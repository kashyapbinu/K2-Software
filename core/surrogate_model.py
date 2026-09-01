"""
K2 AeroSim — Surrogate Model Module
=======================================
Lightweight surrogate models for accelerating optimisation.

Models:
  • Random Forest (sklearn)
  • Gradient Boosting (sklearn)
  • Neural Network / MLP (sklearn)
  • Kriging / Gaussian Process (custom, scipy-based)
  • RBF Interpolation (scipy)
  • Polynomial Response Surface (numpy)

Features:
  • Adaptive / active-learning loop
  • Response-surface mesh generation for contour plots
  • Latin-Hypercube initial sampling

No Qt imports — pure computation, fully thread-safe.
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import qmc as _qmc

from core.batch_simulation import BatchSimConfig, run_batch_simulation

logger = logging.getLogger("K2.Surrogate")


# ══════════════════════════════════════════════════════════════════════════════
#  BASE  CLASS
# ══════════════════════════════════════════════════════════════════════════════

class SurrogateModel:
    """Abstract base for all surrogate models."""

    #: Folds used by :meth:`score`. Capped at n//2 for small sample counts.
    CV_FOLDS = 5

    def __init__(self):
        self.is_trained = False
        self._X_train = None
        self._y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self._X_train = X.copy()
        self._y_train = y.copy()
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict_uncertainty(self, X: np.ndarray) -> np.ndarray:
        """Return predicted std-dev at each point (default=0)."""
        return np.zeros(X.shape[0])

    def _clone_untrained(self) -> "SurrogateModel":
        """A fresh, unfitted copy with the same hyper-parameters.

        Subclasses override this whenever their constructor takes tuning
        arguments, so cross-validation refits the model the user asked for
        rather than the class default.
        """
        return self.__class__()

    @staticmethod
    def _metrics(y: np.ndarray, y_pred: np.ndarray) -> dict:
        y = np.asarray(y, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        return {
            "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0,
            "rmse": float(np.sqrt(np.mean((y - y_pred) ** 2))),
            "mae": float(np.mean(np.abs(y - y_pred))),
        }

    def score(self, cv: bool = True) -> dict:
        """R², RMSE and MAE for the fitted model, by k-fold cross-validation.

        The training-set fit is *not* a usable accuracy measure here. Random
        Forest, RBF and Kriging all reproduce their own training points almost
        exactly, so a training R² sits near 1.0 however badly the model
        generalises — which in turn makes any convergence test built on it fire
        on the first iteration. Cross-validated R² can legitimately come back
        negative; that means the surrogate is worse than predicting the mean.

        Falls back to the training fit (``cv_used=False``) when there are too
        few samples to split, or when every fold refit failed.
        """
        if not self.is_trained or self._X_train is None or len(self._X_train) < 2:
            return {"r2": 0.0, "rmse": 0.0, "mae": 0.0, "r2_train": 0.0,
                    "cv_used": False, "n_folds": 0, "n_samples": 0}

        n = len(self._X_train)
        train = self._metrics(self._y_train, self.predict(self._X_train))

        def _fallback():
            out = dict(train)
            out.update({"r2_train": train["r2"], "cv_used": False,
                        "n_folds": 0, "n_samples": n})
            return out

        n_folds = min(self.CV_FOLDS, n // 2)
        if not cv or n_folds < 2:
            return _fallback()

        # Fixed permutation seed so repeated scoring of the same training set
        # is reproducible (the convergence test compares successive scores).
        y_cv = np.full(n, np.nan)
        for hold in np.array_split(np.random.default_rng(0).permutation(n), n_folds):
            keep = np.ones(n, dtype=bool)
            keep[hold] = False
            try:
                m = self._clone_untrained()
                m.fit(self._X_train[keep], self._y_train[keep])
                y_cv[hold] = m.predict(self._X_train[hold])
            except Exception as e:
                logger.debug(f"CV fold failed ({type(self).__name__}): {e}")

        ok = np.isfinite(y_cv)
        if int(ok.sum()) < 2:
            return _fallback()

        out = self._metrics(self._y_train[ok], y_cv[ok])
        out.update({"r2_train": train["r2"], "cv_used": True,
                    "n_folds": n_folds, "n_samples": n})
        return out

    def feature_importance(self) -> dict:
        """Return {index: importance} if available."""
        return {}


# ══════════════════════════════════════════════════════════════════════════════
#  SKLEARN-BASED  MODELS
# ══════════════════════════════════════════════════════════════════════════════

class RandomForestSurrogate(SurrogateModel):
    def __init__(self, n_estimators: int = 100, max_depth: int = 15, seed: int = 42):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.seed = seed
        self._model = None

    def fit(self, X, y):
        super().fit(X, y)
        from sklearn.ensemble import RandomForestRegressor
        self._model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.seed,
            n_jobs=-1,
        )
        self._model.fit(X, y)
        logger.debug(f"RF fit: {X.shape[0]} samples, {X.shape[1]} features")

    def _clone_untrained(self):
        return RandomForestSurrogate(self.n_estimators, self.max_depth, self.seed)

    def predict(self, X):
        return self._model.predict(X)

    def predict_uncertainty(self, X):
        """Use tree variance as uncertainty proxy."""
        preds = np.array([t.predict(X) for t in self._model.estimators_])
        return np.std(preds, axis=0)

    def feature_importance(self):
        if self._model is None:
            return {}
        return {i: v for i, v in enumerate(self._model.feature_importances_)}


class GradientBoostingSurrogate(SurrogateModel):
    def __init__(self, n_estimators: int = 200, lr: float = 0.1, max_depth: int = 5,
                 seed: int = 42):
        super().__init__()
        self._params = dict(n_estimators=n_estimators, learning_rate=lr,
                            max_depth=max_depth, random_state=seed)
        self._model = None

    def fit(self, X, y):
        super().fit(X, y)
        from sklearn.ensemble import GradientBoostingRegressor
        self._model = GradientBoostingRegressor(**self._params)
        self._model.fit(X, y)

    def _clone_untrained(self):
        p = self._params
        return GradientBoostingSurrogate(
            n_estimators=p["n_estimators"], lr=p["learning_rate"],
            max_depth=p["max_depth"], seed=p["random_state"])

    def predict(self, X):
        return self._model.predict(X)

    def feature_importance(self):
        if self._model is None:
            return {}
        return {i: v for i, v in enumerate(self._model.feature_importances_)}


class NeuralNetworkSurrogate(SurrogateModel):
    def __init__(self, hidden_layers=(64, 32), seed: int = 42):
        super().__init__()
        self.hidden = hidden_layers
        self.seed = seed
        self._model = None

    def fit(self, X, y):
        super().fit(X, y)
        from sklearn.neural_network import MLPRegressor
        self._model = MLPRegressor(
            hidden_layer_sizes=self.hidden,
            activation="relu",
            solver="adam",
            max_iter=500,
            early_stopping=True,
            random_state=self.seed,
            validation_fraction=0.15,
        )
        # Normalise inputs for MLP
        self._X_mean = X.mean(axis=0)
        self._X_std = X.std(axis=0)
        self._X_std[self._X_std < 1e-12] = 1.0
        self._y_mean = y.mean()
        self._y_std = y.std() if y.std() > 0 else 1.0
        X_n = (X - self._X_mean) / self._X_std
        y_n = (y - self._y_mean) / self._y_std
        self._model.fit(X_n, y_n)

    def _clone_untrained(self):
        return NeuralNetworkSurrogate(self.hidden, self.seed)

    def predict(self, X):
        X_n = (X - self._X_mean) / self._X_std
        y_n = self._model.predict(X_n)
        return y_n * self._y_std + self._y_mean


# ══════════════════════════════════════════════════════════════════════════════
#  SCIPY / NUMPY  MODELS  (no sklearn needed)
# ══════════════════════════════════════════════════════════════════════════════

class KrigingSurrogate(SurrogateModel):
    """Simple Ordinary Kriging with squared-exponential kernel.

    Provides analytical uncertainty via the kriging variance.
    """

    def __init__(self, theta: float = 1.0):
        super().__init__()
        self._theta = theta
        self._K_inv = None
        self._alpha = None

    def _kernel(self, X1, X2, theta):
        sq = np.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=2)
        return np.exp(-0.5 * sq / (theta ** 2))

    def fit(self, X, y):
        super().fit(X, y)
        n = X.shape[0]
        # Optimise length scale on [0.1, 10]
        best_ll, best_theta = -1e30, self._theta
        for theta in np.logspace(-1, 1, 15):
            K = self._kernel(X, X, theta) + 1e-6 * np.eye(n)
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
                ll = -0.5 * y @ alpha - np.sum(np.log(np.diag(L)))
                if ll > best_ll:
                    best_ll = ll
                    best_theta = theta
            except np.linalg.LinAlgError:
                continue
        self._theta = best_theta
        K = self._kernel(X, X, best_theta) + 1e-6 * np.eye(n)
        self._K_inv = np.linalg.inv(K)
        self._alpha = self._K_inv @ y
        logger.debug(f"Kriging fit: θ={best_theta:.3f}, n={n}")

    def _clone_untrained(self):
        return KrigingSurrogate(self._theta)

    def predict(self, X):
        k_star = self._kernel(X, self._X_train, self._theta)
        return k_star @ self._alpha

    def predict_uncertainty(self, X):
        k_star = self._kernel(X, self._X_train, self._theta)
        k_ss = np.ones(X.shape[0])  # kernel(x,x) = 1 for SE
        var = k_ss - np.sum(k_star @ self._K_inv * k_star, axis=1)
        return np.sqrt(np.maximum(var, 0.0))


class RBFSurrogate(SurrogateModel):
    """RBF interpolation using scipy."""

    #: Kernels scipy can build without a shape parameter.
    _SCALE_FREE = {"linear", "quintic", "cubic", "thin_plate_spline"}

    def __init__(self, kernel: str = "multiquadric", epsilon: float = 1.0):
        super().__init__()
        self._kernel = kernel
        self._epsilon = epsilon
        self._rbf = None

    def fit(self, X, y):
        super().fit(X, y)
        from scipy.interpolate import RBFInterpolator
        # scipy raises for a scale-dependent kernel with no epsilon, which made
        # every RBF fit fail. Scale-free kernels reject it just as hard, so the
        # argument has to be conditional.
        kw = {} if self._kernel in self._SCALE_FREE else {"epsilon": self._epsilon}
        self._rbf = RBFInterpolator(X, y, kernel=self._kernel, **kw)

    def _clone_untrained(self):
        return RBFSurrogate(self._kernel, self._epsilon)

    def predict(self, X):
        return self._rbf(X)


class PolynomialSurrogate(SurrogateModel):
    """2nd or 3rd order polynomial regression."""

    def __init__(self, degree: int = 2):
        super().__init__()
        self.degree = degree
        self._coefs = None
        self._powers = None

    def fit(self, X, y):
        super().fit(X, y)
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression
        self._poly = PolynomialFeatures(degree=self.degree, include_bias=True)
        X_p = self._poly.fit_transform(X)
        self._reg = LinearRegression().fit(X_p, y)

    def _clone_untrained(self):
        return PolynomialSurrogate(self.degree)

    def predict(self, X):
        X_p = self._poly.transform(X)
        return self._reg.predict(X_p)


# ══════════════════════════════════════════════════════════════════════════════
#  FACTORY
# ══════════════════════════════════════════════════════════════════════════════

_MODEL_MAP = {
    "random_forest": RandomForestSurrogate,
    "gradient_boosting": GradientBoostingSurrogate,
    "neural_network": NeuralNetworkSurrogate,
    "kriging": KrigingSurrogate,
    "rbf": RBFSurrogate,
    "polynomial": PolynomialSurrogate,
}


def create_surrogate(name: str = "random_forest") -> SurrogateModel:
    cls = _MODEL_MAP.get(name, RandomForestSurrogate)
    return cls()


# ══════════════════════════════════════════════════════════════════════════════
#  SAMPLING  &  RESPONSE  SURFACES
# ══════════════════════════════════════════════════════════════════════════════

def build_initial_samples(design_variables, base_config: BatchSimConfig,
                          n_samples: int = 200,
                          method: str = "lhs",
                          target: str = "apogee") -> tuple:
    """Generate space-filling samples and evaluate each via batch simulation.

    Returns (X: ndarray [n × d], y: ndarray [n]).
    """
    enabled = [dv for dv in design_variables if dv.enabled]
    n_vars = len(enabled)
    if n_vars == 0 or n_samples == 0:
        return np.empty((0, 0)), np.empty(0)

    # LHS in [0, 1]^d
    sampler = _qmc.LatinHypercube(d=n_vars, seed=42)
    unit_samples = sampler.random(n=n_samples)

    lo = np.array([dv.min_val for dv in enabled])
    hi = np.array([dv.max_val for dv in enabled])
    X = _qmc.scale(unit_samples, lo, hi)

    y = np.zeros(n_samples)
    for i in range(n_samples):
        cfg = copy.deepcopy(base_config)
        for j, dv in enumerate(enabled):
            val = X[i, j]
            if dv.var_type == "integer":
                val = round(val)
            if hasattr(cfg, dv.name):
                setattr(cfg, dv.name, val)
            if dv.name == "fin_span" and hasattr(cfg, "fin_height"):
                cfg.fin_height = val
        try:
            res = run_batch_simulation(cfg, seed=42 + i)
            if target == "apogee":
                y[i] = res.apogee
            elif target == "mach":
                y[i] = res.max_mach
            elif target == "landing":
                y[i] = res.landing_distance
            elif target == "stability":
                y[i] = res.min_stability_margin
            else:
                y[i] = res.apogee
        except Exception:
            y[i] = 0.0

    logger.info(f"Initial samples: {n_samples} pts, {n_vars} vars, "
                f"y range [{y.min():.1f}, {y.max():.1f}]")
    return X, y


def build_response_surface(model: SurrogateModel,
                           var1_idx: int, var1_range: tuple,
                           var2_idx: int, var2_range: tuple,
                           fixed_values: np.ndarray,
                           n_points: int = 50) -> tuple:
    """Generate a 2-D mesh prediction for contour plots.

    Returns (X_grid, Y_grid, Z_predictions) — all n_points × n_points.
    """
    x1 = np.linspace(var1_range[0], var1_range[1], n_points)
    x2 = np.linspace(var2_range[0], var2_range[1], n_points)
    X1, X2 = np.meshgrid(x1, x2)

    n_features = len(fixed_values)
    grid_pts = np.tile(fixed_values, (n_points * n_points, 1))
    grid_pts[:, var1_idx] = X1.ravel()
    grid_pts[:, var2_idx] = X2.ravel()

    Z = model.predict(grid_pts).reshape(n_points, n_points)
    return X1, X2, Z


# ══════════════════════════════════════════════════════════════════════════════
#  ACTIVE  LEARNING
# ══════════════════════════════════════════════════════════════════════════════

def adaptive_sample(model: SurrogateModel,
                    design_variables, base_config: BatchSimConfig,
                    X_train: np.ndarray, y_train: np.ndarray,
                    n_new: int = 20,
                    target: str = "apogee") -> tuple:
    """Infill sampling: high-uncertainty + near-optimum exploration.

    Returns (X_new, y_new).
    """
    enabled = [dv for dv in design_variables if dv.enabled]
    n_vars = len(enabled)
    lo = np.array([dv.min_val for dv in enabled])
    hi = np.array([dv.max_val for dv in enabled])

    # Generate candidate pool
    sampler = _qmc.LatinHypercube(d=n_vars, seed=int(time.time()) % 10000)
    candidates = _qmc.scale(sampler.random(n=n_new * 10), lo, hi)

    # Split: 50 % exploitation, 50 % exploration
    n_exploit = n_new // 2
    n_explore = n_new - n_exploit

    # Exploitation: best predicted values
    y_pred = model.predict(candidates)
    exploit_idx = np.argsort(y_pred)[-n_exploit:]

    # Exploration: highest uncertainty
    try:
        unc = model.predict_uncertainty(candidates)
        explore_idx = np.argsort(unc)[-n_explore:]
    except Exception:
        explore_idx = np.random.default_rng().choice(
            len(candidates), n_explore, replace=False)

    selected = np.unique(np.concatenate([exploit_idx, explore_idx]))
    if len(selected) > n_new:
        selected = selected[:n_new]

    X_new = candidates[selected]
    y_new = np.zeros(len(X_new))

    for i in range(len(X_new)):
        cfg = copy.deepcopy(base_config)
        for j, dv in enumerate(enabled):
            val = X_new[i, j]
            if dv.var_type == "integer":
                val = round(val)
            if hasattr(cfg, dv.name):
                setattr(cfg, dv.name, val)
            if dv.name == "fin_span" and hasattr(cfg, "fin_height"):
                cfg.fin_height = val
        try:
            res = run_batch_simulation(cfg, seed=1000 + i)
            y_new[i] = getattr(res, target, res.apogee)
        except Exception:
            y_new[i] = 0.0

    return X_new, y_new


def train_with_active_learning(model: SurrogateModel,
                               design_variables,
                               base_config: BatchSimConfig,
                               initial_X: np.ndarray,
                               initial_y: np.ndarray,
                               max_iterations: int = 5,
                               convergence_threshold: float = 0.01,
                               n_new_per_iter: int = 20,
                               target: str = "apogee") -> SurrogateModel:
    """Iterative train → sample → retrain loop.

    Stops when the *cross-validated* R² fails to improve by
    *convergence_threshold* for two iterations running, or at max iterations.
    Returns the trained model.

    The patience of two matters: cross-validated R² is noisy on small samples,
    so a single flat iteration is not evidence of convergence the way a flat
    training R² would (misleadingly) look.
    """
    X = initial_X.copy()
    y = initial_y.copy()
    model.fit(X, y)
    sc = model.score()
    best_r2 = sc["r2"]
    stalled = 0
    logger.info(f"Active learning: iter 0, CV R²={best_r2:.4f} "
                f"(train R²={sc['r2_train']:.4f}), n={len(X)}")

    for it in range(1, max_iterations + 1):
        X_new, y_new = adaptive_sample(
            model, design_variables, base_config, X, y,
            n_new=n_new_per_iter, target=target)

        X = np.vstack([X, X_new])
        y = np.concatenate([y, y_new])
        model.fit(X, y)

        sc = model.score()
        new_r2 = sc["r2"]
        improvement = new_r2 - best_r2
        logger.info(f"Active learning: iter {it}, CV R²={new_r2:.4f} "
                    f"(Δ={improvement:+.4f}, train R²={sc['r2_train']:.4f}), n={len(X)}")

        if improvement < convergence_threshold:
            stalled += 1
            if stalled >= 2:
                logger.info("Active learning converged")
                break
        else:
            stalled = 0
        best_r2 = max(best_r2, new_r2)

    return model
