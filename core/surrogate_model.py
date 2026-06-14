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

    def score(self) -> dict:
        """Compute R², RMSE, MAE on training data (cross-val would be better
        but this is fast enough for the UI)."""
        if not self.is_trained or self._X_train is None:
            return {"r2": 0.0, "rmse": 0.0, "mae": 0.0}
        y_pred = self.predict(self._X_train)
        y = self._y_train
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))
        mae = float(np.mean(np.abs(y - y_pred)))
        return {"r2": float(r2), "rmse": rmse, "mae": mae}

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

    def __init__(self, kernel: str = "multiquadric"):
        super().__init__()
        self._kernel = kernel
        self._rbf = None

    def fit(self, X, y):
        super().fit(X, y)
        from scipy.interpolate import RBFInterpolator
        self._rbf = RBFInterpolator(X, y, kernel=self._kernel)

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

    Stops when R² improvement < *convergence_threshold* or max iterations.
    Returns the trained model.
    """
    X = initial_X.copy()
    y = initial_y.copy()
    model.fit(X, y)
    prev_r2 = model.score()["r2"]
    logger.info(f"Active learning: iter 0, R²={prev_r2:.4f}, n={len(X)}")

    for it in range(1, max_iterations + 1):
        X_new, y_new = adaptive_sample(
            model, design_variables, base_config, X, y,
            n_new=n_new_per_iter, target=target)

        X = np.vstack([X, X_new])
        y = np.concatenate([y, y_new])
        model.fit(X, y)

        new_r2 = model.score()["r2"]
        improvement = new_r2 - prev_r2
        logger.info(f"Active learning: iter {it}, R²={new_r2:.4f} (Δ={improvement:+.4f}), n={len(X)}")

        if improvement < convergence_threshold:
            logger.info("Active learning converged")
            break
        prev_r2 = new_r2

    return model
