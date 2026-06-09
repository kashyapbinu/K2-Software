"""
K2 Aerospace — Global Sensitivity Analysis
=============================================
Methods for identifying dominant design parameters:

  • Sobol indices  (first-order S1, total-order ST)
  • Partial Rank Correlation Coefficients (PRCC)
  • Standardised Regression Coefficients (SRC)
  • Morris Screening  (elementary effects μ*, σ)

No Qt imports — pure computation, thread-safe.
"""

from __future__ import annotations

import copy
import logging
import math

import numpy as np
from scipy import stats as sp_stats
from scipy.stats import qmc as _qmc

from core.batch_simulation import BatchSimConfig, run_batch_simulation

logger = logging.getLogger("K2.Sensitivity")


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER — batch evaluation
# ══════════════════════════════════════════════════════════════════════════════

def _apply_sample(base_config: BatchSimConfig, sample: np.ndarray,
                  enabled_vars: list) -> BatchSimConfig:
    """Create a BatchSimConfig from a single sample vector."""
    cfg = copy.deepcopy(base_config)
    for j, dv in enumerate(enabled_vars):
        val = sample[j]
        if dv.var_type == "integer":
            val = round(val)
        if hasattr(cfg, dv.name):
            setattr(cfg, dv.name, val)
        if dv.name == "fin_span" and hasattr(cfg, "fin_height"):
            cfg.fin_height = val
    return cfg


def _evaluate_samples(X: np.ndarray, enabled_vars: list,
                      base_config: BatchSimConfig,
                      target: str = "apogee") -> np.ndarray:
    """Evaluate each row of X via batch simulation, returning target metric."""
    n = X.shape[0]
    y = np.zeros(n)
    for i in range(n):
        cfg = _apply_sample(base_config, X[i], enabled_vars)
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
            elif target == "rail_exit":
                y[i] = res.rail_exit_velocity
            else:
                y[i] = res.apogee
        except Exception:
            y[i] = 0.0
    return y


# ══════════════════════════════════════════════════════════════════════════════
#  SOBOL  INDICES  (Saltelli sampling scheme)
# ══════════════════════════════════════════════════════════════════════════════

def sobol_analyze(design_variables, base_config: BatchSimConfig,
                  n_samples: int = 512,
                  target: str = "apogee") -> dict:
    """Compute first-order (S1) and total-order (ST) Sobol indices.

    Uses the Saltelli (2002) sampling scheme, requiring N(2p+2) evaluations
    for p variables.

    Returns {var_name: {"S1": float, "ST": float}}.
    """
    enabled = [dv for dv in design_variables if dv.enabled]
    p = len(enabled)
    if p < 1:
        return {}

    lo = np.array([dv.min_val for dv in enabled])
    hi = np.array([dv.max_val for dv in enabled])

    # Sobol' quasi-random samples in [0,1]^(2p)
    sampler = _qmc.Sobol(d=2 * p, scramble=True, seed=42)
    # Need at least n_samples rows
    m = max(6, int(math.ceil(math.log2(n_samples))))
    unit = sampler.random_base2(m=m)[:n_samples]

    A_unit = unit[:, :p]
    B_unit = unit[:, p:]

    A = _qmc.scale(A_unit, lo, hi)
    B = _qmc.scale(B_unit, lo, hi)

    # Evaluate A and B
    Y_A = _evaluate_samples(A, enabled, base_config, target)
    Y_B = _evaluate_samples(B, enabled, base_config, target)

    f0 = np.mean(Y_A)
    var_total = np.var(np.concatenate([Y_A, Y_B]))
    if var_total < 1e-12:
        return {dv.name: {"S1": 0.0, "ST": 0.0} for dv in enabled}

    results = {}
    for j in range(p):
        # AB_j: A with j-th column replaced by B
        AB = A.copy()
        AB[:, j] = B[:, j]
        Y_AB = _evaluate_samples(AB, enabled, base_config, target)

        # First order:  S1_j = V[E[Y|Xj]] / V[Y]
        # Jansen estimator:  S1 = (V[Y] - 0.5*mean((Y_B - Y_AB)^2)) / V[Y]
        s1 = 1.0 - 0.5 * np.mean((Y_B - Y_AB) ** 2) / var_total

        # Total order:  ST_j = 0.5 * mean((Y_A - Y_AB)^2) / V[Y]
        st = 0.5 * np.mean((Y_A - Y_AB) ** 2) / var_total

        s1 = max(0.0, min(1.0, s1))
        st = max(0.0, min(1.0, st))

        results[enabled[j].name] = {"S1": float(s1), "ST": float(st)}

    logger.info(f"Sobol analysis: {p} vars, {n_samples} base samples, "
                f"~{n_samples*(p+2)} total evaluations")
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  PARTIAL  RANK  CORRELATION  COEFFICIENTS  (PRCC)
# ══════════════════════════════════════════════════════════════════════════════

def compute_prcc(X: np.ndarray, y: np.ndarray, var_names: list) -> dict:
    """Partial Rank Correlation Coefficients.

    Rank-transforms all variables, then computes partial correlation of
    each X_j with Y while controlling for all other X variables.

    Returns {var_name: {"prcc": float, "p_value": float}}.
    """
    n, p = X.shape
    if n < p + 2:
        return {name: {"prcc": 0.0, "p_value": 1.0} for name in var_names}

    # Rank-transform
    X_rank = np.zeros_like(X)
    for j in range(p):
        X_rank[:, j] = sp_stats.rankdata(X[:, j])
    y_rank = sp_stats.rankdata(y)

    results = {}
    for j in range(p):
        # Regress X_j on all other X → residual
        others = np.delete(X_rank, j, axis=1)
        try:
            # OLS: residuals of X_j ~ others
            coef_x = np.linalg.lstsq(others, X_rank[:, j], rcond=None)[0]
            res_x = X_rank[:, j] - others @ coef_x

            # OLS: residuals of Y ~ others
            coef_y = np.linalg.lstsq(others, y_rank, rcond=None)[0]
            res_y = y_rank - others @ coef_y

            # Pearson correlation of residuals
            r, pval = sp_stats.pearsonr(res_x, res_y)
        except Exception:
            r, pval = 0.0, 1.0

        results[var_names[j]] = {"prcc": float(r), "p_value": float(pval)}

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  STANDARDISED  REGRESSION  COEFFICIENTS  (SRC)
# ══════════════════════════════════════════════════════════════════════════════

def compute_src(X: np.ndarray, y: np.ndarray, var_names: list) -> dict:
    """Standardised Regression Coefficients via OLS.

    Returns {var_name: {"src": float, "p_value": float}}.
    """
    n, p = X.shape
    if n < p + 1:
        return {name: {"src": 0.0, "p_value": 1.0} for name in var_names}

    # Standardise
    X_std = (X - X.mean(axis=0)) / np.maximum(X.std(axis=0), 1e-12)
    y_std = (y - y.mean()) / max(y.std(), 1e-12)

    # OLS with intercept
    X_aug = np.column_stack([np.ones(n), X_std])
    try:
        beta = np.linalg.lstsq(X_aug, y_std, rcond=None)[0]
        src = beta[1:]  # drop intercept

        # Approximate p-values
        y_pred = X_aug @ beta
        residuals = y_std - y_pred
        mse = np.sum(residuals ** 2) / max(n - p - 1, 1)
        try:
            cov = mse * np.linalg.inv(X_aug.T @ X_aug)
            se = np.sqrt(np.diag(cov)[1:])
            t_stat = src / np.maximum(se, 1e-12)
            p_values = 2 * (1 - sp_stats.t.cdf(np.abs(t_stat), df=max(n - p - 1, 1)))
        except Exception:
            p_values = np.ones(p)
    except Exception:
        src = np.zeros(p)
        p_values = np.ones(p)

    results = {}
    for j in range(p):
        results[var_names[j]] = {
            "src": float(src[j]),
            "p_value": float(p_values[j]),
        }
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  MORRIS  SCREENING  (elementary effects)
# ══════════════════════════════════════════════════════════════════════════════

def morris_screening(design_variables, base_config: BatchSimConfig,
                     n_trajectories: int = 10,
                     target: str = "apogee",
                     n_levels: int = 4) -> dict:
    """Morris (1991) Elementary Effects method.

    Computes μ* (absolute mean of elementary effects — importance)
    and σ (standard deviation — interaction/non-linearity indicator).

    Returns {var_name: {"mu_star": float, "sigma": float}}.
    """
    enabled = [dv for dv in design_variables if dv.enabled]
    p = len(enabled)
    if p < 1:
        return {}

    lo = np.array([dv.min_val for dv in enabled])
    hi = np.array([dv.max_val for dv in enabled])
    rng = np.random.default_rng(42)

    delta = 1.0 / (n_levels - 1) if n_levels > 1 else 0.5

    effects = {dv.name: [] for dv in enabled}

    for _ in range(n_trajectories):
        # Random base point on the grid
        x_base = rng.choice(n_levels, p) / max(n_levels - 1, 1)
        order = rng.permutation(p)

        x_cur = x_base.copy()
        x_phys = lo + x_cur * (hi - lo)
        cfg = _apply_sample(base_config, x_phys, enabled)
        try:
            y_cur = getattr(run_batch_simulation(cfg, seed=rng.integers(10000)), target,
                            run_batch_simulation(cfg, seed=rng.integers(10000)).apogee)
        except Exception:
            y_cur = 0.0

        for j in order:
            # Perturb j-th variable by +/- delta
            x_next = x_cur.copy()
            if x_cur[j] + delta <= 1.0:
                x_next[j] = x_cur[j] + delta
            else:
                x_next[j] = x_cur[j] - delta

            x_phys_next = lo + x_next * (hi - lo)
            cfg_next = _apply_sample(base_config, x_phys_next, enabled)
            try:
                res = run_batch_simulation(cfg_next, seed=rng.integers(10000))
                y_next = getattr(res, target, res.apogee)
            except Exception:
                y_next = 0.0

            ee = (y_next - y_cur) / delta if abs(delta) > 1e-12 else 0.0
            effects[enabled[j].name].append(ee)

            x_cur = x_next
            y_cur = y_next

    results = {}
    for dv in enabled:
        ees = np.array(effects[dv.name])
        if len(ees) > 0:
            mu_star = float(np.mean(np.abs(ees)))
            sigma = float(np.std(ees))
        else:
            mu_star = sigma = 0.0
        results[dv.name] = {"mu_star": mu_star, "sigma": sigma}

    logger.info(f"Morris screening: {p} vars, {n_trajectories} trajectories")
    return results
