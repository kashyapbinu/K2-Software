"""
K2 Aerospace — Design of Experiments (DOE) Engine
===================================================
Methods for systematic design-space exploration:

  • Full Factorial
  • Fractional Factorial  (2^(k-p))
  • Latin Hypercube Sampling
  • Taguchi Orthogonal Arrays  (L9 / L16 / L27)

Post-processing:
  • Main effects computation
  • Two-factor interaction effects

No Qt imports — pure computation, thread-safe.
"""

from __future__ import annotations

import copy
import itertools
import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import qmc as _qmc

from core.batch_simulation import BatchSimConfig, run_batch_simulation

logger = logging.getLogger("K2.DOE")


# ══════════════════════════════════════════════════════════════════════════════
#  DATA  CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DOEConfig:
    method: str = "lhs"           # lhs / full_factorial / fractional / taguchi
    variables: list = field(default_factory=list)  # list of DesignVariable
    levels: int = 3
    n_samples: int = 50
    target: str = "apogee"


@dataclass
class DOEResult:
    design_matrix: np.ndarray = None      # n × p  (physical values)
    responses: np.ndarray = None          # n
    var_names: list = field(default_factory=list)
    main_effects: dict = field(default_factory=dict)
    interactions: dict = field(default_factory=dict)
    method: str = ""
    n_runs: int = 0


# ══════════════════════════════════════════════════════════════════════════════
#  DESIGN  MATRICES
# ══════════════════════════════════════════════════════════════════════════════

def full_factorial(variables, levels: int = 3) -> np.ndarray:
    """Full-factorial design matrix in physical coordinates.

    Each variable is sampled at *levels* equally-spaced points
    between min_val and max_val.  Total runs = levels^p.
    """
    enabled = [dv for dv in variables if dv.enabled]
    p = len(enabled)
    if p == 0:
        return np.empty((0, 0))

    level_vals = []
    for dv in enabled:
        level_vals.append(np.linspace(dv.min_val, dv.max_val, levels))

    combos = list(itertools.product(*level_vals))
    return np.array(combos)


def fractional_factorial(variables, resolution: int = 3) -> np.ndarray:
    """2^(k-p) fractional factorial in physical coordinates.

    Uses a simple 2-level (+1 / -1) scheme with p = max(0, k - resolution).
    """
    enabled = [dv for dv in variables if dv.enabled]
    k = len(enabled)
    if k == 0:
        return np.empty((0, 0))

    p = max(0, k - resolution)
    n_runs = 2 ** (k - p)

    # Build coded matrix (-1, +1)
    base_cols = k - p
    coded = np.array(list(itertools.product([-1, 1], repeat=base_cols)))

    # Augment with confounded columns
    if p > 0:
        # Highest-order interaction confounding
        extra = np.zeros((n_runs, p))
        for c in range(p):
            # Confound with product of first (c+2) base columns
            cols_to_mult = list(range(min(c + 2, base_cols)))
            extra[:, c] = np.prod(coded[:, cols_to_mult], axis=1)
        coded = np.hstack([coded, extra])

    # Scale to physical values
    matrix = np.zeros_like(coded)
    for j, dv in enumerate(enabled):
        mid = (dv.min_val + dv.max_val) / 2.0
        half = (dv.max_val - dv.min_val) / 2.0
        matrix[:, j] = mid + coded[:, j] * half

    return matrix


def latin_hypercube(variables, n_samples: int = 50) -> np.ndarray:
    """Latin Hypercube Sampling in physical coordinates."""
    enabled = [dv for dv in variables if dv.enabled]
    p = len(enabled)
    if p == 0:
        return np.empty((0, 0))

    sampler = _qmc.LatinHypercube(d=p, seed=42)
    unit = sampler.random(n=n_samples)

    lo = np.array([dv.min_val for dv in enabled])
    hi = np.array([dv.max_val for dv in enabled])
    return _qmc.scale(unit, lo, hi)


# ── Taguchi Orthogonal Arrays ────────────────────────────────────────────────

# Standard Taguchi arrays (coded as level indices 0, 1, 2)
_L9 = np.array([
    [0,0,0,0], [0,1,1,1], [0,2,2,2],
    [1,0,1,2], [1,1,2,0], [1,2,0,1],
    [2,0,2,1], [2,1,0,2], [2,2,1,0],
], dtype=int)

_L16 = np.array([
    [0,0,0,0,0], [0,1,1,1,1], [0,2,2,2,2], [0,3,3,3,3],
    [1,0,1,2,3], [1,1,0,3,2], [1,2,3,0,1], [1,3,2,1,0],
    [2,0,2,3,1], [2,1,3,2,0], [2,2,0,1,3], [2,3,1,0,2],
    [3,0,3,1,2], [3,1,2,0,3], [3,2,1,3,0], [3,3,0,2,1],
], dtype=int)

_L27 = np.array([
    [0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,1,1,1,1,1,1,1,1,1],
    [0,0,0,0,2,2,2,2,2,2,2,2,2],
    [0,1,1,1,0,0,0,1,1,1,2,2,2],
    [0,1,1,1,1,1,1,2,2,2,0,0,0],
    [0,1,1,1,2,2,2,0,0,0,1,1,1],
    [0,2,2,2,0,0,0,2,2,2,1,1,1],
    [0,2,2,2,1,1,1,0,0,0,2,2,2],
    [0,2,2,2,2,2,2,1,1,1,0,0,0],
    [1,0,1,2,0,1,2,0,1,2,0,1,2],
    [1,0,1,2,1,2,0,1,2,0,1,2,0],
    [1,0,1,2,2,0,1,2,0,1,2,0,1],
    [1,1,2,0,0,1,2,1,2,0,2,0,1],
    [1,1,2,0,1,2,0,2,0,1,0,1,2],
    [1,1,2,0,2,0,1,0,1,2,1,2,0],
    [1,2,0,1,0,1,2,2,0,1,1,2,0],
    [1,2,0,1,1,2,0,0,1,2,2,0,1],
    [1,2,0,1,2,0,1,1,2,0,0,1,2],
    [2,0,2,1,0,2,1,0,2,1,0,2,1],
    [2,0,2,1,1,0,2,1,0,2,1,0,2],
    [2,0,2,1,2,1,0,2,1,0,2,1,0],
    [2,1,0,2,0,2,1,1,0,2,2,1,0],
    [2,1,0,2,1,0,2,2,1,0,0,2,1],
    [2,1,0,2,2,1,0,0,2,1,1,0,2],
    [2,2,1,0,0,2,1,2,1,0,1,0,2],
    [2,2,1,0,1,0,2,0,2,1,2,1,0],
    [2,2,1,0,2,1,0,1,0,2,0,2,1],
], dtype=int)


def taguchi_array(variables, levels: int = 3) -> np.ndarray:
    """Standard Taguchi orthogonal array in physical coordinates.

    Selects L9 (≤ 4 vars), L16 (≤ 5 vars), or L27 (≤ 13 vars).
    """
    enabled = [dv for dv in variables if dv.enabled]
    p = len(enabled)
    if p == 0:
        return np.empty((0, 0))

    if p <= 4 and levels <= 3:
        coded = _L9[:, :p]
        n_levels = 3
    elif p <= 5 and levels <= 4:
        coded = _L16[:, :p]
        n_levels = 4
    elif p <= 13:
        coded = _L27[:, :p]
        n_levels = 3
    else:
        # Fall back to LHS for many variables
        return latin_hypercube(variables, n_samples=27)

    # Scale coded levels to physical
    matrix = np.zeros((coded.shape[0], p), dtype=float)
    for j, dv in enumerate(enabled):
        levels_vals = np.linspace(dv.min_val, dv.max_val, n_levels)
        for i in range(coded.shape[0]):
            idx = min(coded[i, j], n_levels - 1)
            matrix[i, j] = levels_vals[idx]

    return matrix


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN  EFFECTS  &  INTERACTIONS
# ══════════════════════════════════════════════════════════════════════════════

def compute_main_effects(design_matrix: np.ndarray, responses: np.ndarray,
                         var_names: list, n_levels: int = 3) -> dict:
    """Compute main effects: average response at each level for each variable.

    Returns {var_name: {"levels": [...], "means": [...], "effect": float}}.
    """
    n, p = design_matrix.shape
    results = {}
    for j in range(min(p, len(var_names))):
        col = design_matrix[:, j]
        # Bin into n_levels
        edges = np.linspace(col.min(), col.max() + 1e-12, n_levels + 1)
        level_means = []
        level_centers = []
        for k in range(n_levels):
            mask = (col >= edges[k]) & (col < edges[k + 1])
            if k == n_levels - 1:
                mask = (col >= edges[k]) & (col <= edges[k + 1])
            if np.sum(mask) > 0:
                level_means.append(float(np.mean(responses[mask])))
            else:
                level_means.append(0.0)
            level_centers.append(float((edges[k] + edges[k + 1]) / 2))

        effect = max(level_means) - min(level_means) if level_means else 0.0
        results[var_names[j]] = {
            "levels": level_centers,
            "means": level_means,
            "effect": effect,
        }
    return results


def compute_interactions(design_matrix: np.ndarray, responses: np.ndarray,
                         var_names: list, n_levels: int = 2) -> dict:
    """Compute two-factor interaction effects.

    Returns {("var_a", "var_b"): {"interaction_effect": float, "grid": 2D list}}.
    """
    n, p = design_matrix.shape
    results = {}
    for i in range(p):
        for j in range(i + 1, p):
            if i >= len(var_names) or j >= len(var_names):
                continue
            col_i = design_matrix[:, i]
            col_j = design_matrix[:, j]
            med_i = np.median(col_i)
            med_j = np.median(col_j)

            # 2×2 interaction table
            means = np.zeros((2, 2))
            for li in range(2):
                for lj in range(2):
                    mask_i = col_i >= med_i if li == 1 else col_i < med_i
                    mask_j = col_j >= med_j if lj == 1 else col_j < med_j
                    mask = mask_i & mask_j
                    if np.sum(mask) > 0:
                        means[li, lj] = float(np.mean(responses[mask]))

            # Interaction = difference of differences
            interaction = (means[1, 1] - means[1, 0]) - (means[0, 1] - means[0, 0])
            results[(var_names[i], var_names[j])] = {
                "interaction_effect": float(interaction),
                "grid": means.tolist(),
            }
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  RUN  DOE
# ══════════════════════════════════════════════════════════════════════════════

def run_doe(doe_config: DOEConfig, base_config: BatchSimConfig) -> DOEResult:
    """Generate design matrix, evaluate all points, and compute effects."""
    variables = doe_config.variables
    enabled = [dv for dv in variables if dv.enabled]
    var_names = [dv.name for dv in enabled]

    # Generate design matrix
    method = doe_config.method.lower().replace(" ", "_")
    if method == "full_factorial":
        dm = full_factorial(variables, doe_config.levels)
    elif method == "fractional":
        dm = fractional_factorial(variables, doe_config.levels)
    elif method == "taguchi":
        dm = taguchi_array(variables, doe_config.levels)
    else:
        dm = latin_hypercube(variables, doe_config.n_samples)

    if dm.size == 0:
        return DOEResult(method=method)

    n_runs = dm.shape[0]
    target = doe_config.target

    # Evaluate
    responses = np.zeros(n_runs)
    for i in range(n_runs):
        cfg = copy.deepcopy(base_config)
        for j, dv in enumerate(enabled):
            val = dm[i, j]
            if dv.var_type == "integer":
                val = round(val)
            if hasattr(cfg, dv.name):
                setattr(cfg, dv.name, val)
            if dv.name == "fin_span" and hasattr(cfg, "fin_height"):
                cfg.fin_height = val
        try:
            res = run_batch_simulation(cfg, seed=42 + i)
            if target == "apogee":
                responses[i] = res.apogee
            elif target == "mach":
                responses[i] = res.max_mach
            elif target == "stability":
                responses[i] = res.min_stability_margin
            elif target == "landing":
                responses[i] = res.landing_distance
            else:
                responses[i] = res.apogee
        except Exception:
            responses[i] = 0.0

    # Compute effects
    main_eff = compute_main_effects(dm, responses, var_names, doe_config.levels)
    interactions = compute_interactions(dm, responses, var_names)

    logger.info(f"DOE complete: {method}, {n_runs} runs, {len(enabled)} vars")

    return DOEResult(
        design_matrix=dm,
        responses=responses,
        var_names=var_names,
        main_effects=main_eff,
        interactions=interactions,
        method=method,
        n_runs=n_runs,
    )
