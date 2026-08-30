"""
K2 AeroSim — Optimization Engine
====================================
Aerospace-grade Multidisciplinary Design Optimization (MDO) engine.

Algorithms (pure functions — no Qt, thread-safe):
  • Genetic Algorithm (GA) with SBX crossover & polynomial mutation
  • NSGA-II multi-objective with fast non-dominated sorting
  • Differential Evolution (DE/rand/1/bin)
  • Particle Swarm Optimisation (PSO)

Features:
  • Correlated Monte-Carlo sampling via Cholesky decomposition
  • Robust optimisation (mean / σ / worst-case / reliability / percentile)
  • Mission-driven fitness (target-altitude satisfaction)
  • Physics validation of candidate designs
  • Integrated Sobol sensitivity & PRCC
  • Qt orchestrator (QThread worker + QObject signals)

All heavy computation lives in pure functions so it can safely run inside
a QThread without touching any Qt objects.
"""

from __future__ import annotations

import os
import copy
import math
import time
import logging
from dataclasses import dataclass, field
from typing import Callable, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.stats import qmc

from PyQt6.QtCore import QObject, QThread, pyqtSignal

from core.batch_simulation import (
    BatchSimConfig, BatchSimResult, run_batch_simulation, result_diverged,
)

logger = logging.getLogger("K2.Optimization")

# Motor database for discrete variable
MOTOR_DATABASE = [
    "Estes_D12", "CTI_H100", "CTI_I200", "AT_J350",
    "CTI_K660", "AT_L1000", "CTI_M1400",
]

# Approximate motor properties for evaluation {name: (impulse_Ns, burn_s, prop_mass_kg, avg_thrust_N, max_thrust_N, isp_s)}
_MOTOR_PROPS = {
    "Estes_D12":  (16.85,  1.60, 0.021, 10.53,  28.58, 81.8),
    "CTI_H100":   (176.0,  1.76, 0.094, 100.0,  120.0, 190.7),
    "CTI_I200":   (365.0,  1.82, 0.182, 200.5,  260.0, 204.3),
    "AT_J350":    (652.0,  1.86, 0.310, 350.5,  420.0, 214.3),
    "CTI_K660":   (1417.0, 2.15, 0.650, 659.1,  800.0, 222.2),
    "AT_L1000":   (2758.0, 2.76, 1.221, 999.3, 1260.0, 230.2),
    "CTI_M1400":  (5500.0, 3.93, 2.399, 1399.5, 1750.0, 233.5),
}


# ══════════════════════════════════════════════════════════════════════════════
#  DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DesignVariable:
    name: str
    display_name: str
    category: str              # Geometry / Mass / Propulsion / Recovery / Aerodynamics
    min_val: float
    max_val: float
    current_val: float
    enabled: bool = True
    var_type: str = "continuous"   # continuous / integer / discrete
    discrete_options: list = field(default_factory=list)


@dataclass
class ObjectiveFunction:
    name: str
    display_name: str
    direction: str = "maximize"    # maximize / minimize
    weight: float = 1.0
    enabled: bool = True
    robust_mode: str = "mean"      # mean / std / worst / reliability / p5


@dataclass
class Constraint:
    name: str
    display_name: str
    type: str = "greater_than"     # greater_than / less_than
    limit: float = 0.0
    penalty_weight: float = 1000.0
    enabled: bool = True


@dataclass
class CorrelationEntry:
    param1: str
    param2: str
    coefficient: float


@dataclass
class OptimizationConfig:
    algorithm: str = "ga"
    design_variables: list = field(default_factory=list)
    objectives: list = field(default_factory=list)
    constraints: list = field(default_factory=list)
    correlations: list = field(default_factory=list)
    population_size: int = 50
    max_generations: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    mc_sims_per_candidate: int = 5
    validation_mc_sims: int = 50
    use_surrogate: bool = False
    surrogate_type: str = "random_forest"
    surrogate_initial_samples: int = 200   # standalone response-surface sampling
    surrogate_pool_factor: int = 4         # candidates proposed per slot simulated
    target_apogee: float = 0.0
    mission_mode: bool = False
    robust_mode: bool = False
    parallel: bool = True          # evaluate population in a process pool
    n_workers: int = 0             # 0 → auto (cpu_count - 1)


@dataclass
class CandidateDesign:
    variables: dict = field(default_factory=dict)
    fitness: float = 0.0
    objectives: dict = field(default_factory=dict)
    constraints_eval: dict = field(default_factory=dict)
    feasible: bool = True
    rank: int = 0
    crowding_distance: float = 0.0
    mc_stats: dict = field(default_factory=dict)
    batch_config: object = None


@dataclass
class OptimizationResult:
    best_design: CandidateDesign = None
    pareto_front: list = field(default_factory=list)
    all_designs: list = field(default_factory=list)
    generation_history: list = field(default_factory=list)
    convergence_data: dict = field(default_factory=dict)
    total_evaluations: int = 0
    elapsed_time: float = 0.0
    algorithm_used: str = ""
    surrogate_accuracy: dict = None


# ══════════════════════════════════════════════════════════════════════════════
#  DEFAULT BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def get_default_design_variables(state) -> list:
    """Build full list of design variables from the current rocket state."""
    s = state
    dv = []

    def _add(name, display, cat, lo, hi, cur, vtype="continuous", opts=None):
        cur = float(cur) if cur else (lo + hi) / 2
        cur = max(lo, min(hi, cur))
        dv.append(DesignVariable(
            name=name, display_name=display, category=cat,
            min_val=lo, max_val=hi, current_val=cur,
            enabled=False, var_type=vtype,
            discrete_options=opts or [],
        ))

    d = getattr(s, "diameter", 0.08) or 0.08
    L = getattr(s, "length", 1.5) or 1.5
    nl = getattr(s, "nose_length", L * 0.2) or L * 0.2

    # Geometry
    _add("diameter", "Body Diameter", "Geometry", 0.03, 0.30, d)
    _add("length", "Body Length", "Geometry", 0.3, 3.0, L)
    _add("nose_length", "Nose Length", "Geometry", 0.05, 0.8, nl)
    _add("fin_span", "Fin Span", "Geometry", 0.02, 0.25,
         getattr(s, "fin_span", d * 0.6) or d * 0.6)
    _add("fin_root_chord", "Fin Root Chord", "Geometry", 0.03, 0.40,
         getattr(s, "fin_root_chord", L * 0.08) or L * 0.08)
    _add("fin_tip_chord", "Fin Tip Chord", "Geometry", 0.01, 0.20,
         getattr(s, "fin_tip_chord", L * 0.04) or L * 0.04)
    _add("fin_sweep_angle", "Fin Sweep", "Geometry", 0.0, 1.05,
         getattr(s, "fin_sweep_angle", 0.0))
    _add("fin_thickness", "Fin Thickness", "Geometry", 0.001, 0.01,
         getattr(s, "fin_thickness", 0.003) or 0.003)
    _add("fin_count", "Number of Fins", "Geometry", 3, 6,
         getattr(s, "fin_count", 4) or 4, vtype="integer")

    # Mass
    _add("dry_mass", "Dry Mass", "Mass", 0.1, 20.0,
         getattr(s, "dry_mass", 1.0) or 1.0)

    # Propulsion
    _add("motor_total_impulse", "Total Impulse", "Propulsion", 5, 5000,
         getattr(s, "motor_total_impulse", 200) or 200)
    _add("motor_burn_time", "Burn Time", "Propulsion", 0.3, 10.0,
         getattr(s, "motor_burn_time", 1.8) or 1.8)
    _add("propellant_mass", "Propellant Mass", "Propulsion", 0.01, 5.0,
         getattr(s, "propellant_mass", 0.1) or 0.1)

    # Recovery
    _add("drogue_cd_area", "Drogue CdA", "Recovery", 0.05, 2.0,
         getattr(s, "drogue_cd_area", 0.3) or 0.3)
    _add("main_cd_area", "Main CdA", "Recovery", 0.5, 10.0,
         getattr(s, "main_cd_area", 3.0) or 3.0)
    _add("main_deploy_altitude", "Main Deploy Alt", "Recovery", 100, 600,
         getattr(s, "main_deploy_altitude", 300) or 300)

    # Aerodynamics
    _add("cd", "Cd Correction", "Aerodynamics", 0.1, 1.5,
         getattr(s, "cd", 0.5) or 0.5)

    return dv


def get_default_objectives() -> list:
    return [
        ObjectiveFunction("max_apogee", "Max Apogee", "maximize", 1.0, True),
        ObjectiveFunction("max_rail_exit_velocity", "Max Rail Exit Vel", "maximize", 1.0, False),
        ObjectiveFunction("max_velocity", "Max Velocity", "maximize", 1.0, False),
        ObjectiveFunction("max_payload_fraction", "Max Inert Mass Frac", "maximize", 1.0, False),
        ObjectiveFunction("max_stability_margin", "Max Stability Margin", "maximize", 1.0, False),
        ObjectiveFunction("min_landing_distance", "Min Landing Distance", "minimize", 1.0, False),
        ObjectiveFunction("max_prob_target", "Max P(Target Alt)", "maximize", 1.0, False),
        ObjectiveFunction("max_mission_success", "Max Mission Success", "maximize", 1.0, False),
        ObjectiveFunction("min_mass", "Min Mass", "minimize", 1.0, False),
        ObjectiveFunction("min_cost", "Min Cost", "minimize", 1.0, False),
    ]


def get_default_constraints() -> list:
    return [
        Constraint("stability_min", "Stability > 1.2", "greater_than", 1.2, 1000.0, True),
        Constraint("stability_max", "Stability < 3.0", "less_than", 3.0, 1000.0, True),
        Constraint("rail_exit_min", "Rail Exit > 15", "greater_than", 15.0, 1000.0, True),
        Constraint("mach_max", "Mach < 2", "less_than", 2.0, 1000.0, False),
        Constraint("accel_max", "Accel < 100G", "less_than", 981.0, 500.0, False),
        Constraint("safety_factor_min", "SF > 2.0", "greater_than", 2.0, 500.0, False),
        Constraint("landing_dist_max", "Landing < 1000m", "less_than", 1000.0, 500.0, False),
        Constraint("mass_max", "Mass < 50kg", "less_than", 50.0, 500.0, False),
        Constraint("diameter_min", "Diam > 0.03m", "greater_than", 0.03, 500.0, False),
    ]


def get_default_correlations() -> list:
    return [
        CorrelationEntry("motor_total_impulse", "motor_burn_time", 0.85),
        CorrelationEntry("dry_mass", "length", 0.70),
        CorrelationEntry("fin_span", "fin_root_chord", 0.50),
        CorrelationEntry("drogue_cd_area", "main_cd_area", 0.40),
    ]


# ══════════════════════════════════════════════════════════════════════════════
#  PHYSICS VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def _geom_value(variables: dict, base, key: str, default: float) -> float:
    """Resolve a geometry/mass value for a candidate.

    A candidate's ``variables`` dict only carries the quantities the optimiser
    is actually varying. Everything else still has a real value — the rocket's
    own — and reading a hardcoded default instead silently validated and
    repaired candidates against a fictional 80 mm / 1 m airframe. Prefer the
    candidate's value, then *base* (the BatchSimConfig built from the rocket
    state), and only then the literal fallback.
    """
    val = variables.get(key)
    if val is None and base is not None:
        val = getattr(base, key, None)
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def validate_candidate(variables: dict, design_vars: list, base=None) -> tuple:
    """Validate that a candidate design is physically realisable.

    *base* is the BatchSimConfig the candidate is built on; dimensions the
    optimiser is not varying are read from it rather than guessed.

    Returns (valid: bool, warnings: list[str]).
    """
    warnings = []
    v = variables

    d = _geom_value(v, base, "diameter", 0.08)
    L = _geom_value(v, base, "length", 1.0)
    fs = _geom_value(v, base, "fin_span", 0.05)
    frc = _geom_value(v, base, "fin_root_chord", 0.1)
    ftc = _geom_value(v, base, "fin_tip_chord", 0.05)
    dm = _geom_value(v, base, "dry_mass", 1.0)
    nl = _geom_value(v, base, "nose_length", 0.2)

    if d <= 0.01:
        warnings.append(f"Diameter too small: {d:.4f} m")
    if L <= d:
        warnings.append(f"Length ({L:.3f}) must exceed diameter ({d:.4f})")
    if dm <= 0:
        warnings.append(f"Negative dry mass: {dm:.3f} kg")
    if fs <= 0:
        warnings.append(f"Fin span must be positive: {fs:.4f}")
    if fs > 5 * d:
        warnings.append(f"Fin span ({fs:.3f}) exceeds 5× diameter ({5*d:.3f})")
    if frc <= 0:
        warnings.append(f"Fin root chord must be positive: {frc:.4f}")
    if ftc < 0:
        warnings.append(f"Fin tip chord negative: {ftc:.4f}")
    if ftc > frc:
        warnings.append(f"Tip chord ({ftc:.4f}) > root chord ({frc:.4f})")
    if nl <= 0:
        warnings.append(f"Nose length must be positive: {nl:.4f}")
    if nl > L * 0.8:
        warnings.append(f"Nose length ({nl:.3f}) > 80% of body ({L*0.8:.3f})")

    # The fin-count spin box's own range floor is vmin*0.1, so a user can set
    # the minimum below 1 and have the sampler draw a finless rocket.
    fc = _geom_value(v, base, "fin_count", 4)
    if fc < 1:
        warnings.append(f"Fin count must be at least 1: {fc:.0f}")

    cd = v.get("cd", 0.5)
    if cd < 0.05 or cd > 3.0:
        warnings.append(f"Cd out of physical range: {cd:.3f}")

    return len(warnings) == 0, warnings


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG BUILDING & CORRELATED MONTE-CARLO
# ══════════════════════════════════════════════════════════════════════════════

def build_candidate_config(base_config: BatchSimConfig,
                           variables: dict,
                           design_vars: list) -> BatchSimConfig:
    """Apply candidate variable values onto a base BatchSimConfig."""
    cfg = copy.deepcopy(base_config)
    for dv in design_vars:
        if not dv.enabled:
            continue
        key = dv.name
        val = variables.get(key)
        if val is None:
            continue

        # Map variable names to config attributes
        if key == "fin_span":
            cfg.fin_height = val
            cfg.fin_span = val
        elif key == "motor_designation":
            cfg.motor_designation = val
            if val in _MOTOR_PROPS:
                imp, bt, pm, at, mt, isp = _MOTOR_PROPS[val]
                cfg.motor_total_impulse = imp
                cfg.motor_burn_time = bt
                cfg.propellant_mass = pm
                cfg.motor_avg_thrust = at
                cfg.motor_max_thrust = mt
                cfg.motor_isp = isp
        elif hasattr(cfg, key):
            setattr(cfg, key, val)

    # Axial stations must follow the body when the optimizer changes its length.
    # cg / dry_cg / motor_position / fin_position are absolute distances from
    # the nose; leaving them at the base rocket's values while the body is
    # stretched meant the candidate's CG, motor and fins all stayed put. A
    # rocket stretched from 1.6 m to 3.2 m reported a bit-identical 6.44 cal
    # stability margin with its fins now sitting mid-body — so the search could
    # buy length for free, with no stability consequence at all.
    #
    # Holding each station's FRACTION of body length is the same assumption
    # RocketStateEngine._recompute_derived makes when it auto-estimates
    # (body_cg = 0.45 L, motor_cg = 0.85 L), and it preserves a high-fidelity
    # assembly's real CG fraction rather than inventing one. CP needs no such
    # treatment: AeroModel recomputes it from the candidate's own geometry.
    if base_config.length > 0 and abs(cfg.length - base_config.length) > 1e-12:
        _len_scale = cfg.length / base_config.length
        for _station in ("cg", "dry_cg", "motor_position", "fin_position"):
            setattr(cfg, _station, getattr(base_config, _station, 0.0) * _len_scale)

    # Derived fields
    if cfg.motor_burn_time > 0 and cfg.motor_total_impulse > 0:
        cfg.motor_avg_thrust = cfg.motor_total_impulse / cfg.motor_burn_time

    # Peak thrust must track the candidate's own average, not the base motor's.
    # `max(cfg.motor_max_thrust, ...)` let a shrunk motor keep the inherited
    # peak (base 260 N still reported for a 20 N·s candidate), which then fed
    # the accel_max constraint and the structural safety factor. Rescale the
    # base peak:average ratio instead, and keep it physical.
    if cfg.motor_avg_thrust > 0:
        base_ratio = (base_config.motor_max_thrust / base_config.motor_avg_thrust
                      if base_config.motor_avg_thrust > 0 else 1.3)
        base_ratio = max(1.05, min(3.0, base_ratio))
        cfg.motor_max_thrust = cfg.motor_avg_thrust * base_ratio

    # Enforce the rocket equation's bookkeeping: I = m_prop · Isp · g0. Any two
    # of (impulse, propellant mass, Isp) fix the third, so whichever the
    # optimizer is free to vary, the trio must stay consistent — otherwise
    # impulse is bought for free (0.01 kg delivering 5000 N·s ⇒ Isp ~51000 s)
    # and apogees run away to tens of km.
    #
    # Previously this only fired when impulse was optimised and propellant was
    # NOT, so enabling BOTH (adjacent checkboxes in the Propulsion group)
    # reopened the hole. Now the consistency is unconditional: propellant mass
    # is the dependent quantity unless the user is optimising it directly, in
    # which case Isp is re-derived and sanity-clamped.
    prop_optimized = any(dv.enabled and dv.name == "propellant_mass" for dv in design_vars)
    isp = cfg.motor_isp if cfg.motor_isp > 10 else 200.0

    if cfg.motor_total_impulse > 0:
        if prop_optimized and cfg.propellant_mass > 0:
            # Propellant is a free variable — impulse and propellant together
            # imply an Isp. Clamp it to a physically achievable band for solid
            # / hybrid / liquid chemical propulsion and rescale impulse to match
            # whatever Isp survived the clamp.
            implied_isp = cfg.motor_total_impulse / (cfg.propellant_mass * 9.80665)
            cfg.motor_isp = max(50.0, min(450.0, implied_isp))
            cfg.motor_total_impulse = cfg.propellant_mass * cfg.motor_isp * 9.80665
            if cfg.motor_burn_time > 0:
                cfg.motor_avg_thrust = cfg.motor_total_impulse / cfg.motor_burn_time
                if cfg.motor_avg_thrust > 0:
                    base_ratio = (base_config.motor_max_thrust / base_config.motor_avg_thrust
                                  if base_config.motor_avg_thrust > 0 else 1.3)
                    cfg.motor_max_thrust = cfg.motor_avg_thrust * max(1.05, min(3.0, base_ratio))
        else:
            cfg.propellant_mass = cfg.motor_total_impulse / (isp * 9.80665)

    return cfg


def build_correlated_samples(base_config: BatchSimConfig,
                             variables: dict,
                             design_vars: list,
                             correlations: list,
                             n_mc: int,
                             rng: np.random.Generator) -> list:
    """Generate correlated Monte-Carlo config samples via Cholesky decomposition.

    Returns a list of BatchSimConfigs with correlated perturbations applied.
    """
    enabled = [dv for dv in design_vars if dv.enabled]
    n = len(enabled)
    if n == 0 or n_mc <= 0:
        return []

    # Build correlation matrix (default = identity)
    corr_matrix = np.eye(n)
    name_to_idx = {dv.name: i for i, dv in enumerate(enabled)}

    for ce in correlations:
        i = name_to_idx.get(ce.param1)
        j = name_to_idx.get(ce.param2)
        if i is not None and j is not None and i != j:
            rho = max(-0.99, min(0.99, ce.coefficient))
            corr_matrix[i, j] = rho
            corr_matrix[j, i] = rho

    # Ensure positive definite via nearest PD projection
    try:
        L = np.linalg.cholesky(corr_matrix)
    except np.linalg.LinAlgError:
        # Fall back: regularise
        eigvals = np.linalg.eigvalsh(corr_matrix)
        min_eig = eigvals.min()
        if min_eig < 0:
            corr_matrix += (-min_eig + 0.01) * np.eye(n)
        try:
            L = np.linalg.cholesky(corr_matrix)
        except np.linalg.LinAlgError:
            L = np.eye(n)

    # Standard-deviation for each variable (5% of range)
    stds = np.array([(dv.max_val - dv.min_val) * 0.05 for dv in enabled])

    # Generate correlated standard-normal samples → transform
    Z = rng.standard_normal((n_mc, n))
    correlated = Z @ L.T                   # correlated std-normals
    perturbations = correlated * stds      # scale to physical units

    configs = []
    for k in range(n_mc):
        v_copy = dict(variables)
        for idx, dv in enumerate(enabled):
            base_val = variables.get(dv.name, dv.current_val)
            perturbed = base_val + perturbations[k, idx]
            # Clamp to bounds
            perturbed = max(dv.min_val, min(dv.max_val, perturbed))
            if dv.var_type == "integer":
                perturbed = round(perturbed)
            v_copy[dv.name] = perturbed

        cfg = build_candidate_config(base_config, v_copy, design_vars)
        configs.append(cfg)

    return configs


# ══════════════════════════════════════════════════════════════════════════════
#  CANDIDATE EVALUATION  (MDO — trajectory + structures + aero + recovery)
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_candidate(base_config: BatchSimConfig,
                       variables: dict,
                       design_vars: list,
                       objectives: list,
                       constraints: list,
                       correlations: list,
                       mc_sims: int,
                       seed: int,
                       target_apogee: float = 0.0,
                       mission_mode: bool = False,
                       robust_mode: bool = False) -> CandidateDesign:
    """Full MDO evaluation of one candidate design.

    Runs *mc_sims* batch simulations (with correlated perturbations if
    correlations are provided), then aggregates trajectory metrics and
    computes structural / aerodynamic / recovery discipline outputs.
    """
    rng = np.random.default_rng(seed)

    # Build candidate config
    cfg = build_candidate_config(base_config, variables, design_vars)

    # Run MC simulations (correlated if applicable)
    if mc_sims > 1 and correlations:
        mc_configs = build_correlated_samples(
            base_config, variables, design_vars, correlations, mc_sims, rng)
    else:
        mc_configs = [cfg] * max(mc_sims, 1)

    apogees, machs, accels, stabs = [], [], [], []
    rail_exits, successes, velocities = [], [], []
    # Only runs that actually reached the ground contribute a landing distance.
    landed_dists: list = []
    n_no_landing = 0

    for i, mc_cfg in enumerate(mc_configs):
        try:
            res = run_batch_simulation(mc_cfg, seed=seed + i * 7)
            # Zero the apogee ONLY for a numerically diverged run (the >2000 m/s
            # or >100 km guard truncated it) — that trajectory is meaningless.
            # A run that merely ran out the simulation window still climbed to a
            # real apogee: apogee happens early, only the descent is cut short.
            # Scoring those as 0 m steered the optimizer away from exactly the
            # high-flying designs it was asked to find.
            apogees.append(0.0 if result_diverged(res) else res.apogee)
            machs.append(res.max_mach)
            accels.append(res.max_acceleration)
            stabs.append(res.min_stability_margin)
            if res.final_phase == "Landed":
                landed_dists.append(res.landing_distance)
            else:
                # Still airborne when the run ended — its position is mid-air,
                # a lower bound on where it comes down, not a landing point.
                n_no_landing += 1
            rail_exits.append(res.rail_exit_velocity)
            successes.append(1.0 if res.success else 0.0)
            velocities.append(res.max_velocity)
        except Exception:
            apogees.append(0.0)
            machs.append(0.0)
            accels.append(0.0)
            stabs.append(0.0)
            n_no_landing += 1
            rail_exits.append(0.0)
            successes.append(0.0)
            velocities.append(0.0)

    arr_apogee = np.array(apogees)
    arr_mach = np.array(machs)
    arr_stab = np.array(stabs)
    # No verified landing at all → the sentinel the failed-run path already
    # uses, so "we could not confirm where this lands" reads as a bad landing
    # rather than as a suspiciously short one.
    arr_landing = np.array(landed_dists) if landed_dists else np.array([9999.0])
    arr_rail = np.array(rail_exits)
    arr_accel = np.array(accels)
    arr_success = np.array(successes)

    # MC statistics
    mean_apogee = float(np.mean(arr_apogee))
    std_apogee = float(np.std(arr_apogee)) if len(arr_apogee) > 1 else 0.0
    success_rate = float(np.mean(arr_success))

    # Confidence interval
    ci_low = float(np.percentile(arr_apogee, 2.5)) if len(arr_apogee) > 1 else mean_apogee
    ci_high = float(np.percentile(arr_apogee, 97.5)) if len(arr_apogee) > 1 else mean_apogee

    # Higher moments
    skewness = 0.0
    kurtosis = 0.0
    if len(arr_apogee) > 2 and std_apogee > 0:
        skewness = float(np.mean(((arr_apogee - mean_apogee) / std_apogee) ** 3))
        kurtosis = float(np.mean(((arr_apogee - mean_apogee) / std_apogee) ** 4) - 3.0)

    # P(target altitude)
    p_target = 0.0
    if target_apogee > 0 and len(arr_apogee) > 0:
        tol = target_apogee * 0.10
        within = np.sum(np.abs(arr_apogee - target_apogee) <= tol)
        p_target = float(within / len(arr_apogee))

    mc_stats = {
        "mean_apogee": mean_apogee,
        "std_apogee": std_apogee,
        "min_apogee": float(np.min(arr_apogee)),
        "max_apogee": float(np.max(arr_apogee)),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "mean_mach": float(np.mean(arr_mach)),
        "mean_stability": float(np.mean(arr_stab)),
        "mean_landing_dist": float(np.mean(arr_landing)),
        "mean_rail_exit": float(np.mean(arr_rail)),
        "mean_accel": float(np.mean(arr_accel)),
        "success_rate": success_rate,
        "n_no_landing": n_no_landing,
        "p_target": p_target,
        "p5_apogee": float(np.percentile(arr_apogee, 5)) if len(arr_apogee) > 1 else mean_apogee,
        # Per-objective MC samples, keyed by objective name. Robust modes used
        # to read apogee statistics for EVERY objective, so a robust "worst
        # case" on landing distance silently optimised -max(apogee). Kept as
        # plain lists so the dict pickles cleanly to pool workers.
        "samples": {
            "max_apogee": arr_apogee.tolist(),
            "max_rail_exit_velocity": arr_rail.tolist(),
            "max_velocity": [float(v) for v in velocities],
            "max_stability_margin": arr_stab.tolist(),
            "min_landing_distance": arr_landing.tolist(),
            "max_mach": arr_mach.tolist(),
            "max_accel": arr_accel.tolist(),
        },
    }

    # ── Structural analysis ──
    sf_value = 999.0
    try:
        from physics.structures import compute_all
        d = variables.get("diameter", cfg.diameter)
        wall = 0.002
        force = cfg.motor_max_thrust if cfg.motor_max_thrust > 0 else 200.0
        struct = compute_all(force, d, wall, variables.get("length", cfg.length),
                             "Aluminum 6061-T6")
        sf_value = struct.get("safety_factor", 999.0)
    except Exception:
        pass

    # ── Aerodynamic CP / stability check ──
    aero_stability = float(np.mean(arr_stab))

    # ── Recovery — terminal velocity ──
    landing_vel = 0.0
    try:
        from recovery.parachute_dynamics import ParachuteDynamics
        main_cda = variables.get("main_cd_area", cfg.main_cd_area)
        if main_cda > 0:
            total_mass = variables.get("dry_mass", cfg.dry_mass) + cfg.propellant_mass * 0.05
            rho_sl = 1.225
            landing_vel = math.sqrt(2 * total_mass * 9.81 / (rho_sl * main_cda))
    except Exception:
        pass

    # ── Build objectives dict ──
    obj_vals = {
        "max_apogee": mean_apogee,
        "max_rail_exit_velocity": float(np.mean(arr_rail)),
        "max_velocity": float(np.mean(velocities)) if velocities else 0.0,
        "max_payload_fraction": 0.0,
        "max_stability_margin": aero_stability,
        "min_landing_distance": float(np.mean(arr_landing)),
        "max_prob_target": p_target,
        "max_mission_success": success_rate * (p_target if target_apogee > 0 else 1.0),
        "min_mass": variables.get("dry_mass", cfg.dry_mass),
        "min_cost": variables.get("dry_mass", cfg.dry_mass) * 50.0,
        "max_mach": float(np.mean(arr_mach)),
        "max_accel": float(np.mean(arr_accel)),
        "safety_factor": sf_value,
        "landing_velocity": landing_vel,
        "apogee": mean_apogee,
        "stability": aero_stability,
        "rail_exit_velocity": float(np.mean(arr_rail)),
    }

    # Inert (structure + payload) mass fraction = 1 - propellant/total. Not the
    # true payload fraction (payload mass isn't a design variable here).
    total_mass = variables.get("dry_mass", cfg.dry_mass) + cfg.propellant_mass
    if total_mass > 0:
        obj_vals["max_payload_fraction"] = 1.0 - (cfg.propellant_mass / total_mass)

    # ── Evaluate constraints ──
    cons_eval = {}
    for c in constraints:
        if not c.enabled:
            continue
        val = _constraint_value(c.name, obj_vals, variables, cfg)
        satisfied = True
        if c.type == "greater_than":
            satisfied = val >= c.limit
        elif c.type == "less_than":
            satisfied = val <= c.limit
        cons_eval[c.name] = {"value": val, "limit": c.limit, "satisfied": satisfied}

    feasible = all(info["satisfied"] for info in cons_eval.values())

    # ── Fitness ──
    if mission_mode and target_apogee > 0:
        fitness = _mission_fitness(obj_vals, mc_stats, target_apogee, objectives, constraints, cons_eval)
    elif robust_mode:
        fitness = _robust_fitness(obj_vals, mc_stats, objectives, constraints, cons_eval)
    else:
        fitness = _standard_fitness(obj_vals, objectives, constraints, cons_eval)

    # Expose scalar fitness as an objective so NSGA-II domination can track it
    # in mission mode (otherwise NSGA-II sorts on raw apogee and ignores target).
    obj_vals["fitness"] = fitness

    return CandidateDesign(
        variables=dict(variables),
        fitness=fitness,
        objectives=obj_vals,
        constraints_eval=cons_eval,
        feasible=feasible,
        mc_stats=mc_stats,
        batch_config=cfg,
    )


def _constraint_value(name: str, obj_vals: dict, variables: dict, cfg) -> float:
    """Extract the numeric value for a named constraint."""
    mapping = {
        "stability_min": "max_stability_margin",
        "stability_max": "max_stability_margin",
        "rail_exit_min": "max_rail_exit_velocity",
        "mach_max": "max_mach",
        "accel_max": "max_accel",
        "safety_factor_min": "safety_factor",
        "landing_dist_max": "min_landing_distance",
        "mass_max": "min_mass",
        "diameter_min": "diameter",
    }
    key = mapping.get(name, name)
    if key in obj_vals:
        return obj_vals[key]
    if key in variables:
        return float(variables[key])
    # Not an objective and not an optimised variable — it still has a real
    # value on the candidate's config. Returning 0.0 here made `diameter_min`
    # fail for every candidate whenever diameter wasn't being optimised, which
    # marked the whole population infeasible and penalised it uniformly.
    if cfg is not None:
        val = getattr(cfg, key, None)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                pass
    return 0.0


# Reference magnitudes used to normalise objectives before the weighted sum.
# Without them the sum mixes units outright: apogee (~10^3 m) buries stability
# (~2 cal) and payload fraction (~0.5), so a nominally equal-weighted run is in
# practice a pure apogee run. Dividing by these puts every objective near O(1),
# which is what makes the user's `weight` field mean what it says.
_OBJECTIVE_SCALES = {
    "max_apogee": 1000.0,            # m
    "max_rail_exit_velocity": 20.0,  # m/s
    "max_velocity": 200.0,           # m/s
    "max_payload_fraction": 1.0,     # fraction
    "max_stability_margin": 2.0,     # calibers
    "min_landing_distance": 500.0,   # m
    "max_prob_target": 1.0,          # probability
    "max_mission_success": 1.0,      # probability
    "min_mass": 5.0,                 # kg
    "min_cost": 250.0,               # currency units
    "max_mach": 1.0,                 # Mach
    "max_accel": 100.0,              # m/s^2
}


def _normalise(name: str, value: float) -> float:
    """Scale an objective value into O(1) units."""
    return value / _OBJECTIVE_SCALES.get(name, 1.0)


def _constraint_penalty(constraints, cons_eval) -> float:
    """Total quadratic penalty for violated constraints.

    Uses violation RELATIVE to the limit. The absolute form was unusable across
    constraints of different magnitude: a 500 m overshoot of a 1000 m landing
    limit cost 500 * 500^2 = 1.25e8, swamping every objective and every other
    constraint, while a 0.3 caliber stability miss cost 90. Relative violation
    makes `penalty_weight` comparable between constraints, and squaring still
    grows the pressure as a design drifts further out of bounds.
    """
    penalty = 0.0
    for c in constraints:
        if not c.enabled:
            continue
        info = cons_eval.get(c.name)
        if info and not info["satisfied"]:
            scale = max(abs(info["limit"]), 1e-6)
            rel_violation = abs(info["value"] - info["limit"]) / scale
            penalty += c.penalty_weight * rel_violation ** 2
    return penalty


def _standard_fitness(obj_vals, objectives, constraints, cons_eval) -> float:
    """Weighted-sum fitness over normalised objectives, with penalties."""
    fitness = 0.0
    for o in objectives:
        if not o.enabled:
            continue
        val = _normalise(o.name, obj_vals.get(o.name, 0.0))
        if o.direction == "minimize":
            val = -val
        fitness += o.weight * val

    return fitness - _constraint_penalty(constraints, cons_eval)


def _robust_objective_value(o, obj_vals, mc_stats) -> float:
    """Robust statistic for ONE objective, in its own units.

    Every branch reads that objective's own MC samples. The previous version
    hardcoded apogee statistics regardless of `o.name`, so robust "worst" on
    Min Landing Distance optimised -max(apogee) instead.
    """
    samples = (mc_stats.get("samples") or {}).get(o.name)
    mean_val = obj_vals.get(o.name, 0.0)

    if o.robust_mode == "reliability":
        # Direction-independent: the fraction of MC runs that flew successfully.
        return mc_stats.get("success_rate", 0.0)

    if not samples:
        # No per-objective samples (derived metrics like payload fraction) —
        # fall back to the deterministic mean.
        return -mean_val if o.direction == "minimize" else mean_val

    arr = np.asarray(samples, dtype=np.float64)

    if o.robust_mode == "std":
        # Spread is always bad, whichever way the objective points.
        return -float(np.std(arr))

    if o.robust_mode == "worst":
        # Pessimistic tail: the low end when maximising, the high end when
        # minimising.
        return float(np.min(arr)) if o.direction == "maximize" else -float(np.max(arr))

    if o.robust_mode == "p5":
        # 5th percentile is only the pessimistic tail for a MAXIMISED objective;
        # for a minimised one the bad tail is the 95th.
        if o.direction == "maximize":
            return float(np.percentile(arr, 5))
        return -float(np.percentile(arr, 95))

    # "mean"
    return -mean_val if o.direction == "minimize" else mean_val


def _robust_fitness(obj_vals, mc_stats, objectives, constraints, cons_eval) -> float:
    """Robust fitness using each objective's own MC statistics."""
    fitness = 0.0
    for o in objectives:
        if not o.enabled:
            continue
        val = _robust_objective_value(o, obj_vals, mc_stats)
        # "reliability" is already a probability in [0, 1]; everything else is
        # in the objective's native units and needs the same normalisation the
        # standard fitness applies.
        if o.robust_mode != "reliability":
            val = _normalise(o.name, val)
        fitness += o.weight * val

    return fitness - _constraint_penalty(constraints, cons_eval)


def _mission_fitness(obj_vals, mc_stats, target, objectives, constraints, cons_eval) -> float:
    """Mission-driven fitness: maximise P(apogee within ±10% of target)."""
    p_target = mc_stats.get("p_target", 0.0)
    success = mc_stats.get("success_rate", 0.0)
    mean_apogee = mc_stats.get("mean_apogee", 0.0)

    # Primary: smooth proximity to target × success. A continuous proximity term
    # (1 at target, 0 at ≥100% error) replaces the old binary 1000·p_target,
    # which had a 1000-point cliff at the ±10% tolerance boundary that stalled
    # gradient-following — especially at low mc_sims where p_target is just 0/1.
    if target > 0 and mean_apogee > 0:
        relative_error = abs(mean_apogee - target) / target
    else:
        relative_error = 1.0
    proximity = max(0.0, 1.0 - relative_error)
    fitness = 1000.0 * success * proximity

    # Reliability bonus: still reward the fraction of MC runs inside tolerance,
    # but as a secondary term so it can't dominate the smooth proximity signal.
    fitness += 200.0 * p_target * success

    # Secondary objectives — normalised, so a large-magnitude secondary (e.g.
    # max_velocity in m/s) cannot outweigh the 1000-point mission term.
    for o in objectives:
        if not o.enabled or o.name in ("max_apogee", "max_prob_target", "max_mission_success"):
            continue
        val = _normalise(o.name, obj_vals.get(o.name, 0.0))
        if o.direction == "minimize":
            val = -val
        fitness += o.weight * 0.1 * val

    return fitness - _constraint_penalty(constraints, cons_eval)


# ══════════════════════════════════════════════════════════════════════════════
#  PARALLEL EVALUATION  (process pool — workers run evaluate_candidate)
# ══════════════════════════════════════════════════════════════════════════════

def _eval_task(args: tuple) -> CandidateDesign:
    """Top-level picklable worker: unpack one task tuple → evaluate_candidate.

    Must stay module-level so ProcessPoolExecutor can pickle it. Runs inside a
    worker process; touches no Qt objects.
    """
    (base_config, variables, design_vars, objectives, constraints,
     correlations, mc_sims, seed, target_apogee, mission_mode, robust_mode) = args
    return evaluate_candidate(
        base_config, variables, design_vars, objectives, constraints,
        correlations, mc_sims, seed, target_apogee, mission_mode, robust_mode)


def _make_executor(config: OptimizationConfig):
    """Build a ProcessPoolExecutor for this run, or None to run serially."""
    if not getattr(config, "parallel", True):
        return None
    n = getattr(config, "n_workers", 0) or 0
    if n <= 0:
        n = max(1, (os.cpu_count() or 2) - 1)
    if n <= 1:
        return None
    try:
        return ProcessPoolExecutor(max_workers=n)
    except Exception as e:                 # fall back to serial on any pool failure
        logger.warning(f"Process pool unavailable, running serial: {e}")
        return None


def _parallel_eval(executor,
                   base_config: BatchSimConfig,
                   var_dicts: list,
                   config: OptimizationConfig,
                   seeds: list,
                   on_result: Optional[Callable] = None,
                   cancel_flag: Optional[list] = None) -> list:
    """Evaluate a batch of candidate variable-dicts.

    Returns a list of CandidateDesign in the SAME ORDER as *var_dicts*. Seeds
    are assigned per-position so results are identical to the serial path
    regardless of completion order (deterministic). *on_result(i, cd)* is called
    as each candidate finishes — used for live progress.

    Slots left None mean "not evaluated" (cancelled); callers must filter them.
    A candidate whose evaluation *raised* is not None — it comes back as an
    infeasible design with -inf fitness, so one bad candidate cannot take the
    whole run down with it.
    """
    tasks = [
        (base_config, vd, config.design_variables, config.objectives,
         config.constraints, config.correlations, config.mc_sims_per_candidate,
         seed, config.target_apogee, config.mission_mode, config.robust_mode)
        for vd, seed in zip(var_dicts, seeds)
    ]
    results: list = [None] * len(tasks)

    if executor is None:
        for i, t in enumerate(tasks):
            if cancel_flag and cancel_flag[0]:
                break
            try:
                results[i] = _eval_task(t)
            except Exception as e:
                logger.warning(f"Candidate {i} evaluation failed: {e}")
                results[i] = _failed_candidate(var_dicts[i])
            if on_result:
                on_result(i, results[i])
        return results

    futures = {executor.submit(_eval_task, t): i for i, t in enumerate(tasks)}
    try:
        for fut in as_completed(futures):
            if cancel_flag and cancel_flag[0]:
                # Stop collecting and drop whatever is still queued. Remaining
                # slots stay None so the caller sees them as unevaluated.
                break
            i = futures[fut]
            try:
                results[i] = fut.result()
            except Exception as e:
                # A dead worker (BrokenProcessPool, pickling failure) must not
                # abort the whole optimisation — score the candidate as failed.
                logger.warning(f"Candidate {i} evaluation failed in worker: {e}")
                results[i] = _failed_candidate(var_dicts[i])
            if on_result:
                on_result(i, results[i])
    finally:
        for fut in futures:
            fut.cancel()
    return results


def _failed_candidate(variables: dict) -> CandidateDesign:
    """Placeholder for a candidate whose evaluation raised.

    -inf fitness guarantees selection/elitism never carries it forward, while
    keeping the population size (and every array indexed alongside it) intact.
    """
    return CandidateDesign(
        variables=dict(variables),
        fitness=float("-inf"),
        objectives={},
        constraints_eval={},
        feasible=False,
        mc_stats={},
    )


# ══════════════════════════════════════════════════════════════════════════════
#  SURROGATE  PRE-SCREENING
# ══════════════════════════════════════════════════════════════════════════════

def _encode_variables(variables: dict, enabled_dvs: list) -> np.ndarray:
    """Numeric feature row for one candidate.

    ``_dict_to_vec`` cannot be reused: it assumes every value is numeric, and a
    discrete variable such as the motor holds a string. Those are encoded as
    the option's index in ``discrete_options``.
    """
    row = np.zeros(len(enabled_dvs), dtype=float)
    for j, dv in enumerate(enabled_dvs):
        v = variables.get(dv.name, dv.current_val)
        if dv.var_type == "discrete" and dv.discrete_options:
            try:
                row[j] = float(dv.discrete_options.index(v))
            except ValueError:
                row[j] = 0.0
        else:
            try:
                row[j] = float(v)
            except (TypeError, ValueError):
                row[j] = 0.0
    return row


class _SurrogateScreener:
    """Surrogate-assisted pre-screening of offspring.

    Learns fitness as a function of the design vector from candidates the
    simulator has *already* scored, so building it costs no extra simulations.
    Each generation the algorithm proposes ``surrogate_pool_factor`` times more
    offspring than it can afford to simulate; the surrogate predicts their
    fitness and only the most promising batch is handed to the simulator.

    Every fitness that reaches the population is still a real simulation
    result — the surrogate only chooses *which* designs get simulated, it never
    supplies a fitness value. A badly fitted surrogate can therefore waste a
    generation, but it cannot corrupt the reported optimum.

    Note this does not cut the number of simulations per generation; it raises
    the quality of the designs those simulations are spent on, so the run
    should reach a given fitness in fewer generations.
    """

    POOL_CAP = 2000          # hard ceiling on predictions per generation
    EXPLORE_FRACTION = 0.25  # share of each batch kept unscreened, for diversity
    MAX_TRAIN = 4000         # cap training rows so refit cost stays bounded
    SCORE_EVERY = 5          # generations between (relatively costly) CV scores

    def __init__(self, config: OptimizationConfig, enabled_dvs: list,
                 rng: np.random.Generator):
        from core.surrogate_model import create_surrogate

        self._dvs = enabled_dvs
        self._rng = rng
        self._pool_factor = max(1, int(getattr(config, "surrogate_pool_factor", 4) or 4))
        self._model = create_surrogate(config.surrogate_type)
        self._X: list = []
        self._y: list = []
        self._dirty = False
        self._ready = False
        self._score: dict = None
        self._fits = 0
        self._screened = 0
        # Enough rows to have any hope of a signal: a few per design variable.
        self._min_train = max(20, 4 * len(enabled_dvs))

    # ── data ────────────────────────────────────────────────────────────────

    def observe(self, candidates: list):
        """Record real (simulated) evaluations as training data."""
        for cd in candidates:
            if cd is None or not np.isfinite(cd.fitness):
                continue
            self._X.append(_encode_variables(cd.variables, self._dvs))
            self._y.append(float(cd.fitness))
        if len(self._X) > self.MAX_TRAIN:
            self._X = self._X[-self.MAX_TRAIN:]
            self._y = self._y[-self.MAX_TRAIN:]
        self._dirty = True

    def pool_size(self, n_keep: int) -> int:
        """How many offspring the algorithm should propose for *n_keep* slots."""
        if len(self._X) < self._min_train:
            return n_keep
        return int(min(n_keep * self._pool_factor, self.POOL_CAP))

    # ── model ───────────────────────────────────────────────────────────────

    def _refit(self, generation: int):
        if not self._dirty or len(self._X) < self._min_train:
            return
        X = np.asarray(self._X, dtype=float)
        y = np.asarray(self._y, dtype=float)
        # A constant target carries no gradient to screen on, and several
        # models fail outright on it.
        if float(np.std(y)) <= 0.0:
            self._ready = False
            self._dirty = False
            return
        try:
            self._model.fit(X, y)
            self._ready = True
            self._fits += 1
            self._dirty = False
        except Exception as e:
            logger.warning(f"Surrogate refit failed, screening disabled this "
                           f"generation: {e}")
            self._ready = False
            self._dirty = False
            return
        if generation % self.SCORE_EVERY == 0:
            self._rescore()

    def _rescore(self):
        try:
            self._score = self._model.score()
            logger.info(
                f"Surrogate ({type(self._model).__name__}): CV R²="
                f"{self._score['r2']:.3f} train R²={self._score['r2_train']:.3f} "
                f"n={self._score['n_samples']}")
        except Exception as e:
            logger.debug(f"Surrogate scoring failed: {e}")

    # ── screening ───────────────────────────────────────────────────────────

    def screen(self, child_dicts: list, n_keep: int, generation: int = 0) -> list:
        """Cut a proposed pool down to the *n_keep* designs worth simulating."""
        self._refit(generation)
        if not self._ready or len(child_dicts) <= n_keep:
            return child_dicts[:n_keep]

        try:
            X = np.asarray([_encode_variables(d, self._dvs) for d in child_dicts],
                           dtype=float)
            pred = np.asarray(self._model.predict(X), dtype=float)
        except Exception as e:
            logger.warning(f"Surrogate prediction failed, taking pool head: {e}")
            return child_dicts[:n_keep]

        pred = np.where(np.isfinite(pred), pred, -np.inf)
        order = np.argsort(pred)[::-1]

        # Keep a random slice as well as the predicted best. Trusting the
        # ranking completely would collapse diversity whenever the surrogate is
        # confidently wrong — and on early generations it usually is.
        n_explore = min(int(round(n_keep * self.EXPLORE_FRACTION)),
                        len(child_dicts) - n_keep)
        n_exploit = n_keep - n_explore

        chosen = list(order[:n_exploit])
        rest = order[n_exploit:]
        if n_explore > 0 and len(rest) > 0:
            chosen += list(self._rng.choice(rest, size=min(n_explore, len(rest)),
                                            replace=False))
        self._screened += 1
        return [child_dicts[i] for i in chosen[:n_keep]]

    # ── reporting ───────────────────────────────────────────────────────────

    def final_score(self) -> Optional[dict]:
        """Cross-validated accuracy for the result panel, or None if unused."""
        if self._dirty and len(self._X) >= self._min_train:
            try:
                self._model.fit(np.asarray(self._X, dtype=float),
                                np.asarray(self._y, dtype=float))
                self._ready = True
                self._dirty = False
            except Exception as e:
                logger.debug(f"Final surrogate fit failed: {e}")
        if not self._ready:
            return None
        self._rescore()
        if self._score is None:
            return None
        out = dict(self._score)
        out.update({"model": type(self._model).__name__,
                    "generations_screened": self._screened,
                    "refits": self._fits})
        return out


class _NullScreener:
    """No-op stand-in used when surrogate acceleration is off."""

    def observe(self, candidates: list):
        pass

    def pool_size(self, n_keep: int) -> int:
        return n_keep

    def screen(self, child_dicts: list, n_keep: int, generation: int = 0) -> list:
        return child_dicts[:n_keep]

    def final_score(self):
        return None


def _make_screener(config: OptimizationConfig, enabled_dvs: list,
                   rng: np.random.Generator):
    """Build a screener for this run, falling back to the no-op on any problem."""
    if not getattr(config, "use_surrogate", False) or not enabled_dvs:
        return _NullScreener()
    try:
        s = _SurrogateScreener(config, enabled_dvs, rng)
        logger.info(f"Surrogate screening on: {config.surrogate_type}, "
                    f"pool ×{s._pool_factor}, min train {s._min_train}")
        return s
    except Exception as e:
        # sklearn/scipy missing, or an unknown model name — the run must still
        # go ahead, just without acceleration.
        logger.warning(f"Surrogate unavailable, running without screening: {e}")
        return _NullScreener()


# ══════════════════════════════════════════════════════════════════════════════
#  GENETIC OPERATORS  (pure, no Qt)
# ══════════════════════════════════════════════════════════════════════════════

def _random_individual(design_vars: list, rng: np.random.Generator) -> dict:
    """Create a random individual within variable bounds."""
    ind = {}
    for dv in design_vars:
        if not dv.enabled:
            ind[dv.name] = dv.current_val
            continue
        if dv.var_type == "discrete" and dv.discrete_options:
            ind[dv.name] = rng.choice(dv.discrete_options)
        elif dv.var_type == "integer":
            ind[dv.name] = int(rng.integers(int(dv.min_val), int(dv.max_val) + 1))
        else:
            ind[dv.name] = float(rng.uniform(dv.min_val, dv.max_val))
    return ind


def _repair_individual(ind: dict, design_vars: list, base=None) -> dict:
    """Pull a candidate back into the physically realisable region.

    Only the initial population was ever validated — crossover and mutation
    then produced offspring with tip chord > root chord, nose > 80% of body,
    fin span > 5x diameter and so on, which went straight to the simulator from
    generation 1 onward. Repairing (rather than rejecting) keeps the search
    inside the feasible region without throwing away the evaluation budget.

    Dimensions the optimiser is NOT varying come from *base* (the rocket's own
    geometry). Reading a hardcoded default for them was actively destructive:
    with only tip chord enabled, root chord read as 0.0, so every child had its
    tip chord clamped to its lower bound — the variable was frozen there for
    the whole run.

    Only keys the candidate actually owns are written; a fixed dimension is not
    the repair's to change. Each correction is re-clamped to the variable's own
    bounds, so a repair can never push a value outside the box the user
    configured. Fixes are ordered by dependency: body proportions first, then
    everything measured against them.
    """
    r = dict(ind)
    bounds = {dv.name: (dv.min_val, dv.max_val) for dv in design_vars}

    def _get(key, default):
        return _geom_value(r, base, key, default)

    def _set(key, value):
        # Not an optimised variable → nothing to repair, keep the real value.
        if key not in r:
            return _get(key, value)
        lo, hi = bounds.get(key, (None, None))
        if lo is not None:
            value = max(lo, min(hi, value))
        r[key] = value
        return value

    d = _get("diameter", 0.08)
    L = _get("length", 1.0)

    # Body must be longer than it is wide.
    if L <= d:
        L = _set("length", d * 1.5)
        if L <= d:                       # bounds too tight to fix via length
            d = _set("diameter", L / 1.5)

    # Nose cone cannot eat more than 80% of the body.
    if _get("nose_length", 0.0) > L * 0.8:
        _set("nose_length", L * 0.8)

    # Fin span capped at 5x body diameter.
    if _get("fin_span", 0.0) > 5.0 * d:
        _set("fin_span", 5.0 * d)

    # Tip chord cannot exceed root chord.
    frc = _get("fin_root_chord", 0.0)
    if frc > 0.0 and _get("fin_tip_chord", 0.0) > frc:
        _set("fin_tip_chord", frc)

    return r


def _valid_individual(design_vars: list, enabled_dvs: list,
                      rng: np.random.Generator, tries: int = 10,
                      base=None) -> dict:
    """Random individual that passes physical validation, resampling up to
    *tries* times. Returns the last attempt if none validate (bounds still
    clamped, sim is robust to it)."""
    ind = _random_individual(design_vars, rng)
    for _ in range(tries):
        valid, _ = validate_candidate(ind, enabled_dvs, base)
        if valid:
            break
        ind = _random_individual(design_vars, rng)
    return ind


def _valid_vector(pop_vec: np.ndarray, idx: int, enabled_dvs: list,
                  all_dvs: list, lo: np.ndarray, hi: np.ndarray,
                  rng: np.random.Generator, tries: int = 10,
                  base=None) -> None:
    """Resample row *idx* of *pop_vec* in place until it validates (or tries
    exhausted). Used by DE/PSO whose populations are numpy vectors."""
    for _ in range(tries):
        d = _vec_to_dict(pop_vec[idx], enabled_dvs, all_dvs)
        valid, _ = validate_candidate(d, enabled_dvs, base)
        if valid:
            return
        pop_vec[idx] = rng.uniform(lo, hi)


def _sbx_crossover(p1: dict, p2: dict, design_vars: list,
                    eta: float, rng: np.random.Generator) -> tuple:
    """Simulated Binary Crossover for continuous variables."""
    c1, c2 = dict(p1), dict(p2)
    for dv in design_vars:
        if not dv.enabled or dv.var_type != "continuous":
            continue
        k = dv.name
        if rng.random() > 0.5:
            continue
        x1, x2 = p1[k], p2[k]
        if abs(x1 - x2) < 1e-14:
            continue
        u = rng.random()
        if u <= 0.5:
            beta = (2.0 * u) ** (1.0 / (eta + 1.0))
        else:
            beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0))
        c1[k] = 0.5 * ((1 + beta) * x1 + (1 - beta) * x2)
        c2[k] = 0.5 * ((1 - beta) * x1 + (1 + beta) * x2)
        c1[k] = max(dv.min_val, min(dv.max_val, c1[k]))
        c2[k] = max(dv.min_val, min(dv.max_val, c2[k]))
    # Integer / discrete: uniform crossover
    for dv in design_vars:
        if not dv.enabled or dv.var_type == "continuous":
            continue
        k = dv.name
        if rng.random() < 0.5:
            c1[k], c2[k] = c2[k], c1[k]
    return c1, c2


def _polynomial_mutation(ind: dict, design_vars: list,
                          eta: float, rate: float,
                          rng: np.random.Generator) -> dict:
    """Polynomial mutation for continuous; random reset for integer/discrete."""
    m = dict(ind)
    for dv in design_vars:
        if not dv.enabled:
            continue
        if rng.random() > rate:
            continue
        k = dv.name
        if dv.var_type == "discrete" and dv.discrete_options:
            m[k] = rng.choice(dv.discrete_options)
        elif dv.var_type == "integer":
            m[k] = int(rng.integers(int(dv.min_val), int(dv.max_val) + 1))
        else:
            x = m[k]
            delta_l = (x - dv.min_val) / max(dv.max_val - dv.min_val, 1e-12)
            delta_r = (dv.max_val - x) / max(dv.max_val - dv.min_val, 1e-12)
            u = rng.random()
            if u < 0.5:
                xy = 1.0 - delta_l
                val = (2.0 * u + (1.0 - 2.0 * u) * xy ** (eta + 1.0)) ** (1.0 / (eta + 1.0)) - 1.0
            else:
                xy = 1.0 - delta_r
                val = 1.0 - (2.0 * (1.0 - u) + 2.0 * (u - 0.5) * xy ** (eta + 1.0)) ** (1.0 / (eta + 1.0))
            m[k] = x + val * (dv.max_val - dv.min_val)
            m[k] = max(dv.min_val, min(dv.max_val, m[k]))
    return m


def _tournament_select(pop: list, k: int, rng: np.random.Generator) -> CandidateDesign:
    """Tournament selection: pick best of k random individuals."""
    contenders = rng.choice(len(pop), size=min(k, len(pop)), replace=False)
    best = pop[contenders[0]]
    for idx in contenders[1:]:
        if pop[idx].fitness > best.fitness:
            best = pop[idx]
    return best


# ══════════════════════════════════════════════════════════════════════════════
#  NSGA-II  HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _is_failed(cd: CandidateDesign) -> bool:
    """True for a placeholder from a candidate whose evaluation raised.

    Its ``objectives`` dict is empty, so the ``.get(name, 0.0)`` defaults used
    for domination and crowding would read as a *perfect* score on any
    minimised objective. Every ranking path has to skip these explicitly.
    """
    return not math.isfinite(cd.fitness)


def _finite_fitnesses(pop: list) -> list:
    """Fitnesses of the evaluable designs — for generation summary stats."""
    return [c.fitness for c in pop if not _is_failed(c)]


def _dominates(a: CandidateDesign, b: CandidateDesign, objectives: list) -> bool:
    """Return True if design *a* dominates design *b*."""
    a_failed, b_failed = _is_failed(a), _is_failed(b)
    if a_failed or b_failed:
        # A failed evaluation is dominated by anything real, dominates nothing.
        return b_failed and not a_failed

    dominated_in_all = True
    strictly_better_in_one = False
    for o in objectives:
        if not o.enabled:
            continue
        va = a.objectives.get(o.name, 0.0)
        vb = b.objectives.get(o.name, 0.0)
        if o.direction == "minimize":
            va, vb = -va, -vb
        if va < vb:
            dominated_in_all = False
        if va > vb:
            strictly_better_in_one = True
    return dominated_in_all and strictly_better_in_one


def _fast_non_dominated_sort(pop: list, objectives: list) -> list:
    """Fast non-dominated sorting (Deb et al., 2002). Returns list of fronts."""
    n = len(pop)
    S = [[] for _ in range(n)]
    dom_count = [0] * n
    fronts = [[]]

    for i in range(n):
        for j in range(i + 1, n):
            if _dominates(pop[i], pop[j], objectives):
                S[i].append(j)
                dom_count[j] += 1
            elif _dominates(pop[j], pop[i], objectives):
                S[j].append(i)
                dom_count[i] += 1

    for i in range(n):
        if dom_count[i] == 0:
            pop[i].rank = 0
            fronts[0].append(i)

    k = 0
    while fronts[k]:
        next_front = []
        for i in fronts[k]:
            for j in S[i]:
                dom_count[j] -= 1
                if dom_count[j] == 0:
                    pop[j].rank = k + 1
                    next_front.append(j)
        k += 1
        fronts.append(next_front)

    return [f for f in fronts if f]


def _pareto_from_population(population: list, objectives: list, best) -> list:
    """Non-dominated set of a final population.

    Lets the single-objective drivers (GA / DE / PSO) still produce a real
    Pareto front whenever >=2 objectives are enabled — otherwise their front
    is just ``[best]`` and the UI shows the empty-front placeholder. Falls
    back to ``[best]`` for the genuine single-objective case.
    """
    enabled = [o for o in objectives if getattr(o, "enabled", True)]
    if len(enabled) >= 2 and population:
        fronts = _fast_non_dominated_sort(population, objectives)
        if fronts and fronts[0]:
            return [population[i] for i in fronts[0]]
    return [best] if best is not None else []


def _crowding_distance(pop: list, front: list, objectives: list):
    """Assign crowding distance to individuals in a front."""
    n = len(front)
    if n <= 2:
        for idx in front:
            pop[idx].crowding_distance = float("inf")
        return

    for idx in front:
        pop[idx].crowding_distance = 0.0

    for o in objectives:
        if not o.enabled:
            continue
        sorted_front = sorted(front, key=lambda i: pop[i].objectives.get(o.name, 0.0))
        pop[sorted_front[0]].crowding_distance = float("inf")
        pop[sorted_front[-1]].crowding_distance = float("inf")

        f_min = pop[sorted_front[0]].objectives.get(o.name, 0.0)
        f_max = pop[sorted_front[-1]].objectives.get(o.name, 0.0)
        denom = f_max - f_min if abs(f_max - f_min) > 1e-12 else 1.0

        for k in range(1, n - 1):
            prev_val = pop[sorted_front[k - 1]].objectives.get(o.name, 0.0)
            next_val = pop[sorted_front[k + 1]].objectives.get(o.name, 0.0)
            pop[sorted_front[k]].crowding_distance += (next_val - prev_val) / denom


# ══════════════════════════════════════════════════════════════════════════════
#  OPTIMISATION ALGORITHMS  (pure functions — no Qt)
# ══════════════════════════════════════════════════════════════════════════════

def _run_genetic_algorithm(config: OptimizationConfig,
                           base_config: BatchSimConfig,
                           callback: Callable,
                           cancel_flag: list) -> OptimizationResult:
    """Standard Genetic Algorithm with elitism."""
    t0 = time.time()
    rng = np.random.default_rng(42)
    dvs = [dv for dv in config.design_variables if dv.enabled]
    pop_size = config.population_size
    n_elite = max(2, pop_size // 10)
    total_evals = 0
    estimated_evals = max(1, config.mc_sims_per_candidate * pop_size * (config.max_generations + 1))
    screener = _make_screener(config, dvs, rng)
    executor = _make_executor(config)
    try:
        # Initialise population — build all candidates, evaluate as one batch
        init_dicts = [_valid_individual(config.design_variables, dvs, rng, base=base_config)
                      for _ in range(pop_size)]
        init_seeds = [i * 13 for i in range(pop_size)]

        done = [0]
        def _init_progress(i, cd):
            done[0] += 1
            total = config.mc_sims_per_candidate * done[0]
            if callback:
                callback(0, config.max_generations, cd.fitness, {
                    "phase": "initializing",
                    "message": "Evaluating initial population",
                    "generation": 0,
                    "evaluated_candidates": done[0],
                    "total_candidates": pop_size,
                    "evaluations": total,
                    "estimated_evaluations": estimated_evals,
                    "best_fitness": cd.fitness,
                })

        population = _parallel_eval(executor, base_config, init_dicts, config,
                                    init_seeds, on_result=_init_progress,
                                    cancel_flag=cancel_flag)
        population = [c for c in population if c is not None]
        total_evals += config.mc_sims_per_candidate * len(population)
        screener.observe(population)

        if cancel_flag and cancel_flag[0]:
            return OptimizationResult(
                all_designs=population,
                total_evaluations=total_evals,
                elapsed_time=time.time() - t0,
                algorithm_used="ga",
                surrogate_accuracy=screener.final_score(),
            )

        gen_history = []

        for gen in range(config.max_generations):
            if cancel_flag and cancel_flag[0]:
                break

            # Sort by fitness descending
            population.sort(key=lambda c: c.fitness, reverse=True)

            # Record generation data
            fitnesses = _finite_fitnesses(population)
            apogees = [c.mc_stats.get("mean_apogee", 0) if c.mc_stats else 0 for c in population]
            feasible_pct = sum(1 for c in population if c.feasible) / len(population) * 100

            gen_data = {
                "phase": "generation",
                "generation": gen,
                "best_fitness": fitnesses[0] if fitnesses else 0.0,
                "mean_fitness": float(np.mean(fitnesses)) if fitnesses else 0.0,
                "worst_fitness": fitnesses[-1] if fitnesses else 0.0,
                "feasible_pct": feasible_pct,
                "best_apogee": apogees[0],
                "evaluations": total_evals,
                "estimated_evaluations": estimated_evals,
            }
            gen_history.append(gen_data)

            if callback:
                callback(gen, config.max_generations, fitnesses[0], gen_data)

            # Elitism — carry top n_elite unchanged
            new_pop = list(population[:n_elite])

            # Build all offspring var-dicts first (serial, cheap), then eval batch.
            # With surrogate screening on we build a larger pool and keep only
            # the most promising n_children of it — breeding is cheap, the
            # simulation that follows is not.
            n_children = pop_size - n_elite
            n_pool = screener.pool_size(n_children)
            child_dicts = []
            while len(child_dicts) < n_pool:
                p1 = _tournament_select(population, 3, rng)
                p2 = _tournament_select(population, 3, rng)

                if rng.random() < config.crossover_rate:
                    c1_vars, c2_vars = _sbx_crossover(
                        p1.variables, p2.variables, config.design_variables, 20.0, rng)
                else:
                    c1_vars, c2_vars = dict(p1.variables), dict(p2.variables)

                c1_vars = _polynomial_mutation(c1_vars, config.design_variables,
                                                20.0, config.mutation_rate, rng)
                c2_vars = _polynomial_mutation(c2_vars, config.design_variables,
                                                20.0, config.mutation_rate, rng)
                # Crossover/mutation can leave geometry unphysical — repair
                # before it reaches the simulator (only the initial population
                # was ever validated).
                c1_vars = _repair_individual(c1_vars, config.design_variables, base_config)
                c2_vars = _repair_individual(c2_vars, config.design_variables, base_config)
                child_dicts.append(c1_vars)
                if len(child_dicts) < n_pool:
                    child_dicts.append(c2_vars)

            child_dicts = screener.screen(child_dicts, n_children, generation=gen)
            child_seeds = [total_evals + k for k in range(len(child_dicts))]

            done = [len(new_pop)]
            def _off_progress(i, cd):
                done[0] += 1
                if callback:
                    callback(gen + 1, config.max_generations, cd.fitness, {
                        "phase": "evaluating",
                        "message": "Evaluating offspring",
                        "generation": gen + 1,
                        "evaluated_candidates": done[0],
                        "total_candidates": pop_size,
                        "evaluations": total_evals + config.mc_sims_per_candidate * (done[0] - n_elite),
                        "estimated_evaluations": estimated_evals,
                        "best_fitness": cd.fitness,
                    })

            children = _parallel_eval(executor, base_config, child_dicts, config,
                                      child_seeds, on_result=_off_progress,
                                      cancel_flag=cancel_flag)
            children = [c for c in children if c is not None]
            total_evals += config.mc_sims_per_candidate * len(children)
            screener.observe(children)
            new_pop.extend(children)
            population = new_pop

        population.sort(key=lambda c: c.fitness, reverse=True)
        best = population[0] if population else CandidateDesign()

        return OptimizationResult(
            best_design=best,
            pareto_front=_pareto_from_population(population, config.objectives, best),
            all_designs=population,
            generation_history=gen_history,
            total_evaluations=total_evals,
            elapsed_time=time.time() - t0,
            algorithm_used="ga",
            surrogate_accuracy=screener.final_score(),
        )
    finally:
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)


def _run_nsga2(config: OptimizationConfig,
               base_config: BatchSimConfig,
               callback: Callable,
               cancel_flag: list) -> OptimizationResult:
    """NSGA-II multi-objective optimisation."""
    t0 = time.time()
    rng = np.random.default_rng(42)
    dvs = [dv for dv in config.design_variables if dv.enabled]
    pop_size = config.population_size
    total_evals = 0

    # Mission mode is single-objective (hit target). Drive NSGA-II domination
    # by the scalar mission fitness, otherwise it sorts on raw apogee and
    # blows past the target. Multi-objective runs keep the user's objectives.
    if config.mission_mode and config.target_apogee > 0:
        sort_objs = [ObjectiveFunction("fitness", "Mission Fitness", "maximize", 1.0, True)]
    else:
        sort_objs = config.objectives

    # Screening ranks on the scalar fitness, not the Pareto ordering — it only
    # decides which offspring are worth simulating, and the non-dominated sort
    # that follows still runs on real objective values. The unscreened
    # exploration slice is what keeps the front from collapsing.
    screener = _make_screener(config, dvs, rng)
    executor = _make_executor(config)
    try:
        # Initialise — batch-evaluate the random population
        init_dicts = [_valid_individual(config.design_variables, dvs, rng, base=base_config)
                      for _ in range(pop_size)]
        init_seeds = [i * 17 for i in range(pop_size)]
        population = _parallel_eval(executor, base_config, init_dicts, config,
                                    init_seeds, cancel_flag=cancel_flag)
        population = [c for c in population if c is not None]
        total_evals += config.mc_sims_per_candidate * len(population)
        screener.observe(population)

        gen_history = []

        # Binary tournament draws two distinct members; cancel-filtering above
        # can leave fewer, and rng.choice would raise.
        if len(population) < 2:
            logger.warning(
                f"NSGA-II needs a population of 2+, have {len(population)} — "
                f"stopping after init")
            config = copy.copy(config)
            config.max_generations = 0

        for gen in range(config.max_generations):
            if cancel_flag and cancel_flag[0]:
                break

            # Non-dominated sort + crowding
            fronts = _fast_non_dominated_sort(population, sort_objs)
            for front in fronts:
                _crowding_distance(population, front, sort_objs)

            # Record
            fitnesses = _finite_fitnesses(population)
            apogees = [c.mc_stats.get("mean_apogee", 0) if c.mc_stats else 0 for c in population]
            feasible_pct = sum(1 for c in population if c.feasible) / max(len(population), 1) * 100

            gen_data = {
                "phase": "generation",
                "generation": gen,
                "best_fitness": max(fitnesses) if fitnesses else 0,
                "mean_fitness": float(np.mean(fitnesses)) if fitnesses else 0,
                "worst_fitness": min(fitnesses) if fitnesses else 0,
                "feasible_pct": feasible_pct,
                "best_apogee": max(apogees) if apogees else 0,
            }
            gen_history.append(gen_data)

            if callback:
                callback(gen, config.max_generations, gen_data["best_fitness"], gen_data)

            # Build all offspring var-dicts (serial), then evaluate as one batch
            n_pool = screener.pool_size(pop_size)
            child_dicts = []
            while len(child_dicts) < n_pool:
                # Binary tournament (rank, then crowding)
                i1, i2 = rng.choice(len(population), 2, replace=False)
                p1 = population[i1] if (population[i1].rank < population[i2].rank or
                    (population[i1].rank == population[i2].rank and
                     population[i1].crowding_distance > population[i2].crowding_distance)) \
                    else population[i2]

                i3, i4 = rng.choice(len(population), 2, replace=False)
                p2 = population[i3] if (population[i3].rank < population[i4].rank or
                    (population[i3].rank == population[i4].rank and
                     population[i3].crowding_distance > population[i4].crowding_distance)) \
                    else population[i4]

                if rng.random() < config.crossover_rate:
                    c1_vars, c2_vars = _sbx_crossover(
                        p1.variables, p2.variables, config.design_variables, 20.0, rng)
                else:
                    c1_vars, c2_vars = dict(p1.variables), dict(p2.variables)

                c1_vars = _polynomial_mutation(c1_vars, config.design_variables,
                                                20.0, config.mutation_rate, rng)
                c1_vars = _repair_individual(c1_vars, config.design_variables, base_config)
                child_dicts.append(c1_vars)

            child_dicts = screener.screen(child_dicts, pop_size, generation=gen)
            child_seeds = [total_evals + k for k in range(len(child_dicts))]
            offspring = _parallel_eval(executor, base_config, child_dicts, config,
                                       child_seeds, cancel_flag=cancel_flag)
            offspring = [c for c in offspring if c is not None]
            total_evals += config.mc_sims_per_candidate * len(offspring)
            screener.observe(offspring)

            # Combine parent + offspring, select best pop_size
            combined = population + offspring
            fronts = _fast_non_dominated_sort(combined, sort_objs)
            new_pop = []
            for front in fronts:
                _crowding_distance(combined, front, sort_objs)
                if len(new_pop) + len(front) <= pop_size:
                    new_pop.extend([combined[i] for i in front])
                else:
                    remaining = pop_size - len(new_pop)
                    sorted_front = sorted(front,
                        key=lambda i: combined[i].crowding_distance, reverse=True)
                    new_pop.extend([combined[i] for i in sorted_front[:remaining]])
                    break

            population = new_pop

        # Extract Pareto front (rank 0)
        fronts = _fast_non_dominated_sort(population, sort_objs)
        pareto = [population[i] for i in fronts[0]] if fronts else []

        # "Best" single design must come from the non-dominated front — a high
        # scalar fitness elsewhere can still be Pareto-dominated.
        if pareto:
            best = max(pareto, key=lambda c: c.fitness)
        elif population:
            best = max(population, key=lambda c: c.fitness)
        else:
            best = CandidateDesign()

        return OptimizationResult(
            best_design=best,
            pareto_front=pareto,
            all_designs=population,
            generation_history=gen_history,
            total_evaluations=total_evals,
            elapsed_time=time.time() - t0,
            algorithm_used="nsga2",
            surrogate_accuracy=screener.final_score(),
        )
    finally:
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)


def _run_differential_evolution(config: OptimizationConfig,
                                base_config: BatchSimConfig,
                                callback: Callable,
                                cancel_flag: list) -> OptimizationResult:
    """Differential Evolution  (DE/rand/1/bin)."""
    t0 = time.time()
    rng = np.random.default_rng(42)
    dvs = [dv for dv in config.design_variables if dv.enabled]
    pop_size = config.population_size
    F = 0.8
    CR = 0.9
    total_evals = 0

    # Continuous variable indices
    var_keys = [dv.name for dv in dvs]
    lo = np.array([dv.min_val for dv in dvs])
    hi = np.array([dv.max_val for dv in dvs])

    executor = _make_executor(config)
    try:
        # Initialise — batch-evaluate random population
        pop_vec = rng.uniform(lo, hi, size=(pop_size, len(dvs)))
        for i in range(pop_size):
            _valid_vector(pop_vec, i, dvs, config.design_variables, lo, hi, rng,
                          base=base_config)
        init_dicts = [_vec_to_dict(pop_vec[i], dvs, config.design_variables)
                      for i in range(pop_size)]
        init_seeds = [i * 19 for i in range(pop_size)]
        population = _parallel_eval(executor, base_config, init_dicts, config,
                                    init_seeds, cancel_flag=cancel_flag)
        # Cancelling mid-initialisation leaves None slots. Drop those rows from
        # BOTH the candidate list and the vector population so the two stay
        # index-aligned — the selection loop below indexes them in lockstep.
        keep = [i for i, c in enumerate(population) if c is not None]
        population = [population[i] for i in keep]
        pop_vec = pop_vec[keep]
        pop_size = len(population)
        total_evals += config.mc_sims_per_candidate * pop_size

        # DE/rand/1 draws three DISTINCT donors other than the target, so it
        # needs at least four members. Cancel-filtering above can shrink the
        # population below that, and rng.choice would then raise.
        if pop_size < 4:
            logger.warning(
                f"DE needs a population of 4+, have {pop_size} — stopping after init")
            best = (max(population, key=lambda c: c.fitness)
                    if population else CandidateDesign())
            return OptimizationResult(
                best_design=best if population else None,
                pareto_front=_pareto_from_population(
                    population, config.objectives, best) if population else [],
                all_designs=population,
                total_evaluations=total_evals,
                elapsed_time=time.time() - t0,
                algorithm_used="de",
            )

        gen_history = []

        for gen in range(config.max_generations):
            if cancel_flag and cancel_flag[0]:
                break

            # Build every trial vector against a FROZEN snapshot of this gen
            # (synchronous DE), then evaluate the whole batch in parallel.
            trials = []
            for i in range(pop_size):
                # Mutation: DE/rand/1
                idxs = rng.choice([j for j in range(pop_size) if j != i], 3, replace=False)
                a, b, c = pop_vec[idxs[0]], pop_vec[idxs[1]], pop_vec[idxs[2]]
                mutant = a + F * (b - c)

                # Bounce-back bounds
                for d in range(len(dvs)):
                    if mutant[d] < lo[d]:
                        mutant[d] = lo[d] + rng.random() * (pop_vec[i, d] - lo[d])
                    if mutant[d] > hi[d]:
                        mutant[d] = hi[d] - rng.random() * (hi[d] - pop_vec[i, d])

                # Crossover: binomial
                trial = pop_vec[i].copy()
                j_rand = rng.integers(len(dvs))
                for d in range(len(dvs)):
                    if rng.random() < CR or d == j_rand:
                        trial[d] = mutant[d]

                # Round integers
                for d, dv in enumerate(dvs):
                    if dv.var_type == "integer":
                        trial[d] = round(trial[d])
                trials.append(trial)

            trial_dicts = [_repair_individual(_vec_to_dict(t, dvs, config.design_variables),
                                              config.design_variables, base_config)
                           for t in trials]
            # Keep the numeric trial vectors in step with the repaired dicts —
            # selection below stores trials[i] into pop_vec on acceptance.
            trials = [_dict_to_vec(td, dvs) for td in trial_dicts]
            trial_seeds = [total_evals + k for k in range(pop_size)]
            trial_cds = _parallel_eval(executor, base_config, trial_dicts, config,
                                       trial_seeds, cancel_flag=cancel_flag)
            total_evals += config.mc_sims_per_candidate * sum(1 for c in trial_cds if c is not None)

            # Selection — greedy, against the parent at the same index
            for i in range(pop_size):
                trial_cd = trial_cds[i]
                if trial_cd is not None and trial_cd.fitness >= population[i].fitness:
                    population[i] = trial_cd
                    pop_vec[i] = trials[i]

            fitnesses = _finite_fitnesses(population)
            apogees = [c.mc_stats.get("mean_apogee", 0) if c.mc_stats else 0 for c in population]
            feasible_pct = sum(1 for c in population if c.feasible) / len(population) * 100
            gen_data = {
                "phase": "generation",
                "generation": gen,
                "best_fitness": max(fitnesses) if fitnesses else 0.0,
                "mean_fitness": float(np.mean(fitnesses)) if fitnesses else 0.0,
                "worst_fitness": min(fitnesses) if fitnesses else 0.0,
                "feasible_pct": feasible_pct,
                "best_apogee": max(apogees),
            }
            gen_history.append(gen_data)
            if callback:
                callback(gen, config.max_generations, gen_data["best_fitness"], gen_data)

        best = max(population, key=lambda c: c.fitness) if population else CandidateDesign()
        return OptimizationResult(
            best_design=best,
            pareto_front=_pareto_from_population(population, config.objectives, best),
            all_designs=population,
            generation_history=gen_history,
            total_evaluations=total_evals,
            elapsed_time=time.time() - t0,
            algorithm_used="de",
        )
    finally:
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)


def _run_particle_swarm(config: OptimizationConfig,
                        base_config: BatchSimConfig,
                        callback: Callable,
                        cancel_flag: list) -> OptimizationResult:
    """Particle Swarm Optimisation with inertia weight decay."""
    t0 = time.time()
    rng = np.random.default_rng(42)
    dvs = [dv for dv in config.design_variables if dv.enabled]
    pop_size = config.population_size
    c1 = c2 = 2.0
    w_start, w_end = 0.9, 0.4
    total_evals = 0

    n = len(dvs)
    lo = np.array([dv.min_val for dv in dvs])
    hi = np.array([dv.max_val for dv in dvs])
    v_max = 0.2 * (hi - lo)

    executor = _make_executor(config)
    try:
        # Initialise
        positions = rng.uniform(lo, hi, size=(pop_size, n))
        for i in range(pop_size):
            _valid_vector(positions, i, dvs, config.design_variables, lo, hi, rng,
                          base=base_config)
        velocities = rng.uniform(-v_max, v_max, size=(pop_size, n))
        p_best_pos = positions.copy()
        p_best_fit = np.full(pop_size, -np.inf)
        g_best_pos = positions[0].copy()
        g_best_fit = -np.inf
        # PSO is not elitist: particles fly away from the global best, so the
        # final swarm need not contain it. Keep the winning design itself, or
        # the reported best_fitness and the returned best_design disagree.
        g_best_design = None

        init_dicts = [_vec_to_dict(positions[i], dvs, config.design_variables)
                      for i in range(pop_size)]
        init_seeds = [i * 23 for i in range(pop_size)]
        population = _parallel_eval(executor, base_config, init_dicts, config,
                                    init_seeds, cancel_flag=cancel_flag)
        # Cancelling mid-initialisation leaves None slots. Drop those particles
        # from EVERY per-particle array at once, so positions / velocities /
        # personal bests stay index-aligned with the candidate list below.
        keep = [i for i, c in enumerate(population) if c is not None]
        population = [population[i] for i in keep]
        positions = positions[keep]
        velocities = velocities[keep]
        p_best_pos = p_best_pos[keep]
        p_best_fit = p_best_fit[keep]
        pop_size = len(population)
        total_evals += config.mc_sims_per_candidate * pop_size

        if pop_size == 0:
            return OptimizationResult(
                all_designs=[],
                total_evaluations=total_evals,
                elapsed_time=time.time() - t0,
                algorithm_used="pso",
            )

        for i in range(pop_size):
            cd = population[i]
            if cd.fitness > p_best_fit[i]:
                p_best_fit[i] = cd.fitness
                p_best_pos[i] = positions[i].copy()
            if cd.fitness > g_best_fit:
                g_best_fit = cd.fitness
                g_best_pos = positions[i].copy()
                g_best_design = cd

        gen_history = []

        for gen in range(config.max_generations):
            if cancel_flag and cancel_flag[0]:
                break

            w = w_start - (w_start - w_end) * gen / max(config.max_generations - 1, 1)

            # Synchronous PSO: advance every particle against g_best held fixed
            # for the whole generation, then evaluate the swarm in parallel.
            for i in range(pop_size):
                r1 = rng.random(n)
                r2 = rng.random(n)

                velocities[i] = (w * velocities[i]
                                 + c1 * r1 * (p_best_pos[i] - positions[i])
                                 + c2 * r2 * (g_best_pos - positions[i]))

                # Clamp velocities
                velocities[i] = np.clip(velocities[i], -v_max, v_max)

                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], lo, hi)

                # Round integers
                for d, dv in enumerate(dvs):
                    if dv.var_type == "integer":
                        positions[i, d] = round(positions[i, d])

            swarm_dicts = [_repair_individual(
                               _vec_to_dict(positions[i], dvs, config.design_variables),
                               config.design_variables, base_config)
                           for i in range(pop_size)]
            # Fold the repair back into the swarm, so personal/global bests
            # record the design that was actually flown.
            for i in range(pop_size):
                positions[i] = _dict_to_vec(swarm_dicts[i], dvs)
            swarm_seeds = [total_evals + k for k in range(pop_size)]
            new_cds = _parallel_eval(executor, base_config, swarm_dicts, config,
                                     swarm_seeds, cancel_flag=cancel_flag)
            total_evals += config.mc_sims_per_candidate * sum(1 for c in new_cds if c is not None)

            # Update personal / global bests after the whole swarm is evaluated
            for i in range(pop_size):
                cd = new_cds[i]
                if cd is None:
                    continue
                population[i] = cd
                if cd.fitness > p_best_fit[i]:
                    p_best_fit[i] = cd.fitness
                    p_best_pos[i] = positions[i].copy()
                if cd.fitness > g_best_fit:
                    g_best_fit = cd.fitness
                    g_best_pos = positions[i].copy()
                    g_best_design = cd

            fitnesses = _finite_fitnesses(population)
            apogees = [c.mc_stats.get("mean_apogee", 0) if c.mc_stats else 0 for c in population]
            feasible_pct = sum(1 for c in population if c.feasible) / len(population) * 100
            gen_data = {
                "phase": "generation",
                "generation": gen,
                "best_fitness": g_best_fit,
                "mean_fitness": float(np.mean(fitnesses)) if fitnesses else 0.0,
                "worst_fitness": min(fitnesses) if fitnesses else 0.0,
                "feasible_pct": feasible_pct,
                "best_apogee": max(apogees),
            }
            gen_history.append(gen_data)
            if callback:
                callback(gen, config.max_generations, g_best_fit, gen_data)

        if g_best_design is not None:
            best = g_best_design
        else:
            best = max(population, key=lambda c: c.fitness) if population else CandidateDesign()
        return OptimizationResult(
            best_design=best,
            pareto_front=_pareto_from_population(population, config.objectives, best),
            all_designs=population,
            generation_history=gen_history,
            total_evaluations=total_evals,
            elapsed_time=time.time() - t0,
            algorithm_used="pso",
        )
    finally:
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)


def _dict_to_vec(d: dict, enabled_dvs: list) -> np.ndarray:
    """Inverse of :func:`_vec_to_dict` over the enabled variables.

    Lets DE/PSO write a repaired candidate back into their numeric population,
    so the stored vector always matches the design that was actually simulated.
    """
    return np.array([float(d[dv.name]) for dv in enabled_dvs], dtype=float)


def _vec_to_dict(vec: np.ndarray, enabled_dvs: list, all_dvs: list) -> dict:
    """Convert numpy vector back to variable dict, keeping disabled at current."""
    d = {}
    idx = 0
    for dv in all_dvs:
        if dv.enabled and idx < len(vec):
            val = vec[idx]
            if dv.var_type == "integer":
                val = int(round(val))
            d[dv.name] = float(val)
            idx += 1
        else:
            d[dv.name] = dv.current_val
    return d


# ══════════════════════════════════════════════════════════════════════════════
#  QT  ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

class _OptimizationWorkerThread(QThread):
    """Background worker that runs an optimisation algorithm."""

    progress = pyqtSignal(int, int, float)          # gen, total, best_fit
    status_update = pyqtSignal(object)              # lightweight live status dict
    generation_complete = pyqtSignal(object)         # gen_data dict
    all_done = pyqtSignal(object)                    # OptimizationResult
    failed = pyqtSignal(str)

    def __init__(self, config: OptimizationConfig, base_config: BatchSimConfig):
        super().__init__()
        self._config = config
        self._base = base_config
        self._cancel = [False]

    def cancel(self):
        self._cancel[0] = True

    def run(self):
        try:
            algo_map = {
                "ga": _run_genetic_algorithm,
                "nsga2": _run_nsga2,
                "de": _run_differential_evolution,
                "pso": _run_particle_swarm,
            }
            fn = algo_map.get(self._config.algorithm, _run_genetic_algorithm)

            def cb(gen, total, best_fit, gen_data):
                self.status_update.emit(gen_data)
                if gen_data.get("phase") == "generation":
                    self.progress.emit(gen, total, best_fit)
                    self.generation_complete.emit(gen_data)

            result = fn(self._config, self._base, cb, self._cancel)

            if self._cancel[0]:
                return

            result = self._validate_best(result)

            if self._cancel[0]:
                return

            self.all_done.emit(result)

        except Exception as e:
            logger.error(f"Optimisation worker failed: {e}", exc_info=True)
            self.failed.emit(str(e))

    def _validate_best(self, result: OptimizationResult) -> OptimizationResult:
        """Re-score the winning design with the configured validation sample count.

        The 'Validation Sims' control promised "MC simulations for final
        validation of top designs", but ``validation_mc_sims`` was never read by
        anything — the reported winner carried whatever noise a
        ``mc_sims_per_candidate``-sized sample gave it. Re-evaluating the single
        best design at the larger sample size costs one extra batch and makes
        the headline numbers trustworthy.
        """
        cfg = self._config
        n_val = getattr(cfg, "validation_mc_sims", 0) or 0
        best = result.best_design

        if best is None or not best.variables or n_val <= cfg.mc_sims_per_candidate:
            return result

        self.status_update.emit({
            "phase": "validating",
            "message": f"Validating best design ({n_val} MC sims)",
            "generation": cfg.max_generations,
            "evaluations": result.total_evaluations,
            "best_fitness": best.fitness,
        })

        try:
            validated = evaluate_candidate(
                self._base, best.variables, cfg.design_variables, cfg.objectives,
                cfg.constraints, cfg.correlations, n_val, seed=99991,
                target_apogee=cfg.target_apogee, mission_mode=cfg.mission_mode,
                robust_mode=cfg.robust_mode)
        except Exception as e:
            logger.warning(f"Validation pass failed, keeping search estimate: {e}")
            return result

        validated.mc_stats["validated_sims"] = n_val
        result.best_design = validated
        result.total_evaluations += n_val
        logger.info(
            f"Validation pass: fitness {best.fitness:.2f} -> {validated.fitness:.2f} "
            f"over {n_val} MC sims")
        return result


class OptimizationEngine(QObject):
    """Top-level Qt orchestrator for running optimisation."""

    progress = pyqtSignal(int, int, float)
    status_update = pyqtSignal(object)
    generation_complete = pyqtSignal(object)
    optimization_finished = pyqtSignal(object)
    optimization_failed = pyqtSignal(str)
    optimization_cancelled = pyqtSignal()

    def __init__(self, engine, parent=None):
        super().__init__(parent)
        self.engine = engine
        self._worker: Optional[_OptimizationWorkerThread] = None

    @property
    def is_running(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    def start(self, config: OptimizationConfig):
        """Build base config from rocket state and launch worker."""
        if self.is_running:
            logger.warning("Optimisation already running")
            return

        try:
            base = BatchSimConfig.from_rocket_state(self.engine.state)
        except Exception as exc:
            # Mirror the Monte Carlo engine: report a bad rocket state through
            # the failure signal instead of raising into the caller's click.
            logger.error(f"Failed to build base config: {exc}")
            self.optimization_failed.emit(f"Failed to build base config: {exc}")
            return
        self._worker = _OptimizationWorkerThread(config, base)
        self._worker.progress.connect(self.progress)
        self._worker.status_update.connect(self.status_update)
        self._worker.generation_complete.connect(self.generation_complete)
        self._worker.all_done.connect(self._on_done)
        self._worker.failed.connect(self._on_fail)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()
        logger.info(f"Optimisation started: {config.algorithm}")

    def cancel(self):
        if self._worker:
            self._worker.cancel()
            logger.info("Optimisation cancel requested")

    def _on_done(self, result):
        self.optimization_finished.emit(result)

    def _on_fail(self, msg):
        self.optimization_failed.emit(msg)

    def _on_finished(self):
        """Worker QThread has exited — safe to drop the reference.

        ``is_running`` guards ``start()`` against rebinding a live thread, so
        the reference is only cleared here, never on cancel.
        """
        worker = self._worker
        self._worker = None
        cancelled = worker is not None and worker._cancel[0]
        if worker is not None:
            worker.deleteLater()
        if cancelled:
            self.optimization_cancelled.emit()
