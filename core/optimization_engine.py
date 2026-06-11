"""
K2 Aerospace — Optimization Engine
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

from core.batch_simulation import BatchSimConfig, BatchSimResult, run_batch_simulation

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
    surrogate_initial_samples: int = 200
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
        ObjectiveFunction("max_payload_fraction", "Max Payload Fraction", "maximize", 1.0, False),
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

def validate_candidate(variables: dict, design_vars: list) -> tuple:
    """Validate that a candidate design is physically realisable.

    Returns (valid: bool, warnings: list[str]).
    """
    warnings = []
    v = variables

    d = v.get("diameter", 0.08)
    L = v.get("length", 1.0)
    fs = v.get("fin_span", 0.05)
    frc = v.get("fin_root_chord", 0.1)
    ftc = v.get("fin_tip_chord", 0.05)
    dm = v.get("dry_mass", 1.0)
    nl = v.get("nose_length", 0.2)

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

    # Derived fields
    if cfg.motor_burn_time > 0 and cfg.motor_total_impulse > 0:
        cfg.motor_avg_thrust = cfg.motor_total_impulse / cfg.motor_burn_time
    if cfg.motor_avg_thrust > 0:
        cfg.motor_max_thrust = max(cfg.motor_max_thrust, cfg.motor_avg_thrust * 1.3)

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

    apogees, machs, accels, stabs, landings = [], [], [], [], []
    rail_exits, successes, velocities = [], [], []

    for i, mc_cfg in enumerate(mc_configs):
        try:
            res = run_batch_simulation(mc_cfg, seed=seed + i * 7)
            apogees.append(res.apogee)
            machs.append(res.max_mach)
            accels.append(res.max_acceleration)
            stabs.append(res.min_stability_margin)
            landings.append(res.landing_distance)
            rail_exits.append(res.rail_exit_velocity)
            successes.append(1.0 if res.success else 0.0)
            velocities.append(res.max_velocity)
        except Exception:
            apogees.append(0.0)
            machs.append(0.0)
            accels.append(0.0)
            stabs.append(0.0)
            landings.append(9999.0)
            rail_exits.append(0.0)
            successes.append(0.0)
            velocities.append(0.0)

    arr_apogee = np.array(apogees)
    arr_mach = np.array(machs)
    arr_stab = np.array(stabs)
    arr_landing = np.array(landings)
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
        "p_target": p_target,
        "p5_apogee": float(np.percentile(arr_apogee, 5)) if len(arr_apogee) > 1 else mean_apogee,
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

    # Payload fraction
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
    return variables.get(key, 0.0)


def _standard_fitness(obj_vals, objectives, constraints, cons_eval) -> float:
    """Weighted-sum fitness with quadratic constraint penalties."""
    fitness = 0.0
    for o in objectives:
        if not o.enabled:
            continue
        val = obj_vals.get(o.name, 0.0)
        if o.direction == "minimize":
            val = -val
        fitness += o.weight * val

    # Penalties
    for c in constraints:
        if not c.enabled:
            continue
        info = cons_eval.get(c.name)
        if info and not info["satisfied"]:
            violation = abs(info["value"] - info["limit"])
            fitness -= c.penalty_weight * violation ** 2

    return fitness


def _robust_fitness(obj_vals, mc_stats, objectives, constraints, cons_eval) -> float:
    """Robust fitness using MC statistics."""
    fitness = 0.0
    for o in objectives:
        if not o.enabled:
            continue
        if o.robust_mode == "std":
            val = -mc_stats.get("std_apogee", 0.0)
        elif o.robust_mode == "worst":
            if o.direction == "maximize":
                val = mc_stats.get("min_apogee", 0.0)
            else:
                val = -mc_stats.get("max_apogee", 0.0)
        elif o.robust_mode == "reliability":
            val = mc_stats.get("success_rate", 0.0)
        elif o.robust_mode == "p5":
            val = mc_stats.get("p5_apogee", 0.0)
            if o.direction == "minimize":
                val = -val
        else:  # "mean"
            val = obj_vals.get(o.name, 0.0)
            if o.direction == "minimize":
                val = -val
        fitness += o.weight * val

    for c in constraints:
        if not c.enabled:
            continue
        info = cons_eval.get(c.name)
        if info and not info["satisfied"]:
            violation = abs(info["value"] - info["limit"])
            fitness -= c.penalty_weight * violation ** 2

    return fitness


def _mission_fitness(obj_vals, mc_stats, target, objectives, constraints, cons_eval) -> float:
    """Mission-driven fitness: maximise P(apogee within ±10% of target)."""
    p_target = mc_stats.get("p_target", 0.0)
    success = mc_stats.get("success_rate", 0.0)
    mean_apogee = mc_stats.get("mean_apogee", 0.0)

    # Primary: P(target) × success_rate
    fitness = 1000.0 * p_target * success

    # Penalise deviation from target
    if target > 0 and mean_apogee > 0:
        relative_error = abs(mean_apogee - target) / target
        fitness -= 200.0 * relative_error

    # Secondary objectives
    for o in objectives:
        if not o.enabled or o.name in ("max_apogee", "max_prob_target", "max_mission_success"):
            continue
        val = obj_vals.get(o.name, 0.0)
        if o.direction == "minimize":
            val = -val
        fitness += o.weight * 0.1 * val

    # Constraint penalties
    for c in constraints:
        if not c.enabled:
            continue
        info = cons_eval.get(c.name)
        if info and not info["satisfied"]:
            violation = abs(info["value"] - info["limit"])
            fitness -= c.penalty_weight * violation ** 2

    return fitness


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
            results[i] = _eval_task(t)
            if on_result:
                on_result(i, results[i])
        return results

    futures = {executor.submit(_eval_task, t): i for i, t in enumerate(tasks)}
    for fut in as_completed(futures):
        i = futures[fut]
        results[i] = fut.result()
        if on_result:
            on_result(i, results[i])
    return results


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


def _valid_individual(design_vars: list, enabled_dvs: list,
                      rng: np.random.Generator, tries: int = 10) -> dict:
    """Random individual that passes physical validation, resampling up to
    *tries* times. Returns the last attempt if none validate (bounds still
    clamped, sim is robust to it)."""
    ind = _random_individual(design_vars, rng)
    for _ in range(tries):
        valid, _ = validate_candidate(ind, enabled_dvs)
        if valid:
            break
        ind = _random_individual(design_vars, rng)
    return ind


def _valid_vector(pop_vec: np.ndarray, idx: int, enabled_dvs: list,
                  all_dvs: list, lo: np.ndarray, hi: np.ndarray,
                  rng: np.random.Generator, tries: int = 10) -> None:
    """Resample row *idx* of *pop_vec* in place until it validates (or tries
    exhausted). Used by DE/PSO whose populations are numpy vectors."""
    for _ in range(tries):
        d = _vec_to_dict(pop_vec[idx], enabled_dvs, all_dvs)
        valid, _ = validate_candidate(d, enabled_dvs)
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

def _dominates(a: CandidateDesign, b: CandidateDesign, objectives: list) -> bool:
    """Return True if design *a* dominates design *b*."""
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
    executor = _make_executor(config)
    try:
        # Initialise population — build all candidates, evaluate as one batch
        init_dicts = [_valid_individual(config.design_variables, dvs, rng)
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

        if cancel_flag and cancel_flag[0]:
            return OptimizationResult(
                all_designs=population,
                total_evaluations=total_evals,
                elapsed_time=time.time() - t0,
                algorithm_used="ga",
            )

        gen_history = []

        for gen in range(config.max_generations):
            if cancel_flag and cancel_flag[0]:
                break

            # Sort by fitness descending
            population.sort(key=lambda c: c.fitness, reverse=True)

            # Record generation data
            fitnesses = [c.fitness for c in population]
            apogees = [c.mc_stats.get("mean_apogee", 0) if c.mc_stats else 0 for c in population]
            feasible_pct = sum(1 for c in population if c.feasible) / len(population) * 100

            gen_data = {
                "phase": "generation",
                "generation": gen,
                "best_fitness": fitnesses[0],
                "mean_fitness": float(np.mean(fitnesses)),
                "worst_fitness": fitnesses[-1],
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

            # Build all offspring var-dicts first (serial, cheap), then eval batch
            child_dicts = []
            while len(child_dicts) < pop_size - n_elite:
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
                child_dicts.append(c1_vars)
                if len(child_dicts) < pop_size - n_elite:
                    child_dicts.append(c2_vars)

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
    executor = _make_executor(config)
    try:
        # Initialise — batch-evaluate the random population
        init_dicts = [_valid_individual(config.design_variables, dvs, rng)
                      for _ in range(pop_size)]
        init_seeds = [i * 17 for i in range(pop_size)]
        population = _parallel_eval(executor, base_config, init_dicts, config,
                                    init_seeds, cancel_flag=cancel_flag)
        population = [c for c in population if c is not None]
        total_evals += config.mc_sims_per_candidate * len(population)

        gen_history = []

        for gen in range(config.max_generations):
            if cancel_flag and cancel_flag[0]:
                break

            # Non-dominated sort + crowding
            fronts = _fast_non_dominated_sort(population, config.objectives)
            for front in fronts:
                _crowding_distance(population, front, config.objectives)

            # Record
            fitnesses = [c.fitness for c in population]
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
            child_dicts = []
            while len(child_dicts) < pop_size:
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
                child_dicts.append(c1_vars)

            child_seeds = [total_evals + k for k in range(len(child_dicts))]
            offspring = _parallel_eval(executor, base_config, child_dicts, config,
                                       child_seeds, cancel_flag=cancel_flag)
            offspring = [c for c in offspring if c is not None]
            total_evals += config.mc_sims_per_candidate * len(offspring)

            # Combine parent + offspring, select best pop_size
            combined = population + offspring
            fronts = _fast_non_dominated_sort(combined, config.objectives)
            new_pop = []
            for front in fronts:
                _crowding_distance(combined, front, config.objectives)
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
        fronts = _fast_non_dominated_sort(population, config.objectives)
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
            _valid_vector(pop_vec, i, dvs, config.design_variables, lo, hi, rng)
        init_dicts = [_vec_to_dict(pop_vec[i], dvs, config.design_variables)
                      for i in range(pop_size)]
        init_seeds = [i * 19 for i in range(pop_size)]
        population = _parallel_eval(executor, base_config, init_dicts, config,
                                    init_seeds, cancel_flag=cancel_flag)
        population = [c for c in population if c is not None]
        total_evals += config.mc_sims_per_candidate * len(population)

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

            trial_dicts = [_vec_to_dict(t, dvs, config.design_variables) for t in trials]
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

            fitnesses = [c.fitness for c in population]
            apogees = [c.mc_stats.get("mean_apogee", 0) if c.mc_stats else 0 for c in population]
            feasible_pct = sum(1 for c in population if c.feasible) / len(population) * 100
            gen_data = {
                "phase": "generation",
                "generation": gen,
                "best_fitness": max(fitnesses),
                "mean_fitness": float(np.mean(fitnesses)),
                "worst_fitness": min(fitnesses),
                "feasible_pct": feasible_pct,
                "best_apogee": max(apogees),
            }
            gen_history.append(gen_data)
            if callback:
                callback(gen, config.max_generations, gen_data["best_fitness"], gen_data)

        best = max(population, key=lambda c: c.fitness)
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
            _valid_vector(positions, i, dvs, config.design_variables, lo, hi, rng)
        velocities = rng.uniform(-v_max, v_max, size=(pop_size, n))
        p_best_pos = positions.copy()
        p_best_fit = np.full(pop_size, -np.inf)
        g_best_pos = positions[0].copy()
        g_best_fit = -np.inf

        init_dicts = [_vec_to_dict(positions[i], dvs, config.design_variables)
                      for i in range(pop_size)]
        init_seeds = [i * 23 for i in range(pop_size)]
        population = _parallel_eval(executor, base_config, init_dicts, config,
                                    init_seeds, cancel_flag=cancel_flag)
        population = [c for c in population if c is not None]
        total_evals += config.mc_sims_per_candidate * len(population)

        for i in range(pop_size):
            cd = population[i]
            if cd.fitness > p_best_fit[i]:
                p_best_fit[i] = cd.fitness
                p_best_pos[i] = positions[i].copy()
            if cd.fitness > g_best_fit:
                g_best_fit = cd.fitness
                g_best_pos = positions[i].copy()

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

            swarm_dicts = [_vec_to_dict(positions[i], dvs, config.design_variables)
                           for i in range(pop_size)]
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

            fitnesses = [c.fitness for c in population]
            apogees = [c.mc_stats.get("mean_apogee", 0) if c.mc_stats else 0 for c in population]
            feasible_pct = sum(1 for c in population if c.feasible) / len(population) * 100
            gen_data = {
                "phase": "generation",
                "generation": gen,
                "best_fitness": g_best_fit,
                "mean_fitness": float(np.mean(fitnesses)),
                "worst_fitness": min(fitnesses),
                "feasible_pct": feasible_pct,
                "best_apogee": max(apogees),
            }
            gen_history.append(gen_data)
            if callback:
                callback(gen, config.max_generations, g_best_fit, gen_data)

        best = max(population, key=lambda c: c.fitness)
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

            self.all_done.emit(result)

        except Exception as e:
            logger.error(f"Optimisation worker failed: {e}", exc_info=True)
            self.failed.emit(str(e))


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

        base = BatchSimConfig.from_rocket_state(self.engine.state)
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
        if self._worker and self._worker._cancel[0]:
            self.optimization_cancelled.emit()
        self._worker = None
