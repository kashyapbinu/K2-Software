"""
K2 AeroSim — Monte Carlo Analysis Engine
=============================================
Orchestrates parameter perturbation, parallel execution via QThreadPool,
and statistical aggregation for probabilistic flight performance analysis.

Architecture:
    MonteCarloConfig   — user-specified uncertainty parameters
    MonteCarloResults  — aggregated statistics from all runs
    _SimWorker         — QRunnable that executes one batch sim on a pool thread
    MonteCarloEngine   — QObject orchestrator: perturbs, dispatches, collects, analyzes

Usage:
    engine = MonteCarloEngine(state_engine)
    engine.analysis_finished.connect(on_results)
    engine.start(MonteCarloConfig(num_simulations=500))
"""

from __future__ import annotations

import copy
import math
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from PyQt6.QtCore import QObject, QThread, pyqtSignal

from core.batch_simulation import BatchSimConfig, BatchSimResult, run_batch_simulation

logger = logging.getLogger("K2.MonteCarlo")


# ─────────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────────

@dataclass
class MonteCarloConfig:
    """
    User-specified uncertainty parameters and failure thresholds
    for Monte Carlo analysis.

    All ``*_pct`` values are percentage of nominal (σ).
    All ``*_deg`` values are in degrees (σ).
    """
    # Run count
    num_simulations: int = 500

    # Perturbation σ values
    wind_speed_uncertainty_pct: float = 10.0       # % of nominal wind speed
    wind_direction_uncertainty_deg: float = 15.0    # degrees σ
    motor_impulse_uncertainty_pct: float = 5.0      # % of nominal
    dry_mass_uncertainty_pct: float = 3.0           # % of nominal
    drag_coeff_uncertainty_pct: float = 10.0        # % of nominal
    launch_angle_uncertainty_deg: float = 2.0       # degrees σ
    cg_uncertainty_mm: float = 5.0                  # mm σ

    # Failure thresholds
    min_stability_cal: float = 1.0                  # calibers
    min_rail_exit_velocity: float = 15.0            # m/s
    max_mach_limit: float = 5.0                     # Mach number

    # Target apogee (0 = auto-detect from a nominal baseline run)
    target_apogee: float = 0.0


# ─────────────────────────────────────────────────────────────────────
#  Results
# ─────────────────────────────────────────────────────────────────────

@dataclass
class MonteCarloResults:
    """
    Aggregated statistics from a complete Monte Carlo analysis.
    """
    # Raw data
    all_runs: list = field(default_factory=list)            # list[BatchSimResult]
    perturbed_params: list = field(default_factory=list)    # list[dict]

    # Apogee statistics
    apogee_mean: float = 0.0
    apogee_std: float = 0.0
    apogee_min: float = 0.0
    apogee_max: float = 0.0
    apogee_ci_95: tuple = (0.0, 0.0)
    apogee_median: float = 0.0
    apogee_values: list = field(default_factory=list)

    # Velocity statistics
    velocity_mean: float = 0.0
    velocity_std: float = 0.0
    velocity_values: list = field(default_factory=list)

    # Mach statistics
    mach_mean: float = 0.0
    mach_std: float = 0.0
    mach_values: list = field(default_factory=list)

    # Acceleration statistics
    accel_mean: float = 0.0
    accel_std: float = 0.0
    accel_values: list = field(default_factory=list)

    # Landing statistics
    landing_distance_mean: float = 0.0
    landing_distance_std: float = 0.0
    landing_distance_max: float = 0.0
    landing_x_values: list = field(default_factory=list)
    landing_y_values: list = field(default_factory=list)
    landing_distance_values: list = field(default_factory=list)

    # Mission success / failure
    success_count: int = 0
    failure_count: int = 0
    success_percentage: float = 0.0
    failure_percentage: float = 0.0
    failure_breakdown: dict = field(default_factory=dict)   # reason -> count

    # Probabilities
    prob_target_altitude: float = 0.0
    prob_safe_recovery: float = 0.0
    mission_success_probability: float = 0.0

    # Rail exit velocity
    rail_exit_velocity_mean: float = 0.0
    rail_exit_velocity_std: float = 0.0

    # Best / worst runs
    best_run_index: int = 0
    worst_run_index: int = 0

    # Outlier filtering
    n_valid: int = 0
    n_outliers: int = 0
    n_physics_invalid: int = 0
    n_unstable: int = 0          # flew but static margin < required caliber (would tumble)
    unstable_landing_x: list = field(default_factory=list)   # landing East (m) of unstable runs
    unstable_landing_y: list = field(default_factory=list)   # landing North (m) of unstable runs
    unstable_apogee_values: list = field(default_factory=list)  # apogee (m) of unstable runs

    # Sensitivity analysis
    sensitivity_correlations: dict = field(default_factory=dict)

    # Extended distribution metrics (Problem 5)
    apogee_skewness: float = 0.0
    apogee_kurtosis: float = 0.0
    apogee_mode: float = 0.0
    apogee_percentiles: dict = field(default_factory=dict)  # {5,10,25,50,75,90,95}
    exceedance_probabilities: dict = field(default_factory=dict)  # altitude -> P(apogee > alt)

    # Reliability metrics (Issue 9)
    p_apogee_above_target: float = 0.0
    p_mach_below_limit: float = 0.0
    p_accel_below_limit: float = 0.0
    p_stability_above_limit: float = 0.0
    p_rail_exit_above_min: float = 0.0
    reliability_index_beta: float = 0.0   # beta = (mean - limit) / sigma
    reliability_confidence_interval: tuple = (0.0, 0.0)  # 95% CI on mission success

    @property
    def runs(self): return self.all_runs
    
    @property
    def apogee_ci_low(self): return self.apogee_ci_95[0]
    
    @property
    def apogee_ci_high(self): return self.apogee_ci_95[1]
    
    @property
    def max_velocity_mean(self): return self.velocity_mean
    
    @property
    def max_mach_mean(self): return self.mach_mean
    
    @property
    def max_accel_mean(self): return self.accel_mean
    
    @property
    def landing_dist_mean(self): return self.landing_distance_mean
    
    @property
    def landing_dist_std(self): return self.landing_distance_std
    
    @property
    def landing_dist_max(self): return self.landing_distance_max
    
    @property
    def success_rate(self): return self.success_percentage
    
    @property
    def failure_rate(self): return self.failure_percentage
    
    @property
    def p_target_alt(self): return self.prob_target_altitude * 100.0
    
    @property
    def p_safe_recovery(self): return self.prob_safe_recovery * 100.0
    
    @property
    def mission_success(self): return self.mission_success_probability * 100.0
    
    @property
    def best_apogee(self): return self.all_runs[self.best_run_index].apogee if self.all_runs else 0.0
    
    @property
    def worst_apogee(self): return self.all_runs[self.worst_run_index].apogee if self.all_runs else 0.0

    @property
    def rail_exit_vel_mean(self): return self.rail_exit_velocity_mean

    @property
    def rail_exit_vel_std(self): return self.rail_exit_velocity_std


# ─────────────────────────────────────────────────────────────────────
#  Background worker thread (runs sims sequentially, no pickling needed)
# ─────────────────────────────────────────────────────────────────────

class _BatchWorkerThread(QThread):
    """
    Runs all Monte Carlo simulations sequentially on a background QThread.
    Emits progress every 2% to keep UI overhead low.

    With RK4 at dt=0.05 each sim takes ~75ms, so:
      200 sims ≈ 15s, 500 sims ≈ 37s, 1000 sims ≈ 75s
    """
    progress = pyqtSignal(int, int)
    all_done = pyqtSignal(object)         # list[(run_index, BatchSimResult)]
    failed = pyqtSignal(str)

    def __init__(self, work_items: list):
        """work_items: list[(run_index, BatchSimConfig, seed)]"""
        super().__init__()
        self._work_items = work_items
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        results = []
        total = len(self._work_items)
        last_pct_emitted = -1

        for run_index, config, seed in self._work_items:
            if self._cancelled:
                return

            try:
                result = run_batch_simulation(config, seed=seed)
            except Exception as exc:
                result = BatchSimResult(
                    success=False,
                    failure_reasons=[f"Simulation error: {exc}"],
                    final_phase="Terminated",
                )

            results.append((run_index, result))

            # Batch progress: emit every 2% or on last run
            completed = len(results)
            pct = completed * 100 // total
            if pct >= last_pct_emitted + 2 or completed == total:
                self.progress.emit(completed, total)
                last_pct_emitted = pct

        if not self._cancelled:
            self.all_done.emit(results)


# ─────────────────────────────────────────────────────────────────────
#  Monte Carlo Engine
# ─────────────────────────────────────────────────────────────────────

class MonteCarloEngine(QObject):
    """
    Main orchestrator for Monte Carlo flight analysis.

    Workflow:
        1. ``start(mc_config)`` — perturb parameters, dispatch workers
        2. Workers run in parallel on QThreadPool
        3. ``_on_run_complete`` collects results thread-safely
        4. ``_compute_statistics`` aggregates and emits ``analysis_finished``
    """

    # Signals
    progress = pyqtSignal(int, int)              # (completed, total)
    run_completed = pyqtSignal(int, object)      # (run_index, BatchSimResult)
    analysis_finished = pyqtSignal(object)       # MonteCarloResults
    analysis_failed = pyqtSignal(str)            # error message
    analysis_cancelled = pyqtSignal()

    def __init__(self, state_engine):
        super().__init__()
        self.engine = state_engine

        self._worker_thread: Optional[_BatchWorkerThread] = None

        self._cancelled = False
        self._completed_count = 0
        self._run_results: list[tuple[int, BatchSimResult]] = []
        self._perturbed_params: list[dict] = []
        self._mc_config: Optional[MonteCarloConfig] = None
        self._total: int = 0
        self._running = False

    # ── Public API ────────────────────────────────────────────────

    def start(self, mc_config: MonteCarloConfig) -> None:
        """
        Launch the full Monte Carlo analysis.

        Args:
            mc_config: User-specified uncertainty configuration.
        """
        if self._running:
            logger.warning("Monte Carlo analysis already running")
            return

        logger.info(
            f"Starting Monte Carlo analysis: {mc_config.num_simulations} runs"
        )

        # Reset state
        self._cancelled = False
        self._completed_count = 0
        self._run_results = []
        self._perturbed_params = []
        self._mc_config = mc_config
        self._total = mc_config.num_simulations
        self._running = True

        # Create base config from current rocket state
        try:
            base_config = BatchSimConfig.from_rocket_state(self.engine.state)
        except Exception as exc:
            self._running = False
            self.analysis_failed.emit(f"Failed to build base config: {exc}")
            return

        # Auto-detect target apogee from nominal run
        target_apogee = mc_config.target_apogee
        if target_apogee <= 0.0:
            try:
                nominal_result = run_batch_simulation(base_config, seed=42)
                target_apogee = nominal_result.apogee
                logger.info(f"Nominal apogee baseline: {target_apogee:.1f} m")
            except Exception as exc:
                self._running = False
                self.analysis_failed.emit(
                    f"Nominal baseline simulation failed: {exc}"
                )
                return
            if target_apogee <= 0.0:
                self._running = False
                self.analysis_failed.emit(
                    "Nominal simulation produced zero apogee — check rocket configuration."
                )
                return
        # Store resolved target for statistics
        self._target_apogee = target_apogee

        # RNG for reproducible perturbation generation
        rng = np.random.default_rng(12345)

        # Build work items (pass configs directly — no pickling needed)
        work_items = []
        for i in range(mc_config.num_simulations):
            if self._cancelled:
                break

            perturbed, params_dict = self._perturb_config(
                base_config, mc_config, rng
            )
            self._perturbed_params.append(params_dict)

            work_items.append((i, perturbed, 42 + i))

        if self._cancelled:
            self._running = False
            self.analysis_cancelled.emit()
            return

        # Launch background thread
        self._worker_thread = _BatchWorkerThread(work_items)
        self._worker_thread.progress.connect(self._on_progress)
        self._worker_thread.all_done.connect(self._on_all_done)
        self._worker_thread.failed.connect(self._on_pool_failed)
        self._worker_thread.start()

        logger.info(f"Monte Carlo analysis launched: {len(work_items)} runs")

    def cancel(self) -> None:
        """Cancel the running analysis."""
        self._cancelled = True
        if self._worker_thread is not None:
            self._worker_thread.cancel()
        self._running = False
        logger.info("Monte Carlo analysis cancelled")
        self.analysis_cancelled.emit()

    @property
    def is_running(self) -> bool:
        """True while analysis is in progress."""
        return self._running

    # ── Parameter Perturbation ────────────────────────────────────

    @staticmethod
    def _clamp_gaussian(rng, sigma: float, n_sigma: float = 3.0) -> float:
        """Sample from N(0, σ) clamped to ±n_sigma*σ."""
        sigma = max(sigma, 1e-12)
        val = float(rng.normal(0, sigma))
        limit = n_sigma * sigma
        return max(-limit, min(limit, val))

    @staticmethod
    def _perturb_config(
        base: BatchSimConfig,
        mc: MonteCarloConfig,
        rng,
    ) -> tuple[BatchSimConfig, dict]:
        """
        Create a perturbed copy of *base* by sampling Gaussian noise
        from the uncertainties defined in *mc*.

        All perturbations are clamped to ±3σ to prevent physically
        impossible parameter combinations.

        Returns:
            (perturbed_config, params_dict) where params_dict logs the
            perturbed values for post-analysis.
        """
        _cg = MonteCarloEngine._clamp_gaussian
        cfg = copy.deepcopy(base)
        params: dict = {}

        if base.wind_mode == "multi_level" and base.wind_layers:
            # Multi-level wind: perturb the whole profile coherently — one
            # speed scale factor and one direction offset applied to every
            # layer (forecast uncertainty shifts the profile as a unit;
            # per-layer independent noise would create unphysical shear).
            speed_scale = max(0.0, 1.0 + _cg(
                rng, mc.wind_speed_uncertainty_pct / 100.0))
            dir_offset = _cg(rng, mc.wind_direction_uncertainty_deg)
            cfg.wind_layers = [
                (alt, spd * speed_scale, (drn + dir_offset) % 360.0)
                for alt, spd, drn in base.wind_layers
            ]
            # Log ground-layer values so correlation/scatter plots stay meaningful
            params["wind_speed"] = cfg.wind_layers[0][1]
            params["wind_direction"] = cfg.wind_layers[0][2]
        else:
            # Wind speed: σ = base * pct / 100
            ws_sigma = max(base.wind_speed * mc.wind_speed_uncertainty_pct / 100.0, 0.01)
            cfg.wind_speed = max(0.0, base.wind_speed + _cg(rng, ws_sigma))
            params["wind_speed"] = cfg.wind_speed

            # Wind direction: σ = deg
            cfg.wind_direction = base.wind_direction + _cg(rng, mc.wind_direction_uncertainty_deg)
            params["wind_direction"] = cfg.wind_direction

        # Dry mass: σ = base * pct / 100
        dm_sigma = max(base.dry_mass * mc.dry_mass_uncertainty_pct / 100.0, 0.001)
        cfg.dry_mass = max(0.05, base.dry_mass + _cg(rng, dm_sigma))
        params["dry_mass"] = cfg.dry_mass

        # Drag coefficient: σ = base * pct / 100
        cd_sigma = max(base.cd * mc.drag_coeff_uncertainty_pct / 100.0, 0.001)
        cfg.cd = max(0.05, min(2.0, base.cd + _cg(rng, cd_sigma)))
        params["cd"] = cfg.cd

        # Launch angle: σ = deg
        cfg.launch_angle = max(0.0, min(90.0,
            base.launch_angle + _cg(rng, mc.launch_angle_uncertainty_deg)
        ))
        params["launch_angle"] = cfg.launch_angle

        # CG: σ = mm / 1000 (convert to metres) — perturb both cg and dry_cg
        cg_sigma_m = mc.cg_uncertainty_mm / 1000.0
        cg_delta = _cg(rng, cg_sigma_m)
        cfg.cg = max(0.001, base.cg + cg_delta)
        cfg.dry_cg = max(0.001, base.dry_cg + cg_delta)  # Fix: keep dry_cg in sync
        params["cg"] = cfg.cg

        # Motor impulse: multiplicative scale factor = 1 + N(0, pct/100)
        impulse_sigma = mc.motor_impulse_uncertainty_pct / 100.0
        impulse_scale = 1.0 + _cg(rng, impulse_sigma)
        impulse_scale = max(0.5, min(1.5, impulse_scale))
        cfg.motor_avg_thrust = base.motor_avg_thrust * impulse_scale
        cfg.motor_max_thrust = base.motor_max_thrust * impulse_scale
        cfg.motor_total_impulse = base.motor_total_impulse * impulse_scale
        # Scale propellant mass proportionally to maintain energy balance
        cfg.propellant_mass = base.propellant_mass * impulse_scale
        params["impulse_scale"] = impulse_scale

        # Scale custom thrust curve points if present
        if cfg.custom_thrust_curve:
            cfg.custom_thrust_curve = [
                (t, v * impulse_scale) for t, v in cfg.custom_thrust_curve
            ]

        return cfg, params

    # ── Callbacks from _PoolManagerThread ──────────────────────────

    def _on_progress(self, completed: int, total: int) -> None:
        """Relay progress from pool manager thread."""
        self._completed_count = completed
        self.progress.emit(completed, total)


    def _on_pool_failed(self, error_msg: str) -> None:
        """Handle process pool failure."""
        self._running = False
        self.analysis_failed.emit(error_msg)

    def _on_all_done(self, results: list) -> None:
        """All simulations finished — compute statistics."""
        if self._cancelled:
            return
        self._run_results = results

        # Log any failed runs for debugging
        failed = [(i, r) for i, r in results if not r.success]
        if failed:
            logger.warning(f"{len(failed)} of {len(results)} runs failed")
            for idx, r in failed[:3]:  # Show first 3 errors
                logger.warning(f"  Run {idx}: {r.failure_reasons}")

        try:
            self._compute_statistics()
        except Exception as exc:
            logger.error(f"Statistics computation failed: {exc}")
            self.analysis_failed.emit(f"Statistics error: {exc}")
            self._running = False

    # ── Statistical Aggregation ───────────────────────────────────

    def _compute_statistics(self) -> None:
        """
        Three-tier run classification → valid-only statistics.

        Tier 1: Physics gate (deterministic) — reject diverged / impossible runs
        Tier 2: Mission criteria — stability, recovery, Mach limits
        Tier 3: Statistical outlier (IQR+3σ) — applied to physics-valid runs only
        """
        mc = self._mc_config
        if mc is None:
            return

        # Sort results by run_index for deterministic ordering
        self._run_results.sort(key=lambda x: x[0])
        runs = [r for _, r in self._run_results]

        n = len(runs)
        if n == 0:
            self._running = False
            self.analysis_failed.emit("No completed runs to analyze")
            return

        # ══════════════════════════════════════════════════════════
        #  TIER 1: Physics validity gate (deterministic)
        # ══════════════════════════════════════════════════════════
        # These thresholds catch numerical divergence / impossible states.
        # Runs failing these are NOT real flights — they are sim artifacts.
        PHYSICS_LIMITS = {
            "max_velocity": 2000.0,       # m/s  (≈ Mach 6 at sea level)
            "max_acceleration": 200*9.81, # 200G — above any solid motor
            "max_apogee": 100_000,        # 100 km (Kármán line)
        }

        physics_valid_indices = []
        physics_invalid_indices = []
        failure_breakdown: dict[str, int] = {}

        for i, r in enumerate(runs):
            reasons: list[str] = []

            # Check for sim-level failures first (integration errors, etc.)
            if r.failure_reasons:
                for fr in r.failure_reasons:
                    if fr not in reasons:
                        reasons.append(fr)

            # Physics sanity checks
            if r.max_velocity > PHYSICS_LIMITS["max_velocity"]:
                reasons.append(f"Numerical divergence (V={r.max_velocity:.0f} m/s)")
            if r.max_acceleration > PHYSICS_LIMITS["max_acceleration"]:
                reasons.append(f"Numerical divergence (a={r.max_acceleration/9.81:.0f}G)")
            if r.apogee > PHYSICS_LIMITS["max_apogee"]:
                reasons.append(f"Numerical divergence (alt={r.apogee:.0f} m)")

            if reasons:
                physics_invalid_indices.append(i)
                r.success = False
                r.failure_reasons = reasons
                for reason in reasons:
                    failure_breakdown[reason] = failure_breakdown.get(reason, 0) + 1
            else:
                physics_valid_indices.append(i)

        n_physics_invalid = len(physics_invalid_indices)
        logger.info(
            f"Physics gate: {len(physics_valid_indices)} valid, "
            f"{n_physics_invalid} invalid"
        )

        # ══════════════════════════════════════════════════════════
        #  TIER 2: Mission criteria (on physics-valid runs only)
        # ══════════════════════════════════════════════════════════
        success_count = 0
        failure_count = 0

        for i in physics_valid_indices:
            r = runs[i]
            reasons: list[str] = []

            # Stability check
            if r.min_stability_margin < mc.min_stability_cal and r.apogee > 1.0:
                reasons.append("Low stability margin")

            # Rail exit velocity
            if r.rail_exit_velocity < mc.min_rail_exit_velocity and r.apogee > 1.0:
                reasons.append("Low rail exit velocity")

            # Mach limit
            if r.max_mach > mc.max_mach_limit:
                reasons.append("Exceeded Mach limit")

            # Structural overload (>100G)
            if r.max_acceleration > 100 * 9.81:
                reasons.append("Structural overload (>100G)")

            # Non-recovery
            if r.final_phase not in ("Landed",):
                reasons.append("Recovery failure")

            if reasons:
                failure_count += 1
                r.success = False
                r.failure_reasons = reasons
                for reason in reasons:
                    failure_breakdown[reason] = failure_breakdown.get(reason, 0) + 1
            else:
                success_count += 1
                r.success = True
                r.failure_reasons = []

        # Also count physics-invalid as failures
        failure_count += n_physics_invalid

        success_pct = (success_count / n) * 100.0 if n > 0 else 0.0
        failure_pct = (failure_count / n) * 100.0 if n > 0 else 0.0

        # ══════════════════════════════════════════════════════════
        #  Extract arrays from PHYSICS-VALID runs only
        # ══════════════════════════════════════════════════════════
        pv_runs = [runs[i] for i in physics_valid_indices]
        npv = len(pv_runs)

        if npv == 0:
            # All runs diverged — report failure with all-run data
            self._running = False
            self.analysis_failed.emit(
                f"All {n} simulations failed physics validation. "
                f"Check rocket configuration."
            )
            return

        apogees = np.array([r.apogee for r in pv_runs], dtype=np.float64)
        velocities = np.array([r.max_velocity for r in pv_runs], dtype=np.float64)
        machs = np.array([r.max_mach for r in pv_runs], dtype=np.float64)
        accels = np.array([r.max_acceleration for r in pv_runs], dtype=np.float64)
        landing_xs = np.array([r.landing_x for r in pv_runs], dtype=np.float64)
        landing_ys = np.array([r.landing_y for r in pv_runs], dtype=np.float64)
        landing_dists = np.array([r.landing_distance for r in pv_runs], dtype=np.float64)
        rail_vels = np.array([r.rail_exit_velocity for r in pv_runs], dtype=np.float64)

        # ══════════════════════════════════════════════════════════
        #  TIER 3: Statistical outlier filtering (IQR + 3σ)
        #  Applied to physics-valid runs only
        # ══════════════════════════════════════════════════════════
        q1 = float(np.percentile(apogees, 25))
        q3 = float(np.percentile(apogees, 75))
        iqr = q3 - q1
        iqr_mask = (apogees >= q1 - 3.0 * iqr) & (apogees <= q3 + 3.0 * iqr)

        mu_raw = float(np.mean(apogees))
        sigma_raw = float(np.std(apogees, ddof=1)) if npv > 1 else 0.0
        if sigma_raw > 1e-6:
            sigma_mask = np.abs(apogees - mu_raw) <= 3.0 * sigma_raw
        else:
            sigma_mask = np.ones(npv, dtype=bool)

        valid_mask = iqr_mask & sigma_mask
        n_valid = int(np.sum(valid_mask))
        n_outliers = npv - n_valid

        # Fall back to all physics-valid if too few survive
        if n_valid < max(5, npv // 4):
            valid_mask = np.ones(npv, dtype=bool)
            n_valid = npv
            n_outliers = 0

        v_apogees = apogees[valid_mask]
        v_velocities = velocities[valid_mask]
        v_machs = machs[valid_mask]
        v_accels = accels[valid_mask]
        v_landing_xs = landing_xs[valid_mask]
        v_landing_ys = landing_ys[valid_mask]
        v_landing_dists = landing_dists[valid_mask]
        v_rail_vels = rail_vels[valid_mask]
        nv = len(v_apogees)

        # ══════════════════════════════════════════════════════════
        #  Compute statistics from valid runs
        # ══════════════════════════════════════════════════════════
        from scipy import stats as sp_stats

        # ── Apogee stats ──
        apogee_mean = float(np.mean(v_apogees))
        apogee_std = float(np.std(v_apogees, ddof=1)) if nv > 1 else 0.0
        apogee_min = float(np.min(v_apogees))
        apogee_max = float(np.max(v_apogees))
        apogee_median = float(np.median(v_apogees))
        ci_half = 1.96 * apogee_std / math.sqrt(nv) if nv > 0 else 0.0
        apogee_ci_95 = (apogee_mean - ci_half, apogee_mean + ci_half)

        # ── Extended distribution metrics (Problem 5) ──
        apogee_skewness = float(sp_stats.skew(v_apogees)) if nv > 2 else 0.0
        apogee_kurtosis = float(sp_stats.kurtosis(v_apogees)) if nv > 3 else 0.0

        # Mode via KDE (more robust than histogram for continuous data)
        if nv > 5:
            try:
                kde = sp_stats.gaussian_kde(v_apogees)
                x_grid = np.linspace(apogee_min, apogee_max, 200)
                apogee_mode = float(x_grid[np.argmax(kde(x_grid))])
            except Exception:
                apogee_mode = apogee_median
        else:
            apogee_mode = apogee_median

        # Percentile table
        pct_keys = [5, 10, 25, 50, 75, 90, 95]
        apogee_percentiles = {
            p: float(np.percentile(v_apogees, p)) for p in pct_keys
        }

        # Exceedance probabilities
        target = self._target_apogee
        exceedance_alts = set()
        if target > 0:
            for frac in [0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2]:
                exceedance_alts.add(round(target * frac))
        # Also add round numbers near the mean
        base = round(apogee_mean / 500) * 500
        for offset in [-1000, -500, 0, 500, 1000]:
            alt = base + offset
            if alt > 0:
                exceedance_alts.add(alt)
        exceedance_probabilities = {}
        for alt in sorted(exceedance_alts):
            prob = float(np.sum(v_apogees >= alt)) / nv
            exceedance_probabilities[alt] = prob

        # ── Velocity stats ──
        vel_mean = float(np.mean(v_velocities))
        vel_std = float(np.std(v_velocities, ddof=1)) if nv > 1 else 0.0

        # ── Mach stats ──
        mach_mean = float(np.mean(v_machs))
        mach_std = float(np.std(v_machs, ddof=1)) if nv > 1 else 0.0

        # ── Acceleration stats ──
        accel_mean = float(np.mean(v_accels))
        accel_std = float(np.std(v_accels, ddof=1)) if nv > 1 else 0.0

        # ── Landing stats ──
        ldist_mean = float(np.mean(v_landing_dists))
        ldist_std = float(np.std(v_landing_dists, ddof=1)) if nv > 1 else 0.0
        ldist_max = float(np.max(v_landing_dists))

        # ── Rail exit velocity ──
        rev_mean = float(np.mean(v_rail_vels))
        rev_std = float(np.std(v_rail_vels, ddof=1)) if nv > 1 else 0.0

        # ── Probabilities (from physics-valid runs) ──
        prob_target = float(np.sum(v_apogees >= target * 0.95)) / nv if nv > 0 else 0.0
        prob_recovery = float(
            sum(1 for r in pv_runs if r.final_phase == "Landed")
        ) / npv if npv > 0 else 0.0
        mission_prob = success_count / n if n > 0 else 0.0

        # ── Reliability metrics (Issue 9) ──
        p_mach_below = float(np.sum(v_machs <= mc.max_mach_limit)) / nv if nv > 0 else 0.0
        p_accel_below = float(np.sum(v_accels <= 100 * 9.81)) / nv if nv > 0 else 0.0
        p_stability_above = float(
            sum(1 for r in pv_runs if r.min_stability_margin >= mc.min_stability_cal)
        ) / npv if npv > 0 else 0.0
        p_rail_above = float(
            sum(1 for r in pv_runs if r.rail_exit_velocity >= mc.min_rail_exit_velocity)
        ) / npv if npv > 0 else 0.0

        # Unstable runs: flew (apogee > 1 m) but the minimum static margin dropped
        # below the required caliber, i.e. they would weathercock / tumble.
        # Counted across ALL runs (distinct from the numerically-diverged runs
        # already tallied in n_physics_invalid).
        unstable_runs = [r for r in runs
                         if r.min_stability_margin < mc.min_stability_cal
                         and r.apogee > 1.0]
        n_unstable = len(unstable_runs)
        unstable_landing_x = [getattr(r, "landing_x", 0.0) for r in unstable_runs]
        unstable_landing_y = [getattr(r, "landing_y", 0.0) for r in unstable_runs]
        unstable_apogee_values = [r.apogee for r in unstable_runs]

        # Reliability index beta = (mean - limit) / sigma
        if apogee_std > 1e-6 and target > 0:
            beta_idx = (apogee_mean - target) / apogee_std
        else:
            beta_idx = 0.0

        # Wilson score 95% CI on mission success probability
        if n > 0:
            p_hat = mission_prob
            z = 1.96
            denom = 1 + z**2 / n
            centre = (p_hat + z**2 / (2*n)) / denom
            spread = z * math.sqrt((p_hat*(1-p_hat) + z**2/(4*n)) / n) / denom
            mission_ci = (max(0, centre - spread), min(1, centre + spread))
        else:
            mission_ci = (0.0, 0.0)

        # ── Best / worst (from physics-valid runs) ──
        best_idx = physics_valid_indices[int(np.argmax(apogees))]
        worst_idx = physics_valid_indices[int(np.argmin(apogees))]

        # ── Sensitivity analysis (valid runs only!) ──
        valid_pv_params = [self._perturbed_params[i] for i in physics_valid_indices]
        sensitivity = self._compute_sensitivity(pv_runs, valid_pv_params)

        # ── Build results ──
        results = MonteCarloResults(
            all_runs=runs,
            perturbed_params=self._perturbed_params[:n],

            apogee_mean=apogee_mean,
            apogee_std=apogee_std,
            apogee_min=apogee_min,
            apogee_max=apogee_max,
            apogee_ci_95=apogee_ci_95,
            apogee_median=apogee_median,
            apogee_values=v_apogees.tolist(),

            velocity_mean=vel_mean,
            velocity_std=vel_std,
            velocity_values=v_velocities.tolist(),

            mach_mean=mach_mean,
            mach_std=mach_std,
            mach_values=v_machs.tolist(),

            accel_mean=accel_mean,
            accel_std=accel_std,
            accel_values=v_accels.tolist(),

            landing_distance_mean=ldist_mean,
            landing_distance_std=ldist_std,
            landing_distance_max=ldist_max,
            landing_x_values=v_landing_xs.tolist(),
            landing_y_values=v_landing_ys.tolist(),
            landing_distance_values=v_landing_dists.tolist(),

            success_count=success_count,
            failure_count=failure_count,
            success_percentage=success_pct,
            failure_percentage=failure_pct,
            failure_breakdown=failure_breakdown,

            prob_target_altitude=prob_target,
            prob_safe_recovery=prob_recovery,
            mission_success_probability=mission_prob,

            rail_exit_velocity_mean=rev_mean,
            rail_exit_velocity_std=rev_std,

            best_run_index=best_idx,
            worst_run_index=worst_idx,

            n_valid=n_valid,
            n_outliers=n_outliers,
            n_physics_invalid=n_physics_invalid,
            n_unstable=n_unstable,
            unstable_landing_x=unstable_landing_x,
            unstable_landing_y=unstable_landing_y,
            unstable_apogee_values=unstable_apogee_values,
            sensitivity_correlations=sensitivity,

            # Extended distribution metrics
            apogee_skewness=apogee_skewness,
            apogee_kurtosis=apogee_kurtosis,
            apogee_mode=apogee_mode,
            apogee_percentiles=apogee_percentiles,
            exceedance_probabilities=exceedance_probabilities,

            # Reliability metrics
            p_apogee_above_target=prob_target,
            p_mach_below_limit=p_mach_below,
            p_accel_below_limit=p_accel_below,
            p_stability_above_limit=p_stability_above,
            p_rail_exit_above_min=p_rail_above,
            reliability_index_beta=beta_idx,
            reliability_confidence_interval=mission_ci,
        )

        self._running = False

        logger.info(
            f"Monte Carlo complete — {n} runs "
            f"({n_physics_invalid} physics-invalid, {n_unstable} unstable, "
            f"{n_outliers} outliers, {n_valid} clean) | "
            f"Apogee: {apogee_mean:.1f} ± {apogee_std:.1f} m | "
            f"Success: {success_pct:.1f}% | "
            f"P(target): {prob_target:.1%} | "
            f"Skew: {apogee_skewness:.2f}, Kurt: {apogee_kurtosis:.2f}"
        )

        self.analysis_finished.emit(results)

    @staticmethod
    def _compute_sensitivity(runs, perturbed_params) -> dict:
        """
        Compute Pearson and Spearman correlations between each perturbed
        parameter and apogee.

        IMPORTANT: Must be called with physics-valid runs only to avoid
        diverged runs distorting correlations.
        """
        from scipy import stats as sp_stats

        if len(runs) < 10 or len(perturbed_params) < len(runs):
            return {}

        apogees = np.array([r.apogee for r in runs], dtype=np.float64)

        # Only use successful runs for sensitivity (failed runs distort correlations)
        valid_mask = np.array([r.success for r in runs], dtype=bool)
        if np.sum(valid_mask) < 10:
            # Not enough successful runs — use all physics-valid runs
            valid_mask = np.ones(len(runs), dtype=bool)

        valid_apogees = apogees[valid_mask]
        valid_params = [p for p, m in zip(perturbed_params, valid_mask) if m]

        correlations = {}

        param_labels = {
            "wind_speed": "Wind Speed",
            "wind_direction": "Wind Direction",
            "dry_mass": "Dry Mass",
            "cd": "Drag Coefficient",
            "launch_angle": "Launch Angle",
            "cg": "CG Location",
            "impulse_scale": "Motor Impulse",
        }

        for key, label in param_labels.items():
            try:
                values = np.array([p.get(key, 0) for p in valid_params], dtype=np.float64)
                if np.std(values) < 1e-12:
                    continue
                r_pearson, p_val = sp_stats.pearsonr(values, valid_apogees)
                r_spearman, _ = sp_stats.spearmanr(values, valid_apogees)
                correlations[label] = {
                    "pearson_r": float(r_pearson),
                    "spearman_r": float(r_spearman),
                    "p_value": float(p_val),
                    "param_key": key,
                }
            except Exception:
                continue

        return correlations

