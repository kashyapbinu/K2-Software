"""
K2 Aerospace — CFD Parameter Sweep Engine
==========================================
Runs a series of SU2 solves over a swept flow variable (Angle of Attack or
Mach number) reusing a SINGLE computational mesh, then extracts engineering
curves and derived stability/drag metrics.

Why one mesh for the whole sweep
--------------------------------
The mesh depends only on geometry + refinement + boundary-layer settings —
NOT on Mach or AoA. Standard CFD practice is to build one mesh sized for the
most demanding condition and reuse it across all flow points. This turns an
N-point AoA polar from "N full mesh+solve" into "1 mesh + N solves", which is
the dominant cost saving for sweeps.

Outputs
-------
  - Cl / Cd / Cm vs AoA   (lift curve, drag polar, pitch-stability)
  - Cd vs Mach            (drag-rise curve)
  - Transonic drag spike + CP migration
  - dCl/dα, dCm/dα (per-radian), drag-divergence Mach, CP travel envelope
"""
from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from cfd.solvers.base import CFDConfig, CFDResult
from cfd.solvers.su2_solver import SU2Solver

logger = logging.getLogger("K2.CFD.Sweep")

# Variables that can be swept and the CFDConfig attribute each maps to.
SWEEP_VARS = {
    "aoa":  "angle_of_attack_deg",
    "mach": "mach",
}


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class SweepPoint:
    """One solved point in a sweep: the swept input value + its CFD result."""
    var: str                 # "aoa" | "mach"
    value: float             # swept input (deg or Mach)
    result: CFDResult        # full CFD result at this point


@dataclass
class SweepData:
    """Container for a completed (or in-progress) sweep."""
    var: str = "aoa"                       # swept variable
    points: list[SweepPoint] = field(default_factory=list)

    # ── Curve accessors (return parallel lists, sorted by swept value) ──────
    def _sorted(self) -> list[SweepPoint]:
        return sorted(self.points, key=lambda p: p.value)

    def x(self) -> list[float]:
        return [p.value for p in self._sorted()]

    def cl(self) -> list[float]:
        return [p.result.cl for p in self._sorted()]

    def cd(self) -> list[float]:
        return [p.result.cd for p in self._sorted()]

    def cm(self) -> list[float]:
        return [p.result.cm for p in self._sorted()]

    def cm_cg(self) -> list[float]:
        """Pitching moment about the CG (static-stability moment) per point."""
        return [p.result.cm_cg for p in self._sorted()]

    def cp(self) -> list[float]:
        return [p.result.cp_location_m for p in self._sorted()]

    def cp_smooth(self) -> list[float]:
        """CP-from-nozzle with the indeterminate AoA≈0 point(s) filled in.

        Each point's CP is computed independently from its own integrated forces;
        only points where the normal force vanished (CP stored as the 0 sentinel)
        are interpolated from their loaded neighbours so the plotted CP-vs-AoA
        curve is smooth and shows the real few-mm migration instead of dropping
        to zero. No CP value is invented for a point that produced one.
        """
        pts = self._sorted()
        xs = [p.value for p in pts]
        ys = [p.result.cp_location_m for p in pts]
        valid = [(x, y) for x, y in zip(xs, ys) if y > 0.01]
        if len(valid) < 2:
            return ys
        vx = [x for x, _ in valid]
        vy = [y for _, y in valid]
        out = []
        for x, y in zip(xs, ys):
            if y > 0.01:
                out.append(y)
            else:
                # Linear interp/extrapolate from the nearest loaded points.
                if x <= vx[0]:
                    j = 1
                elif x >= vx[-1]:
                    j = len(vx) - 1
                else:
                    j = next(k for k in range(1, len(vx)) if vx[k] >= x)
                x0, x1 = vx[j - 1], vx[j]
                y0, y1 = vy[j - 1], vy[j]
                t = (x - x0) / (x1 - x0) if abs(x1 - x0) > 1e-12 else 0.0
                out.append(y0 + t * (y1 - y0))
        return out

    def cd_pressure(self) -> list[float]:
        return [p.result.cd_pressure for p in self._sorted()]

    def cd_friction(self) -> list[float]:
        return [p.result.cd_friction for p in self._sorted()]

    def cd_wave(self) -> list[float]:
        """Sweep-derived wave drag.

        SU2 splits total drag into pressure + viscous with no leftover, and
        wave drag lives *inside* pressure drag — so the solver's per-point
        ``cd_wave`` is always ~0. Across a Mach sweep we can recover it: wave
        drag at Mach M ≈ pressure drag at M minus the subsonic-bucket minimum
        pressure drag (the shock-free baseline). Only meaningful for a Mach
        sweep with ≥2 points; returns the solver field otherwise.
        """
        if self.var != "mach":
            return [p.result.cd_wave for p in self._sorted()]
        cdp = self.cd_pressure()
        if len(cdp) < 2:
            return [p.result.cd_wave for p in self._sorted()]
        baseline = min(cdp)   # drag-bucket minimum = shock-free pressure drag
        return [max(0.0, c - baseline) for c in cdp]


# ── Value-list builder ───────────────────────────────────────────────────────

def build_value_list(start: float, stop: float, step: float) -> list[float]:
    """Inclusive arange that tolerates float drift and reversed bounds.

    Always includes the start; includes the stop if it lands within half a
    step of the last grid point. Step sign is inferred from start/stop.
    """
    if step == 0:
        return [round(start, 6)]
    step = abs(step)
    if stop < start:
        step = -step
    n = int(math.floor((stop - start) / step + 1e-9))
    vals = [round(start + i * step, 6) for i in range(n + 1)]
    # Append the exact stop if the grid stopped just short of it.
    if vals and abs(vals[-1] - stop) > abs(step) * 0.5:
        vals.append(round(stop, 6))
    return vals


# ── Mesh staging ─────────────────────────────────────────────────────────────

def stage_mesh(mesh_path: Path, work_dir: Path) -> Path:
    """Make the shared mesh reachable from a point's work_dir.

    SU2 runs with cwd = work_dir and references the mesh by basename, so the
    mesh must live inside each point folder. We hardlink it (instant, no extra
    disk) and fall back to a copy across volumes or on filesystems without
    hardlink support.
    """
    import os, shutil
    mesh_path = Path(mesh_path)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    local = work_dir / mesh_path.name
    if local.exists():
        return local
    try:
        os.link(mesh_path, local)          # hardlink — same NTFS volume
    except Exception:
        shutil.copy2(mesh_path, local)     # cross-volume / unsupported → copy
    return local


# ── Per-point solve ──────────────────────────────────────────────────────────

def run_sweep_point(
    base_config: CFDConfig,
    var: str,
    value: float,
    mesh_path: Path,
    progress_cb: Optional[Callable[[int, float], None]] = None,
) -> CFDResult:
    """Solve ONE sweep point on a pre-built mesh.

    Clones ``base_config``, overrides the swept variable, points the solver at
    the already-generated ``mesh_path`` (skipping mesh generation), writes a
    fresh SU2 case, runs it, and returns the parsed CFDResult.

    Each point gets its own work sub-directory so VTK/history files don't clash.
    """
    if var not in SWEEP_VARS:
        raise ValueError(f"Unknown sweep variable '{var}'. Use one of {list(SWEEP_VARS)}.")

    cfg = copy.deepcopy(base_config)
    setattr(cfg, SWEEP_VARS[var], value)

    # Isolate each point's outputs in its own folder.
    tag = f"{var}_{value:+.3f}".replace("+", "p").replace("-", "m").replace(".", "_")
    cfg.work_dir = Path(base_config.work_dir) / "sweep" / tag
    cfg.work_dir.mkdir(parents=True, exist_ok=True)

    solver = SU2Solver(cfg)
    # Reuse the shared mesh — stage it into this point's folder, skip remesh.
    solver._mesh_path = stage_mesh(mesh_path, cfg.work_dir)
    if progress_cb is not None:
        solver.set_progress_callback(progress_cb)

    solver.generate_case()
    for _it, _rms in solver.run():
        pass  # progress already streamed via callback
    result = solver.parse_results()
    logger.info(
        f"Sweep point {var}={value:g} → Cd={result.cd:.4f} Cl={result.cl:.4f} "
        f"Cm={result.cm:.4f} conv={result.converged}"
    )
    return result


# ── Derived metrics ──────────────────────────────────────────────────────────

def _linfit_slope(xs: list[float], ys: list[float]) -> float:
    """Least-squares slope dy/dx. Returns 0 if degenerate."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx < 1e-12:
        return 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / sxx


def compute_sweep_metrics(data: SweepData) -> dict:
    """Extract engineering metrics from a completed sweep.

    AoA sweep → lift-curve slope, pitch-stability slope, CP travel, zero-α drag.
    Mach sweep → drag-divergence Mach, peak transonic Cd, CP shift.
    """
    metrics: dict = {}
    pts = data._sorted()
    if len(pts) < 2:
        return metrics

    xs = [p.value for p in pts]

    if data.var == "aoa":
        # Use the near-linear region |AoA| <= 8 deg for slopes.
        lin = [(p.value, p.result) for p in pts if abs(p.value) <= 8.0]
        if len(lin) < 2:
            lin = [(p.value, p.result) for p in pts]
        a_deg = [v for v, _ in lin]
        a_rad = [math.radians(v) for v in a_deg]
        cl = [r.cl for _, r in lin]
        cm = [r.cm for _, r in lin]
        metrics["cl_alpha_per_rad"] = _linfit_slope(a_rad, cl)
        metrics["cl_alpha_per_deg"] = _linfit_slope(a_deg, cl)
        # Nose-tip moment slope (informational only — NOT the stability metric).
        metrics["cm_alpha_per_rad"] = _linfit_slope(a_rad, cm)
        metrics["cm_alpha_per_deg"] = _linfit_slope(a_deg, cm)

        # ── Static pitch stability: dCm/dα about the CG ─────────────────────
        # The stability sign must come from the moment about the CG, not the
        # nose tip. Use only points that carry a valid CG-moment (CG known and
        # AoA loaded). dCm_cg/dα < 0 ⇒ restoring ⇒ statically stable.
        cg_pts = [(v, r) for v, r in lin if r.x_cg_m > 0.0 and abs(r.cl) > 1e-4]
        if len(cg_pts) >= 2:
            cg_rad = [math.radians(v) for v, _ in cg_pts]
            cm_cg = [r.cm_cg for _, r in cg_pts]
            slope = _linfit_slope(cg_rad, cm_cg)
            metrics["cm_cg_alpha_per_rad"] = slope
            metrics["cm_cg_alpha_per_deg"] = _linfit_slope([v for v, _ in cg_pts], cm_cg)
            # Marginal band: |dCm/dα| within this of zero ⇒ neutral.
            eps = 0.02   # per-rad neutral-stability tolerance
            if slope < -eps:
                metrics["stability_verdict"] = "Stable"
                metrics["statically_stable"] = True
            elif slope > eps:
                metrics["stability_verdict"] = "Unstable"
                metrics["statically_stable"] = False
            else:
                metrics["stability_verdict"] = "Marginal"
                metrics["statically_stable"] = None
        # Else: CG not supplied → leave stability undetermined (no false verdict).

        # Zero-AoA drag (min |α| point ≈ Cd0).
        zero = min(pts, key=lambda p: abs(p.value))
        metrics["cd0"] = zero.result.cd

        # ── Induced-drag fit: Cd ≈ Cd0 + k·Cl² ──────────────────────────────
        # Least-squares fit over the full sweep. Reports the physical drag-polar
        # parameters; it does NOT reshape the plotted curve (raw Cd is plotted).
        cl_all = [p.result.cl for p in pts]
        cd_all = [p.result.cd for p in pts]
        cl2 = [c * c for c in cl_all]
        k = _linfit_slope(cl2, cd_all)
        if k > 0:
            metrics["k_induced"] = k
            metrics["cd0_fit"] = (sum(cd_all) - k * sum(cl2)) / len(cd_all)

        # CP travel envelope over the sweep, using neighbour-filled CP so the
        # AoA≈0 indeterminate point doesn't masquerade as huge travel.
        cps = [c for c in data.cp_smooth() if c > 0.01]
        if cps:
            metrics["cp_min_m"] = min(cps)
            metrics["cp_max_m"] = max(cps)
            metrics["cp_travel_m"] = max(cps) - min(cps)

    elif data.var == "mach":
        cd = [p.result.cd for p in pts]
        # Peak (transonic/supersonic) drag.
        i_peak = max(range(len(cd)), key=lambda i: cd[i])
        metrics["cd_peak"] = cd[i_peak]
        metrics["mach_at_cd_peak"] = xs[i_peak]

        # Drag bucket: minimum Cd and its Mach (the shock-free baseline).
        i_min = min(range(len(cd)), key=lambda i: cd[i])
        metrics["cd_min"] = cd[i_min]
        metrics["mach_at_cd_min"] = xs[i_min]

        # Drag-divergence Mach. Two definitions:
        #   classic   — first Mach where dCd/dM ≥ 0.10 /Mach (aircraft-style;
        #               often never tripped by slender rockets).
        #   onset     — first Mach AFTER the drag minimum where Cd rises ≥5%
        #               above the bucket minimum. Robust for mild rocket rise.
        m_dd_classic = None
        for i in range(1, len(pts)):
            dm = xs[i] - xs[i - 1]
            if dm <= 1e-9:
                continue
            if (cd[i] - cd[i - 1]) / dm >= 0.10:
                m_dd_classic = xs[i - 1]
                break
        metrics["drag_divergence_mach_classic"] = m_dd_classic

        m_dd = None
        thresh = cd[i_min] * 1.05
        for i in range(i_min + 1, len(pts)):
            if cd[i] >= thresh:
                m_dd = xs[i]
                break
        # Prefer the onset definition; fall back to classic if onset not found.
        metrics["drag_divergence_mach"] = m_dd if m_dd is not None else m_dd_classic

        # Sweep-derived wave drag (pressure drag above the bucket baseline).
        wd = data.cd_wave()
        if wd:
            metrics["wave_drag_peak"] = max(wd)
            metrics["mach_at_wave_peak"] = xs[max(range(len(wd)), key=lambda i: wd[i])]

        # Subsonic baseline drag (lowest-Mach point) for rise ratio.
        metrics["cd_subsonic"] = cd[0]
        if cd[i_min] > 1e-9:
            metrics["transonic_drag_rise_ratio"] = cd[i_peak] / cd[i_min]
        # CP shift across the sweep.
        cps = [p.result.cp_location_m for p in pts if p.result.cp_location_m > 0.01]
        if len(cps) >= 2:
            metrics["cp_shift_m"] = max(cps) - min(cps)

    return metrics
