"""
K2 AeroSim — CFD Parameter Sweep Engine
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
        """Per-point wave drag, straight from the solver.

        This used to reconstruct wave drag across a Mach sweep by subtracting
        the drag-bucket minimum pressure drag, because the solver's per-point
        ``cd_wave`` was the residue of (total - pressure - friction) and so was
        always ~0. It is now solved per point — the forebody part of the
        integrated pressure drag, base drag removed by its own surface integral
        (cfd/drag_decomposition.py) — so the reconstruction is both unnecessary
        and worse: the old baseline was a *pressure* drag minimum, which carries
        base drag in it, and subtracting it took base drag out of a wave-drag
        number instead of out of the forebody term.
        """
        return [p.result.cd_wave for p in self._sorted()]

    def cd_forebody_pressure(self) -> list[float]:
        """Forebody (non-base) pressure drag per point. Equals cd_wave at
        M >= 0.8; below that it is the numerical/form pressure drag that an
        inviscid body should not have, kept visible rather than relabelled."""
        return [p.result.cd_forebody_pressure for p in self._sorted()]

    def cd_base(self) -> list[float]:
        return [p.result.cd_base for p in self._sorted()]


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


# ── Analytic skin-friction build-up (Euler hybrid polar mode) ────────────────

def analytic_friction_cd(
    geometry: dict,
    reynolds: float,
    mach: float,
    ref_area: float,
) -> Optional[float]:
    """Component flat-plate skin-friction drag coefficient (about ref_area).

    Used with inviscid (Euler) sweep points: SU2 supplies the pressure field
    (lift, moments, CP, wave drag) and this supplies the friction drag the
    inviscid solve cannot. Standard engineering build-up — the same approach
    OpenRocket/Barrowman-class tools use:

      Cf   = 0.455 / (log10 Re)^2.58            (Schlichting, fully turbulent)
      Cf  *= (1 + 0.144 M²)^-0.65               (compressibility correction)
      body FF = 1 + 60/f³ + 0.0025·f            (Hoerner, fineness f = L/d)
      fin  FF = 1 + 2(t/c)                      (thin-airfoil form factor)
      Cd_f = Σ Cf_i · FF_i · S_wet_i / S_ref

    Fully-turbulent Cf is the right default at flight Reynolds numbers
    (≥1e6); it slightly over-predicts below that. Returns None when the
    geometry dict or Reynolds number is unusable.
    """
    if not geometry or reynolds <= 1e3 or ref_area <= 0:
        return None
    L = geometry.get("length", 0.0)
    r = geometry.get("body_radius", 0.0)
    if L <= 0 or r <= 0:
        return None
    nose_L = geometry.get("nose_length", 0.3 * L)
    body_L = max(0.0, geometry.get("body_length", L - nose_L))

    def cf_turb(re: float) -> float:
        re = max(re, 1e4)
        cf = 0.455 / (math.log10(re) ** 2.58)
        return cf * (1.0 + 0.144 * mach * mach) ** -0.65

    # ── Wetted area of the body ──────────────────────────────────────────────
    # Integrate the REAL meridian when the geometry carries one. The previous
    # form was a cone slant plus a cylinder:
    #
    #     s_nose = pi * r * sqrt(nose_L**2 + r**2)
    #     s_body = 2 * pi * r * body_L
    #
    # which is exact only for a straight cone on a constant-radius tube. An
    # ogive, a conical transition, a flare or a boattail has no representation
    # there at all, and every one of them changes the wetted area this drag is
    # proportional to. The mesher revolves the true profile
    # (_revolve_profile_solid) and this is now the same shape.
    #
    # Frustum lateral area between consecutive stations: pi*(r0+r1)*slant.
    prof = [(float(p[0]), float(p[1]))
            for p in (geometry.get("profile") or []) if len(p) >= 2]
    if len(prof) >= 2:
        s_body_total = sum(
            math.pi * (r0 + r1) * math.hypot(x1 - x0, r1 - r0)
            for (x0, r0), (x1, r1) in zip(prof, prof[1:])
        )
        r_max = max(rr for _, rr in prof)
    else:
        s_body_total = (math.pi * r * math.sqrt(nose_L * nose_L + r * r)
                        + 2.0 * math.pi * r * body_L)
        r_max = r
    # Hoerner's body form factor wants the body's own slenderness, so it takes
    # the widest station rather than the nominal tube radius.
    fineness = L / (2.0 * max(r_max, 1e-9))
    ff_body = 1.0 + 60.0 / fineness ** 3 + 0.0025 * fineness
    cd_f = cf_turb(reynolds) * ff_body * s_body_total / ref_area

    # Fins: both faces of each panel, Re over the mean chord.
    n_fin = int(geometry.get("fin_count", 0))
    cr = geometry.get("fin_root", 0.0)
    ct = geometry.get("fin_tip", cr * 0.5)
    h = geometry.get("fin_height", 0.0)
    if n_fin > 0 and cr > 0 and h > 0:
        mean_chord = 0.5 * (cr + ct)
        s_fins = 2.0 * n_fin * mean_chord * h
        t_over_c = geometry.get("fin_thick", 0.003) / max(mean_chord, 1e-6)
        ff_fin = 1.0 + 2.0 * t_over_c
        re_fin = reynolds * mean_chord / L
        cd_f += cf_turb(re_fin) * ff_fin * s_fins / ref_area

    return cd_f


def analytic_friction_cd_cad(
    cad_info: dict,
    reynolds: float,
    mach: float,
    ref_area: float,
) -> Optional[float]:
    """Flat-plate skin-friction Cd for an imported CAD body.

    Same Schlichting/compressibility build-up as :func:`analytic_friction_cd`,
    but driven by the measured wetted area instead of a rocket component
    breakdown — the only wetted-area information an arbitrary body carries.

    The form factor uses the body's own slenderness (flow-wise length over
    equivalent cross-stream diameter) via Hoerner's body-of-revolution
    expression, clamped for stubby shapes where it blows up. Coarser than the
    per-component rocket version, and it does not know a wing from a fuselage —
    treat it as an engineering estimate, not a validated number.
    """
    if not cad_info or reynolds <= 1e3 or ref_area <= 0:
        return None
    s_wet = float(cad_info.get("wetted_area", 0.0) or 0.0)
    L = float(cad_info.get("length", 0.0) or 0.0)
    if s_wet <= 0 or L <= 0:
        return None

    re = max(reynolds, 1e4)
    cf = 0.455 / (math.log10(re) ** 2.58)
    cf *= (1.0 + 0.144 * mach * mach) ** -0.65

    # Equivalent diameter from the measured frontal area (falls back to the
    # cross-stream bbox when the projection came out unusable).
    frontal = float(cad_info.get("frontal_area", 0.0) or 0.0)
    if frontal > 0:
        d_eq = 2.0 * math.sqrt(frontal / math.pi)
    else:
        d_eq = max(
            float(cad_info.get("cross_width", 0.0) or 0.0),
            float(cad_info.get("cross_height", 0.0) or 0.0),
        )
    if d_eq <= 0:
        return None

    # Hoerner body form factor. Below f≈2 the 60/f³ term dominates
    # unphysically, so the fineness used here is floored.
    fineness = max(L / d_eq, 2.0)
    ff = 1.0 + 60.0 / fineness ** 3 + 0.0025 * fineness
    return cf * ff * s_wet / ref_area


# ── Mesh staging ─────────────────────────────────────────────────────────────

def staged_mesh_matches(mesh_path: Path, work_dir: Path) -> bool:
    """True if ``work_dir`` already holds *this* mesh, not an older one."""
    local = Path(work_dir) / Path(mesh_path).name
    if not local.is_file() or not Path(mesh_path).is_file():
        return False
    src, dst = Path(mesh_path).stat(), local.stat()
    return src.st_size == dst.st_size and abs(src.st_mtime - dst.st_mtime) < 1.0


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
        # Only reuse it if it IS the current mesh. A leftover from an earlier
        # sweep in the same folder — a different refinement, say — would
        # otherwise be solved silently instead of the mesh just generated, and
        # the points of one sweep would not all be on the same grid.
        if staged_mesh_matches(mesh_path, work_dir):
            return local
        logger.info(f"Restaging stale mesh in {work_dir.name} "
                    f"({local.stat().st_size} B -> {mesh_path.stat().st_size} B)")
        local.unlink()
    try:
        os.link(mesh_path, local)          # hardlink — same NTFS volume
    except Exception:
        shutil.copy2(mesh_path, local)     # cross-volume / unsupported → copy
    return local


# ── Per-point solve ──────────────────────────────────────────────────────────

def sweep_point_dir(base_work_dir: Path, var: str, value: float) -> Path:
    """Work directory a given sweep point writes into."""
    tag = f"{var}_{value:+.3f}".replace("+", "p").replace("-", "m").replace(".", "_")
    return Path(base_work_dir) / "sweep" / tag


def point_is_complete(work_dir: Path, mesh_path: Optional[Path] = None) -> bool:
    """True if ``work_dir`` holds a SU2 solve that ran to completion.

    A killed run leaves a perfectly parseable ``history.csv`` that simply stops
    mid-transient, so the history alone cannot be trusted; SU2 prints its exit
    banner only after the final output files are written.

    Pass ``mesh_path`` to also require that the solve was run on that exact mesh
    — otherwise results from a superseded grid would look reusable. Two things
    are checked, because either alone can be fooled: the staged copy must be the
    current mesh, *and* the history must postdate it. The staged copy is normally
    a hardlink, so re-meshing to the same path updates the point's copy along with
    it and makes a stale point look current; only the history's own age shows that
    its numbers came from the older grid.
    """
    wd = Path(work_dir)
    log = wd / "su2_run.log"
    history = wd / "history.csv"
    if not history.is_file() or not log.is_file():
        return False
    if mesh_path is not None:
        if not staged_mesh_matches(mesh_path, wd):
            return False
        if history.stat().st_mtime < Path(mesh_path).stat().st_mtime - 1.0:
            return False
    try:
        return "Exit Success" in log.read_text(errors="replace")
    except OSError:
        return False


def run_sweep_point(
    base_config: CFDConfig,
    var: str,
    value: float,
    mesh_path: Path,
    progress_cb: Optional[Callable[[int, float], None]] = None,
    reuse_existing: bool = False,
) -> CFDResult:
    """Solve ONE sweep point on a pre-built mesh.

    Clones ``base_config``, overrides the swept variable, points the solver at
    the already-generated ``mesh_path`` (skipping mesh generation), writes a
    fresh SU2 case, runs it, and returns the parsed CFDResult.

    Each point gets its own work sub-directory so VTK/history files don't clash.

    With ``reuse_existing``, a point whose folder already holds a completed solve
    is re-parsed instead of re-solved. This makes a long sweep resumable after an
    interruption. The .cfg is still regenerated — that is what populates the flow
    metadata (Reynolds, q, reference area) the parse depends on — so a reused
    point is only trusted when its own solver log shows a clean exit.
    """
    if var not in SWEEP_VARS:
        raise ValueError(f"Unknown sweep variable '{var}'. Use one of {list(SWEEP_VARS)}.")

    cfg = copy.deepcopy(base_config)
    setattr(cfg, SWEEP_VARS[var], value)

    # Isolate each point's outputs in its own folder.
    cfg.work_dir = sweep_point_dir(base_config.work_dir, var, value)
    cfg.work_dir.mkdir(parents=True, exist_ok=True)

    # Decided before staging: staging is what would make a stale point's mesh
    # look current again.
    reused = reuse_existing and point_is_complete(cfg.work_dir, mesh_path)

    solver = SU2Solver(cfg)
    # Reuse the shared mesh — stage it into this point's folder, skip remesh.
    solver._mesh_path = stage_mesh(mesh_path, cfg.work_dir)
    if progress_cb is not None:
        solver.set_progress_callback(progress_cb)

    solver.generate_case()
    if reused:
        logger.info(f"Sweep point {var}={value:g} already solved in "
                    f"{cfg.work_dir.name} — re-parsing, not re-solving.")
    else:
        for _it, _rms in solver.run():
            pass  # progress already streamed via callback
    result = solver.parse_results()

    # Hybrid Euler polar: the friction build-up is applied inside
    # SU2Solver.parse_results now, so it reaches the single-run path as well as
    # the sweep. Applying it here too would double-count it.

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

        # Wave drag, solved per point (forebody share of the integrated
        # pressure drag) — not reconstructed off the drag-bucket baseline.
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
