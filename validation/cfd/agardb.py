"""
AGARD-B benchmarks against wind-tunnel measurement.
===================================================

Two comparisons share one geometry and one reference dataset
(:mod:`validation.data.agardb_aedc`, from AEDC-TR-70-100):

``barrowman_vs_experiment``
    K2's analytic :class:`physics.aerodynamics.AeroModel` lift-curve slope vs
    the measured one. No solver, runs in milliseconds. This is a *validity
    envelope* probe, not a pass/fail gate — see the note in
    :func:`barrowman_cl_alpha_per_deg`.

``su2_vs_experiment``
    The full K2 CFD pipeline (geometry → STL → gmsh → SU2) swept over Mach and
    angle of attack, with the lift-curve slope at each Mach compared against the
    measurement. This is the real test of the CFD stack against physical data.

Coefficient bookkeeping — the one thing that will silently ruin this comparison
-------------------------------------------------------------------------------
AEDC normalises by the **wing planform area** S = 4*sqrt(3)*D^2 = 6.93 D^2.
K2 and SU2 normalise by the **body frontal area** pi*D^2/4 = 0.785 D^2. The two
differ by a factor of 8.8, so every K2-side coefficient here is rescaled by
``area_k2 / S_ref`` before it is compared.
"""
from __future__ import annotations

import math
from pathlib import Path

from validation.data import agardb_aedc as REF
from validation.cfd.agardb_geometry import (
    agardb_assembly, reference_quantities, DEFAULT_DIAMETER_M,
)

_WORK = Path("cfd_run") / "validation" / "agardb"


# ── K2 analytic (Barrowman) ───────────────────────────────────────────────────

def _find(assembly, cls):
    """First component of type *cls* anywhere in the assembly tree."""
    stack = list(getattr(assembly, "stages", []) or [])
    while stack:
        node = stack.pop(0)
        if isinstance(node, cls):
            return node
        stack.extend(getattr(node, "children", []) or [])
    raise LookupError(f"no {cls.__name__} in the AGARD-B assembly")


def _agardb_aero_model(diameter_m: float = DEFAULT_DIAMETER_M):
    """(AeroModel, reference_quantities) for the AGARD-B model.

    Getting the geometry in is the fiddly part, and getting it wrong is silent.
    ``AeroModel.from_state`` reads FLAT fields off the state object; it never
    looks at ``state.assembly``. Setting only the assembly (as this module used
    to) leaves every fin field at zero, and ``from_state``'s ``or`` fallbacks
    then invent a generic 4-fin rocket — 0.6 D span, unswept, 0.68 D root chord
    — so the comparison silently measures a fabricated vehicle. Copy the panel
    out explicitly, and mind that the component stores sweep in DEGREES while
    the state field is RADIANS.
    """
    from physics.aerodynamics import AeroModel
    from core.rocket_state import RocketState
    from core.components import NoseCone, TrapezoidalFinSet

    asm = agardb_assembly(diameter_m)
    ref = reference_quantities(diameter_m)

    state = RocketState(
        name="AGARD-B", length=ref["length_m"], diameter=diameter_m,
        dry_mass=1.0, propellant_mass=0.0, propellant_mass_initial=0.0,
    )
    state.assembly = asm

    nose = _find(asm, NoseCone)
    fins = _find(asm, TrapezoidalFinSet)
    state.nose_type = "ogive"
    state.nose_length = nose.component_length()
    state.fin_count = fins.fin_count
    state.fin_root_chord = fins.root_chord
    state.fin_tip_chord = fins.tip_chord
    state.fin_span = state.fin_height = fins.height
    state.fin_sweep_angle = math.radians(fins.sweep_angle)   # deg -> rad
    state.fin_thickness = fins.thickness
    state.fin_position = fins.position
    state.fin_cross_section = fins.cross_section
    return AeroModel.from_state(state), ref


def barrowman_cd0(diameter_m: float = DEFAULT_DIAMETER_M,
                  mach: float = 0.5) -> float:
    """K2's zero-lift drag for AGARD-B, wing-planform-area referenced.

    The AEDC report's C_D0 column is the weaker of its two aggregates (see
    :mod:`validation.data.agardb_aedc`) and it includes a sting-dependent base
    term, so this is a 10%-class comparison, not a 2% one. It is still the only
    *measured* drag K2 can be held against, and it is what caught the base-drag
    model: with OpenRocket's Mach-only ``0.12 + 0.13 M^2`` this ran +22.6% at
    M=0.2 rising to +48.4% at M=0.9.
    """
    aero, ref = _agardb_aero_model(diameter_m)
    from environment.atmosphere_model import Atmosphere

    atm = Atmosphere()
    v = mach * atm.speed_of_sound(0.0)
    q = 0.5 * atm.density(0.0) * v * v
    cd_body_ref = aero.compute(alpha=1.0e-6, mach=mach, q_dyn=q, pitch_rate=0.0,
                               v_rel=v, cg=0.6 * ref["length_m"])["cd"]
    return cd_body_ref * ref["body_area_m2"] / ref["s_ref_m2"]


def barrowman_cl_alpha_per_deg(diameter_m: float = DEFAULT_DIAMETER_M,
                               mach: float = 0.2) -> float:
    """K2's Barrowman lift-curve slope for AGARD-B, per degree, wing-area based.

    AGARD-B's wing spans 4 body diameters, well outside the small-fin
    assumption Barrowman's fin term is derived under, so this used to be read
    as an envelope probe that could only be low. It is now within 5% of the
    measurement at every Mach in the dataset — see
    :func:`validation.cfd.benchmarks.bench_agardb_barrowman` for what the two
    intervening bugs were.
    """
    aero, ref = _agardb_aero_model(diameter_m)

    # Finite-difference the normal-force coefficient about alpha = 0.
    da = math.radians(0.5)
    q, v = 1.0e4, 100.0
    cg = 0.6 * ref["length_m"]
    cn_p = aero.compute(alpha=da, mach=mach, q_dyn=q, pitch_rate=0.0,
                        v_rel=v, cg=cg)["cn"]
    cn_m = aero.compute(alpha=-da, mach=mach, q_dyn=q, pitch_rate=0.0,
                        v_rel=v, cg=cg)["cn"]
    cn_alpha_per_rad = (cn_p - cn_m) / (2.0 * da)      # body-area referenced

    # -> wing-area referenced, per degree
    scale = ref["body_area_m2"] / ref["s_ref_m2"]
    return cn_alpha_per_rad * scale * math.pi / 180.0


# ── K2 CFD (SU2) ──────────────────────────────────────────────────────────────

def _base_config(stl, geom, diameter_m, refinement, max_iterations):
    from cfd.solvers.base import CFDConfig

    ref = reference_quantities(diameter_m)
    return CFDConfig(
        mach=0.6, angle_of_attack_deg=0.0, altitude_m=0.0,
        mesh_refinement=refinement, work_dir=_WORK,
        geometry_stl=stl, geometry_dict=geom,
        cg_from_nose_m=ref["length_m"] * 0.6,
        # Hybrid polar mode: inviscid solve + analytic flat-plate friction.
        # The tet-only mesh cannot resolve a wall layer, so a RANS run here would
        # report a fictitious viscous lift contribution on top of a wall-shear
        # field that is not converged — see the CFD notes on wall functions.
        turbulence_model="Euler", euler_analytic_friction=True,
        max_iterations=max_iterations,
    )


def mesh_key(cfg) -> str:
    """Fingerprint of everything the mesh depends on.

    Mach and alpha are deliberately absent — they are solver inputs, not mesh
    inputs, which is why one mesh serves the whole sweep in the first place.

    ``cfd.meshing.MESHER_REVISION`` is in the key because the mesher itself is an
    input. Without it a rewrite of the size fields left the cached mesh in place
    and the resumed sweep reproduced the previous lift slopes to six digits.
    """
    import hashlib
    import json

    from cfd.meshing import MESHER_REVISION

    payload = json.dumps({
        "mesher": MESHER_REVISION,
        "geometry": cfg.geometry_dict,
        "refinement": cfg.mesh_refinement,
        "domain": [cfg.domain_length_scale, cfg.domain_radius_scale],
        "bl": [cfg.boundary_layer_layers, cfg.boundary_layer_growth],
        "wall_size": cfg.custom_wall_size,
        "target_elements": cfg.target_element_count,
    }, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


def _shared_mesh(cfg, reuse: bool = True) -> Path:
    """The one mesh the sweep runs on — generated, or reused from disk.

    Reuse is keyed on :func:`mesh_key` written beside the mesh, not on file
    timestamps: the STL is re-exported with a fresh mtime on every call even when
    its bytes are identical, so mtimes would force a needless re-mesh (and a
    re-mesh invalidates every already-solved point, since the points are matched
    to the mesh file they were solved on).
    """
    from cfd.solvers.su2_solver import SU2Solver

    mesh = Path(cfg.work_dir) / "rocket_mesh.su2"
    stamp = Path(cfg.work_dir) / "mesh_key.txt"
    key = mesh_key(cfg)

    if reuse and mesh.is_file() and stamp.is_file():
        if stamp.read_text(encoding="utf-8").strip() == key:
            return mesh

    path = SU2Solver(cfg).generate_mesh()
    stamp.write_text(key, encoding="utf-8")
    return path


def run_su2_sweep(machs=(0.5, 0.6, 0.7, 0.8), alphas=(0.0, 2.0, 4.0, 6.0),
                  diameter_m: float = DEFAULT_DIAMETER_M,
                  refinement: str = "medium", max_iterations: int = 3000,
                  progress=None, resume: bool = True) -> dict:
    """Solve the (Mach x alpha) grid on ONE shared mesh.

    Returns ``{mach: [(alpha_deg, cl_wing_ref, cd_wing_ref, converged), ...]}``.
    The mesh depends only on the geometry, so it is generated once and staged
    into each point's directory by :func:`cfd.sweep.run_sweep_point`.

    16 points at a few minutes each is over an hour of wall clock, so ``resume``
    (default on) re-parses points already solved in ``_WORK`` instead of solving
    them again, and keeps the mesh already on disk. Both are keyed on the mesh
    inputs (see :func:`_shared_mesh`) and on per-point mesh identity, so a
    resumed sweep cannot mix grids: change the geometry or the refinement and
    the mesh plus every point is rebuilt.
    """
    from cfd.geometry_exporter import extract_cfd_geometry, export_assembly_to_stl
    from cfd.sweep import run_sweep_point

    _WORK.mkdir(parents=True, exist_ok=True)
    asm = agardb_assembly(diameter_m)
    geom = extract_cfd_geometry(asm)
    stl = export_assembly_to_stl(asm, _WORK / "agardb.stl")

    cfg = _base_config(stl, geom, diameter_m, refinement, max_iterations)
    mesh_path = _shared_mesh(cfg, reuse=resume)

    s_ref = reference_quantities(diameter_m)["s_ref_m2"]
    out: dict = {}
    for mach in machs:
        cfg_m = _base_config(stl, geom, diameter_m, refinement, max_iterations)
        cfg_m.mach = mach
        cfg_m.work_dir = _WORK / f"M{mach:.2f}".replace(".", "_")
        rows = []
        for alpha in alphas:
            if progress:
                progress(f"AGARD-B SU2: M={mach:.2f} alpha={alpha:.1f}deg")
            res = run_sweep_point(cfg_m, "aoa", alpha, mesh_path,
                                  reuse_existing=resume)
            # SU2 normalises by the body frontal area it derived from the STL;
            # rescale onto the wing area the measurements use.
            scale = (res.reference_area_m2 or 0.0) / s_ref
            rows.append((alpha, res.cl * scale, res.cd * scale, bool(res.converged)))
        out[mach] = rows
    return out


def slope_per_deg(rows) -> float:
    """Least-squares dC_L/dalpha [1/deg] from [(alpha, cl, ...), ...]."""
    pts = [(a, cl) for a, cl, *_ in rows]
    n = len(pts)
    if n < 2:
        return float("nan")
    sa = sum(a for a, _ in pts)
    sc = sum(c for _, c in pts)
    saa = sum(a * a for a, _ in pts)
    sac = sum(a * c for a, c in pts)
    den = n * saa - sa * sa
    if abs(den) < 1e-12:
        return float("nan")
    return (n * sac - sa * sc) / den
