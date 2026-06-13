"""
Structural benchmarks.
======================

Published / solver-proof (direct CalculiX, textbook-exact) — proves the bundled
``ccx.exe``, units, material card and BC handling:

    * uniaxial bar tension   → δ=FL/EA, σ=F/A
    * cantilever bending      → δ_tip=FL³/3EI

Closed-form fast-math gates (no solver, always-on) — proves K2's analytic
formula layer in :mod:`physics.structures`:

    * thin-wall hoop stress  → σ_h = p·r/t
    * Euler column buckling   → P_cr = π²EI/L²

Analytic-vs-solver (K2 pipeline) — proves K2's airframe FEM agrees with K2's
closed form on one object:

    * 1st bending frequency: workstation.modal_estimate (Euler–Bernoulli
      cantilever) vs CalculiX modal_analysis (3-D tube mesh, cantilever BC).
"""
from __future__ import annotations

import math
import statistics
from pathlib import Path

from core.validation import ValidationLevel
from validation.harness import Benchmark, Comparison


_WORK = Path("fem_run") / "validation"


# ── direct-CCX textbook cases ─────────────────────────────────────────────────

def bench_bar_tension() -> Benchmark:
    from validation.structures import ccx_direct as C
    L, b, h, F = 0.5, 0.02, 0.02, 1000.0
    wd = _WORK / "bar"; wd.mkdir(parents=True, exist_ok=True)
    mesh = C.make_beam_mesh(L, b, h, nx=10, ny=2, nz=2)
    C._write_deck(wd / "case.inp", mesh, "C3D8", mesh.nset_xL, 1,
                  F / len(mesh.nset_xL))
    dat = C.run_ccx(wd)
    disp = C.parse_displacements(dat)
    A = b * h
    delta_fe = statistics.mean(disp[n][0] for n in mesh.nset_xL)
    sigma_fe = C.parse_mean_stress(dat, component=1)

    bm = Benchmark(name="Uniaxial bar (CalculiX)", domain="structures",
                   reference="Exact δ=FL/EA, σ=F/A",
                   level=ValidationLevel.VALIDATED)
    bm.add(Comparison.make("Tip elongation δ", delta_fe, F * L / (A * C.E_AL),
                           "FL/EA", "m", tol_rel=0.03))
    bm.add(Comparison.make("Axial stress σ (mean)", sigma_fe, F / A,
                           "F/A", "Pa", tol_rel=0.02))
    return bm


def bench_cantilever_bending() -> Benchmark:
    from validation.structures import ccx_direct as C
    L, b, h, F = 0.5, 0.02, 0.02, 50.0
    wd = _WORK / "cantilever"; wd.mkdir(parents=True, exist_ok=True)
    mesh = C.make_beam_mesh(L, b, h, nx=24, ny=2, nz=4)
    # C3D8I (incompatible modes) avoids shear locking in bending.
    C._write_deck(wd / "case.inp", mesh, "C3D8I", mesh.nset_xL, 3,
                  F / len(mesh.nset_xL))
    dat = C.run_ccx(wd)
    disp = C.parse_displacements(dat)
    uz_tip = statistics.mean(disp[n][2] for n in mesh.nset_xL)
    I = b * h ** 3 / 12.0
    delta_exact = F * L ** 3 / (3 * C.E_AL * I)

    bm = Benchmark(name="Cantilever bending (CalculiX)", domain="structures",
                   reference="Euler–Bernoulli δ=FL³/3EI",
                   level=ValidationLevel.VALIDATED)
    # FE vs slender-beam theory: a few % (finite aspect ratio, shear).
    bm.add(Comparison.make("Tip deflection δ", uz_tip, delta_exact,
                           "FL³/3EI", "m", tol_rel=0.05))
    return bm


# ── closed-form fast-math gates (no solver) ───────────────────────────────────

def bench_closed_form_formulas() -> Benchmark:
    """K2 physics.structures closed-form helpers vs textbook formulas."""
    from physics import structures as S

    bm = Benchmark(name="Closed-form structural formulas", domain="structures",
                   reference="Textbook thin-wall / Euler relations",
                   level=ValidationLevel.VALIDATED)

    # Thin-wall hoop stress σ_h = p·r/t (mean radius).
    p, d, t = 2.0e6, 0.10, 0.0025
    r = d / 2
    bm.add(Comparison.make("Hoop stress σ_h",
                           S.hoop_stress(p, d, t), p * r / t,
                           "p·r/t", "Pa", tol_rel=0.02))

    # Euler buckling of a tube column: P_cr = π²EI/L². Match K2's wall
    # convention (r_i = r_o − t, with r_o = d/2) so this is an exact identity.
    E, L = 68.9e9, 1.5
    r_o, r_i = d / 2, d / 2 - t
    I = (math.pi / 4) * (r_o ** 4 - r_i ** 4)
    bm.add(Comparison.make("Euler buckling P_cr",
                           S.euler_buckling(E, d, t, L),
                           math.pi ** 2 * E * I / L ** 2,
                           "π²EI/L²", "N", tol_rel=0.05))

    # von Mises of a pure axial stress reduces to the axial stress itself.
    bm.add(Comparison.make("von Mises (uniaxial)",
                           S.von_mises(150e6), 150e6,
                           "σ", "Pa", tol_rel=1e-6))
    return bm


# ── K2 pipeline vs CalculiX (modal) ───────────────────────────────────────────

def _uniform_tube_modal_inputs():
    """A uniform aluminium tube used identically on the beam and FE sides.

    Removing the nose cone and propellant makes mass *and* stiffness uniform, so
    the only remaining difference is beam theory vs 3-D shell FE — exactly what
    this benchmark is meant to expose. The state's dry_mass is set to the tube's
    analytic mass so ``modal_estimate`` and the FE mesh carry the same mass.
    """
    from core.components import RocketAssembly, BodyTube
    from core.rocket_state import RocketState

    OD, t, L = 0.102, 0.0025, 2.0
    r_o, r_i = OD / 2, OD / 2 - t
    mass = 2700.0 * math.pi * (r_o ** 2 - r_i ** 2) * L

    state = RocketState(
        name="modal-tube", length=L, diameter=OD, wall_thickness=t,
        dry_mass=mass, propellant_mass=0.0, propellant_mass_initial=0.0,
        material_name="Aluminum 6061-T6", elastic_modulus=68.9e9,
    )

    asm = RocketAssembly()
    tube = BodyTube()
    tube.length = L
    tube.outer_diameter_val = OD
    tube.inner_diameter = OD - 2 * t
    tube.material = "Aluminum 6061-T6"
    asm.add_component(asm.stages[0], tube)
    return state, asm


def bench_modal_vs_ccx() -> Benchmark:
    """K2 closed-form 1st bending frequency vs CalculiX modal on the airframe.

    Both use a cantilever (clamped-aft) boundary condition and an identical
    uniform aluminium tube with matched mass, so they describe the same
    idealisation; the 3-D shell mesh still differs from the Euler–Bernoulli beam
    (shell ovalisation lowers FE frequency), hence a moderate tolerance and an
    ESTIMATED level.
    """
    from structures.workstation import modal_estimate
    from structures.fem_interface import FEMInterface

    bm = Benchmark(name="1st bending mode: closed-form vs CalculiX",
                   domain="structures", reference="CalculiX modal (cantilever)",
                   level=ValidationLevel.ESTIMATED)

    state, asm = _uniform_tube_modal_inputs()
    est = modal_estimate(state, "Aluminum 6061-T6")
    fem = FEMInterface(work_dir=_WORK / "modal")
    modal = fem.modal_analysis(asm, material_name="Aluminum 6061-T6",
                               num_modes=6, refinement="coarse")
    freqs = [f for f in (modal.frequencies_hz or []) if f > 1.0]
    if not freqs:
        bm.skipped = True
        bm.skip_reason = "CalculiX modal returned no non-rigid frequencies"
        return bm

    bm.add(Comparison.make("f1 (1st lateral bending)", est.f1_hz, freqs[0],
                           "CalculiX mode 1", "Hz", tol_rel=0.25,
                           note="beam Euler–Bernoulli vs 3-D shell FE"))
    return bm


def run_benchmarks(include_ccx: bool = True) -> list:
    """All structures benchmarks. `include_ccx` runs the slow CalculiX cases."""
    out = [bench_closed_form_formulas()]
    if include_ccx:
        out.append(bench_bar_tension())
        out.append(bench_cantilever_bending())
        out.append(bench_modal_vs_ccx())
    return out
