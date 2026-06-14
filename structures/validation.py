"""
K2 AeroSim — Structural Physics Consistency Checks
=====================================================
Audits a WorkstationReport for results that violate basic structural
mechanics or fall outside the range expected for a 1–3 m amateur/research
rocket, and returns human-readable warnings.

Two classes of check
--------------------
1. Internal consistency — combinations that are physically impossible
   (e.g. stress above yield with zero displacement, negative margin while
   SF > 1, buckling loads far above the material's own strength).
2. Range plausibility — values outside the bands known for aluminium
   airframes (Mode-1 20–80 Hz, tip deflection 0.5–20 mm, von Mises
   20–250 MPa, SF 1.2–3.0, wall temperature 300–700 K).

Each warning is a dataclass with a severity so the UI can colour it.
"""
from __future__ import annotations

from dataclasses import dataclass

from structures.solvers.base import get_structural_material


@dataclass
class Warning_:
    severity: str   # "error" | "warn" | "info"
    message: str


_COLOR = {"error": "#f85149", "warn": "#d29922", "info": "#58a6ff"}


def severity_color(sev: str) -> str:
    return _COLOR.get(sev, "#8b949e")


def validate_report(state, rep) -> list:
    """Return a list of Warning_ for any inconsistent or out-of-range result."""
    w = []
    mat = get_structural_material(getattr(state, "material_name", "Aluminum 6061-T6"))
    yield_pa = mat.yield_strength
    vm = rep.body_condition.get("von_mises", 0.0)
    sf = rep.body_condition.get("safety_factor", 0.0)
    mos = rep.body_condition.get("margin_of_safety", 0.0)
    defl = rep.deflection.max_deflection_mm
    L = getattr(state, "length", 0.0)
    is_metal_small = (L > 0 and L <= 3.0 and "alum" in mat.name.lower())

    # ── 1. Internal consistency ──────────────────────────────────────────────
    # 1a. Stress above yield but ~zero displacement
    if vm > yield_pa and defl < 0.05:
        w.append(Warning_("error",
            f"σ_vm {vm/1e6:.0f} MPa exceeds yield ({yield_pa/1e6:.0f} MPa) yet "
            f"deflection is ~0 mm — inconsistent stress/displacement."))

    # 1b. High stress with tiny applied loads
    F = max(rep.flight.max_thrust, rep.body_condition.get("axial", 0.0) *
            (3.14159 * state.diameter * state.wall_thickness)) if state.diameter else 0.0
    if vm > 0.5 * yield_pa and rep.flight.max_dynamic_pressure < 5000 and \
            rep.flight.max_thrust < 200:
        w.append(Warning_("warn",
            f"High stress ({vm/1e6:.0f} MPa) from very small loads — check load "
            f"case; expected low stress at q<5 kPa, thrust<200 N."))

    # 1c. Negative margin while SF > 1
    if mos < 0 and sf > 1.0:
        w.append(Warning_("error",
            f"Margin of safety is negative ({mos:+.2f}) while SF = {sf:.2f} > 1 — "
            f"margin convention error (MoS should be SF−1)."))

    # 1d. Buckling critical load far above material strength (sanity)
    cap = yield_pa * 3.14159 * state.diameter * state.wall_thickness if state.diameter else 0.0
    euler = next((m.critical for m in rep.buckling.modes if m.name == "Euler Column"), 0.0)
    if cap > 0 and euler > 100 * cap:
        w.append(Warning_("warn",
            f"Euler buckling load ({euler/1000:.0f} kN) is >100× the squash load "
            f"({cap/1000:.0f} kN) — column buckling not credible; shell mode governs."))

    # 1e. Temperature above 700 K at low Mach
    if rep.thermal.skin_temp_K > 700 and rep.flight.max_mach < 1.5:
        w.append(Warning_("error",
            f"Skin temperature {rep.thermal.skin_temp_K:.0f} K at Mach "
            f"{rep.flight.max_mach:.2f} is impossible — aero heating ∝ M²."))

    # ── 2. Range plausibility (1–3 m aluminium) ──────────────────────────────
    if is_metal_small:
        f1 = rep.modal.f1_hz
        if f1 > 0 and not (20 <= f1 <= 80):
            sev = "warn" if 10 <= f1 <= 120 else "error"
            w.append(Warning_(sev,
                f"Mode-1 frequency {f1:.0f} Hz outside the 20–80 Hz band typical "
                f"for a {L:.1f} m aluminium airframe."))
        if defl > 0 and not (0.5 <= defl <= 20):
            w.append(Warning_("warn",
                f"Tip deflection {defl:.2f} mm outside the 0.5–20 mm band for a "
                f"stiff metal airframe."))
        if vm > 0 and not (20e6 <= vm <= 250e6):
            sev = "info" if vm < 20e6 else "warn"
            note = "very lightly loaded / overbuilt" if vm < 20e6 else "approaching limits"
            w.append(Warning_(sev,
                f"Von Mises {vm/1e6:.0f} MPa outside 20–250 MPa band ({note})."))
        if sf > 0 and not (1.2 <= sf <= 3.0):
            if sf < 1.2:
                w.append(Warning_("error",
                    f"Safety factor {sf:.2f} below 1.2 — insufficient structural margin."))
            else:
                w.append(Warning_("info",
                    f"Safety factor {sf:.1f} above 3.0 — structure is overbuilt."))
        Tk = rep.thermal.skin_temp_K
        if Tk > 0 and not (300 <= Tk <= 700):
            if Tk > 700:
                w.append(Warning_("warn",
                    f"Wall temperature {Tk:.0f} K above 700 K — exceeds aluminium service range."))

    if not w:
        w.append(Warning_("info", "All results physically consistent and within "
                                   "expected ranges for the analysed load cases."))
    return w
