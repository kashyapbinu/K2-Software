"""
Taylor–Maccoll exact supersonic cone-flow solver.
==================================================

The flow over a sharp cone at zero incidence in supersonic flow has an *exact*
solution (Taylor & Maccoll, 1933): a conical flowfield bounded by an attached
straight shock. This module solves it numerically and is the analytic reference
the SU2 cone case is benchmarked against — the CFD analogue of the textbook
cantilever beam on the structures side.

Method (Anderson, *Modern Compressible Flow*, ch. 10):

1. Normalise velocity by V_max so V' ∈ [0, 1], with a'² = (γ−1)/2·(1−V'²).
2. For a trial shock angle β, the oblique-shock relations give the post-shock
   Mach and the velocity components on the ray θ=β.
3. Integrate the Taylor–Maccoll ODE inward in θ until the meridional component
   V_θ = 0 — that θ is the cone half-angle produced by this β.
4. Root-find β so the produced cone angle equals the requested θc.
5. The surface state (M_c) then gives the surface pressure coefficient via the
   shock static-pressure jump followed by isentropic compression to the cone.

Pure NumPy/SciPy — no solver binary, fully deterministic.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

GAMMA = 1.4


# ── helpers ───────────────────────────────────────────────────────────────────

def _v_prime(mach: float, gamma: float = GAMMA) -> float:
    """Non-dimensional velocity V/V_max for a given Mach number."""
    m2 = mach * mach
    return math.sqrt(m2 / (m2 + 2.0 / (gamma - 1.0)))


def _mach_from_vprime(vp: float, gamma: float = GAMMA) -> float:
    a2 = (gamma - 1.0) / 2.0 * (1.0 - vp * vp)   # a'^2
    return math.sqrt(vp * vp / a2) if a2 > 0 else float("inf")


def oblique_shock_deflection(M1: float, beta: float, gamma: float = GAMMA) -> float:
    """Flow deflection δ (rad) for shock angle β (rad) and upstream Mach M1."""
    mn1 = M1 * math.sin(beta)
    if mn1 <= 1.0:
        return 0.0
    num = 2.0 / math.tan(beta) * (mn1 ** 2 - 1.0)
    den = M1 ** 2 * (gamma + math.cos(2.0 * beta)) + 2.0
    return math.atan(num / den)


def _post_shock_mach(M1: float, beta: float, gamma: float = GAMMA) -> float:
    mn1 = M1 * math.sin(beta)
    mn2_sq = (1.0 + (gamma - 1.0) / 2.0 * mn1 ** 2) / (gamma * mn1 ** 2 - (gamma - 1.0) / 2.0)
    delta = oblique_shock_deflection(M1, beta, gamma)
    mn2 = math.sqrt(mn2_sq)
    return mn2 / math.sin(beta - delta)


# ── Taylor–Maccoll ODE ─────────────────────────────────────────────────────────

def _tm_rhs(theta, y, gamma=GAMMA):
    """dy/dθ for y=[V_r', V_θ'] (velocities normalised by V_max)."""
    vr, vth = y
    a2 = (gamma - 1.0) / 2.0 * (1.0 - vr * vr - vth * vth)   # a'^2
    denom = a2 - vth * vth
    if abs(denom) < 1e-12:
        denom = math.copysign(1e-12, denom)
    dvth = (vr * vth * vth - a2 * (2.0 * vr + vth / math.tan(theta))) / denom
    return [vth, dvth]


def _cone_angle_for_beta(M1: float, beta: float, gamma: float = GAMMA):
    """Integrate inward from the shock; return (theta_cone, surface_vr')."""
    delta = oblique_shock_deflection(M1, beta, gamma)
    M2 = _post_shock_mach(M1, beta, gamma)
    v2 = _v_prime(M2, gamma)
    # Components on the ray θ=β: radial along the ray, θ-component normal to it.
    vr0 = v2 * math.cos(beta - delta)
    vth0 = -v2 * math.sin(beta - delta)

    # V_θ = 0 marks the cone surface (flow tangent to the cone).
    def surface_event(theta, y, *a):
        return y[1]
    surface_event.terminal = True
    surface_event.direction = 0

    sol = solve_ivp(_tm_rhs, [beta, 1e-4], [vr0, vth0],
                    events=surface_event, rtol=1e-9, atol=1e-12,
                    max_step=math.radians(0.25))
    if sol.t_events[0].size == 0:
        return None, None
    theta_c = float(sol.t_events[0][0])
    vr_c = float(sol.y_events[0][0][0])
    return theta_c, vr_c


@dataclass
class ConeSolution:
    mach_inf: float
    cone_half_angle_deg: float
    shock_angle_deg: float
    surface_mach: float
    cp_surface: float          # surface pressure coefficient
    p_ratio_surface: float     # p_cone / p_inf


def solve_cone(M1: float, cone_half_angle_deg: float,
               gamma: float = GAMMA) -> ConeSolution:
    """Exact Taylor–Maccoll solution for a sharp cone at zero incidence."""
    theta_c_target = math.radians(cone_half_angle_deg)
    mu = math.asin(1.0 / M1)                       # Mach angle (weakest shock)

    def residual(beta):
        tc, _ = _cone_angle_for_beta(M1, beta, gamma)
        if tc is None:
            return 1.0
        return tc - theta_c_target

    # Bracket β between just above the Mach angle and near 90°. The cone angle
    # rises from 0 at β=μ to a max near detachment, so the attached (weak)
    # solution is the smaller root.
    lo, hi = mu + math.radians(0.05), math.radians(89.0)
    # Scan for a sign change from lo upward.
    betas = np.linspace(lo, hi, 200)
    prev_b, prev_r = lo, residual(lo)
    beta_sol = None
    for b in betas[1:]:
        r = residual(b)
        if prev_r == 1.0 or r == 1.0:
            prev_b, prev_r = b, r
            continue
        if prev_r * r < 0:
            beta_sol = brentq(residual, prev_b, b, xtol=1e-8)
            break
        prev_b, prev_r = b, r
    if beta_sol is None:
        raise RuntimeError(
            f"No attached cone solution for M={M1}, θc={cone_half_angle_deg}° "
            "(shock likely detached).")

    theta_c, vr_c = _cone_angle_for_beta(M1, beta_sol, gamma)
    Mc = _mach_from_vprime(vr_c, gamma)            # V_θ=0 → V'=V_r' at surface

    # Pressure: static jump across the shock × isentropic compression to cone.
    mn1 = M1 * math.sin(beta_sol)
    p2_p1 = 1.0 + 2.0 * gamma / (gamma + 1.0) * (mn1 ** 2 - 1.0)
    M2 = _post_shock_mach(M1, beta_sol, gamma)
    # Isentropic (total pressure constant behind the shock) from M2 to Mc.
    g = gamma
    pc_p2 = ((1.0 + (g - 1.0) / 2.0 * M2 ** 2) /
             (1.0 + (g - 1.0) / 2.0 * Mc ** 2)) ** (g / (g - 1.0))
    pc_p1 = pc_p2 * p2_p1
    cp = (pc_p1 - 1.0) / (0.5 * gamma * M1 ** 2)

    return ConeSolution(
        mach_inf=M1, cone_half_angle_deg=cone_half_angle_deg,
        shock_angle_deg=math.degrees(beta_sol), surface_mach=Mc,
        cp_surface=cp, p_ratio_surface=pc_p1,
    )
