"""
K2 AeroSim — Shared 6DOF Aerodynamic Frame
==============================================
Body-frame resolution of the relative wind, shared by every trajectory
integrator so the flight physics cannot drift apart between them.

Callers
-------
- ``core.simulation_engine.SimulationEngine._derivatives``  (interactive/UI)
- ``core.batch_simulation.run_batch_simulation._derivatives`` (Monte Carlo,
  DOE, sensitivity, trade study, optimizer)

Why this module exists
----------------------
Both integrators used to resolve the relative wind like this::

    vel_angle = atan2(vrel_z, sqrt(vrel_x**2 + vrel_y**2))
    alpha     = pitch - vel_angle
    ...
    perp      = vel_angle + pi/2
    normal_x  = F_normal * cos(perp) * copysign(1, alpha)
    normal_z  = F_normal * sin(perp) * copysign(1, alpha)

``sqrt(vrel_x² + vrel_y²)`` collapses the horizontal relative wind to a
magnitude, discarding its bearing. Two consequences, both measured:

1. Any horizontal flow — whatever its compass bearing — read as a flow angle
   in the global X–Z plane, so a pure crosswind raised a spurious angle of
   attack and put the pitch normal force on the global X axis. The crosswind
   was then counted twice: once correctly as sideslip, once spuriously as AoA
   on the wrong axis. Pitch angular acceleration came out *identical* for wind
   from 0°, 90° and 180°, and the horizontal response varied 51% with bearing
   alone. End to end that moved apogee 4.1% and landing range 39.4% purely
   with the wind's compass bearing — both of which must be exactly 0% for an
   axisymmetric vehicle, since rotating the wind about the vertical axis is
   just a rotation of the whole problem.

2. Dropping the sign of ``vrel_x`` meant the X-axis normal force could never
   change sign, so wind from the south weathercocked the rocket south.

The replacement resolves the relative wind onto the actual body axes, so the
answer rotates with the wind instead of leaking into a fixed global plane.

Conventions (shared with both integrators)
------------------------------------------
- Euler angles: ``pitch`` is elevation of the body axis above horizontal,
  ``yaw`` is azimuth about +Z. Roll does not enter axisymmetric aero.
- Body axis (nose direction)::

      b = (cos(pitch)·cos(yaw), cos(pitch)·sin(yaw), sin(pitch))

  which is exactly the direction both integrators already thrust along.
- ``alpha`` > 0 means the nose is pitched above the relative velocity, and
  ``beta`` carries the same sign convention the engine's yaw channel uses
  (``yaw - flow_bearing``). Both match the previous definitions in the
  nominal in-plane case, so downstream moment code is unchanged.
"""
from __future__ import annotations

import math
from typing import NamedTuple

__all__ = ["AeroFrame", "resolve_aero_frame", "yaw_euler_rate", "wrap_angle"]


# Floor on |cos(pitch)| in the yaw kinematic relation. The relation is exact;
# this only bounds the integrand at the pole. 1e-6 measures isotropic to 0.1%
# and is stable from dt = 1e-2 down to 5e-4.
_YAW_KIN_EPS = 1e-6


def yaw_euler_rate(yaw_body_rate: float, pitch: float) -> float:
    """Euler yaw rate dψ/dt from the body angular rate about the yaw axis.

        dψ/dt = r / cos(θ)

    The moment equation ``yaw_accel = M_yaw / I`` is written about the body
    lateral axis ê_ψ, so the state's ``yaw_rate`` is the BODY rate r, not dψ/dt.
    Both integrators used to return r directly as dψ/dt, dropping the 1/cos(θ)
    metric factor. Since the nose moves at dψ/dt·cos(θ), that made the yaw
    weathercock channel fade out exactly where rockets fly — near vertical,
    where cos(θ) → 0 — while the pitch channel was unaffected.

    Measured on a vertical launch in an 8 m/s wind: the tilt angle off
    vertical, a scalar that cannot depend on the wind's compass bearing,
    varied 3.4x with bearing (6.9° to 23.7°) and horizontal drift 4.5x
    (24 m to 108 m). With the factor restored both are isotropic to 0.2%.

    ψ is only ever used through cos ψ / sin ψ, so the large rates this
    produces within a hair of the pole are harmless; ``wrap_angle`` keeps the
    stored value bounded.
    """
    c = math.cos(pitch)
    c = math.copysign(max(abs(c), _YAW_KIN_EPS), c if c != 0.0 else 1.0)
    return yaw_body_rate / c


def wrap_angle(angle: float) -> float:
    """Wrap an angle to (-π, π]. Keeps ψ bounded after a near-pole slew."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class AeroFrame(NamedTuple):
    """Relative wind resolved onto the body axes."""

    alpha: float          # pitch-plane angle of attack (rad), nose-up positive
    beta: float           # yaw-plane sideslip (rad), engine sign convention
    alpha_total: float    # total angle between body axis and relative wind (rad)
    normal_dir: tuple     # unit vector the aerodynamic normal force acts along
    u: float              # relative speed along the body axis (m/s)


def _unit(vx: float, vy: float, vz: float) -> tuple:
    n = math.sqrt(vx * vx + vy * vy + vz * vz)
    if n <= 1e-12:
        return (0.0, 0.0, 0.0)
    return (vx / n, vy / n, vz / n)


def resolve_aero_frame(vrel: tuple, v_rel: float,
                       pitch: float, yaw: float) -> AeroFrame:
    """Resolve the relative wind onto the body axes.

    Parameters
    ----------
    vrel : (float, float, float)
        Wind-relative velocity in world axes (m/s).
    v_rel : float
        Magnitude of *vrel*. Passed in because both callers already have it.
    pitch, yaw : float
        Body Euler angles (rad).

    Returns
    -------
    AeroFrame
        ``alpha``/``beta`` are the two body-frame aerodynamic angles that drive
        the pitch and yaw moment channels; ``normal_dir`` is the exact 3D
        direction of the aerodynamic normal force — the component of the body
        axis perpendicular to the relative wind, normalised, which is where
        lift acts. It carries the bearing of the flow, so it rotates with a
        crosswind instead of being pinned to the global X–Z plane, and it
        reverses correctly when the flow reverses.
    """
    vrx, vry, vrz = vrel

    cp_, sp_ = math.cos(pitch), math.sin(pitch)
    cy_, sy_ = math.cos(yaw), math.sin(yaw)

    # Body axis (nose), and the two directions it moves in when pitch and yaw
    # increase. These form the body-frame triad used for the aero angles.
    bx, by, bz = cp_ * cy_, cp_ * sy_, sp_
    ex, ey, ez = -sp_ * cy_, -sp_ * sy_, cp_      # d(b)/d(pitch), unit length
    fx, fy, fz = -sy_, cy_, 0.0                   # d(b)/d(yaw), unit length

    # Relative wind projected onto the body triad.
    u = vrx * bx + vry * by + vrz * bz            # along the body axis
    w_n = vrx * ex + vry * ey + vrz * ez          # body "pitch" direction
    v_s = vrx * fx + vry * fy + vrz * fz          # body "yaw" direction

    # Aerodynamic angles. Negated so that a nose pitched ABOVE the relative
    # wind gives alpha > 0, matching the previous `pitch - vel_angle`, and so
    # beta matches the engine's `yaw - flow_bearing`.
    alpha = math.atan2(-w_n, u)
    beta = math.atan2(-v_s, u)

    if v_rel > 1e-9:
        cos_at = max(-1.0, min(1.0, u / v_rel))
        alpha_total = math.acos(cos_at)
        # Component of the body axis perpendicular to the relative wind: the
        # direction lift acts. Zero-length only when the body axis and the
        # flow are exactly aligned (or exactly opposed), where there is no
        # normal force to place anyway.
        vhx, vhy, vhz = vrx / v_rel, vry / v_rel, vrz / v_rel
        normal_dir = _unit(bx - cos_at * vhx,
                           by - cos_at * vhy,
                           bz - cos_at * vhz)
    else:
        alpha_total = 0.0
        normal_dir = (0.0, 0.0, 0.0)

    return AeroFrame(alpha=alpha, beta=beta, alpha_total=alpha_total,
                     normal_dir=normal_dir, u=u)
