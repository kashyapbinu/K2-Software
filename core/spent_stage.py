"""
K2 AeroSim — Spent-Stage Ballistic Tracker
=============================================
After a stage separates it is dead weight: no thrust, no active stabilisation,
tumbling. This propagates each dropped stage from its separation state to the
ground so the UI/report can show where spent boosters land (range safety).

The propagation is a lightweight point-mass ballistic integrator (semi-implicit
Euler) — far cheaper than the full 6DOF and accurate enough for an impact
footprint:
    * tumbling drag: Cd ≈ 1.0 on the stage's side (planform) area, opposing the
      wind-relative velocity (a tumbling body presents ~its side, not the small
      nose area), unless a recovery CdA is supplied.
    * same atmosphere + wind field as the main flight (so it drifts realistically).
"""

import math

TUMBLE_CD = 1.0          # tumbling bluff-body drag coefficient
G0 = 9.80665


def _drag_area(snapshot: dict) -> float:
    """Effective Cd·A for the falling stage. Recovery CdA if chuted, else the
    tumbling side (planform) area × tumbling Cd."""
    cda = snapshot.get("recovery_cd_area", 0.0)
    if cda and cda > 0:
        return cda
    length = max(snapshot.get("length", 0.5), 0.01)
    diameter = max(snapshot.get("diameter", 0.05), 0.01)
    return TUMBLE_CD * (length * diameter)      # side-on planform area


def propagate(snapshot: dict, atmosphere, wind_model,
              gravity_fn, dt: float = 0.05, max_time: float = 3600.0) -> dict:
    """Integrate one spent stage to the ground.

    Args:
        snapshot: dict from separation — keys: stage, t, x, y, z, vx, vy, vz,
                  mass, length, diameter, [recovery_cd_area].
        atmosphere:  object with .density(z).
        wind_model:  object with .get_wind_velocity(z, t) -> (vx, vy, vz).
        gravity_fn:  callable(z) -> g (m/s²).
        dt:          integration step (s).
        max_time:    safety cap (s).

    Returns dict: stage, landing_x, landing_y, drift, impact_velocity,
                  flight_time, peak_altitude.
    """
    x = snapshot["x"]; y = snapshot["y"]; z = max(0.0, snapshot["z"])
    vx = snapshot["vx"]; vy = snapshot["vy"]; vz = snapshot["vz"]
    mass = max(snapshot.get("mass", 1.0), 1e-3)
    cda = _drag_area(snapshot)
    t0 = snapshot.get("t", 0.0)

    peak = z
    t = 0.0
    while z > 0.0 and t < max_time:
        rho = atmosphere.density(z)
        g = gravity_fn(z)
        wvx, wvy, wvz = wind_model.get_wind_velocity(z, t0 + t)

        # wind-relative velocity drives drag
        rvx, rvy, rvz = vx - wvx, vy - wvy, vz - wvz
        vrel = math.sqrt(rvx * rvx + rvy * rvy + rvz * rvz)
        if vrel > 1e-6:
            fd = 0.5 * rho * vrel * vrel * cda
            ax = -fd * rvx / vrel / mass
            ay = -fd * rvy / vrel / mass
            az = -fd * rvz / vrel / mass - g
        else:
            ax = ay = 0.0
            az = -g

        vx += ax * dt; vy += ay * dt; vz += az * dt
        x += vx * dt; y += vy * dt; z += vz * dt
        if z > peak:
            peak = z
        t += dt

    impact_v = math.sqrt(vx * vx + vy * vy + vz * vz)
    return {
        "stage": snapshot.get("stage"),
        "landing_x": x,
        "landing_y": y,
        "drift": math.sqrt(x * x + y * y),
        "impact_velocity": impact_v,
        "flight_time": t,
        "peak_altitude": peak,
        "separation_altitude": snapshot["z"],
    }


def propagate_all(snapshots, atmosphere, wind_model, gravity_fn, **kw) -> list:
    """Propagate every spent-stage snapshot to the ground."""
    return [propagate(s, atmosphere, wind_model, gravity_fn, **kw)
            for s in snapshots]
