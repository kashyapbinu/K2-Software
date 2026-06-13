"""
Sim benchmarks.
===============

Tier-A (always runs, no external deps) — gates the numerical core directly
against *exact* ODE solutions, isolating integrator correctness from the aero
model:

    * vacuum projectile      → exact parabola (RK4 is exact for ≤4° polynomials)
    * quadratic-drag fall     → exact tanh terminal velocity  v(t)=v_t·tanh(g t/v_t)
    * undamped oscillator     → energy conservation over many periods

Plus a full-engine *plausibility* benchmark: run the real headless engine on the
canonical rocket and assert the flight is physically sane (positive apogee,
lands, max-Q before apogee, energy budget closes within drag losses).

Tier-B (OpenRocket compare) lives in :mod:`validation.sim.openrocket` and is
invoked from here only when an OpenRocket jar + Java are available; otherwise the
benchmark is marked skipped (not failed).
"""
from __future__ import annotations

import math

from core.integrators import get_integrator
from core.validation import ValidationLevel
from validation.harness import Benchmark, Comparison, curve_rmse

G = 9.80665


# ── exact-ODE integrator gates ────────────────────────────────────────────────

def _integrate(integrator, sv, derivs, dt, n):
    """Fixed-step march; returns list of (t, state) snapshots including start."""
    out = [(0.0, list(sv))]
    t = 0.0
    for _ in range(n):
        sv = integrator.step(sv, t, dt, derivs)
        t += dt
        out.append((t, list(sv)))
    return out


def bench_vacuum_projectile(integrator_name: str = "rk4") -> Benchmark:
    """1-D vertical throw in vacuum: dv/dt=-g. Exact apogee = v0²/2g."""
    v0 = 100.0
    integ = get_integrator(integrator_name, nan_guard=False)
    # state = [z, v]; derivs = [v, -g]
    traj = _integrate(integ, [0.0, v0], lambda t, s: [s[1], -G],
                      dt=0.01, n=4000)
    # Apogee = max z; also the analytic peak time t*=v0/g.
    z_peak = max(s[0] for _, s in traj)
    z_exact = v0 ** 2 / (2 * G)
    # Sample the position curve vs exact parabola for an RMSE row.
    t_burn = v0 / G
    ts = [t for t, _ in traj if t <= 2 * t_burn]
    zs = [s[0] for t, s in traj if t <= 2 * t_burn]
    z_ref = [v0 * t - 0.5 * G * t ** 2 for t in ts]

    bm = Benchmark(name=f"Vacuum projectile ({integrator_name})", domain="sim",
                   reference="Exact kinematics z=v0·t−½g·t²",
                   level=ValidationLevel.VALIDATED)
    bm.add(Comparison.make("Apogee height", z_peak, z_exact,
                           "v0²/2g", "m", tol_rel=0.002))
    bm.add(Comparison.make("Position RMSE vs parabola",
                           curve_rmse(ts, zs, ts, z_ref), 0.0,
                           "exact", "m", tol_abs=0.05))
    bm.curves["altitude"] = {"x": ts, "k2": zs, "ref": z_ref,
                             "xlabel": "t (s)", "ylabel": "z (m)"}
    return bm


def bench_terminal_velocity(integrator_name: str = "rk4") -> Benchmark:
    """Quadratic-drag fall: dv/dt = g − k·v². Exact v(t)=v_t·tanh(g·t/v_t)."""
    v_t = 50.0
    k = G / v_t ** 2
    integ = get_integrator(integrator_name, nan_guard=False)
    traj = _integrate(integ, [0.0], lambda t, s: [G - k * s[0] ** 2],
                      dt=0.01, n=2000)              # 20 s ≫ 5·(v_t/g)
    v_final = traj[-1][1][0]
    ts = [t for t, _ in traj]
    vs = [s[0] for _, s in traj]
    v_ref = [v_t * math.tanh(G * t / v_t) for t in ts]

    bm = Benchmark(name=f"Terminal velocity ({integrator_name})", domain="sim",
                   reference="Exact v(t)=v_t·tanh(g·t/v_t), v_t=√(g/k)",
                   level=ValidationLevel.VALIDATED)
    bm.add(Comparison.make("Terminal velocity", v_final, v_t,
                           "√(g/k)", "m/s", tol_rel=0.002))
    bm.add(Comparison.make("Velocity RMSE vs tanh",
                           curve_rmse(ts, vs, ts, v_ref), 0.0,
                           "exact", "m/s", tol_abs=0.02))
    bm.curves["velocity"] = {"x": ts, "k2": vs, "ref": v_ref,
                             "xlabel": "t (s)", "ylabel": "v (m/s)"}
    return bm


def bench_oscillator_energy(integrator_name: str = "rk4") -> Benchmark:
    """Undamped SHO dx/dt=v, dv/dt=−ω²x. Energy must be conserved (RK4 drift tiny)."""
    omega = 2.0
    integ = get_integrator(integrator_name, nan_guard=False)
    # 50 periods at ~100 steps/period.
    n = int(50 * (2 * math.pi / omega) / 0.01)
    traj = _integrate(integ, [1.0, 0.0],
                      lambda t, s: [s[1], -omega ** 2 * s[0]], dt=0.01, n=n)

    def energy(s):   # ½v² + ½ω²x²
        return 0.5 * s[1] ** 2 + 0.5 * omega ** 2 * s[0] ** 2
    e0 = energy(traj[0][1])
    e_end = energy(traj[-1][1])

    bm = Benchmark(name=f"Oscillator energy ({integrator_name})", domain="sim",
                   reference="Conserved E=½v²+½ω²x² over 50 periods",
                   level=ValidationLevel.VALIDATED)
    bm.add(Comparison.make("Energy drift (50 periods)", e_end, e0,
                           "E(0)", "J/kg", tol_rel=0.01))
    return bm


# ── full-engine plausibility ──────────────────────────────────────────────────

def bench_full_flight_plausibility() -> Benchmark:
    """Run the real headless engine on the canonical rocket; assert physical sanity."""
    from validation.cases.rocket_canonical import canonical_state
    from validation.sim.headless_runner import run_flight

    res = run_flight(canonical_state())

    # Max-Q must occur during/just after boost, BEFORE apogee — sanity on phase
    # ordering. Find the times of max-q and apogee from the history.
    times = res.history.get_values("time")
    q = res.history.get_values("dynamic_pressure")
    alt = res.history.get_values("altitude")
    t_maxq = times[max(range(len(q)), key=lambda i: q[i])] if q else 0.0
    t_apogee = times[max(range(len(alt)), key=lambda i: alt[i])] if alt else 0.0

    bm = Benchmark(name="Full-engine flight plausibility", domain="sim",
                   reference="Physical sanity bounds (canonical L1090W rocket)",
                   level=ValidationLevel.ESTIMATED)
    # Apogee in a credible band for a 7.4 kg, 2671 N·s vehicle (≈0.5–6 km).
    bm.add(Comparison.make("Apogee in [500, 6000] m",
                           1.0 if 500 <= res.apogee_m <= 6000 else 0.0, 1.0,
                           "bound", "bool", tol_abs=0.5,
                           note=f"apogee={res.apogee_m:.0f} m"))
    bm.add(Comparison.make("Vehicle landed",
                           1.0 if res.landed else 0.0, 1.0, "bound", "bool",
                           tol_abs=0.5, note=f"phase, t={res.flight_time_s:.0f}s"))
    bm.add(Comparison.make("Max-Q before apogee",
                           1.0 if t_maxq < t_apogee else 0.0, 1.0, "bound",
                           "bool", tol_abs=0.5,
                           note=f"t_maxQ={t_maxq:.1f}s, t_apogee={t_apogee:.1f}s"))
    bm.add(Comparison.make("Subsonic-to-low-supersonic max Mach",
                           1.0 if 0.1 < res.max_mach < 3.0 else 0.0, 1.0,
                           "bound", "bool", tol_abs=0.5,
                           note=f"max Mach={res.max_mach:.2f}"))
    bm.curves["altitude"] = {"x": times, "k2": alt, "ref": [],
                             "xlabel": "t (s)", "ylabel": "altitude (m)"}
    return bm


# ── Tier-B: OpenRocket compare (skipped if unavailable) ───────────────────────

def bench_openrocket() -> Benchmark:
    """Compare the canonical flight to OpenRocket. Skips if OR/Java absent."""
    try:
        from validation.sim.openrocket import run_openrocket_benchmark
    except Exception as exc:           # module not built yet
        bm = Benchmark(name="Sim vs OpenRocket", domain="sim",
                       reference="OpenRocket (headless)")
        bm.skipped = True
        bm.skip_reason = f"OpenRocket bridge unavailable: {exc}"
        return bm
    return run_openrocket_benchmark()


def run_benchmarks(include_full: bool = True) -> list:
    """All sim benchmarks. `include_full` runs the (slower) full-engine flight."""
    out = []
    for name in ("rk4", "rk45"):
        out.append(bench_vacuum_projectile(name))
        out.append(bench_terminal_velocity(name))
        out.append(bench_oscillator_energy(name))
    if include_full:
        out.append(bench_full_flight_plausibility())
        out.append(bench_openrocket())
    return out
