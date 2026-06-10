# K2 Physics Validation Suite

Benchmarks every analysis engine against an **independent reference**, so the
physics is *validated*, not merely internally consistent. (For internal
consistency / NASA-STD-7009A credibility tags see `core/validation.py` and
`structures/validation.py` — this suite is the external cross-check.)

| Domain | Fast reference (no external tools) | High-fidelity reference |
|---|---|---|
| **Sim (6DOF)** | exact ODE solutions (vacuum projectile, tanh terminal velocity, SHO energy) | OpenRocket (head-less) |
| **CFD (aero)** | Taylor–Maccoll exact cone flow vs NACA-1135 | SU2 |
| **Structures** | textbook closed form (hoop, Euler buckling) | CalculiX (`bin/ccx.exe`) |

## Layout

```
validation/
  harness.py            # Benchmark / Comparison, tolerance gate, JSON cache
  cases/rocket_canonical.py   # one HPR rocket (L1090W) shared by all domains
  sim/        headless_runner.py (drives the REAL engine), benchmarks.py, openrocket.py*
  cfd/        taylor_maccoll.py (exact), cone_geometry.py, benchmarks.py
  structures/ ccx_direct.py (textbook .inp decks), benchmarks.py
  report.py             # markdown report + overlay plots + JSON cache
tests/validation/       # pytest gates (test_sim/_cfd/_structures.py)
```
`*openrocket.py` is the Tier-B bridge — see *Bootstrap* below; not yet present.

## Running

```bash
pip install -r requirements.txt

# Fast gates only (seconds) — integrator exact solutions, Taylor–Maccoll,
# closed-form structures. No SU2/CalculiX/OpenRocket.
pytest tests/validation -m "not slow" -q

# Full suite (minutes) — adds CalculiX textbook cases and attempts SU2.
pytest tests/validation -q

# Generate the credibility report (fast benchmarks + plots):
python -m validation.report
# Include the slow solver benchmarks:
python -m validation.report --full
# Re-render from the cached JSON without re-solving:
python -m validation.report --from-cache
```

Report output: `validation/report/REPORT.md` + `validation/report/plots/*.png`,
with `benchmarks.json` caching every result.

## What is gated vs. skipped

* **Always gated (fast, deterministic):** RK4/RK45 vs exact ODEs, full-engine
  flight plausibility, Taylor–Maccoll vs NACA-1135, closed-form structural
  formulas.
* **Gated when the solver is available:** CalculiX bar tension / cantilever
  bending / 1st-mode modal (uses bundled `bin/ccx.exe`).
* **Attempt-then-skip:** SU2 cone and Barrowman-vs-SU2, and Sim-vs-OpenRocket.
  These run the real pipeline but **skip (not fail)** when the head-less path is
  untrustworthy — e.g. an auto-exported STL that is not watertight, a
  non-converged SU2 solve, or a missing OpenRocket/Java install. Run those cases
  from the CFD workspace (supervised mesh) for a trustworthy reference.

## Bootstrap (you)

* **OpenRocket (Tier-B sim):** install Java (JRE 17+) and an OpenRocket `.jar`.
  Point `OPENROCKET_JAR` at it; the bridge (`validation/sim/openrocket.py`) then
  runs OpenRocket head-less and compares apogee / max-v / max-q to K2. Until
  then the OpenRocket benchmark skips and the exact-ODE gates still validate the
  integrator.
