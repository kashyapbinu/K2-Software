# K2 AeroSim — working notes for Claude

Written 2026-11-13. Sections 1–4 describe the code as it stands; section 5
onward is the plan. When the code and this file disagree, trust the code and
fix this file.

## 1. What this is

A Windows desktop app (PyQt6) for designing and analysing high-power and
experimental rockets: design tree, motors, 6DOF flight, Monte Carlo,
optimisation, CFD (SU2), FEM (CalculiX), flutter, a 3D mission view and an AI
assistant. Shipped version: **v0.2.0** (2026-10-04), installer built by GitHub
Actions and delivered through the in-app updater.

**Who it is for:** university competition teams (Spaceport America Cup / IREC,
EuRoC, Launch Canada and similar) first, small launch startups second. Hobby
flyers are not the target. This decides priorities: accuracy, evidence a judge
or a range-safety officer will accept, control and avionics, scripted use.

**The pitch:** one tool from design to a judged report, with structures, CFD
and control included. OpenRocket and RocketPy each cover part of that.

## 2. Where things are

| Path | What |
|---|---|
| `main.py` | Entry point. `--run-script <file>` runs a script inside the frozen app |
| `core/rocket_state.py` | `RocketState` (every scalar the app shares) and `RocketStateEngine` (Qt signals). `FLIGHT_FIELDS` = what one flight writes |
| `core/simulation_engine.py` | Interactive 6DOF flight (QTimer driven) |
| `core/batch_simulation.py` | Headless 6DOF flight, no Qt. Used by Monte Carlo, optimiser, DOE, sensitivity |
| `core/flight_dynamics.py` | Aero frame, step limits shared by both integrators |
| `core/staging.py` | Stages, thrust curves, and the Design → state hops (`assembly_aero_geometry`, `assembly_recovery`) |
| `core/components.py` | Design-tab component tree (`RocketAssembly`) |
| `core/monte_carlo_engine.py`, `core/optimization_engine.py` | Dispersion and optimiser |
| `physics/aerodynamics.py` | `AeroModel` (Barrowman + drag build-up) |
| `recovery/parachute_dynamics.py` | `RecoverySystem`, used by both flight engines |
| `environment/` | ISA atmosphere (`temperature_offset` only), wind models |
| `avionics/` | Sensor models; `FlightComputer.tick` only reads sensors. Kalman filter and apogee detector exist and are not wired to anything |
| `cfd/`, `structures/`, `dynamics/` | SU2, CalculiX, flutter/vibration |
| `ui/main_window.py`, `ui/workspaces/` | 12 tabs; one workspace per file |
| `ui/pdf_report.py` | `save_report()`: cover table + figures → PDF |
| `validation/`, `tests/validation/` | Benchmarks against OpenRocket, SU2, CalculiX and published data |
| `website/` | Marketing site (Netlify) |

Built with no UI yet: `core/trade_study.py`, `core/doe_engine.py`,
`core/sensitivity_analysis.py`, `environment/weather_profiles.py`,
`environment/turbulence.py`. Dead: `ui/workspaces/aero_workspace.py`,
`ui/rocket_params_panel.py`, `plugins/` (a stub).

## 3. Running and testing

```
python main.py                                  # the app
python -m pytest -m "not slow" -q               # fast suite, ~2.5 min, 435 tests
python -m pytest -m slow                        # SU2 / CalculiX / OpenRocket, tens of minutes
```

- Widget tests need a GUI `QApplication`, and the headless sim tests leave a
  bare `QCoreApplication` in the pytest process. So widget tests run their body
  in a subprocess with `QT_QPA_PLATFORM=offscreen` (pattern:
  `tests/validation/test_dock_toggle.py`, `test_simulation_tab_sync.py`).
- To exercise Design-tab logic without an OpenGL viewport, bind the method to a
  plain object (`_design_sync` in `tests/validation/test_design_sync_geometry.py`).
- To probe the flight sim with odd inputs, run each case in its own subprocess
  with a timeout. A hang otherwise stalls the whole run with nothing printed.
- CI: `.github/workflows/build-windows.yml` builds and publishes on a `v*` tag.
  `.github/workflows/tests.yml` runs the fast suite on push and PR; it was added
  2026-11-12 and **has not yet run on a real runner**. It may need an OpenGL
  fix there. It does not gate the release.

## 4. Rules that past bugs taught

Most bugs found so far were silent: a believable number for the wrong rocket.
These rules exist because each was broken at least once.

1. **One source for each number.** If two tabs show the same quantity, both
   read it from one function. Do not recompute it locally.
2. **Never invent a default for something the user drew.** No
   `or <made-up fin>` in a consumer of geometry, mass or recovery fields. Zero
   means zero; a missing component means none.
3. **Design → state goes through `core/staging.py`.** Anything the sim needs
   from the component tree is added to `assembly_aero_geometry` or
   `assembly_recovery` and pushed in `DesignWorkspace._sync_to_engine`. Then
   `BatchSimConfig.from_rocket_state` has to carry it too, or Monte Carlo and
   the optimiser fly a different rocket from the Simulation tab.
4. **Two flight engines, one physics.** `simulation_engine.py` and
   `batch_simulation.py` each have a `_derivatives`. A change to one is a change
   to both; shared pieces live in `core/flight_dynamics.py`,
   `recovery/parachute_dynamics.py`, `core/staging.py`. `batch_simulation.py`
   must stay free of Qt imports (it runs in worker processes).
5. **State ↔ widget both ways.** An input that writes to the state must also be
   shown from the state (`SimulationWorkspace._sync_inputs_from_state`), or an
   opened project flies values the screen does not show.
6. **New `RocketState` field: decide which kind it is.** Rocket or launch setup
   → it is saved. Written by a flight → add it to `FLIGHT_FIELDS` (not saved,
   cleared by `reset_flight`).
7. **Validate at the boundary.** Headless runs refuse unflyable input
   (`config_problems`); project load type-checks each field.
8. **Units.** SI on the state. Fin sweep is degrees on the component and
   radians on the state. Angles and bearings: say which in the field comment.
9. **Mesher changes** bump `cfd.meshing.MESHER_REVISION`, or resumed sweeps
   reuse the old mesh.

Working habits for this repo:

- Source files are CRLF in the working tree. Edit with the Edit/Write tools,
  not shell redirection or PowerShell `Set-Content` (both have corrupted
  unicode or flipped line endings here). Check with `git ls-files --eol`.
- Both PyQt5 and PyQt6 are installed on the dev machine; pin `QT_API=pyqt6`.
  Import QtWebEngine before creating the `QApplication`.
- The native stack (`PyQt6`, `pyvista`, `vtk`, `numpy<2`) is pinned in
  `requirements.txt` for ABI reasons. Bump those together and deliberately.
- Every fix comes with a test that fails without it. Test names are sentences
  about behaviour (`test_rail_exit_velocity_depends_on_the_rail`). Comments
  and docstrings say what the code does and, where it replaced something
  wrong, what the wrong version did and the measured effect.
- Commits: conventional prefix, subject says the user-visible effect
  (`fix(aero): fly every fin set and diameter change the design has`). One
  concern per commit; each commit passes the fast suite on its own.
- Do not commit, push, tag or build unless the user asks. The `website/` edits
  in the working tree are the user's own; leave them out of code commits.
- Release: bump `core/version.py` and `installer.iss` together, tag
  `v<version>`, CI publishes. Run the slow suite before a release.
- Website: keep the cards / centred / animated look. A flat redesign was
  rejected.

## 5. Current position

Branch `fix/post-0.1.10-audit`, local only (nothing pushed since `v0.2.0` at
`150cf24`). Version not bumped.

**Done** (each with tests; see `git log`):

- November audit fixes (7 commits; `test_audit_2026_11.py`,
  `test_simulation_tab_sync.py`).
- **A0** `core/flight_summary.py` — one `FlightSummary` from both engines;
  one trapezoid thrust curve (`staging.trapezoid_thrust_curve`), one static
  margin (`AeroModel.static_margin`), one sensed-acceleration definition.
  Engines agree within 0.02 % on the canonical rocket.
- **A1** launch site (`launch_altitude/latitude/longitude/azimuth`,
  `launch_site_atmosphere`, `core/launch_sites.py`, `data/launch_sites.json`);
  `ground_temperature` 0 = standard day at the site; batch now flies the pad
  temperature; Monte Carlo absolute wind σ.
- **A2** weather soundings (`environment/sounding.py`, `ProfileAtmosphere`,
  `RocketState.atmosphere_profile`).
- **A3** per-parachute deployment (`RecoverySystem` canopy list with its own
  triggers in both engines, `RocketState.recovery_canopies`,
  `Parachute.deploy_event/altitude/delay`, `.ork` deploy import, structures
  recovery loads from the real canopies, canopy lines in Results).

**Not yet done in A0:** `tests.yml` has never run on GitHub (needs a push, which
the user has to approve).

**Known open, not started:**

- Wind-bearing asymmetry: landing range varies about ±1.8 % with wind bearing
  on a vertical launch. It is the limit of the Euler-angle attitude model,
  gated at 2 % by `test_apogee_does_not_depend_on_wind_bearing`. The fix is
  quaternion attitude in both engines (Release C, item C1).
- Transition and boat-tail drag are not modelled; launch-lug and rail-button
  drag neither.
- Flat Earth, 100 km ceiling, batch runner stops at 600 s of flight (a slow
  descent from a high apogee times out; the interactive engine allows 3600 s).
- Sounding wind layers are heights above the pad at import time: moving the
  pad afterwards needs a re-import (the UI says so).
- Structures and flutter still use ISA (above sea level), not an imported
  sounding. Flutter is judged at sea level on purpose (worst case).
- The `.ork` importer maps OpenRocket "ejection" deploys to apogee + delay
  (K2 does not model the motor delay) and logs it.
- Launch-site presets: only "Sea level" and an approximate Spaceport America
  point (the whole spaceport, not the launch area). Real pad coordinates
  should come from the user.
- `ShockCord` has no rated strength, so shock-cord margin is not checked.
- AI assistant: Gemini key is stored in plain text in QSettings.
- Prism boundary-layer meshing is unsupported (SU2 stalls); RANS polars are not
  trustworthy on the tet-only mesh, so the hybrid Euler + flat-plate friction
  mode is the default.

## 6. Release A — "Competition ready" (target v0.3.0)

Goal: a team can design, set the real launch site, tune to a target apogee,
check the design against competition rules and hand a judge one PDF.

Estimates are working days for one developer including tests, and they are
estimates. Total: about 22–27 days.

Do them in this order. Each item is its own commit series and leaves the fast
suite green.

### A0. Groundwork — 1.5 days
- Commit the audit fixes (group by concern), run the slow suite, get
  `tests.yml` green on a runner.
- **Flight summary, one source.** Add `core/flight_summary.py`: a dataclass
  with apogee, max velocity/Mach/acceleration, max-Q, rail-exit velocity,
  stability at rail exit and minimum, drogue and main descent rates, landing
  velocity, drift, flight time. Build it from an interactive flight (history +
  state) and from a `BatchSimResult`. The interactive engine must start
  recording rail-exit velocity (same definition as batch: speed at
  `launch_rod_length`, interpolated). Results tab shows it. A5 and A8 read
  only this object.
- Done when: Results shows rail-exit velocity, and interactive vs batch
  summaries of the canonical rocket agree within 1 %.

### A1. Launch site — 3 days
- State (saved): `launch_altitude` (m above sea level), `launch_latitude`,
  `launch_longitude`, `launch_azimuth` (bearing the rail leans toward, deg),
  optional `ground_pressure`.
- Physics: altitude stays above-ground on the state; every atmosphere and
  gravity lookup uses `z + launch_altitude`. Call sites: both `_derivatives`,
  both main loops, `core/spent_stage.py`, `avionics/sensors/barometer.py`, the
  dynamics/flutter "worst altitude" logic. Grep `atmosphere.` and
  `gravity_at_altitude` to find them all.
- Azimuth: the rail currently always leans along +X. Set the initial yaw from
  the azimuth, and write down the axis convention (X north or east) next to
  the wind-bearing convention in `environment/wind_model.py`.
- Monte Carlo: add an absolute wind-speed σ (m/s) beside the percentage.
- UI: a "Launch Site" group in the Simulation tab with a few presets (editable
  JSON in `data/launch_sites.json`); synced both ways (rule 5).
- Tests: density at pad equals ISA at site elevation; apogee above ground
  rises with elevation; rotating azimuth and wind together rotates the landing
  point and leaves apogee unchanged (within the existing 2 % gate).

### A2. Weather import — 2.5 days
- Parse a sounding (start with University of Wyoming text and a plain CSV:
  altitude, temperature, pressure, wind speed, wind direction) into
  `wind_layers` plus a temperature/pressure profile.
- `environment/atmosphere_model.py`: accept an optional tabulated profile and
  fall back to ISA above its top. One `Atmosphere` object is built per run in
  both engines; pass the profile through `BatchSimConfig`.
- UI: "Import sounding…" in the Wind group; show the profile in the table.
- Tests: a sounding equal to ISA reproduces the ISA flight; a hot day flies
  higher than a cold one by the expected density ratio.

### A3. Recovery — 4 days
- `Parachute` component gains `deploy_event` ("apogee" / "altitude") and
  `deploy_altitude`; the component editor shows them; the `.ork` importer
  reads OpenRocket's `deployevent` / `deployaltitude` (see
  `import_export/ork_importer.py`, and `rocket.ork` for a real file).
- Generalise `RecoverySystem` from drogue + main to a list of canopies, each
  with its own trigger. `assembly_recovery` returns that list; the two scalar
  areas stay as the fallback when the design has no parachute. Both engines
  use it (rule 4).
- Report per canopy: deployment time/altitude/speed, peak opening force,
  descent rate; shock-cord load against `ShockCord` strength.
- Tests: single-deploy, dual-deploy and main-at-apogee designs each fly the
  event sequence drawn; imported `.ork` deploy settings survive a save/load.

### A4. Target-apogee tools — 3.5 days
- `core/apogee_tools.py`, built on `run_batch_simulation`:
  - ballast solver: mass (and its position) to hit a target apogee, bounded
    by a minimum stability margin;
  - sensitivity table: d(apogee)/d(mass, Cd, impulse, wind, launch angle,
    temperature) — reuse `core/sensitivity_analysis.py`;
  - airbrake sizing: extra drag area deployed from burnout, to shave a given
    overshoot (fixed deployment only; closed-loop control is B1).
- UI: a "Target Apogee" panel (Simulation or Results tab). Target entry in m
  or ft.
- Tests: solver's ballast flown through the batch sim lands within 0.5 % of
  target; it reports "unreachable" when the rocket falls short with zero
  ballast.

### A5. Rules checker — 3 days
- `core/rules.py`: a rule = name, quantity from `FlightSummary`, comparison,
  limit, unit, citation text. Presets in `data/rules/*.json`, user-editable.
- **Do not write limits from memory.** Rules change every year. Take each
  preset's numbers from the organiser's current published rules, record the
  document name and year in the preset, and ask the user to confirm them.
- UI: a checklist with pass / fail / not-evaluated and the margin; also
  evaluated over a Monte Carlo run ("passes in 97 % of flights").
- Tests: each comparison type; a preset with an unknown quantity is reported,
  not ignored.

### A6. Landing map — 2.5 days
- Convert landing points to lat/long from the launch site (A1). Fit a
  dispersion ellipse (1σ, 2σ, 3σ) in `core/monte_carlo_engine.py`.
- Export KML (pad, nominal track, landing points, ellipses) and CSV.
- Map view in the Monte Carlo tab: lat/long grid and scale bar always;
  background tiles only if the user turns them on (network use must be
  opt-in, and the app must work offline).
- Tests: a known offset maps to the right lat/long; KML parses and holds the
  expected placemarks.

### A7. Technical report — 3.5 days
- One PDF built from `FlightSummary`, the rules check, the Monte Carlo
  summary and the design: vehicle table and drawing, mass and stability
  budget, motor, flight profile plots, recovery sequence, dispersion map,
  structural and flutter margins when those analyses have been run, rules
  checklist, the assumptions and model limits (section 5 above).
- Extend `ui/pdf_report.py` or use `reportlab` (already a dependency; see
  `structures/report.py`). A section whose analysis was not run says so; it
  is never filled with defaults (rule 2).
- Tests: report builds headless for the canonical rocket; every number in it
  equals the `FlightSummary` value.

### A8. Display units — 2.5 days
- A display-unit setting (SI / imperial) applied at the widget edge only. The
  state stays SI (rule 8). Start with what competitions quote: altitude (ft),
  velocity (ft/s), mass (lb), length (in), force (lbf).
- One `ui/units.py` with format and parse helpers; no ad-hoc conversions.
- Tests: round-trip format → parse for each unit; saved files are unchanged
  by the setting.

### Release A exit
- Fast and slow suites green; `validation/` report regenerated.
- A project made in v0.2.0 opens and flies the same apogee (within 0.5 %)
  when launch altitude is 0 and no parachute deploy settings are present.
- Website and README updated; version bumped; user approves the tag.

## 7. Release B — "Control and avionics" (target v0.4.0)

Rough size: 5–7 weeks. Plan each item in detail when Release A is out.

- **B1. Airbrake control in the loop.** A controller interface called at a
  fixed rate inside both engines: inputs are *estimated* state, output is
  brake deployment. Ship a reference apogee-targeting controller and allow a
  user Python file. Actuator rate and lag are modelled.
- **B2. Flight computer in the loop.** Wire `avionics/state_estimation`
  (Kalman filter, apogee detector) into `FlightComputer.tick`, make it a real
  state machine, and let it — not the truth phase — fire the parachutes.
  Keep a "perfect avionics" switch so existing results stay reproducible.
  Model barometer lag and the transonic pressure spike.
- **B3. Hardware in the loop.** Stream simulated sensor data over serial at
  the sensor's real rate; read deployment commands back. Define the packet
  format in one documented module.
- **B4. Post-flight reconstruction.** Fit Cd scale, impulse scale and wind to
  an imported flight log (`core/flight_log.py` already parses them), report
  the fitted values with their uncertainty, and offer to apply them.
- **B5. Wire the unused engines to the UI.** Trade study, DOE, Sobol/Morris.

How: B1 and B2 both need a clean per-step hook in the flight loop. Do C1 (the
shared kernel) first if the hook would otherwise be written twice.

## 8. Release C — "Startup grade" (target v0.5.0)

Rough size: 8–12 weeks.

- **C1. One fast flight kernel.** A single implementation of the equations of
  motion used by both engines, with quaternion attitude (removes the
  wind-bearing asymmetry and the pole handling in `flight_dynamics.py`).
  Steps: (1) profile a batch flight, currently about 1 s each; (2) write the
  kernel so many flights integrate as one NumPy array, or compile it with
  Numba; (3) only if that misses the target (10,000-run Monte Carlo in
  minutes), write it in Rust with PyO3 and maturin. Rust adds a toolchain to
  CI and a compiled module to the PyInstaller bundle, so it has to earn its
  place with a measurement. The rest of the app stays Python.
  Gate: every flight in `validation/` reproduces within its existing
  tolerance, and the bearing test tightens from 2 % to 0.1 %.
- **C2. High-altitude physics.** Rotating round Earth (WGS-84), a proper
  gravity model, an atmosphere above 86 km, removal of the 100 km and 600 s
  cut-offs in `batch_simulation.py`.
- **C3. Thrust vectoring and guidance** through the B1 controller interface.
- **C4. Range safety.** Impact-point dispersion with failure modes (motor
  cut-off at any time, no parachute, fin loss), impact probability contours.
- **C5. Python API and CLI.** `import k2; k2.fly(project)`; headless runs for
  scripts and CI. The API is the same code the UI calls.
- **C6. Propulsion depth.** Feed-system pressure drop, tank blow-down,
  regenerative-cooling check, hybrid regression model
  (`physics/liquid_propulsion.py`, `physics/internal_ballistics.py`).

## 9. Across every release

- **Validation is the product.** Each new model ships with a benchmark in
  `validation/` against flight data or a reference code, and the limits it
  was checked over are written in the report. A feature without one is not
  finished.
- **Say what is not modelled.** The technical report and the UI state the
  model's limits rather than leaving a gap that looks like a zero.
- **Old projects keep opening.** New state fields get defaults that reproduce
  the old behaviour; `RocketState.from_dict` ignores what it does not know.
- **Ask before deciding for the user:** competition rule numbers, anything
  that changes results for existing projects, network use, new dependencies
  (especially compiled ones), and anything published (tag, release, website).
