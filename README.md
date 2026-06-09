# K2 Aerospace — Rocket Simulation Platform

Integrated aerospace digital twin for high-power and experimental rockets: 6DOF
flight simulation, aerodynamics, propulsion / internal ballistics, structures,
recovery, CFD and FEM tooling, with a PyQt6 desktop UI.

## Modules

| Package | Purpose |
|---|---|
| `physics/` | Aerodynamics (Barrowman port), propulsion, internal ballistics, structures, drag tables |
| `core/` | Simulation engine (6DOF), integrators (RK4 / RK45 Dormand-Prince), Monte Carlo, optimization, DOE |
| `environment/` | ISA atmosphere, wind, turbulence, weather profiles |
| `dynamics/` | Flutter (NACA TN-4197 + p-k), aeroelastic, vibration |
| `recovery/` | Drogue / main chute, descent dynamics, deployment logic |
| `avionics/` | Flight computer, sensors, Kalman state estimation, telemetry |
| `cfd/` | SU2 / meshing interface, boundary layer, shock detection, sweeps |
| `structures/` | FEM interface (CalculiX), thermal, pressure mapping, reporting |
| `vehicle/` | Component model, builder, mass/inertia |
| `ui/` | PyQt6 workspaces (design, sim, aero, propulsion, structures, dynamics, Monte Carlo, results) |
| `visualization/` | 3D viewer, mission visualizer |

## Requirements

```
PyQt6>=6.5
pyvista>=0.43
pyvistaqt>=0.11
numpy>=1.24
scipy>=1.10
scikit-learn>=1.3
```

## Run

```bash
pip install -r requirements.txt
python main.py
```

## Notes

- Physics verified against ISA tables, OpenRocket Barrowman method, NACA TN-4197
  flutter, and Roark thin-shell stress formulas.
- Large run artifacts (`cfd_run/`, `fem_run/`), bundled SU2 binaries (`bin/`),
  and the vendored upstream `scratch/openMotor_repo/` are excluded from version
  control — see `.gitignore`.
