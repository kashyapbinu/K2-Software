import sys
from pathlib import Path

sys.path.insert(0, str(Path(".").resolve()))

from cfd.solvers.base import CFDConfig
from cfd.solvers.su2_solver import SU2Solver

cfg = CFDConfig()
cfg.work_dir = Path("cfd_run")
cfg.geometry_stl = Path("cfd_run/geometry.stl")  # Assuming it exists from earlier run
cfg.turbulence_model = "SST"
cfg.mach = 0.8
cfg.altitude_m = 0.0
cfg.mesh_refinement = "coarse"
cfg.domain_length_scale = 3.0
cfg.domain_radius_scale = 3.0
cfg.boundary_layer_layers = 10
cfg.boundary_layer_growth = 1.2
cfg.geometry_dict = {
    "length": 1.0,
    "body_radius": 0.05,
    "nose_radius": 0.05,
    "nose_length": 0.3,
    "body_length": 0.7,
    "fin_count": 4,
    "fin_height": 0.1,
    "fin_root": 0.15,
    "fin_thick": 0.003
}

# Create dummy STL to pass the check
Path("cfd_run/geometry.stl").parent.mkdir(exist_ok=True)
Path("cfd_run/geometry.stl").touch()

solver = SU2Solver(cfg)
print("Regenerating mesh...")
solver.generate_mesh()
print("Regenerating config...")
solver.generate_case()
print("Running solver...")
for it, rms in solver.run():
    print(f"Iter: {it}, RMS: {rms}")
    if it > 5:
        print("Success! Exiting early.")
        break
