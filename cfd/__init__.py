"""
K2 Aerospace — CFD Module
==========================
Open-source CFD pipeline using Gmsh (meshing) and SU2 (solver).

Architecture:
  - cfd.solvers.base      → CFDSolver abstract base class
  - cfd.solvers.su2_solver → SU2 implementation
  - cfd.meshing           → Gmsh-based mesh generation
  - cfd.post_processing   → Results parsing and VTK loading
  - cfd.geometry_exporter → Assembly-to-STL conversion
"""

from cfd.solvers.base import CFDSolver, CFDConfig, CFDResult
from cfd.solvers.su2_solver import SU2Solver

__all__ = ["CFDSolver", "CFDConfig", "CFDResult", "SU2Solver"]
