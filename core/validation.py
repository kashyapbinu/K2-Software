"""
K2 Aerospace — Result Validation Framework
============================================
Every analysis result carries a ValidationInfo describing:
- Validation level  (VALIDATED / ESTIMATED / SIMPLIFIED / PLACEHOLDER)
- Governing equations used
- Assumptions made
- Checks that passed or failed
- Units of the primary output
- Numerical confidence metric (0–1)

Reference: NASA-STD-7009A — Standard for Models and Simulations
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("K2.Validation")


class ValidationLevel(Enum):
    """Classification of result trustworthiness."""
    VALIDATED   = "Validated"       # cross-checked against FEM, test, or reference
    ESTIMATED   = "Estimated"       # physics-based but not fully validated
    SIMPLIFIED  = "Simplified"      # closed-form with known limitations
    PLACEHOLDER = "Placeholder"     # artificial / hardcoded — must be replaced


@dataclass
class ValidationInfo:
    """Metadata attached to every analysis result.

    Follows NASA-STD-7009A guidance for model credibility.
    """
    level: ValidationLevel = ValidationLevel.ESTIMATED
    governing_equations: list = field(default_factory=list)
    # e.g. ["Euler-Bernoulli beam bending", "NACA TN-4197 flutter"]
    assumptions: list = field(default_factory=list)
    # e.g. ["Thin-wall approximation", "Isotropic material"]
    checks_passed: list = field(default_factory=list)
    # e.g. ["Mass conservation < 1%", "Eigenvalue convergence"]
    checks_failed: list = field(default_factory=list)
    units: str = ""
    # Primary output unit, e.g. "Pa", "Hz", "m/s"
    confidence: float = 0.5
    # 0.0 = no confidence, 1.0 = fully verified
    notes: str = ""

    @property
    def level_str(self) -> str:
        return self.level.value

    def flag(self, level: ValidationLevel, confidence: float,
             equations: list = None, assumptions: list = None,
             units: str = "", notes: str = ""):
        """Convenience setter for bulk-populating fields."""
        self.level = level
        self.confidence = max(0.0, min(1.0, confidence))
        if equations:
            self.governing_equations = equations
        if assumptions:
            self.assumptions = assumptions
        if units:
            self.units = units
        if notes:
            self.notes = notes
        return self

    def add_check(self, description: str, passed: bool):
        """Record a validation check."""
        if passed:
            self.checks_passed.append(description)
        else:
            self.checks_failed.append(description)
            # Downgrade confidence if a check fails
            self.confidence = max(0.0, self.confidence - 0.15)
        return self

    def summary(self) -> str:
        """One-line summary for logging / status bars."""
        n_pass = len(self.checks_passed)
        n_fail = len(self.checks_failed)
        return (f"[{self.level.value}] confidence={self.confidence:.0%}, "
                f"checks={n_pass}✓/{n_fail}✗")


def make_validation(level: ValidationLevel = ValidationLevel.ESTIMATED,
                    confidence: float = 0.5,
                    equations: list = None,
                    assumptions: list = None,
                    units: str = "",
                    notes: str = "") -> ValidationInfo:
    """Factory for creating a pre-populated ValidationInfo."""
    v = ValidationInfo()
    v.flag(level, confidence, equations, assumptions, units, notes)
    return v
