"""
Validation harness — shared comparison + tolerance machinery.
==============================================================

A *Benchmark* is one named comparison of K2 output against a reference. It holds
a list of *Comparison* rows (one per quantity), each tagging pass/fail against a
relative and/or absolute tolerance. Benchmarks are JSON-serialisable so the slow
SU2/CalculiX/OpenRocket runs can be cached and the report regenerated without
re-solving.

Reference provenance is recorded on every row (textbook formula, published case,
solver name) so the generated report doubles as a credibility trail, tying back
to the NASA-STD-7009A levels in :mod:`core.validation`.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path

# Reuse the existing credibility classification rather than inventing a parallel one.
from core.validation import ValidationLevel


# ── Error metrics ─────────────────────────────────────────────────────────────

def rel_error(k2: float, ref: float) -> float:
    """Relative error |k2 - ref| / |ref|. Falls back to absolute when ref≈0."""
    denom = abs(ref)
    if denom < 1e-12:
        return abs(k2 - ref)
    return abs(k2 - ref) / denom


def passes(k2: float, ref: float, tol_rel: float = 0.0,
           tol_abs: float = 0.0) -> bool:
    """True if k2 is within EITHER the relative OR absolute tolerance of ref."""
    if tol_abs > 0 and abs(k2 - ref) <= tol_abs:
        return True
    if tol_rel > 0 and rel_error(k2, ref) <= tol_rel:
        return True
    # If only one tolerance was supplied, the other check above already ran.
    return False


def curve_rmse(k2_x, k2_y, ref_x, ref_y) -> float:
    """RMSE of K2 curve vs reference, with K2 linearly interpolated onto ref_x.

    Both curves are sampled on the reference abscissa over their overlapping
    range so differing time/AoA grids compare cleanly.
    """
    if not k2_x or not ref_x:
        return float("nan")
    lo = max(min(k2_x), min(ref_x))
    hi = min(max(k2_x), max(ref_x))
    pts = [(x, y) for x, y in zip(ref_x, ref_y) if lo <= x <= hi]
    if not pts:
        return float("nan")
    sq = 0.0
    for x, yr in pts:
        sq += (_interp(k2_x, k2_y, x) - yr) ** 2
    return math.sqrt(sq / len(pts))


def _interp(xs, ys, x):
    """Linear interpolation of (xs, ys) at x. Assumes xs sorted ascending."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            span = xs[i + 1] - xs[i]
            if span < 1e-15:
                return ys[i]
            f = (x - xs[i]) / span
            return ys[i] + f * (ys[i + 1] - ys[i])
    return ys[-1]


# ── Data model ──────────────────────────────────────────────────────────────

@dataclass
class Comparison:
    """One scalar quantity: K2 value vs reference value, with pass/fail."""
    label: str
    k2: float
    ref: float
    source: str               # provenance of the reference value
    units: str = ""
    tol_rel: float = 0.0
    tol_abs: float = 0.0
    passed: bool = False
    rel_err: float = 0.0
    note: str = ""

    @classmethod
    def make(cls, label, k2, ref, source, units="", tol_rel=0.0,
             tol_abs=0.0, note="") -> "Comparison":
        return cls(
            label=label, k2=float(k2), ref=float(ref), source=source,
            units=units, tol_rel=tol_rel, tol_abs=tol_abs,
            passed=passes(k2, ref, tol_rel, tol_abs),
            rel_err=rel_error(k2, ref), note=note,
        )


@dataclass
class Benchmark:
    """A named group of Comparisons against one reference."""
    name: str
    domain: str               # "sim" | "cfd" | "structures"
    reference: str            # "OpenRocket 23.09" | "Taylor-Maccoll" | "CalculiX 2.21"
    level: ValidationLevel = ValidationLevel.ESTIMATED
    comparisons: list = field(default_factory=list)
    curves: dict = field(default_factory=dict)   # plot_key -> series dict (for report plots)
    notes: str = ""
    skipped: bool = False
    skip_reason: str = ""

    def add(self, comp: Comparison) -> "Benchmark":
        self.comparisons.append(comp)
        return self

    @property
    def passed(self) -> bool:
        if self.skipped:
            return True   # a skip is not a failure; the report flags it separately
        return all(c.passed for c in self.comparisons) and bool(self.comparisons)

    def summary(self) -> str:
        if self.skipped:
            return f"[SKIP] {self.name}: {self.skip_reason}"
        n_pass = sum(c.passed for c in self.comparisons)
        flag = "PASS" if self.passed else "FAIL"
        return f"[{flag}] {self.name}: {n_pass}/{len(self.comparisons)} within tolerance"

    # ── persistence (so slow runs are cached for the report) ──
    def to_dict(self) -> dict:
        d = asdict(self)
        d["level"] = self.level.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Benchmark":
        d = dict(d)
        d["level"] = ValidationLevel(d.get("level", "Estimated"))
        comps = [Comparison(**c) for c in d.pop("comparisons", [])]
        bm = cls(**d)
        bm.comparisons = comps
        return bm


def save_benchmarks(benchmarks: list, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([b.to_dict() for b in benchmarks], indent=2),
        encoding="utf-8",
    )


def load_benchmarks(path: Path) -> list:
    path = Path(path)
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [Benchmark.from_dict(d) for d in raw]
