"""Tolerance semantics of the benchmark harness itself."""
from validation.harness import Comparison, passes


def test_two_sided_tolerance_is_a_band():
    assert passes(1.05, 1.0, tol_rel=0.10)
    assert passes(0.95, 1.0, tol_rel=0.10)
    assert not passes(1.20, 1.0, tol_rel=0.10)
    assert not passes(0.80, 1.0, tol_rel=0.10)


def test_one_sided_below_accepts_any_improvement():
    """A refinement gate: smaller than the reference is the desired outcome."""
    assert passes(0.01, 0.5, one_sided="below")          # far better, still passes
    assert passes(0.5, 0.5, one_sided="below")           # equal is not worse
    assert passes(0.55, 0.5, tol_rel=0.15, one_sided="below")   # inside the slack
    assert not passes(0.75, 0.5, tol_rel=0.15, one_sided="below")


def test_one_sided_above_is_the_mirror():
    assert passes(10.0, 0.5, one_sided="above")
    assert not passes(0.2, 0.5, tol_rel=0.15, one_sided="above")


def test_comparison_records_the_direction():
    c = Comparison.make("trend", 0.02, 0.4, "src", one_sided="below")
    assert c.passed and c.one_sided == "below"
    # rel_err still reports the plain deviation — a one-sided pass does not mean
    # the two numbers agree, and the report shows both.
    assert c.rel_err > 0.9
