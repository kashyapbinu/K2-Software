"""
Shared fixtures + path setup for the validation test suite.
"""
import sys
from pathlib import Path

import pytest

# Make the project root importable when pytest is invoked from anywhere.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@pytest.fixture
def canonical():
    """Fresh canonical validation rocket state."""
    from validation.cases.rocket_canonical import canonical_state
    return canonical_state()


def assert_benchmark(bm):
    """Assert a Benchmark passed, with a readable message listing failed rows."""
    if bm.skipped:
        pytest.skip(bm.skip_reason)
    failed = [c for c in bm.comparisons if not c.passed]
    detail = "\n".join(
        f"    {c.label}: k2={c.k2:.4g} ref={c.ref:.4g} "
        f"rel_err={c.rel_err:.2%} (tol_rel={c.tol_rel}, tol_abs={c.tol_abs}) {c.note}"
        for c in failed
    )
    assert bm.passed, f"{bm.name} failed {len(failed)} comparison(s):\n{detail}"
