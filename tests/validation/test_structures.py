"""
Structural validation gates.

The closed-form formula gate runs fast and always. The CalculiX cases need the
bundled ccx.exe and are marked slow (minutes).
"""
import pytest

from validation.structures import benchmarks as B
from tests.validation.conftest import assert_benchmark


@pytest.mark.structures
def test_closed_form_formulas():
    assert_benchmark(B.bench_closed_form_formulas())


@pytest.mark.structures
@pytest.mark.slow
def test_bar_tension():
    assert_benchmark(B.bench_bar_tension())


@pytest.mark.structures
@pytest.mark.slow
def test_cantilever_bending():
    assert_benchmark(B.bench_cantilever_bending())


@pytest.mark.structures
@pytest.mark.slow
def test_modal_vs_ccx():
    assert_benchmark(B.bench_modal_vs_ccx())
