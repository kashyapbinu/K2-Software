"""
CFD validation gates.

The Taylor–Maccoll reference gate is fast and always runs. The SU2 cases are
slow (gmsh + SU2, minutes) and skip cleanly if the head-less pipeline fails.
"""
import pytest

from validation.cfd import benchmarks as B
from tests.validation.conftest import assert_benchmark


@pytest.mark.cfd
def test_taylor_maccoll_reference():
    assert_benchmark(B.bench_taylor_maccoll_reference())


@pytest.mark.cfd
@pytest.mark.slow
def test_su2_cone_vs_taylor_maccoll():
    assert_benchmark(B.bench_su2_cone())


@pytest.mark.cfd
@pytest.mark.slow
def test_barrowman_vs_su2():
    assert_benchmark(B.bench_barrowman_vs_su2())
