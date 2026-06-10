"""
Sim 6DOF validation gates.

Tier-A integrator benchmarks run fast and always. The full-engine plausibility
run is a touch slower but needs no externals. The OpenRocket comparison skips
cleanly when no jar/Java is present.
"""
import pytest

from validation.sim import benchmarks as B
from tests.validation.conftest import assert_benchmark


@pytest.mark.sim
@pytest.mark.parametrize("integrator", ["rk4", "rk45"])
def test_vacuum_projectile(integrator):
    assert_benchmark(B.bench_vacuum_projectile(integrator))


@pytest.mark.sim
@pytest.mark.parametrize("integrator", ["rk4", "rk45"])
def test_terminal_velocity(integrator):
    assert_benchmark(B.bench_terminal_velocity(integrator))


@pytest.mark.sim
@pytest.mark.parametrize("integrator", ["rk4", "rk45"])
def test_oscillator_energy(integrator):
    assert_benchmark(B.bench_oscillator_energy(integrator))


@pytest.mark.sim
def test_full_flight_plausibility():
    assert_benchmark(B.bench_full_flight_plausibility())


@pytest.mark.sim
@pytest.mark.slow
def test_sim_vs_openrocket():
    assert_benchmark(B.bench_openrocket())
