"""
Surrogate accuracy and screening gates.

None of these touch the simulator: the surrogate is scored against analytic
functions whose true value is known, so a regression shows up as a number
rather than as a slow optimisation run that merely converges badly.
"""
import numpy as np
import pytest

from core.surrogate_model import create_surrogate, _MODEL_MAP


def _smooth_dataset(n=120, d=4, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, (n, d))
    y = (3.0 * X[:, 0] ** 2 - 2.0 * X[:, 1] + 0.5 * X[:, 2] * X[:, 3]
         + rng.normal(0.0, noise, n))
    return X, y


@pytest.mark.optimization
@pytest.mark.parametrize("name", sorted(_MODEL_MAP))
def test_model_fits_smooth_function(name):
    """Every registered surrogate must fit a smooth function and cross-validate.

    This also guards the constructors themselves: RBF's default kernel used to
    raise on every fit because scipy needs an explicit ``epsilon`` for it.
    """
    X, y = _smooth_dataset()
    m = create_surrogate(name)
    m.fit(X, y)
    sc = m.score()
    assert sc["cv_used"], f"{name}: every CV fold failed"
    assert sc["r2"] > 0.8, f"{name}: cross-validated R²={sc['r2']:.3f}"
    assert m.predict(X[:5]).shape == (5,)


@pytest.mark.optimization
@pytest.mark.parametrize("name", ["random_forest", "rbf", "kriging"])
def test_cv_score_rejects_pure_noise(name):
    """Cross-validated R² must expose a surrogate that has learned nothing.

    The training-set fit cannot: these models interpolate their own training
    points, so they report a high training R² on unlearnable data. Anything
    built on that number — the active-learning convergence test, the accuracy
    shown in the results panel — would be reading noise as a perfect fit.
    """
    X, _ = _smooth_dataset()
    y = np.random.default_rng(7).normal(0.0, 1.0, len(X))
    m = create_surrogate(name)
    m.fit(X, y)
    sc = m.score()
    assert sc["r2"] < 0.2, f"{name}: CV R²={sc['r2']:.3f} on pure noise"
    assert sc["r2_train"] > sc["r2"], f"{name}: training fit should be the optimistic one"


@pytest.mark.optimization
def test_score_falls_back_below_two_folds():
    X, y = _smooth_dataset(n=3)
    m = create_surrogate("random_forest")
    m.fit(X, y)
    sc = m.score()
    assert sc["cv_used"] is False and sc["n_folds"] == 0
    assert sc["r2"] == pytest.approx(sc["r2_train"])


# ── screening ────────────────────────────────────────────────────────────────

def _screener_fixtures(pool_factor=4):
    from core.optimization_engine import (
        DesignVariable, OptimizationConfig, _make_screener)
    dvs = [
        DesignVariable("a", "A", "Geometry", 0.0, 1.0, 0.5, True),
        DesignVariable("b", "B", "Geometry", 0.0, 1.0, 0.5, True),
        DesignVariable("m", "M", "Propulsion", 0.0, 2.0, 0.0, True,
                       "discrete", ["x", "y", "z"]),
    ]
    cfg = OptimizationConfig(use_surrogate=True, surrogate_pool_factor=pool_factor)
    return dvs, _make_screener(cfg, dvs, np.random.default_rng(1))


def _truth(d):
    return -((d["a"] - 0.8) ** 2) - ((d["b"] - 0.2) ** 2)


def _designs(n, rng):
    from core.optimization_engine import CandidateDesign
    out = []
    for _ in range(n):
        d = {"a": float(rng.random()), "b": float(rng.random()),
             "m": str(rng.choice(["x", "y", "z"]))}
        out.append(CandidateDesign(variables=d, fitness=_truth(d)))
    return out


@pytest.mark.optimization
def test_encode_handles_discrete_and_junk():
    from core.optimization_engine import _encode_variables
    dvs, _ = _screener_fixtures()
    assert list(_encode_variables({"a": 0.2, "b": 0.3, "m": "z"}, dvs)) == [0.2, 0.3, 2.0]
    # A missing option or a non-numeric value must degrade to 0, not raise:
    # one malformed candidate cannot be allowed to kill the run.
    assert list(_encode_variables({"a": 0.2, "b": None, "m": "nope"}, dvs)) == [0.2, 0.0, 0.0]


@pytest.mark.optimization
def test_screening_selects_better_designs():
    rng = np.random.default_rng(3)
    dvs, s = _screener_fixtures()
    assert s.pool_size(10) == 10, "no pool widening before there is training data"

    s.observe(_designs(60, rng))
    assert s.pool_size(10) == 40

    pool = [c.variables for c in _designs(40, rng)]
    kept = s.screen(pool, 10, generation=0)
    assert len(kept) == 10
    assert np.mean([_truth(d) for d in kept]) > np.mean([_truth(d) for d in pool])

    acc = s.final_score()
    assert acc["cv_used"] and acc["r2"] > 0.5


@pytest.mark.optimization
def test_screener_survives_degenerate_input():
    from core.optimization_engine import CandidateDesign
    rng = np.random.default_rng(4)
    dvs, s = _screener_fixtures()

    # Failed candidates carry -inf fitness and must never enter training data.
    s.observe([None] + _designs(30, rng)
              + [CandidateDesign(variables={"a": .1, "b": .1, "m": "x"},
                                 fitness=float("-inf"))])
    assert all(np.isfinite(v) for v in s._y)

    # A constant fitness landscape has no gradient to screen on.
    _, s2 = _screener_fixtures()
    s2.observe([CandidateDesign(variables={"a": .5, "b": .5, "m": "x"}, fitness=1.0)
                for _ in range(40)])
    assert len(s2.screen([{"a": .1, "b": .1, "m": "x"}] * 20, 5, 0)) == 5
    assert s2.final_score() is None


@pytest.mark.optimization
def test_null_screener_is_transparent():
    from core.optimization_engine import OptimizationConfig, _make_screener
    dvs, _ = _screener_fixtures()
    s = _make_screener(OptimizationConfig(use_surrogate=False), dvs,
                       np.random.default_rng(0))
    assert s.pool_size(7) == 7
    assert s.screen(list(range(20)), 7) == list(range(7))
    assert s.final_score() is None
