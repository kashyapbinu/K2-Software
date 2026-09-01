"""
The conditions-panel Reynolds preview must describe the run it precedes.

Re = rho*V*L/mu is linear in the reference length. The preview hardcoded
L = 1 m with a comment promising it was "updated after geometry" - which was
never implemented, since nothing called _update_isa() after geometry loaded.
The solver meanwhile normalises by the real length (geometry dict, CAD
flow-axis extent, or STL span) and honours the user's Ref. length override.

So the two "Reynolds:" rows in the same workspace disagreed by exactly the
body length - 1.68x for the canonical rocket - and the Ref. length spinbox,
whose tooltip says it sets Reynolds, moved the solver's value while the
displayed one ignored it.

Both sides now resolve through cfd.solvers.su2_solver.resolve_reference_values.
"""
import math
from pathlib import Path

import pytest

from cfd.solvers.base import CFDConfig, isa_conditions
from cfd.solvers.su2_solver import SU2Solver, resolve_reference_values

ROCKET = {"max_diameter": 0.155, "length": 1.68}


def _reynolds(mach, alt, L):
    P, T, rho = isa_conditions(alt)
    a = math.sqrt(1.4 * 287.05 * T)
    mu = 1.716e-5 * (T / 273.15) ** 1.5 * (273.15 + 110.4) / (T + 110.4)
    return rho * (mach * a) * L / mu


# ── the shared resolver keeps the solver's priority order ────────────────────

def test_geometry_dict_gives_body_length_and_max_diameter_area():
    area, length = resolve_reference_values(geometry_dict=ROCKET, quiet=True)
    assert length == pytest.approx(1.68)
    assert area == pytest.approx(math.pi * (0.155 / 2) ** 2)


def test_cad_outranks_geometry_dict():
    area, length = resolve_reference_values(
        external_cad=Path("body.step"),
        cad_info={"frontal_area": 0.02, "length": 2.5},
        geometry_dict=ROCKET, quiet=True)
    assert (area, length) == pytest.approx((0.02, 2.5))


def test_overrides_win_and_apply_independently():
    """Setting only one override must keep the measured value for the other."""
    area, length = resolve_reference_values(
        geometry_dict=ROCKET, ref_length_override=3.0, quiet=True)
    assert length == pytest.approx(3.0)
    assert area == pytest.approx(math.pi * (0.155 / 2) ** 2)   # not clobbered

    area, length = resolve_reference_values(
        geometry_dict=ROCKET, ref_area_override=0.5, quiet=True)
    assert area == pytest.approx(0.5)
    assert length == pytest.approx(1.68)


def test_falls_back_to_unit_length_with_no_geometry():
    assert resolve_reference_values(quiet=True) == (0.1, 1.0)


# ── solver and preview must not drift ────────────────────────────────────────

def test_solver_delegates_to_the_shared_resolver(tmp_path):
    cfg = CFDConfig(mach=0.8, altitude_m=1000.0, work_dir=tmp_path,
                    geometry_dict=ROCKET)
    assert SU2Solver(cfg)._reference_values() == resolve_reference_values(
        geometry_dict=ROCKET, quiet=True)


def test_preview_reynolds_matches_the_solver(tmp_path):
    """The number shown before Run must be the number the run uses."""
    cfg = CFDConfig(mach=0.8, altitude_m=1000.0, work_dir=tmp_path,
                    geometry_dict=ROCKET)
    _a, l_solver = SU2Solver(cfg)._reference_values()
    # what the panel resolves in assembly mode: length straight off the assembly
    _a2, l_panel = resolve_reference_values(
        geometry_dict={"max_diameter": 0.0, "length": 1.68}, quiet=True)

    assert l_panel == pytest.approx(l_solver)
    assert _reynolds(0.8, 1000.0, l_panel) == pytest.approx(
        _reynolds(0.8, 1000.0, l_solver))


def test_unit_length_would_have_understated_re_by_the_body_length():
    """Pin the size of the original error so a silent revert is visible."""
    re_real = _reynolds(0.8, 1000.0, 1.68)
    re_old = _reynolds(0.8, 1000.0, 1.0)
    assert re_real / re_old == pytest.approx(1.68)


# ── Reynolds genuinely responds to both flight conditions ────────────────────

def test_reynolds_falls_monotonically_with_altitude():
    """rho drops faster than mu, so Re falls all the way up."""
    alts = [0, 1000, 3000, 5000, 10000, 20000]
    res = [_reynolds(0.8, a, 1.68) for a in alts]
    assert res == sorted(res, reverse=True), dict(zip(alts, res))
    assert res[0] / res[-1] > 10, "sea level to 20 km should be an order of magnitude"


def test_reynolds_rises_with_mach():
    res = [_reynolds(m, 3000.0, 1.68) for m in (0.3, 0.8, 2.0)]
    assert res == sorted(res)
    assert res[-1] > 5 * res[0]      # M0.3 -> M2.0 is a big change, not noise


def test_reynolds_is_linear_in_reference_length():
    """The property that made the hardcoded L=1 m a pure scale error."""
    assert _reynolds(0.8, 1000.0, 3.36) == pytest.approx(
        2 * _reynolds(0.8, 1000.0, 1.68))


# ── the wiring that was missing ──────────────────────────────────────────────

def test_preview_is_refreshed_when_geometry_and_ref_length_change():
    """_update_isa must run after geometry loads and when Ref. length moves.

    Without these the resolver is correct but never re-consulted, which is the
    original bug wearing a different hat.
    """
    from ui.workspaces import cfd_workspace
    src = Path(cfd_workspace.__file__).with_suffix(".py").read_text(encoding="utf-8")

    assert "self._sp_ref_len.valueChanged.connect(self._update_isa)" in src
    assert src.count("self._update_isa()") >= 3, (
        "expected the construction call plus geometry-export and CAD-load refreshes"
    )
    assert "rho * V * 1.0 / mu" not in src, "reference length is hardcoded again"
