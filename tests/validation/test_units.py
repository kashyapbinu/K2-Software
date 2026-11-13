"""
Display units (ui.units): SI inside, SI or imperial at the edge, one table.
"""
import pytest

from ui import units as U


@pytest.mark.parametrize("kind", list(U.KINDS))
@pytest.mark.parametrize("system", U.SYSTEMS)
def test_format_parse_round_trip(kind, system):
    for si in (0.0, 1.0, 123.456, 3048.0):
        text = U.format_value(si, kind, system, decimals=6)
        # Six display decimals: the round trip is good to 1e-6 display units.
        step = U.KINDS[kind][system][1] * 1e-6
        assert U.parse_value(text, kind, system) == pytest.approx(si, abs=step)


def test_competition_numbers():
    assert U.format_value(3048.0, "altitude", "imperial") == "10,000 ft"
    assert U.format_value(3048.0, "altitude", "SI") == "3,048 m"
    assert U.format_value(30.48, "velocity", "imperial") == "100 ft/s"
    assert U.format_value(4.4482216152605, "force", "imperial") == "1.0 lbf"
    assert U.format_value(0.45359237, "mass", "imperial") == "1.00 lb"
    assert U.format_value(None, "mass") == "—"
    assert U.to_si(10000, "altitude", "imperial") == pytest.approx(3048.0)


def test_parse_refuses_text_that_is_not_a_number():
    with pytest.raises(ValueError):
        U.parse_value("ten thousand", "altitude", "imperial")


def test_saved_numbers_do_not_depend_on_the_setting():
    """Units are a display matter: the project file is SI either way."""
    from core.rocket_state import RocketState
    from validation.cases.rocket_canonical import canonical_state
    a = canonical_state().to_dict()
    # Nothing in RocketState knows about display units.
    assert not any("unit" in k for k in a)
    assert RocketState.from_dict(a).to_dict() == a
