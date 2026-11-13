"""
Landing map (core.landing_map): pad-relative landing points to latitude /
longitude, a dispersion ellipse, and KML / CSV for the recovery team.
"""
import math
import random
import xml.etree.ElementTree as ET

import pytest

from core import landing_map as L


def test_known_distances_on_the_ellipsoid():
    # One degree of latitude is ~110.57 km at the equator, ~111.69 km at the
    # pole; one degree of longitude is ~111.32 km at the equator.
    m0, n0 = L.radii_of_curvature(0.0)
    assert math.radians(1) * m0 == pytest.approx(110574, rel=1e-4)
    assert math.radians(1) * n0 == pytest.approx(111320, rel=1e-4)
    lat, lon = L.offset_to_latlon(1000.0, 0.0, 32.99, -106.97)
    assert lat - 32.99 == pytest.approx(1000.0 / 110900.0, rel=0.01)
    assert lon == pytest.approx(-106.97)


@pytest.mark.parametrize("lat0,lon0", [(32.99, -106.97), (39.2, -8.0), (-33.9, 151.2),
                                       (0.0, 179.99)])
def test_round_trip(lat0, lon0):
    for n, e in ((0, 0), (1500, -2300), (-12000, 8000)):
        lat, lon = L.offset_to_latlon(n, e, lat0, lon0)
        n2, e2 = L.latlon_to_offset(lat, lon, lat0, lon0)
        assert (n2, e2) == pytest.approx((n, e), abs=1e-6)


def _cloud(n=4000, sn=300.0, se=100.0, bearing=30.0, mn=500.0, me=-200.0, seed=3):
    rng = random.Random(seed)
    th = math.radians(bearing)
    pts = []
    for _ in range(n):
        a, b = rng.gauss(0, sn), rng.gauss(0, se)
        pts.append((mn + a * math.cos(th) - b * math.sin(th),
                    me + a * math.sin(th) + b * math.cos(th)))
    return pts


def test_dispersion_ellipse_recovers_a_known_spread():
    pts = _cloud()
    d = L.fit_dispersion(pts)
    assert (d.mean_north, d.mean_east) == pytest.approx((500.0, -200.0), abs=15.0)
    assert d.semi_major_1sigma == pytest.approx(300.0, rel=0.05)
    assert d.semi_minor_1sigma == pytest.approx(100.0, rel=0.05)
    assert d.major_bearing == pytest.approx(30.0, abs=2.0)
    for k in (1.0, 2.0, 3.0):
        inside = sum(d.contains(n, e, k) for n, e in pts) / len(pts)
        assert inside == pytest.approx(L.ellipse_coverage(k), abs=0.02)
    poly = d.polygon(2.0)
    assert poly[0] == pytest.approx(poly[-1])
    assert all(d.contains(n, e, 2.0001) for n, e in poly)


def test_too_few_points_give_no_ellipse():
    assert L.fit_dispersion([(0, 0), (1, 1)]) is None
    assert L.fit_dispersion([(0, 0), (1, 1), (float("nan"), 2)]) is None


def test_kml_holds_the_pad_points_track_and_ellipses():
    ls = L.LandingSet("Pad A", 32.99, -106.97, 1401.0, landings=_cloud(50),
                      nominal_track=[(0, 0, 0), (10, 5, 1000), (40, 20, 2600)])
    root = ET.fromstring(L.to_kml(ls))
    ns = {"k": "http://www.opengis.net/kml/2.2"}
    names = [p.findtext("k:name", namespaces=ns) for p in root.iter("{%s}Placemark" % ns["k"])]
    assert "Pad" in names and "Nominal flight" in names
    assert sum(n.isdigit() for n in names) == 50
    assert [n for n in names if "σ" in n] == ["1σ (39 %)", "2σ (86 %)", "3σ (99 %)"]
    pad = next(p for p in root.iter("{%s}Placemark" % ns["k"])
               if p.findtext("k:name", namespaces=ns) == "Pad")
    lon, lat, _ = pad.find(".//k:coordinates", ns).text.split(",")
    assert (float(lat), float(lon)) == pytest.approx((32.99, -106.97))


def test_monte_carlo_helpers_read_north_then_east():
    """landing_x is NORTH in the flight frame; the MC plot had it on the
    East axis."""
    import types
    from core.batch_simulation import BatchSimResult
    from ui.workspaces.monte_carlo_workspace import (landing_dispersion,
                                                     landing_points, run_summaries)
    ok = BatchSimResult(apogee=900.0, final_phase="Landed", landing_x=300.0, landing_y=-40.0)
    bad = BatchSimResult(success=False, failure_reasons=["Invalid configuration: x"])
    r = types.SimpleNamespace(landing_x_values=[300.0, 310.0, float("nan"), 290.0],
                              landing_y_values=[-40.0, -35.0, 0.0, -50.0],
                              runs=[ok, bad])
    assert landing_points(r) == [(300.0, -40.0), (310.0, -35.0), (290.0, -50.0)]
    d = landing_dispersion(r)
    assert d.mean_north == pytest.approx(300.0) and d.mean_east == pytest.approx(-41.667, abs=1e-3)
    sums = run_summaries(r)
    assert len(sums) == 1 and sums[0].apogee == 900.0


def test_monte_carlo_tab_exports_kml_and_csv(tmp_path):
    import os
    import subprocess
    import sys
    import textwrap
    from pathlib import Path

    script = textwrap.dedent(f"""
        import sys, types
        sys.path.insert(0, {str(Path(__file__).resolve().parents[2])!r})
        from PyQt6.QtWidgets import QApplication
        app = QApplication([])
        from core.rocket_state import RocketStateEngine
        from core.simulation_engine import SimulationEngine
        from ui.workspaces.monte_carlo_workspace import MonteCarloWorkspace
        from validation.cases.rocket_canonical import canonical_state
        s = canonical_state()
        s.launch_latitude, s.launch_longitude, s.launch_site_name = 32.99, -106.97, "Pad"
        eng = RocketStateEngine(s)
        ws = MonteCarloWorkspace(eng, SimulationEngine(eng))
        ws._results = types.SimpleNamespace(
            landing_x_values=[100.0, 140.0, 90.0, 120.0],
            landing_y_values=[10.0, -20.0, 5.0, 0.0], runs=[])
        ws._on_export_landing_map({str(tmp_path / 'm.kml')!r})
        ws._on_export_landing_map({str(tmp_path / 'm.csv')!r})
        print("OK")
    """)
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    proc = subprocess.run([sys.executable, "-c", script], env=env,
                          capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0 and "OK" in proc.stdout, (proc.stdout + proc.stderr)[-3000:]
    kml = (tmp_path / "m.kml").read_text(encoding="utf-8")
    assert "<name>Pad</name>" in kml and kml.count("<Point>") == 5     # pad + 4 landings
    assert (tmp_path / "m.csv").read_text().count("\n") == 5


def test_csv_rows():
    ls = L.LandingSet("x", 10.0, 20.0, landings=[(100.0, 0.0), (0.0, -50.0)])
    rows = L.to_csv(ls).strip().splitlines()
    assert rows[0].startswith("run,north_m,east_m")
    assert rows[1].split(",")[3:5] == ["100.00", "0.0"]
    assert rows[2].split(",")[4] == "270.0"
