"""
K2 AeroSim — Landing map
==========================
Landing points on the ground: from the flight frame (metres north and east of
the pad) to latitude/longitude, a dispersion ellipse fitted to a Monte Carlo
run, and KML / CSV export for recovery teams.

Frame: the flight engines use +X = north, +Y = east (see
environment.wind_model), with the pad at the origin.

Geodesy: offsets are converted with the WGS-84 radii of curvature at the
pad's latitude. That is a local tangent-plane approximation; over the tens
of kilometres a recovery area spans the position error is metres, well
inside the dispersion itself. It is not a long-range ballistic model.

No Qt imports.
"""
from __future__ import annotations

import csv
import io
import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple
from xml.sax.saxutils import escape

WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3


def radii_of_curvature(lat_deg: float) -> Tuple[float, float]:
    """(meridional M, prime-vertical N) radii in metres at a latitude."""
    s = math.sin(math.radians(lat_deg))
    w = 1.0 - WGS84_E2 * s * s
    return WGS84_A * (1.0 - WGS84_E2) / w ** 1.5, WGS84_A / math.sqrt(w)


def offset_to_latlon(north: float, east: float, lat0: float, lon0: float) -> Tuple[float, float]:
    """Pad-relative offset (m north, m east) → (latitude, longitude) degrees."""
    m, n = radii_of_curvature(lat0)
    lat = lat0 + math.degrees(north / m)
    coslat = math.cos(math.radians(lat0))
    lon = lon0 + (math.degrees(east / (n * coslat)) if abs(coslat) > 1e-9 else 0.0)
    lon = (lon + 180.0) % 360.0 - 180.0
    return lat, lon


def latlon_to_offset(lat: float, lon: float, lat0: float, lon0: float) -> Tuple[float, float]:
    """Inverse of offset_to_latlon."""
    m, n = radii_of_curvature(lat0)
    dlon = (lon - lon0 + 180.0) % 360.0 - 180.0
    return (math.radians(lat - lat0) * m,
            math.radians(dlon) * n * math.cos(math.radians(lat0)))


# ── Dispersion ellipse ────────────────────────────────────────────────────

# Fraction of a 2-D normal inside its k-sigma ellipse: 1 - exp(-k²/2).
def ellipse_coverage(k: float) -> float:
    return 1.0 - math.exp(-0.5 * k * k)


@dataclass
class Dispersion:
    """A bivariate-normal fit to landing points (north, east)."""
    n: int
    mean_north: float
    mean_east: float
    cov: Tuple[Tuple[float, float], Tuple[float, float]]
    semi_major_1sigma: float          # m
    semi_minor_1sigma: float          # m
    major_bearing: float              # deg, compass bearing of the major axis (0..180)

    def axes(self, k: float) -> Tuple[float, float]:
        return self.semi_major_1sigma * k, self.semi_minor_1sigma * k

    def polygon(self, k: float, points: int = 72) -> List[Tuple[float, float]]:
        """The k-sigma ellipse as (north, east) points, closed."""
        a, b = self.axes(k)
        th = math.radians(self.major_bearing)
        # Major axis along bearing th: unit vector (north=cos, east=sin).
        un, ue = math.cos(th), math.sin(th)
        vn, ve = -math.sin(th), math.cos(th)
        out = []
        for i in range(points + 1):
            t = 2.0 * math.pi * i / points
            ca, sb = a * math.cos(t), b * math.sin(t)
            out.append((self.mean_north + ca * un + sb * vn,
                        self.mean_east + ca * ue + sb * ve))
        return out

    def contains(self, north: float, east: float, k: float) -> bool:
        """Is (north, east) inside the k-sigma ellipse (Mahalanobis <= k)?"""
        (a, b), (_, d) = self.cov
        det = a * d - b * b
        if det <= 0:
            return False
        dn, de = north - self.mean_north, east - self.mean_east
        m2 = (d * dn * dn - 2 * b * dn * de + a * de * de) / det
        return m2 <= k * k


def fit_dispersion(points: Sequence[Tuple[float, float]]) -> Optional[Dispersion]:
    """Fit a dispersion ellipse to (north, east) landing points. Needs 3+."""
    pts = [(float(n), float(e)) for n, e in points
           if math.isfinite(n) and math.isfinite(e)]
    if len(pts) < 3:
        return None
    k = len(pts)
    mn = sum(p[0] for p in pts) / k
    me = sum(p[1] for p in pts) / k
    snn = sum((p[0] - mn) ** 2 for p in pts) / (k - 1)
    see = sum((p[1] - me) ** 2 for p in pts) / (k - 1)
    sne = sum((p[0] - mn) * (p[1] - me) for p in pts) / (k - 1)
    tr, det = snn + see, snn * see - sne * sne
    disc = math.sqrt(max(0.0, tr * tr / 4.0 - det))
    l1, l2 = tr / 2.0 + disc, max(0.0, tr / 2.0 - disc)
    # Eigenvector of l1 in (north, east) → compass bearing of the major axis.
    if abs(sne) > 1e-12:
        vn, ve = l1 - see, sne
    elif snn >= see:
        vn, ve = 1.0, 0.0
    else:
        vn, ve = 0.0, 1.0
    bearing = math.degrees(math.atan2(ve, vn)) % 180.0
    return Dispersion(k, mn, me, ((snn, sne), (sne, see)),
                      math.sqrt(l1), math.sqrt(l2), bearing)


# ── Export ────────────────────────────────────────────────────────────────

@dataclass
class LandingSet:
    """What a landing map shows."""
    site_name: str
    lat0: float
    lon0: float
    pad_altitude: float = 0.0                         # m above sea level
    landings: List[Tuple[float, float]] = field(default_factory=list)   # (north, east)
    nominal_track: List[Tuple[float, float, float]] = field(default_factory=list)  # (n, e, h above pad)
    sigmas: Tuple[float, ...] = (1.0, 2.0, 3.0)


def _kml_coord(lat0, lon0, north, east, h=0.0):
    lat, lon = offset_to_latlon(north, east, lat0, lon0)
    return f"{lon:.7f},{lat:.7f},{h:.1f}"


def to_kml(ls: LandingSet) -> str:
    """Pad, nominal track, every landing point and the k-sigma ellipses."""
    d = fit_dispersion(ls.landings)
    parts = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>',
             f"<name>{escape(ls.site_name or 'K2 landing map')}</name>",
             '<Style id="pad"><IconStyle><scale>1.2</scale></IconStyle></Style>',
             '<Style id="land"><IconStyle><scale>0.5</scale></IconStyle></Style>',
             '<Style id="ell"><LineStyle><color>ff0000ff</color><width>2</width>'
             '</LineStyle><PolyStyle><fill>0</fill></PolyStyle></Style>',
             '<Style id="trk"><LineStyle><color>ff00ffff</color><width>3</width>'
             '</LineStyle></Style>',
             f'<Placemark><name>Pad</name><styleUrl>#pad</styleUrl><Point>'
             f'<coordinates>{_kml_coord(ls.lat0, ls.lon0, 0, 0)}</coordinates>'
             f'</Point></Placemark>']
    if ls.nominal_track:
        coords = " ".join(_kml_coord(ls.lat0, ls.lon0, n, e, h) for n, e, h in ls.nominal_track)
        parts.append('<Placemark><name>Nominal flight</name><styleUrl>#trk</styleUrl>'
                     '<LineString><altitudeMode>relativeToGround</altitudeMode>'
                     f'<coordinates>{coords}</coordinates></LineString></Placemark>')
    if ls.landings:
        parts.append("<Folder><name>Landing points</name>")
        for i, (n, e) in enumerate(ls.landings, 1):
            parts.append(f'<Placemark><name>{i}</name><styleUrl>#land</styleUrl>'
                         f'<Point><coordinates>{_kml_coord(ls.lat0, ls.lon0, n, e)}'
                         f'</coordinates></Point></Placemark>')
        parts.append("</Folder>")
    if d is not None:
        parts.append("<Folder><name>Dispersion ellipses</name>")
        for k in ls.sigmas:
            ring = " ".join(_kml_coord(ls.lat0, ls.lon0, n, e) for n, e in d.polygon(k))
            parts.append(f'<Placemark><name>{k:g}σ ({ellipse_coverage(k) * 100:.0f} %)</name>'
                         '<styleUrl>#ell</styleUrl><Polygon><outerBoundaryIs><LinearRing>'
                         f'<coordinates>{ring}</coordinates></LinearRing></outerBoundaryIs>'
                         '</Polygon></Placemark>')
        parts.append("</Folder>")
    parts.append("</Document></kml>")
    return "\n".join(parts)


def to_csv(ls: LandingSet) -> str:
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\n")
    w.writerow(["run", "north_m", "east_m", "distance_m", "bearing_deg",
                "latitude", "longitude"])
    for i, (n, e) in enumerate(ls.landings, 1):
        lat, lon = offset_to_latlon(n, e, ls.lat0, ls.lon0)
        w.writerow([i, f"{n:.2f}", f"{e:.2f}", f"{math.hypot(n, e):.2f}",
                    f"{math.degrees(math.atan2(e, n)) % 360.0:.1f}",
                    f"{lat:.7f}", f"{lon:.7f}"])
    return out.getvalue()
