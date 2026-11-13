"""
K2 AeroSim — Launch sites
===========================
Named pad positions: elevation above sea level, latitude, longitude.

Two sources, one list:

- built-in sites, shipped in ``data/launch_sites.json`` (read-only in the
  installed app). Each carries a ``note`` saying where its numbers came from
  and how far to trust them. Keep that note honest: a pad 100 m off in
  elevation moves apogee measurably.
- the user's own sites, saved by the UI (as JSON text in the app settings).

A site sets the four launch-site fields of RocketState; it does not touch the
rail azimuth, which belongs to the launch, not the place.

No Qt imports.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

logger = logging.getLogger("K2.LaunchSites")

BUILTIN_PATH = Path(__file__).resolve().parents[1] / "data" / "launch_sites.json"


@dataclass
class LaunchSite:
    name: str
    altitude: float = 0.0        # pad elevation above sea level (m)
    latitude: float = 0.0        # deg, + north
    longitude: float = 0.0       # deg, + east
    note: str = ""
    builtin: bool = False

    def state_fields(self) -> dict:
        """The RocketState fields this site sets."""
        return {"launch_site_name": self.name,
                "launch_altitude": self.altitude,
                "launch_latitude": self.latitude,
                "launch_longitude": self.longitude}

    def problems(self) -> List[str]:
        """Why this site cannot be used; empty when it can."""
        out = []
        if not self.name.strip():
            out.append("a site needs a name")
        for label, value, lo, hi in (("elevation", self.altitude, -500.0, 9000.0),
                                     ("latitude", self.latitude, -90.0, 90.0),
                                     ("longitude", self.longitude, -180.0, 180.0)):
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                out.append(f"{label} is not a number")
            elif not lo <= value <= hi:
                out.append(f"{label} {value:g} is outside {lo:g}…{hi:g}")
        return out


def _from_dict(d: dict, builtin: bool) -> LaunchSite:
    return LaunchSite(
        name=str(d.get("name", "")).strip(),
        altitude=float(d.get("altitude", 0.0)),
        latitude=float(d.get("latitude", 0.0)),
        longitude=float(d.get("longitude", 0.0)),
        note=str(d.get("note", "")),
        builtin=builtin,
    )


def parse_sites(text: str, builtin: bool = False) -> List[LaunchSite]:
    """Sites from JSON text (a list of objects). Entries that are not valid
    sites are skipped with a warning; bad JSON gives an empty list."""
    try:
        raw = json.loads(text) if text else []
    except (TypeError, ValueError):
        logger.warning("Launch sites: not valid JSON, ignored")
        return []
    if not isinstance(raw, list):
        return []
    sites = []
    for entry in raw:
        try:
            site = _from_dict(entry, builtin)
        except (AttributeError, TypeError, ValueError):
            logger.warning("Launch sites: skipping malformed entry %r", entry)
            continue
        if site.problems():
            logger.warning("Launch sites: skipping %r (%s)", site.name,
                           "; ".join(site.problems()))
            continue
        sites.append(site)
    return sites


def dump_sites(sites) -> str:
    """User sites as JSON text (built-ins are never written back)."""
    rows = []
    for s in sites:
        if s.builtin:
            continue
        row = asdict(s)
        row.pop("builtin", None)
        rows.append(row)
    return json.dumps(rows, ensure_ascii=False, indent=1)


def builtin_sites(path: Path = BUILTIN_PATH) -> List[LaunchSite]:
    try:
        return parse_sites(path.read_text(encoding="utf-8"), builtin=True)
    except OSError:
        logger.warning("Launch sites: %s not found", path)
        return []


def all_sites(user_json: str = "") -> List[LaunchSite]:
    """Built-in sites, then the user's. A user site with a built-in's name
    replaces it."""
    user = parse_sites(user_json)
    names = {s.name for s in user}
    return [s for s in builtin_sites() if s.name not in names] + user


def with_site(user_json: str, site: LaunchSite) -> str:
    """The user's site list with *site* added (replacing one of the same
    name), as JSON text."""
    keep = [s for s in parse_sites(user_json) if s.name != site.name]
    return dump_sites(keep + [LaunchSite(**{**asdict(site), "builtin": False})])


def without_site(user_json: str, name: str) -> str:
    return dump_sites([s for s in parse_sites(user_json) if s.name != name])
