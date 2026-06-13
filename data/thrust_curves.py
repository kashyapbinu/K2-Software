"""
Fetch real motor thrust-curve samples from ThrustCurve.org, with a local cache.

The motor catalog (motors.json, see fetch_motors.py) stores summary stats only;
the actual measured curve points come from the /api/v1/download.json endpoint
on demand. Curves are cached as data/thrust_curves/<motor_id>.json so each
motor hits the network at most once. Offline or on any failure the caller
falls back to the engine's impulse-normalized trapezoid.

No third-party deps — uses urllib from the stdlib.
"""
import json
import logging
import urllib.request
import urllib.error
from pathlib import Path

logger = logging.getLogger("K2.ThrustCurves")

API = "https://www.thrustcurve.org/api/v1/download.json"
# Per-user writable cache (the install dir is read-only in a frozen build).
# In a source run this resolves back to data/thrust_curves as before.
from core.paths import user_data_dir
CACHE_DIR = user_data_dir("data/thrust_curves")


def curve_impulse(curve) -> float:
    """Trapezoid-integrated total impulse (N·s) of a [(t, N), ...] curve."""
    return sum((t2 - t1) * (v1 + v2) / 2.0
               for (t1, v1), (t2, v2) in zip(curve, curve[1:]))


def _matches_catalog(curve, expected_impulse, motor_id) -> bool:
    """True if the curve's impulse is within 20% of the catalog value.

    ThrustCurve.org data files are user-submitted and occasionally belong to
    a different motor entirely (e.g. the RATT K600TR-P file integrates to
    ~614 N·s against a 2170 N·s catalog entry). Such a curve silently flies
    the sim on a much smaller motor, so reject it and let the caller fall
    back to the impulse-normalized trapezoid.
    """
    if not expected_impulse:
        return True
    imp = curve_impulse(curve)
    if abs(imp - expected_impulse) <= 0.2 * expected_impulse:
        return True
    logger.warning(
        f"Rejecting thrust curve for {motor_id}: integrates to {imp:.0f} N*s "
        f"but catalog says {expected_impulse:.0f} N*s")
    return False


def load_cached(motor_id: str, expected_impulse: float = 0.0):
    """Return the cached curve [(t, N), ...] or None if not cached.

    A cached curve whose impulse contradicts expected_impulse is treated as
    corrupt: the cache file is removed and None is returned.
    """
    if not motor_id:
        return None
    f = CACHE_DIR / f"{motor_id}.json"
    if not f.is_file():
        return None
    try:
        curve = json.loads(f.read_text(encoding="utf-8"))
        curve = [(float(t), float(v)) for t, v in curve] or None
    except Exception as exc:
        logger.warning(f"Bad curve cache for {motor_id}: {exc}")
        return None
    if curve and not _matches_catalog(curve, expected_impulse, motor_id):
        try:
            f.unlink()
        except OSError:
            pass
        return None
    return curve


def _pick_result(results: list) -> list:
    """Choose the best data file: prefer cert data, then most samples."""
    def rank(r):
        return (r.get("source") == "cert", len(r.get("samples") or []))
    best = max(results, key=rank)
    return best.get("samples") or []


def fetch_thrust_curve(motor_id: str, timeout: float = 15.0,
                       expected_impulse: float = 0.0):
    """Fetch the measured thrust curve for a motor_id.

    Returns [(time_s, thrust_N), ...] or None on any failure. Successful
    fetches are cached; failures are not (so a later retry can succeed).
    If expected_impulse is given, curves deviating more than 20% from it
    are rejected (and not cached).
    """
    if not motor_id:
        return None
    cached = load_cached(motor_id, expected_impulse)
    if cached is not None:
        return cached

    body = json.dumps({"motorIds": [motor_id], "data": "samples"}).encode()
    req = urllib.request.Request(
        API, data=body,
        headers={"Content-Type": "application/json",
                 "User-Agent": "K2-Software/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            resp = json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError) as exc:
        logger.info(f"Thrust-curve fetch failed for {motor_id}: {exc}")
        return None

    results = resp.get("results") or []
    if not results:
        return None
    samples = _pick_result(results)
    curve = [(float(s["time"]), float(s["thrust"])) for s in samples
             if s.get("time") is not None and s.get("thrust") is not None]
    if len(curve) < 3:
        return None
    if curve[0][0] > 0:                  # RASP files may omit the t=0 point
        curve.insert(0, (0.0, 0.0))
    if not _matches_catalog(curve, expected_impulse, motor_id):
        return None

    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        (CACHE_DIR / f"{motor_id}.json").write_text(
            json.dumps(curve), encoding="utf-8")
    except OSError as exc:
        logger.warning(f"Could not cache curve for {motor_id}: {exc}")
    return curve
