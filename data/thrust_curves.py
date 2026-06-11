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
CACHE_DIR = Path(__file__).parent / "thrust_curves"


def load_cached(motor_id: str):
    """Return the cached curve [(t, N), ...] or None if not cached."""
    if not motor_id:
        return None
    f = CACHE_DIR / f"{motor_id}.json"
    if not f.is_file():
        return None
    try:
        curve = json.loads(f.read_text(encoding="utf-8"))
        return [(float(t), float(v)) for t, v in curve] or None
    except Exception as exc:
        logger.warning(f"Bad curve cache for {motor_id}: {exc}")
        return None


def _pick_result(results: list) -> list:
    """Choose the best data file: prefer cert data, then most samples."""
    def rank(r):
        return (r.get("source") == "cert", len(r.get("samples") or []))
    best = max(results, key=rank)
    return best.get("samples") or []


def fetch_thrust_curve(motor_id: str, timeout: float = 15.0):
    """Fetch the measured thrust curve for a motor_id.

    Returns [(time_s, thrust_N), ...] or None on any failure. Successful
    fetches are cached; failures are not (so a later retry can succeed).
    """
    if not motor_id:
        return None
    cached = load_cached(motor_id)
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

    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        (CACHE_DIR / f"{motor_id}.json").write_text(
            json.dumps(curve), encoding="utf-8")
    except OSError as exc:
        logger.warning(f"Could not cache curve for {motor_id}: {exc}")
    return curve
