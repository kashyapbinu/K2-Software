"""
Fetch the full motor catalog from ThrustCurve.org and write data/motors.json.

ThrustCurve API: https://www.thrustcurve.org/info/api.html
We query per impulse class (A..O) because the search endpoint caps results per
call, then normalize every record into the schema the K2 propulsion workspace
expects.

Usage:
    python data/fetch_motors.py
    python data/fetch_motors.py --available-only   # drop OOP (out of production)

No third-party deps -- uses urllib from the stdlib.
"""
import json
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

API = "https://www.thrustcurve.org/api/v1/search.json"
CLASSES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
           "K", "L", "M", "N", "O"]
OUT = Path(__file__).parent / "motors.json"


def _post(body):
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        API, data=data,
        headers={"Content-Type": "application/json",
                 "User-Agent": "K2-Software/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def fetch_class(cls):
    """Return all results for one impulse class."""
    body = {"impulseClass": cls, "maxResults": 2000}
    try:
        resp = _post(body)
    except urllib.error.URLError as e:
        print(f"  class {cls}: request failed: {e}", file=sys.stderr)
        return []
    results = resp.get("results", [])
    print(f"  class {cls}: {len(results)} motors")
    return results


def normalize(m):
    """Map a ThrustCurve record onto the K2 motor schema."""
    avg = m.get("avgThrustN") or 0.0
    # search endpoint has no max thrust; estimate from avg if absent
    mx = m.get("maxThrustN") or (avg * 1.4 if avg else 0.0)
    prop_g = m.get("propWeightG") or 0.0
    tot_g = m.get("totalWeightG")
    if not tot_g:
        # rough: casing+nozzle ~= propellant mass again, plus a bit
        tot_g = prop_g * 2.0 if prop_g else 0.0
    return {
        "designation": m.get("commonName") or m.get("designation") or "?",
        "full_designation": m.get("designation") or "",
        "manufacturer": m.get("manufacturerAbbrev") or m.get("manufacturer") or "?",
        "class": m.get("impulseClass") or "",
        "total_impulse": round(float(m.get("totImpulseNs") or 0.0), 2),
        "avg_thrust": round(float(avg), 2),
        "max_thrust": round(float(mx), 2),
        "burn_time": round(float(m.get("burnTimeS") or 0.0), 3),
        "propellant_mass": round(prop_g / 1000.0, 5),
        "total_mass": round(tot_g / 1000.0, 5),
        "diameter": round(float(m.get("diameter") or 0) / 1000.0, 4),  # mm -> m
        "length": round(float(m.get("length") or 0) / 1000.0, 4),      # mm -> m
        "type": m.get("type") or "",
        "availability": m.get("availability") or "",
        "motor_id": m.get("motorId") or "",
    }


def main(available_only=False):
    print(f"Fetching motor catalog from {API} ...")
    seen = set()
    motors = []
    for cls in CLASSES:
        for m in fetch_class(cls):
            n = normalize(m)
            if n["total_impulse"] <= 0 or n["burn_time"] <= 0:
                continue  # skip records with no usable performance data
            if available_only and n["availability"] == "OOP":
                continue
            key = (n["manufacturer"], n["full_designation"] or n["designation"])
            if key in seen:
                continue
            seen.add(key)
            motors.append(n)
        time.sleep(0.3)  # be polite to the API

    cls_order = {c: i for i, c in enumerate(CLASSES)}
    motors.sort(key=lambda x: (cls_order.get(x["class"], 99),
                               x["diameter"], x["total_impulse"]))

    OUT.write_text(json.dumps(motors, indent=2), encoding="utf-8")
    print(f"\nWrote {len(motors)} motors -> {OUT}")


if __name__ == "__main__":
    main(available_only="--available-only" in sys.argv)
