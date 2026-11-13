"""
K2 AeroSim — Rules checker
============================
Check a flight against a competition's rules: rail-exit speed, stability,
descent rates, drift, apogee accuracy...

A rule names one quantity (see QUANTITIES), a comparison and a limit, all in
SI units, plus the text it came from. A rule set is a named list of rules
with the document it was taken from.

**Limits are never invented here.** Rule numbers change every year and
differ between competitions, so built-in rule sets (``data/rules/*.json``)
either cite the organiser's document and year and are marked confirmed, or
carry no limits at all and say so. A rule without a limit is reported as
"not evaluated", never as a pass.

Evaluated against one FlightSummary (core.flight_summary) or a list of them
(a Monte Carlo run), which gives a pass rate per rule.

No Qt imports.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("K2.Rules")

BUILTIN_DIR = Path(__file__).resolve().parents[1] / "data" / "rules"

PASS, FAIL, NOT_EVALUATED, UNKNOWN = "pass", "fail", "not evaluated", "unknown quantity"
COMPARISONS = (">=", "<=", "between")


@dataclass
class Context:
    """What a rule may need besides the flight itself."""
    target_apogee: float = 0.0                    # m above the pad; 0 = none declared
    liftoff_thrust_to_weight: Optional[float] = None


@dataclass
class Quantity:
    label: str
    unit: str                                     # SI unit of the value and limits
    get: Callable                                 # (summary, context) -> float | None
    help: str = ""


def _attr(name):
    return lambda s, c: getattr(s, name, None)


def _apogee_error(s, c):
    if not c.target_apogee or s.apogee is None:
        return None
    return abs(s.apogee - c.target_apogee) / c.target_apogee * 100.0


def _accel_g(s, c):
    return s.max_acceleration / 9.80665 if s.max_acceleration is not None else None


def _main_canopy_rate(s, c):
    """Descent rate under the last canopy to open (the main, on a dual
    deploy): what that canopy measured, or the touchdown rate."""
    opened = sorted((x for x in (s.canopies or []) if x.get("deploy_time") is not None),
                    key=lambda x: x["deploy_time"])
    if opened and opened[-1].get("descent_rate") is not None:
        return opened[-1]["descent_rate"]
    return s.landing_velocity


QUANTITIES: Dict[str, Quantity] = {
    "apogee": Quantity("Apogee", "m", _attr("apogee")),
    "apogee_error_pct": Quantity("Apogee error vs target", "%", _apogee_error,
                                 "Needs a declared target apogee."),
    "rail_exit_velocity": Quantity("Rail exit velocity", "m/s", _attr("rail_exit_velocity")),
    "stability_at_rail_exit": Quantity("Stability at rail exit", "cal",
                                       _attr("stability_at_rail_exit")),
    "min_stability": Quantity("Minimum stability (rail exit → apogee)", "cal",
                              _attr("min_stability")),
    "max_mach": Quantity("Max Mach", "", _attr("max_mach")),
    "max_acceleration_g": Quantity("Max sensed acceleration", "g", _accel_g),
    "max_q": Quantity("Max dynamic pressure", "Pa", _attr("max_q")),
    "drogue_descent_rate": Quantity("Descent rate under drogue", "m/s",
                                    _attr("drogue_descent_rate")),
    "main_descent_rate": Quantity("Descent rate under main", "m/s", _main_canopy_rate),
    "landing_velocity": Quantity("Touchdown speed", "m/s", _attr("landing_velocity")),
    "landing_drift": Quantity("Landing distance from pad", "m", _attr("landing_drift")),
    "descent_time": Quantity("Apogee to touchdown", "s", _attr("descent_time")),
    "flight_time": Quantity("Flight time", "s", _attr("flight_time")),
    "liftoff_thrust_to_weight": Quantity(
        "Liftoff thrust-to-weight", "",
        lambda s, c: c.liftoff_thrust_to_weight,
        "Average motor thrust / liftoff weight."),
}


@dataclass
class Rule:
    name: str
    quantity: str
    comparison: str = ">="
    limit: Optional[float] = None
    limit_high: Optional[float] = None            # "between" only
    citation: str = ""                            # rulebook section / wording

    def problems(self) -> List[str]:
        out = []
        if self.quantity not in QUANTITIES:
            out.append(f"unknown quantity {self.quantity!r}")
        if self.comparison not in COMPARISONS:
            out.append(f"unknown comparison {self.comparison!r}")
        for v in (self.limit, self.limit_high):
            if v is not None and not (isinstance(v, (int, float)) and math.isfinite(v)):
                out.append("limit is not a number")
        if self.comparison == "between" and self.limit is not None and \
                self.limit_high is not None and self.limit_high < self.limit:
            out.append("upper limit is below the lower one")
        return out


@dataclass
class RuleSet:
    name: str
    source: str = ""          # document and year the limits were taken from
    confirmed: bool = False   # limits checked against that document by a person
    notes: str = ""
    rules: List[Rule] = field(default_factory=list)
    builtin: bool = False

    def to_json(self) -> str:
        d = asdict(self)
        d.pop("builtin", None)
        return json.dumps(d, indent=1, ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict, builtin: bool = False) -> "RuleSet":
        rules = []
        for r in d.get("rules", []):
            rule = Rule(name=str(r.get("name", "")), quantity=str(r.get("quantity", "")),
                        comparison=str(r.get("comparison", ">=")),
                        limit=_num(r.get("limit")), limit_high=_num(r.get("limit_high")),
                        citation=str(r.get("citation", "")))
            rules.append(rule)
        return cls(name=str(d.get("name", "Rules")), source=str(d.get("source", "")),
                   confirmed=bool(d.get("confirmed", False)),
                   notes=str(d.get("notes", "")), rules=rules, builtin=builtin)


def _num(v):
    if v is None or v == "":
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


@dataclass
class RuleResult:
    rule: Rule
    status: str
    value: Optional[float] = None
    margin: Optional[float] = None    # distance to the limit, + on the passing side
    pass_rate: Optional[float] = None  # 0..1 over a set of flights
    evaluated: int = 0                 # flights the rule could be evaluated on
    note: str = ""


def _check(rule: Rule, value: float):
    """(passed, margin) for one value."""
    if rule.comparison == ">=":
        return value >= rule.limit, value - rule.limit
    if rule.comparison == "<=":
        return value <= rule.limit, rule.limit - value
    lo, hi = rule.limit, rule.limit_high
    return lo <= value <= hi, min(value - lo, hi - value)


def evaluate(rule: Rule, summaries, context: Context = None) -> RuleResult:
    """One rule against one flight summary, or a list of them."""
    context = context or Context()
    many = isinstance(summaries, (list, tuple))
    flights = list(summaries) if many else [summaries]
    problems = rule.problems()
    if problems:
        status = UNKNOWN if rule.quantity not in QUANTITIES else NOT_EVALUATED
        return RuleResult(rule, status, note="; ".join(problems))
    if rule.limit is None or (rule.comparison == "between" and rule.limit_high is None):
        return RuleResult(rule, NOT_EVALUATED, note="no limit entered")
    q = QUANTITIES[rule.quantity]
    values = [q.get(s, context) for s in flights]
    checked = [(v, _check(rule, v)) for v in values if v is not None and math.isfinite(v)]
    if not checked:
        why = q.help or "the flight did not produce this quantity"
        return RuleResult(rule, NOT_EVALUATED, note=why)
    passes = sum(1 for _, (ok, _) in checked if ok)
    worst_v, (worst_ok, worst_margin) = min(checked, key=lambda x: x[1][1])
    status = PASS if passes == len(checked) else FAIL
    return RuleResult(rule, status, value=worst_v if many else checked[0][0],
                      margin=worst_margin, pass_rate=passes / len(checked),
                      evaluated=len(checked),
                      note="" if len(checked) == len(flights) else
                      f"{len(flights) - len(checked)} flight(s) could not be evaluated")


def evaluate_all(rule_set: RuleSet, summaries, context: Context = None) -> List[RuleResult]:
    return [evaluate(r, summaries, context) for r in rule_set.rules]


# ── Rule-set library ───────────────────────────────────────────────────────

def builtin_rule_sets(folder: Path = BUILTIN_DIR) -> List[RuleSet]:
    out = []
    for p in sorted(folder.glob("*.json")):
        try:
            out.append(RuleSet.from_dict(json.loads(p.read_text(encoding="utf-8")),
                                         builtin=True))
        except (OSError, ValueError, AttributeError) as exc:
            logger.warning("Rule set %s not loaded: %s", p.name, exc)
    return out


def parse_user_rule_sets(text: str) -> List[RuleSet]:
    try:
        raw = json.loads(text) if text else []
    except ValueError:
        return []
    return [RuleSet.from_dict(d) for d in raw if isinstance(d, dict)]


def dump_user_rule_sets(sets) -> str:
    return json.dumps([json.loads(s.to_json()) for s in sets if not s.builtin],
                      indent=1, ensure_ascii=False)


def context_from_state(state, target_apogee: float = 0.0) -> Context:
    from core.constants import G_EARTH
    mass = state.total_mass() if callable(getattr(state, "total_mass", None)) else 0.0
    twr = None
    if mass > 0 and getattr(state, "motor_avg_thrust", 0.0) > 0:
        twr = state.motor_avg_thrust / (mass * G_EARTH)
    return Context(target_apogee=target_apogee, liftoff_thrust_to_weight=twr)
