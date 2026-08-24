"""
Validation report generator.
============================

Runs the benchmark suites and renders a single markdown credibility report with
per-domain result tables and overlay plots (K2 vs reference), plus a JSON cache
of every Benchmark so the report can be regenerated — or the slow SU2/CalculiX
results reused — without re-solving.

Usage:
    python -m validation.report              # fast benchmarks only
    python -m validation.report --full       # include slow SU2/CalculiX/OpenRocket
    python -m validation.report --from-cache # re-render from the last JSON cache
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # headless rendering
import matplotlib.pyplot as plt

from validation.harness import Benchmark, save_benchmarks, load_benchmarks

REPORT_DIR = Path(__file__).resolve().parent / "report"
CACHE = REPORT_DIR / "benchmarks.json"
PLOTS = REPORT_DIR / "plots"


# ── collection ────────────────────────────────────────────────────────────────

def collect(full: bool) -> list:
    from validation.sim import benchmarks as sim_b
    from validation.cfd import benchmarks as cfd_b
    from validation.structures import benchmarks as str_b

    out = []
    out += sim_b.run_benchmarks(include_full=True)
    out += cfd_b.run_benchmarks(include_su2=full)
    out += str_b.run_benchmarks(include_ccx=full)
    if not full:
        # The OpenRocket bench is the only "slow" sim one; drop it in fast mode.
        out = [b for b in out if b.name != "Sim vs OpenRocket"]
    return out


# ── plotting ──────────────────────────────────────────────────────────────────

def _plot_curves(bm: Benchmark) -> list:
    """Render every curve attached to a benchmark; return relative PNG paths."""
    PLOTS.mkdir(parents=True, exist_ok=True)
    paths = []
    for key, c in (bm.curves or {}).items():
        x, k2 = c.get("x", []), c.get("k2", [])
        ref = c.get("ref", [])
        if not x or not k2:
            continue
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(x, k2, label="K2", lw=1.8)
        if ref:
            ax.plot(x, ref, "--", label="reference", lw=1.4)
        ax.set_xlabel(c.get("xlabel", "")); ax.set_ylabel(c.get("ylabel", ""))
        ax.set_title(f"{bm.name} — {key}")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        slug = f"{bm.domain}_{bm.name}_{key}".lower()
        for ch in r' /\:()–—°':
            slug = slug.replace(ch, "_")
        fname = PLOTS / f"{slug}.png"
        fig.savefig(fname, dpi=110); plt.close(fig)
        paths.append(f"plots/{fname.name}")
    return paths


# ── markdown ──────────────────────────────────────────────────────────────────

_STATUS = {True: "PASS", False: "FAIL"}


def _bench_md(bm: Benchmark, plot_paths: list) -> str:
    if bm.skipped:
        return (f"### {bm.name}  *(skipped)*\n\n"
                f"_Reference:_ {bm.reference}  \n"
                f"_Reason:_ {bm.skip_reason}\n")
    head = _STATUS[bm.passed]
    lines = [f"### {head} — {bm.name}", ""]
    lines.append(f"_Reference:_ {bm.reference} &nbsp;|&nbsp; "
                 f"_Credibility:_ {bm.level.value}")
    lines.append("")
    lines.append("| Quantity | K2 | Reference | Source | Rel. err | Tol | Status |")
    lines.append("|---|---|---|---|---|---|---|")
    for c in bm.comparisons:
        tol = (f"{c.tol_rel:.0%}" if c.tol_rel else "") + \
              (f" / {c.tol_abs:g}" if c.tol_abs else "")
        # A one-sided row passes on the strength of its direction, so it can show
        # a large relative deviation next to a small tolerance and still be a
        # pass. Say so in the tolerance cell rather than leaving the reader to
        # reconcile "30% error, 15% tolerance, ✓".
        if getattr(c, "one_sided", ""):
            arrow = "≤" if c.one_sided == "below" else "≥"
            tol = f"{arrow} ref ({tol})" if tol else f"{arrow} ref"
        lines.append(
            f"| {c.label} | {c.k2:.4g} | {c.ref:.4g} | {c.source} | "
            f"{c.rel_err:.2%} | {tol or '—'} | {'✓' if c.passed else '✗'} |")
    lines.append("")
    for p in plot_paths:
        lines.append(f"![{bm.name}]({p})")
    if plot_paths:
        lines.append("")
    return "\n".join(lines)


DOMAIN_TITLES = {
    "sim": "Flight Simulation (6DOF) ↔ Integrator exact solutions / OpenRocket",
    "cfd": "Aerodynamics ↔ Wind-tunnel measurement / Taylor–Maccoll exact / SU2",
    "structures": "Structures ↔ NAFEMS benchmarks / textbook closed form / CalculiX",
}

# Benchmarks whose reference is published external data rather than a formula
# evaluated here or another solver run here. These are the rows that answer "has
# this been checked against something outside this program", so the report calls
# them out separately — labelled by what the published numbers actually are,
# because the three kinds do not carry the same weight.
#
# "Measurement" means an instrument reading. A published *tabulation* of an exact
# solution (NACA-1135) is not one, and neither is a NAFEMS target value, which is
# an agreed converged answer the FE community reproduces. Lumping them together
# would inflate the strongest claim the report makes.
PUBLISHED_REFERENCES = {
    "AEDC-TR-70-100": "wind-tunnel measurement",
    "AGARD AR-138": "wind-tunnel measurement",
    "NACA Report 1135": "published exact-solution tables",
    "NAFEMS": "agreed benchmark target value",
}


def _published_kind(bm: Benchmark) -> str:
    """What kind of published data this benchmark is checked against, if any."""
    for tag, kind in PUBLISHED_REFERENCES.items():
        if tag in bm.reference:
            return kind
    return ""


def _is_measurement(bm: Benchmark) -> bool:
    return _published_kind(bm) == "wind-tunnel measurement"


def _provenance_table(benchmarks: list) -> str:
    """One row per published-data benchmark: what it is checked against."""
    rows = [b for b in benchmarks if _published_kind(b)]
    if not rows:
        return ""
    out = ["### Comparisons against published data", "",
           "| Benchmark | Source | Kind | Status |", "|---|---|---|---|"]
    for b in rows:
        status = "skipped" if b.skipped else _STATUS[b.passed]
        out.append(f"| {b.name} | {b.reference} | {_published_kind(b)} | {status} |")
    out.append("")
    return "\n".join(out)


def render(benchmarks: list) -> str:
    n_pass = sum(b.passed and not b.skipped for b in benchmarks)
    n_skip = sum(b.skipped for b in benchmarks)
    n_fail = sum(not b.passed and not b.skipped for b in benchmarks)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    n_meas = sum(_is_measurement(b) for b in benchmarks)
    md = ["# K2 Physics Validation Report", "",
          f"_Generated {ts}_", "",
          f"**{n_pass} passed · {n_fail} failed · {n_skip} skipped**", "",
          "Each engine is benchmarked against an *independent* reference, in "
          "three tiers of increasing strength:", "",
          "1. **Exact solutions** — the 6DOF integrator against closed-form ODE "
          "solutions, structures against textbook formulas, the cone solver "
          "against Taylor–Maccoll.",
          "2. **Code-to-code** — flight against OpenRocket, aerodynamics against "
          "SU2, the airframe FEM against CalculiX.",
          f"3. **Published measurement** — {n_meas} benchmark"
          f"{'' if n_meas == 1 else 's'} compare K2 against wind-tunnel data "
          "from the open literature. Agreement with another program proves "
          "consistency; agreement with a measurement is the only tier that can "
          "show the physics is right. The table below lists these alongside the "
          "other published references (exact-solution tables, agreed benchmark "
          "target values), which are external but are not measurements.", "",
          _provenance_table(benchmarks)]

    for domain in ("sim", "cfd", "structures"):
        items = [b for b in benchmarks if b.domain == domain]
        if not items:
            continue
        md.append(f"## {DOMAIN_TITLES.get(domain, domain)}")
        md.append("")
        for bm in items:
            md.append(_bench_md(bm, _plot_curves(bm) if not bm.skipped else []))
            md.append("")
    return "\n".join(md)


# ── entry ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Generate the K2 validation report.")
    ap.add_argument("--full", action="store_true",
                    help="include slow SU2 / CalculiX / OpenRocket benchmarks")
    ap.add_argument("--from-cache", action="store_true",
                    help="re-render from the last benchmarks.json instead of re-running")
    args = ap.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if args.from_cache:
        benchmarks = load_benchmarks(CACHE)
        if not benchmarks:
            raise SystemExit(f"No cache at {CACHE}; run without --from-cache first.")
    else:
        benchmarks = collect(full=args.full)
        save_benchmarks(benchmarks, CACHE)

    md = render(benchmarks)
    out = REPORT_DIR / "REPORT.md"
    out.write_text(md, encoding="utf-8")
    print(f"Wrote {out}")
    for b in benchmarks:
        print("  " + b.summary())


if __name__ == "__main__":
    main()
