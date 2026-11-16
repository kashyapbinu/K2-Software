"""
K2 AeroSim — Technical report
===============================
One PDF a team can hand a judge or a range-safety officer: the vehicle, the
launch conditions, the flight, the recovery sequence, the dispersion, the
structural and flutter margins, the rules checklist, and what the models
leave out.

Two steps, so the numbers can be checked without reading a PDF:

1. :func:`build_content` turns the inputs into a :class:`Report` — titled
   sections of (label, value) rows, figures and notes. Every flight number
   comes from the FlightSummary (core.flight_summary); nothing is
   recomputed here.
2. :func:`render_pdf` lays the Report out with reportlab.

A section whose analysis was not run says so. It is never filled with
defaults.
"""
from __future__ import annotations

import datetime
import io
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

NOT_RUN = "Not run"


@dataclass
class Section:
    title: str
    rows: List[Tuple[str, str]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    figures: List[Tuple[str, bytes]] = field(default_factory=list)   # (caption, PNG)
    table: Optional[List[List[str]]] = None                          # header + rows
    # Printed first, in red: this section's result is out of date.
    warnings: List[str] = field(default_factory=list)


@dataclass
class Report:
    title: str
    subtitle: str
    sections: List[Section] = field(default_factory=list)

    def section(self, title: str) -> Optional[Section]:
        return next((s for s in self.sections if s.title == title), None)

    def value(self, section: str, label: str) -> Optional[str]:
        s = self.section(section)
        return dict(s.rows).get(label) if s else None

    @property
    def warnings(self) -> List[str]:
        """Every out-of-date mark in the report, as "Section: text"."""
        return [f"{s.title}: {w}" for s in self.sections for w in s.warnings]


def _stale(reason: str, what: str = "The numbers below are") -> str:
    return (f"OUT OF DATE: {reason or 'inputs changed'}. {what} from before "
            "that change; run the analysis again.")


# ── formatting (one place, so the PDF and the tests agree) ────────────────

def fmt(value, unit: str = "", digits: int = 1) -> str:
    if value is None:
        return "—"
    if isinstance(value, float) and not math.isfinite(value):
        return "—"
    text = f"{value:,.{digits}f}"
    return f"{text} {unit}".strip()


# Assumptions and limits of the models a reader should know about. Kept in
# step with CLAUDE.md section 5.
MODEL_LIMITS = [
    "6-DOF point-body flight over a flat Earth; gravity varies with altitude only.",
    "Aerodynamics: Barrowman normal-force/CP with a component drag build-up "
    "(skin friction, base, fin pressure and wave drag); no transition or "
    "boat-tail drag, no launch-lug or rail-button drag.",
    "Static margin is reported at the current Mach and zero incidence.",
    "Atmosphere: ISA (offset to the pad temperature) or the imported sounding; "
    "structures and flutter use ISA above sea level.",
    "Wind: constant-direction power-law profile or the multi-level table, "
    "optional pink-noise turbulence; no vertical wind.",
    "Parachutes: drag area with a cubic inflation transient; opening shock is "
    "the peak summed canopy drag in the simulation.",
    "Attitude uses Euler angles; landing range varies by about ±2 % with the "
    "wind's compass bearing on a vertical launch.",
    "Monte Carlo landing ellipses assume a bivariate normal dispersion.",
]


# ── content ───────────────────────────────────────────────────────────────

def build_content(state, summary, *, history=None, rule_results=None,
                  rule_set=None, dispersion=None, landings=None,
                  margins=None, out_of_date=None, title: str = "") -> Report:
    """Assemble the report.

    state:        RocketState (vehicle, motor, launch setup)
    summary:      FlightSummary of the reported flight, or None
    history:      HistoryManager of that flight (profile plots), optional
    rule_results: list of core.rules.RuleResult, optional; rule_set: its RuleSet
    dispersion:   core.landing_map.Dispersion of a Monte Carlo run, optional
    landings:     its (north, east) landing points, optional
    margins:      {"flutter_speed": m/s, "structural_sf": x, "structures_source": str}
                  — whatever the Dynamics / Structures tabs produced, optional
    out_of_date:  {readiness stage key: reason} for each result that was
                  made from inputs that have since changed (core.readiness).
                  Its section is printed with a mark saying so. The report
                  used to print such a result like any other: change a fin
                  after flying and the PDF still gave the old apogee.
    """
    stale = out_of_date or {}
    from core.version import __version__

    s = state
    name = title or s.name or "Rocket"
    rep = Report(f"{name} — Flight Analysis",
                 f"K2 AeroSim {__version__} · "
                 f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # Vehicle
    # The rocket on the pad (RocketState.liftoff_*), not the state's live
    # mass, CG and margin: a report is written after the flight, when those
    # are the burnt-out rocket's. This section used to read them, and for the
    # canonical rocket printed 6.001 kg, 0.900 m and 6.78 cal under "liftoff"
    # and "on the pad" (7.400 kg, 1.051 m, 5.30 cal), and a liftoff
    # thrust-to-weight of 18.52 (15.02).
    liftoff = s.liftoff_mass()
    weight = liftoff * 9.80665
    pad_margin = s.pad_stability_margin()
    rep.sections.append(Section("Vehicle", [
        ("Length", fmt(s.length, "m", 3)),
        ("Body diameter", fmt(s.diameter * 1000.0, "mm", 1)),
        ("Dry mass (airframe)", fmt(s.dry_mass, "kg", 3)),
        ("Liftoff mass", fmt(liftoff, "kg", 3)),
        ("CG at liftoff (from nose)", fmt(s.liftoff_cg(), "m", 3)),
        ("CP, subsonic (from nose)", fmt(s.cp, "m", 3)),
        ("Static margin on the pad", fmt(pad_margin, "cal", 2)),
        ("Fins", f"{s.fin_count}" if s.fin_count else "none"),
        ("Airbrake area (Cd×A)", fmt(s.airbrake_cd_area * 1e4, "cm²", 1)
         if s.airbrake_cd_area > 0 else "none"),
    ]))

    # Motor
    # Liftoff thrust: the first stage to light on a staged rocket.
    liftoff_thrust = s.liftoff_avg_thrust()
    twr = liftoff_thrust / weight if weight > 0 and liftoff_thrust > 0 else None
    rep.sections.append(Section("Motor", [
        ("Designation", s.motor_designation),
        ("Total impulse", fmt(s.motor_total_impulse, "N·s", 0)),
        ("Average thrust", fmt(s.motor_avg_thrust, "N", 0)),
        # No rated peak: the trapezoid curve assumes one
        # (staging.trapezoid_thrust_curve). This row used to print "0 N".
        ("Peak thrust", fmt(s.motor_max_thrust, "N", 0) if s.motor_max_thrust > 0
         or s.custom_thrust_curve else
         "not given (1.4 × average assumed for the thrust curve)"),
        ("Burn time", fmt(s.motor_burn_time, "s", 2)),
        ("Propellant mass", fmt(s.propellant_mass_initial, "kg", 3)),
        ("Thrust curve", "measured" if s.custom_thrust_curve else
         "trapezoid from catalogue values"),
        ("Liftoff thrust-to-weight (average thrust)", fmt(twr, "", 2)),
    ]))

    # Launch conditions
    air = (f"sounding: {s.atmosphere_source or 'imported'}" if s.atmosphere_profile
           else ("standard day (ISA)" if s.ground_temperature <= 0
                 else f"ISA offset to {fmt(s.ground_temperature, 'K', 1)} at the pad"))
    wind = (f"{len(s.wind_layers)}-level profile" if s.wind_mode == "multi_level" and s.wind_layers
            else f"{fmt(s.wind_speed, 'm/s', 1)} from {s.wind_direction:.0f}°"
                 + (f", turbulence {s.wind_gust_intensity * 100:.0f} %"
                    if s.wind_gust_intensity > 0 else ""))
    rep.sections.append(Section("Launch conditions", [
        ("Site", s.launch_site_name or "custom"),
        ("Pad elevation", fmt(s.launch_altitude, "m", 0)),
        ("Position", f"{s.launch_latitude:.5f}°, {s.launch_longitude:.5f}°"
         if (s.launch_latitude or s.launch_longitude) else "not set"),
        ("Rail length", fmt(s.launch_rod_length, "m", 2)),
        ("Rail elevation / azimuth", f"{s.launch_angle:.1f}° / {s.launch_azimuth:.0f}°"),
        ("Air", air),
        ("Wind", wind),
    ]))

    # Flight
    if summary is None:
        rep.sections.append(Section("Flight", notes=[NOT_RUN + ": no flight simulated."]))
    else:
        f = summary
        sec = Section("Flight", [
            ("Result", f.final_phase),
            ("Apogee above pad", fmt(f.apogee, "m", 0)),
            ("Time to apogee", fmt(f.apogee_time, "s", 1)),
            ("Max velocity", fmt(f.max_velocity, "m/s", 1)),
            ("Max Mach", fmt(f.max_mach, "", 3)),
            ("Max sensed acceleration", fmt(f.max_acceleration / 9.80665, "g", 1)),
            ("Max dynamic pressure", fmt(f.max_q / 1000.0, "kPa", 1)),
            ("Rail exit velocity", fmt(f.rail_exit_velocity, "m/s", 1)),
            ("Stability at rail exit", fmt(f.stability_at_rail_exit, "cal", 2)),
            ("Minimum stability (rail exit → apogee)", fmt(f.min_stability, "cal", 2)),
            ("Touchdown speed", fmt(f.landing_velocity, "m/s", 1)),
            ("Landing distance from pad", fmt(f.landing_drift, "m", 0)),
            ("Flight time", fmt(f.flight_time, "s", 1)),
        ], notes=[f"⚠ {w}" for w in f.warnings])
        if "flight" in stale:
            sec.warnings.append(_stale(stale["flight"]))
        if history is not None and len(history) > 0:
            sec.figures.extend(_profile_figures(history))
        rep.sections.append(sec)

        # Recovery
        rec = Section("Recovery")
        if "flight" in stale:
            rec.warnings.append(_stale(stale["flight"]))
        if f.canopies:
            rec.table = [["Canopy", "Trigger", "Cd×A (m²)", "Opened at", "Speed",
                          "Peak force", "Descent rate"]]
            for c in f.canopies:
                trig = ("apogee" if c.get("event") == "apogee"
                        else f"{fmt(c.get('trigger_altitude'), 'm', 0)}")
                if c.get("delay"):
                    trig += f" + {c['delay']:.1f} s"
                opened = c.get("deploy_time") is not None
                rec.table.append([
                    c.get("name", ""), trig, f"{c.get('cd_area', 0.0):.3f}",
                    (f"{fmt(c.get('deploy_altitude'), 'm', 0)}, T+{c['deploy_time']:.1f} s"
                     if opened else "did not open"),
                    fmt(c.get("deploy_speed"), "m/s", 1) if opened else "—",
                    fmt(c.get("peak_force"), "N", 0) if opened else "—",
                    fmt(c.get("descent_rate"), "m/s", 1) if opened else "—"])
            rec.rows.append(("Peak opening shock (all canopies)",
                             fmt(f.recovery_shock_force, "N", 0)))
        else:
            rec.notes.append("No parachute on the vehicle.")
        rep.sections.append(rec)

    # Dispersion
    disp = Section("Landing dispersion (Monte Carlo)")
    if dispersion is None:
        disp.notes.append(NOT_RUN + ": no Monte Carlo run with landings.")
    else:
        from core.landing_map import ellipse_coverage
        d = dispersion
        if "monte_carlo" in stale:
            disp.warnings.append(_stale(stale["monte_carlo"]))
        disp.rows += [
            ("Flights", f"{d.n}"),
            ("Mean landing point", f"{d.mean_north:,.0f} m N, {d.mean_east:,.0f} m E"),
        ] + [(f"{k:g}σ ellipse ({ellipse_coverage(k) * 100:.0f} % of landings)",
              f"{d.axes(k)[0]:,.0f} × {d.axes(k)[1]:,.0f} m semi-axes, "
              f"major axis {d.major_bearing:.0f}°")
             for k in (1.0, 2.0, 3.0)]
        if landings:
            disp.figures.append(("Landing points and ellipses",
                                 _dispersion_figure(d, landings)))
    rep.sections.append(disp)

    # Margins
    m = margins or {}
    marg = Section("Structural and flutter margins")
    vmax = summary.max_velocity if summary is not None else None
    if m.get("flutter_speed"):
        if "flutter" in stale:
            marg.warnings.append(_stale(stale["flutter"], "The fin flutter speed is"))
        marg.rows.append(("Fin flutter speed (governing)", fmt(m["flutter_speed"], "m/s", 0)))
        if vmax:
            marg.rows.append(("Flutter margin over max velocity",
                              fmt((m["flutter_speed"] / vmax - 1.0) * 100.0, "%", 0)))
    else:
        marg.rows.append(("Fin flutter", NOT_RUN + " (Dynamics tab)"))
    if m.get("structural_sf"):
        # The structural margin is worked from the flight's loads, so it is
        # as old as the flight.
        if "flight" in stale:
            marg.warnings.append(_stale(
                stale["flight"], "The structural safety factor uses the loads of a flight"))
        marg.rows.append(("Governing structural safety factor",
                          fmt(m["structural_sf"], "", 2)
                          + (f" ({m['structures_source']})" if m.get("structures_source") else "")))
    else:
        marg.rows.append(("Airframe structure", NOT_RUN + " (Structures tab)"))
    rep.sections.append(marg)

    # Rules
    rules = Section("Rules check")
    if not rule_results:
        rules.notes.append(NOT_RUN + ": no rule set evaluated.")
    else:
        if "rules" in stale or "flight" in stale:
            rules.warnings.append(_stale(stale.get("rules") or stale["flight"]))
        if rule_set is not None:
            src = rule_set.source or "no source given"
            rules.notes.append(f"Rule set: {rule_set.name} — limits from {src}"
                               + ("" if rule_set.confirmed else " (NOT confirmed)"))
        rules.table = [["Rule", "Test", "Value", "Result", "Margin"]]
        for r in rule_results:
            test = (f"{r.rule.comparison} {r.rule.limit:g}" if r.rule.limit is not None
                    and r.rule.comparison != "between" else
                    (f"{r.rule.limit:g} … {r.rule.limit_high:g}"
                     if r.rule.limit is not None and r.rule.limit_high is not None else "—"))
            rules.table.append([r.rule.name, test,
                                fmt(r.value, "", 2), r.status, fmt(r.margin, "", 2)])
    rep.sections.append(rules)

    rep.sections.append(Section("Assumptions and model limits", notes=list(MODEL_LIMITS)))
    return rep


# ── figures ───────────────────────────────────────────────────────────────

def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return buf.getvalue()


def _profile_figures(history) -> List[Tuple[str, bytes]]:
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    t = history.get_values("time")
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 7.5), sharex=True)
    for ax, key, label in zip(axes, ("altitude", "velocity", "acceleration"),
                              ("Altitude (m)", "Velocity (m/s)", "Acceleration (m/s²)")):
        ax.plot(t, history.get_values(key), lw=1.4)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    return [("Flight profile", _png(fig))]


def _dispersion_figure(d, landings) -> bytes:
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    ax.scatter([e for _, e in landings], [n for n, _ in landings], s=6, alpha=0.6,
               label="Landings")
    for k in (1.0, 2.0, 3.0):
        poly = d.polygon(k)
        ax.plot([e for _, e in poly], [n for n, _ in poly], lw=1.2, label=f"{k:g}σ")
    ax.plot([0], [0], marker="^", ms=10, color="k", label="Pad")
    ax.set_xlabel("East (m)"); ax.set_ylabel("North (m)")
    ax.set_aspect("equal", adjustable="datalim"); ax.grid(alpha=0.3); ax.legend()
    return _png(fig)


# ── PDF ───────────────────────────────────────────────────────────────────

def render_pdf(report: Report, path: str) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (Image, Paragraph, SimpleDocTemplate, Spacer,
                                    Table, TableStyle)
    from xml.sax.saxutils import escape

    ss = getSampleStyleSheet()
    doc = SimpleDocTemplate(path, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm,
                            title=report.title, author="K2 AeroSim")
    width = A4[0] - 36 * mm
    story = [Paragraph(escape(report.title), ss["Title"]),
             Paragraph(escape(report.subtitle), ss["Normal"]), Spacer(1, 6 * mm)]
    if report.warnings:
        # On the first page, so nobody has to find the mark in its section.
        story += [Paragraph(
            '<font color="#b00020"><b>This report contains results that are out of '
            'date for the rocket as it is now: '
            + escape(", ".join(sorted({s.title for s in report.sections if s.warnings})))
            + '.</b></font>', ss["BodyText"]), Spacer(1, 4 * mm)]
    grid = TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bbbbbb")),
                       ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                       ("VALIGN", (0, 0), (-1, -1), "TOP")])
    head = TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#222a35")),
                       ("TEXTCOLOR", (0, 0), (-1, 0), colors.white)])
    for sec in report.sections:
        story.append(Paragraph(escape(sec.title), ss["Heading2"]))
        for w in sec.warnings:
            story.append(Paragraph(
                f'<font color="#b00020"><b>{escape(w)}</b></font>', ss["BodyText"]))
            story.append(Spacer(1, 1.5 * mm))
        if sec.rows:
            t = Table([[escape(a), escape(b)] for a, b in sec.rows],
                      colWidths=[width * 0.5, width * 0.5])
            t.setStyle(grid)
            story.append(t)
        if sec.table:
            cells = [[Paragraph(escape(str(c)), ss["BodyText"]) for c in row]
                     for row in sec.table]
            t = Table(cells, repeatRows=1)
            t.setStyle(grid)
            t.setStyle(head)
            story.append(t)
        for note in sec.notes:
            story.append(Paragraph(escape(note), ss["BodyText"]))
        for caption, png in sec.figures:
            img = Image(io.BytesIO(png))
            scale = min(1.0, width / img.drawWidth)
            img.drawWidth *= scale
            img.drawHeight *= scale
            story += [Spacer(1, 3 * mm), img, Paragraph(escape(caption), ss["Italic"])]
        story.append(Spacer(1, 5 * mm))
    doc.build(story)
