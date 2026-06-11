"""
K2 Aerospace — Structural Analysis Report Generator
====================================================
Produces a professional multi-section PDF from a WorkstationReport:

    Executive Summary · Vehicle Information · Stress Analysis ·
    Buckling Analysis · Recovery Loads · Fin Analysis · Temperature ·
    Failure Map · Safety Score · Recommendations · Final Verdict

Uses ReportLab (platypus). No Qt dependency — callable from the workspace
export button or head-less.
"""
from __future__ import annotations

import logging
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
)
from reportlab.lib.enums import TA_CENTER

logger = logging.getLogger("K2.StructReport")

_NAVY = colors.HexColor("#0d2440")
_BLUE = colors.HexColor("#1f6feb")
_GREY = colors.HexColor("#8b949e")
_LIGHT = colors.HexColor("#eef2f6")
_GREEN = colors.HexColor("#2ecc71")
_AMBER = colors.HexColor("#f1c40f")
_ORANGE = colors.HexColor("#e67e22")
_RED = colors.HexColor("#e74c3c")


def _styles():
    ss = getSampleStyleSheet()
    ss.add(ParagraphStyle("K2Title", parent=ss["Title"], textColor=_NAVY,
                          fontSize=22, spaceAfter=4))
    ss.add(ParagraphStyle("K2Sub", parent=ss["Normal"], textColor=_GREY,
                          fontSize=10, alignment=TA_CENTER, spaceAfter=2))
    ss.add(ParagraphStyle("K2H", parent=ss["Heading2"], textColor=_BLUE,
                          fontSize=13, spaceBefore=12, spaceAfter=4))
    ss.add(ParagraphStyle("K2Body", parent=ss["Normal"], fontSize=9.5,
                          textColor=colors.HexColor("#222222"), leading=13))
    return ss


def _status_color(sf):
    if sf >= 1.5:
        return _GREEN
    if sf >= 1.2:
        return _AMBER
    if sf >= 1.0:
        return _ORANGE
    return _RED


def _kv_table(rows, col_w=(70 * mm, 95 * mm)):
    t = Table(rows, colWidths=col_w)
    t.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), "Helvetica", 9.5),
        ("FONT", (0, 0), (0, -1), "Helvetica-Bold", 9.5),
        ("TEXTCOLOR", (0, 0), (0, -1), _NAVY),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, _LIGHT]),
        ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#d0d7de")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    return t


def _recommendations(state, rep):
    recs = []
    if rep.mass.overbuilt_pct >= 30:
        recs.append(f"Structure is ~{rep.mass.overbuilt_pct:.0f}% overbuilt "
                    f"(efficiency {rep.mass.efficiency_pct:.0f}%). Consider thinner "
                    f"wall or lighter material to recover mass margin.")
    weakest = rep.failure.weakest
    if weakest and weakest.margin < 1.5:
        recs.append(f"{weakest.name} is the governing subsystem (SF "
                    f"{weakest.margin:.2f}, {weakest.status}) — reinforce or "
                    f"de-rate the {weakest.detail.lower()}.")
    if rep.buckling.governing and rep.buckling.governing.margin < 2.0:
        recs.append(f"Buckling margin governed by {rep.buckling.governing.name} "
                    f"(×{rep.buckling.governing.margin:.2f}); add stiffeners or "
                    f"increase wall thickness aft.")
    if rep.fin.flutter_margin < 1.5:
        recs.append(f"Fin flutter margin is {rep.fin.flutter_margin:.2f}× — stiffen "
                    f"fins (thicker section / lower aspect ratio) before flight.")
    if rep.recovery.safety_factor < 2.0:
        recs.append(f"Recovery hardware SF is {rep.recovery.safety_factor:.2f}; "
                    f"upsize eyebolt / harness for the {rep.recovery.peak_force_N:.0f} N "
                    f"peak deployment shock.")
    if rep.thermal.exceeds_limit:
        recs.append(f"Skin temperature ({rep.thermal.skin_temp_K:.0f} K) exceeds the "
                    f"material service limit ({rep.thermal.service_limit_K:.0f} K) — "
                    f"add ablative / insulating layer.")
    if not recs:
        recs.append("All structural margins are healthy. No corrective action "
                    "required for the analysed load cases.")
    return recs


def generate_structural_report(path, state, rep, material_name="Aluminum 6061-T6"):
    """Write a full structural PDF report to *path*. Returns the path."""
    ss = _styles()
    doc = SimpleDocTemplate(
        str(path), pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=16 * mm, bottomMargin=16 * mm,
        title="K2 Structural Analysis Report",
    )
    story = []
    name = getattr(state, "name", None) or "Untitled Rocket"

    # ── Header ──
    story.append(Paragraph("K2 AEROSPACE", ss["K2Title"]))
    story.append(Paragraph("Structural Analysis Report", ss["K2Sub"]))
    story.append(Paragraph(
        f"{name} &nbsp;·&nbsp; {datetime.now():%Y-%m-%d %H:%M} &nbsp;·&nbsp; "
        f"Material: {material_name}", ss["K2Sub"]))
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=1.2, color=_BLUE))
    story.append(Spacer(1, 8))

    # ── Executive Summary ──
    sc = rep.score
    verdict_col = {"PASS": _GREEN, "PASS WITH MARGIN": _AMBER, "FAIL": _RED}.get(
        rep.verdict, _GREY)
    story.append(Paragraph("Executive Summary", ss["K2H"]))
    summ = Table([[
        Paragraph(f"<b>Safety Score</b><br/><font size=20 color='{sc.color}'>"
                  f"{sc.score}/100</font><br/>{sc.grade}", ss["K2Body"]),
        Paragraph(f"<b>Final Verdict</b><br/><font size=16 color='{verdict_col}'>"
                  f"{rep.verdict}</font>", ss["K2Body"]),
        Paragraph(f"<b>Governing Subsystem</b><br/>{rep.failure.weakest.name}<br/>"
                  f"SF {rep.failure.weakest.margin:.2f} ({rep.failure.weakest.status})"
                  if rep.failure.weakest else "—", ss["K2Body"]),
    ]], colWidths=[55 * mm, 55 * mm, 55 * mm])
    summ.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#d0d7de")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d0d7de")),
        ("BACKGROUND", (0, 0), (-1, -1), _LIGHT),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(summ)
    story.append(Spacer(1, 4))
    bc = rep.body_condition
    story.append(Paragraph(
        f"The vehicle was evaluated across thrust, max-Q, recovery and thermal "
        f"load cases. Governing body von Mises stress is "
        f"<b>{bc.get('von_mises', 0)/1e6:.0f} MPa</b> at a safety factor of "
        f"<b>{bc.get('safety_factor', 0):.2f}</b>. "
        f"Structural mass is {rep.mass.current_mass_kg:.2f} kg "
        f"({rep.mass.efficiency_pct:.0f}% efficient, "
        f"{rep.mass.optimization_potential} optimization potential).",
        ss["K2Body"]))

    # ── Vehicle Information ──
    story.append(Paragraph("Vehicle Information", ss["K2H"]))
    story.append(_kv_table([
        ["Name", name],
        ["Length", f"{getattr(state,'length',0):.3f} m"],
        ["Diameter", f"{getattr(state,'diameter',0)*1000:.0f} mm"],
        ["Wall Thickness", f"{getattr(state,'wall_thickness',0)*1000:.2f} mm"],
        ["Fin Count", f"{getattr(state,'fin_count',0)}"],
        ["Total Mass", f"{(state.total_mass() if callable(getattr(state,'total_mass',None)) else 0):.2f} kg"],
        ["Material", material_name],
    ]))

    # ── Stress Analysis ──
    story.append(Paragraph("Stress Analysis", ss["K2H"]))
    story.append(_kv_table([
        ["Axial Stress", f"{bc.get('axial',0)/1e6:.1f} MPa"],
        ["Hoop Stress", f"{bc.get('hoop',0)/1e6:.1f} MPa"],
        ["Bending Stress", f"{bc.get('bending',0)/1e6:.1f} MPa"],
        ["Shear Stress", f"{bc.get('shear',0)/1e6:.1f} MPa"],
        ["Thermal Stress", f"{bc.get('thermal',0)/1e6:.1f} MPa"],
        ["Von Mises (peak)", f"{bc.get('von_mises',0)/1e6:.1f} MPa"],
        ["Safety Factor", f"{bc.get('safety_factor',0):.2f}"],
    ]))

    # ── Buckling Analysis ──
    story.append(Paragraph("Buckling Analysis", ss["K2H"]))
    brows = [["Mode", "Critical", "Margin", "Status"]]
    for m in rep.buckling.modes:
        crit = f"{m.critical:.0f} N" if m.unit == "N" else f"{m.critical/1e6:.0f} MPa"
        brows.append([m.name, crit, f"×{m.margin:.2f}", m.status])
    bt = Table(brows, colWidths=[55 * mm, 45 * mm, 30 * mm, 35 * mm])
    bt.setStyle(_grid_style())
    story.append(bt)

    # ── Recovery Loads ──
    rl = rep.recovery
    story.append(Paragraph("Recovery Loads", ss["K2H"]))
    story.append(_kv_table([
        ["Drogue Deployment Shock", f"{rl.drogue_shock_N:.0f} N"],
        ["Main Deployment Shock", f"{rl.main_shock_N:.0f} N"],
        ["Harness Tension", f"{rl.harness_tension_N:.0f} N"],
        ["Nose Cone Separation", f"{rl.nosecone_separation_N:.0f} N"],
        ["Bulkhead Load", f"{rl.bulkhead_load_N:.0f} N"],
        ["Eye Bolt Load", f"{rl.eyebolt_load_N:.0f} N"],
        ["Peak Deployment Force", f"{rl.peak_force_N:.0f} N"],
        ["Recovery Safety Factor", f"{rl.safety_factor:.2f}  ({rl.status})"],
    ]))

    # ── Fin Analysis ──
    fa = rep.fin
    story.append(Paragraph("Fin Analysis", ss["K2H"]))
    story.append(_kv_table([
        ["Root Bending Stress", f"{fa.root_bending_MPa:.1f} MPa"],
        ["Root Shear Stress", f"{fa.root_shear_MPa:.1f} MPa"],
        ["Tip Deflection", f"{fa.tip_deflection_mm:.2f} mm"],
        ["Natural Frequency", f"{fa.natural_frequency_Hz:.0f} Hz"],
        ["Flutter Speed", f"{fa.flutter_speed_m_s:.0f} m/s"],
        ["Flutter Margin", f"{fa.flutter_margin:.2f}×"],
        ["Safety Factor", f"{fa.safety_factor:.2f}  ({fa.status})"],
    ]))

    # ── Temperature ──
    tp = rep.thermal
    story.append(Paragraph("Temperature Analysis", ss["K2H"]))
    story.append(_kv_table([
        ["Skin Temperature", f"{tp.skin_temp_K:.0f} K ({tp.skin_temp_K-273.15:.0f} °C)"],
        ["Internal Temperature", f"{tp.internal_temp_K:.0f} K"],
        ["Thermal Gradient", f"{tp.gradient_K:.0f} K"],
        ["Thermal Expansion", f"{tp.expansion_mm:.2f} mm"],
        ["Convective Heat Flux", f"{tp.heat_flux_W_m2/1000:.1f} kW/m²"],
        ["Max Thermal Stress", f"{tp.max_thermal_stress_MPa:.1f} MPa @ {tp.max_stress_altitude:.0f} m"],
        ["Service Limit", f"{tp.service_limit_K:.0f} K " + ("EXCEEDED" if tp.exceeds_limit else "OK")],
    ]))

    # ── Deflection & Vibration ──
    bd = rep.deflection
    md = rep.modal
    story.append(Paragraph("Deflection &amp; Vibration", ss["K2H"]))
    story.append(_kv_table([
        ["Max Deflection", f"{bd.max_deflection_mm:.2f} mm ({bd.location})"],
        ["Tip Deflection", f"{bd.tip_deflection_mm:.2f} mm"],
        ["Applied Normal Force", f"{bd.applied_normal_force_N:.0f} N"],
        ["Mode 1 (bending)", f"{md.f1_hz:.0f} Hz"],
        ["Mode 2", f"{md.f2_hz:.0f} Hz"],
        ["Mode 3", f"{md.f3_hz:.0f} Hz"],
    ]))
    if md.low_freq:
        story.append(Paragraph(f"<font color='#e67e22'>{md.warning}</font>", ss["K2Body"]))

    # ── Physics Consistency Checks ──
    if getattr(rep, "warnings", None):
        story.append(Paragraph("Physics Consistency Checks", ss["K2H"]))
        from structures.validation import severity_color
        for wn in rep.warnings:
            story.append(Paragraph(
                f"<font color='{severity_color(wn.severity)}'>[{wn.severity.upper()}]</font> "
                f"{wn.message}", ss["K2Body"]))
            story.append(Spacer(1, 1))

    # ── Failure Map ──
    story.append(Paragraph("Structural Failure Map", ss["K2H"]))
    frows = [["Subsystem", "Safety Factor", "Status", "Governing Check"]]
    for c in rep.failure.components:
        sfx = "∞" if c.margin > 1e6 else f"{c.margin:.2f}"
        frows.append([c.name, sfx, c.status, c.detail])
    ft = Table(frows, colWidths=[38 * mm, 28 * mm, 24 * mm, 75 * mm])
    fstyle = _grid_style()
    for i, c in enumerate(rep.failure.components, start=1):
        fstyle.add("TEXTCOLOR", (2, i), (2, i), _status_color(c.margin))
        fstyle.add("FONT", (2, i), (2, i), "Helvetica-Bold", 8.5)
    ft.setStyle(fstyle)
    story.append(ft)

    # ── Safety Score breakdown ──
    story.append(Paragraph("Safety Score Breakdown", ss["K2H"]))
    srows = [["Margin", "Safety Factor", "Sub-score (/100)"]]
    labels = [("Yield", sc.yield_margin), ("Buckling", sc.buckling_margin),
              ("Recovery", sc.recovery_margin), ("Fin", sc.fin_margin),
              ("Thermal", sc.thermal_margin)]
    keymap = {"Yield": "yield", "Buckling": "buckling", "Recovery": "recovery",
              "Fin": "fin", "Thermal": "thermal"}
    for lbl, sf in labels:
        sub = sc.subscores.get(keymap[lbl], 0)
        sfx = "∞" if sf > 1e6 else f"{sf:.2f}"
        srows.append([lbl, sfx, f"{sub:.0f}"])
    srows.append(["TOTAL", "", f"{sc.score} ({sc.grade})"])
    sst = Table(srows, colWidths=[55 * mm, 55 * mm, 55 * mm])
    sst.setStyle(_grid_style(total_row=True))
    story.append(sst)

    # ── Recommendations ──
    story.append(Paragraph("Recommendations", ss["K2H"]))
    for i, r in enumerate(_recommendations(state, rep), 1):
        story.append(Paragraph(f"{i}. {r}", ss["K2Body"]))
        story.append(Spacer(1, 2))

    # ── Final Verdict ──
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=1.0, color=_BLUE))
    vt = Table([[Paragraph(
        f"<b>FINAL VERDICT:</b> <font size=15 color='{verdict_col}'>{rep.verdict}"
        f"</font>", ss["K2Body"])]], colWidths=[170 * mm])
    vt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), _LIGHT),
        ("BOX", (0, 0), (-1, -1), 0.5, verdict_col),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(vt)

    doc.build(story)
    logger.info(f"Structural report written: {path}")
    return path


def _grid_style(total_row=False):
    s = TableStyle([
        ("FONT", (0, 0), (-1, -1), "Helvetica", 8.5),
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 9),
        ("BACKGROUND", (0, 0), (-1, 0), _NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, _LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d0d7de")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
    ])
    if total_row:
        s.add("FONT", (0, -1), (-1, -1), "Helvetica-Bold", 9)
        s.add("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#dce4ec"))
    return s


if __name__ == "__main__":
    # Standalone test: build a report from the workstation smoke-test state
    from structures.workstation import full_analysis

    class _S:
        name = "Test Rocket"; diameter = 0.10; length = 1.5; wall_thickness = 0.002
        material_name = "Aluminum 6061-T6"; dry_mass = 3.0; propellant_mass = 1.1
        fin_root_chord = 0.15; fin_tip_chord = 0.07; fin_span = 0.06; fin_height = 0.06
        fin_thickness = 0.003; fin_count = 3; thrust = 1200.0; weight = 40.0
        motor_max_thrust = 1500.0; motor_avg_thrust = 1100.0; max_altitude = 3000.0
        max_velocity = 280.0; max_mach = 0.85; max_acceleration = 120.0
        dynamic_pressure = 48000.0; temperature_ambient = 288.15
        drogue_cd_area = 0.5; main_cd_area = 3.0; main_deploy_altitude = 300.0
        wind_speed = 5.0
        def total_mass(self): return self.dry_mass + self.propellant_mass

    s = _S()
    rep = full_analysis(s, None, None, "Aluminum 6061-T6", "Max-Q")
    out = generate_structural_report("structural_report_test.pdf", s, rep)
    print(f"Report written: {out}")
