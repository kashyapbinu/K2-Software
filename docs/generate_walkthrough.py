"""Generate the K2 AeroSim walkthrough PDF (docs/K2_AeroSim_Walkthrough.pdf).

Detailed, control-by-control manual. Every workspace gets:
  - a purpose line,
  - a screenshot slot (embeds docs/shots/<file>.png if present, else a
    labelled placeholder box you can drop a picture into later),
  - one or more panels, each a two-column table: Control -> what it does.

Re-run after dropping new screenshots into docs/shots/:
    python docs/generate_walkthrough.py
"""
from datetime import datetime
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, HRFlowable, PageBreak, Table,
    TableStyle, Image, KeepTogether)
from reportlab.lib.utils import ImageReader

HERE = Path(__file__).parent
SHOTS = HERE / "shots"

BLUE = colors.HexColor("#1f6feb")
DARK = colors.HexColor("#0d1117")
GREY = colors.HexColor("#57606a")
LIGHT = colors.HexColor("#eef2f7")
ROWALT = colors.HexColor("#f6f8fa")
LINE = colors.HexColor("#d0d7de")

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], textColor=BLUE, fontSize=18, spaceAfter=2)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], textColor=colors.HexColor("#24292f"),
                    fontSize=12, spaceBefore=8, spaceAfter=3)
BODY = ParagraphStyle("Body", parent=ss["BodyText"], fontSize=9.5, leading=13, spaceAfter=3)
SMALL = ParagraphStyle("Small", parent=ss["BodyText"], fontSize=8.5, textColor=GREY, leading=11)
CAP = ParagraphStyle("Cap", parent=ss["BodyText"], fontSize=8, textColor=GREY,
                     leading=10, alignment=TA_CENTER)
CELL = ParagraphStyle("Cell", parent=ss["BodyText"], fontSize=8.7, leading=11.5, spaceAfter=0)
CELLB = ParagraphStyle("CellB", parent=CELL, textColor=colors.HexColor("#0a3069"))
TITLE = ParagraphStyle("Title", parent=ss["Title"], textColor=BLUE, fontSize=30)
SUB = ParagraphStyle("Sub", parent=ss["Normal"], fontSize=12, textColor=GREY, alignment=TA_CENTER)

CONTENT_W = (210 - 36) * mm  # page width minus margins


def shot(name):
    """Image flowable for docs/shots/<name>, scaled to content width; or a
    placeholder box if the file is missing."""
    p = SHOTS / name
    if p.exists():
        try:
            iw, ih = ImageReader(str(p)).getSize()
            w = CONTENT_W
            h = w * ih / iw
            return Image(str(p), width=w, height=h)
        except Exception:
            pass
    # placeholder
    t = Table([[Paragraph(f"[ screenshot slot — drop <b>docs/shots/{name}</b> here ]",
                          CAP)]], colWidths=[CONTENT_W], rowHeights=[70])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
        ("BOX", (0, 0), (-1, -1), 0.6, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t


def control_table(rows):
    """rows: list of (control, description). Two-column reference table."""
    data = [[Paragraph("<b>Control</b>", CELLB), Paragraph("<b>What it does</b>", CELLB)]]
    for ctrl, desc in rows:
        data.append([Paragraph(ctrl, CELLB), Paragraph(desc, CELL)])
    t = Table(data, colWidths=[CONTENT_W * 0.30, CONTENT_W * 0.70], repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dce3ea")),
        ("GRID", (0, 0), (-1, -1), 0.4, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]
    for r in range(1, len(data)):
        if r % 2 == 0:
            style.append(("BACKGROUND", (0, r), (-1, r), ROWALT))
    t.setStyle(TableStyle(style))
    return t


# ════════════════════════════════════════════════════════════════════════════
# GLOBAL TOOLBAR (top of every workspace)
# ════════════════════════════════════════════════════════════════════════════
TOOLBAR = [
    ("New", "Start a fresh project. Opens a brand-new window so the current one is left untouched. (Ctrl+N)"),
    ("Open", "Load a saved <b>.k2</b> project file. (Ctrl+O)"),
    ("Save", "Save the current project to its <b>.k2</b> file. (Ctrl+S)"),
    ("Save As", "Save the project to a new file / name. (Ctrl+Shift+S)"),
    ("Import .ork", "Import an <b>OpenRocket</b> design (.ork). Components, positions and "
                    "materials are translated into the K2 component tree. (Ctrl+I)"),
    ("Reset View", "Recenter the 3D camera in whichever 3D view is active."),
    ("Settings", "Application settings."),
    ("Run Sim (F5)", "Keyboard shortcut to start the flight simulation from anywhere."),
    ("Console (bottom)", "Dockable log panel. Every analysis writes progress, warnings and "
                         "errors here — check it if something looks stuck. Drag it out or hide it."),
]

# ════════════════════════════════════════════════════════════════════════════
# WORKSPACES  —  (num, title, image, purpose, [ (panel, [(ctrl, desc)...]) ], [tips])
# ════════════════════════════════════════════════════════════════════════════
WORKSPACES = [
("1. Design", "00_design.png",
 "Build the airframe component-by-component and watch stability update live. "
 "Everything downstream reads this model.",
 [
  ("Rocket Structure tree (left)", [
    ("Component tree", "The rocket as a hierarchy: <b>Stage → body components → inner "
        "components</b>. Click any item to select and edit it. Right-click for the same "
        "move / duplicate / delete actions as the toolbar below."),
    ("▲ Move up / ▼ Move down", "Reorder the selected component within its parent. Order is "
        "nose-to-tail, so this changes where the part sits along the body."),
    ("Duplicate", "Copy the selected component (with all its settings) as a sibling."),
    ("Delete", "Remove the selected component."),
  ]),
  ("Add New Component palette (top center)", [
    ("Body Components", "<b>Nose Cone</b>, <b>Body Tube</b>, <b>Transition</b> (diameter "
        "change). These form the outer airframe and go directly under a stage."),
    ("Fin Sets", "<b>Trapezoidal</b> fin set — count, root/tip chord, span, sweep, thickness, "
        "cross-section. Must be added onto a body tube."),
    ("Propulsion", "<b>Nozzle</b> — external nozzle geometry."),
    ("Inner Components", "<b>Inner Tube</b> (motor mount), <b>Centering Ring</b>, "
        "<b>Bulkhead</b>, <b>Engine Block</b>. Placed <i>inside</i> a body tube."),
    ("Recovery", "<b>Parachute</b> and <b>Shock Cord</b> — go inside a body tube."),
    ("Mass / Attach", "<b>Mass Component</b> (ballast / payload), <b>Launch Lug</b>, "
        "<b>Rail Button</b>."),
    ("+ Add New Stage (Booster)", "Add another stage for multi-stage rockets."),
  ]),
  ("3D Preview (center)", [
    ("Live 3D model", "Rebuilds as you edit. Drag to orbit, scroll to zoom."),
    ("Toggle 2D Line / Wireframe Mode", "Switch between the shaded 3D model and a flat "
        "2D side-profile / wireframe — handy for checking proportions and station positions."),
  ]),
  ("Component Properties (right)", [
    ("Editor fields", "Edit the selected part's geometry, wall thickness, material and colour. "
        "Fields change with the component type (e.g. a nose cone exposes shape + length; a fin "
        "set exposes count, chords, sweep)."),
  ]),
  ("Stability Analysis (right)", [
    ("CP", "Centre of Pressure — where aerodynamic force acts (Barrowman)."),
    ("CG", "Centre of Gravity — mass balance point (includes motor)."),
    ("Margin", "Static margin in calibers = (CP−CG) / body diameter."),
    ("Status", "<b>UNSTABLE</b> / <b>MARGINAL</b> / <b>✓ STABLE</b> / <b>OVERSTABLE</b> "
        "verdict from the margin."),
  ]),
 ],
 ["Aim for ~1–2 cal static margin. Add a motor in Propulsion — the motor's mass "
  "shifts CG and the stability readout updates."]),

("2. Propulsion", "01_propulsion.png",
 "Pick a motor from the full ThrustCurve.org catalogue or build a custom one.",
 [
  ("Motor Selection (left)", [
    ("Class", "Filter by motor impulse class (A…O)."),
    ("Diameter", "Filter by motor case diameter (mm)."),
    ("Fit current body", "Only show motors that physically fit inside the current airframe."),
    ("Hide out-of-production", "Hide discontinued motors."),
    ("Motor", "The selectable list (counter shows how many pass the filters). Picking a motor "
        "applies it to the rocket and drives every analysis."),
    ("Create Custom Motor", "Open the custom-motor builder: choose a grain "
        "(BATES / Tubular / End-Burner / Star), propellant and nozzle, <b>Simulate Ballistics</b>, "
        "then <b>Apply to Rocket</b> (a motor that doesn't produce thrust is flagged, not applied)."),
  ]),
  ("Motor Properties (left)", [
    ("Total Impulse / Avg Thrust / Max Thrust / Burn Time / Prop Mass",
        "Catalogue figures for the selected motor."),
  ]),
  ("Computed Performance (left)", [
    ("Isp / Mass Flow / Chamber P", "Derived performance numbers for the motor."),
  ]),
  ("Thrust Curve (center)", [
    ("Plot", "The real measured thrust-vs-time curve (ThrustCurve.org samples). The simulation "
        "flies this curve, not a trapezoid approximation, when it is available."),
  ]),
 ],
 ["The selected motor is shared with the simulation, structures, dynamics, Monte "
  "Carlo and optimization tabs."]),

("3. CFD", "02_cfd.png",
 "SU2 aerodynamics — drag, lift, moments, pressure and flow fields.",
 [
  ("Geometry Source (left)", [
    ("Use current rocket design", "Mesh the airframe you built in Design."),
    ("Import external CAD file / Browse CAD File…", "Use an external STL instead."),
    ("Export Geometry to STL", "Write the current rocket as a watertight STL."),
  ]),
  ("Flow Conditions (left)", [
    ("Mach", "Freestream Mach number."),
    ("Altitude", "Sets the atmosphere (pressure, temperature, density) below."),
    ("Angle of Attack", "Pitch angle of the flow, degrees."),
    ("Turbulence Model", "Euler / Laminar / Spalart-Allmaras / k-ω SST."),
  ]),
  ("Atmosphere & Flow (left)", [
    ("Pressure / Temperature / Density / Speed of Sound / Reynolds / Dyn. Pressure",
        "Read-only freestream values computed from the altitude above."),
  ]),
  ("Analysis Mode (left)", [
    ("Single point", "Solve one Mach/AoA condition."),
    ("Sweep (polar)", "Solve a series over Angle of Attack or Mach (one mesh, N solves) to "
        "build a drag/lift polar — see the <b>Polars</b> tab."),
    ("Euler + flat-plate friction (recommended)", "Fast hybrid mode: inviscid Euler pressure "
        "drag plus an analytical skin-friction estimate. Default — the tet-only RANS mesh can "
        "give untrustworthy viscous drag."),
  ]),
  ("Mesh Settings (left)", [
    ("Refinement", "Coarse (fast) … Ultra Fine / Custom. Trades runtime for accuracy."),
  ]),
  ("Run / Export (left)", [
    ("Run CFD Analysis", "Mesh, then solve. Progress shows in the Console."),
    ("Export Results", "PDF report (all contours + plots) and CSV."),
  ]),
  ("Center — 3D Field tab", [
    ("View dropdown", "Field to display: Surface Cp, Pressure/Temperature/Velocity/Mach volume "
        "slices, Streamlines, Vorticity, Q-criterion / Lambda-2, Boundary-layer Y+, surface "
        "temperature, and more (17 modes)."),
    ("Interactive Slice", "Drag a cutting plane through the volume field."),
    ("Contour Lines", "Overlay iso-contour lines on the field."),
    ("Mesh Edges", "Show the computational mesh."),
    ("Probe", "Click a point to read its field value."),
    ("Reset Camera", "Recenter the 3D view."),
  ]),
  ("Center — Polars tab", [
    ("Cl / Cd / Cm vs AoA, or Cd vs Mach", "Populated after a Sweep run."),
    ("Export CSV", "Save the polar data."),
  ]),
  ("Results (right)", [
    ("Aerodynamic Coefficients", "Total Cd with <b>Pressure / Friction / Base / Wave</b> split, "
        "Lift Cl, Moment Cm, and a Converged flag."),
    ("Forces & Center of Pressure", "Axial / Normal force and CP location."),
    ("Solver & Flow / Mesh Quality", "Solver, turbulence model, Reynolds, cell/node counts, "
        "mesh quality and Y+ range."),
    ("Cp Distribution / Convergence Residuals / Solver Log", "Pressure-coefficient plot, "
        "residual history and raw solver output."),
  ]),
  ("Hand-off (right)", [
    ("Export VTK Results", "Write the solved field for ParaView."),
    ("Map Pressure to FEM", "Send the CFD surface pressure to the Structures tab as a load."),
  ]),
 ],
 ["Run a Sweep to get a drag polar across AoA/Mach; the simulation can use it."]),

("4. Structures", "03_structures.png",
 "CalculiX FEA — stress, deflection, buckling, modes and thermal.",
 [
  ("Material (left)", [
    ("Material", "Pick the airframe material (aluminium alloys, composites, …)."),
    ("Wall Thickness", "Shell thickness used by the FE model."),
  ]),
  ("Load Case (left)", [
    ("Load case", "<b>Max Thrust</b> / <b>Max-Q</b> / <b>Recovery Shock</b> / <b>Thermal</b> / "
        "<b>Custom</b>. Each preset auto-fills a representative Mach/altitude (from your last "
        "flight when one exists)."),
    ("Axial Force / Int. Pressure / Altitude / Static ΔT", "The load inputs. Editable in Custom; "
        "preset in the others."),
    ("Map Pressure from CFD", "Use the CFD surface-pressure field as the distributed load "
        "instead of a single axial force."),
  ]),
  ("Flight Loads (left)", [
    ("Import flight loads", "Pull the real peak loads (Max-Q etc.) from the last simulation."),
  ]),
  ("FEM Settings (left)", [
    ("Refinement", "Coarse … Ultra Fine / Custom mesh."),
    ("Circumferential divisions / axial per-caliber", "Custom mesh density controls."),
  ]),
  ("Run buttons (left)", [
    ("Run Static Analysis", "Stress + deformation under the load case."),
    ("Run Modal Analysis", "Natural frequencies and mode shapes."),
    ("Run Thermal Analysis", "Temperature field / thermal stress."),
    ("Export PDF Report", "Full report: stress contours, deformation, fin and thermal plots."),
  ]),
  ("Center result tabs", [
    ("3D Stress", "Interactive stress contour viewer (von Mises and component stresses)."),
    ("Stress Profile", "Von Mises stress vs position along the airframe."),
    ("Modal / Deformation / Fin Analysis / Buckling / Temperature", "Mode shapes, deflected "
        "shape, fin root stress + tip deflection, buckling modes, and thermal results."),
  ]),
  ("Right column", [
    ("Structural Safety Assessment", "<b>Good / Marginal / Poor</b> verdict (colour-coded) with safety factor."),
    ("Physics Checks / Airframe Modes / Stress Analysis / Buckling / Resonance / Fin Flutter / "
        "Thermal", "Per-phenomenon numeric breakdown."),
  ]),
 ],
 ["The smooth 3D stress map is an analytical estimate; the reported safety factor "
  "is FE-backed."]),

("5. Dynamics", "04_dynamics.png",
 "Aeroelasticity — flutter, divergence, vibration and the flight envelope.",
 [
  ("Flight Loads (Max-Q) (left)", [
    ("Max-Q loads", "Pulled from the trajectory — the worst-case dynamic pressure point."),
  ]),
  ("Run (left)", [
    ("Run Full Assessment", "Runs Flutter, Divergence/Aeroelastic, Vibration Response and "
        "Flight Envelope together."),
  ]),
  ("Results panels", [
    ("Flutter", "Fin flutter speed/margin vs flight Mach."),
    ("Divergence / Aeroelastic", "Static aeroelastic divergence margin."),
    ("Vibration Response", "Forced-response / resonance behaviour."),
    ("Flight Envelope", "Safe region vs the flight profile."),
    ("Consistency Checks", "Cross-checks between the phenomena and the flight data."),
    ("Flight Safety Assessment", "<b>Good / Marginal / Poor</b> banner (colour-coded)."),
  ]),
  ("Report Export (left)", [
    ("JSON / CSV / PDF", "Export the dynamics results."),
  ]),
 ],
 ["Check the fin flutter margin against your maximum flight Mach."]),

("6. Avionics", "05_avionics.png",
 "Flight-computer and sensor modelling.",
 [
  ("Sensor Configuration (left)", [
    ("IMU / Gyroscope", "Inertial sensor noise / bias model."),
    ("Barometer / Sensor settings", "Pressure-sensor model used for apogee detection."),
  ]),
  ("Flight Computer", [
    ("Flight Computer settings", "The simulated flight-computer / state machine."),
  ]),
  ("Live Telemetry Data", [
    ("Telemetry readouts", "Live values streamed from the flight computer during a run."),
  ]),
 ],
 []),

("7. Simulation", "06_simulation.png",
 "Mission Control — fly the rocket with the 6-DOF solver.",
 [
  ("Simulation Controls (top)", [
    ("RUN", "Start the flight. A dialog lets you pick the view: stay on the readouts, or jump "
        "to a 3D visualizer."),
    ("PAUSE", "Pause / resume the running simulation."),
    ("STOP", "Abort the run."),
    ("RESET", "Clear the simulation state and readouts."),
  ]),
  ("Environment (left)", [
    ("Launch Angle", "Rail tilt from vertical (90° = straight up)."),
    ("Launch Rod", "Rail/rod length — affects off-rod velocity and weathercocking."),
    ("Temperature", "Ground-level air temperature (sets the atmosphere)."),
  ]),
  ("Wind (left)", [
    ("Wind mode", "<b>Average Wind</b> (single) or a multi-level profile."),
    ("Avg Speed / Std Deviation / Turbulence / Direction", "Wind parameters."),
    ("Add Layer / Delete", "Build a multi-level wind profile (altitude bands)."),
  ]),
  ("Simulation Settings (left)", [
    ("Integrator", "Numerical integrator for the 6-DOF solver."),
    ("Time step", "Solver step size — smaller = more accurate, slower."),
  ]),
  ("Recovery System (left)", [
    ("Drogue / Main + deploy altitude", "Parachute deployment configuration."),
  ]),
  ("Live Flight Data + Flight Phase Timeline (right)", [
    ("Readouts", "Altitude, velocity, acceleration, thrust, drag, Mach, mass, dyn. pressure — "
        "update live during the flight."),
    ("Flight Phase Timeline", "Lights up Liftoff → Burnout → Apogee → Drogue → Main → Touchdown "
        "as events occur."),
  ]),
 ],
 ["Run a flight first — Results, Structures, Dynamics and Monte Carlo all read it."]),

("8. Mission Visualizer", "07_mission_visualizer.png",
 "Live 3D flight with telemetry, event flags and recovery status.",
 [
  ("3D scene (center)", [
    ("Live 3D flight", "Watch the rocket fly on a launch pad with a ground grid and trail."),
  ]),
  ("Playback bar (bottom)", [
    ("Play / Pause / Restart", "Control the replay."),
    ("Reset Camera", "Frame the whole scene."),
    ("Camera", "Camera mode (e.g. Chase)."),
    ("Trail", "Colour the flight trail by a channel (Mach, velocity, …)."),
    ("Alt Planes / Scale Bar / Envelope / Effects / Vectors / Flags / HUD", "Toggle scene "
        "overlays."),
    ("Target / Recov R / Quality / Stats", "Target altitude ring, recovery radius, render "
        "quality and an FPS/stats overlay."),
  ]),
  ("Telemetry panel (right)", [
    ("Telemetry / Flight Events / Timeline tabs", "Mission time, altitude, velocity, Mach, "
        "mass, thrust, drag, dyn. pressure, stability, phase, and the recovery block (state, "
        "descent rate, deploy altitude, canopy area)."),
  ]),
 ],
 []),

("9. Advanced Visualizer", "08_advanced_visualizer.png",
 "Webcast-style cinematic replay with a SpaceX-style HUD (three.js / WebEngine).",
 [
  ("Cinematic view", [
    ("Real-time 3D replay", "Follow-cameras and a broadcast HUD."),
    ("Mission timeline ribbon", "Liftoff, Max-Q, Burnout, Apogee, Drogue, Main, Touchdown — each "
        "shows the altitude it was reached at."),
    ("Replay scrubber + speed", "Scrub the flight; <b>LIVE</b> jumps back to the latest run."),
  ]),
 ],
 ["Needs QtWebEngine installed; otherwise the tab shows an install hint."]),

("10. Results", "09_results.png",
 "Full flight history across every channel.",
 [
  ("Plots (center)", [
    ("Channel plots", "Scrub the entire flight: altitude, velocity, Mach, acceleration, dynamic "
        "pressure, stability, and more."),
  ]),
  ("Controls", [
    ("Refresh Plots", "Re-read the last simulation run."),
    ("Export CSV", "Save the full flight data to CSV."),
  ]),
 ],
 []),

("11. Monte Carlo", "10_monte_carlo.png",
 "Uncertainty and reliability over many randomized flights.",
 [
  ("Inputs (left)", [
    ("Launch Uncertainty", "Spread on launch angle / conditions."),
    ("Motor Uncertainty", "Spread on motor performance."),
    ("Simulation Count", "Number of randomized runs (more = tighter stats, longer runtime)."),
    ("Target", "Target apogee to score against."),
    ("Failure Criteria", "What counts as a failed flight."),
  ]),
  ("Run / Export (left)", [
    ("RUN ANALYSIS", "Run the batch."),
    ("CANCEL", "Stop a running batch."),
    ("EXPORT CSV", "Per-run data."),
    ("EXPORT PDF", "Stats + distribution plots."),
  ]),
  ("Results", [
    ("Apogee / Landing / Performance Statistics", "Mean, spread and percentiles."),
    ("Reliability Analysis / Failure Breakdown / Mission Assessment", "Success probability, "
        "failure causes, overall verdict."),
    ("Best Case / Worst Case", "Jump to the extreme runs."),
  ]),
 ],
 ["More runs → tighter statistics, longer runtime."]),

("12. Optimization", "11_optimization.png",
 "Multidisciplinary design optimization (MDO).",
 [
  ("Setup (left)", [
    ("Algorithm", "GA, NSGA-II (multi-objective), Differential Evolution or Particle Swarm."),
    ("Optimization Mode", "Standard, <b>Mission Target</b> (hit an apogee), or Robust."),
    ("Target Apogee", "Goal altitude for Mission Target mode (0 = auto)."),
    ("Surrogate Model", "Optional surrogate to speed up evaluations."),
    ("Monte Carlo Settings", "Sampling used for robust / reliability objectives."),
    ("Design Variables", "Enable variables and set their bounds (Select All / Deselect All)."),
    ("Objectives", "Pick objectives and weight/direction; statistical objectives expose a mode "
        "(mean / std / worst / reliability / p5)."),
    ("Constraints", "e.g. Stability &gt; 1.2 and &lt; 3.0."),
  ]),
  ("Actions (left)", [
    ("START OPTIMIZATION", "Run the optimizer."),
    ("CANCEL", "Stop the run."),
    ("CSV / JSON / PDF", "Export results."),
  ]),
  ("Result tabs (center)", [
    ("Single / Multi Objective", "Convergence and objective space."),
    ("Design Space Explorer", "Scatter the sampled designs; colour by Fitness / Feasibility / "
        "Apogee / Stability / Mach."),
    ("Pareto Front", "Trade-off curve for multi-objective runs; click a point to inspect it."),
    ("DOE & Surfaces", "Latin Hypercube / Full Factorial / Taguchi designs and response "
        "surfaces (Run DOE)."),
    ("Sensitivity", "Sobol indices / PRCC / Morris screening (Analyze)."),
    ("Trade Study", "Compare named configurations on a radar plot (Add Current Design / "
        "Add Best Optimized / Compare / Clear)."),
    ("History", "Every evaluated design."),
  ]),
  ("Summary (right)", [
    ("Best Design — Parameters", "Winning design variables."),
    ("Performance / Reliability & Uncertainty", "Apogee, P(Target Alt), etc."),
    ("Improvement Over Baseline / Constraint Status / Optimization Statistics", "How much better, "
        "and whether constraints are met."),
    ("Pareto Solutions", "Shortcuts: Best Apogee / Best Reliability / Best Mass Efficiency / "
        "Best Balanced."),
  ]),
 ],
 ["Set a Target Apogee (Mission Target mode) to make the optimizer hit a number "
  "rather than just maximize altitude."]),
]


def build(path):
    doc = SimpleDocTemplate(str(path), pagesize=A4,
                            leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=14 * mm,
                            title="K2 AeroSim — Walkthrough")
    story = []

    # ── Cover ──
    story.append(Spacer(1, 40))
    logo = HERE / "k2_logo.png"
    if logo.exists():
        lw = 34 * mm
        img = Image(str(logo), width=lw, height=lw)
        img.hAlign = "CENTER"
        story.append(img)
        story.append(Spacer(1, 14))
    story.append(Paragraph("K2 AeroSim", TITLE))
    story.append(Spacer(1, 6))
    story.append(Paragraph("Software Walkthrough — every workspace, every control", SUB))
    story.append(Spacer(1, 4))
    story.append(Paragraph(f"v0.1.1 &nbsp;·&nbsp; {datetime.now():%Y-%m-%d}", SUB))
    story.append(Spacer(1, 26))
    story.append(HRFlowable(width="60%", thickness=1.2, color=BLUE))
    story.append(Spacer(1, 18))
    story.append(Paragraph(
        "K2 AeroSim is an integrated, multi-physics platform for high-power and experimental "
        "rockets: airframe design, propulsion, CFD, structural FEM, flight dynamics, avionics, "
        "6-DOF simulation and recovery — all sharing one live rocket model. This manual walks "
        "every workspace left-to-right across the tab bar and documents each button, input box, "
        "dropdown and checkbox.", BODY))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Recommended order", H2))
    story.append(Paragraph(
        "Design → Propulsion → (CFD for drag) → <b>Simulation</b> (fly it) → Results → "
        "Structures &amp; Dynamics (safety) → Monte Carlo (reliability) → Optimization.", BODY))

    # ── Global toolbar ──
    story.append(Spacer(1, 12))
    story.append(Paragraph("The top toolbar (always visible)", H2))
    story.append(control_table(TOOLBAR))
    story.append(PageBreak())

    # ── Each workspace ──
    for num, (title, img, purpose, panels, tips) in enumerate(WORKSPACES):
        story.append(Paragraph(title, H1))
        story.append(Paragraph(f"<i>{purpose}</i>", SMALL))
        story.append(HRFlowable(width="100%", thickness=0.6, color=LIGHT))
        story.append(Spacer(1, 5))
        story.append(shot(img))
        story.append(Paragraph(f"Figure {num + 1}. The {title.split('. ', 1)[1]} workspace.", CAP))
        story.append(Spacer(1, 6))
        for panel, rows in panels:
            story.append(KeepTogether([Paragraph(panel, H2), control_table(rows)]))
            story.append(Spacer(1, 4))
        for t in tips:
            story.append(Spacer(1, 2))
            story.append(Paragraph(f"<b>&#9656; Tip:</b> {t}", SMALL))
        story.append(PageBreak())

    # ── Closing ──
    story.append(Paragraph("Saving &amp; exporting", H1))
    story.append(HRFlowable(width="100%", thickness=0.6, color=LIGHT))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "Only the design and component tree are stored in a <b>.k2</b> project — all analyses "
        "recompute from the design. To keep CFD, Structures, Dynamics, Monte Carlo or "
        "Optimization results, use that tab's <b>Export</b> (PDF / CSV / JSON). Free and open "
        "source under GPL-3.0.", BODY))

    doc.build(story)
    return path


if __name__ == "__main__":
    out = HERE / "K2_AeroSim_Walkthrough.pdf"
    build(out)
    print("written", out)
