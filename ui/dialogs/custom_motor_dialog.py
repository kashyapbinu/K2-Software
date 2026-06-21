"""
K2 AeroSim — Custom (Solid) Motor Builder
============================================
Simulates a solid-motor grain's internal ballistics and now also reports the
same class of conceptual-design output the Liquid Engine Designer provides
where it makes sense for a solid: nozzle & chamber geometry with a live 2D
profile, ideal-vs-delivered performance, combustion-chamber metrics, ambient
optimisation, a propellant/inert mass breakdown, Δv / thrust-to-weight and a
sensitivity sweep, plus engineering validation warnings — organised into
collapsible sections and tabs that match the liquid dialog. Applies to the
rocket through the same `motor_created` signal.
"""

import csv
import logging
import math

from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QPushButton, QLabel, QGroupBox, QSpinBox, QDoubleSpinBox, QWidget,
    QComboBox, QFileDialog, QMessageBox, QGridLayout, QScrollArea, QFrame,
    QTabWidget)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush

from ui.widgets.design_widgets import (CollapsibleBox, MetricGrid, MplCanvas,
    ACCENT, MUTED, WARN, GOOD, BG, PANEL)
from physics.internal_ballistics import (Propellant, BatesGrain, TubularGrain,
    EndBurnerGrain, StarGrain, MotorSimulator)

logger = logging.getLogger("K2.CustomMotor")

P_ATM = 101325.0


class GrainRenderer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(200, 200)
        self.outer_d = 38.0
        self.inner_d = 15.0
        self.regression = 0.0
        self.grain_type = 0  # 0: BATES, 1: Tubular, 2: End-Burner, 3: Star
        self.star_points = 5
        self.star_depth = 5.0

    def update_dimensions(self, outer_d_mm, inner_d_mm, regression_mm=0.0,
                          grain_type=0, star_points=5, star_depth=5.0):
        self.outer_d = outer_d_mm
        self.inner_d = inner_d_mm
        self.regression = regression_mm
        self.grain_type = grain_type
        self.star_points = star_points
        self.star_depth = star_depth
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        painter.fillRect(0, 0, w, h, QColor("#1e1e1e"))
        cx, cy = w / 2, h / 2
        max_d = max(self.outer_d, 1.0)
        scale = min(w, h) * 0.8 / max_d
        r_out = (self.outer_d * scale) / 2

        painter.setPen(QPen(QColor("#555555"), 2))
        painter.setBrush(QBrush(QColor("#808080")))
        painter.drawEllipse(int(cx - r_out), int(cy - r_out), int(r_out * 2), int(r_out * 2))

        if self.grain_type in [0, 1]:  # BATES or Tubular (core burners)
            current_inner_d = min(self.inner_d + 2 * self.regression, self.outer_d)
            r_in = (current_inner_d * scale) / 2
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor("#1e1e1e")))
            painter.drawEllipse(int(cx - r_in), int(cy - r_in), int(r_in * 2), int(r_in * 2))
        elif self.grain_type == 2:  # End-Burner
            if self.regression >= self.outer_d * 5:
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(QColor("#1e1e1e")))
                painter.drawEllipse(int(cx - r_out), int(cy - r_out), int(r_out * 2), int(r_out * 2))
        elif self.grain_type == 3:  # Star
            from PyQt6.QtGui import QPolygonF
            from PyQt6.QtCore import QPointF
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor("#1e1e1e")))
            poly = QPolygonF()
            N = self.star_points
            r_inner = self.inner_d / 2.0
            r_outer = r_inner + self.star_depth
            if r_outer > self.outer_d / 2.0:
                r_outer = self.outer_d / 2.0
            r_in_draw = min((r_inner + self.regression) * scale, r_out)
            r_out_draw = min((r_outer + self.regression) * scale, r_out)
            for i in range(N * 2):
                angle = i * math.pi / N
                r = r_out_draw if i % 2 == 0 else r_in_draw
                poly.append(QPointF(cx + r * math.sin(angle), cy - r * math.cos(angle)))
            painter.drawPolygon(poly)
        painter.end()


class CustomMotorDialog(QDialog):

    motor_created = pyqtSignal(dict)  # emits the simulation result dict

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Custom Motor Builder")
        self.setMinimumSize(1180, 760)
        self.setStyleSheet("background-color:#1a1a1a; color:#ffffff;")
        self._last_result = None
        self._summary = None

        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(self._on_anim_tick)
        self._anim_idx = 0

        root = QHBoxLayout(self)
        root.addWidget(self._build_inputs(), 0)
        root.addWidget(self._build_outputs(), 1)

        self._on_grain_type_changed(0)
        self._update_visual()

    # ── input panel ──────────────────────────────────────────────────────
    def _build_inputs(self):
        panel = QWidget()
        panel.setFixedWidth(380)
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget()
        left = QVBoxLayout(inner)
        scroll.setWidget(inner)
        outer.addWidget(scroll, 1)

        # Propellant
        prop_group = QGroupBox("Propellant Parameters")
        pf = QFormLayout(prop_group)
        self.spin_a = QDoubleSpinBox(); self.spin_a.setDecimals(6)
        self.spin_a.setRange(0, 1.0); self.spin_a.setValue(0.000015)
        self.spin_a.setSingleStep(0.000001)
        self.spin_n = QDoubleSpinBox(); self.spin_n.setDecimals(3)
        self.spin_n.setRange(0, 0.999); self.spin_n.setValue(0.350)
        self.spin_rho = QDoubleSpinBox(); self.spin_rho.setRange(500, 3000)
        self.spin_rho.setValue(1600.0)
        self.spin_cstar = QDoubleSpinBox(); self.spin_cstar.setRange(500, 3000)
        self.spin_cstar.setValue(1400.0)
        self.spin_gamma = QDoubleSpinBox(); self.spin_gamma.setDecimals(3)
        self.spin_gamma.setRange(1.05, 1.40); self.spin_gamma.setValue(1.20)
        pf.addRow("Burn Rate Coeff 'a':", self.spin_a)
        pf.addRow("Pressure Exp 'n':", self.spin_n)
        pf.addRow("Density (kg/m³):", self.spin_rho)
        pf.addRow("C* (m/s):", self.spin_cstar)
        pf.addRow("γ (gamma):", self.spin_gamma)
        left.addWidget(prop_group)

        # Grain
        grain_group = QGroupBox("Grain Configuration")
        grain_layout = QFormLayout(grain_group)
        self.combo_grain_type = QComboBox()
        self.combo_grain_type.addItems(["BATES (Standard)", "Tubular", "End-Burner", "Star"])
        self.combo_grain_type.currentIndexChanged.connect(self._on_grain_type_changed)
        grain_layout.addRow("Grain Type:", self.combo_grain_type)
        self.spin_gcount = QSpinBox(); self.spin_gcount.setRange(1, 10); self.spin_gcount.setValue(3)
        self.spin_glen = QDoubleSpinBox(); self.spin_glen.setRange(10, 1000); self.spin_glen.setValue(100.0)
        self.spin_god = QDoubleSpinBox(); self.spin_god.setRange(10, 500); self.spin_god.setValue(38.0)
        self.spin_gid = QDoubleSpinBox(); self.spin_gid.setRange(2, 400); self.spin_gid.setValue(15.0)
        self.spin_star_points = QSpinBox(); self.spin_star_points.setRange(3, 12); self.spin_star_points.setValue(5)
        self.spin_star_depth = QDoubleSpinBox(); self.spin_star_depth.setRange(1, 200); self.spin_star_depth.setValue(5.0)
        self.row_gcount = grain_layout.rowCount()
        grain_layout.addRow("Number of Grains:", self.spin_gcount)
        grain_layout.addRow("Grain Length (mm):", self.spin_glen)
        grain_layout.addRow("Outer Diameter (mm):", self.spin_god)
        self.row_gid = grain_layout.rowCount()
        grain_layout.addRow("Core Diameter (mm):", self.spin_gid)
        self.row_star_points = grain_layout.rowCount()
        grain_layout.addRow("Star Points:", self.spin_star_points)
        self.row_star_depth = grain_layout.rowCount()
        grain_layout.addRow("Point Depth (mm):", self.spin_star_depth)
        left.addWidget(grain_group)
        self._grain_layout = grain_layout
        for s in (self.spin_god, self.spin_gid, self.spin_star_points, self.spin_star_depth):
            s.valueChanged.connect(self._update_visual)

        # Nozzle + ambient
        nozzle_group = QGroupBox("Nozzle & Ambient")
        nf = QFormLayout(nozzle_group)
        self.spin_nthroat = QDoubleSpinBox(); self.spin_nthroat.setRange(1, 100); self.spin_nthroat.setValue(9.0)
        self.spin_nexit = QDoubleSpinBox(); self.spin_nexit.setRange(1, 200); self.spin_nexit.setValue(20.0)
        self.combo_ambient = QComboBox()
        self.combo_ambient.addItems(["Sea level", "Vacuum", "Custom pressure"])
        self.combo_ambient.currentIndexChanged.connect(self._on_ambient_changed)
        self.spin_pamb = QDoubleSpinBox(); self.spin_pamb.setDecimals(3)
        self.spin_pamb.setRange(0.0, 1.5); self.spin_pamb.setValue(1.013); self.spin_pamb.setSuffix(" bar")
        self.spin_eff = QDoubleSpinBox(); self.spin_eff.setDecimals(3)
        self.spin_eff.setRange(0.7, 1.0); self.spin_eff.setValue(0.95)
        self.spin_struct = QDoubleSpinBox(); self.spin_struct.setDecimals(2)
        self.spin_struct.setRange(0.05, 2.0); self.spin_struct.setValue(0.50); self.spin_struct.setSuffix(" inert/prop")
        nf.addRow("Throat Dia (mm):", self.spin_nthroat)
        nf.addRow("Exit Dia (mm):", self.spin_nexit)
        nf.addRow("Design ambient:", self.combo_ambient)
        nf.addRow("Custom Pa:", self.spin_pamb)
        nf.addRow("Nozzle efficiency:", self.spin_eff)
        nf.addRow("Inert mass frac:", self.spin_struct)
        left.addWidget(nozzle_group)
        self.spin_pamb.setVisible(False)

        left.addStretch()

        btn_sim = QPushButton("SIMULATE BALLISTICS")
        btn_sim.setStyleSheet("background:#0078D7; color:white; padding:10px; font-weight:bold;")
        btn_sim.clicked.connect(self._run_sim)
        outer.addWidget(btn_sim)
        bh = QHBoxLayout()
        self.btn_apply = QPushButton("APPLY TO ROCKET")
        self.btn_apply.setStyleSheet("background:#28a745; color:white; padding:8px; font-weight:bold;")
        self.btn_apply.setEnabled(False)
        self.btn_apply.clicked.connect(self._apply)
        self.btn_export = QPushButton("EXPORT CSV")
        self.btn_export.setStyleSheet("background:#555; color:white; padding:8px; font-weight:bold;")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self._export_csv)
        bh.addWidget(self.btn_apply); bh.addWidget(self.btn_export)
        outer.addLayout(bh)
        return panel

    # ── output panel ─────────────────────────────────────────────────────
    def _build_outputs(self):
        tabs = QTabWidget()

        # Summary
        sa = QScrollArea(); sa.setWidgetResizable(True); sa.setFrameShape(QFrame.Shape.NoFrame)
        host = QWidget(); sv = QVBoxLayout(host); sa.setWidget(host)

        self.g_ball = MetricGrid(["Initial Kn", "Max Kn", "Max Pc", "Vol loading",
            "Port/Throat", "Throat/Port", "Core L/D", "Web", "Burn time",
            "Prop length", "Prop mass", "Peak mass flux", "Motor class"])
        self.g_geom = MetricGrid(["Throat Ø", "Exit Ø", "Chamber Ø", "Chamber length",
            "Expansion ratio", "Contraction ratio", "L* (gross)", "Nozzle length",
            "Residence time", "Exit pressure", "Optimum ε"])
        self.g_perf = MetricGrid(["Ideal Isp", "Delivered Isp", "Isp (vac)",
            "Cf (SL)", "Cf (vac)", "Overall efficiency", "Loss %",
            "Ideal thrust", "Delivered thrust", "Avg thrust", "Total impulse"])
        self.g_dv = MetricGrid(["Total impulse", "Prop mass", "Wet mass", "Dry mass",
            "Mass ratio", "Delta-V", "Engine TWR", "Liftoff TWR", "Burnout TWR"])

        for title, grid in [("Internal ballistics", self.g_ball),
                            ("Nozzle & chamber geometry", self.g_geom),
                            ("Performance — ideal vs delivered", self.g_perf),
                            ("Delta-V & thrust/weight", self.g_dv)]:
            box = CollapsibleBox(title); box.add_widget(grid); sv.addWidget(box)
        self.warn_box = CollapsibleBox("Engineering validation")
        self.lbl_warn = QLabel("Simulate a motor to see validation checks.")
        self.lbl_warn.setWordWrap(True); self.lbl_warn.setStyleSheet(f"color:{MUTED};")
        self.warn_box.add_widget(self.lbl_warn); sv.addWidget(self.warn_box)
        sv.addStretch()
        tabs.addTab(sa, "Summary")

        # Grain cross-section (existing renderer + burn animation)
        self.renderer = GrainRenderer()
        gw = QWidget(); gl = QVBoxLayout(gw)
        gl.addWidget(QLabel("Grain cross-section (animates burn regression after a run)"))
        gl.addWidget(self.renderer, 1)
        tabs.addTab(gw, "Grain")

        # Nozzle schematic
        self.canvas_schem = MplCanvas(toolbar=True)
        tabs.addTab(self.canvas_schem, "Nozzle Schematic")

        # Thrust curve
        thr = QWidget(); tl = QVBoxLayout(thr)
        self.combo_graph = QComboBox()
        self.combo_graph.addItems(["Thrust vs Time", "Pressure vs Time",
                                   "Kn vs Time", "Mass Flux vs Time"])
        self.combo_graph.currentIndexChanged.connect(self._draw_curve)
        self.canvas_curve = MplCanvas(toolbar=True)
        self.canvas_curve.canvas.mpl_connect("motion_notify_event", self._on_hover)
        self._hover_annot = None
        tl.addWidget(self.combo_graph); tl.addWidget(self.canvas_curve, 1)
        tabs.addTab(thr, "Thrust Curve")

        # Mass breakdown
        self.canvas_pie = MplCanvas(toolbar=False)
        tabs.addTab(self.canvas_pie, "Mass Breakdown")

        # Sensitivity
        self.canvas_sens = MplCanvas(toolbar=True)
        tabs.addTab(self.canvas_sens, "Sensitivity")

        self.tabs = tabs
        return tabs

    # ── input helpers ──────────────────────────────────────────────────────
    def _on_ambient_changed(self, *_):
        self.spin_pamb.setVisible(self.combo_ambient.currentText() == "Custom pressure")

    def _ambient_pa(self):
        mode = self.combo_ambient.currentText()
        if mode == "Vacuum":
            return 0.0
        if mode == "Custom pressure":
            return self.spin_pamb.value() * 1e5
        return P_ATM

    def _rocket_mass(self):
        try:
            return float(self.parent().engine.state.total_mass())
        except Exception:
            return 0.0

    def _on_grain_type_changed(self, idx):
        layout = self._grain_layout

        def show(row, w, vis):
            layout.itemAt(row, QFormLayout.ItemRole.LabelRole).widget().setVisible(vis)
            w.setVisible(vis)
        show(self.row_gcount, self.spin_gcount, idx == 0)
        show(self.row_gid, self.spin_gid, idx in (0, 1, 3))
        show(self.row_star_points, self.spin_star_points, idx == 3)
        show(self.row_star_depth, self.spin_star_depth, idx == 3)
        self._update_visual()

    def _update_visual(self):
        self.renderer.update_dimensions(
            self.spin_god.value(), self.spin_gid.value(), 0.0,
            self.combo_grain_type.currentIndex(),
            self.spin_star_points.value(), self.spin_star_depth.value())

    def _on_anim_tick(self):
        if not self._last_result or "regression" not in self._last_result:
            self._anim_timer.stop(); return
        regs = self._last_result["regression"]
        if self._anim_idx >= len(regs):
            self._anim_timer.stop(); return
        self.renderer.update_dimensions(
            self.spin_god.value(), self.spin_gid.value(), regs[self._anim_idx] * 1000.0,
            self.combo_grain_type.currentIndex(),
            self.spin_star_points.value(), self.spin_star_depth.value())
        self._anim_idx += max(1, len(regs) // 100)

    # ── build a simulator from current inputs (reused for sensitivity) ──────
    def _build_grains(self):
        g_len = self.spin_glen.value() / 1000.0
        g_od = self.spin_god.value() / 1000.0
        g_id = self.spin_gid.value() / 1000.0
        g_type = self.combo_grain_type.currentIndex()
        if g_type == 0:
            return [BatesGrain(length=g_len, outer_diameter=g_od, core_diameter=g_id)] * self.spin_gcount.value()
        if g_type == 1:
            return [TubularGrain(length=g_len, outer_diameter=g_od, core_diameter=g_id)]
        if g_type == 2:
            return [EndBurnerGrain(length=g_len, diameter=g_od)]
        pts = self.spin_star_points.value()
        dep = self.spin_star_depth.value() / 1000.0
        return [StarGrain(length=g_len, outer_diameter=g_od, core_diameter=g_id,
                          points=pts, point_depth=dep)]

    def _build_sim(self, throat_d=None, a_mult=1.0, cstar_mult=1.0):
        prop = Propellant(a=self.spin_a.value() * a_mult, n=self.spin_n.value(),
                          density=self.spin_rho.value(),
                          c_star=self.spin_cstar.value() * cstar_mult,
                          gamma=self.spin_gamma.value())
        t = (throat_d if throat_d is not None else self.spin_nthroat.value() / 1000.0)
        return MotorSimulator(propellant=prop, grains=self._build_grains(),
                              throat_diameter=t, exit_diameter=self.spin_nexit.value() / 1000.0,
                              ambient_pressure=self._ambient_pa(), efficiency=self.spin_eff.value())

    # ── run ─────────────────────────────────────────────────────────────────
    def _run_sim(self):
        try:
            sim = self._build_sim()
            res = sim.simulate(dt=0.01)
            g_type = self.combo_grain_type.currentIndex()
            res["motor_name"] = "Custom " + ["BATES", "Tubular", "End-Burner", "Star"][g_type]
            self._sim = sim
            self._last_result = res
            self._summary = sim.design_summary(
                res, chamber_diameter=self.spin_god.value() / 1000.0,
                rocket_total_mass=self._rocket_mass(), struct_frac=self.spin_struct.value())
        except Exception as e:
            logger.exception("sim failed")
            QMessageBox.critical(self, "Simulation Error", str(e))
            return

        self._refresh_metrics()
        self._draw_curve()
        self._draw_schematic()
        self._draw_pie()
        self._draw_sensitivity()

        # burn animation
        self._anim_idx = 0
        frames = min(len(res["regression"]), 100)
        self._anim_timer.start(max(20, 2000 // frames if frames > 0 else 20))

        impulse = res["metrics"]["total_impulse"]
        max_pc = res["metrics"].get("max_pc", 0.0)
        if impulse < 0.1 or max_pc <= P_ATM * 1.01:
            self.btn_apply.setEnabled(False); self.btn_export.setEnabled(True)
            QMessageBox.warning(self, "Motor Did Not Fire",
                "The motor never reached a sustainable chamber pressure, so thrust "
                f"is ~0 for the whole burn (total impulse {impulse:.2f} N·s).\n\nThe "
                "burn area is too small for this throat. Try a smaller throat, more "
                "grains, or a more energetic propellant (higher a / C*).")
            return
        self.btn_apply.setEnabled(True); self.btn_export.setEnabled(True)

    def _refresh_metrics(self):
        res, m, s = self._last_result, self._last_result["metrics"], self._summary

        def mm(x): return f"{x*1000:.1f} mm"

        self.g_ball.set("Initial Kn", f"{m['initial_kn']:.1f}")
        self.g_ball.set("Max Kn", f"{m['max_kn']:.1f}")
        self.g_ball.set("Max Pc", f"{m['max_pc']/1e5:.1f} bar")
        self.g_ball.set("Vol loading", f"{m['vol_loading']:.1f} %")
        self.g_ball.set("Port/Throat", f"{m['port_to_throat']:.2f}")
        self.g_ball.set("Throat/Port", f"{m['throat_to_port']:.2f}")
        self.g_ball.set("Core L/D", f"{m['core_l_d']:.2f}")
        self.g_ball.set("Web", mm(m["web"]))
        self.g_ball.set("Burn time", f"{res['time'][-1]:.2f} s")
        self.g_ball.set("Prop length", mm(m["prop_len"]))
        self.g_ball.set("Prop mass", f"{res['prop_mass']*1000:.0f} g")
        self.g_ball.set("Peak mass flux", f"{m['peak_mass_flux']:.0f} kg/m²s")
        self.g_ball.set("Motor class", self._motor_class(m["total_impulse"]))

        self.g_geom.set("Throat Ø", mm(s["throat_diameter"]))
        self.g_geom.set("Exit Ø", mm(s["exit_diameter"]))
        self.g_geom.set("Chamber Ø", mm(s["chamber_diameter"]))
        self.g_geom.set("Chamber length", mm(s["chamber_length"]))
        self.g_geom.set("Expansion ratio", f"{s['expansion_ratio']:.2f}")
        self.g_geom.set("Contraction ratio", f"{s['contraction_ratio']:.1f}")
        self.g_geom.set("L* (gross)", f"{s['l_star']:.2f} m")
        self.g_geom.set("Nozzle length", mm(s["nozzle_length"]))
        self.g_geom.set("Residence time", f"{s['residence_time']*1e3:.2f} ms")
        self.g_geom.set("Exit pressure", f"{s['exit_pressure']/1e5:.2f} bar")
        self.g_geom.set("Optimum ε", f"{s['opt_expansion']:.1f}")

        loss = (1.0 - s["efficiency"]) * 100.0
        self.g_perf.set("Ideal Isp", f"{s['isp_ideal']:.1f} s")
        self.g_perf.set("Delivered Isp", f"{s['isp_delivered']:.1f} s")
        self.g_perf.set("Isp (vac)", f"{s['isp_vac']:.1f} s")
        self.g_perf.set("Cf (SL)", f"{s['cf_sl']:.3f}")
        self.g_perf.set("Cf (vac)", f"{s['cf_vac']:.3f}")
        self.g_perf.set("Overall efficiency", f"{s['efficiency']*100:.1f} %")
        self.g_perf.set("Loss %", f"{loss:.1f} %", WARN if loss > 12 else ACCENT)
        self.g_perf.set("Ideal thrust", f"{s['thrust_ideal']:.0f} N")
        self.g_perf.set("Delivered thrust", f"{s['thrust_delivered']:.0f} N")
        self.g_perf.set("Avg thrust", f"{s['avg_thrust']:.0f} N")
        self.g_perf.set("Total impulse", f"{m['total_impulse']:.1f} N·s")

        self.g_dv.set("Total impulse", f"{m['total_impulse']:.1f} N·s")
        self.g_dv.set("Prop mass", f"{s['prop_mass']*1000:.0f} g")
        self.g_dv.set("Wet mass", f"{s['wet_mass']:.2f} kg")
        self.g_dv.set("Dry mass", f"{s['dry_mass']:.2f} kg")
        self.g_dv.set("Mass ratio", f"{s['mass_ratio']:.2f}")
        self.g_dv.set("Delta-V", f"{s['delta_v']:.0f} m/s")
        self.g_dv.set("Engine TWR", f"{s['twr_engine']:.1f}")
        self.g_dv.set("Liftoff TWR", f"{s['twr_liftoff']:.2f}" if s["twr_liftoff"] else "— (link rocket)")
        self.g_dv.set("Burnout TWR", f"{s['twr_burnout']:.2f}" if s["twr_burnout"] else "—")

        w = s["warnings"]
        if not w:
            self.lbl_warn.setText("✓ No issues flagged — inputs within typical ranges.")
            self.lbl_warn.setStyleSheet(f"color:{GOOD};")
            self.warn_box.toggle.setText("Engineering validation  ✓")
        else:
            self.lbl_warn.setText("⚠ " + "\n\n⚠ ".join(w))
            self.lbl_warn.setStyleSheet(f"color:{WARN};")
            self.warn_box.toggle.setText(f"Engineering validation  ⚠ {len(w)}")

    @staticmethod
    def _motor_class(impulse):
        classes = [(0.625, '1/4A'), (1.25, '1/2A'), (2.50, 'A'), (5.0, 'B'),
                   (10.0, 'C'), (20.0, 'D'), (40.0, 'E'), (80.0, 'F'), (160.0, 'G'),
                   (320.0, 'H'), (640.0, 'I'), (1280.0, 'J'), (2560.0, 'K'),
                   (5120.0, 'L'), (10240.0, 'M'), (20480.0, 'N'), (40960.0, 'O')]
        if impulse > 40960.0:
            return "P+"
        for limit, cls in classes:
            if impulse <= limit:
                return cls
        return "Micro"

    # ── drawing ─────────────────────────────────────────────────────────────
    def _draw_curve(self):
        if not self._last_result:
            return
        c = self.canvas_curve
        c.ax.clear()
        res = self._last_result
        t = res["time"]
        idx = self.combo_graph.currentIndex()
        if idx == 1:
            y = [p / 1e5 for p in res["pressure"]]; col = "#FF4500"; ttl = "Pressure vs Time"; yl = "Pc (bar)"
        elif idx == 2:
            y = res["kn"]; col = "#32CD32"; ttl = "Kn vs Time"; yl = "Kn"
        elif idx == 3:
            y = res["mass_flux"]; col = "#FFD700"; ttl = "Mass Flux vs Time"; yl = "kg/m²s"
        else:
            y = res["thrust"]; col = "#00BFFF"; ttl = "Thrust vs Time"; yl = "Thrust (N)"
        c.style_ax(ttl, "Time (s)", yl)
        c.ax.plot(t, y, color=col, linewidth=1.6)
        c.ax.fill_between(t, y, alpha=0.12, color=col)
        self._hover_xy = (t, y); self._hover_annot = None
        c.figure.tight_layout(); c.canvas.draw()

    def _on_hover(self, event):
        if not getattr(self, "_hover_xy", None) or event.inaxes != self.canvas_curve.ax:
            return
        t, y = self._hover_xy
        if not t or event.xdata is None:
            return
        i = min(range(len(t)), key=lambda k: abs(t[k] - event.xdata))
        ax = self.canvas_curve.ax
        if self._hover_annot:
            self._hover_annot.remove()
        self._hover_annot = ax.annotate(
            f"t={t[i]:.2f}s\n{y[i]:.1f}", xy=(t[i], y[i]), xytext=(10, 10),
            textcoords="offset points", color="#e6edf3", fontsize=9,
            bbox=dict(boxstyle="round", fc="#21262d", ec="#30363d"))
        self.canvas_curve.canvas.draw_idle()

    def _draw_schematic(self):
        s = self._summary
        if not s:
            return
        c = self.canvas_schem; c.clear(); ax = c.ax
        c.style_ax("Chamber & Nozzle Profile (axisymmetric, mm)", "Axial (mm)", "Radius (mm)")
        u = 1000.0
        rc = s["chamber_diameter"] / 2 * u
        rt = s["throat_diameter"] / 2 * u
        re = s["exit_diameter"] / 2 * u
        lc = s["chamber_length"] * u
        lconv = max((rc - rt) / 0.577, 1.0)
        ln = s["nozzle_length"] * u
        xs = [0.0, lc, lc + lconv, lc + lconv + ln]
        ys = [rc, rc, rt, re]
        ax.plot(xs, ys, color=ACCENT, lw=2)
        ax.plot(xs, [-v for v in ys], color=ACCENT, lw=2)
        ax.fill_between(xs, ys, [-v for v in ys], color=ACCENT, alpha=0.10)
        ax.plot([0, xs[-1]], [0, 0], color=MUTED, ls="--", lw=0.8)
        ax.plot([0, 0], [-rc, rc], color="#ff7b72", lw=3)
        ax.annotate("forward closure", (0, rc), color=MUTED, fontsize=8, ha="left", va="bottom")
        ax.annotate("throat", (xs[2], rt), color=MUTED, fontsize=8, ha="center", va="bottom")
        ax.annotate("exit", (xs[3], re), color=MUTED, fontsize=8, ha="right", va="bottom")
        ax.set_aspect("equal", adjustable="datalim")
        c.figure.tight_layout(); c.canvas.draw()

    def _draw_pie(self):
        s = self._summary
        if not s:
            return
        c = self.canvas_pie; c.clear(); ax = c.ax
        ax.set_facecolor(BG)
        ax.set_title("Mass Breakdown", color=ACCENT, fontsize=12, fontweight="bold")
        inert = max(s["dry_mass"], 1e-6)
        case = 0.55 * inert; nozzle = 0.25 * inert; struct = 0.20 * inert
        vals = [s["prop_mass"], case, nozzle, struct]
        labels = ["Propellant", "Casing", "Nozzle", "Structural"]
        colors = ["#58a6ff", "#ff7b72", "#d2a8ff", "#8b949e"]
        vals, labels, colors = zip(*[(v, l, col) for v, l, col in
                                     zip(vals, labels, colors) if v > 1e-6])
        ax.pie(vals, labels=labels, colors=colors, autopct="%1.1f%%",
               textprops={"color": "#e6edf3", "fontsize": 9})
        ax.text(0, -1.35, f"Wet mass {s['wet_mass']:.2f} kg", ha="center", color=MUTED, fontsize=9)
        c.figure.tight_layout(); c.canvas.draw()

    def _draw_sensitivity(self):
        if not self._summary:
            return
        c = self.canvas_sens; c.clear(); ax = c.ax
        c.style_ax("Sensitivity — total impulse response to ±20%", "Impulse change (%)", "")
        base = self._last_result["metrics"]["total_impulse"]
        if base <= 0:
            c.canvas.draw(); return
        params = [("Throat Ø", "throat"), ("Burn rate a", "a"), ("C*", "cstar")]
        rows, lows, highs = [], [], []
        for label, key in params:
            vals = []
            for f in (0.8, 1.2):
                try:
                    if key == "throat":
                        sim = self._build_sim(throat_d=self.spin_nthroat.value() / 1000.0 * f)
                    elif key == "a":
                        sim = self._build_sim(a_mult=f)
                    else:
                        sim = self._build_sim(cstar_mult=f)
                    r = sim.simulate(dt=0.02)
                    vals.append(r["metrics"]["total_impulse"])
                except Exception:
                    vals.append(base)
            rows.append(label)
            lows.append((min(vals) - base) / base * 100.0)
            highs.append((max(vals) - base) / base * 100.0)
        y = range(len(rows))
        ax.barh(list(y), [h - l for h, l in zip(highs, lows)], left=lows,
                color=ACCENT, alpha=0.7, height=0.5)
        ax.axvline(0, color="#ff7b72", lw=1, ls="--")
        ax.set_yticks(list(y)); ax.set_yticklabels(rows, color="#c9d1d9")
        for i, (l, h) in enumerate(zip(lows, highs)):
            ax.text(h, i, f" {h:+.0f}%", va="center", color=MUTED, fontsize=8)
            ax.text(l, i, f"{l:+.0f}% ", va="center", ha="right", color=MUTED, fontsize=8)
        c.figure.tight_layout(); c.canvas.draw()

    # ── export / apply ───────────────────────────────────────────────────────
    def _export_csv(self):
        if not self._last_result:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export motor", "custom_motor.csv", "CSV Files (*.csv)")
        if not path:
            return
        res, m, s = self._last_result, self._last_result["metrics"], self._summary
        try:
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["Parameter", "Value", "Unit"])
                rows = [
                    ("Motor", res.get("motor_name", ""), ""),
                    ("Motor class", self._motor_class(m["total_impulse"]), ""),
                    ("Total impulse", m["total_impulse"], "N*s"),
                    ("Burn time", res["time"][-1], "s"),
                    ("Max Pc", m["max_pc"] / 1e5, "bar"),
                    ("Initial Kn", m["initial_kn"], ""),
                    ("Max Kn", m["max_kn"], ""),
                    ("Port/Throat", m["port_to_throat"], ""),
                    ("Peak mass flux", m["peak_mass_flux"], "kg/m2s"),
                    ("Prop mass", res["prop_mass"], "kg"),
                    ("Throat diameter", s["throat_diameter"] * 1000, "mm"),
                    ("Exit diameter", s["exit_diameter"] * 1000, "mm"),
                    ("Chamber diameter", s["chamber_diameter"] * 1000, "mm"),
                    ("Chamber length", s["chamber_length"] * 1000, "mm"),
                    ("Expansion ratio", s["expansion_ratio"], ""),
                    ("Contraction ratio", s["contraction_ratio"], ""),
                    ("L* gross", s["l_star"], "m"),
                    ("Nozzle length", s["nozzle_length"] * 1000, "mm"),
                    ("Residence time", s["residence_time"] * 1000, "ms"),
                    ("Exit pressure", s["exit_pressure"] / 1e5, "bar"),
                    ("Optimum expansion", s["opt_expansion"], ""),
                    ("Ideal Isp", s["isp_ideal"], "s"),
                    ("Delivered Isp", s["isp_delivered"], "s"),
                    ("Vacuum Isp", s["isp_vac"], "s"),
                    ("Overall efficiency", s["efficiency"], ""),
                    ("Cf SL", s["cf_sl"], ""),
                    ("Cf vac", s["cf_vac"], ""),
                    ("Wet mass", s["wet_mass"], "kg"),
                    ("Dry mass", s["dry_mass"], "kg"),
                    ("Mass ratio", s["mass_ratio"], ""),
                    ("Delta-V", s["delta_v"], "m/s"),
                    ("Engine TWR", s["twr_engine"], ""),
                    ("Liftoff TWR", s["twr_liftoff"], ""),
                    ("Burnout TWR", s["twr_burnout"], ""),
                ]
                for r in rows:
                    w.writerow(r)
                w.writerow([])
                w.writerow(["Time (s)", "Thrust (N)", "Pressure (Pa)", "Kn", "Mass Flux (kg/m2s)"])
                for i in range(len(res["time"])):
                    w.writerow([res["time"][i], res["thrust"][i], res["pressure"][i],
                                res["kn"][i], res["mass_flux"][i]])
            QMessageBox.information(self, "Export Successful", "Data exported successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export: {str(e)}")

    def _apply(self):
        if self._last_result:
            self.motor_created.emit(self._last_result)
            self.accept()
