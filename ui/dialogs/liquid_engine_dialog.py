"""
K2 AeroSim — Liquid / Bi-Propellant Engine Designer
======================================================
Conceptual liquid rocket engine design environment. Sizes a liquid (or
monopropellant) engine from chamber pressure, mixture ratio and nozzle
expansion ratio and reports full preliminary-design output — nozzle & chamber
geometry, tank sizing, injector & cooling estimates, ideal-vs-delivered
performance, Δv / TWR, a sensitivity sweep and engineering validation
warnings — with live 2D schematics. Applies to the rocket through the same
`motor_created` signal the solid Custom Motor Builder uses, so it feeds
single-stage and per-stage multistage unchanged.
"""

import csv
import logging

from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from PyQt6.QtWidgets import (QDialog, QHBoxLayout, QVBoxLayout, QFormLayout,
    QGroupBox, QComboBox, QDoubleSpinBox, QSpinBox, QPushButton, QLabel,
    QGridLayout, QMessageBox, QFileDialog, QScrollArea, QWidget, QFrame,
    QTabWidget, QSizePolicy, QCheckBox, QToolButton)
from PyQt6.QtCore import pyqtSignal, Qt, QTimer

from ui.widgets.design_widgets import (CollapsibleBox, MetricGrid, MplCanvas,
    ACCENT, MUTED, WARN, GOOD, BG, PANEL)
from physics.liquid_propulsion import (LiquidEngineDesign, PropellantCombo,
    PROPELLANT_COMBOS, ENGINE_CYCLES, COOLING_METHODS, THRUST_PROFILES,
    OPT_MODES, ambient_pressure_at_altitude)

logger = logging.getLogger("K2.LiquidEngine")


# ──────────────────────────────────────────────────────────────────────────
class LiquidEngineDialog(QDialog):

    motor_created = pyqtSignal(dict)   # same schema as the solid Custom Motor

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Liquid Engine Designer")
        self.setMinimumSize(1180, 780)
        self.setStyleSheet("background-color:#1a1a1a; color:#ffffff;")
        self._result = None
        self._design = None
        self._loading = True

        # debounce live recompute
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(160)
        self._timer.timeout.connect(self._recompute)

        root = QHBoxLayout(self)
        root.addWidget(self._build_inputs(), 0)
        root.addWidget(self._build_outputs(), 1)

        self._on_combo_changed(self.combo_prop.currentText())
        self._on_ambient_changed()
        self._loading = False
        self._recompute()

    # ── input panel ────────────────────────────────────────────────────────
    def _build_inputs(self):
        panel = QWidget()
        panel.setFixedWidth(390)
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
        pg = QGroupBox("Propellant")
        pf = QFormLayout(pg)
        self.combo_prop = QComboBox()
        self.combo_prop.addItems(list(PROPELLANT_COMBOS.keys()))
        self.combo_prop.currentTextChanged.connect(self._on_combo_changed)
        pf.addRow("Combination:", self.combo_prop)
        self.spin_cstar = self._dspin(500, 3000, 1, 1823, " m/s")
        self.spin_gamma = self._dspin(1.05, 1.40, 3, 1.24)
        self.spin_of = self._dspin(0.0, 20.0, 2, 2.56, " O/F")
        self.spin_rho_ox = self._dspin(1.0, 2000, 0, 1141, " kg/m³")
        self.spin_rho_fuel = self._dspin(1.0, 2000, 0, 810, " kg/m³")
        pf.addRow("C* (eff. base):", self.spin_cstar)
        pf.addRow("γ (gamma):", self.spin_gamma)
        pf.addRow("Mixture ratio:", self.spin_of)
        pf.addRow("Ox density:", self.spin_rho_ox)
        pf.addRow("Fuel density:", self.spin_rho_fuel)
        left.addWidget(pg)

        # Optimisation
        og = QGroupBox("Mixture-ratio optimisation")
        of_ = QFormLayout(og)
        self.combo_opt = QComboBox()
        self.combo_opt.addItems(OPT_MODES)
        self.combo_opt.currentIndexChanged.connect(self._on_opt_changed)
        of_.addRow("Mode:", self.combo_opt)
        self.lbl_opt = QLabel("—")
        self.lbl_opt.setStyleSheet(f"color:{GOOD};")
        of_.addRow("Optimum O/F:", self.lbl_opt)
        left.addWidget(og)

        # Engine / nozzle
        cg = QGroupBox("Engine / Nozzle")
        cf = QFormLayout(cg)
        self.combo_cycle = QComboBox()
        self.combo_cycle.addItems(list(ENGINE_CYCLES.keys()))
        self.combo_cycle.currentIndexChanged.connect(self._mark_dirty)
        self.combo_cooling = QComboBox()
        self.combo_cooling.addItems(COOLING_METHODS)
        self.combo_cooling.currentIndexChanged.connect(self._mark_dirty)
        self.spin_pc = self._dspin(1.0, 300.0, 1, 70.0, " bar")
        self.spin_eps = self._dspin(1.5, 200.0, 1, 12.0, " Ae/At")
        self.spin_thrust = self._dspin(10.0, 1.0e7, 0, 5000.0, " N")
        self.spin_burn = self._dspin(0.5, 1000.0, 1, 10.0, " s")
        self.spin_cstar_eff = self._dspin(0.7, 1.0, 3, 0.95)
        self.spin_cf_eff = self._dspin(0.7, 1.0, 3, 0.97)
        self.spin_struct = self._dspin(0.05, 2.0, 2, 0.30, " dry/prop")
        cf.addRow("Engine cycle:", self.combo_cycle)
        cf.addRow("Cooling:", self.combo_cooling)
        cf.addRow("Chamber pressure:", self.spin_pc)
        cf.addRow("Expansion ratio:", self.spin_eps)
        cf.addRow("Target thrust:", self.spin_thrust)
        cf.addRow("Burn time:", self.spin_burn)
        cf.addRow("C* efficiency:", self.spin_cstar_eff)
        cf.addRow("Cf efficiency:", self.spin_cf_eff)
        cf.addRow("Struct. mass frac:", self.spin_struct)
        left.addWidget(cg)

        # Ambient
        ag = QGroupBox("Design Ambient")
        af = QFormLayout(ag)
        self.combo_ambient = QComboBox()
        self.combo_ambient.addItems(["Sea level", "Vacuum", "Custom pressure",
                                     "Optimum for altitude"])
        self.combo_ambient.currentIndexChanged.connect(self._on_ambient_changed)
        self.spin_pamb = self._dspin(0.0, 1.5, 3, 1.013, " bar")
        self.spin_alt = self._dspin(0.0, 100000.0, 0, 0.0, " m")
        self.chk_autoeps = QCheckBox("Auto ε for ambient (perfect expansion)")
        self.chk_autoeps.stateChanged.connect(self._mark_dirty)
        af.addRow("Mode:", self.combo_ambient)
        af.addRow("Custom Pa:", self.spin_pamb)
        af.addRow("Altitude:", self.spin_alt)
        af.addRow(self.chk_autoeps)
        left.addWidget(ag)

        # Geometry / tanks
        gg = QGroupBox("Chamber / Tanks")
        gf = QFormLayout(gg)
        self.spin_lstar = self._dspin(0.0, 3.0, 2, 0.0, " m (0=auto)")
        self.spin_tankd = self._dspin(0.0, 5.0, 3, 0.0, " m (0=auto)")
        self.spin_ullage = self._dspin(0.0, 0.5, 2, 0.06, " ullage")
        gf.addRow("L* :", self.spin_lstar)
        gf.addRow("Tank Ø:", self.spin_tankd)
        gf.addRow("Ullage frac:", self.spin_ullage)
        left.addWidget(gg)

        # Injector
        ig = QGroupBox("Injector")
        igf = QFormLayout(ig)
        self.spin_inj_dp = self._dspin(0.05, 0.5, 2, 0.20, " ΔP/Pc")
        self.spin_inj_cd = self._dspin(0.4, 0.95, 2, 0.75, " Cd")
        self.spin_inj_d = self._dspin(0.2, 5.0, 2, 1.0, " mm hole")
        igf.addRow("Pressure drop:", self.spin_inj_dp)
        igf.addRow("Discharge Cd:", self.spin_inj_cd)
        igf.addRow("Hole Ø:", self.spin_inj_d)
        left.addWidget(ig)

        # Thrust profile
        tg = QGroupBox("Thrust profile")
        tf = QFormLayout(tg)
        self.combo_profile = QComboBox()
        self.combo_profile.addItems(THRUST_PROFILES)
        self.combo_profile.currentIndexChanged.connect(self._on_profile_changed)
        self.btn_csv = QPushButton("Load CSV…")
        self.btn_csv.clicked.connect(self._load_csv)
        self.btn_csv.setEnabled(False)
        tf.addRow("Profile:", self.combo_profile)
        tf.addRow(self.btn_csv)
        left.addWidget(tg)

        # connect every numeric input to the debounced recompute
        for s in (self.spin_cstar, self.spin_gamma, self.spin_of, self.spin_rho_ox,
                  self.spin_rho_fuel, self.spin_pc, self.spin_eps, self.spin_thrust,
                  self.spin_burn, self.spin_cstar_eff, self.spin_cf_eff,
                  self.spin_struct, self.spin_pamb, self.spin_alt, self.spin_lstar,
                  self.spin_tankd, self.spin_ullage, self.spin_inj_dp,
                  self.spin_inj_cd, self.spin_inj_d):
            s.valueChanged.connect(self._mark_dirty)

        left.addStretch()

        # action buttons
        btns = QVBoxLayout()
        self.btn_apply = QPushButton("APPLY TO ROCKET")
        self.btn_apply.setStyleSheet("background:#2ea043; color:white; padding:8px; font-weight:bold;")
        self.btn_apply.clicked.connect(self._apply)
        self.btn_export = QPushButton("Export all parameters (CSV)")
        self.btn_export.clicked.connect(self._export_csv)
        btns.addWidget(self.btn_apply)
        btns.addWidget(self.btn_export)
        outer.addLayout(btns)
        return panel

    # ── output panel ────────────────────────────────────────────────────────
    def _build_outputs(self):
        tabs = QTabWidget()

        # Summary tab — collapsible sections in a scroll area
        sa = QScrollArea()
        sa.setWidgetResizable(True)
        sa.setFrameShape(QFrame.Shape.NoFrame)
        host = QWidget()
        sv = QVBoxLayout(host)
        sa.setWidget(host)

        self.g_perf = MetricGrid(["Ideal Isp", "Delivered Isp", "Isp (vac)",
            "Ideal thrust", "Delivered thrust", "Cf (ideal)", "Cf (delivered)",
            "Overall efficiency", "Loss %", "Mass flow", "Ox flow", "Fuel flow",
            "Exit pressure", "Total impulse"])
        self.g_geom = MetricGrid(["Throat Ø", "Exit Ø", "Chamber Ø", "Chamber length",
            "Nozzle length", "L*", "Contraction ratio", "Expansion ratio",
            "Residence time", "Combustion eff"])
        self.g_tank = MetricGrid(["Ox mass", "Fuel mass", "Ox volume", "Fuel volume",
            "Ullage volume", "Tank Ø", "Ox tank len", "Fuel tank len",
            "Tank mass", "Engine mass", "Wet mass", "Dry mass"])
        self.g_inj = MetricGrid(["Injector ΔP", "Total area", "Inj. velocity",
            "Hole count", "Hole Ø", "Pattern"])
        self.g_cool = MetricGrid(["Wall heat flux", "Coolant flow", "Cooling ΔP",
            "Cooling effectiveness"])
        self.g_dv = MetricGrid(["Total impulse", "Propellant mass", "Wet mass",
            "Dry mass", "Mass ratio", "Delta-V", "Engine TWR", "Liftoff TWR",
            "Burnout TWR"])

        for title, grid in [("Performance — ideal vs delivered", self.g_perf),
                            ("Nozzle & chamber geometry", self.g_geom),
                            ("Tank sizing & masses", self.g_tank),
                            ("Injector design", self.g_inj),
                            ("Cooling estimates", self.g_cool),
                            ("Delta-V & thrust/weight", self.g_dv)]:
            box = CollapsibleBox(title)
            box.add_widget(grid)
            sv.addWidget(box)

        self.warn_box = CollapsibleBox("Engineering validation")
        self.lbl_warn = QLabel("Run a design to see validation checks.")
        self.lbl_warn.setWordWrap(True)
        self.lbl_warn.setStyleSheet(f"color:{MUTED};")
        self.warn_box.add_widget(self.lbl_warn)
        sv.addWidget(self.warn_box)
        sv.addStretch()
        tabs.addTab(sa, "Summary")

        # Schematic
        self.canvas_schem = MplCanvas(toolbar=True)
        tabs.addTab(self.canvas_schem, "Nozzle Schematic")

        # Flow diagram
        self.canvas_flow = MplCanvas(toolbar=False)
        tabs.addTab(self.canvas_flow, "Flow Diagram")

        # Thrust curve
        thr_host = QWidget()
        tl = QVBoxLayout(thr_host)
        self.combo_graph = QComboBox()
        self.combo_graph.addItems(["Thrust vs Time", "Chamber Pressure vs Time"])
        self.combo_graph.currentIndexChanged.connect(self._draw_thrust)
        self.canvas_thrust = MplCanvas(toolbar=True)
        self.canvas_thrust.canvas.mpl_connect("motion_notify_event", self._on_hover)
        self._hover_annot = None
        tl.addWidget(self.combo_graph)
        tl.addWidget(self.canvas_thrust, 1)
        tabs.addTab(thr_host, "Thrust Curve")

        # Mass breakdown pie
        self.canvas_pie = MplCanvas(toolbar=False)
        tabs.addTab(self.canvas_pie, "Mass Breakdown")

        # Sensitivity
        self.canvas_sens = MplCanvas(toolbar=True)
        tabs.addTab(self.canvas_sens, "Sensitivity")

        self.tabs = tabs
        return tabs

    # ── helpers ──────────────────────────────────────────────────────────
    @staticmethod
    def _dspin(lo, hi, decimals, val, suffix=""):
        s = QDoubleSpinBox()
        s.setDecimals(decimals)
        s.setRange(lo, hi)
        s.setValue(val)
        if suffix:
            s.setSuffix(suffix)
        return s

    def _mark_dirty(self, *_):
        if not self._loading:
            self._timer.start()

    def _on_combo_changed(self, name):
        c = PROPELLANT_COMBOS.get(name)
        if not c:
            return
        self._loading = True
        self.spin_cstar.setValue(c.c_star)
        self.spin_gamma.setValue(c.gamma)
        self.spin_of.setValue(c.of_ratio)
        self.spin_rho_ox.setValue(c.rho_ox)
        self.spin_rho_fuel.setValue(c.rho_fuel)
        self.spin_of.setEnabled(not c.monoprop)
        self._loading = False
        self._mark_dirty()

    def _on_opt_changed(self, *_):
        self._mark_dirty()

    def _on_profile_changed(self, *_):
        self.btn_csv.setEnabled(self.combo_profile.currentText() == "User-defined CSV")
        self._mark_dirty()

    def _on_ambient_changed(self, *_):
        mode = self.combo_ambient.currentText()
        self.spin_pamb.setVisible(mode == "Custom pressure")
        self.spin_alt.setVisible(mode == "Optimum for altitude")
        self._mark_dirty()

    def _ambient_pa(self):
        mode = self.combo_ambient.currentText()
        if mode == "Vacuum":
            return 0.0
        if mode == "Custom pressure":
            return self.spin_pamb.value() * 1e5
        if mode == "Optimum for altitude":
            return ambient_pressure_at_altitude(self.spin_alt.value())
        return 101325.0

    def _rocket_mass(self):
        """Full-vehicle wet mass from the parent workspace, if reachable."""
        try:
            return float(self.parent().engine.state.total_mass())
        except Exception:
            return 0.0

    def _build_design(self):
        name = self.combo_prop.currentText()
        base = PROPELLANT_COMBOS[name]
        combo = PropellantCombo(
            name=name, c_star=self.spin_cstar.value(), Tc=base.Tc,
            gamma=self.spin_gamma.value(), mol_wt=base.mol_wt,
            of_ratio=self.spin_of.value(), rho_ox=self.spin_rho_ox.value(),
            rho_fuel=self.spin_rho_fuel.value(), monoprop=base.monoprop,
            l_star=base.l_star, of_min=base.of_min, of_max=base.of_max)
        d = LiquidEngineDesign(
            combo=combo,
            chamber_pressure=self.spin_pc.value() * 1e5,
            expansion_ratio=self.spin_eps.value(),
            target_thrust=self.spin_thrust.value(),
            burn_time=self.spin_burn.value(),
            of_ratio=self.spin_of.value(),
            ambient_pressure=self._ambient_pa(),
            cstar_eff=self.spin_cstar_eff.value(),
            cf_eff=self.spin_cf_eff.value(),
            struct_frac=self.spin_struct.value(),
            cycle=self.combo_cycle.currentText(),
            cooling=self.combo_cooling.currentText(),
            thrust_profile=self.combo_profile.currentText(),
            l_star=self.spin_lstar.value(),
            tank_diameter=self.spin_tankd.value(),
            ullage_frac=self.spin_ullage.value(),
            injector_cd=self.spin_inj_cd.value(),
            injector_dp_frac=self.spin_inj_dp.value(),
            injector_hole_d=self.spin_inj_d.value() / 1000.0,
            rocket_total_mass=self._rocket_mass())
        if self.combo_profile.currentText() == "Throttle schedule":
            d.throttle_points = [(0.0, 1.0), (0.4, 0.6), (0.75, 0.6), (1.0, 1.0)]
        if self.combo_profile.currentText() == "User-defined CSV":
            d.csv_curve = getattr(self, "_csv_curve", [])
        return d

    # ── recompute everything (live) ─────────────────────────────────────────
    def _recompute(self):
        try:
            d = self._build_design()
            # Apply optimisation / auto-ε before the final solve.
            opt = self.combo_opt.currentText()
            d.solve()
            if opt != "Off" and not d.combo.monoprop:
                best = d.optimum_of(opt)
                self.lbl_opt.setText(f"{best:.2f}")
                d.of_ratio = best
            else:
                self.lbl_opt.setText("—")
            if self.chk_autoeps.isChecked():
                eps = d.optimum_expansion()
                d.expansion_ratio = eps
                self._loading = True
                self.spin_eps.setValue(eps)
                self._loading = False
            self._result = d.simulate(dt=0.02)
            self._design = d
        except Exception as exc:
            logger.exception("design failed")
            self.lbl_warn.setText(f"Error: {exc}")
            return
        self._refresh_metrics()
        self._draw_all()

    def _refresh_metrics(self):
        d, m = self._design, self._result["metrics"]

        def mm(x): return f"{x*1000:.1f} mm"
        def kg(x): return f"{x:.2f} kg"

        loss = (1.0 - m["efficiency"]) * 100.0
        self.g_perf.set("Ideal Isp", f"{m['isp_ideal']:.1f} s")
        self.g_perf.set("Delivered Isp", f"{m['isp_sl']:.1f} s")
        self.g_perf.set("Isp (vac)", f"{m['isp_vac']:.1f} s")
        self.g_perf.set("Ideal thrust", f"{m['thrust_ideal']:.0f} N")
        self.g_perf.set("Delivered thrust", f"{d.target_thrust:.0f} N")
        self.g_perf.set("Cf (ideal)", f"{m['cf_ideal']:.3f}")
        self.g_perf.set("Cf (delivered)", f"{m['cf']:.3f}")
        self.g_perf.set("Overall efficiency", f"{m['efficiency']*100:.1f} %")
        self.g_perf.set("Loss %", f"{loss:.1f} %", WARN if loss > 12 else ACCENT)
        self.g_perf.set("Mass flow", f"{m['mdot']:.2f} kg/s")
        self.g_perf.set("Ox flow", f"{m['mdot_ox']:.2f} kg/s")
        self.g_perf.set("Fuel flow", f"{m['mdot_fuel']:.2f} kg/s")
        self.g_perf.set("Exit pressure", f"{m['exit_pressure']/1e5:.2f} bar")
        self.g_perf.set("Total impulse", f"{m['total_impulse']:.0f} N·s")

        self.g_geom.set("Throat Ø", mm(m["throat_diameter"]))
        self.g_geom.set("Exit Ø", mm(m["exit_diameter"]))
        self.g_geom.set("Chamber Ø", mm(m["chamber_diameter"]))
        self.g_geom.set("Chamber length", mm(m["chamber_length"]))
        self.g_geom.set("Nozzle length", mm(m["nozzle_length"]))
        self.g_geom.set("L*", f"{m['l_star']:.2f} m")
        self.g_geom.set("Contraction ratio", f"{m['contraction_ratio']:.2f}")
        self.g_geom.set("Expansion ratio", f"{m['expansion_ratio']:.1f}")
        self.g_geom.set("Residence time", f"{m['residence_time']*1e3:.2f} ms")
        self.g_geom.set("Combustion eff", f"{m['comb_efficiency']*100:.1f} %")

        self.g_tank.set("Ox mass", kg(m["ox_mass"]))
        self.g_tank.set("Fuel mass", kg(m["fuel_mass"]))
        self.g_tank.set("Ox volume", f"{m['ox_volume']*1000:.1f} L")
        self.g_tank.set("Fuel volume", f"{m['fuel_volume']*1000:.1f} L")
        self.g_tank.set("Ullage volume", f"{m['ullage_volume']*1000:.1f} L")
        self.g_tank.set("Tank Ø", mm(m["tank_dia"]))
        self.g_tank.set("Ox tank len", f"{m['ox_tank_len']:.2f} m")
        self.g_tank.set("Fuel tank len", f"{m['fuel_tank_len']:.2f} m")
        self.g_tank.set("Tank mass", kg(m["tank_mass"]))
        self.g_tank.set("Engine mass", kg(m["engine_mass"]))
        self.g_tank.set("Wet mass", kg(m["wet_mass"]))
        self.g_tank.set("Dry mass", kg(m["dry_mass"]))

        self.g_inj.set("Injector ΔP", f"{m['inj_dp']/1e5:.2f} bar")
        self.g_inj.set("Total area", f"{m['inj_area']*1e6:.1f} mm²")
        self.g_inj.set("Inj. velocity", f"{m['inj_velocity']:.1f} m/s")
        self.g_inj.set("Hole count", f"{m['inj_n_holes']}")
        self.g_inj.set("Hole Ø", f"{self.spin_inj_d.value():.2f} mm")
        self.g_inj.set("Pattern", m["inj_pattern"])

        self.g_cool.set("Wall heat flux", f"{m['heat_flux']/1e6:.1f} MW/m²")
        self.g_cool.set("Coolant flow", f"{m['coolant_flow']:.2f} kg/s")
        self.g_cool.set("Cooling ΔP", f"{m['cooling_dp']/1e5:.2f} bar")
        self.g_cool.set("Cooling effectiveness", f"{m['cooling_eff']*100:.0f} %")

        self.g_dv.set("Total impulse", f"{m['total_impulse']:.0f} N·s")
        self.g_dv.set("Propellant mass", kg(self._result["prop_mass"]))
        self.g_dv.set("Wet mass", kg(m["wet_mass"]))
        self.g_dv.set("Dry mass", kg(m["dry_mass"]))
        self.g_dv.set("Mass ratio", f"{m['mass_ratio']:.2f}")
        self.g_dv.set("Delta-V", f"{m['delta_v']:.0f} m/s")
        self.g_dv.set("Engine TWR", f"{m['twr_engine']:.1f}")
        self.g_dv.set("Liftoff TWR", f"{m['twr_liftoff']:.2f}" if m["twr_liftoff"] else "— (link rocket)")
        self.g_dv.set("Burnout TWR", f"{m['twr_burnout']:.2f}" if m["twr_burnout"] else "—")

        # warnings
        w = d.warnings
        if not w:
            self.lbl_warn.setText("✓ No issues flagged — inputs within typical ranges.")
            self.lbl_warn.setStyleSheet(f"color:{GOOD};")
            self.warn_box.toggle.setText("Engineering validation  ✓")
        else:
            self.lbl_warn.setText("⚠ " + "\n\n⚠ ".join(w))
            self.lbl_warn.setStyleSheet(f"color:{WARN};")
            self.warn_box.toggle.setText(f"Engineering validation  ⚠ {len(w)}")

    # ── drawing ─────────────────────────────────────────────────────────────
    def _draw_all(self):
        self._draw_thrust()
        self._draw_schematic()
        self._draw_flow()
        self._draw_pie()
        self._draw_sensitivity()

    def _draw_thrust(self):
        if not self._result:
            return
        c = self.canvas_thrust
        c.ax.clear()
        t = self._result["time"]
        if self.combo_graph.currentIndex() == 1:
            y = [p / 1e5 for p in self._result["pressure"]]
            c.style_ax("Chamber Pressure vs Time", "Time (s)", "Pc (bar)")
            col = "#ff7b72"
        else:
            y = self._result["thrust"]
            c.style_ax("Thrust vs Time", "Time (s)", "Thrust (N)")
            col = "#00BFFF"
        c.ax.plot(t, y, color=col, linewidth=1.6)
        c.ax.fill_between(t, y, alpha=0.12, color=col)
        self._hover_xy = (t, y)
        self._hover_annot = None
        c.figure.tight_layout()
        c.canvas.draw()

    def _on_hover(self, event):
        if not getattr(self, "_hover_xy", None) or event.inaxes != self.canvas_thrust.ax:
            return
        t, y = self._hover_xy
        if not t:
            return
        # nearest sample
        i = min(range(len(t)), key=lambda k: abs(t[k] - event.xdata))
        ax = self.canvas_thrust.ax
        if self._hover_annot:
            self._hover_annot.remove()
        self._hover_annot = ax.annotate(
            f"t={t[i]:.2f}s\n{y[i]:.0f}", xy=(t[i], y[i]),
            xytext=(10, 10), textcoords="offset points",
            color="#e6edf3", fontsize=9,
            bbox=dict(boxstyle="round", fc="#21262d", ec="#30363d"))
        self.canvas_thrust.canvas.draw_idle()

    def _draw_schematic(self):
        d = self._design
        if not d:
            return
        c = self.canvas_schem
        c.clear()
        ax = c.ax
        c.style_ax("Chamber & Nozzle Profile (axisymmetric, mm)", "Axial (mm)", "Radius (mm)")
        s = 1000.0
        rc = d.chamber_diameter / 2 * s
        rt = d.throat_diameter / 2 * s
        re = d.exit_diameter / 2 * s
        lc = d.chamber_length * s
        # convergent length ~ (rc-rt)/tan(30°); nozzle from solver
        lconv = (rc - rt) / 0.577
        ln = d.nozzle_length * s
        x_inj, x_cyl, x_thr, x_exit = 0.0, lc, lc + lconv, lc + lconv + ln
        xs = [x_inj, x_cyl, x_thr, x_exit]
        ys = [rc, rc, rt, re]
        ax.plot(xs, ys, color=ACCENT, lw=2)
        ax.plot(xs, [-v for v in ys], color=ACCENT, lw=2)
        ax.fill_between(xs, ys, [-v for v in ys], color=ACCENT, alpha=0.10)
        ax.plot([x_inj, x_exit], [0, 0], color=MUTED, ls="--", lw=0.8)
        ax.plot([x_inj, x_inj], [-rc, rc], color="#ff7b72", lw=3)  # injector face
        ax.annotate("injector", (x_inj, rc), color=MUTED, fontsize=8, ha="left", va="bottom")
        ax.annotate("throat", (x_thr, rt), color=MUTED, fontsize=8, ha="center", va="bottom")
        ax.annotate("exit", (x_exit, re), color=MUTED, fontsize=8, ha="right", va="bottom")
        ax.set_aspect("equal", adjustable="datalim")
        c.figure.tight_layout()
        c.canvas.draw()

    def _draw_flow(self):
        d = self._design
        if not d:
            return
        c = self.canvas_flow
        c.clear()
        ax = c.ax
        ax.set_facecolor(PANEL)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6)
        ax.axis("off")
        ax.set_title("Engine Flow Schematic", color=ACCENT, fontsize=12, fontweight="bold")

        def box(x, y, w, h, text, color):
            ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                         fc=color, ec="#30363d", alpha=0.85))
            ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                    color="#0d1117", fontsize=8, fontweight="bold")

        def arrow(x0, y0, x1, y1):
            ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                         arrowstyle="-|>", mutation_scale=12, color="#8b949e", lw=1.4))

        mono = d.combo.monoprop
        # Ox row
        if not mono:
            box(0.2, 4.0, 1.6, 1.0, f"LOX tank\n{d.ox_mass:.1f} kg", "#58a6ff")
            box(2.4, 4.0, 1.6, 1.0, f"Feed\n{d.mdot_ox:.2f} kg/s", "#79c0ff")
            arrow(1.8, 4.5, 2.4, 4.5)
            arrow(4.0, 4.5, 4.8, 3.7)
        # Fuel row
        box(0.2, 1.0, 1.6, 1.0, f"Fuel tank\n{d.fuel_mass:.1f} kg", "#f0883e")
        box(2.4, 1.0, 1.6, 1.0, f"Feed\n{d.mdot_fuel:.2f} kg/s", "#ffa657")
        arrow(1.8, 1.5, 2.4, 1.5)
        arrow(4.0, 1.5, 4.8, 2.3)
        # injector / chamber / nozzle
        box(4.8, 2.3, 1.6, 1.4, f"Injector\n{d.inj_n_holes} holes\nΔP {d.inj_dp/1e5:.1f} bar", "#d2a8ff")
        box(6.8, 2.3, 1.4, 1.4, f"Chamber\nPc {d.chamber_pressure/1e5:.0f} bar\nØ{d.chamber_diameter*1000:.0f}mm", "#ff7b72")
        box(8.4, 2.5, 1.4, 1.0, f"Nozzle\nε {d.expansion_ratio:.0f}\nIsp {d.isp:.0f}s", "#3fb950")
        arrow(6.4, 3.0, 6.8, 3.0)
        arrow(8.2, 3.0, 8.4, 3.0)
        c.figure.tight_layout()
        c.canvas.draw()

    def _draw_pie(self):
        d = self._design
        if not d:
            return
        c = self.canvas_pie
        c.clear()
        ax = c.ax
        ax.set_facecolor(BG)
        ax.set_title("Mass Breakdown", color=ACCENT, fontsize=12, fontweight="bold")
        dry = max(d.dry_mass, 1e-6)
        eng = 0.30 * dry
        tank = 0.50 * dry
        struct = 0.20 * dry
        vals = [d.ox_mass, d.fuel_mass, eng, tank, struct]
        labels = ["Oxidizer", "Fuel", "Engine dry", "Tank", "Structural"]
        colors = ["#58a6ff", "#f0883e", "#d2a8ff", "#ff7b72", "#8b949e"]
        vals, labels, colors = zip(*[(v, l, col) for v, l, col in
                                     zip(vals, labels, colors) if v > 1e-6])
        ax.pie(vals, labels=labels, colors=colors, autopct="%1.1f%%",
               textprops={"color": "#e6edf3", "fontsize": 9})
        ax.text(0, -1.35, f"Wet mass {d.wet_mass:.1f} kg", ha="center",
                color=MUTED, fontsize=9)
        c.figure.tight_layout()
        c.canvas.draw()

    def _draw_sensitivity(self):
        d = self._design
        if not d:
            return
        c = self.canvas_sens
        c.clear()
        ax = c.ax
        c.style_ax("Sensitivity — Isp response to ±20%", "Isp change (%)", "")
        try:
            s = d.sensitivity(span=0.20, n=9)
        except Exception:
            return
        names = {"chamber_pressure": "Chamber Pc", "of_ratio": "Mixture O/F",
                 "expansion_ratio": "Expansion ε"}
        base = d.isp
        rows, lows, highs = [], [], []
        for k, label in names.items():
            isp = s[k]["isp"]
            rows.append(label)
            lows.append((min(isp) - base) / base * 100.0)
            highs.append((max(isp) - base) / base * 100.0)
        y = range(len(rows))
        ax.barh(list(y), [h - l for h, l in zip(highs, lows)],
                left=lows, color=ACCENT, alpha=0.7, height=0.5)
        ax.axvline(0, color="#ff7b72", lw=1, ls="--")
        ax.set_yticks(list(y))
        ax.set_yticklabels(rows, color="#c9d1d9")
        for i, (l, h) in enumerate(zip(lows, highs)):
            ax.text(h, i, f" {h:+.1f}%", va="center", color=MUTED, fontsize=8)
            ax.text(l, i, f"{l:+.1f}% ", va="center", ha="right", color=MUTED, fontsize=8)
        c.figure.tight_layout()
        c.canvas.draw()

    # ── CSV thrust profile load ─────────────────────────────────────────────
    def _load_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load thrust curve CSV", "",
                                              "CSV Files (*.csv)")
        if not path:
            return
        pts = []
        try:
            with open(path, newline="") as f:
                for row in csv.reader(f):
                    if len(row) < 2:
                        continue
                    try:
                        pts.append((float(row[0]), float(row[1])))
                    except ValueError:
                        continue
        except Exception as exc:
            QMessageBox.critical(self, "CSV Error", str(exc))
            return
        if not pts:
            QMessageBox.warning(self, "CSV", "No valid (time,thrust) rows found.")
            return
        self._csv_curve = pts
        self._mark_dirty()

    # ── export / apply ───────────────────────────────────────────────────────
    def _export_csv(self):
        if not self._design:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export all parameters", "liquid_engine.csv", "CSV Files (*.csv)")
        if not path:
            return
        m = self._result["metrics"]
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Parameter", "Value", "Unit"])
            rows = [
                ("Propellant", self.combo_prop.currentText(), ""),
                ("Engine cycle", self._design.cycle, ""),
                ("Cooling", self._design.cooling, ""),
                ("Chamber pressure", m["chamber_pressure"] / 1e5, "bar"),
                ("Expansion ratio", m["expansion_ratio"], ""),
                ("Ideal Isp", m["isp_ideal"], "s"),
                ("Delivered Isp", m["isp_sl"], "s"),
                ("Vacuum Isp", m["isp_vac"], "s"),
                ("Overall efficiency", m["efficiency"], ""),
                ("Cf delivered", m["cf"], ""),
                ("Throat diameter", m["throat_diameter"] * 1000, "mm"),
                ("Exit diameter", m["exit_diameter"] * 1000, "mm"),
                ("Chamber diameter", m["chamber_diameter"] * 1000, "mm"),
                ("Chamber length", m["chamber_length"] * 1000, "mm"),
                ("Nozzle length", m["nozzle_length"] * 1000, "mm"),
                ("L*", m["l_star"], "m"),
                ("Contraction ratio", m["contraction_ratio"], ""),
                ("Residence time", m["residence_time"] * 1000, "ms"),
                ("Mass flow", m["mdot"], "kg/s"),
                ("Ox flow", m["mdot_ox"], "kg/s"),
                ("Fuel flow", m["mdot_fuel"], "kg/s"),
                ("Ox mass", m["ox_mass"], "kg"),
                ("Fuel mass", m["fuel_mass"], "kg"),
                ("Ox volume", m["ox_volume"] * 1000, "L"),
                ("Fuel volume", m["fuel_volume"] * 1000, "L"),
                ("Ullage volume", m["ullage_volume"] * 1000, "L"),
                ("Tank diameter", m["tank_dia"] * 1000, "mm"),
                ("Ox tank length", m["ox_tank_len"], "m"),
                ("Fuel tank length", m["fuel_tank_len"], "m"),
                ("Tank mass", m["tank_mass"], "kg"),
                ("Engine mass", m["engine_mass"], "kg"),
                ("Wet mass", m["wet_mass"], "kg"),
                ("Dry mass", m["dry_mass"], "kg"),
                ("Injector dP", m["inj_dp"] / 1e5, "bar"),
                ("Injector area", m["inj_area"] * 1e6, "mm^2"),
                ("Injector velocity", m["inj_velocity"], "m/s"),
                ("Injector holes", m["inj_n_holes"], ""),
                ("Injector pattern", m["inj_pattern"], ""),
                ("Wall heat flux", m["heat_flux"] / 1e6, "MW/m^2"),
                ("Coolant flow", m["coolant_flow"], "kg/s"),
                ("Cooling dP", m["cooling_dp"] / 1e5, "bar"),
                ("Cooling effectiveness", m["cooling_eff"], ""),
                ("Total impulse", m["total_impulse"], "N*s"),
                ("Propellant mass", self._result["prop_mass"], "kg"),
                ("Mass ratio", m["mass_ratio"], ""),
                ("Delta-V", m["delta_v"], "m/s"),
                ("Engine TWR", m["twr_engine"], ""),
                ("Liftoff TWR", m["twr_liftoff"], ""),
                ("Burnout TWR", m["twr_burnout"], ""),
            ]
            for r in rows:
                w.writerow(r)
            w.writerow([])
            w.writerow(["Time (s)", "Thrust (N)", "Chamber Pressure (Pa)"])
            res = self._result
            for i in range(len(res["time"])):
                w.writerow([res["time"][i], res["thrust"][i], res["pressure"][i]])
        QMessageBox.information(self, "Export", "All parameters exported.")

    def _apply(self):
        if self._result:
            self.motor_created.emit(self._result)
            self.accept()
