"""
K2 Aerospace — Propulsion Workspace
Motor analysis, thrust curves, Isp, mass flow, chamber pressure.
"""
import json, logging
from pathlib import Path
from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QGroupBox,
    QFormLayout, QLabel, QComboBox, QSplitter, QFrame, QScrollArea, QCheckBox,
    QPushButton)
from PyQt6.QtCore import Qt
from ui.widgets.plot_widget import PlotWidget
from physics.propulsion import compute_isp, compute_mass_flow_rate, estimate_chamber_pressure, generate_thrust_curve

logger = logging.getLogger("K2.PropulsionWS")


class ValueLabel(QLabel):
    def __init__(self, text="—", parent=None):
        super().__init__(text, parent)
        self.setStyleSheet(
            "color: #e6edf3; font-family: 'Cascadia Code', monospace; font-size: 13px; "
            "font-weight: 600; padding: 2px 4px; background-color: #161b22; border-radius: 4px;")


class PropulsionWorkspace(QWidget):
    def __init__(self, engine, parent=None):
        super().__init__(parent)
        self.engine = engine
        self._motors = self._load_motors()
        self._setup_ui()
        self.engine.state_changed.connect(self._on_state_changed)
        self._update_display()

    def _load_motors(self):
        p = Path(__file__).parent.parent.parent / "data" / "motors.json"
        try:
            with open(p, encoding="utf-8") as f: return json.load(f)
        except: return []

    def _unique_diameters_mm(self):
        ds = {round(m.get("diameter", 0) * 1000) for m in self._motors if m.get("diameter")}
        return sorted(ds)

    def _current_body_diameter(self):
        try:
            return float(self.engine.state.diameter)
        except Exception:
            return 0.0

    def _passes_filters(self, m):
        if self.class_combo.currentIndex() > 0 and m.get("class") != self.class_combo.currentText():
            return False
        d_mm = self.diam_combo.currentData()
        if d_mm is not None and round(m.get("diameter", 0) * 1000) != d_mm:
            return False
        if self.fit_body_chk.isChecked():
            body = self._current_body_diameter()
            if body > 0 and m.get("diameter", 0) > body + 1e-9:
                return False
        if self.hide_oop_chk.isChecked() and m.get("availability") == "OOP":
            return False
        return True

    def _rebuild_motor_combo(self):
        # remember current selection so we can restore it after refiltering
        cur = self.engine.state.motor_designation
        self._filtered = [m for m in self._motors if self._passes_filters(m)]

        self.motor_combo.blockSignals(True)
        self.motor_combo.clear()
        self.motor_combo.addItem("None")
        restore_idx = 0
        for i, m in enumerate(self._filtered):
            d_mm = round(m.get("diameter", 0) * 1000)
            self.motor_combo.addItem(
                f"{m['designation']} — {m['manufacturer']} ({d_mm}mm)")
            if m["designation"] == cur:
                restore_idx = i + 1
        self.motor_combo.setCurrentIndex(restore_idx)
        self.motor_combo.blockSignals(False)

        body = self._current_body_diameter()
        note = f"  •  body Ø {body*1000:.0f} mm" if (self.fit_body_chk.isChecked() and body > 0) else ""
        self.lbl_count.setText(f"{len(self._filtered)} motors{note}")

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel
        left = QScrollArea()
        left.setWidgetResizable(True)
        left.setMaximumWidth(350)
        left.setFrameShape(QFrame.Shape.NoFrame)
        lw = QWidget()
        ll = QVBoxLayout(lw)
        ll.setContentsMargins(12, 12, 12, 12)
        ll.setSpacing(12)

        g = QGroupBox("Motor Selection")
        fl = QFormLayout()

        # --- Filters ---
        self.class_combo = QComboBox()
        self.class_combo.addItem("All classes")
        for c in sorted({m.get("class", "") for m in self._motors if m.get("class")}):
            self.class_combo.addItem(c)
        self.class_combo.currentIndexChanged.connect(self._rebuild_motor_combo)
        fl.addRow("Class:", self.class_combo)

        self.diam_combo = QComboBox()
        self.diam_combo.addItem("All diameters")
        for d_mm in self._unique_diameters_mm():
            self.diam_combo.addItem(f"{d_mm} mm", d_mm)
        self.diam_combo.currentIndexChanged.connect(self._rebuild_motor_combo)
        fl.addRow("Diameter:", self.diam_combo)

        self.fit_body_chk = QCheckBox("Fit current body")
        self.fit_body_chk.setToolTip("Only show motors whose diameter fits the airframe diameter")
        self.fit_body_chk.stateChanged.connect(self._rebuild_motor_combo)
        fl.addRow("", self.fit_body_chk)

        self.hide_oop_chk = QCheckBox("Hide out-of-production")
        self.hide_oop_chk.setChecked(True)
        self.hide_oop_chk.stateChanged.connect(self._rebuild_motor_combo)
        fl.addRow("", self.hide_oop_chk)

        # --- Motor list ---
        self.motor_combo = QComboBox()
        self.motor_combo.currentIndexChanged.connect(self._on_motor_selected)
        fl.addRow("Motor:", self.motor_combo)

        self.lbl_count = QLabel("")
        self.lbl_count.setStyleSheet("color: #8b949e; font-size: 11px;")
        fl.addRow("", self.lbl_count)

        self.btn_custom_motor = QPushButton("Create Custom Motor")
        self.btn_custom_motor.setStyleSheet("background-color: #0078D7; color: white; padding: 5px; margin-top: 5px;")
        self.btn_custom_motor.clicked.connect(self._open_custom_motor_dialog)
        fl.addRow("", self.btn_custom_motor)

        g.setLayout(fl)
        ll.addWidget(g)

        self._filtered = []
        self._rebuild_motor_combo()

        g2 = QGroupBox("Motor Properties")
        fl2 = QFormLayout(); fl2.setSpacing(6)
        self.lbl_impulse = ValueLabel(); fl2.addRow("Total Impulse:", self.lbl_impulse)
        self.lbl_avg = ValueLabel(); fl2.addRow("Avg Thrust:", self.lbl_avg)
        self.lbl_max = ValueLabel(); fl2.addRow("Max Thrust:", self.lbl_max)
        self.lbl_burn = ValueLabel(); fl2.addRow("Burn Time:", self.lbl_burn)
        self.lbl_prop = ValueLabel(); fl2.addRow("Prop Mass:", self.lbl_prop)
        g2.setLayout(fl2)
        ll.addWidget(g2)

        g3 = QGroupBox("Computed Performance")
        fl3 = QFormLayout(); fl3.setSpacing(6)
        self.lbl_isp = ValueLabel(); fl3.addRow("Isp:", self.lbl_isp)
        self.lbl_mdot = ValueLabel(); fl3.addRow("Mass Flow:", self.lbl_mdot)
        self.lbl_pc = ValueLabel(); fl3.addRow("Chamber P:", self.lbl_pc)
        g3.setLayout(fl3)
        ll.addWidget(g3)
        ll.addStretch()

        left.setWidget(lw)
        splitter.addWidget(left)

        # Right: thrust curve plot
        self.thrust_plot = PlotWidget(title="Thrust Curve", xlabel="Time (s)", ylabel="Thrust (N)")
        splitter.addWidget(self.thrust_plot)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)

    def _open_custom_motor_dialog(self):
        from ui.dialogs.custom_motor_dialog import CustomMotorDialog
        dlg = CustomMotorDialog(self)
        dlg.motor_created.connect(self._on_custom_motor_created)
        dlg.exec()

    def _on_custom_motor_created(self, sim_data):
        # Calculate derived values from the simulation
        import numpy as np
        t = np.array(sim_data["time"])
        f = np.array(sim_data["thrust"])

        burn_time = float(t[-1])
        max_thrust = float(np.max(f))

        # Integrate for total impulse; average = impulse/burn (exact, not
        # sample-mean which depends on time spacing)
        total_impulse = float(np.trapz(f, t))
        avg_thrust = total_impulse / burn_time if burn_time > 0 else 0.0
        prop_mass = float(sim_data.get("prop_mass", 0.5))

        # MotorSimulator doesn't model the hardware, so estimate it:
        # motor length ≈ grain stack + nozzle/closures; casing mass ≈ half the
        # propellant mass (typical HPR reload hardware ratio). Zero here would
        # corrupt the CG/stability calc (motor dry mass matters post-burnout).
        metrics = sim_data.get("metrics", {})
        prop_len = float(metrics.get("prop_len", 0.0))
        motor_length = float(sim_data.get("length", 0.0)) or prop_len * 1.2
        case_mass = float(sim_data.get("case_mass", 0.0)) or prop_mass * 0.5

        # Reset combo box to "None" so it doesn't show a pre-selected motor
        self.motor_combo.blockSignals(True)
        self.motor_combo.setCurrentIndex(0)
        self.motor_combo.blockSignals(False)

        # Update engine
        self.engine.update(
            motor_designation="Custom BATES",
            motor_avg_thrust=avg_thrust,
            motor_max_thrust=max_thrust,
            motor_total_impulse=total_impulse,
            motor_burn_time=burn_time,
            propellant_mass=prop_mass,
            propellant_mass_initial=prop_mass,
            motor_dry_mass=case_mass,
            motor_length=motor_length,
            custom_thrust_curve=list(zip(t.tolist(), f.tolist()))
        )
        self._update_display()

    def _on_motor_selected(self, idx):
        # Always clear any custom thrust curve — the sim engine and the plot
        # both prefer it over the trapezoid, so a stale one would silently fly
        # the OLD custom motor under the newly selected motor's name.
        if idx == 0:
            self.engine.update(motor_designation="None", motor_avg_thrust=0, motor_max_thrust=0,
                motor_total_impulse=0, motor_burn_time=0, propellant_mass=0, propellant_mass_initial=0,
                motor_dry_mass=0, motor_length=0, custom_thrust_curve=[])
        else:
            m = self._filtered[idx - 1]
            prop, dry = self._sanitized_masses(m)
            self.engine.update(
                motor_designation=m["designation"], motor_avg_thrust=m["avg_thrust"],
                motor_max_thrust=m.get("max_thrust", m["avg_thrust"] * 1.4),
                motor_total_impulse=m["total_impulse"], motor_burn_time=m["burn_time"],
                propellant_mass=prop, propellant_mass_initial=prop,
                motor_dry_mass=dry,
                motor_length=m.get("length", 0.0),
                custom_thrust_curve=[])
        self._update_display()

    @staticmethod
    def _sanitized_masses(m) -> tuple:
        """(propellant_mass, dry_mass) with catalog-data repair.

        ThrustCurve hybrids (Contrail, SkyRipper, RATT…) list only the fuel
        grain as propellant_mass — the oxidizer is missing, so the implied
        Isp is absurd (500–16000 s) and the sim would deplete almost no mass
        while delivering the full impulse. Reconstruct an effective expended
        mass from the impulse at a typical delivered Isp instead. Also guards
        corrupt entries where propellant_mass exceeds total_mass.
        """
        imp = m.get("total_impulse", 0.0)
        prop = m.get("propellant_mass", 0.0)
        total = m.get("total_mass", 0.0)
        isp = imp / (prop * 9.81) if prop > 0 else 0.0
        if prop <= 0 or isp > 350.0 or (total > 0 and prop >= total):
            prop_fixed = imp / (9.81 * 200.0)   # typical delivered Isp ≈ 200 s
            if total > 0:
                prop_fixed = min(prop_fixed, 0.9 * total)
            logger.warning(
                f"Motor {m.get('designation')}: implausible catalog masses "
                f"(prop={prop*1000:.1f} g, Isp={isp:.0f} s) — using effective "
                f"expended mass {prop_fixed*1000:.1f} g (hybrid oxidizer not in catalog)")
            prop = prop_fixed
        dry = max(0.0, total - prop)
        return prop, dry

    def _on_state_changed(self, state):
        # Body diameter may have changed -> refresh fit-body filter + selection sync
        idx = 0
        for i, m in enumerate(self._filtered):
            if m["designation"] == state.motor_designation:
                idx = i + 1; break
        self.motor_combo.blockSignals(True)
        self.motor_combo.setCurrentIndex(idx)
        self.motor_combo.blockSignals(False)
        if self.fit_body_chk.isChecked():
            self._rebuild_motor_combo()
        self._update_display()

    def _update_display(self):
        s = self.engine.state
        if s.motor_designation == "None":
            for l in [self.lbl_impulse, self.lbl_avg, self.lbl_max, self.lbl_burn, self.lbl_prop,
                      self.lbl_isp, self.lbl_mdot, self.lbl_pc]:
                l.setText("—")
            self.thrust_plot.ax.clear()
            self.thrust_plot._style_axis("Thrust Curve", "Time (s)", "Thrust (N)")
            self.thrust_plot.canvas.draw()
            return

        self.lbl_impulse.setText(f"{s.motor_total_impulse:.1f} N·s")
        self.lbl_avg.setText(f"{s.motor_avg_thrust:.1f} N")
        self.lbl_max.setText(f"{s.motor_max_thrust:.1f} N")
        self.lbl_burn.setText(f"{s.motor_burn_time:.2f} s")
        pm = s.propellant_mass_initial
        self.lbl_prop.setText(f"{pm*1000:.1f} g")

        isp = compute_isp(s.motor_total_impulse, pm)
        mdot = compute_mass_flow_rate(pm, s.motor_burn_time)
        pc = estimate_chamber_pressure(s.motor_avg_thrust)
        self.lbl_isp.setText(f"{isp:.1f} s")
        self.lbl_mdot.setText(f"{mdot*1000:.2f} g/s")
        self.lbl_pc.setText(f"{pc/1e6:.2f} MPa")

        self.engine.update(motor_isp=isp, motor_mass_flow=mdot, motor_chamber_pressure=pc, emit=False)
        
        if hasattr(s, "custom_thrust_curve") and s.custom_thrust_curve:
            t = [pt[0] for pt in s.custom_thrust_curve]
            f = [pt[1] for pt in s.custom_thrust_curve]
        else:
            t, f = generate_thrust_curve(s.motor_avg_thrust, s.motor_max_thrust, s.motor_burn_time)
            
        self.thrust_plot.update_plot(t, f, "Thrust Curve", "Time (s)", "Thrust (N)", "#f0883e")
