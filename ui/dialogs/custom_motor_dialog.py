import math
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, 
                             QLineEdit, QPushButton, QLabel, QGroupBox, QSpinBox, 
                             QDoubleSpinBox, QWidget, QComboBox, QFileDialog, QMessageBox, QGridLayout)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush

from ui.widgets.plot_widget import PlotWidget
from physics.internal_ballistics import Propellant, BatesGrain, TubularGrain, EndBurnerGrain, MotorSimulator

class GrainRenderer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(200, 200)
        self.outer_d = 38.0
        self.inner_d = 15.0
        self.regression = 0.0
        self.grain_type = 0 # 0: BATES, 1: Tubular, 2: End-Burner, 3: Star
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
        
        w = self.width()
        h = self.height()
        
        painter.fillRect(0, 0, w, h, QColor("#1e1e1e"))
        
        # Scale drawing
        cx = w / 2
        cy = h / 2
        max_d = max(self.outer_d, 1.0)
        scale = min(w, h) * 0.8 / max_d
        
        r_out = (self.outer_d * scale) / 2
        
        # Draw casing (outer)
        painter.setPen(QPen(QColor("#555555"), 2))
        painter.setBrush(QBrush(QColor("#808080")))
        painter.drawEllipse(int(cx - r_out), int(cy - r_out), int(r_out * 2), int(r_out * 2))
        
        if self.grain_type in [0, 1]: # BATES or Tubular (both core burners)
            # Effective inner diameter
            current_inner_d = min(self.inner_d + 2 * self.regression, self.outer_d)
            r_in = (current_inner_d * scale) / 2
            # Draw core (inner void)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor("#1e1e1e")))
            painter.drawEllipse(int(cx - r_in), int(cy - r_in), int(r_in * 2), int(r_in * 2))
            
        elif self.grain_type == 2: # End-Burner (solid cylinder)
            # Draw solid core (no void, unless it burned completely)
            if self.regression >= self.outer_d * 5: # arbitrary hack for end burner view
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(QColor("#1e1e1e")))
                painter.drawEllipse(int(cx - r_out), int(cy - r_out), int(r_out * 2), int(r_out * 2))
                
        elif self.grain_type == 3: # Star
            import math
            from PyQt6.QtGui import QPolygonF
            from PyQt6.QtCore import QPointF
            
            # Draw star void
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor("#1e1e1e")))
            
            poly = QPolygonF()
            N = self.star_points
            
            # Calculate dynamic star points based on regression approximation
            # For visualization, we just expand the inner radius and outer points.
            r_inner = self.inner_d / 2.0
            r_outer = r_inner + self.star_depth
            if r_outer > self.outer_d / 2.0:
                r_outer = self.outer_d / 2.0
                
            r_in_draw = min((r_inner + self.regression) * scale, r_out)
            r_out_draw = min((r_outer + self.regression) * scale, r_out)
            
            for i in range(N * 2):
                angle = i * math.pi / N
                r = r_out_draw if i % 2 == 0 else r_in_draw
                px = cx + r * math.sin(angle)
                py = cy - r * math.cos(angle)
                poly.append(QPointF(px, py))
                
            painter.drawPolygon(poly)
        
        painter.end()


class CustomMotorDialog(QDialog):
    
    motor_created = pyqtSignal(dict)  # emits the simulation result dict
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Custom Motor Builder")
        self.setMinimumSize(1000, 700)
        self.setStyleSheet("background-color: #1a1a1a; color: #ffffff;")
        
        layout = QHBoxLayout(self)
        
        # Left Panel (Inputs)
        left_layout = QVBoxLayout()
        
        # -- Propellant Group
        prop_group = QGroupBox("Propellant Parameters")
        prop_layout = QFormLayout()
        
        self.spin_a = QDoubleSpinBox()
        self.spin_a.setDecimals(6)
        self.spin_a.setRange(0, 1.0)
        self.spin_a.setValue(0.000015)
        self.spin_a.setSingleStep(0.000001)
        
        self.spin_n = QDoubleSpinBox()
        self.spin_n.setDecimals(3)
        self.spin_n.setRange(0, 0.999)
        self.spin_n.setValue(0.350)
        
        self.spin_rho = QDoubleSpinBox()
        self.spin_rho.setRange(500, 3000)
        self.spin_rho.setValue(1600.0)
        
        self.spin_cstar = QDoubleSpinBox()
        self.spin_cstar.setRange(500, 3000)
        self.spin_cstar.setValue(1400.0)
        
        prop_layout.addRow("Burn Rate Coeff 'a':", self.spin_a)
        prop_layout.addRow("Pressure Exp 'n':", self.spin_n)
        prop_layout.addRow("Density (kg/m³):", self.spin_rho)
        prop_layout.addRow("C* (m/s):", self.spin_cstar)
        prop_group.setLayout(prop_layout)
        left_layout.addWidget(prop_group)
        
        # -- Grain Group
        grain_group = QGroupBox("Grain Configuration")
        grain_layout = QFormLayout()
        
        self.combo_grain_type = QComboBox()
        self.combo_grain_type.addItems(["BATES (Standard)", "Tubular", "End-Burner", "Star"])
        self.combo_grain_type.currentIndexChanged.connect(self._on_grain_type_changed)
        grain_layout.addRow("Grain Type:", self.combo_grain_type)
        
        self.spin_gcount = QSpinBox()
        self.spin_gcount.setRange(1, 10)
        self.spin_gcount.setValue(3)
        
        self.spin_glen = QDoubleSpinBox()
        self.spin_glen.setRange(10, 1000)
        self.spin_glen.setValue(100.0)
        
        self.spin_god = QDoubleSpinBox()
        self.spin_god.setRange(10, 500)
        self.spin_god.setValue(38.0)
        
        self.spin_gid = QDoubleSpinBox()
        self.spin_gid.setRange(2, 400)
        self.spin_gid.setValue(15.0)
        
        # Star parameters
        self.spin_star_points = QSpinBox()
        self.spin_star_points.setRange(3, 12)
        self.spin_star_points.setValue(5)
        
        self.spin_star_depth = QDoubleSpinBox()
        self.spin_star_depth.setRange(1, 200)
        self.spin_star_depth.setValue(5.0)
        
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
        
        grain_group.setLayout(grain_layout)
        left_layout.addWidget(grain_group)
        
        # Hide star parameters initially
        grain_layout.itemAt(self.row_star_points, QFormLayout.ItemRole.LabelRole).widget().hide()
        self.spin_star_points.hide()
        grain_layout.itemAt(self.row_star_depth, QFormLayout.ItemRole.LabelRole).widget().hide()
        self.spin_star_depth.hide()
        
        # Connect visuals
        self.spin_god.valueChanged.connect(self._update_visual)
        self.spin_gid.valueChanged.connect(self._update_visual)
        self.spin_star_points.valueChanged.connect(self._update_visual)
        self.spin_star_depth.valueChanged.connect(self._update_visual)
        
        # -- Nozzle Group
        nozzle_group = QGroupBox("Nozzle Configuration")
        nozzle_layout = QFormLayout()
        self.spin_nthroat = QDoubleSpinBox()
        self.spin_nthroat.setRange(1, 100)
        self.spin_nthroat.setValue(9.0)
        self.spin_nexit = QDoubleSpinBox()
        self.spin_nexit.setRange(1, 200)
        self.spin_nexit.setValue(20.0)
        
        nozzle_layout.addRow("Throat Dia (mm):", self.spin_nthroat)
        nozzle_layout.addRow("Exit Dia (mm):", self.spin_nexit)
        nozzle_group.setLayout(nozzle_layout)
        left_layout.addWidget(nozzle_group)
        
        # -- Simulate Button
        btn_sim = QPushButton("SIMULATE BALLISTICS")
        btn_sim.setStyleSheet("background-color: #0078D7; color: white; padding: 10px; font-weight: bold;")
        btn_sim.clicked.connect(self._run_sim)
        left_layout.addWidget(btn_sim)
        
        # -- Apply & Export Buttons
        btn_h = QHBoxLayout()
        self.btn_apply = QPushButton("APPLY TO ROCKET")
        self.btn_apply.setStyleSheet("background-color: #28a745; color: white; padding: 8px; font-weight: bold;")
        self.btn_apply.setEnabled(False)
        self.btn_apply.clicked.connect(self._apply)
        btn_h.addWidget(self.btn_apply)
        
        self.btn_export = QPushButton("EXPORT CSV")
        self.btn_export.setStyleSheet("background-color: #555555; color: white; padding: 8px; font-weight: bold;")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self._export_csv)
        btn_h.addWidget(self.btn_export)
        
        left_layout.addLayout(btn_h)
        left_layout.addStretch()
        layout.addLayout(left_layout, 1)
        
        # Right Panel (Visuals)
        right_layout = QVBoxLayout()
        
        # Top right: Metrics and 2D Cross Section
        top_right_h = QHBoxLayout()
        
        # Metrics Grid
        metrics_group = QGroupBox("Simulation Results")
        self.metrics_grid = QGridLayout()
        self.lbl_init_kn = QLabel("Initial Kn: —"); self.metrics_grid.addWidget(self.lbl_init_kn, 0, 0)
        self.lbl_max_kn = QLabel("Max Kn: —"); self.metrics_grid.addWidget(self.lbl_max_kn, 1, 0)
        self.lbl_max_pc = QLabel("Max Pc: —"); self.metrics_grid.addWidget(self.lbl_max_pc, 2, 0)
        self.lbl_vol_load = QLabel("Vol Loading: —"); self.metrics_grid.addWidget(self.lbl_vol_load, 3, 0)
        self.lbl_pt_ratio = QLabel("Port/Throat: —"); self.metrics_grid.addWidget(self.lbl_pt_ratio, 4, 0)
        self.lbl_tp_ratio = QLabel("Throat/Port: —"); self.metrics_grid.addWidget(self.lbl_tp_ratio, 5, 0)
        self.lbl_core_ld = QLabel("Core L/D: —"); self.metrics_grid.addWidget(self.lbl_core_ld, 6, 0)
        
        self.lbl_web = QLabel("Web: —"); self.metrics_grid.addWidget(self.lbl_web, 0, 1)
        self.lbl_burn_t = QLabel("Burn Time: —"); self.metrics_grid.addWidget(self.lbl_burn_t, 1, 1)
        self.lbl_prop_len = QLabel("Prop Length: —"); self.metrics_grid.addWidget(self.lbl_prop_len, 2, 1)
        self.lbl_prop_mass = QLabel("Prop Mass: —"); self.metrics_grid.addWidget(self.lbl_prop_mass, 3, 1)
        self.lbl_tot_impulse = QLabel("Total Impulse: —"); self.metrics_grid.addWidget(self.lbl_tot_impulse, 4, 1)
        self.lbl_motor_class = QLabel("Motor Class: —"); self.metrics_grid.addWidget(self.lbl_motor_class, 5, 1)
        self.lbl_del_isp = QLabel("Delivered ISP: —"); self.metrics_grid.addWidget(self.lbl_del_isp, 6, 1)
        self.lbl_mass_flux = QLabel("Peak Mass Flux: —"); self.metrics_grid.addWidget(self.lbl_mass_flux, 7, 1)
        
        metrics_group.setLayout(self.metrics_grid)
        top_right_h.addWidget(metrics_group, 2)
        
        # 2D Cross Section
        self.renderer = GrainRenderer()
        top_right_h.addWidget(self.renderer, 1)
        right_layout.addLayout(top_right_h, 1)
        
        # Plot Controls
        plot_controls_h = QHBoxLayout()
        plot_controls_h.addWidget(QLabel("Select Graph:"))
        self.combo_graph = QComboBox()
        self.combo_graph.addItems(["Thrust vs Time", "Pressure vs Time", "Kn vs Time", "Mass Flux vs Time"])
        self.combo_graph.currentIndexChanged.connect(self._update_plot)
        plot_controls_h.addWidget(self.combo_graph)
        plot_controls_h.addStretch()
        right_layout.addLayout(plot_controls_h)
        
        # Plot
        self.plot_widget = PlotWidget(title="Thrust Curve", xlabel="Time (s)", ylabel="Thrust (N)")
        right_layout.addWidget(self.plot_widget, 3)
        
        layout.addLayout(right_layout, 2)
        
        self._last_result = None
        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(self._on_anim_tick)
        self._anim_idx = 0

    def _on_grain_type_changed(self, idx):
        # 0: BATES, 1: Tubular, 2: End-Burner, 3: Star
        layout = self.spin_gcount.parentWidget().layout()
        
        # Hide all conditional fields first
        layout.itemAt(self.row_gcount, QFormLayout.ItemRole.LabelRole).widget().hide()
        self.spin_gcount.hide()
        layout.itemAt(self.row_gid, QFormLayout.ItemRole.LabelRole).widget().hide()
        self.spin_gid.hide()
        layout.itemAt(self.row_star_points, QFormLayout.ItemRole.LabelRole).widget().hide()
        self.spin_star_points.hide()
        layout.itemAt(self.row_star_depth, QFormLayout.ItemRole.LabelRole).widget().hide()
        self.spin_star_depth.hide()
        
        if idx == 0: # BATES
            layout.itemAt(self.row_gcount, QFormLayout.ItemRole.LabelRole).widget().show()
            self.spin_gcount.show()
            layout.itemAt(self.row_gid, QFormLayout.ItemRole.LabelRole).widget().show()
            self.spin_gid.show()
        elif idx == 1: # Tubular
            layout.itemAt(self.row_gid, QFormLayout.ItemRole.LabelRole).widget().show()
            self.spin_gid.show()
        elif idx == 2: # End-Burner
            pass # Only needs Length and OD
        elif idx == 3: # Star
            layout.itemAt(self.row_gid, QFormLayout.ItemRole.LabelRole).widget().show()
            self.spin_gid.show()
            layout.itemAt(self.row_star_points, QFormLayout.ItemRole.LabelRole).widget().show()
            self.spin_star_points.show()
            layout.itemAt(self.row_star_depth, QFormLayout.ItemRole.LabelRole).widget().show()
            self.spin_star_depth.show()
            
        self._update_visual()

    def _update_visual(self):
        self.renderer.update_dimensions(
            self.spin_god.value(), 
            self.spin_gid.value(), 
            0.0,
            self.combo_grain_type.currentIndex(),
            self.spin_star_points.value(),
            self.spin_star_depth.value()
        )

    def _on_anim_tick(self):
        if not self._last_result or "regression" not in self._last_result:
            self._anim_timer.stop()
            return
            
        regs = self._last_result["regression"]
        if self._anim_idx >= len(regs):
            self._anim_timer.stop()
            return
            
        reg_mm = regs[self._anim_idx] * 1000.0
        self.renderer.update_dimensions(
            self.spin_god.value(), 
            self.spin_gid.value(), 
            reg_mm,
            self.combo_grain_type.currentIndex(),
            self.spin_star_points.value(),
            self.spin_star_depth.value()
        )
        
        # Advance faster if the array is huge
        step = max(1, len(regs) // 100)
        self._anim_idx += step
        
    def _run_sim(self):
        # Gather inputs
        prop = Propellant(
            a=self.spin_a.value(),
            n=self.spin_n.value(),
            density=self.spin_rho.value(),
            c_star=self.spin_cstar.value(),
            gamma=1.2
        )
        
        g_len = self.spin_glen.value() / 1000.0
        g_od = self.spin_god.value() / 1000.0
        g_id = self.spin_gid.value() / 1000.0
        count = self.spin_gcount.value()
        g_type = self.combo_grain_type.currentIndex()
        
        if g_type == 0: # BATES
            grain = BatesGrain(length=g_len, outer_diameter=g_od, core_diameter=g_id)
            grains = [grain] * count
        elif g_type == 1: # Tubular
            grain = TubularGrain(length=g_len, outer_diameter=g_od, core_diameter=g_id)
            grains = [grain]
        elif g_type == 2: # End-Burner
            grain = EndBurnerGrain(length=g_len, diameter=g_od)
            grains = [grain]
        elif g_type == 3: # Star
            pts = self.spin_star_points.value()
            dep = self.spin_star_depth.value() / 1000.0
            from physics.internal_ballistics import StarGrain
            grain = StarGrain(length=g_len, outer_diameter=g_od, core_diameter=g_id, 
                              points=pts, point_depth=dep)
            grains = [grain]
        else:
            grains = []
        
        t_dia = self.spin_nthroat.value() / 1000.0
        e_dia = self.spin_nexit.value() / 1000.0
        
        sim = MotorSimulator(propellant=prop, grains=grains, throat_diameter=t_dia, exit_diameter=e_dia)
        try:
            res = sim.simulate(dt=0.01)
            self._last_result = res
            
            # Update Metrics
            m = res["metrics"]
            self.lbl_init_kn.setText(f"Initial Kn: {m['initial_kn']:.1f}")
            self.lbl_max_kn.setText(f"Max Kn: {m['max_kn']:.1f}")
            self.lbl_max_pc.setText(f"Max Pc: {m['max_pc']/1e5:.1f} bar")
            self.lbl_vol_load.setText(f"Vol Loading: {m['vol_loading']:.1f} %")
            self.lbl_pt_ratio.setText(f"Port/Throat: {m['port_to_throat']:.2f}")
            self.lbl_tp_ratio.setText(f"Throat/Port: {m['throat_to_port']:.2f}")
            self.lbl_core_ld.setText(f"Core L/D: {m['core_l_d']:.2f}")
            
            self.lbl_web.setText(f"Web: {m['web']*1000:.1f} mm")
            self.lbl_burn_t.setText(f"Burn Time: {res['time'][-1]:.2f} s")
            self.lbl_prop_len.setText(f"Prop Length: {m['prop_len']*1000:.1f} mm")
            self.lbl_prop_mass.setText(f"Prop Mass: {res['prop_mass']*1000:.1f} g")
            
            impulse = m["total_impulse"]
            self.lbl_tot_impulse.setText(f"Total Impulse: {impulse:.1f} Ns")
            self.lbl_del_isp.setText(f"Delivered ISP: {m['delivered_isp']:.1f} s")
            self.lbl_mass_flux.setText(f"Peak Mass Flux: {m['peak_mass_flux']:.1f} kg/m²s")
            
            # Calculate Motor Class
            motor_class = "Micro"
            classes = [
                (0.625, '1/4A'), (1.25, '1/2A'), (2.50, 'A'), (5.0, 'B'), 
                (10.0, 'C'), (20.0, 'D'), (40.0, 'E'), (80.0, 'F'), 
                (160.0, 'G'), (320.0, 'H'), (640.0, 'I'), (1280.0, 'J'),
                (2560.0, 'K'), (5120.0, 'L'), (10240.0, 'M'), (20480.0, 'N'), 
                (40960.0, 'O')
            ]
            for limit, cls in classes:
                if impulse <= limit:
                    motor_class = cls
                    break
            if impulse > 40960.0:
                motor_class = "P+"
            self.lbl_motor_class.setText(f"Motor Class: {motor_class}")
            
            self._update_plot()
            
            # Start animation
            self._anim_idx = 0
            # Target ~2 seconds for animation (2000 ms). Determine interval based on max 100 frames.
            frames = min(len(res["regression"]), 100)
            interval = max(20, 2000 // frames if frames > 0 else 20)
            self._anim_timer.start(interval)
            
            # Enable apply and export
            self.btn_apply.setEnabled(True)
            self.btn_export.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Simulation Error", str(e))

    def _update_plot(self):
        if not self._last_result:
            return
            
        res = self._last_result
        times = res["time"]
        idx = self.combo_graph.currentIndex()
        
        if idx == 0:
            y = res["thrust"]
            self.plot_widget.update_plot(times, y, "Thrust vs Time", "Time (s)", "Thrust (N)", "#00BFFF")
        elif idx == 1:
            y = [p / 100000.0 for p in res["pressure"]]
            self.plot_widget.update_plot(times, y, "Pressure vs Time", "Time (s)", "Chamber Pressure (bar)", "#FF4500")
        elif idx == 2:
            y = res["kn"]
            self.plot_widget.update_plot(times, y, "Kn vs Time", "Time (s)", "Kneser Number (Kn)", "#32CD32")
        elif idx == 3:
            y = res["mass_flux"]
            self.plot_widget.update_plot(times, y, "Mass Flux vs Time", "Time (s)", "Mass Flux (kg/m²s)", "#FFD700")

    def _export_csv(self):
        if not self._last_result:
            return
            
        file_path, _ = QFileDialog.getSaveFileName(self, "Export Simulation to CSV", "custom_motor.csv", "CSV Files (*.csv)")
        if file_path:
            try:
                import csv
                res = self._last_result
                with open(file_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Time (s)", "Thrust (N)", "Pressure (Pa)", "Kn", "Mass Flux (kg/m2s)"])
                    for i in range(len(res["time"])):
                        writer.writerow([
                            res["time"][i],
                            res["thrust"][i],
                            res["pressure"][i],
                            res["kn"][i],
                            res["mass_flux"][i]
                        ])
                QMessageBox.information(self, "Export Successful", "Data exported successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export: {str(e)}")

    def _apply(self):
        if self._last_result:
            self.motor_created.emit(self._last_result)
            self.accept()
