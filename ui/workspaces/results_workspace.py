"""
K2 Aerospace — Results Workspace
Professional post-flight data visualization with synchronized cursor.
"""
import csv, logging
from pathlib import Path
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QTabWidget, QLabel, QFileDialog, QSlider, QFrame, QGridLayout)
from PyQt6.QtCore import Qt
from ui.icons import icon
from ui.widgets.plot_widget import PlotWidget

logger = logging.getLogger("K2.ResultsWS")


class ReadoutValue(QLabel):
    def __init__(self, text="—", parent=None):
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(
            "color: #e6edf3; font-family: 'Cascadia Code', monospace; "
            "font-size: 13px; font-weight: 600; padding: 4px 8px; "
            "background-color: #161b22; border: 1px solid #21262d; border-radius: 6px;"
        )


class ResultsWorkspace(QWidget):
    def __init__(self, engine, sim_engine=None, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.sim_engine = sim_engine
        self._history = None
        self._setup_ui()
        self.engine.state_changed.connect(self._check_data)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Header
        top = QHBoxLayout()
        title = QLabel("FLIGHT RESULTS")
        title.setStyleSheet("color: #58a6ff; font-size: 16px; font-weight: 700; letter-spacing: 2px;")
        top.addWidget(title)
        top.addStretch()

        self.btn_refresh = QPushButton(icon("refresh"), "Refresh Plots")
        self.btn_refresh.clicked.connect(self.refresh_plots)
        top.addWidget(self.btn_refresh)

        self.btn_export = QPushButton(icon("export"), "Export CSV")
        self.btn_export.clicked.connect(self._export_csv)
        top.addWidget(self.btn_export)
        layout.addLayout(top)

        # Summary
        self.summary_label = QLabel("No simulation data. Run a simulation first.")
        self.summary_label.setStyleSheet("color: #8b949e; font-size: 13px; padding: 8px; "
            "background-color: #161b22; border-radius: 6px;")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        # Timeline scrubber
        scrub_frame = QFrame()
        scrub_frame.setStyleSheet("background-color: #161b22; border: 1px solid #21262d; border-radius: 6px; padding: 4px;")
        sl = QHBoxLayout(scrub_frame)
        sl.setContentsMargins(8, 4, 8, 4)
        sl.addWidget(QLabel("T+"))
        self.scrub_slider = QSlider(Qt.Orientation.Horizontal)
        self.scrub_slider.setRange(0, 1000)
        self.scrub_slider.setValue(0)
        self.scrub_slider.valueChanged.connect(self._on_scrub)
        sl.addWidget(self.scrub_slider, 1)
        self.scrub_time = QLabel("0.00 s")
        self.scrub_time.setStyleSheet("color: #58a6ff; font-family: 'Cascadia Code', monospace; font-weight: 600;")
        sl.addWidget(self.scrub_time)
        layout.addWidget(scrub_frame)

        # Cursor readouts
        readout_frame = QFrame()
        readout_frame.setStyleSheet("background-color: #0d1117; border: 1px solid #21262d; border-radius: 6px;")
        rg = QGridLayout(readout_frame)
        rg.setContentsMargins(8, 6, 8, 6)
        rg.setSpacing(6)

        readout_defs = [
            ("Time", "s"), ("Altitude", "m"), ("Velocity", "m/s"), ("Accel", "m/s²"),
            ("Mach", ""), ("Thrust", "N"), ("Drag", "N"), ("Mass", "kg"),
            ("Phase", ""), ("Dyn Press", "Pa"), ("Stability", "cal"), ("Cd", ""),
        ]
        self.cursor_readouts = {}
        for i, (name, unit) in enumerate(readout_defs):
            col = i % 6
            row = (i // 6) * 2
            header = QLabel(name)
            header.setStyleSheet("color: #58a6ff; font-weight: 600; font-size: 10px;")
            header.setAlignment(Qt.AlignmentFlag.AlignCenter)
            rg.addWidget(header, row, col)
            val = ReadoutValue("—")
            self.cursor_readouts[name] = (val, unit)
            rg.addWidget(val, row + 1, col)

        layout.addWidget(readout_frame)

        # Plot tabs
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("QTabWidget::pane { border: 1px solid #21262d; }")

        self.alt_plot = PlotWidget(title="Altitude vs Time", xlabel="Time (s)", ylabel="Altitude (m)")
        self.tabs.addTab(self.alt_plot, "Altitude")

        self.vel_plot = PlotWidget(title="Velocity vs Time", xlabel="Time (s)", ylabel="Velocity (m/s)")
        self.tabs.addTab(self.vel_plot, "Velocity")

        self.accel_plot = PlotWidget(title="Acceleration vs Time", xlabel="Time (s)", ylabel="Accel (m/s²)")
        self.tabs.addTab(self.accel_plot, "Acceleration")

        self.thrust_plot = PlotWidget(title="Thrust & Drag vs Time", xlabel="Time (s)", ylabel="Force (N)")
        self.tabs.addTab(self.thrust_plot, "Forces")

        self.mass_plot = PlotWidget(title="Mass vs Time", xlabel="Time (s)", ylabel="Mass (kg)")
        self.tabs.addTab(self.mass_plot, "Mass")

        self.mach_plot = PlotWidget(title="Mach Number vs Time", xlabel="Time (s)", ylabel="Mach")
        self.tabs.addTab(self.mach_plot, "Mach")

        self.stability_plot = PlotWidget(title="Stability Margin vs Time", xlabel="Time (s)", ylabel="Calibers")
        self.tabs.addTab(self.stability_plot, "Stability")

        self.dyn_press_plot = PlotWidget(title="Dynamic Pressure vs Time", xlabel="Time (s)", ylabel="q (Pa)")
        self.tabs.addTab(self.dyn_press_plot, "Dyn. Pressure")

        layout.addWidget(self.tabs, 1)

    def _check_data(self, state):
        if self.sim_engine and self.sim_engine.history.count > 0:
            n = self.sim_engine.history.count
            self.summary_label.setText(f"Simulation data available: {n} data points. Click 'Refresh Plots'.")

    def _get_history(self):
        """Get history from the new HistoryManager or legacy engine."""
        if self.sim_engine and hasattr(self.sim_engine, 'history') and self.sim_engine.history.count > 0:
            return self.sim_engine.history
        # Legacy fallback
        if hasattr(self.engine, 'sim_history') and self.engine.sim_history:
            return None  # signal to use legacy path
        return None

    def refresh_plots(self):
        history = self._get_history()
        if history is None:
            self.summary_label.setText("No simulation data. Run a simulation first.")
            return

        self._history = history
        t_vals = history.get_values("time")
        if not t_vals:
            return

        # Update slider range
        self.scrub_slider.setRange(0, max(1, len(t_vals) - 1))
        self.scrub_slider.setValue(len(t_vals) - 1)

        # Plot all
        self.alt_plot.update_plot(t_vals, history.get_values("altitude"),
            "Altitude vs Time", "Time (s)", "Altitude (m)", "#58a6ff")

        self.vel_plot.update_plot(t_vals, history.get_values("velocity"),
            "Velocity vs Time", "Time (s)", "Velocity (m/s)", "#7ee787")

        self.accel_plot.update_plot(t_vals, history.get_values("acceleration"),
            "Acceleration vs Time", "Time (s)", "Accel (m/s²)", "#f0883e")

        self.thrust_plot.multi_plot([
            (t_vals, history.get_values("thrust"), "#f0883e", "Thrust"),
            (t_vals, history.get_values("drag"), "#f85149", "Drag"),
        ], "Thrust & Drag vs Time", "Time (s)", "Force (N)")

        self.mass_plot.update_plot(t_vals, history.get_values("mass"),
            "Mass vs Time", "Time (s)", "Mass (kg)", "#d29922")

        self.mach_plot.update_plot(t_vals, history.get_values("mach"),
            "Mach Number vs Time", "Time (s)", "Mach", "#bc8cff")

        self.stability_plot.update_plot(t_vals, history.get_values("stability_margin"),
            "Stability Margin vs Time", "Time (s)", "Calibers", "#f778ba")

        self.dyn_press_plot.update_plot(t_vals, history.get_values("dynamic_pressure"),
            "Dynamic Pressure vs Time", "Time (s)", "q (Pa)", "#79c0ff")

        # Summary
        s = self.engine.state
        self.summary_label.setText(
            f"Apogee: {s.max_altitude:.1f} m  |  Max Vel: {s.max_velocity:.1f} m/s  |  "
            f"Max Mach: {s.max_mach:.3f}  |  Max Accel: {s.max_acceleration:.1f} m/s²  |  "
            f"Flight Time: {t_vals[-1]:.2f} s  |  {history.count} data points"
        )
        self.summary_label.setStyleSheet("color: #7ee787; font-size: 13px; padding: 8px; "
            "background-color: #161b22; border: 1px solid #21262d; border-radius: 6px; font-weight: 600;")

    def _on_scrub(self, index):
        """Update cursor readouts when scrubber moves."""
        if self._history is None or index >= self._history.count:
            return

        snap = self._history.get_snapshot(index)
        if not snap:
            return

        self.scrub_time.setText(f"{snap.get('time', 0):.2f} s")
        
        t_val = snap.get('time', 0)
        self.alt_plot.set_cursor(t_val)
        self.vel_plot.set_cursor(t_val)
        self.accel_plot.set_cursor(t_val)
        self.thrust_plot.set_cursor(t_val)
        self.mass_plot.set_cursor(t_val)
        self.mach_plot.set_cursor(t_val)
        self.stability_plot.set_cursor(t_val)
        self.dyn_press_plot.set_cursor(t_val)

        field_map = {
            "Time": ("time", "{:.2f}"),
            "Altitude": ("altitude", "{:.1f}"),
            "Velocity": ("velocity", "{:.1f}"),
            "Accel": ("acceleration", "{:.1f}"),
            "Mach": ("mach", "{:.3f}"),
            "Thrust": ("thrust", "{:.1f}"),
            "Drag": ("drag", "{:.1f}"),
            "Mass": ("mass", "{:.3f}"),
            "Phase": ("phase", "{}"),
            "Dyn Press": ("dynamic_pressure", "{:.0f}"),
            "Stability": ("stability_margin", "{:.2f}"),
            "Cd": ("cd", "{:.3f}"),
        }

        for label, (field, fmt) in field_map.items():
            val = snap.get(field, 0)
            widget, unit = self.cursor_readouts[label]
            text = fmt.format(val)
            if unit:
                text += f" {unit}"
            widget.setText(text)

    def _export_csv(self):
        if self._history is None or self._history.count == 0:
            # Try legacy
            if self.sim_engine and hasattr(self.sim_engine, 'history'):
                self.sim_engine.history.export_csv(
                    str(Path.home() / "Documents" / "K2_flight_results.csv"))
            return

        path, _ = QFileDialog.getSaveFileName(self, "Export Results",
            str(Path.home() / "Documents" / "K2_flight_results.csv"),
            "CSV Files (*.csv)")
        if not path:
            return

        self._history.export_csv(path)
        self.engine.log_message.emit(f"Results exported: {path}")
