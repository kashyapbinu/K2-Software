"""
K2 AeroSim — Results Workspace
Professional post-flight data visualization with synchronized cursor.
"""
import csv, logging
from pathlib import Path
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QTabWidget, QLabel, QFileDialog, QSlider, QFrame, QGridLayout)
from PyQt6.QtCore import Qt
from ui.icons import icon
from ui.widgets.plot_widget import PlotWidget
from core.flight_log import parse_flight_log_file, compare_apogee

from ui import theme
from ui.widgets.readout import Readout

logger = logging.getLogger("K2.ResultsWS")

# (FlightSummary field, tile label). Formatting lives in summary_text().
SUMMARY_TILES = [
    ("apogee", "Apogee"),
    ("max_velocity", "Max Velocity"),
    ("max_mach", "Max Mach"),
    ("max_acceleration", "Max Accel (sensed)"),
    ("max_q", "Max-Q"),
    ("rail_exit_velocity", "Rail Exit"),
    ("stability_at_rail_exit", "Stability @ Rail"),
    ("min_stability", "Min Stability"),
    ("drogue_descent_rate", "Drogue Descent"),
    ("landing_velocity", "Landing Speed"),
    ("landing_drift", "Drift"),
    ("flight_time", "Flight Time"),
]


def canopy_line(c: dict) -> str:
    """One canopy report as a line: when and where it opened, the speed it
    opened at, its peak force, and the descent rate under it."""
    name = c.get("name", "Parachute")
    if c.get("deploy_time") is None:
        return f"{name}: did not open"
    parts = [f"opened at {c['deploy_altitude']:.0f} m, T+{c['deploy_time']:.1f} s",
             f"{c.get('deploy_speed') or 0.0:.0f} m/s",
             f"peak {c.get('peak_force', 0.0):.0f} N"]
    if c.get("descent_rate") is not None:
        parts.append(f"descent {c['descent_rate']:.1f} m/s")
    return f"{name}: " + ", ".join(parts)


def summary_text(summary, key: str) -> str:
    """One FlightSummary value as tile text; "—" for a value the flight did
    not produce (None), never a made-up zero."""
    from ui.units import format_value
    value = getattr(summary, key, None) if summary is not None else None
    if value is None:
        return "—"
    if key == "max_acceleration":
        return f"{format_value(value, 'acceleration')} ({value / 9.80665:.1f} g)"
    if key == "max_q":
        return format_value(value, "pressure")
    if key == "max_mach":
        return f"{value:.3f}"
    if key in ("stability_at_rail_exit", "min_stability"):
        return f"{value:.2f} cal"
    if key == "flight_time":
        return f"{value:.1f} s"
    if key == "apogee":
        return format_value(value, "altitude")
    if key == "landing_drift":
        return format_value(value, "distance")
    return format_value(value, "velocity")


class ResultsWorkspace(QWidget):
    def __init__(self, engine, sim_engine=None, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.sim_engine = sim_engine
        self._history = None
        self._flight_log = None      # imported real flight log for sim-vs-real overlay
        # The cursor is the shared replay instant (ui.replay_clock): moving it
        # here moves the rocket in the Mission Visualizer, and the reverse.
        self._clock = None
        self._times = []             # recorded times, for time -> slider index
        self._following_clock = False
        self._setup_ui()
        self.engine.state_changed.connect(self._check_data)
        if sim_engine is not None and hasattr(sim_engine, "sim_finished"):
            from ui.replay_clock import clock_for
            self._clock = clock_for(sim_engine)
            self._clock.time_changed.connect(self._on_clock_time)
        if sim_engine is not None and hasattr(sim_engine, "flight_cleared"):
            sim_engine.flight_cleared.connect(self._on_flight_cleared)

    def _on_flight_cleared(self):
        """The engine dropped its flight (project opened, Reset, a new run
        starting): blank the plots, tiles and cursor that showed it. They
        used to stay until the next Refresh, and the report took its history
        from the copy kept here."""
        self._history = None
        self._times = []
        self.reset_workspace()

    def _on_clock_time(self, t: float):
        """Another view moved the replay instant: bring the cursor to it.
        A hidden tab waits until it is shown (eight plots redraw per move)."""
        if not self._times or not self.isVisible():
            return
        import bisect
        i = bisect.bisect_left(self._times, t)
        if i > 0 and (i == len(self._times) or t - self._times[i - 1] <= self._times[i] - t):
            i -= 1
        if i != self.scrub_slider.value():
            self._following_clock = True
            try:
                self.scrub_slider.setValue(i)
            finally:
                self._following_clock = False

    def showEvent(self, event):
        super().showEvent(event)
        if self._clock is not None and self._clock.replay is not None:
            self._on_clock_time(self._clock.time)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Header
        top = QHBoxLayout()
        title = QLabel("Flight Results")
        title.setStyleSheet(theme.panel_title_qss())
        top.addWidget(title)
        top.addStretch()

        self.btn_refresh = QPushButton(icon("refresh"), "Refresh Plots")
        self.btn_refresh.clicked.connect(self.refresh_plots)
        top.addWidget(self.btn_refresh)

        self.btn_export = QPushButton(icon("export"), "Export CSV")
        self.btn_export.clicked.connect(self._export_csv)
        top.addWidget(self.btn_export)

        # Import a real flight log (altimeter/GPS CSV) to overlay on the sim.
        self.btn_import_log = QPushButton(icon("import"), "Import Flight Log")
        self.btn_import_log.setToolTip(
            "Overlay a real altimeter/GPS CSV (time + altitude, optional "
            "velocity/accel) on the simulated trajectory for validation.")
        self.btn_import_log.clicked.connect(self._import_flight_log)
        top.addWidget(self.btn_import_log)

        self.btn_clear_log = QPushButton("Clear Log")
        self.btn_clear_log.clicked.connect(self._clear_flight_log)
        self.btn_clear_log.setVisible(False)
        top.addWidget(self.btn_clear_log)

        self.btn_report = QPushButton(icon("report"), "Technical Report…")
        self.btn_report.setToolTip(
            "One PDF for judges / range safety: vehicle, launch conditions, "
            "flight, recovery, Monte Carlo dispersion, margins, rules check "
            "and model limits.")
        self.btn_report.clicked.connect(self._on_report)
        top.addWidget(self.btn_report)
        layout.addLayout(top)

        # Summary
        self.summary_label = QLabel("No simulation data. Run a simulation first.")
        self.summary_label.setStyleSheet(f"color: {theme.TEXT_DIM}; font-size: 12px;")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        # Flight summary (core.flight_summary): the headline numbers, from
        # the same object the rules check and the report read.
        sg = QGridLayout()
        sg.setContentsMargins(0, 0, 0, 0)
        sg.setSpacing(6)
        self.summary_tiles = {}
        for i, (key, label) in enumerate(SUMMARY_TILES):
            val = Readout(label)
            self.summary_tiles[key] = val
            sg.addWidget(val, i // 6, i % 6)
            sg.setColumnStretch(i % 6, 1)
        layout.addLayout(sg)

        # Timeline scrubber
        sl = QHBoxLayout()
        sl.setContentsMargins(0, 4, 0, 0)
        sl.setSpacing(10)
        scrub_caption = QLabel("Cursor")
        scrub_caption.setStyleSheet(f"color: {theme.TEXT_DIM}; font-size: 11px; font-weight: 600;")
        sl.addWidget(scrub_caption)
        self.scrub_slider = QSlider(Qt.Orientation.Horizontal)
        self.scrub_slider.setRange(0, 1000)
        self.scrub_slider.setValue(0)
        self.scrub_slider.valueChanged.connect(self._on_scrub)
        sl.addWidget(self.scrub_slider, 1)
        self.scrub_time = QLabel("T+ 0.00 s")
        self.scrub_time.setStyleSheet(
            f"color: {theme.TEXT_BRIGHT}; font-family: {theme.MONO}; font-weight: 600;")
        sl.addWidget(self.scrub_time)
        layout.addLayout(sl)

        # Cursor readouts
        rg = QGridLayout()
        rg.setContentsMargins(0, 0, 0, 0)
        rg.setSpacing(6)

        readout_defs = [
            ("Time", "s"), ("Altitude", "m"), ("Velocity", "m/s"), ("Accel", "m/s²"),
            ("Mach", ""), ("Thrust", "N"), ("Drag", "N"), ("Mass", "kg"),
            ("Phase", ""), ("Dyn Press", "Pa"), ("Stability", "cal"), ("Cd", ""),
        ]
        self.cursor_readouts = {}
        for i, (name, unit) in enumerate(readout_defs):
            val = Readout(name, compact=True)
            self.cursor_readouts[name] = (val, unit)
            rg.addWidget(val, i // 6, i % 6)
            rg.setColumnStretch(i % 6, 1)
        layout.addLayout(rg)

        # Plot tabs
        self.tabs = QTabWidget()

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

        # Top-down landing footprint: pad, main vehicle, spent boosters.
        self.landing_plot = PlotWidget(title="Landing Footprint", xlabel="Downrange X (m)", ylabel="Crossrange Y (m)")
        self.tabs.addTab(self.landing_plot, "Landing")

        from ui.widgets.rules_panel import RulesPanel
        self.rules_panel = RulesPanel()
        self.tabs.addTab(self.rules_panel, icon("check"), "Rules")
        self.rules_panel.checked.connect(self._stamp_rules)

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

    def reset_workspace(self):
        """Clear results to the empty state (called on New Project)."""
        from ui.workspace_reset import clear_visuals
        clear_visuals(self)
        self.refresh_plots()   # history is already cleared → shows the empty state

    def refresh_plots(self):
        history = self._get_history()
        if history is None:
            self.summary_label.setText("No simulation data. Run a simulation first.")
            self._show_summary(None)
            return

        self._history = history
        t_vals = history.get_values("time")
        if not t_vals:
            return
        self._times = list(t_vals)

        # Update slider range
        self.scrub_slider.setRange(0, max(1, len(t_vals) - 1))
        self.scrub_slider.setValue(len(t_vals) - 1)

        # Plot all
        self._plot_with_overlay(self.alt_plot, t_vals, history.get_values("altitude"),
            "altitude", "Altitude vs Time", "Time (s)", "Altitude (m)", theme.ACCENT)

        self._plot_with_overlay(self.vel_plot, t_vals, history.get_values("velocity"),
            "velocity", "Velocity vs Time", "Time (s)", "Velocity (m/s)", theme.OK)

        self._plot_with_overlay(self.accel_plot, t_vals, history.get_values("acceleration"),
            "acceleration", "Acceleration vs Time", "Time (s)", "Accel (m/s²)", theme.ACCENT)

        self.thrust_plot.multi_plot([
            (t_vals, history.get_values("thrust"), theme.ACCENT, "Thrust"),
            (t_vals, history.get_values("drag"), theme.ERR, "Drag"),
        ], "Thrust & Drag vs Time", "Time (s)", "Force (N)")

        self.mass_plot.update_plot(t_vals, history.get_values("mass"),
            "Mass vs Time", "Time (s)", "Mass (kg)", theme.WARN)

        self.mach_plot.update_plot(t_vals, history.get_values("mach"),
            "Mach Number vs Time", "Time (s)", "Mach", theme.SERIES_VIOLET)

        self.stability_plot.update_plot(t_vals, history.get_values("stability_margin"),
            "Stability Margin vs Time", "Time (s)", "Calibers", theme.SERIES_TEAL)

        self.dyn_press_plot.update_plot(t_vals, history.get_values("dynamic_pressure"),
            "Dynamic Pressure vs Time", "Time (s)", "q (Pa)", theme.ACCENT_HOVER)

        self._plot_landing()

        # Summary — every number from the flight's FlightSummary.
        summary = getattr(self.sim_engine, "summary", None)
        self._show_summary(summary)
        self.rules_panel.set_flights(summary, self.engine.state)
        status = summary.final_phase if summary is not None else "In flight"
        text = f"Flight: {status}  |  {history.count} data points"
        for c in (summary.canopies if summary is not None else []):
            text += "\n" + canopy_line(c)
        if summary is not None and summary.warnings:
            text += "\n⚠ " + "\n⚠ ".join(summary.warnings)
        self.summary_label.setText(text)
        self.summary_label.setStyleSheet(f"color: {theme.TEXT}; font-size: 12px;")

        # Append sim-vs-measured comparison when a flight log is loaded.
        if self._flight_log and summary is not None:
            cmp = compare_apogee(summary.apogee, self._flight_log)
            self.summary_label.setText(
                self.summary_label.text() +
                f"\nLog '{self._flight_log['source']}': "
                f"Measured apogee: {cmp['measured_apogee']:.1f} m  |  "
                f"Sim error: {cmp['error_m']:+.1f} m ({cmp['error_pct']:+.1f}%)")

    # ── Technical report ──

    def report_inputs(self) -> dict:
        """Everything the technical report can include, from wherever it
        was computed. Missing pieces stay missing (the report says 'Not run')."""
        main = self.window()
        s = self.engine.state
        out = {"state": s,
               "summary": getattr(self.sim_engine, "summary", None),
               "history": self._history if self._history is not None
               else self._get_history(),
               "rule_set": self.rules_panel.current_set(),
               "rule_results": self.rules_panel.results() or None}
        mc = getattr(main, "monte_carlo_ws", None)
        if mc is not None and getattr(mc, "_results", None) is not None:
            from ui.workspaces.monte_carlo_workspace import landing_dispersion, landing_points
            out["dispersion"] = landing_dispersion(mc._results)
            out["landings"] = landing_points(mc._results)
        margins = {}
        if getattr(s, "flutter_speed", 0.0) > 0:
            margins["flutter_speed"] = s.flutter_speed
        wks = getattr(getattr(main, "structures_ws", None), "_wks_report", None)
        weakest = getattr(getattr(wks, "failure", None), "weakest", None)
        if weakest is not None and weakest.margin > 0:
            margins["structural_sf"] = weakest.margin
            margins["structures_source"] = f"{weakest.name}, Structures workstation"
        out["margins"] = margins
        # Which of those results were made for an earlier version of the
        # rocket; the report marks them (core.technical_report).
        from core.readiness import STALE
        out["out_of_date"] = {st.stage.key: st.reason
                              for st in self.engine.readiness.all(s) if st.status == STALE}
        return out

    def _on_report(self, path: str = None):
        from core.technical_report import build_content, render_pdf
        if not path:
            path, _ = QFileDialog.getSaveFileName(
                self, "Technical Report",
                str(Path.home() / "Documents" / f"{self.engine.state.name}_report.pdf"),
                "PDF (*.pdf)")
            if not path:
                return
        inp = self.report_inputs()
        rep = build_content(inp.pop("state"), inp.pop("summary"), **inp)
        render_pdf(rep, path)
        self.engine.analysis_done("report", Path(path).name)
        self.engine.log_message.emit(f"Technical report saved: {path}")

    def _stamp_rules(self, *_):
        """Record the rules check on the Readiness board: how many rules the
        flight passed, of those that have a limit. A set with no limits (the
        built-in template) is not a check, so it stays 'not run'."""
        from core import rules as R
        summary = getattr(self.sim_engine, "summary", None)
        if summary is None or summary.final_phase != "Landed":
            return
        judged = [r for r in self.rules_panel.results() if r.status in (R.PASS, R.FAIL)]
        if not judged:
            self.engine.readiness.forget("rules")
            self.engine.readiness_changed.emit()
            return
        passed = sum(1 for r in judged if r.status == R.PASS)
        self.engine.analysis_done("rules", f"{passed} of {len(judged)} rules pass",
                                  verdict=passed == len(judged))

    def _show_summary(self, summary):
        for key, tile in self.summary_tiles.items():
            tile.setText(summary_text(summary, key))

    def _plot_with_overlay(self, plot, t, y, meas_key, title, xlabel, ylabel, color):
        """Plot the sim series, overlaying the imported measured channel if present."""
        log = self._flight_log
        if log and log.get(meas_key):
            plot.multi_plot([
                (t, y, color, "Simulated"),
                (log["time"], log[meas_key], theme.SERIES_TEAL, "Measured"),
            ], title, xlabel, ylabel)
        else:
            plot.update_plot(t, y, title, xlabel, ylabel, color)

    def _plot_landing(self):
        """Top-down landing footprint: pad at origin, main vehicle touchdown,
        and every spent-stage ballistic impact (range safety)."""
        import math as _m
        from matplotlib.patches import Circle

        plot = self.landing_plot
        ax = plot.ax
        ax.clear()
        plot._style_axis("Landing Footprint", "Downrange X (m)", "Crossrange Y (m)")

        s = self.engine.state

        # Collected impact points: (x, y, label, color, impact_velocity).
        points = [(0.0, 0.0, "Pad", theme.TEXT_DIM, None)]

        main_x = getattr(s, "landing_x", 0.0)
        main_y = getattr(s, "landing_y", 0.0)
        main_v = getattr(s, "main_descent_rate", 0.0) or getattr(s, "touchdown_rate", 0.0)
        if getattr(s, "landing_drift", 0.0) or main_x or main_y:
            points.append((main_x, main_y, "Main vehicle", theme.OK, main_v))

        # Spent stages (multistage only). Names come from the snapshots; the
        # ballistic results carry the landing position + impact speed.
        results = getattr(self.sim_engine, "spent_stage_results", None) or []
        snaps = getattr(self.sim_engine, "spent_stages", None) or []
        palette = [theme.ACCENT, theme.ERR, theme.WARN, theme.SERIES_VIOLET]
        for i, r in enumerate(results):
            name = snaps[i].get("name") if i < len(snaps) else None
            name = name or f"Stage {r.get('stage', i)}"
            points.append((r.get("landing_x", 0.0), r.get("landing_y", 0.0),
                           name, palette[i % len(palette)],
                           r.get("impact_velocity", 0.0)))

        # Range rings sized to the farthest impact.
        max_r = max((_m.hypot(x, y) for x, y, *_ in points), default=0.0)
        if max_r > 0:
            step = max(10.0, round(max_r / 3.0 / 10.0) * 10.0)
            ring = step
            while ring <= max_r * 1.15 + step:
                ax.add_patch(Circle((0, 0), ring, fill=False, ls="--",
                                    ec=theme.LINE, lw=0.8, alpha=0.7))
                ax.text(ring * 0.7071, ring * 0.7071, f"{ring:.0f} m",
                        color=theme.LINE_STRONG, fontsize=7, ha="center", va="center")
                ring += step

        for x, y, label, color, vimp in points:
            ax.scatter([x], [y], c=color, s=90, edgecolors=theme.BG,
                       linewidths=1.2, zorder=5, label=label)
            tag = label if vimp is None else f"{label}\n{vimp:.0f} m/s"
            ax.annotate(tag, (x, y), textcoords="offset points", xytext=(8, 6),
                        color=theme.TEXT, fontsize=8, zorder=6)

        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(facecolor=theme.PANEL, edgecolor=theme.LINE,
                  labelcolor=theme.TEXT, fontsize=8, loc="best")
        plot.figure.tight_layout()
        plot.canvas.draw()

    def _import_flight_log(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Flight Log", str(Path.home() / "Documents"),
            "Flight Log (*.csv *.txt);;All Files (*)")
        if not path:
            return
        try:
            self._flight_log = parse_flight_log_file(path)
        except Exception as exc:
            self.engine.log_message.emit(f"Flight log import failed: {exc}")
            return
        self.btn_clear_log.setVisible(True)
        self.engine.log_message.emit(
            f"Imported flight log '{self._flight_log['source']}' "
            f"({self._flight_log['n_points']} pts, apogee {self._flight_log['apogee']:.0f} m)")
        self.refresh_plots()

    def _clear_flight_log(self):
        self._flight_log = None
        self.btn_clear_log.setVisible(False)
        self.refresh_plots()

    def _on_scrub(self, index):
        """Update cursor readouts when scrubber moves."""
        if self._history is None or index >= self._history.count:
            return

        snap = self._history.get_snapshot(index)
        if not snap:
            return

        self.scrub_time.setText(f"T+ {snap.get('time', 0):.2f} s")
        
        t_val = snap.get('time', 0)
        if self._clock is not None and not self._following_clock:
            self._clock.seek(t_val)
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
            if isinstance(val, float) and val != val:   # NaN → undefined
                text = "—"
            else:
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
