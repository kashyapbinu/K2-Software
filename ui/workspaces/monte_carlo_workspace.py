"""
K2 AeroSim — Monte Carlo Workspace
======================================
Professional Monte Carlo analysis interface with configuration inputs,
matplotlib visualizations, and statistical results in a 3-panel layout.
"""

import csv
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse
from scipy import stats

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout, QLabel,
    QPushButton, QSplitter, QFrame, QScrollArea, QSpinBox,
    QDoubleSpinBox, QProgressBar, QTabWidget, QFileDialog, QMessageBox,
    QDialog, QDialogButtonBox, QTextEdit,
)
from PyQt6.QtCore import Qt
from ui.icons import icon

from core.monte_carlo_engine import (
    MonteCarloConfig, MonteCarloEngine, MonteCarloResults,
)

logger = logging.getLogger("K2.MonteCarloWS")

# ── Stylesheet constants ────────────────────────────────────────────────────

_GRP = """
QGroupBox { color:#8b949e; font-size:11px; font-weight:600;
  border:1px solid #21262d; border-radius:6px; margin-top:10px; padding-top:6px; }
QGroupBox::title { subcontrol-origin:margin; left:10px; padding:0 4px; }
"""

_BTN_P = """
QPushButton { background:#1f6feb; color:#fff; font-weight:700; font-size:12px;
  border:none; border-radius:6px; padding:9px 14px; }
QPushButton:hover { background:#388bfd; }
QPushButton:disabled { background:#21262d; color:#484f58; }
"""

_BTN_D = """
QPushButton { background:#da3633; color:#fff; font-weight:700; font-size:12px;
  border:none; border-radius:6px; padding:9px 14px; }
QPushButton:hover { background:#f85149; }
QPushButton:disabled { background:#21262d; color:#484f58; }
"""

_BTN_S = """
QPushButton { background:#21262d; color:#c9d1d9; font-weight:600; font-size:11px;
  border:1px solid #30363d; border-radius:6px; padding:7px 12px; }
QPushButton:hover { background:#30363d; border-color:#58a6ff; }
QPushButton:disabled { color:#484f58; }
"""

_VAL = ("color:#e6edf3; font-family:'Cascadia Code',monospace; font-size:13px;"
        "font-weight:600; padding:2px 6px; background:#161b22; border-radius:4px;")


def _vl(text: str = "—") -> QLabel:
    """Create a monospace value label."""
    lbl = QLabel(text)
    lbl.setStyleSheet(_VAL)
    return lbl


# ── Dark-themed matplotlib helpers ───────────────────────────────────────────

def _style_ax(ax, title: str = "", xlabel: str = "", ylabel: str = ""):
    """Apply K2 dark-theme styling to a matplotlib Axes."""
    ax.set_facecolor("#161b22")
    ax.set_title(title, color="#58a6ff", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, color="#8b949e", fontsize=10)
    ax.set_ylabel(ylabel, color="#8b949e", fontsize=10)
    ax.tick_params(colors="#484f58", labelsize=9)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color("#30363d")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(True, alpha=0.15, color="#30363d")


def _dark_figure(rows: int = 1, cols: int = 1,
                 figsize: tuple = (8, 5)) -> tuple:
    """Return (Figure, axes) with dark background."""
    fig = Figure(figsize=figsize, dpi=100)
    fig.patch.set_facecolor("#0d1117")
    if rows == 1 and cols == 1:
        ax = fig.add_subplot(111)
        _style_ax(ax)
        return fig, ax
    axes = fig.subplots(rows, cols)
    for a in (axes.flat if hasattr(axes, "flat") else [axes]):
        _style_ax(a)
    return fig, axes


# ═════════════════════════════════════════════════════════════════════════════
# Workspace
# ═════════════════════════════════════════════════════════════════════════════

class MonteCarloWorkspace(QWidget):
    """3-panel Monte Carlo analysis workspace."""

    def __init__(self, engine, sim_engine=None, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.sim_engine = sim_engine
        self._mc_engine: Optional[MonteCarloEngine] = None
        self._results: Optional[MonteCarloResults] = None
        self._setup_ui()
        self.engine.state_changed.connect(self._on_state_changed)

    # ── UI construction ──────────────────────────────────────────────────────

    def _setup_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_left())
        splitter.addWidget(self._build_center())
        splitter.addWidget(self._build_right())
        splitter.setSizes([320, 760, 340])
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter)

    # ── LEFT PANEL ───────────────────────────────────────────────────────────

    def _build_left(self) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumWidth(340)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(12, 14, 12, 14)
        lay.setSpacing(12)

        # Title
        title = QLabel("MONTE CARLO ANALYSIS")
        title.setStyleSheet(
            "color:#58a6ff; font-size:16px; font-weight:700; "
            "letter-spacing:2px; padding:2px 0 6px 0;"
        )
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(title)

        # ── Simulation Count ──
        g1 = QGroupBox("Simulation Count")
        g1.setStyleSheet(_GRP)
        f1 = QFormLayout()
        f1.setSpacing(8)
        self.spin_num = QSpinBox()
        self.spin_num.setRange(10, 10000)
        self.spin_num.setSingleStep(10)
        self.spin_num.setValue(500)
        f1.addRow("Runs:", self.spin_num)
        g1.setLayout(f1)
        lay.addWidget(g1)

        # ── Wind Uncertainty ──
        g2 = QGroupBox("Wind Uncertainty")
        g2.setStyleSheet(_GRP)
        f2 = QFormLayout()
        f2.setSpacing(8)
        self.spin_wind_speed = QDoubleSpinBox()
        self.spin_wind_speed.setRange(0, 50)
        self.spin_wind_speed.setSuffix(" %")
        self.spin_wind_speed.setDecimals(1)
        self.spin_wind_speed.setValue(10.0)
        f2.addRow("Speed ±:", self.spin_wind_speed)
        self.spin_wind_dir = QDoubleSpinBox()
        self.spin_wind_dir.setRange(0, 45)
        self.spin_wind_dir.setSuffix(" °")
        self.spin_wind_dir.setDecimals(1)
        self.spin_wind_dir.setValue(15.0)
        f2.addRow("Direction ±:", self.spin_wind_dir)
        g2.setLayout(f2)
        lay.addWidget(g2)

        # ── Vehicle Uncertainty ──
        g3 = QGroupBox("Vehicle Uncertainty")
        g3.setStyleSheet(_GRP)
        f3 = QFormLayout()
        f3.setSpacing(8)
        self.spin_dry_mass = QDoubleSpinBox()
        self.spin_dry_mass.setRange(0, 20)
        self.spin_dry_mass.setSuffix(" %")
        self.spin_dry_mass.setDecimals(1)
        self.spin_dry_mass.setValue(3.0)
        f3.addRow("Dry Mass ±:", self.spin_dry_mass)
        self.spin_cd = QDoubleSpinBox()
        self.spin_cd.setRange(0, 30)
        self.spin_cd.setSuffix(" %")
        self.spin_cd.setDecimals(1)
        self.spin_cd.setValue(10.0)
        f3.addRow("Drag Coeff ±:", self.spin_cd)
        self.spin_cg = QDoubleSpinBox()
        self.spin_cg.setRange(0, 50)
        self.spin_cg.setSuffix(" mm")
        self.spin_cg.setDecimals(1)
        self.spin_cg.setValue(5.0)
        f3.addRow("CG ±:", self.spin_cg)
        g3.setLayout(f3)
        lay.addWidget(g3)

        # ── Motor Uncertainty ──
        g4 = QGroupBox("Motor Uncertainty")
        g4.setStyleSheet(_GRP)
        f4 = QFormLayout()
        f4.setSpacing(8)
        self.spin_impulse = QDoubleSpinBox()
        self.spin_impulse.setRange(0, 15)
        self.spin_impulse.setSuffix(" %")
        self.spin_impulse.setDecimals(1)
        self.spin_impulse.setValue(5.0)
        f4.addRow("Impulse ±:", self.spin_impulse)
        g4.setLayout(f4)
        lay.addWidget(g4)

        # ── Launch Uncertainty ──
        g5 = QGroupBox("Launch Uncertainty")
        g5.setStyleSheet(_GRP)
        f5 = QFormLayout()
        f5.setSpacing(8)
        self.spin_launch_angle = QDoubleSpinBox()
        self.spin_launch_angle.setRange(0, 10)
        self.spin_launch_angle.setSuffix(" °")
        self.spin_launch_angle.setDecimals(1)
        self.spin_launch_angle.setValue(2.0)
        f5.addRow("Angle ±:", self.spin_launch_angle)
        g5.setLayout(f5)
        lay.addWidget(g5)

        # ── Failure Criteria ──
        g6 = QGroupBox("Failure Criteria")
        g6.setStyleSheet(_GRP)
        f6 = QFormLayout()
        f6.setSpacing(8)
        self.spin_min_stab = QDoubleSpinBox()
        self.spin_min_stab.setRange(0, 3)
        self.spin_min_stab.setSuffix(" cal")
        self.spin_min_stab.setDecimals(1)
        self.spin_min_stab.setValue(1.0)
        f6.addRow("Min Stability:", self.spin_min_stab)
        self.spin_min_rev = QDoubleSpinBox()
        self.spin_min_rev.setRange(0, 30)
        self.spin_min_rev.setSuffix(" m/s")
        self.spin_min_rev.setDecimals(0)
        self.spin_min_rev.setValue(15.0)
        f6.addRow("Min Rail Exit:", self.spin_min_rev)
        self.spin_max_mach = QDoubleSpinBox()
        self.spin_max_mach.setRange(0.5, 5.0)
        self.spin_max_mach.setDecimals(1)
        self.spin_max_mach.setValue(2.0)
        f6.addRow("Max Mach:", self.spin_max_mach)
        g6.setLayout(f6)
        lay.addWidget(g6)

        # ── Target ──
        g7 = QGroupBox("Target")
        g7.setStyleSheet(_GRP)
        f7 = QFormLayout()
        f7.setSpacing(8)
        self.spin_target = QDoubleSpinBox()
        self.spin_target.setRange(0, 100000)
        self.spin_target.setSuffix(" m")
        self.spin_target.setDecimals(0)
        self.spin_target.setValue(0)
        f7.addRow("Apogee (0=auto):", self.spin_target)
        g7.setLayout(f7)
        lay.addWidget(g7)

        # ── Actions ──
        g8 = QGroupBox("Actions")
        g8.setStyleSheet(_GRP)
        al = QVBoxLayout()
        al.setSpacing(8)

        self.btn_run = QPushButton(icon("run", color="#fff"), "RUN ANALYSIS")
        self.btn_run.setStyleSheet(_BTN_P)
        self.btn_run.setMinimumHeight(40)
        self.btn_run.clicked.connect(self._on_run)
        al.addWidget(self.btn_run)

        self.btn_cancel = QPushButton(icon("stop", color="#fff"), "CANCEL")
        self.btn_cancel.setStyleSheet(_BTN_D)
        self.btn_cancel.setMinimumHeight(40)
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._on_cancel)
        al.addWidget(self.btn_cancel)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFixedHeight(12)
        self.progress_bar.setStyleSheet(
            "QProgressBar { background:#21262d; border-radius:6px; border:none; "
            "color:#c9d1d9; font-size:9px; }"
            "QProgressBar::chunk { background:#1f6feb; border-radius:6px; }"
        )
        al.addWidget(self.progress_bar)

        self.progress_label = QLabel("Ready")
        self.progress_label.setStyleSheet("color:#8b949e; font-size:11px;")
        al.addWidget(self.progress_label)

        self.btn_export = QPushButton(icon("export"), "EXPORT CSV")
        self.btn_export.setStyleSheet(_BTN_S)
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self._on_export)
        al.addWidget(self.btn_export)

        self.btn_export_pdf = QPushButton(icon("report"), "EXPORT PDF")
        self.btn_export_pdf.setStyleSheet(_BTN_S)
        self.btn_export_pdf.setEnabled(False)
        self.btn_export_pdf.clicked.connect(self._on_export_pdf)
        al.addWidget(self.btn_export_pdf)

        g8.setLayout(al)
        lay.addWidget(g8)

        lay.addStretch()
        scroll.setWidget(inner)
        return scroll

    # ── CENTER PANEL ─────────────────────────────────────────────────────────

    def _build_center(self) -> QWidget:
        wrapper = QWidget()
        vl = QVBoxLayout(wrapper)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(0)

        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(
            "QTabWidget::pane { border:1px solid #21262d; }"
            "QTabBar::tab { background:#161b22; color:#8b949e; padding:6px 14px; "
            "  border:1px solid #21262d; border-bottom:none; border-radius:4px 4px 0 0; }"
            "QTabBar::tab:selected { background:#0d1117; color:#58a6ff; font-weight:700; }"
            "QTabBar::scroller { width:30px; }"
            "QTabBar QToolButton { background:#21262d; border:1px solid #30363d; "
            "  border-radius:4px; margin:2px 1px; width:22px; color:#c9d1d9; }"
            "QTabBar QToolButton:hover { background:#1f6feb; border-color:#1f6feb; }"
            "QTabBar QToolButton:disabled { background:#161b22; border-color:#21262d; }"
        )

        # Tab 1: Apogee Distribution
        fig1, self._ax_apogee = _dark_figure(figsize=(8, 5))
        self._canvas_apogee = FigureCanvas(fig1)
        _style_ax(self._ax_apogee, "Apogee Distribution", "Apogee (m)", "Frequency")
        self.tabs.addTab(self._canvas_apogee, "Apogee Distribution")

        # Tab 2: Landing Dispersion
        fig2, self._ax_landing = _dark_figure(figsize=(8, 5))
        self._canvas_landing = FigureCanvas(fig2)
        _style_ax(self._ax_landing, "Landing Dispersion", "East (m)", "North (m)")
        self.tabs.addTab(self._canvas_landing, "Landing Dispersion")

        # Tab 3: Landing Distance
        fig3, self._ax_dist = _dark_figure(figsize=(8, 5))
        self._canvas_dist = FigureCanvas(fig3)
        _style_ax(self._ax_dist, "Landing Distance", "Distance (m)", "Frequency")
        self.tabs.addTab(self._canvas_dist, "Landing Distance")

        # Tab 4: Performance Box Plots
        fig4, self._ax_box = _dark_figure(rows=1, cols=5, figsize=(14, 5))
        self._canvas_box = FigureCanvas(fig4)
        self.tabs.addTab(self._canvas_box, "Performance Box Plots")

        # Tab 5: Sensitivity / Tornado Chart
        fig5, self._ax_tornado = _dark_figure(figsize=(8, 5))
        self._canvas_tornado = FigureCanvas(fig5)
        _style_ax(self._ax_tornado, "Parameter Sensitivity", "Correlation with Apogee", "")
        self.tabs.addTab(self._canvas_tornado, "Sensitivity Analysis")

        vl.addWidget(self.tabs, 1)
        return wrapper

    # ── RIGHT PANEL ──────────────────────────────────────────────────────────

    def _build_right(self) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumWidth(360)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(12, 14, 12, 14)
        lay.setSpacing(12)

        t = QLabel("Statistics")
        t.setStyleSheet(
            "color:#58a6ff; font-size:15px; font-weight:700; padding:2px 0 6px 0;"
        )
        lay.addWidget(t)

        # ── Apogee Statistics ──
        ga = QGroupBox("Apogee Statistics")
        ga.setStyleSheet(_GRP)
        fa = QFormLayout()
        fa.setSpacing(6)
        self.lbl_ap_mean   = _vl(); fa.addRow("Mean:",    self.lbl_ap_mean)
        self.lbl_ap_std    = _vl(); fa.addRow("Std Dev:", self.lbl_ap_std)
        self.lbl_ap_min    = _vl(); fa.addRow("Min:",     self.lbl_ap_min)
        self.lbl_ap_max    = _vl(); fa.addRow("Max:",     self.lbl_ap_max)
        self.lbl_ap_median = _vl(); fa.addRow("Median:",  self.lbl_ap_median)
        self.lbl_ap_mode   = _vl(); fa.addRow("Mode:",    self.lbl_ap_mode)
        self.lbl_ap_ci     = _vl(); fa.addRow("95% CI:",  self.lbl_ap_ci)
        self.lbl_ap_skew   = _vl(); fa.addRow("Skewness:", self.lbl_ap_skew)
        self.lbl_ap_kurt   = _vl(); fa.addRow("Kurtosis:", self.lbl_ap_kurt)
        ga.setLayout(fa)
        lay.addWidget(ga)

        # ── Performance Statistics ──
        gp = QGroupBox("Performance Statistics")
        gp.setStyleSheet(_GRP)
        fp = QFormLayout()
        fp.setSpacing(6)
        self.lbl_vel_mean  = _vl(); fp.addRow("Max Velocity:", self.lbl_vel_mean)
        self.lbl_mach_mean = _vl(); fp.addRow("Max Mach:",     self.lbl_mach_mean)
        self.lbl_acc_mean  = _vl(); fp.addRow("Max Accel:",    self.lbl_acc_mean)
        self.lbl_rev_mean  = _vl(); fp.addRow("Rail Exit Vel:", self.lbl_rev_mean)
        gp.setLayout(fp)
        lay.addWidget(gp)

        # ── Landing Statistics ──
        gl = QGroupBox("Landing Statistics")
        gl.setStyleSheet(_GRP)
        fl = QFormLayout()
        fl.setSpacing(6)
        self.lbl_ld_mean = _vl(); fl.addRow("Mean Dist:", self.lbl_ld_mean)
        self.lbl_ld_max  = _vl(); fl.addRow("Max Dist:",  self.lbl_ld_max)
        self.lbl_ld_std  = _vl(); fl.addRow("Std Dev:",   self.lbl_ld_std)
        gl.setLayout(fl)
        lay.addWidget(gl)

        # ── Mission Assessment ──
        gm = QGroupBox("Mission Assessment")
        gm.setStyleSheet(_GRP)
        fm = QFormLayout()
        fm.setSpacing(6)
        self.lbl_success  = _vl(); fm.addRow("Success Rate:", self.lbl_success)
        self.lbl_failure  = _vl(); fm.addRow("Failure Rate:", self.lbl_failure)
        self.lbl_p_target = _vl(); fm.addRow("P(Target Alt):", self.lbl_p_target)
        self.lbl_p_safe   = _vl(); fm.addRow("P(Safe Rec.):", self.lbl_p_safe)
        self.lbl_mission  = _vl(); fm.addRow("Mission Success:", self.lbl_mission)
        gm.setLayout(fm)
        lay.addWidget(gm)

        # ── Failure Breakdown ──
        gf = QGroupBox("Failure Breakdown")
        gf.setStyleSheet(_GRP)
        self._fail_layout = QVBoxLayout()
        self._fail_layout.setSpacing(4)
        self._fail_placeholder = QLabel("No data")
        self._fail_placeholder.setStyleSheet("color:#484f58; font-size:11px;")
        self._fail_layout.addWidget(self._fail_placeholder)
        gf.setLayout(self._fail_layout)
        lay.addWidget(gf)

        # ── Scenarios (clickable) ──
        gs = QGroupBox("Scenarios")
        gs.setStyleSheet(_GRP)
        fs = QVBoxLayout()
        fs.setSpacing(6)
        self.btn_best = QPushButton("Best Case: —")
        self.btn_best.setStyleSheet(
            "QPushButton{color:#7ee787;background:#161b22;border:1px solid #30363d;"
            "border-radius:4px;padding:6px;font-size:12px;text-align:left;}"
            "QPushButton:hover{background:#1c2333;border-color:#58a6ff;}"
        )
        self.btn_worst = QPushButton("Worst Case: —")
        self.btn_worst.setStyleSheet(
            "QPushButton{color:#f85149;background:#161b22;border:1px solid #30363d;"
            "border-radius:4px;padding:6px;font-size:12px;text-align:left;}"
            "QPushButton:hover{background:#1c2333;border-color:#58a6ff;}"
        )
        fs.addWidget(self.btn_best)
        fs.addWidget(self.btn_worst)
        gs.setLayout(fs)
        lay.addWidget(gs)

        # ── Reliability Analysis ──
        gr = QGroupBox("Reliability Analysis")
        gr.setStyleSheet(_GRP)
        fr = QFormLayout()
        fr.setSpacing(6)
        self.lbl_r_apogee   = _vl(); fr.addRow("P(Apogee>Target):", self.lbl_r_apogee)
        self.lbl_r_mach     = _vl(); fr.addRow("P(Mach<Limit):",    self.lbl_r_mach)
        self.lbl_r_accel    = _vl(); fr.addRow("P(Accel<100G):",    self.lbl_r_accel)
        self.lbl_r_stab     = _vl(); fr.addRow("P(Stability>Min):", self.lbl_r_stab)
        self.lbl_r_rail     = _vl(); fr.addRow("P(RailV>Min):",     self.lbl_r_rail)
        self.lbl_r_beta     = _vl(); fr.addRow("Reliability (beta):", self.lbl_r_beta)
        self.lbl_r_ci       = _vl(); fr.addRow("Mission 95% CI:",   self.lbl_r_ci)
        gr.setLayout(fr)
        lay.addWidget(gr)

        lay.addStretch()
        scroll.setWidget(inner)
        return scroll

    # ═════════════════════════════════════════════════════════════════════════
    # Actions
    # ═════════════════════════════════════════════════════════════════════════

    def reset_workspace(self):
        """Blank Monte-Carlo results + plots (called on New Project)."""
        self._results = None
        from ui.workspace_reset import clear_visuals
        clear_visuals(self)
        try:
            if hasattr(self, "summary_label"):
                self.summary_label.setText("Run a Monte Carlo analysis to see results.")
        except Exception:
            pass

    def _on_run(self):
        """Collect UI values, create engine, and start analysis."""
        config = MonteCarloConfig(
            num_simulations=self.spin_num.value(),
            wind_speed_uncertainty_pct=self.spin_wind_speed.value(),
            wind_direction_uncertainty_deg=self.spin_wind_dir.value(),
            dry_mass_uncertainty_pct=self.spin_dry_mass.value(),
            drag_coeff_uncertainty_pct=self.spin_cd.value(),
            cg_uncertainty_mm=self.spin_cg.value(),
            motor_impulse_uncertainty_pct=self.spin_impulse.value(),
            launch_angle_uncertainty_deg=self.spin_launch_angle.value(),
            min_stability_cal=self.spin_min_stab.value(),
            min_rail_exit_velocity=self.spin_min_rev.value(),
            max_mach_limit=self.spin_max_mach.value(),
            target_apogee=self.spin_target.value(),
        )

        if self._mc_engine is None:
            self._mc_engine = MonteCarloEngine(self.engine)
        else:
            # Disconnect old signals before reconnecting to avoid duplicates
            try:
                self._mc_engine.progress.disconnect(self._on_progress)
                self._mc_engine.analysis_finished.disconnect(self._on_finished)
                self._mc_engine.analysis_cancelled.disconnect(self._on_cancelled)
                self._mc_engine.analysis_failed.disconnect(self._on_failed)
            except TypeError:
                pass

        self._mc_engine.progress.connect(self._on_progress)
        self._mc_engine.analysis_finished.connect(self._on_finished)
        self._mc_engine.analysis_cancelled.connect(self._on_cancelled)
        self._mc_engine.analysis_failed.connect(self._on_failed)

        # Validate that a motor is configured
        if self.engine.state.motor_total_impulse <= 0.0 and not self.engine.state.custom_thrust_curve:
            QMessageBox.warning(self, "Configuration Error",
                "No motor is configured for this rocket.\n\nPlease select or create a motor in the Propulsion workspace before running Monte Carlo analysis.")
            return

        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.btn_export.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Starting…")

        self._mc_engine.start(config)
        logger.info(f"Monte Carlo analysis launched: {config.num_simulations} runs")

    def _on_failed(self, error_msg: str):
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress_label.setText("Failed")
        QMessageBox.critical(self, "Analysis Failed", f"Monte Carlo analysis failed:\n\n{error_msg}")

    def _on_cancel(self):
        if self._mc_engine is not None:
            self._mc_engine.cancel()
            self.progress_label.setText("Cancelling…")

    def _on_progress(self, completed: int, total: int):
        pct = completed * 100 / max(total, 1)
        self.progress_bar.setValue(int(pct))
        self.progress_label.setText(
            f"Run {completed} / {total} — {pct:.1f}%"
        )

    def _on_finished(self, results: MonteCarloResults):
        self._results = results
        n = len(results.runs)

        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.btn_export.setEnabled(True)
        self.btn_export_pdf.setEnabled(True)
        self.progress_bar.setValue(100)

        n_valid = getattr(results, 'n_valid', n)
        n_outliers = getattr(results, 'n_outliers', 0)
        n_phys_bad = getattr(results, 'n_physics_invalid', 0)

        parts = [f"Complete — {n} simulations"]
        if n_phys_bad > 0:
            parts.append(f"{n_phys_bad} physics-invalid")
        if n_outliers > 0:
            parts.append(f"{n_outliers} outliers")
        parts.append(f"{n_valid} used for stats")
        self.progress_label.setText(" | ".join(parts))

        self._update_plots(results)
        self._update_statistics(results)

        logger.info(f"Monte Carlo finished: {n} runs "
                    f"({n_phys_bad} physics-invalid, {n_outliers} outliers), "
                    f"apogee={results.apogee_mean:.1f}±{results.apogee_std:.1f}m")

    def _on_cancelled(self):
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress_label.setText("Cancelled")
        logger.info("Monte Carlo analysis cancelled by user")

    # ═════════════════════════════════════════════════════════════════════════
    # Plot Updates
    # ═════════════════════════════════════════════════════════════════════════

    def _update_plots(self, r: MonteCarloResults):
        self._plot_apogee_dist(r)
        self._plot_landing_dispersion(r)
        self._plot_landing_distance(r)
        self._plot_box_plots(r)
        self._plot_tornado(r)

    # ── Tab 1: Apogee Distribution ───────────────────────────────────────────

    def _plot_apogee_dist(self, r: MonteCarloResults):
        ax = self._ax_apogee
        ax.clear()
        _style_ax(ax, "Apogee Distribution", "Apogee (m)", "Frequency")

        # Use pre-filtered valid apogee values from statistics engine
        apogees = np.array(r.apogee_values) if r.apogee_values else np.array([run.apogee for run in r.runs])
        n = len(apogees)
        bins = min(50, max(30, n // 10))

        # Histogram
        counts, bin_edges, patches = ax.hist(
            apogees, bins=bins, color="#58a6ff", alpha=0.7,
            edgecolor="#30363d", linewidth=0.5,
        )

        # Normal distribution fit overlay
        mu, sigma = r.apogee_mean, r.apogee_std
        if sigma > 0:
            x_fit = np.linspace(apogees.min(), apogees.max(), 200)
            pdf = stats.norm.pdf(x_fit, mu, sigma)
            # Scale PDF to match histogram
            bin_width = bin_edges[1] - bin_edges[0]
            ax.plot(x_fit, pdf * n * bin_width, color="#c9d1d9",
                    linewidth=1.5, alpha=0.8, label="Normal fit")

        # Mean line
        ax.axvline(mu, color="#ffffff", linestyle="--", linewidth=1.2,
                   alpha=0.9, label=f"Mean: {mu:.1f} m")

        # ±1σ lines
        ax.axvline(mu - sigma, color="#7ee787", linestyle="--", linewidth=1,
                   alpha=0.7, label=f"±1σ: {sigma:.1f} m")
        ax.axvline(mu + sigma, color="#7ee787", linestyle="--", linewidth=1,
                   alpha=0.7)

        # 95% CI bounds
        ax.axvline(r.apogee_ci_low, color="#f0883e", linestyle="--",
                   linewidth=1, alpha=0.7,
                   label=f"95% CI: {r.apogee_ci_low:.0f}–{r.apogee_ci_high:.0f} m")
        ax.axvline(r.apogee_ci_high, color="#f0883e", linestyle="--",
                   linewidth=1, alpha=0.7)

        ax.legend(facecolor="#161b22", edgecolor="#30363d",
                  labelcolor="#c9d1d9", fontsize=8, loc="upper right")

        self._ax_apogee.figure.tight_layout()
        self._canvas_apogee.draw()

    # ── Tab 2: Landing Dispersion ────────────────────────────────────────────

    def _plot_landing_dispersion(self, r: MonteCarloResults):
        ax = self._ax_landing
        ax.clear()
        _style_ax(ax, "Landing Dispersion", "East (m)", "North (m)")

        # Use pre-filtered valid landing values from statistics engine
        xs = np.array(r.landing_x_values) if r.landing_x_values else np.array([run.landing_x for run in r.runs])
        ys = np.array(r.landing_y_values) if r.landing_y_values else np.array([run.landing_y for run in r.runs])

        # All points in the filtered set are physics-valid
        ax.scatter(xs, ys, c="#58a6ff", s=8,
                   alpha=0.5, label=f"Valid ({len(xs)})", zorder=3)

        # Dispersion ellipses (1σ, 2σ, 3σ)
        cx, cy = float(np.mean(xs)), float(np.mean(ys))
        sx, sy = float(np.std(xs)), float(np.std(ys))
        for n_sigma, alpha_val in [(1, 0.5), (2, 0.35), (3, 0.2)]:
            w = max(sx * 2 * n_sigma, 1.0)
            h = max(sy * 2 * n_sigma, 1.0)
            ellipse = Ellipse(
                (cx, cy), w, h,
                fill=False, edgecolor="#58a6ff", linewidth=1.2,
                alpha=alpha_val, linestyle="--",
                label=f"{n_sigma}σ" if n_sigma == 1 else f"{n_sigma}σ",
            )
            ax.add_patch(ellipse)

        # Launch point crosshair
        ax.axhline(0, color="#484f58", linewidth=0.5, alpha=0.5)
        ax.axvline(0, color="#484f58", linewidth=0.5, alpha=0.5)
        ax.plot(0, 0, "+", color="#ffffff", markersize=12, markeredgewidth=2,
                zorder=5, label="Launch")

        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(facecolor="#161b22", edgecolor="#30363d",
                  labelcolor="#c9d1d9", fontsize=8, loc="upper right")

        self._ax_landing.figure.tight_layout()
        self._canvas_landing.draw()

    # ── Tab 3: Landing Distance ──────────────────────────────────────────────

    def _plot_landing_distance(self, r: MonteCarloResults):
        ax = self._ax_dist
        ax.clear()
        _style_ax(ax, "Landing Distance", "Distance (m)", "Frequency")

        # Use pre-filtered valid landing distances
        dists = np.array(r.landing_distance_values) if r.landing_distance_values else np.array([run.landing_distance for run in r.runs])
        bins = min(50, max(30, len(dists) // 10))

        ax.hist(dists, bins=bins, color="#bc8cff", alpha=0.7,
                edgecolor="#30363d", linewidth=0.5)

        mean_d = float(np.mean(dists))
        max_d = float(np.max(dists))
        ax.axvline(mean_d, color="#ffffff", linestyle="--", linewidth=1.2,
                   alpha=0.9, label=f"Mean: {mean_d:.0f} m")
        ax.axvline(max_d, color="#f85149", linestyle="--", linewidth=1,
                   alpha=0.7, label=f"Max: {max_d:.0f} m")

        ax.legend(facecolor="#161b22", edgecolor="#30363d",
                  labelcolor="#c9d1d9", fontsize=8, loc="upper right")

        self._ax_dist.figure.tight_layout()
        self._canvas_dist.draw()

    # ── Tab 4: Performance Box Plots ─────────────────────────────────────────

    def _plot_box_plots(self, r: MonteCarloResults):
        axes = self._ax_box
        fig = axes[0].figure
        for a in axes:
            a.clear()
            _style_ax(a)

        # Use pre-filtered valid values for box plots (avoids contamination)
        datasets = [
            (r.apogee_values if r.apogee_values else [run.apogee for run in r.runs],
             "Apogee (m)", "#58a6ff"),
            (r.velocity_values if r.velocity_values else [run.max_velocity for run in r.runs],
             "Max Velocity (m/s)", "#7ee787"),
            (r.mach_values if r.mach_values else [run.max_mach for run in r.runs],
             "Max Mach", "#bc8cff"),
            (r.accel_values if r.accel_values else [run.max_acceleration for run in r.runs],
             "Max Accel (m/s²)", "#f0883e"),
            (r.landing_distance_values if r.landing_distance_values else [run.landing_distance for run in r.runs],
             "Landing Dist (m)", "#f778ba"),
        ]

        for ax, (data, label, color) in zip(axes, datasets):
            bp = ax.boxplot(
                data, patch_artist=True, widths=0.6,
                boxprops=dict(facecolor=color, alpha=0.4, edgecolor=color),
                whiskerprops=dict(color=color, linewidth=1.2),
                capprops=dict(color=color, linewidth=1.2),
                medianprops=dict(color="#ffffff", linewidth=1.5),
                flierprops=dict(marker="o", markerfacecolor=color,
                                markeredgecolor=color, markersize=3, alpha=0.5),
            )
            ax.set_title(label, color="#8b949e", fontsize=9, fontweight="bold")
            ax.set_xticks([])

        fig.tight_layout()
        self._canvas_box.draw()

    # ═════════════════════════════════════════════════════════════════════════
    # Statistics Update
    # ═════════════════════════════════════════════════════════════════════════

    def _update_statistics(self, r: MonteCarloResults):
        # Scenarios (clickable buttons)
        self.btn_best.setText(
            f"Best Case: Run #{r.best_run_index} — {r.best_apogee:.1f} m"
        )
        try: self.btn_best.clicked.disconnect()
        except TypeError: pass
        self.btn_best.clicked.connect(lambda _, idx=r.best_run_index: self._show_scenario_detail(idx))
        self.btn_worst.setText(
            f"Worst Case: Run #{r.worst_run_index} — {r.worst_apogee:.1f} m"
        )
        try: self.btn_worst.clicked.disconnect()
        except TypeError: pass
        self.btn_worst.clicked.connect(lambda _, idx=r.worst_run_index: self._show_scenario_detail(idx))

        # Apogee
        self.lbl_ap_mean.setText(f"{r.apogee_mean:.1f} m")
        self.lbl_ap_std.setText(f"{r.apogee_std:.1f} m")
        self.lbl_ap_min.setText(f"{r.apogee_min:.1f} m")
        self.lbl_ap_max.setText(f"{r.apogee_max:.1f} m")
        self.lbl_ap_median.setText(f"{r.apogee_median:.1f} m")
        self.lbl_ap_mode.setText(f"{r.apogee_mode:.1f} m")
        self.lbl_ap_ci.setText(
            f"{r.apogee_ci_low:.0f} – {r.apogee_ci_high:.0f} m"
        )
        # Distribution shape indicators
        skew_txt = f"{r.apogee_skewness:+.2f}"
        if abs(r.apogee_skewness) > 1.0:
            skew_txt += " highly skewed"
        self.lbl_ap_skew.setText(skew_txt)

        kurt_txt = f"{r.apogee_kurtosis:+.2f}"
        if r.apogee_kurtosis > 3.0:
            kurt_txt += " heavy-tailed"
        self.lbl_ap_kurt.setText(kurt_txt)

        # Performance
        self.lbl_vel_mean.setText(f"{r.max_velocity_mean:.1f} m/s")
        self.lbl_mach_mean.setText(f"{r.max_mach_mean:.3f}")
        self.lbl_acc_mean.setText(f"{r.max_accel_mean:.1f} m/s²")
        self.lbl_rev_mean.setText(f"{r.rail_exit_velocity_mean:.1f} m/s")

        # Landing
        self.lbl_ld_mean.setText(f"{r.landing_dist_mean:.1f} m")
        self.lbl_ld_max.setText(f"{r.landing_dist_max:.1f} m")
        self.lbl_ld_std.setText(f"{r.landing_dist_std:.1f} m")

        # Mission Assessment (color-coded)
        self._set_rate_label(self.lbl_success, r.success_rate, " %")
        self.lbl_failure.setText(f"{r.failure_rate:.1f} %")
        self._set_rate_label(self.lbl_p_target, r.p_target_alt, " %")
        self._set_rate_label(self.lbl_p_safe, r.p_safe_recovery, " %")
        self._set_rate_label(self.lbl_mission, r.mission_success, " %")

        # Failure Breakdown
        self._update_failure_breakdown(r)

        # Reliability Analysis
        self._set_rate_label(self.lbl_r_apogee, r.p_apogee_above_target * 100, " %")
        self._set_rate_label(self.lbl_r_mach, r.p_mach_below_limit * 100, " %")
        self._set_rate_label(self.lbl_r_accel, r.p_accel_below_limit * 100, " %")
        self._set_rate_label(self.lbl_r_stab, r.p_stability_above_limit * 100, " %")
        self._set_rate_label(self.lbl_r_rail, r.p_rail_exit_above_min * 100, " %")

        beta = r.reliability_index_beta
        beta_color = "#7ee787" if beta > 3 else ("#d29922" if beta > 1 else "#f85149")
        self.lbl_r_beta.setText(f"{beta:+.2f}")
        self.lbl_r_beta.setStyleSheet(
            f"color:{beta_color};font-family:'Cascadia Code',monospace;font-size:13px;"
            f"font-weight:600;padding:2px 6px;background:#161b22;border-radius:4px;"
        )

        ci_lo, ci_hi = r.reliability_confidence_interval
        self.lbl_r_ci.setText(f"{ci_lo*100:.1f}% – {ci_hi*100:.1f}%")

    def _set_rate_label(self, label: QLabel, value: float, suffix: str):
        """Set label text and colour based on rate thresholds."""
        if value >= 90:
            color = "#7ee787"
        elif value >= 70:
            color = "#d29922"
        else:
            color = "#f85149"
        label.setText(f"{value:.1f}{suffix}")
        label.setStyleSheet(
            f"color:{color}; font-family:'Cascadia Code',monospace; font-size:13px;"
            f"font-weight:600; padding:2px 6px; background:#161b22; border-radius:4px;"
        )

    def _update_failure_breakdown(self, r: MonteCarloResults):
        """Rebuild the failure breakdown list."""
        # Clear existing items
        while self._fail_layout.count():
            item = self._fail_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not r.failure_breakdown:
            lbl = QLabel("✓ No failures")
            lbl.setStyleSheet("color:#7ee787; font-size:11px; font-weight:600;")
            self._fail_layout.addWidget(lbl)
            return

        n = len(r.runs)
        for reason, count in sorted(r.failure_breakdown.items(),
                                    key=lambda x: -x[1]):
            pct = 100.0 * count / n
            lbl = QLabel(f"• {reason}: {count} ({pct:.1f}%)")
            lbl.setStyleSheet(
                "color:#f85149; font-size:11px; font-weight:500; padding:1px 0;"
            )
            lbl.setWordWrap(True)
            self._fail_layout.addWidget(lbl)

    # ═════════════════════════════════════════════════════════════════════════
    # Scenario Inspector (Issue 6)
    # ═════════════════════════════════════════════════════════════════════════

    def _show_scenario_detail(self, run_index: int):
        """Show a detail dialog for a specific MC run."""
        if self._results is None or run_index >= len(self._results.runs):
            return

        run = self._results.runs[run_index]
        params = (
            self._results.perturbed_params[run_index]
            if run_index < len(self._results.perturbed_params)
            else {}
        )

        # Build rich text content
        lines = []
        lines.append(f"<h3 style='color:#58a6ff;'>Run #{run_index} Detail</h3>")

        # Inputs
        lines.append("<h4 style='color:#7ee787;'>INPUTS</h4>")
        lines.append("<table style='font-family:monospace;color:#c9d1d9;'>")
        param_labels = {
            "impulse_scale": ("Motor Impulse Scale", ""),
            "dry_mass": ("Dry Mass", "kg"),
            "cd": ("Drag Coefficient", ""),
            "wind_speed": ("Wind Speed", "m/s"),
            "wind_direction": ("Wind Direction", "\u00b0"),
            "launch_angle": ("Launch Angle", "\u00b0"),
            "cg": ("CG Position", "m"),
        }
        for key, (label, unit) in param_labels.items():
            val = params.get(key, "N/A")
            if isinstance(val, float):
                val = f"{val:.4f}"
            lines.append(f"<tr><td style='padding:2px 8px;'>{label}:</td>"
                        f"<td style='padding:2px 8px;color:#58a6ff;'>{val} {unit}</td></tr>")
        lines.append("</table>")

        # Outputs
        lines.append("<h4 style='color:#7ee787;'>OUTPUTS</h4>")
        lines.append("<table style='font-family:monospace;color:#c9d1d9;'>")
        outputs = [
            ("Apogee", f"{run.apogee:.1f} m"),
            ("Max Velocity", f"{run.max_velocity:.1f} m/s"),
            ("Max Mach", f"{run.max_mach:.3f}"),
            ("Max Acceleration", f"{run.max_acceleration:.1f} m/s\u00b2 ({run.max_acceleration/9.81:.1f}G)"),
            ("Landing Distance", f"{run.landing_distance:.1f} m"),
            ("Landing X", f"{run.landing_x:+.1f} m"),
            ("Landing Y", f"{run.landing_y:+.1f} m"),
            ("Rail Exit Velocity", f"{run.rail_exit_velocity:.1f} m/s"),
            ("Min Stability", f"{run.min_stability_margin:.2f} cal"),
            ("Flight Time", f"{run.flight_time:.1f} s"),
            ("Final Phase", run.final_phase),
        ]
        for label, val in outputs:
            lines.append(f"<tr><td style='padding:2px 8px;'>{label}:</td>"
                        f"<td style='padding:2px 8px;color:#58a6ff;'>{val}</td></tr>")
        lines.append("</table>")

        # Failures
        if run.failure_reasons:
            lines.append("<h4 style='color:#f85149;'>FAILURES</h4>")
            lines.append("<ul style='color:#f85149;'>")
            for fr in run.failure_reasons:
                lines.append(f"<li>{fr}</li>")
            lines.append("</ul>")
        else:
            lines.append("<h4 style='color:#7ee787;'>\u2713 No failures</h4>")

        # Create dialog
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Run #{run_index} — Scenario Detail")
        dlg.setMinimumSize(450, 500)
        dlg.setStyleSheet("background:#0d1117; color:#c9d1d9;")

        layout = QVBoxLayout(dlg)
        text = QTextEdit()
        text.setReadOnly(True)
        text.setHtml("".join(lines))
        text.setStyleSheet(
            "QTextEdit{background:#161b22;color:#c9d1d9;border:1px solid #30363d;"
            "border-radius:6px;padding:8px;font-size:12px;}"
        )
        layout.addWidget(text)

        btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btn_box.setStyleSheet("QPushButton{background:#21262d;color:#c9d1d9;border:1px solid #30363d;"
                              "border-radius:4px;padding:6px 16px;}"
                              "QPushButton:hover{background:#30363d;}")
        btn_box.rejected.connect(dlg.close)
        layout.addWidget(btn_box)

        dlg.exec()

    # ═════════════════════════════════════════════════════════════════════════
    # Sensitivity / Tornado Chart
    # ═════════════════════════════════════════════════════════════════════════

    def _plot_tornado(self, r: MonteCarloResults):
        """Draw tornado chart showing parameter sensitivity to apogee."""
        ax = self._ax_tornado
        ax.clear()
        _style_ax(ax, "Parameter Sensitivity (Pearson r)", "Correlation Coefficient", "")

        correlations = getattr(r, 'sensitivity_correlations', {})
        if not correlations:
            ax.text(
                0.5, 0.5, "Insufficient data for sensitivity analysis",
                transform=ax.transAxes, ha="center", va="center",
                color="#484f58", fontsize=12,
            )
            self._ax_tornado.figure.tight_layout()
            self._canvas_tornado.draw()
            return

        # Sort by absolute correlation
        sorted_items = sorted(
            correlations.items(),
            key=lambda x: abs(x[1]["pearson_r"]),
        )
        labels = [item[0] for item in sorted_items]
        pearson_vals = [item[1]["pearson_r"] for item in sorted_items]
        colors = ["#58a6ff" if v >= 0 else "#f85149" for v in pearson_vals]

        bars = ax.barh(labels, pearson_vals, color=colors, height=0.55,
                       alpha=0.85, edgecolor="#30363d", linewidth=0.5)

        # Value labels on bars
        for bar, val in zip(bars, pearson_vals):
            x_pos = val + 0.02 if val >= 0 else val - 0.02
            ha = "left" if val >= 0 else "right"
            ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                    f"{val:+.3f}", va="center", ha=ha,
                    color="#c9d1d9", fontsize=9, fontweight="bold")

        ax.axvline(0, color="#484f58", linewidth=0.8)
        ax.set_xlim(-1.05, 1.05)
        ax.tick_params(axis="y", labelsize=10, colors="#c9d1d9")

        # Legend explaining colors
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#58a6ff", alpha=0.85, label="Positive (↑ param → ↑ apogee)"),
            Patch(facecolor="#f85149", alpha=0.85, label="Negative (↑ param → ↓ apogee)"),
        ]
        ax.legend(handles=legend_elements, loc="lower right",
                  facecolor="#161b22", edgecolor="#30363d",
                  labelcolor="#c9d1d9", fontsize=8)

        self._ax_tornado.figure.tight_layout()
        self._canvas_tornado.draw()

    # ═════════════════════════════════════════════════════════════════════════
    # Export
    # ═════════════════════════════════════════════════════════════════════════

    def _on_export_pdf(self):
        """Monte Carlo PDF: statistics + distribution / landing / box / tornado plots."""
        if self._results is None:
            return
        from PyQt6.QtWidgets import QFileDialog, QMessageBox
        from pathlib import Path
        from ui.pdf_report import save_report
        path, _ = QFileDialog.getSaveFileName(
            self, "Export PDF Report", "monte_carlo_report.pdf", "PDF Files (*.pdf)")
        if not path:
            return
        r = self._results
        ci = getattr(r, "apogee_ci_95", (0.0, 0.0))
        n = len(getattr(r, "all_runs", []) or getattr(r, "runs", []))
        kv = [
            ("Runs", n),
            ("Apogee mean", f"{r.apogee_mean:.1f} m"),
            ("Apogee std", f"{r.apogee_std:.1f} m"),
            ("Apogee min / max", f"{r.apogee_min:.1f} / {r.apogee_max:.1f} m"),
            ("Apogee median", f"{r.apogee_median:.1f} m"),
            ("Apogee 95% CI", f"{ci[0]:.0f} – {ci[1]:.0f} m"),
            ("Max velocity mean", f"{r.velocity_mean:.1f} m/s"),
            ("Max Mach mean", f"{r.mach_mean:.3f}"),
            ("Peak accel mean", f"{r.accel_mean:.1f} m/s²"),
            ("Landing dist mean", f"{r.landing_distance_mean:.0f} m"),
            ("Success rate", f"{r.success_percentage:.1f} %"),
            ("P(target altitude)", f"{r.prob_target_altitude * 100:.1f} %"),
            ("P(mission success)", f"{r.mission_success_probability * 100:.1f} %"),
        ]
        figs = [getattr(self, a).figure for a in
                ("_ax_apogee", "_ax_landing", "_ax_dist", "_ax_tornado")
                if getattr(self, a, None) is not None]
        # box-plot axes are a row of subplots; grab the shared figure once
        if getattr(self, "_ax_box", None) is not None:
            try:
                figs.append(self._ax_box[0].figure)
            except Exception:
                pass
        ok = save_report(path, "Monte Carlo Report",
                         f"{n} runs · {r.success_percentage:.0f}% success", kv, figures=figs)
        if ok:
            self.status_label.setText(f"PDF report saved: {Path(path).name}") if hasattr(self, "status_label") else None
        else:
            QMessageBox.warning(self, "Export Error", "Could not write the PDF report.")

    def _on_export(self):
        if self._results is None or not self._results.runs:
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Monte Carlo Results",
            str(Path.home() / "Documents" / "K2_monte_carlo.csv"),
            "CSV Files (*.csv)",
        )
        if not path:
            return

        header = [
            "run", "apogee", "max_velocity", "max_mach", "max_acceleration",
            "landing_x", "landing_y", "landing_distance", "flight_time",
            "rail_exit_velocity", "min_stability_margin", "success",
            "failure_reasons",
            "wind_speed", "wind_direction", "dry_mass", "cd",
            "launch_angle", "cg", "motor_impulse_factor",
        ]

        try:
            with open(path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(header)
                for i, run in enumerate(self._results.runs):
                    params = (
                        self._results.perturbed_params[i]
                        if i < len(self._results.perturbed_params)
                        else {}
                    )
                    writer.writerow([
                        i,
                        f"{run.apogee:.2f}",
                        f"{run.max_velocity:.2f}",
                        f"{run.max_mach:.4f}",
                        f"{run.max_acceleration:.2f}",
                        f"{run.landing_x:.2f}",
                        f"{run.landing_y:.2f}",
                        f"{run.landing_distance:.2f}",
                        f"{run.flight_time:.2f}",
                        f"{run.rail_exit_velocity:.2f}",
                        f"{run.min_stability_margin:.2f}",
                        "1" if run.success else "0",
                        "; ".join(run.failure_reasons),
                        f"{params.get('wind_speed', '')}",
                        f"{params.get('wind_direction', '')}",
                        f"{params.get('dry_mass', '')}",
                        f"{params.get('cd', '')}",
                        f"{params.get('launch_angle', '')}",
                        f"{params.get('cg', '')}",
                        f"{params.get('impulse_scale', '')}",
                    ])
            self.engine.log_message.emit(f"Monte Carlo results exported: {path}")
            logger.info(f"Exported {len(self._results.runs)} runs to {path}")
        except Exception as exc:
            logger.error(f"Export failed: {exc}")
            self.engine.log_message.emit(f"Export failed: {exc}")

    # ═════════════════════════════════════════════════════════════════════════
    # State Change
    # ═════════════════════════════════════════════════════════════════════════

    def _on_state_changed(self, state):
        """Optionally update target apogee from latest simulation."""
        pass
