"""
K2 Aerospace — Optimization Workspace
=======================================
Professional MDO (Multidisciplinary Design Optimization) interface with:
  - 8 analysis sub-tabs
  - Interactive design-space exploration
  - Pareto front visualisation
  - DOE / response-surface contour maps
  - Sobol sensitivity bar charts
  - Trade-study radar plots
  - Robust / mission-driven optimisation controls
  - CSV / JSON export

Layout follows the 3-panel pattern established by MonteCarloWorkspace.
"""

from __future__ import annotations

import os
import csv
import json
import copy
import math
import time
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout, QLabel,
    QPushButton, QSplitter, QFrame, QScrollArea, QSpinBox,
    QDoubleSpinBox, QProgressBar, QTabWidget, QFileDialog, QMessageBox,
    QCheckBox, QComboBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QRadioButton, QButtonGroup, QSizePolicy,
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread

from ui.icons import icon as app_icon

logger = logging.getLogger("K2.OptimizationWS")

# ── Stylesheet constants (matches Monte Carlo workspace) ────────────────────

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

_BTN_SUCCESS = """
QPushButton { background:#238636; color:#fff; font-weight:600; font-size:11px;
  border:none; border-radius:6px; padding:7px 12px; }
QPushButton:hover { background:#2ea043; }
QPushButton:disabled { background:#21262d; color:#484f58; }
"""

_VAL = ("color:#e6edf3; font-family:'Cascadia Code',monospace; font-size:13px;"
        "font-weight:600; padding:2px 6px; background:#161b22; border-radius:4px;")

_CHK = """
QCheckBox { color:#c9d1d9; spacing:6px; font-size:11px; }
QCheckBox::indicator { width:14px; height:14px; border:1px solid #30363d; border-radius:3px;
  background:#0d1117; }
QCheckBox::indicator:checked { background:#1f6feb; border-color:#1f6feb; }
"""

_COMBO_SMALL = """
QComboBox { font-size:11px; padding:3px 6px; min-width:90px; }
"""


def _vl(text: str = "—") -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(_VAL)
    return lbl


# ── Dark-themed matplotlib helpers ───────────────────────────────────────────

def _style_ax(ax, title="", xlabel="", ylabel=""):
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


def _dark_figure(rows=1, cols=1, figsize=(8, 5)):
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


class _DOEWorker(QThread):
    """Runs the DOE sample sweep off the UI thread.

    Builds the design matrix and evaluates each point with a full batch
    simulation, then hands ``(dm, responses)`` back for plotting on the main
    thread. Running this inline froze the GUI for the whole sweep.
    """
    finished_ok = pyqtSignal(object, object)   # dm (ndarray), responses (ndarray)
    failed = pyqtSignal(str)

    def __init__(self, base_config, enabled_vars, method, n_samples, swept_fn):
        super().__init__()
        self._base = base_config
        self._ev = enabled_vars
        self._method = method
        self._n = n_samples
        self._swept = swept_fn

    def run(self):
        try:
            from core.batch_simulation import run_batch_simulation
            n_vars = len(self._ev)
            if self._method == "Full Factorial":
                levels = max(2, int(round(self._n ** (1.0 / n_vars))))
                grids = [np.linspace(0, 1, levels) for _ in range(n_vars)]
                dm = np.array(np.meshgrid(*grids)).T.reshape(-1, n_vars)[:self._n]
            else:  # Latin Hypercube / Taguchi
                from scipy.stats.qmc import LatinHypercube
                n = min(self._n, 27) if self._method == "Taguchi" else self._n
                dm = LatinHypercube(d=n_vars, seed=42).random(n=n)

            responses = np.zeros(len(dm))
            for i in range(len(dm)):
                values = [vmin + dm[i, j] * (vmax - vmin)
                          for j, (key, vmin, vmax) in enumerate(self._ev)]
                cfg = self._swept(self._base, self._ev, values)
                try:
                    responses[i] = run_batch_simulation(cfg, seed=42 + i).apogee
                except Exception:
                    responses[i] = 0.0
            self.finished_ok.emit(dm, responses)
        except Exception as e:
            self.failed.emit(str(e))


class _SensitivityWorker(QThread):
    """Runs sensitivity analysis (LHS sweep + method-specific sims) off the UI
    thread. Returns a payload dict the workspace plots on the main thread.
    Morris screening fires extra paired sims, so all simulation must live
    here — not in the render path."""
    finished_ok = pyqtSignal(object)   # payload dict
    failed = pyqtSignal(str)

    def __init__(self, base_config, enabled_vars, method, n_samples, swept_fn):
        super().__init__()
        self._base = base_config
        self._ev = enabled_vars
        self._method = method
        self._n = n_samples
        self._swept = swept_fn

    def run(self):
        try:
            from core.batch_simulation import run_batch_simulation
            from scipy.stats.qmc import LatinHypercube
            ev = self._ev
            n_vars = len(ev)
            n = self._n
            dm = LatinHypercube(d=n_vars, seed=42).random(n=n)
            X = np.zeros((n, n_vars))
            y = np.zeros(n)
            for i in range(n):
                vals = [vmin + dm[i, j] * (vmax - vmin)
                        for j, (key, vmin, vmax) in enumerate(ev)]
                for j, v in enumerate(vals):
                    X[i, j] = v
                cfg = self._swept(self._base, ev, vals)
                try:
                    y[i] = run_batch_simulation(cfg, seed=42 + i).apogee
                except Exception:
                    y[i] = 0.0

            out = {"method": self._method, "X": X, "y": y}
            if self._method == "Sobol Indices":
                from scipy.stats import pearsonr
                tv = np.var(y)
                s1, st = [], []
                for j in range(n_vars):
                    bins = np.linspace(X[:, j].min(), X[:, j].max(), 11)
                    bidx = np.digitize(X[:, j], bins)
                    cms = [np.mean(y[bidx == b]) for b in range(1, len(bins))
                           if np.sum(bidx == b) > 0]
                    s1v = (np.var(cms) / tv) if (cms and tv > 0) else 0
                    s1.append(min(s1v, 1.0))
                    try:
                        r, _ = pearsonr(X[:, j], y)
                        stv = r ** 2
                    except Exception:
                        stv = s1v
                    st.append(min(max(stv, s1v), 1.0))
                out["s1"], out["st"] = s1, st
            elif self._method == "PRCC":
                from scipy.stats import spearmanr
                prcc = []
                for j in range(n_vars):
                    try:
                        r, _ = spearmanr(X[:, j], y)
                        prcc.append(r)
                    except Exception:
                        prcc.append(0)
                out["prcc"] = prcc
            else:  # Morris Screening — extra paired elementary-effect sims
                mu_star, sigma_vals = [], []
                for j in range(n_vars):
                    key, vmin, vmax = ev[j]
                    delta = 0.1
                    effects = []
                    for i in range(min(n - 1, 50)):
                        v1 = X[i, j]
                        v2 = min(v1 + delta * (vmax - vmin), vmax)
                        base_vals = [X[i, k] for k in range(n_vars)]
                        cfg1 = self._swept(self._base, ev, base_vals)
                        pert = list(base_vals)
                        pert[j] = v2
                        cfg2 = self._swept(self._base, ev, pert)
                        try:
                            r1 = run_batch_simulation(cfg1, seed=1000 + i).apogee
                            r2 = run_batch_simulation(cfg2, seed=1000 + i).apogee
                            dx = v2 - v1
                            if abs(dx) > 1e-12:
                                effects.append((r2 - r1) / dx)
                        except Exception:
                            pass
                    if effects:
                        mu_star.append(np.mean(np.abs(effects)))
                        sigma_vals.append(np.std(effects))
                    else:
                        mu_star.append(0)
                        sigma_vals.append(0)
                out["mu_star"], out["sigma"] = mu_star, sigma_vals
            self.finished_ok.emit(out)
        except Exception as e:
            self.failed.emit(str(e))


# ══════════════════════════════════════════════════════════════════════════════
#  WORKSPACE
# ══════════════════════════════════════════════════════════════════════════════

class OptimizationWorkspace(QWidget):
    """Aerospace-grade MDO workspace with 8 analysis sub-tabs."""

    design_loaded = pyqtSignal(dict)  # Emitted when user clicks a design point

    def __init__(self, engine, sim_engine=None, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.sim_engine = sim_engine
        self._opt_engine = None
        self._result = None
        self._var_widgets: dict = {}  # name -> (checkbox, min_spin, max_spin)
        self._obj_widgets: dict = {}  # name -> (checkbox, weight_spin, mode_combo)
        self._con_widgets: dict = {}  # name -> (checkbox, limit_spin)
        self._selected_design = None
        self._setup_ui()
        self.engine.state_changed.connect(self._on_state_changed)

    # ─── UI Construction ─────────────────────────────────────────────────────

    def _setup_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_left())
        splitter.addWidget(self._build_center())
        splitter.addWidget(self._build_right())
        splitter.setSizes([400, 720, 380])
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter)

    # ═════════════════════════════════════════════════════════════════════════
    #  LEFT PANEL — Configuration
    # ═════════════════════════════════════════════════════════════════════════

    def _build_left(self) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumWidth(420)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(12, 14, 12, 14)
        lay.setSpacing(10)

        # Title
        title = QLabel("DESIGN OPTIMIZATION")
        title.setStyleSheet(
            "color:#58a6ff; font-size:16px; font-weight:700; "
            "letter-spacing:2px; padding:2px 0 6px 0;"
        )
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(title)

        lay.addWidget(self._build_algorithm_group())
        lay.addWidget(self._build_mode_group())
        lay.addWidget(self._build_mc_group())
        lay.addWidget(self._build_surrogate_group())
        lay.addWidget(self._build_variables_group())
        lay.addWidget(self._build_objectives_group())
        lay.addWidget(self._build_constraints_group())
        lay.addWidget(self._build_actions_group())

        lay.addStretch()
        scroll.setWidget(inner)
        return scroll

    # ── Algorithm Selection ──────────────────────────────────────────────────

    def _build_algorithm_group(self) -> QGroupBox:
        g = QGroupBox("Algorithm")
        g.setStyleSheet(_GRP)
        f = QFormLayout()
        f.setSpacing(6)

        self.combo_algorithm = QComboBox()
        self.combo_algorithm.addItems([
            "Genetic Algorithm", "NSGA-II (Multi-Obj)",
            "Differential Evolution", "Particle Swarm",
        ])
        f.addRow("Method:", self.combo_algorithm)

        self.spin_pop = QSpinBox()
        self.spin_pop.setRange(10, 500)
        self.spin_pop.setValue(50)
        f.addRow("Population:", self.spin_pop)

        self.spin_gens = QSpinBox()
        self.spin_gens.setRange(5, 1000)
        self.spin_gens.setValue(100)
        f.addRow("Generations:", self.spin_gens)

        self.spin_mutation = QDoubleSpinBox()
        self.spin_mutation.setRange(0.01, 0.5)
        self.spin_mutation.setSingleStep(0.01)
        self.spin_mutation.setValue(0.1)
        f.addRow("Mutation Rate:", self.spin_mutation)

        self.spin_crossover = QDoubleSpinBox()
        self.spin_crossover.setRange(0.3, 1.0)
        self.spin_crossover.setSingleStep(0.05)
        self.spin_crossover.setValue(0.8)
        f.addRow("Crossover Rate:", self.spin_crossover)

        g.setLayout(f)
        return g

    # ── Optimization Mode ────────────────────────────────────────────────────

    def _build_mode_group(self) -> QGroupBox:
        g = QGroupBox("Optimization Mode")
        g.setStyleSheet(_GRP)
        vl = QVBoxLayout()
        vl.setSpacing(4)

        self._mode_group = QButtonGroup(self)
        modes = [
            ("Standard (maximize mean)", "standard"),
            ("Robust (minimize variance)", "robust"),
            ("Mission Target", "mission"),
        ]
        for i, (label, val) in enumerate(modes):
            rb = QRadioButton(label)
            rb.setStyleSheet("color:#c9d1d9; font-size:11px;")
            rb.setProperty("mode_value", val)
            if i == 0:
                rb.setChecked(True)
            self._mode_group.addButton(rb, i)
            vl.addWidget(rb)

        g.setLayout(vl)
        return g

    # ── Monte Carlo Settings ─────────────────────────────────────────────────

    def _build_mc_group(self) -> QGroupBox:
        g = QGroupBox("Monte Carlo Settings")
        g.setStyleSheet(_GRP)
        f = QFormLayout()
        f.setSpacing(6)

        self.spin_mc_per = QSpinBox()
        self.spin_mc_per.setRange(1, 50)
        self.spin_mc_per.setValue(5)
        self.spin_mc_per.setToolTip("MC simulations per candidate during search")
        f.addRow("Sims/Candidate:", self.spin_mc_per)

        self.spin_mc_val = QSpinBox()
        self.spin_mc_val.setRange(10, 500)
        self.spin_mc_val.setValue(50)
        self.spin_mc_val.setToolTip("MC simulations for final validation of top designs")
        f.addRow("Validation Sims:", self.spin_mc_val)

        self.spin_target_apogee = QDoubleSpinBox()
        self.spin_target_apogee.setRange(0, 100000)
        self.spin_target_apogee.setSuffix(" m")
        self.spin_target_apogee.setDecimals(0)
        self.spin_target_apogee.setValue(0)
        self.spin_target_apogee.setToolTip("Target apogee for mission mode (0 = auto)")
        f.addRow("Target Apogee:", self.spin_target_apogee)

        self.chk_parallel = QCheckBox("Parallel Evaluation")
        self.chk_parallel.setStyleSheet(_CHK)
        self.chk_parallel.setChecked(True)
        self.chk_parallel.setToolTip(
            "Evaluate the population across multiple worker processes.\n"
            "Disable to run serially (easier debugging).")
        f.addRow("", self.chk_parallel)

        self.spin_workers = QSpinBox()
        self.spin_workers.setRange(0, max(1, os.cpu_count() or 1))
        self.spin_workers.setValue(0)
        self.spin_workers.setSpecialValueText("Auto")
        self.spin_workers.setToolTip(
            f"Worker processes (0 = auto: cpu_count-1). "
            f"This machine has {os.cpu_count() or 1} CPUs.")
        f.addRow("Workers:", self.spin_workers)

        self.chk_parallel.toggled.connect(self.spin_workers.setEnabled)

        g.setLayout(f)
        return g

    # ── Surrogate Model ──────────────────────────────────────────────────────

    def _build_surrogate_group(self) -> QGroupBox:
        g = QGroupBox("Surrogate Model")
        g.setStyleSheet(_GRP)
        vl = QVBoxLayout()
        vl.setSpacing(6)

        self.chk_surrogate = QCheckBox("Enable Surrogate Acceleration")
        self.chk_surrogate.setStyleSheet(_CHK)
        # NOTE: surrogate acceleration is not yet wired into the optimization
        # engine (config flag is read but no algorithm consumes it). Disabled
        # to avoid implying a speedup that does not happen. Re-enable once
        # _run_* algorithms use core.surrogate_model.
        self.chk_surrogate.setEnabled(False)
        self.chk_surrogate.setChecked(False)
        self.chk_surrogate.setToolTip("Not yet implemented — full simulation is always used.")
        vl.addWidget(self.chk_surrogate)

        f = QFormLayout()
        f.setSpacing(6)
        self.combo_surrogate = QComboBox()
        self.combo_surrogate.addItems([
            "Random Forest", "Gradient Boosting", "Neural Network",
            "Kriging (GP)", "RBF Interpolation", "Polynomial",
        ])
        self.combo_surrogate.setEnabled(False)
        f.addRow("Model:", self.combo_surrogate)

        self.spin_surrogate_samples = QSpinBox()
        self.spin_surrogate_samples.setRange(50, 2000)
        self.spin_surrogate_samples.setValue(200)
        self.spin_surrogate_samples.setEnabled(False)
        f.addRow("Initial Samples:", self.spin_surrogate_samples)

        self.chk_active_learning = QCheckBox("Active Learning")
        self.chk_active_learning.setStyleSheet(_CHK)
        self.chk_active_learning.setChecked(True)
        self.chk_active_learning.setEnabled(False)
        f.addRow("", self.chk_active_learning)

        vl.addLayout(f)
        g.setLayout(vl)
        return g

    # ── Design Variables ─────────────────────────────────────────────────────

    def _build_variables_group(self) -> QGroupBox:
        g = QGroupBox("Design Variables")
        g.setStyleSheet(_GRP)
        vl = QVBoxLayout()
        vl.setSpacing(4)

        # Quick actions row
        hl = QHBoxLayout()
        btn_all = QPushButton("Select All")
        btn_all.setStyleSheet(_BTN_S)
        btn_all.setFixedHeight(24)
        btn_all.clicked.connect(lambda: self._toggle_all_vars(True))
        btn_none = QPushButton("Deselect All")
        btn_none.setStyleSheet(_BTN_S)
        btn_none.setFixedHeight(24)
        btn_none.clicked.connect(lambda: self._toggle_all_vars(False))
        hl.addWidget(btn_all)
        hl.addWidget(btn_none)
        vl.addLayout(hl)

        # Variable categories
        categories = {
            "Geometry": [
                ("diameter", "Diameter", 0.03, 0.30, "m"),
                ("length", "Body Length", 0.3, 3.0, "m"),
                ("nose_length", "Nose Length", 0.05, 0.8, "m"),
                ("fin_span", "Fin Span", 0.02, 0.25, "m"),
                ("fin_root_chord", "Fin Root Chord", 0.03, 0.40, "m"),
                ("fin_tip_chord", "Fin Tip Chord", 0.01, 0.20, "m"),
                ("fin_sweep_angle", "Fin Sweep", 0, 60, "°"),
                ("fin_thickness", "Fin Thickness", 0.001, 0.01, "m"),
                ("fin_count", "Num Fins", 3, 6, ""),
            ],
            "Mass": [
                ("dry_mass", "Dry Mass", 0.1, 20.0, "kg"),
            ],
            "Propulsion": [
                ("motor_total_impulse", "Total Impulse", 5, 5000, "Ns"),
                ("motor_burn_time", "Burn Time", 0.3, 10.0, "s"),
                ("propellant_mass", "Propellant Mass", 0.01, 5.0, "kg"),
            ],
            "Recovery": [
                ("drogue_cd_area", "Drogue CdA", 0.05, 2.0, "m²"),
                ("main_cd_area", "Main CdA", 0.5, 10.0, "m²"),
                ("main_deploy_altitude", "Main Deploy Alt", 100, 600, "m"),
            ],
            "Aerodynamics": [
                ("cd", "Cd Correction", 0.1, 1.5, ""),
            ],
        }

        self._var_widgets = {}
        for cat_name, variables in categories.items():
            cat_lbl = QLabel(f"  ▸ {cat_name}")
            cat_lbl.setStyleSheet(
                "color:#58a6ff; font-size:11px; font-weight:700; padding:4px 0 2px 0;"
            )
            vl.addWidget(cat_lbl)

            for var_key, display, vmin, vmax, unit in variables:
                row = QHBoxLayout()
                row.setSpacing(4)

                chk = QCheckBox()
                chk.setStyleSheet(_CHK)
                chk.setFixedWidth(18)
                row.addWidget(chk)

                lbl = QLabel(display)
                lbl.setStyleSheet("color:#c9d1d9; font-size:10px;")
                lbl.setFixedWidth(90)
                row.addWidget(lbl)

                spin_min = QDoubleSpinBox()
                spin_min.setRange(vmin * 0.1, vmax * 5)
                spin_min.setValue(vmin)
                spin_min.setDecimals(3 if vmax < 1 else 1)
                spin_min.setFixedWidth(70)
                spin_min.setToolTip(f"Min {unit}")
                row.addWidget(spin_min)

                dash = QLabel("–")
                dash.setStyleSheet("color:#484f58;")
                dash.setFixedWidth(8)
                row.addWidget(dash)

                spin_max = QDoubleSpinBox()
                spin_max.setRange(vmin * 0.1, vmax * 5)
                spin_max.setValue(vmax)
                spin_max.setDecimals(3 if vmax < 1 else 1)
                spin_max.setFixedWidth(70)
                spin_max.setToolTip(f"Max {unit}")
                row.addWidget(spin_max)

                if unit:
                    u = QLabel(unit)
                    u.setStyleSheet("color:#484f58; font-size:9px;")
                    u.setFixedWidth(22)
                    row.addWidget(u)

                row.addStretch()
                vl.addLayout(row)
                self._var_widgets[var_key] = (chk, spin_min, spin_max, cat_name)

        g.setLayout(vl)
        return g

    # ── Objectives ───────────────────────────────────────────────────────────

    def _build_objectives_group(self) -> QGroupBox:
        g = QGroupBox("Objectives")
        g.setStyleSheet(_GRP)
        vl = QVBoxLayout()
        vl.setSpacing(4)

        objectives_defs = [
            ("  ▸ Performance", [
                ("max_apogee", "Max Apogee", "maximize"),
                ("max_rail_exit_velocity", "Max Rail Exit Vel", "maximize"),
                ("max_velocity", "Max Velocity", "maximize"),
                ("max_payload_fraction", "Max Payload Fraction", "maximize"),
            ]),
            ("  ▸ Safety", [
                ("max_stability_margin", "Max Stability Margin", "maximize"),
                ("min_landing_distance", "Min Landing Distance", "minimize"),
            ]),
            ("  ▸ Mission", [
                ("max_prob_target", "Max P(Target Alt)", "maximize"),
                ("max_mission_success", "Max Mission Success", "maximize"),
            ]),
            ("  ▸ Economics", [
                ("min_mass", "Min Mass", "minimize"),
                ("min_cost", "Min Cost", "minimize"),
            ]),
        ]

        self._obj_widgets = {}
        for cat_label, objs in objectives_defs:
            cl = QLabel(cat_label)
            cl.setStyleSheet(
                "color:#7ee787; font-size:11px; font-weight:700; padding:4px 0 2px 0;"
            )
            vl.addWidget(cl)

            for key, display, direction in objs:
                row = QHBoxLayout()
                row.setSpacing(4)

                chk = QCheckBox(display)
                chk.setStyleSheet(_CHK)
                chk.setFixedWidth(150)
                if key == "max_apogee":
                    chk.setChecked(True)
                row.addWidget(chk)

                wspin = QDoubleSpinBox()
                wspin.setRange(0.0, 10.0)
                wspin.setValue(1.0)
                wspin.setDecimals(2)
                wspin.setFixedWidth(55)
                wspin.setToolTip("Weight")
                row.addWidget(wspin)

                mode_combo = QComboBox()
                mode_combo.addItems(["mean", "std", "worst", "reliability", "p5"])
                mode_combo.setStyleSheet(_COMBO_SMALL)
                mode_combo.setFixedWidth(80)
                row.addWidget(mode_combo)

                row.addStretch()
                vl.addLayout(row)
                self._obj_widgets[key] = (chk, wspin, mode_combo, direction)

        g.setLayout(vl)
        return g

    # ── Constraints ──────────────────────────────────────────────────────────

    def _build_constraints_group(self) -> QGroupBox:
        g = QGroupBox("Constraints")
        g.setStyleSheet(_GRP)
        vl = QVBoxLayout()
        vl.setSpacing(4)

        constraints_defs = [
            ("stability_min", "Stability >", 1.2, "cal", "greater_than"),
            ("rail_exit_min", "Rail Exit Vel >", 15.0, "m/s", "greater_than"),
            ("mach_max", "Max Mach <", 2.0, "", "less_than"),
            ("accel_max", "Max Accel <", 100.0, "G", "less_than"),
            ("safety_factor_min", "Safety Factor >", 2.0, "", "greater_than"),
            ("landing_dist_max", "Landing Dist <", 1000.0, "m", "less_than"),
            ("mass_max", "Total Mass <", 50.0, "kg", "less_than"),
            ("diameter_min", "Diameter >", 0.03, "m", "greater_than"),
        ]

        self._con_widgets = {}
        for key, display, default_val, unit, con_type in constraints_defs:
            row = QHBoxLayout()
            row.setSpacing(4)

            chk = QCheckBox(display)
            chk.setStyleSheet(_CHK)
            chk.setFixedWidth(130)
            if key in ("stability_min", "rail_exit_min"):
                chk.setChecked(True)
            row.addWidget(chk)

            spin = QDoubleSpinBox()
            spin.setRange(0, 100000)
            spin.setValue(default_val)
            spin.setDecimals(1)
            spin.setFixedWidth(75)
            row.addWidget(spin)

            if unit:
                u = QLabel(unit)
                u.setStyleSheet("color:#484f58; font-size:9px;")
                u.setFixedWidth(28)
                row.addWidget(u)

            row.addStretch()
            vl.addLayout(row)
            self._con_widgets[key] = (chk, spin, con_type)

        g.setLayout(vl)
        return g

    # ── Actions ──────────────────────────────────────────────────────────────

    def _build_actions_group(self) -> QGroupBox:
        g = QGroupBox("Actions")
        g.setStyleSheet(_GRP)
        vl = QVBoxLayout()
        vl.setSpacing(8)

        self.btn_run = QPushButton(app_icon("run", color="#fff"), "START OPTIMIZATION")
        self.btn_run.setStyleSheet(_BTN_P)
        self.btn_run.setMinimumHeight(42)
        self.btn_run.clicked.connect(self._on_run)
        vl.addWidget(self.btn_run)

        self.btn_cancel = QPushButton(app_icon("stop", color="#fff"), "CANCEL")
        self.btn_cancel.setStyleSheet(_BTN_D)
        self.btn_cancel.setMinimumHeight(42)
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._on_cancel)
        vl.addWidget(self.btn_cancel)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFixedHeight(14)
        self.progress_bar.setStyleSheet(
            "QProgressBar { background:#21262d; border-radius:7px; border:none; "
            "color:#c9d1d9; font-size:9px; }"
            "QProgressBar::chunk { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            "stop:0 #1f6feb, stop:1 #58a6ff); border-radius:7px; }"
        )
        vl.addWidget(self.progress_bar)

        self.progress_label = QLabel("Ready — configure parameters and start")
        self.progress_label.setStyleSheet("color:#8b949e; font-size:11px;")
        self.progress_label.setWordWrap(True)
        vl.addWidget(self.progress_label)

        # Export row
        eh = QHBoxLayout()
        self.btn_export_csv = QPushButton(app_icon("export"), "CSV")
        self.btn_export_csv.setStyleSheet(_BTN_S)
        self.btn_export_csv.setEnabled(False)
        self.btn_export_csv.clicked.connect(lambda: self._on_export("csv"))
        eh.addWidget(self.btn_export_csv)

        self.btn_export_json = QPushButton(app_icon("export"), "JSON")
        self.btn_export_json.setStyleSheet(_BTN_S)
        self.btn_export_json.setEnabled(False)
        self.btn_export_json.clicked.connect(lambda: self._on_export("json"))
        eh.addWidget(self.btn_export_json)
        vl.addLayout(eh)

        g.setLayout(vl)
        return g

    # ═════════════════════════════════════════════════════════════════════════
    #  CENTER PANEL — 8 Sub-Tabs
    # ═════════════════════════════════════════════════════════════════════════

    def _build_center(self) -> QWidget:
        wrapper = QWidget()
        vl = QVBoxLayout(wrapper)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(0)

        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(
            "QTabWidget::pane { border:1px solid #21262d; }"
            "QTabBar::tab { background:#161b22; color:#8b949e; padding:6px 12px; "
            "  border:1px solid #21262d; border-bottom:none; border-radius:4px 4px 0 0; "
            "  font-size:11px; }"
            "QTabBar::tab:selected { background:#0d1117; color:#58a6ff; font-weight:700; }"
        )

        # Tab 1: Single Objective
        fig1, self._ax_conv = _dark_figure(figsize=(8, 5))
        self._canvas_conv = FigureCanvas(fig1)
        _style_ax(self._ax_conv, "Fitness Convergence", "Generation", "Fitness")
        self.tabs.addTab(self._canvas_conv, "Single Objective")

        # Tab 2: Multi Objective
        fig2, self._ax_multi = _dark_figure(figsize=(8, 5))
        self._canvas_multi = FigureCanvas(fig2)
        _style_ax(self._ax_multi, "Multi-Objective Space", "Objective 1", "Objective 2")
        self.tabs.addTab(self._canvas_multi, "Multi Objective")

        # Tab 3: Design Space Explorer
        w3 = QWidget()
        v3 = QVBoxLayout(w3)
        v3.setContentsMargins(4, 4, 4, 4)

        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(8)
        lx = QLabel("X-Axis:")
        lx.setStyleSheet("color:#8b949e; font-size:11px;")
        ctrl_row.addWidget(lx)
        self.combo_dse_x = QComboBox()
        self.combo_dse_x.setStyleSheet(_COMBO_SMALL)
        self.combo_dse_x.setMinimumWidth(120)
        ctrl_row.addWidget(self.combo_dse_x)

        ly = QLabel("Y-Axis:")
        ly.setStyleSheet("color:#8b949e; font-size:11px;")
        ctrl_row.addWidget(ly)
        self.combo_dse_y = QComboBox()
        self.combo_dse_y.setStyleSheet(_COMBO_SMALL)
        self.combo_dse_y.setMinimumWidth(120)
        ctrl_row.addWidget(self.combo_dse_y)

        lc = QLabel("Color:")
        lc.setStyleSheet("color:#8b949e; font-size:11px;")
        ctrl_row.addWidget(lc)
        self.combo_dse_color = QComboBox()
        self.combo_dse_color.addItems(["Fitness", "Feasibility", "Apogee", "Stability", "Mach"])
        self.combo_dse_color.setStyleSheet(_COMBO_SMALL)
        ctrl_row.addWidget(self.combo_dse_color)

        btn_refresh = QPushButton("Refresh")
        btn_refresh.setStyleSheet(_BTN_S)
        btn_refresh.setFixedHeight(26)
        btn_refresh.clicked.connect(self._refresh_dse)
        ctrl_row.addWidget(btn_refresh)
        ctrl_row.addStretch()
        v3.addLayout(ctrl_row)

        fig3, self._ax_dse = _dark_figure(figsize=(8, 5))
        self._canvas_dse = FigureCanvas(fig3)
        _style_ax(self._ax_dse, "Design Space Explorer", "", "")
        self._canvas_dse.mpl_connect("button_press_event", self._on_dse_click)
        v3.addWidget(self._canvas_dse, 1)

        self.tabs.addTab(w3, "Design Space Explorer")

        # Tab 4: Pareto Front
        fig4, self._ax_pareto = _dark_figure(figsize=(8, 5))
        self._canvas_pareto = FigureCanvas(fig4)
        _style_ax(self._ax_pareto, "Pareto Front", "Objective 1", "Objective 2")
        self.tabs.addTab(self._canvas_pareto, "Pareto Front")

        # Tab 5: DOE & Response Surfaces
        w5 = QWidget()
        v5 = QVBoxLayout(w5)
        v5.setContentsMargins(4, 4, 4, 4)

        doe_ctrl = QHBoxLayout()
        doe_ctrl.setSpacing(6)
        self.combo_doe_method = QComboBox()
        self.combo_doe_method.addItems(["Latin Hypercube", "Full Factorial", "Taguchi"])
        self.combo_doe_method.setStyleSheet(_COMBO_SMALL)
        doe_ctrl.addWidget(QLabel("DOE Method:"))
        doe_ctrl.addWidget(self.combo_doe_method)

        self.spin_doe_samples = QSpinBox()
        self.spin_doe_samples.setRange(10, 1000)
        self.spin_doe_samples.setValue(50)
        doe_ctrl.addWidget(QLabel("Samples:"))
        doe_ctrl.addWidget(self.spin_doe_samples)

        self.btn_run_doe = QPushButton(app_icon("run"), "Run DOE")
        self.btn_run_doe.setStyleSheet(_BTN_SUCCESS)
        self.btn_run_doe.setFixedHeight(26)
        self.btn_run_doe.clicked.connect(self._on_run_doe)
        doe_ctrl.addWidget(self.btn_run_doe)
        doe_ctrl.addStretch()
        v5.addLayout(doe_ctrl)

        fig5, self._ax_doe = _dark_figure(rows=1, cols=2, figsize=(12, 5))
        self._canvas_doe = FigureCanvas(fig5)
        v5.addWidget(self._canvas_doe, 1)

        self.tabs.addTab(w5, "DOE & Surfaces")

        # Tab 6: Sensitivity Analysis
        w6 = QWidget()
        v6 = QVBoxLayout(w6)
        v6.setContentsMargins(4, 4, 4, 4)

        sens_ctrl = QHBoxLayout()
        self.combo_sens_method = QComboBox()
        self.combo_sens_method.addItems(["Sobol Indices", "PRCC", "Morris Screening"])
        self.combo_sens_method.setStyleSheet(_COMBO_SMALL)
        sens_ctrl.addWidget(QLabel("Method:"))
        sens_ctrl.addWidget(self.combo_sens_method)

        self.spin_sens_samples = QSpinBox()
        self.spin_sens_samples.setRange(64, 4096)
        self.spin_sens_samples.setValue(256)
        sens_ctrl.addWidget(QLabel("Samples:"))
        sens_ctrl.addWidget(self.spin_sens_samples)

        self.btn_run_sens = QPushButton(app_icon("run"), "Analyze")
        self.btn_run_sens.setStyleSheet(_BTN_SUCCESS)
        self.btn_run_sens.setFixedHeight(26)
        self.btn_run_sens.clicked.connect(self._on_run_sensitivity)
        sens_ctrl.addWidget(self.btn_run_sens)
        sens_ctrl.addStretch()
        v6.addLayout(sens_ctrl)

        fig6, self._ax_sens = _dark_figure(rows=1, cols=2, figsize=(12, 5))
        self._canvas_sens = FigureCanvas(fig6)
        v6.addWidget(self._canvas_sens, 1)

        self.tabs.addTab(w6, "Sensitivity")

        # Tab 7: Trade Study
        w7 = QWidget()
        v7 = QVBoxLayout(w7)
        v7.setContentsMargins(4, 4, 4, 4)

        trade_ctrl = QHBoxLayout()
        self.btn_add_current = QPushButton("+ Add Current Design")
        self.btn_add_current.setStyleSheet(_BTN_S)
        self.btn_add_current.setFixedHeight(26)
        self.btn_add_current.clicked.connect(self._on_add_trade_config)
        trade_ctrl.addWidget(self.btn_add_current)

        self.btn_add_best = QPushButton("+ Add Best Optimized")
        self.btn_add_best.setStyleSheet(_BTN_S)
        self.btn_add_best.setFixedHeight(26)
        self.btn_add_best.clicked.connect(self._on_add_best_trade)
        trade_ctrl.addWidget(self.btn_add_best)

        self.btn_run_trade = QPushButton(app_icon("run"), "Compare")
        self.btn_run_trade.setStyleSheet(_BTN_SUCCESS)
        self.btn_run_trade.setFixedHeight(26)
        self.btn_run_trade.clicked.connect(self._on_run_trade)
        trade_ctrl.addWidget(self.btn_run_trade)

        self.btn_clear_trade = QPushButton("Clear")
        self.btn_clear_trade.setStyleSheet(_BTN_D + "QPushButton{font-size:11px;padding:5px 10px;}")
        self.btn_clear_trade.setFixedHeight(26)
        self.btn_clear_trade.clicked.connect(self._on_clear_trade)
        trade_ctrl.addWidget(self.btn_clear_trade)
        trade_ctrl.addStretch()
        v7.addLayout(trade_ctrl)

        self.trade_configs = []
        self.trade_table = QTableWidget(0, 7)
        self.trade_table.setHorizontalHeaderLabels([
            "Name", "Apogee (m)", "Max Mach", "Stability", "Landing (m)", "Mass (kg)", "Success %",
        ])
        self.trade_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.trade_table.setStyleSheet(
            "QTableWidget { background:#0d1117; gridline-color:#21262d; font-size:11px; }"
            "QHeaderView::section { background:#161b22; color:#58a6ff; border:1px solid #21262d; "
            "  padding:4px; font-weight:600; font-size:10px; }"
            "QTableWidget::item { padding:4px; color:#c9d1d9; }"
        )
        v7.addWidget(self.trade_table)

        fig7, self._ax_trade = _dark_figure(figsize=(8, 5))
        self._canvas_trade = FigureCanvas(fig7)
        v7.addWidget(self._canvas_trade, 1)

        self.tabs.addTab(w7, "Trade Study")

        # Tab 8: Optimization History
        self.history_table = QTableWidget(0, 6)
        self.history_table.setHorizontalHeaderLabels([
            "Gen", "Best Fitness", "Mean Fitness", "Worst Fitness",
            "Feasible %", "Best Apogee (m)",
        ])
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.history_table.setStyleSheet(
            "QTableWidget { background:#0d1117; gridline-color:#21262d; font-size:11px; }"
            "QHeaderView::section { background:#161b22; color:#58a6ff; border:1px solid #21262d; "
            "  padding:4px; font-weight:600; font-size:10px; }"
            "QTableWidget::item { padding:4px; color:#c9d1d9; }"
        )
        self.tabs.addTab(self.history_table, "History")

        vl.addWidget(self.tabs, 1)
        return wrapper

    # ═════════════════════════════════════════════════════════════════════════
    #  RIGHT PANEL — Results
    # ═════════════════════════════════════════════════════════════════════════

    def _build_right(self) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumWidth(400)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(12, 14, 12, 14)
        lay.setSpacing(10)

        t = QLabel("Optimization Results")
        t.setStyleSheet(
            "color:#58a6ff; font-size:15px; font-weight:700; padding:2px 0 6px 0;"
        )
        lay.addWidget(t)

        # ── Best Design ──
        gb = QGroupBox("Best Design — Parameters")
        gb.setStyleSheet(_GRP)
        fb = QFormLayout()
        fb.setSpacing(5)
        self.lbl_best_diameter = _vl(); fb.addRow("Diameter:", self.lbl_best_diameter)
        self.lbl_best_length = _vl(); fb.addRow("Length:", self.lbl_best_length)
        self.lbl_best_nose = _vl(); fb.addRow("Nose Length:", self.lbl_best_nose)
        self.lbl_best_fin_span = _vl(); fb.addRow("Fin Span:", self.lbl_best_fin_span)
        self.lbl_best_fin_root = _vl(); fb.addRow("Fin Root:", self.lbl_best_fin_root)
        self.lbl_best_fin_tip = _vl(); fb.addRow("Fin Tip:", self.lbl_best_fin_tip)
        self.lbl_best_mass = _vl(); fb.addRow("Dry Mass:", self.lbl_best_mass)
        self.lbl_best_motor = _vl(); fb.addRow("Motor:", self.lbl_best_motor)
        gb.setLayout(fb)
        lay.addWidget(gb)

        # ── Performance ──
        gp = QGroupBox("Performance")
        gp.setStyleSheet(_GRP)
        fp = QFormLayout()
        fp.setSpacing(5)
        self.lbl_perf_apogee = _vl(); fp.addRow("Apogee:", self.lbl_perf_apogee)
        self.lbl_perf_mach = _vl(); fp.addRow("Max Mach:", self.lbl_perf_mach)
        self.lbl_perf_stability = _vl(); fp.addRow("Stability:", self.lbl_perf_stability)
        self.lbl_perf_landing = _vl(); fp.addRow("Landing Dist:", self.lbl_perf_landing)
        self.lbl_perf_rail = _vl(); fp.addRow("Rail Exit Vel:", self.lbl_perf_rail)
        self.lbl_perf_accel = _vl(); fp.addRow("Max Accel:", self.lbl_perf_accel)
        gp.setLayout(fp)
        lay.addWidget(gp)

        # ── Reliability ──
        gr = QGroupBox("Reliability & Uncertainty")
        gr.setStyleSheet(_GRP)
        fr = QFormLayout()
        fr.setSpacing(5)
        self.lbl_rel_success = _vl(); fr.addRow("Success Rate:", self.lbl_rel_success)
        self.lbl_rel_p_target = _vl(); fr.addRow("P(Target Alt):", self.lbl_rel_p_target)
        self.lbl_rel_apogee_std = _vl(); fr.addRow("Apogee σ:", self.lbl_rel_apogee_std)
        self.lbl_rel_ci = _vl(); fr.addRow("95% CI:", self.lbl_rel_ci)
        self.lbl_rel_skew = _vl(); fr.addRow("Skewness:", self.lbl_rel_skew)
        self.lbl_rel_kurt = _vl(); fr.addRow("Kurtosis:", self.lbl_rel_kurt)
        gr.setLayout(fr)
        lay.addWidget(gr)

        # ── Pareto Solutions ──
        gps = QGroupBox("Pareto Solutions")
        gps.setStyleSheet(_GRP)
        fps = QVBoxLayout()
        fps.setSpacing(4)

        sol_style = (
            "QPushButton{{color:{c};background:#161b22;border:1px solid #30363d;"
            "border-radius:4px;padding:6px;font-size:11px;text-align:left;}}"
            "QPushButton:hover{{background:#1c2333;border-color:#58a6ff;}}"
        )
        self.btn_sol_apogee = QPushButton(app_icon("apogee", color="#7ee787"), "Best Apogee: —")
        self.btn_sol_apogee.setStyleSheet(sol_style.format(c="#7ee787"))
        self.btn_sol_apogee.clicked.connect(lambda: self._load_pareto_solution("apogee"))
        fps.addWidget(self.btn_sol_apogee)

        self.btn_sol_reliability = QPushButton(app_icon("reliability", color="#58a6ff"), "Best Reliability: —")
        self.btn_sol_reliability.setStyleSheet(sol_style.format(c="#58a6ff"))
        self.btn_sol_reliability.clicked.connect(lambda: self._load_pareto_solution("reliability"))
        fps.addWidget(self.btn_sol_reliability)

        self.btn_sol_mass = QPushButton(app_icon("mass", color="#bc8cff"), "Best Mass Efficiency: —")
        self.btn_sol_mass.setStyleSheet(sol_style.format(c="#bc8cff"))
        self.btn_sol_mass.clicked.connect(lambda: self._load_pareto_solution("mass"))
        fps.addWidget(self.btn_sol_mass)

        self.btn_sol_balanced = QPushButton(app_icon("balanced", color="#f0883e"), "Best Balanced: —")
        self.btn_sol_balanced.setStyleSheet(sol_style.format(c="#f0883e"))
        self.btn_sol_balanced.clicked.connect(lambda: self._load_pareto_solution("balanced"))
        fps.addWidget(self.btn_sol_balanced)

        gps.setLayout(fps)
        lay.addWidget(gps)

        # ── Constraint Status ──
        gc = QGroupBox("Constraint Status")
        gc.setStyleSheet(_GRP)
        self._constraint_layout = QVBoxLayout()
        self._constraint_layout.setSpacing(3)
        self._con_placeholder = QLabel("No results yet")
        self._con_placeholder.setStyleSheet("color:#484f58; font-size:11px;")
        self._constraint_layout.addWidget(self._con_placeholder)
        gc.setLayout(self._constraint_layout)
        lay.addWidget(gc)

        # ── Improvement Over Baseline ──
        gi = QGroupBox("Improvement Over Baseline")
        gi.setStyleSheet(_GRP)
        fi = QFormLayout()
        fi.setSpacing(5)
        self.lbl_imp_apogee = _vl(); fi.addRow("Apogee Δ:", self.lbl_imp_apogee)
        self.lbl_imp_stability = _vl(); fi.addRow("Stability Δ:", self.lbl_imp_stability)
        self.lbl_imp_mass = _vl(); fi.addRow("Mass Δ:", self.lbl_imp_mass)
        self.lbl_imp_landing = _vl(); fi.addRow("Landing Δ:", self.lbl_imp_landing)
        gi.setLayout(fi)
        lay.addWidget(gi)

        # ── Optimization Stats ──
        gs = QGroupBox("Optimization Statistics")
        gs.setStyleSheet(_GRP)
        fs = QFormLayout()
        fs.setSpacing(5)
        self.lbl_stat_evals = _vl(); fs.addRow("Evaluations:", self.lbl_stat_evals)
        self.lbl_stat_time = _vl(); fs.addRow("Time:", self.lbl_stat_time)
        self.lbl_stat_algo = _vl(); fs.addRow("Algorithm:", self.lbl_stat_algo)
        self.lbl_stat_surrogate = _vl(); fs.addRow("Surrogate R²:", self.lbl_stat_surrogate)
        gs.setLayout(fs)
        lay.addWidget(gs)

        lay.addStretch()
        scroll.setWidget(inner)
        return scroll

    # ═════════════════════════════════════════════════════════════════════════
    #  ACTIONS — Run / Cancel / Export
    # ═════════════════════════════════════════════════════════════════════════

    def _collect_config(self):
        """Build OptimizationConfig from UI widgets."""
        try:
            from core.optimization_engine import (
                OptimizationConfig, DesignVariable, ObjectiveFunction,
                Constraint, CorrelationEntry, get_default_correlations,
            )
        except ImportError as e:
            QMessageBox.critical(self, "Import Error",
                f"Cannot import optimization engine:\n\n{e}")
            return None

        # Algorithm
        algo_map = {0: "ga", 1: "nsga2", 2: "de", 3: "pso"}
        algorithm = algo_map.get(self.combo_algorithm.currentIndex(), "ga")

        # Design variables
        design_vars = []
        state = self.engine.state
        for var_key, (chk, spin_min, spin_max, cat) in self._var_widgets.items():
            if not chk.isChecked():
                continue
            current = getattr(state, var_key, (spin_min.value() + spin_max.value()) / 2)
            if current == 0:
                current = (spin_min.value() + spin_max.value()) / 2
            vtype = "continuous"
            if var_key == "fin_count":
                vtype = "integer"
            design_vars.append(DesignVariable(
                name=var_key,
                display_name=chk.parent().findChild(QLabel).text() if chk.parent() else var_key,
                category=cat,
                min_val=spin_min.value(),
                max_val=spin_max.value(),
                current_val=float(current),
                enabled=True,
                var_type=vtype,
            ))

        if not design_vars:
            QMessageBox.warning(self, "Configuration Error",
                "No design variables selected.\n\nPlease check at least one variable to optimize.")
            return None

        # Objectives
        objectives = []
        for key, (chk, wspin, mode_combo, direction) in self._obj_widgets.items():
            if chk.isChecked():
                objectives.append(ObjectiveFunction(
                    name=key,
                    display_name=chk.text(),
                    direction=direction,
                    weight=wspin.value(),
                    enabled=True,
                    robust_mode=mode_combo.currentText(),
                ))

        if not objectives:
            QMessageBox.warning(self, "Configuration Error",
                "No objectives selected.\n\nPlease check at least one objective function.")
            return None

        # Constraints
        constraints = []
        for key, (chk, spin, con_type) in self._con_widgets.items():
            if chk.isChecked():
                constraints.append(Constraint(
                    name=key,
                    display_name=chk.text(),
                    type=con_type,
                    limit=spin.value(),
                    penalty_weight=1000.0,
                    enabled=True,
                ))

        # Mode
        mode_btn = self._mode_group.checkedButton()
        mode_val = mode_btn.property("mode_value") if mode_btn else "standard"

        # Auto mission mode: a non-zero target apogee in Standard mode is
        # otherwise silently ignored (Standard maximizes mean apogee). Treat
        # it as a Mission Target so the optimizer actually hits the target.
        if mode_val == "standard" and self.spin_target_apogee.value() > 0:
            mode_val = "mission"

        # Surrogate
        surr_map = {
            0: "random_forest", 1: "gradient_boosting", 2: "neural_network",
            3: "kriging", 4: "rbf", 5: "polynomial",
        }

        config = OptimizationConfig(
            algorithm=algorithm,
            design_variables=design_vars,
            objectives=objectives,
            constraints=constraints,
            correlations=get_default_correlations(),
            population_size=self.spin_pop.value(),
            max_generations=self.spin_gens.value(),
            mutation_rate=self.spin_mutation.value(),
            crossover_rate=self.spin_crossover.value(),
            mc_sims_per_candidate=self.spin_mc_per.value(),
            validation_mc_sims=self.spin_mc_val.value(),
            use_surrogate=self.chk_surrogate.isChecked(),
            surrogate_type=surr_map.get(self.combo_surrogate.currentIndex(), "random_forest"),
            surrogate_initial_samples=self.spin_surrogate_samples.value(),
            target_apogee=self.spin_target_apogee.value(),
            mission_mode=(mode_val == "mission"),
            robust_mode=(mode_val == "robust"),
            parallel=self.chk_parallel.isChecked(),
            n_workers=self.spin_workers.value(),
        )
        # Remember the user's enabled objectives so plots show THOSE axes,
        # not just the first keys of the objectives dict.
        self._active_objectives = [(o.name, o.direction) for o in objectives]
        return config

    def _plot_objective_axes(self, fallback_design):
        """First two user-enabled objectives as [(key, direction), ...].
        Falls back to the first two dict keys when fewer than 2 enabled."""
        active = getattr(self, "_active_objectives", [])
        if len(active) >= 2:
            return active[0], active[1]
        keys = list(fallback_design.objectives.keys())
        if len(keys) < 2:
            keys = keys + keys
        pairs = [(k, "maximize") for k in keys[:2]]
        if len(active) == 1:
            other = next((p for p in pairs if p[0] != active[0][0]), pairs[1])
            return active[0], other
        return pairs[0], pairs[1]

    def _on_run(self):
        """Collect config and start optimization."""
        config = self._collect_config()
        if config is None:
            return

        # Validate motor
        if self.engine.state.motor_total_impulse <= 0.0:
            if not getattr(self.engine.state, 'custom_thrust_curve', None):
                QMessageBox.warning(self, "Configuration Error",
                    "No motor configured.\n\nPlease select a motor in the Propulsion "
                    "workspace before running optimization.")
                return

        try:
            from core.optimization_engine import OptimizationEngine
        except ImportError as e:
            QMessageBox.critical(self, "Import Error", f"Cannot import OptimizationEngine:\n{e}")
            return

        if self._opt_engine is None:
            self._opt_engine = OptimizationEngine(self.engine)
        else:
            try:
                self._opt_engine.progress.disconnect()
                self._opt_engine.status_update.disconnect()
                self._opt_engine.generation_complete.disconnect()
                self._opt_engine.optimization_finished.disconnect()
                self._opt_engine.optimization_failed.disconnect()
                self._opt_engine.optimization_cancelled.disconnect()
            except TypeError:
                pass

        self._opt_engine.progress.connect(self._on_progress)
        self._opt_engine.status_update.connect(self._on_status_update)
        self._opt_engine.generation_complete.connect(self._on_generation)
        self._opt_engine.optimization_finished.connect(self._on_finished)
        self._opt_engine.optimization_failed.connect(self._on_failed)
        self._opt_engine.optimization_cancelled.connect(self._on_cancelled)

        # UI state
        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.btn_export_csv.setEnabled(False)
        self.btn_export_json.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Starting optimization…")
        self.history_table.setRowCount(0)
        self._run_started_at = time.time()
        self._conv_bests = []
        self._conv_means = []
        self._conv_worsts = []
        self._ax_conv.clear()
        _style_ax(self._ax_conv, "Fitness Convergence", "Generation", "Fitness")
        self._canvas_conv.draw_idle()
        self.engine.log_message.emit(
            f"Optimization started: {config.algorithm.upper()} | "
            f"population={config.population_size}, generations={config.max_generations}, "
            f"sims/candidate={config.mc_sims_per_candidate}"
        )

        self._opt_engine.start(config)
        logger.info(f"Optimization started: {config.algorithm}, pop={config.population_size}, "
                     f"gens={config.max_generations}")

    def _on_cancel(self):
        if self._opt_engine:
            self._opt_engine.cancel()
            self.progress_label.setText("Cancelling…")

    def _on_status_update(self, status: dict):
        """Show live worker status while candidate simulations are running."""
        phase = status.get("phase", "running")
        if phase == "generation":
            return

        elapsed = time.time() - getattr(self, "_run_started_at", time.time())
        evaluations = int(status.get("evaluations", 0) or 0)
        estimated = int(status.get("estimated_evaluations", 0) or 0)
        if estimated > 0:
            self.progress_bar.setValue(min(99, int(evaluations * 100 / estimated)))

        candidate = int(status.get("evaluated_candidates", 0) or 0)
        total_candidates = int(status.get("total_candidates", 0) or 0)
        generation = int(status.get("generation", 0) or 0)
        best = status.get("best_fitness", 0.0) or 0.0
        message = status.get("message", "Running optimization")

        candidate_text = ""
        if total_candidates:
            candidate_text = f" | candidate {candidate}/{total_candidates}"

        self.progress_label.setText(
            f"{message} | gen {generation}/{self.spin_gens.value()}"
            f"{candidate_text} | evals {evaluations:,}"
            f" | best {best:.2f} | {elapsed:.0f}s"
        )

    def _on_progress(self, gen: int, total: int, best_fitness: float):
        gen_done = min(gen + 1, total)
        pct = gen_done * 100 / max(total, 1)
        self.progress_bar.setValue(int(pct))
        self.progress_label.setText(
            f"Generation {gen_done}/{total} complete — Best fitness: {best_fitness:.2f}"
        )

    def _on_generation(self, gen_data):
        """Update history table and convergence plot on each generation."""
        row = self.history_table.rowCount()
        self.history_table.insertRow(row)

        gen = gen_data.get("generation", row)
        best = gen_data.get("best_fitness", 0)
        mean = gen_data.get("mean_fitness", 0)
        worst = gen_data.get("worst_fitness", 0)
        feas_pct = gen_data.get("feasible_pct", 0)
        best_apogee = gen_data.get("best_apogee", 0)

        items = [
            f"{gen + 1}", f"{best:.2f}", f"{mean:.2f}", f"{worst:.2f}",
            f"{feas_pct:.0f}%", f"{best_apogee:.1f}",
        ]
        for col, text in enumerate(items):
            item = QTableWidgetItem(text)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.history_table.setItem(row, col, item)

        self.history_table.scrollToBottom()

        # Update convergence plot incrementally
        self._update_convergence_incremental(gen_data)

    def _on_finished(self, result):
        """Optimization complete — update all displays."""
        self._result = result
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.btn_export_csv.setEnabled(True)
        self.btn_export_json.setEnabled(True)
        self.progress_bar.setValue(100)

        n = result.total_evaluations
        t = result.elapsed_time
        self.progress_label.setText(
            f"Complete — {n} evaluations in {t:.1f}s | "
            f"Algorithm: {result.algorithm_used}"
        )

        self._update_all_plots(result)
        self._update_results_panel(result)
        self._populate_dse_combos(result)

        logger.info(f"Optimization complete: {n} evals, {t:.1f}s, "
                     f"best fitness={result.best_design.fitness:.3f}")

    def _on_failed(self, error_msg: str):
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress_label.setText("Failed")
        QMessageBox.critical(self, "Optimization Failed", f"Optimization failed:\n\n{error_msg}")

    def _on_cancelled(self):
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress_label.setText("Cancelled by user")
        logger.info("Optimization cancelled")

    # ═════════════════════════════════════════════════════════════════════════
    #  PLOT UPDATES
    # ═════════════════════════════════════════════════════════════════════════

    def _update_convergence_incremental(self, gen_data):
        """Incrementally update the convergence plot."""
        ax = self._ax_conv
        if not hasattr(self, '_conv_bests'):
            self._conv_bests = []
            self._conv_means = []
            self._conv_worsts = []

        self._conv_bests.append(gen_data.get("best_fitness", 0))
        self._conv_means.append(gen_data.get("mean_fitness", 0))
        self._conv_worsts.append(gen_data.get("worst_fitness", 0))

        ax.clear()
        _style_ax(ax, "Fitness Convergence", "Generation", "Fitness")

        gens = list(range(len(self._conv_bests)))
        ax.plot(gens, self._conv_bests, color="#7ee787", linewidth=2, label="Best")
        ax.fill_between(gens, self._conv_worsts, self._conv_bests,
                         alpha=0.15, color="#58a6ff")
        ax.plot(gens, self._conv_means, color="#58a6ff", linewidth=1.2,
                alpha=0.7, linestyle="--", label="Mean")
        ax.plot(gens, self._conv_worsts, color="#f85149", linewidth=0.8,
                alpha=0.5, label="Worst")

        ax.legend(facecolor="#161b22", edgecolor="#30363d",
                  labelcolor="#c9d1d9", fontsize=8, loc="lower right")

        ax.figure.tight_layout()
        self._canvas_conv.draw_idle()

    def _update_all_plots(self, result):
        """Update all plots after optimization completes."""
        self._conv_bests = []
        self._conv_means = []
        self._conv_worsts = []

        self._plot_convergence(result)
        self._plot_multi_objective(result)
        self._plot_pareto(result)

    def _plot_convergence(self, result):
        ax = self._ax_conv
        ax.clear()
        _style_ax(ax, "Fitness Convergence", "Generation", "Fitness")

        if not result.generation_history:
            self._canvas_conv.draw()
            return

        gens = [g.get("generation", i) for i, g in enumerate(result.generation_history)]
        bests = [g.get("best_fitness", 0) for g in result.generation_history]
        means = [g.get("mean_fitness", 0) for g in result.generation_history]
        worsts = [g.get("worst_fitness", 0) for g in result.generation_history]

        ax.plot(gens, bests, color="#7ee787", linewidth=2, label="Best", zorder=3)
        ax.fill_between(gens, worsts, bests, alpha=0.12, color="#58a6ff")
        ax.plot(gens, means, color="#58a6ff", linewidth=1.2, alpha=0.8,
                linestyle="--", label="Mean")
        ax.plot(gens, worsts, color="#f85149", linewidth=0.8, alpha=0.5, label="Worst")

        ax.legend(facecolor="#161b22", edgecolor="#30363d",
                  labelcolor="#c9d1d9", fontsize=8, loc="lower right")
        ax.figure.tight_layout()
        self._canvas_conv.draw()

    def _plot_multi_objective(self, result):
        ax = self._ax_multi
        ax.clear()
        _style_ax(ax, "Multi-Objective Space", "Objective 1", "Objective 2")

        if not result.all_designs or len(result.all_designs) < 2:
            self._canvas_multi.draw()
            return

        # Plot the objectives the user actually optimised
        (k1, _), (k2, _) = self._plot_objective_axes(result.all_designs[0])

        # All designs
        x_all = [d.objectives.get(k1, 0) for d in result.all_designs]
        y_all = [d.objectives.get(k2, 0) for d in result.all_designs]
        feasible = [d.feasible for d in result.all_designs]

        colors = ["#58a6ff" if f else "#484f58" for f in feasible]
        ax.scatter(x_all, y_all, c=colors, s=12, alpha=0.4, zorder=2)

        # Pareto front
        if result.pareto_front:
            x_p = [d.objectives.get(k1, 0) for d in result.pareto_front]
            y_p = [d.objectives.get(k2, 0) for d in result.pareto_front]
            ax.scatter(x_p, y_p, c="#7ee787", s=40, zorder=4,
                       edgecolors="#ffffff", linewidths=0.8, label="Pareto Front")
            # Sort and connect
            pairs = sorted(zip(x_p, y_p))
            if pairs:
                ax.plot([p[0] for p in pairs], [p[1] for p in pairs],
                        color="#7ee787", linewidth=1.5, alpha=0.6, zorder=3)

        # Best design
        if result.best_design:
            bx = result.best_design.objectives.get(k1, 0)
            by = result.best_design.objectives.get(k2, 0)
            ax.scatter([bx], [by], c="#f0883e", s=100, marker="*",
                       zorder=5, label="Best Design")

        ax.set_xlabel(k1.replace("_", " ").title(), color="#8b949e", fontsize=10)
        ax.set_ylabel(k2.replace("_", " ").title(), color="#8b949e", fontsize=10)
        ax.legend(facecolor="#161b22", edgecolor="#30363d",
                  labelcolor="#c9d1d9", fontsize=8, loc="upper right")
        ax.figure.tight_layout()
        self._canvas_multi.draw()

    def _plot_pareto(self, result):
        ax = self._ax_pareto
        ax.clear()
        _style_ax(ax, "Pareto Front — Trade-Off Curve", "", "")

        if not result.pareto_front or len(result.pareto_front) < 2:
            ax.text(0.5, 0.5, "Select ≥2 objectives, then run\nto generate a Pareto front",
                    transform=ax.transAxes, ha="center", va="center",
                    color="#484f58", fontsize=13)
            self._canvas_pareto.draw()
            return

        (k1, dir1), (k2, dir2) = self._plot_objective_axes(result.pareto_front[0])
        x_vals = [d.objectives.get(k1, 0) for d in result.pareto_front]
        y_vals = [d.objectives.get(k2, 0) for d in result.pareto_front]

        # Rank by color gradient
        n = len(result.pareto_front)
        colors = np.linspace(0.2, 0.9, n)
        scatter = ax.scatter(x_vals, y_vals, c=colors, cmap="cool",
                             s=60, zorder=3, edgecolors="#30363d", linewidths=0.5)

        # Connect the front
        pairs = sorted(zip(x_vals, y_vals))
        ax.plot([p[0] for p in pairs], [p[1] for p in pairs],
                color="#58a6ff", linewidth=1.5, alpha=0.4, zorder=2)

        # Annotate best solutions (direction-aware: best of a minimize
        # objective is its minimum)
        if x_vals:
            idx_best_x = int(np.argmax(x_vals) if dir1 == "maximize" else np.argmin(x_vals))
            ax.annotate("Best " + k1.replace("_", " "),
                        (x_vals[idx_best_x], y_vals[idx_best_x]),
                        textcoords="offset points", xytext=(10, 10),
                        fontsize=8, color="#7ee787",
                        arrowprops=dict(arrowstyle="->", color="#7ee787", lw=0.8))

            idx_best_y = int(np.argmax(y_vals) if dir2 == "maximize" else np.argmin(y_vals))
            if idx_best_y != idx_best_x:
                ax.annotate("Best " + k2.replace("_", " "),
                            (x_vals[idx_best_y], y_vals[idx_best_y]),
                            textcoords="offset points", xytext=(10, -15),
                            fontsize=8, color="#bc8cff",
                            arrowprops=dict(arrowstyle="->", color="#bc8cff", lw=0.8))

        # Utopia point (per-axis ideal, respecting objective direction)
        ux = (max(x_vals) if dir1 == "maximize" else min(x_vals)) if x_vals else 0
        uy = (max(y_vals) if dir2 == "maximize" else min(y_vals)) if y_vals else 0
        ax.scatter([ux], [uy], c="#f0883e", s=120, marker="D", zorder=5,
                   label="Utopia Point", edgecolors="#ffffff", linewidths=1)

        ax.set_xlabel(k1.replace("_", " ").title(), color="#8b949e", fontsize=10)
        ax.set_ylabel(k2.replace("_", " ").title(), color="#8b949e", fontsize=10)
        ax.legend(facecolor="#161b22", edgecolor="#30363d",
                  labelcolor="#c9d1d9", fontsize=8, loc="upper right")
        ax.figure.tight_layout()
        self._canvas_pareto.draw()

    # ═════════════════════════════════════════════════════════════════════════
    #  DESIGN SPACE EXPLORER
    # ═════════════════════════════════════════════════════════════════════════

    def _populate_dse_combos(self, result):
        """Populate the Design Space Explorer axis combos."""
        self.combo_dse_x.clear()
        self.combo_dse_y.clear()

        if not result.all_designs:
            return

        # Variable keys
        var_keys = list(result.all_designs[0].variables.keys())
        # Objective keys
        obj_keys = list(result.all_designs[0].objectives.keys())
        all_keys = var_keys + obj_keys

        self.combo_dse_x.addItems(all_keys)
        self.combo_dse_y.addItems(all_keys)

        # Default: first variable vs first objective
        if obj_keys:
            idx = all_keys.index(obj_keys[0]) if obj_keys[0] in all_keys else 0
            self.combo_dse_y.setCurrentIndex(idx)

    def _refresh_dse(self):
        """Refresh the Design Space Explorer plot."""
        if not self._result or not self._result.all_designs:
            return

        ax = self._ax_dse
        ax.clear()

        x_key = self.combo_dse_x.currentText()
        y_key = self.combo_dse_y.currentText()
        color_key = self.combo_dse_color.currentText()

        designs = self._result.all_designs

        def _get_val(d, key):
            if key in d.variables:
                return d.variables[key]
            if key in d.objectives:
                return d.objectives[key]
            return 0

        x_vals = [_get_val(d, x_key) for d in designs]
        y_vals = [_get_val(d, y_key) for d in designs]

        # Color mapping
        if color_key == "Fitness":
            c_vals = [d.fitness for d in designs]
            cmap = "viridis"
        elif color_key == "Feasibility":
            c_vals = [1.0 if d.feasible else 0.0 for d in designs]
            cmap = "RdYlGn"
        elif color_key == "Apogee":
            c_vals = [d.objectives.get("max_apogee", d.objectives.get("apogee", 0))
                      for d in designs]
            cmap = "plasma"
        elif color_key == "Stability":
            c_vals = [d.objectives.get("max_stability_margin",
                      d.objectives.get("stability", 0)) for d in designs]
            cmap = "cool"
        else:  # Mach
            c_vals = [d.objectives.get("max_mach", 0) for d in designs]
            cmap = "hot"

        sc = ax.scatter(x_vals, y_vals, c=c_vals, cmap=cmap,
                        s=20, alpha=0.6, edgecolors="#30363d", linewidths=0.3)

        # Best design highlight
        if self._result.best_design:
            bx = _get_val(self._result.best_design, x_key)
            by = _get_val(self._result.best_design, y_key)
            ax.scatter([bx], [by], c="#f0883e", s=100, marker="*",
                       zorder=5, edgecolors="#ffffff", linewidths=1)

        # Colorbar
        try:
            cb = ax.figure.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
            cb.ax.tick_params(colors="#484f58", labelsize=8)
            cb.set_label(color_key, color="#8b949e", fontsize=9)
        except Exception:
            pass

        _style_ax(ax, "Design Space Explorer",
                  x_key.replace("_", " ").title(),
                  y_key.replace("_", " ").title())

        # Store scatter data for click interaction
        self._dse_x_vals = x_vals
        self._dse_y_vals = y_vals
        self._dse_designs = designs

        ax.figure.tight_layout()
        self._canvas_dse.draw()

    def _on_dse_click(self, event):
        """Handle click on Design Space Explorer to load a design."""
        if event.inaxes != self._ax_dse:
            return
        if not hasattr(self, '_dse_x_vals') or not self._dse_x_vals:
            return

        # Find nearest point
        x_click, y_click = event.xdata, event.ydata
        if x_click is None or y_click is None:
            return

        x_arr = np.array(self._dse_x_vals)
        y_arr = np.array(self._dse_y_vals)

        # Normalize to plot range
        x_range = x_arr.max() - x_arr.min() if x_arr.max() != x_arr.min() else 1
        y_range = y_arr.max() - y_arr.min() if y_arr.max() != y_arr.min() else 1

        dists = ((x_arr - x_click) / x_range) ** 2 + ((y_arr - y_click) / y_range) ** 2
        idx = int(np.argmin(dists))

        design = self._dse_designs[idx]
        self._selected_design = design
        self._update_results_panel_from_design(design)

        # Highlight selection
        ax = self._ax_dse
        # Remove old selection markers
        for c in list(ax.collections):
            if hasattr(c, '_is_selection'):
                c.remove()
        sc = ax.scatter([self._dse_x_vals[idx]], [self._dse_y_vals[idx]],
                        c="none", s=200, edgecolors="#ffffff", linewidths=2, zorder=10)
        sc._is_selection = True
        self._canvas_dse.draw_idle()

    # ═════════════════════════════════════════════════════════════════════════
    #  DOE & SENSITIVITY (run via thread)
    # ═════════════════════════════════════════════════════════════════════════

    def _make_swept_config(self, base_config, enabled_vars, values):
        """Apply swept variable values onto a base config using the SAME mapping
        the optimizer uses (fin_span→fin_height, motor designation lookup, and
        derived avg/max thrust from total impulse). Direct setattr would leave
        motor_avg_thrust / fin_height stale, so propulsion and fin sweeps would
        have no effect on the simulation."""
        from core.optimization_engine import build_candidate_config, DesignVariable
        dv_list = [
            DesignVariable(
                name=k, display_name=k, category="",
                min_val=mn, max_val=mx, current_val=mn, enabled=True,
                var_type="integer" if k == "fin_count" else "continuous",
            )
            for (k, mn, mx) in enabled_vars
        ]
        variables = {ev[0]: val for ev, val in zip(enabled_vars, values)}
        return build_candidate_config(base_config, variables, dv_list)

    def _on_run_doe(self):
        """Launch Design of Experiments analysis on a worker thread."""
        if getattr(self, "_doe_worker", None) is not None and self._doe_worker.isRunning():
            return
        from core.batch_simulation import BatchSimConfig

        enabled_vars = []
        for key, (chk, smin, smax, cat) in self._var_widgets.items():
            if chk.isChecked():
                enabled_vars.append((key, smin.value(), smax.value()))
        if len(enabled_vars) < 1:
            QMessageBox.warning(self, "DOE Error", "Enable at least 1 design variable.")
            return

        self.progress_label.setText("Running DOE analysis…")
        self.btn_run_doe.setEnabled(False)
        try:
            base_config = BatchSimConfig.from_rocket_state(self.engine.state)
            self._doe_worker = _DOEWorker(
                base_config, enabled_vars, self.combo_doe_method.currentText(),
                self.spin_doe_samples.value(), self._make_swept_config)
            self._doe_worker.finished_ok.connect(
                lambda dm, resp, ev=enabled_vars: self._render_doe(ev, dm, resp))
            self._doe_worker.failed.connect(self._on_doe_failed)
            self._doe_worker.start()
        except Exception as e:
            self._on_doe_failed(str(e))

    def _on_doe_failed(self, msg):
        QMessageBox.warning(self, "DOE Error", f"DOE analysis failed:\n\n{msg}")
        logger.error(f"DOE failed: {msg}")
        self.progress_label.setText("DOE failed.")
        self.btn_run_doe.setEnabled(True)

    def _render_doe(self, enabled_vars, dm, responses):
        """Plot DOE main effects + response surface (runs on the UI thread)."""
        try:
            n_vars = len(enabled_vars)
            n_samples = len(dm)
            responses = np.asarray(responses)

            ax_left, ax_right = self._ax_doe
            ax_left.clear()
            ax_right.clear()

            # Main effects plot
            var_names = [v[0] for v in enabled_vars]
            _style_ax(ax_left, "Main Effects", "Variable", "Mean Response")
            if len(enabled_vars) <= 8:
                effects = []
                for j in range(n_vars):
                    median_val = np.median(dm[:, j])
                    low_mask = dm[:, j] < median_val
                    high_mask = dm[:, j] >= median_val
                    if np.sum(low_mask) > 0 and np.sum(high_mask) > 0:
                        effect = np.mean(responses[high_mask]) - np.mean(responses[low_mask])
                    else:
                        effect = 0
                    effects.append(effect)

                colors = ["#7ee787" if e > 0 else "#f85149" for e in effects]
                ax_left.barh(var_names, effects, color=colors, alpha=0.8,
                             edgecolor="#30363d")
                ax_left.axvline(0, color="#484f58", linewidth=0.5)

            # Response surface contour (first two variables)
            _style_ax(ax_right, "Response Surface", "", "")
            if n_vars >= 2:
                v1 = enabled_vars[0]
                v2 = enabled_vars[1]
                x_grid = np.linspace(v1[1], v1[2], 30)
                y_grid = np.linspace(v2[1], v2[2], 30)
                X_g, Y_g = np.meshgrid(x_grid, y_grid)

                # Fit simple polynomial surface
                from numpy.polynomial import polynomial as P
                x_data = v1[1] + dm[:, 0] * (v1[2] - v1[1])
                y_data = v2[1] + dm[:, 1] * (v2[2] - v2[1])

                # Simple 2D polynomial fit
                try:
                    from sklearn.preprocessing import PolynomialFeatures
                    from sklearn.linear_model import LinearRegression
                    xy = np.column_stack([x_data, y_data])
                    poly = PolynomialFeatures(degree=2)
                    xy_poly = poly.fit_transform(xy)
                    reg = LinearRegression().fit(xy_poly, responses)

                    grid_pts = np.column_stack([X_g.ravel(), Y_g.ravel()])
                    grid_poly = poly.transform(grid_pts)
                    Z_g = reg.predict(grid_poly).reshape(X_g.shape)

                    cs = ax_right.contourf(X_g, Y_g, Z_g, levels=20,
                                            cmap="viridis", alpha=0.8)
                    ax_right.contour(X_g, Y_g, Z_g, levels=10,
                                     colors="#c9d1d9", linewidths=0.3, alpha=0.5)
                    try:
                        cb = ax_right.figure.colorbar(cs, ax=ax_right,
                                                       fraction=0.03, pad=0.02)
                        cb.ax.tick_params(colors="#484f58", labelsize=8)
                        cb.set_label("Apogee (m)", color="#8b949e", fontsize=9)
                    except Exception:
                        pass

                    ax_right.scatter(x_data, y_data, c=responses, cmap="viridis",
                                     s=15, edgecolors="#ffffff", linewidths=0.3,
                                     zorder=3, alpha=0.7)
                except Exception:
                    ax_right.text(0.5, 0.5, "Could not fit surface",
                                  transform=ax_right.transAxes, ha="center",
                                  color="#484f58")

                ax_right.set_xlabel(v1[0].replace("_", " ").title(),
                                     color="#8b949e", fontsize=10)
                ax_right.set_ylabel(v2[0].replace("_", " ").title(),
                                     color="#8b949e", fontsize=10)

            for a in self._ax_doe:
                a.figure.tight_layout()
            self._canvas_doe.draw()

            self.progress_label.setText(
                f"DOE complete — {len(dm)} samples evaluated"
            )

        except Exception as e:
            QMessageBox.warning(self, "DOE Error", f"DOE analysis failed:\n\n{e}")
            logger.error(f"DOE failed: {e}", exc_info=True)
        finally:
            self.btn_run_doe.setEnabled(True)

    def _on_run_sensitivity(self):
        """Launch sensitivity analysis on a worker thread."""
        if getattr(self, "_sens_worker", None) is not None and self._sens_worker.isRunning():
            return
        from core.batch_simulation import BatchSimConfig

        enabled_vars = []
        for key, (chk, smin, smax, cat) in self._var_widgets.items():
            if chk.isChecked():
                enabled_vars.append((key, smin.value(), smax.value()))
        if len(enabled_vars) < 2:
            QMessageBox.warning(self, "Sensitivity Error",
                "Enable at least 2 design variables.")
            return

        self.progress_label.setText("Running sensitivity analysis…")
        self.btn_run_sens.setEnabled(False)
        try:
            base_config = BatchSimConfig.from_rocket_state(self.engine.state)
            self._sens_worker = _SensitivityWorker(
                base_config, enabled_vars, self.combo_sens_method.currentText(),
                self.spin_sens_samples.value(), self._make_swept_config)
            self._sens_worker.finished_ok.connect(
                lambda out, ev=enabled_vars: self._render_sensitivity(ev, out))
            self._sens_worker.failed.connect(self._on_sens_failed)
            self._sens_worker.start()
        except Exception as e:
            self._on_sens_failed(str(e))

    def _on_sens_failed(self, msg):
        QMessageBox.warning(self, "Sensitivity Error",
            f"Sensitivity analysis failed:\n\n{msg}")
        logger.error(f"Sensitivity analysis failed: {msg}")
        self.progress_label.setText("Sensitivity analysis failed.")
        self.btn_run_sens.setEnabled(True)

    def _render_sensitivity(self, enabled_vars, out):
        """Plot sensitivity results (runs on the UI thread — no simulation)."""
        try:
            n_vars = len(enabled_vars)
            var_names = [v[0].replace("_", " ").title() for v in enabled_vars]
            X_data, y_data, method = out["X"], out["y"], out["method"]

            ax_left, ax_right = self._ax_sens
            ax_left.clear()
            ax_right.clear()

            if method == "Sobol Indices":
                _style_ax(ax_left, "First-Order Sobol Indices (S1)", "", "")
                _style_ax(ax_right, "Total-Order Sobol Indices (ST)", "", "")
                ax_left.barh(var_names, out["s1"], color=["#58a6ff"] * n_vars,
                             alpha=0.8, edgecolor="#30363d")
                ax_right.barh(var_names, out["st"], color=["#bc8cff"] * n_vars,
                              alpha=0.8, edgecolor="#30363d")
                ax_left.set_xlim(0, 1)
                ax_right.set_xlim(0, 1)

            elif method == "PRCC":
                _style_ax(ax_left, "PRCC — Tornado Plot", "PRCC", "")
                _style_ax(ax_right, "Scatter vs Apogee", "", "Apogee (m)")
                prcc_vals = out["prcc"]
                sorted_idx = np.argsort(np.abs(prcc_vals))
                sorted_names = [var_names[i] for i in sorted_idx]
                sorted_vals = [prcc_vals[i] for i in sorted_idx]
                colors = ["#7ee787" if v > 0 else "#f85149" for v in sorted_vals]
                ax_left.barh(sorted_names, sorted_vals, color=colors, alpha=0.8,
                             edgecolor="#30363d")
                ax_left.axvline(0, color="#484f58", linewidth=0.5)
                ax_left.set_xlim(-1, 1)
                if sorted_idx.size > 0:
                    best_j = sorted_idx[-1]
                    ax_right.scatter(X_data[:, best_j], y_data,
                                     c="#58a6ff", s=8, alpha=0.5)
                    ax_right.set_xlabel(var_names[best_j], color="#8b949e", fontsize=10)

            else:  # Morris Screening
                _style_ax(ax_left, "Morris μ* (Importance)", "μ*", "")
                _style_ax(ax_right, "Morris σ (Interaction)", "σ", "")
                ax_left.barh(var_names, out["mu_star"], color="#58a6ff", alpha=0.8,
                             edgecolor="#30363d")
                ax_right.barh(var_names, out["sigma"], color="#f0883e", alpha=0.8,
                              edgecolor="#30363d")

            for a in self._ax_sens:
                a.figure.tight_layout()
            self._canvas_sens.draw()
            self.progress_label.setText(
                f"Sensitivity analysis complete — {len(y_data)} samples, "
                f"{n_vars} variables")
        except Exception as e:
            QMessageBox.warning(self, "Sensitivity Error",
                f"Sensitivity render failed:\n\n{e}")
            logger.error(f"Sensitivity render failed: {e}", exc_info=True)
        finally:
            self.btn_run_sens.setEnabled(True)

    # ═════════════════════════════════════════════════════════════════════════
    #  TRADE STUDY
    # ═════════════════════════════════════════════════════════════════════════

    def _on_add_trade_config(self):
        """Add current rocket configuration to trade study."""
        try:
            from core.batch_simulation import BatchSimConfig, run_batch_simulation
            config = BatchSimConfig.from_rocket_state(self.engine.state)
            name = f"Config {len(self.trade_configs) + 1}"

            # Quick evaluation (5 MC runs)
            results = []
            for i in range(5):
                r = run_batch_simulation(config, seed=42 + i)
                results.append(r)

            metrics = {
                "name": name,
                "apogee": np.mean([r.apogee for r in results]),
                "max_mach": np.mean([r.max_mach for r in results]),
                "stability": np.mean([r.min_stability_margin for r in results]),
                "landing": np.mean([r.landing_distance for r in results]),
                "mass": config.dry_mass + config.propellant_mass,
                "success": np.mean([1.0 if r.success else 0.0 for r in results]) * 100,
            }

            self.trade_configs.append(metrics)
            self._update_trade_table()
            self.progress_label.setText(f"Added '{name}' to trade study")
        except Exception as e:
            QMessageBox.warning(self, "Trade Study Error", f"Failed to add configuration:\n{e}")

    def _on_add_best_trade(self):
        """Add best optimized design to trade study."""
        if not self._result or not self._result.best_design:
            QMessageBox.warning(self, "Trade Study", "No optimization result available.")
            return

        design = self._result.best_design
        mc = design.mc_stats or {}
        metrics = {
            "name": f"Optimized {len(self.trade_configs) + 1}",
            "apogee": mc.get("mean_apogee", design.objectives.get("max_apogee", 0)),
            "max_mach": mc.get("mean_mach", design.objectives.get("max_mach", 0)),
            "stability": mc.get("mean_stability", design.objectives.get("max_stability_margin", 0)),
            "landing": mc.get("mean_landing_dist", design.objectives.get("min_landing_distance", 0)),
            "mass": design.variables.get("dry_mass", 0),
            "success": mc.get("success_rate", 0) * 100,
        }
        self.trade_configs.append(metrics)
        self._update_trade_table()

    def _on_run_trade(self):
        """Generate trade study radar plot."""
        if len(self.trade_configs) < 2:
            QMessageBox.warning(self, "Trade Study",
                "Add at least 2 configurations to compare.")
            return

        ax = self._ax_trade
        ax.clear()

        # Radar plot
        categories = ["Apogee", "Stability", "Success %", "1/Landing", "1/Mach", "1/Mass"]
        n_cats = len(categories)
        angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
        angles += angles[:1]

        ax.figure.clear()
        ax_r = ax.figure.add_subplot(111, polar=True)
        ax_r.set_facecolor("#161b22")
        ax.figure.patch.set_facecolor("#0d1117")

        colors = ["#58a6ff", "#7ee787", "#f0883e", "#bc8cff", "#f85149", "#d29922"]

        for i, cfg in enumerate(self.trade_configs):
            values = [
                cfg["apogee"],
                cfg["stability"],
                cfg["success"],
                1000.0 / max(cfg["landing"], 1),
                1.0 / max(cfg["max_mach"], 0.1),
                10.0 / max(cfg["mass"], 0.1),
            ]
            # Normalize to 0-1 per axis across all configs
            values_n = []
            for j, v in enumerate(values):
                all_vals = [c[[
                    "apogee", "stability", "success", "landing", "max_mach", "mass"
                ][j]] for c in self.trade_configs]
                if j >= 3:
                    all_vals = [1.0/max(av, 0.001) if j == 3 else (
                        1.0/max(av, 0.001) if j == 4 else 10.0/max(av, 0.001))
                        for av in [c[[
                            "apogee", "stability", "success", "landing", "max_mach", "mass"
                        ][j]] for c in self.trade_configs]]
                vmin, vmax = min(all_vals) if all_vals else 0, max(all_vals) if all_vals else 1
                if vmax - vmin > 1e-9:
                    values_n.append((v - vmin) / (vmax - vmin))
                else:
                    values_n.append(0.5)

            values_n += values_n[:1]
            c = colors[i % len(colors)]
            ax_r.plot(angles, values_n, color=c, linewidth=2, label=cfg["name"])
            ax_r.fill(angles, values_n, color=c, alpha=0.1)

        ax_r.set_xticks(angles[:-1])
        ax_r.set_xticklabels(categories, color="#8b949e", fontsize=9)
        ax_r.tick_params(axis="y", colors="#484f58", labelsize=7)
        ax_r.set_ylim(0, 1)
        ax_r.grid(color="#30363d", alpha=0.3)
        ax_r.set_title("Trade Study Comparison", color="#58a6ff",
                        fontsize=13, fontweight="bold", pad=20)
        ax_r.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1),
                     facecolor="#161b22", edgecolor="#30363d",
                     labelcolor="#c9d1d9", fontsize=8)

        self._canvas_trade.draw()

    def _on_clear_trade(self):
        self.trade_configs = []
        self.trade_table.setRowCount(0)
        self._ax_trade.clear()
        _style_ax(self._ax_trade, "Trade Study", "", "")
        self._canvas_trade.draw()

    def _update_trade_table(self):
        self.trade_table.setRowCount(len(self.trade_configs))
        for row, cfg in enumerate(self.trade_configs):
            items = [
                cfg["name"],
                f"{cfg['apogee']:.1f}",
                f"{cfg['max_mach']:.3f}",
                f"{cfg['stability']:.2f}",
                f"{cfg['landing']:.0f}",
                f"{cfg['mass']:.2f}",
                f"{cfg['success']:.0f}",
            ]
            for col, text in enumerate(items):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.trade_table.setItem(row, col, item)

    # ═════════════════════════════════════════════════════════════════════════
    #  RESULTS PANEL UPDATES
    # ═════════════════════════════════════════════════════════════════════════

    def _update_results_panel(self, result):
        """Update right panel with best design from optimization result."""
        if not result.best_design:
            return
        self._update_results_panel_from_design(result.best_design)
        self._update_pareto_buttons(result)
        self._update_optimization_stats(result)
        self._update_improvement(result)

    def _update_results_panel_from_design(self, design):
        """Update results panel from a specific CandidateDesign."""
        v = design.variables
        o = design.objectives
        mc = design.mc_stats or {}

        # Parameters
        self.lbl_best_diameter.setText(f"{v.get('diameter', 0):.4f} m")
        self.lbl_best_length.setText(f"{v.get('length', 0):.3f} m")
        self.lbl_best_nose.setText(f"{v.get('nose_length', 0):.3f} m")
        self.lbl_best_fin_span.setText(f"{v.get('fin_span', 0):.4f} m")
        self.lbl_best_fin_root.setText(f"{v.get('fin_root_chord', 0):.4f} m")
        self.lbl_best_fin_tip.setText(f"{v.get('fin_tip_chord', 0):.4f} m")
        self.lbl_best_mass.setText(f"{v.get('dry_mass', 0):.3f} kg")
        self.lbl_best_motor.setText(f"{v.get('motor_designation', 'N/A')}")

        # Performance
        apogee = mc.get("mean_apogee", o.get("max_apogee", o.get("apogee", 0)))
        self.lbl_perf_apogee.setText(f"{apogee:.1f} m")
        self.lbl_perf_mach.setText(f"{mc.get('mean_mach', o.get('max_mach', 0)):.3f}")
        self.lbl_perf_stability.setText(
            f"{mc.get('mean_stability', o.get('max_stability_margin', 0)):.2f} cal"
        )
        self.lbl_perf_landing.setText(
            f"{mc.get('mean_landing_dist', o.get('min_landing_distance', 0)):.0f} m"
        )
        self.lbl_perf_rail.setText(
            f"{mc.get('mean_rail_exit', o.get('rail_exit_velocity', 0)):.1f} m/s"
        )
        self.lbl_perf_accel.setText(
            f"{mc.get('mean_accel', o.get('max_accel', 0)):.1f} m/s²"
        )

        # Reliability
        self.lbl_rel_success.setText(f"{mc.get('success_rate', 0) * 100:.1f} %")
        self.lbl_rel_p_target.setText(f"{mc.get('p_target', 0) * 100:.1f} %")
        self.lbl_rel_apogee_std.setText(f"{mc.get('std_apogee', 0):.1f} m")

        ci_low = mc.get("ci_low", 0)
        ci_high = mc.get("ci_high", 0)
        self.lbl_rel_ci.setText(f"{ci_low:.0f} – {ci_high:.0f} m")
        self.lbl_rel_skew.setText(f"{mc.get('skewness', 0):.3f}")
        self.lbl_rel_kurt.setText(f"{mc.get('kurtosis', 0):.3f}")

        # Constraint status
        self._update_constraint_status(design)

    def _update_constraint_status(self, design):
        """Update constraint status indicators."""
        # Clear old
        while self._constraint_layout.count():
            item = self._constraint_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not design.constraints_eval:
            lbl = QLabel("No constraints evaluated")
            lbl.setStyleSheet("color:#484f58; font-size:11px;")
            self._constraint_layout.addWidget(lbl)
            return

        for name, info in design.constraints_eval.items():
            satisfied = info.get("satisfied", True)
            value = info.get("value", 0)
            limit = info.get("limit", 0)
            icon = "✓" if satisfied else "✗"
            color = "#7ee787" if satisfied else "#f85149"

            lbl = QLabel(f"{icon} {name}: {value:.2f} (limit: {limit:.2f})")
            lbl.setStyleSheet(f"color:{color}; font-size:10px; padding:1px 4px;")
            self._constraint_layout.addWidget(lbl)

    def _update_pareto_buttons(self, result):
        """Update Pareto solution buttons."""
        if not result.pareto_front:
            return

        designs = result.pareto_front

        # Best Apogee
        apogees = [d.objectives.get("max_apogee", d.objectives.get("apogee", 0))
                    for d in designs]
        if apogees:
            best_idx = int(np.argmax(apogees))
            self.btn_sol_apogee.setText(f"Best Apogee: {apogees[best_idx]:.0f} m")
            self._pareto_best_apogee = designs[best_idx]

        # Best Reliability
        rel_vals = [d.mc_stats.get("success_rate", 0) if d.mc_stats else 0 for d in designs]
        if rel_vals:
            best_idx = int(np.argmax(rel_vals))
            self.btn_sol_reliability.setText(
                f"Best Reliability: {rel_vals[best_idx] * 100:.0f}%"
            )
            self._pareto_best_reliability = designs[best_idx]

        # Best Mass (lowest)
        masses = [d.variables.get("dry_mass", 999) for d in designs]
        if masses:
            best_idx = int(np.argmin(masses))
            self.btn_sol_mass.setText(f"Best Mass: {masses[best_idx]:.2f} kg")
            self._pareto_best_mass = designs[best_idx]

        # Balanced (closest to utopia) — over the user-enabled objectives only,
        # with minimize objectives sign-flipped so "utopia" is the true ideal.
        if len(designs) > 2:
            active = getattr(self, "_active_objectives", [])
            if not active:
                active = [(k, "maximize") for k in list(designs[0].objectives.keys())[:2]]
            obj_matrix = np.array([
                [(d.objectives.get(k, 0) if direction == "maximize"
                  else -d.objectives.get(k, 0)) for k, direction in active]
                for d in designs])
            if obj_matrix.shape[0] > 0:
                mins = obj_matrix.min(axis=0)
                maxs = obj_matrix.max(axis=0)
                ranges = maxs - mins
                ranges[ranges < 1e-9] = 1
                normed = (obj_matrix - mins) / ranges
                utopia = normed.max(axis=0)
                dists = np.sqrt(np.sum((normed - utopia) ** 2, axis=1))
                best_idx = int(np.argmin(dists))
                self.btn_sol_balanced.setText(
                    f"Balanced: Apogee {apogees[best_idx]:.0f} m"
                )
                self._pareto_best_balanced = designs[best_idx]

    def _update_optimization_stats(self, result):
        """Update optimization statistics."""
        self.lbl_stat_evals.setText(f"{result.total_evaluations:,}")
        self.lbl_stat_time.setText(f"{result.elapsed_time:.1f} s")
        self.lbl_stat_algo.setText(result.algorithm_used.upper())

        if result.surrogate_accuracy:
            r2 = result.surrogate_accuracy.get("r2", 0)
            self.lbl_stat_surrogate.setText(f"{r2:.3f}")
        else:
            self.lbl_stat_surrogate.setText("N/A")

    def _update_improvement(self, result):
        """Show improvement over baseline."""
        if not result.best_design:
            return

        try:
            from core.batch_simulation import BatchSimConfig, run_batch_simulation
            base = BatchSimConfig.from_rocket_state(self.engine.state)
            baseline = run_batch_simulation(base, seed=42)

            mc = result.best_design.mc_stats or {}
            opt_apogee = mc.get("mean_apogee",
                result.best_design.objectives.get("max_apogee", 0))
            opt_stab = mc.get("mean_stability",
                result.best_design.objectives.get("max_stability_margin", 0))
            opt_land = mc.get("mean_landing_dist",
                result.best_design.objectives.get("min_landing_distance", 0))
            opt_mass = result.best_design.variables.get("dry_mass", base.dry_mass)

            d_apogee = opt_apogee - baseline.apogee
            d_stab = opt_stab - baseline.min_stability_margin
            d_land = opt_land - baseline.landing_distance
            d_mass = opt_mass - base.dry_mass

            pct_a = (d_apogee / baseline.apogee * 100) if baseline.apogee > 0 else 0

            c_pos = "#7ee787"
            c_neg = "#f85149"

            self.lbl_imp_apogee.setText(f"{d_apogee:+.1f} m ({pct_a:+.1f}%)")
            self.lbl_imp_apogee.setStyleSheet(
                _VAL.replace("#e6edf3", c_pos if d_apogee > 0 else c_neg)
            )

            self.lbl_imp_stability.setText(f"{d_stab:+.2f} cal")
            self.lbl_imp_stability.setStyleSheet(
                _VAL.replace("#e6edf3", c_pos if d_stab > 0 else c_neg)
            )

            self.lbl_imp_landing.setText(f"{d_land:+.0f} m")
            self.lbl_imp_landing.setStyleSheet(
                _VAL.replace("#e6edf3", c_pos if d_land < 0 else c_neg)
            )

            self.lbl_imp_mass.setText(f"{d_mass:+.3f} kg")
            self.lbl_imp_mass.setStyleSheet(
                _VAL.replace("#e6edf3", c_pos if d_mass < 0 else c_neg)
            )
        except Exception as e:
            logger.warning(f"Could not compute improvement: {e}")

    def _load_pareto_solution(self, solution_type: str):
        """Load a Pareto solution into the results panel."""
        design = None
        if solution_type == "apogee":
            design = getattr(self, '_pareto_best_apogee', None)
        elif solution_type == "reliability":
            design = getattr(self, '_pareto_best_reliability', None)
        elif solution_type == "mass":
            design = getattr(self, '_pareto_best_mass', None)
        elif solution_type == "balanced":
            design = getattr(self, '_pareto_best_balanced', None)

        if design:
            self._update_results_panel_from_design(design)

    # ═════════════════════════════════════════════════════════════════════════
    #  EXPORT
    # ═════════════════════════════════════════════════════════════════════════

    def _on_export(self, fmt: str):
        """Export optimization results."""
        if not self._result:
            return

        if fmt == "csv":
            path, _ = QFileDialog.getSaveFileName(
                self, "Export Results", "optimization_results.csv",
                "CSV Files (*.csv)")
            if path:
                self._export_csv(path)

        elif fmt == "json":
            path, _ = QFileDialog.getSaveFileName(
                self, "Export Results", "optimization_results.json",
                "JSON Files (*.json)")
            if path:
                self._export_json(path)

    def _export_csv(self, path: str):
        """Export all designs to CSV."""
        try:
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)

                # Header
                if self._result.all_designs:
                    d0 = self._result.all_designs[0]
                    var_keys = sorted(d0.variables.keys())
                    obj_keys = sorted(d0.objectives.keys())
                    header = ["index", "fitness", "feasible"] + var_keys + obj_keys
                    writer.writerow(header)

                    for i, d in enumerate(self._result.all_designs):
                        row = [i, f"{d.fitness:.6f}", d.feasible]
                        row += [d.variables.get(k, "") for k in var_keys]
                        row += [d.objectives.get(k, "") for k in obj_keys]
                        writer.writerow(row)

            self.progress_label.setText(f"Exported {len(self._result.all_designs)} designs to CSV")
            logger.info(f"CSV export: {path}")
        except Exception as e:
            QMessageBox.warning(self, "Export Error", f"CSV export failed:\n{e}")

    def _export_json(self, path: str):
        """Export best design and Pareto front to JSON."""
        try:
            data = {
                "algorithm": self._result.algorithm_used,
                "total_evaluations": self._result.total_evaluations,
                "elapsed_time": self._result.elapsed_time,
                "best_design": {
                    "variables": self._result.best_design.variables,
                    "objectives": self._result.best_design.objectives,
                    "fitness": self._result.best_design.fitness,
                    "feasible": self._result.best_design.feasible,
                    "mc_stats": self._result.best_design.mc_stats,
                },
                "pareto_front": [
                    {
                        "variables": d.variables,
                        "objectives": d.objectives,
                        "fitness": d.fitness,
                    }
                    for d in (self._result.pareto_front or [])
                ],
                "generation_history": self._result.generation_history,
            }

            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=str)

            self.progress_label.setText("Exported to JSON")
            logger.info(f"JSON export: {path}")
        except Exception as e:
            QMessageBox.warning(self, "Export Error", f"JSON export failed:\n{e}")

    # ═════════════════════════════════════════════════════════════════════════
    #  HELPERS
    # ═════════════════════════════════════════════════════════════════════════

    def _toggle_all_vars(self, checked: bool):
        """Toggle all variable checkboxes."""
        for chk, _, _, _ in self._var_widgets.values():
            chk.setChecked(checked)

    def _on_state_changed(self, state):
        """Update variable bounds from current rocket state."""
        pass  # Variables use fixed bounds, state is read when optimization starts
