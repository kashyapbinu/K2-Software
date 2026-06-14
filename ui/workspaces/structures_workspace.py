"""
K2 AeroSim — Structures Workspace (Full Rebuild)
====================================================
3-panel layout matching CFD workspace: config | visualization | results.
Material selection, stress analysis, FEM, modal, thermal, buckling.
"""
import logging, math, threading
from pathlib import Path
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QGroupBox, QFormLayout, QLabel,
    QComboBox, QDoubleSpinBox, QSplitter, QFrame, QScrollArea,
    QPushButton, QProgressBar, QSpinBox, QTabWidget, QSlider, QGridLayout,
    QSizePolicy,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from physics.structures import MATERIALS, compute_all, compute_for_condition
from structures.solvers.base import STRUCTURAL_MATERIALS, LoadCase
from structures import workstation as wks
from ui.icons import icon

logger = logging.getLogger("K2.StructWS")

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
_BTN_S = """
QPushButton { background:#21262d; color:#c9d1d9; font-weight:500;
  border:1px solid #30363d; border-radius:6px; padding:7px 14px; }
QPushButton:hover { background:#30363d; border-color:#8b949e; }
QPushButton:disabled { color:#484f58; }
"""
_VAL = ("color:#e6edf3; font-family:'Cascadia Code',monospace; font-size:13px;"
        "font-weight:600; padding:2px 6px; background:#161b22; border-radius:4px;")

def _vl(t="—"):
    l = QLabel(t); l.setStyleSheet(_VAL); return l


class AnalysisThread(QThread):
    progress = pyqtSignal(str, float)
    finished = pyqtSignal(object, str)  # (result, type)
    errored = pyqtSignal(str)

    def __init__(self, func, args, result_type="static"):
        super().__init__()
        self._func = func
        self._args = args
        self._type = result_type

    def run(self):
        try:
            result = self._func(*self._args)
            self.finished.emit(result, self._type)
        except Exception as e:
            self.errored.emit(str(e))


class StructuresWorkspace(QWidget):
    def __init__(self, engine, parent=None):
        super().__init__(parent)
        self.engine = engine
        self._fem_result = None
        self._modal_result = None
        self._thermal_result = None
        self._thread = None
        self._wks_report = None
        self._last_bc = None
        self._flight_loads = None
        self._has_run = False  # graphs/tabs stay empty until a Run Analysis click
        self._setup_ui()
        self.engine.state_changed.connect(self._refresh)
        self._center_tabs.currentChanged.connect(self._on_tab_changed)
        self._refresh_flight_indicator()
        # Auto-import real flight loads if a simulation already exists
        if self._flight_loads and self._flight_loads.available:
            self._import_flight_loads()
        self._refresh()
        self._update_q()

    def _setup_ui(self):
        root = QHBoxLayout(self); root.setContentsMargins(0,0,0,0)
        sp = QSplitter(Qt.Orientation.Horizontal)
        sp.addWidget(self._build_left())
        sp.addWidget(self._build_center())
        sp.addWidget(self._build_right())
        sp.setSizes([320, 760, 340]); sp.setStretchFactor(1, 1)
        root.addWidget(sp)

    # ── LEFT: Configuration ──────────────────────────────────────────────────
    def _build_left(self):
        sc = QScrollArea(); sc.setWidgetResizable(True); sc.setMaximumWidth(340)
        sc.setFrameShape(QFrame.Shape.NoFrame)
        w = QWidget(); lay = QVBoxLayout(w); lay.setContentsMargins(12,14,12,14); lay.setSpacing(12)

        t = QLabel("Structural Analysis")
        t.setStyleSheet("color:#58a6ff;font-size:15px;font-weight:700;padding:2px 0 6px 0;")
        lay.addWidget(t)

        # Material
        g1 = QGroupBox("Material"); g1.setStyleSheet(_GRP)
        f1 = QFormLayout(); f1.setSpacing(8)
        self.mat_combo = QComboBox()
        for name in STRUCTURAL_MATERIALS: self.mat_combo.addItem(name)
        self.mat_combo.currentTextChanged.connect(self._on_mat)
        f1.addRow("Material:", self.mat_combo)
        self.lbl_yield = _vl(); f1.addRow("Yield:", self.lbl_yield)
        self.lbl_E = _vl(); f1.addRow("Young's Mod:", self.lbl_E)
        self.lbl_G = _vl(); f1.addRow("Shear Mod:", self.lbl_G)
        self.lbl_dens = _vl(); f1.addRow("Density:", self.lbl_dens)
        self.lbl_cte = _vl(); f1.addRow("CTE:", self.lbl_cte)
        g1.setLayout(f1); lay.addWidget(g1)

        # Geometry
        g2 = QGroupBox("Geometry"); g2.setStyleSheet(_GRP)
        f2 = QFormLayout(); f2.setSpacing(8)
        self.thick_spin = QDoubleSpinBox()
        self.thick_spin.setRange(0.0005, 0.05); self.thick_spin.setValue(0.002)
        self.thick_spin.setDecimals(4); self.thick_spin.setSuffix(" m"); self.thick_spin.setSingleStep(0.0005)
        self.thick_spin.valueChanged.connect(self._on_thick)
        f2.addRow("Wall Thickness:", self.thick_spin)
        g2.setLayout(f2); lay.addWidget(g2)

        # Load Case
        g3 = QGroupBox("Load Case"); g3.setStyleSheet(_GRP)
        f3 = QFormLayout(); f3.setSpacing(8)
        self.lc_combo = QComboBox()
        self.lc_combo.addItems(["Max Thrust", "Max-Q", "Recovery Shock", "Thermal", "Custom"])
        self.lc_combo.currentTextChanged.connect(self._on_condition_changed)
        f3.addRow("Condition:", self.lc_combo)
        self.sp_force = QDoubleSpinBox(); self.sp_force.setRange(0,50000); self.sp_force.setValue(500)
        self.sp_force.setSuffix(" N"); f3.addRow("Axial Force:", self.sp_force)
        self.sp_press = QDoubleSpinBox(); self.sp_press.setRange(0,1e7); self.sp_press.setValue(0)
        self.sp_press.setSuffix(" Pa"); f3.addRow("Int. Pressure:", self.sp_press)
        
        # Flight conditions for aero loads / thermal
        self.sp_mach = QDoubleSpinBox(); self.sp_mach.setRange(0, 10); self.sp_mach.setValue(0.80)
        self.sp_mach.setDecimals(2); f3.addRow("Flight Mach:", self.sp_mach)
        self.sp_alt = QDoubleSpinBox(); self.sp_alt.setRange(0, 100000); self.sp_alt.setValue(3000.0)
        self.sp_alt.setSuffix(" m"); f3.addRow("Altitude:", self.sp_alt)
        
        from PyQt6.QtWidgets import QCheckBox
        self.chk_cfd_map = QCheckBox("Map Pressure from CFD")
        self.chk_cfd_map.setChecked(False)
        self.chk_cfd_map.setStyleSheet("color:#c9d1d9; font-weight:600;")
        f3.addRow("", self.chk_cfd_map)

        self.lbl_q = _vl(); f3.addRow("Dyn. Pressure:", self.lbl_q)
        self.lbl_lf = _vl(); f3.addRow("Load Factor:", self.lbl_lf)

        self.sp_dT = QDoubleSpinBox(); self.sp_dT.setRange(-200,500); self.sp_dT.setValue(0)
        self.sp_dT.setSuffix(" K"); f3.addRow("Static ΔT:", self.sp_dT)

        self.sp_mach.valueChanged.connect(self._update_q)
        self.sp_alt.valueChanged.connect(self._update_q)
        self.chk_cfd_map.stateChanged.connect(self._update_q)

        g3.setLayout(f3); lay.addWidget(g3)

        # Mesh
        g4 = QGroupBox("FEM Settings"); g4.setStyleSheet(_GRP)
        f4 = QFormLayout(); f4.setSpacing(8)
        self.ref_combo = QComboBox()
        self.ref_combo.addItems(["Coarse", "Medium", "Fine", "Very Fine", "Ultra Fine", "Custom…"])
        self.ref_combo.setCurrentIndex(1)
        self.ref_combo.currentIndexChanged.connect(self._on_fem_ref_changed)
        f4.addRow("Refinement:", self.ref_combo)

        # Custom FEM mesh controls (visible when "Custom…" selected)
        self._fem_custom_widget = QWidget()
        fcl = QFormLayout(self._fem_custom_widget)
        fcl.setSpacing(6)
        fcl.setContentsMargins(0, 4, 0, 0)

        self._sp_circum = QSpinBox()
        self._sp_circum.setRange(8, 120)
        self._sp_circum.setValue(24)
        self._sp_circum.setSuffix(" divisions")
        fcl.addRow("Circumferential:", self._sp_circum)

        self._sp_axial_cal = QSpinBox()
        self._sp_axial_cal.setRange(2, 128)
        self._sp_axial_cal.setValue(8)
        self._sp_axial_cal.setSuffix(" / caliber")
        fcl.addRow("Axial Density:", self._sp_axial_cal)

        self._lbl_fem_warn = QLabel("")
        self._lbl_fem_warn.setStyleSheet(
            "color:#d29922; font-size:11px; font-weight:600; padding:2px 0;"
        )
        self._lbl_fem_warn.setWordWrap(True)
        self._lbl_fem_warn.setVisible(False)
        fcl.addRow("", self._lbl_fem_warn)

        self._fem_custom_widget.setVisible(False)
        f4.addRow(self._fem_custom_widget)

        # Warning for preset ultra-fine
        self._lbl_fem_preset_warn = QLabel("")
        self._lbl_fem_preset_warn.setStyleSheet(
            "color:#d29922; font-size:11px; font-weight:600; padding:2px 0;"
        )
        self._lbl_fem_preset_warn.setWordWrap(True)
        self._lbl_fem_preset_warn.setVisible(False)
        f4.addRow(self._lbl_fem_preset_warn)

        g4.setLayout(f4); lay.addWidget(g4)

        # ── Flight Load Import ──
        gfl = QGroupBox("Flight Loads"); gfl.setStyleSheet(_GRP)
        ffl2 = QVBoxLayout(); ffl2.setSpacing(6)
        self.lbl_flight_src = QLabel("○ No simulation data")
        self.lbl_flight_src.setStyleSheet("color:#8b949e;font-size:11px;font-weight:600;")
        ffl2.addWidget(self.lbl_flight_src)
        fl_form = QFormLayout(); fl_form.setSpacing(4)
        self.lbl_fl_v = _vl(); fl_form.addRow("Max Velocity:", self.lbl_fl_v)
        self.lbl_fl_m = _vl(); fl_form.addRow("Max Mach:", self.lbl_fl_m)
        self.lbl_fl_a = _vl(); fl_form.addRow("Max Accel:", self.lbl_fl_a)
        self.lbl_fl_q = _vl(); fl_form.addRow("Max-Q:", self.lbl_fl_q)
        ffl2.addLayout(fl_form)
        self.btn_import_loads = QPushButton(icon("import"), "Import Last Simulation")
        self.btn_import_loads.setStyleSheet(_BTN_S)
        self.btn_import_loads.clicked.connect(self._import_flight_loads)
        ffl2.addWidget(self.btn_import_loads)
        gfl.setLayout(ffl2); lay.addWidget(gfl)

        # ── Worst-case search ──
        self.btn_worst = QPushButton(icon("search"), "Find Worst Structural Condition")
        self.btn_worst.setStyleSheet(_BTN_S)
        self.btn_worst.clicked.connect(self._on_worst_case)
        lay.addWidget(self.btn_worst)
        self.lbl_worst = QLabel("")
        self.lbl_worst.setWordWrap(True)
        self.lbl_worst.setStyleSheet(
            "color:#c9d1d9;background:#161b22;border:1px solid #21262d;"
            "border-radius:6px;padding:8px;font-size:11px;")
        self.lbl_worst.setVisible(False)
        lay.addWidget(self.lbl_worst)

        # Buttons
        self.btn_static = QPushButton(icon("static", color="#fff"), "Run Static Analysis"); self.btn_static.setStyleSheet(_BTN_P)
        self.btn_static.clicked.connect(self._run_static); lay.addWidget(self.btn_static)
        self.btn_modal = QPushButton(icon("modal"), "Run Modal Analysis"); self.btn_modal.setStyleSheet(_BTN_S)
        self.btn_modal.clicked.connect(self._run_modal); lay.addWidget(self.btn_modal)
        self.btn_thermal = QPushButton(icon("thermal"), "Run Thermal Analysis"); self.btn_thermal.setStyleSheet(_BTN_S)
        self.btn_thermal.clicked.connect(self._run_thermal); lay.addWidget(self.btn_thermal)
        self.btn_report = QPushButton(icon("report"), "Export PDF Report"); self.btn_report.setStyleSheet(_BTN_S)
        self.btn_report.clicked.connect(self._export_report); lay.addWidget(self.btn_report)

        self._progress = QProgressBar(); self._progress.setRange(0,0); self._progress.setVisible(False)
        self._progress.setFixedHeight(6)
        self._progress.setStyleSheet("QProgressBar{background:#21262d;border-radius:3px;border:none;}"
                                      "QProgressBar::chunk{background:#1f6feb;border-radius:3px;}")
        lay.addWidget(self._progress)
        lay.addStretch()
        sc.setWidget(w); return sc

    # ── CENTER: Visualization ────────────────────────────────────────────────
    def _build_center(self):
        w = QWidget(); lay = QVBoxLayout(w); lay.setContentsMargins(0,0,0,0); lay.setSpacing(0)
        bar = QWidget(); bar.setStyleSheet("background:#161b22; border-bottom:1px solid #21262d;")
        bar.setFixedHeight(44); bl = QHBoxLayout(bar); bl.setContentsMargins(12,0,8,0)
        lbl = QLabel("Structural Visualization"); lbl.setStyleSheet("color:#58a6ff;font-weight:700;font-size:13px;")
        bl.addWidget(lbl); bl.addStretch()
        lay.addWidget(bar)

        # Stress profile chart area
        self._center_tabs = QTabWidget()
        self._center_tabs.setDocumentMode(True)

        # ── 3D Stress tab (interactive ANSYS-style viewer) ──
        viz_w = QWidget(); vzl = QVBoxLayout(viz_w); vzl.setContentsMargins(0,0,0,0)
        try:
            from ui.widgets.stress_viewer import StressViewer
            self._stress3d = StressViewer()
        except Exception as e:
            logger.error(f"StressViewer load failed: {e}")
            self._stress3d = QLabel(f"3D viewer unavailable: {e}")
        vzl.addWidget(self._stress3d)
        self._center_tabs.addTab(viz_w, icon("stress3d"), "3D Stress")

        # Stress profile tab
        stress_w = QWidget(); sl = QVBoxLayout(stress_w); sl.setContentsMargins(8,8,8,8)
        try:
            from ui.widgets.plot_widget import PlotWidget
            self._stress_plot = PlotWidget(title="", xlabel="Position (m)", ylabel="Von Mises Stress (MPa)")
            self._stress_plot.setMinimumHeight(300)
        except Exception:
            self._stress_plot = QLabel("Plot widget not available")
        sl.addWidget(self._stress_plot)
        self._center_tabs.addTab(stress_w, icon("stress_profile"), "Stress Profile")

        # Modal tab
        modal_w = QWidget(); ml = QVBoxLayout(modal_w); ml.setContentsMargins(8,8,8,8)

        # Mode selection dropdown
        mode_hdr = QHBoxLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.setMinimumWidth(280)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_select)
        mode_hdr.addWidget(QLabel("Active Mode:"))
        mode_hdr.addWidget(self.mode_combo, 1)
        ml.addLayout(mode_hdr)

        # Mode detail info bar
        self._mode_info_bar = QLabel("Run modal analysis to see mode shapes")
        self._mode_info_bar.setStyleSheet(
            "background:#161b22; color:#8b949e; padding:6px 10px; "
            "border-radius:6px; font-size:11px; border:1px solid #21262d;"
        )
        self._mode_info_bar.setWordWrap(True)
        ml.addWidget(self._mode_info_bar)

        try:
            from ui.widgets.mode_shape_viewer import ModeShapeViewer
            self._modal_plot = ModeShapeViewer()
            self._modal_plot.setMinimumHeight(300)
        except Exception as e:
            logger.error(f"Failed to load ModeShapeViewer: {e}")
            self._modal_plot = QLabel(f"Plot widget not available: {e}")
        ml.addWidget(self._modal_plot)
        self._center_tabs.addTab(modal_w, icon("modal"), "Modal Analysis")

        # Thermal tab
        therm_w = QWidget(); tl = QVBoxLayout(therm_w); tl.setContentsMargins(8,8,8,8)
        try:
            from ui.widgets.plot_widget import PlotWidget
            self._temp_plot = PlotWidget(title="", xlabel="Position (m)", ylabel="Wall Temperature (K)")
            self._temp_plot.setMinimumHeight(300)
        except Exception:
            self._temp_plot = QLabel("Plot widget not available")
        tl.addWidget(self._temp_plot)
        self._center_tabs.addTab(therm_w, icon("temperature"), "Temperature")

        # ── New workstation tabs ──
        self._center_tabs.addTab(self._build_deformation_tab(), icon("deformation"), "Deformation")
        self._center_tabs.addTab(self._build_fin_tab(), icon("fin"), "Fin Analysis")
        self._center_tabs.addTab(self._build_recovery_tab(), icon("recovery"), "Recovery Loads")
        self._center_tabs.addTab(self._build_buckling_tab(), icon("buckling"), "Buckling")
        self._center_tabs.addTab(self._build_loadpath_tab(), icon("loadpath"), "Load Paths")
        self._center_tabs.addTab(self._build_failuremap_tab(), icon("failuremap"), "Failure Map")
        self._center_tabs.addTab(self._build_mass_tab(), icon("mass"), "Mass Efficiency")

        lay.addWidget(self._center_tabs, 1)

        self._status = QLabel("Configure material and load case, then run analysis.")
        self._status.setStyleSheet("color:#8b949e;padding:5px 12px;font-size:11px;"
                                    "background:#161b22;border-top:1px solid #21262d;")
        self._status.setFixedHeight(28); lay.addWidget(self._status)
        return w

    # ── Tab helpers ──────────────────────────────────────────────────────────
    def _placeholder(self, title, instr):
        ph = QWidget(); v = QVBoxLayout(ph)
        v.setAlignment(Qt.AlignmentFlag.AlignCenter)
        t = QLabel(title); t.setAlignment(Qt.AlignmentFlag.AlignCenter)
        t.setStyleSheet("color:#6e7681;font-size:16px;font-weight:700;")
        s = QLabel(instr); s.setAlignment(Qt.AlignmentFlag.AlignCenter)
        s.setStyleSheet("color:#484f58;font-size:12px;padding-top:8px;")
        s.setWordWrap(True)
        v.addWidget(t); v.addWidget(s)
        return ph

    def _stacked(self, placeholder):
        """Return (QStackedWidget, content_widget). Page0=placeholder, page1=content."""
        from PyQt6.QtWidgets import QStackedWidget
        st = QStackedWidget()
        st.addWidget(placeholder)
        content = QWidget()
        st.addWidget(content)
        st.setCurrentIndex(0)
        return st, content

    def _metric(self, form, label, big=False):
        l = _vl()
        if big:
            l.setStyleSheet(_VAL + "font-size:18px;")
        form.addRow(label, l)
        return l

    def _new_plot(self, ylabel, xlabel="Position (m)"):
        try:
            from ui.widgets.plot_widget import PlotWidget
            p = PlotWidget(title="", xlabel=xlabel, ylabel=ylabel)
            p.setMinimumHeight(260)
            return p
        except Exception:
            return QLabel("Plot unavailable")

    # ── Deformation tab ──────────────────────────────────────────────────────
    def _build_deformation_tab(self):
        ph = self._placeholder("No Results Available",
            "Run Static Analysis to view structural deformation.\n"
            "Undeformed (wireframe) vs deformed (contour) with exaggeration.")
        st, content = self._stacked(ph)
        v = QVBoxLayout(content); v.setContentsMargins(8,8,8,8); v.setSpacing(6)

        # Exaggeration control
        ctl = QHBoxLayout()
        ctl.addWidget(QLabel("Exaggeration:"))
        self._exag_combo = QComboBox()
        self._exag_combo.addItems(["1×", "10×", "50×", "100×"])
        self._exag_combo.setCurrentIndex(1)
        self._exag_combo.currentIndexChanged.connect(self._apply_exaggeration)
        ctl.addWidget(self._exag_combo)
        ctl.addStretch()
        self._defl_summary = QLabel("—")
        self._defl_summary.setStyleSheet("color:#c9d1d9;font-weight:600;")
        ctl.addWidget(self._defl_summary)
        v.addLayout(ctl)

        try:
            from ui.widgets.deformation_viewer import DeformationViewer
            self._defo_view = DeformationViewer()
        except Exception as e:
            logger.error(f"DeformationViewer load failed: {e}")
            self._defo_view = QLabel(f"Viewer unavailable: {e}")
        v.addWidget(self._defo_view, 1)
        self._tab_deform = st
        return st

    # ── Fin Analysis tab ─────────────────────────────────────────────────────
    def _build_fin_tab(self):
        ph = self._placeholder("No Results Available",
            "Run Static Analysis to evaluate fin root stress, tip deflection,\n"
            "natural frequency and flutter margin.")
        st, content = self._stacked(ph)
        v = QVBoxLayout(content); v.setContentsMargins(8,8,8,8); v.setSpacing(8)
        g = QGroupBox("Fin Structural Results"); g.setStyleSheet(_GRP)
        f = QFormLayout(); f.setSpacing(7)
        self.lbl_fin_bend = self._metric(f, "Root Bending:")
        self.lbl_fin_shear = self._metric(f, "Root Shear:")
        self.lbl_fin_defl = self._metric(f, "Tip Deflection:")
        self.lbl_fin_freq = self._metric(f, "Natural Frequency:")
        self.lbl_fin_flutter = self._metric(f, "Flutter Speed:")
        self.lbl_fin_margin = self._metric(f, "Flutter Margin:")
        self.lbl_fin_force = self._metric(f, "Fin Normal Force:")
        self.lbl_fin_loaded = self._metric(f, "Highest-Loaded Fin:")
        self.lbl_fin_sf = self._metric(f, "Safety Factor:", big=True)
        g.setLayout(f); v.addWidget(g)
        self._fin_plot = self._new_plot("Deflection (mm)", "Span Fraction")
        v.addWidget(self._fin_plot, 1)
        self._tab_fin = st
        return st

    # ── Recovery Loads tab ───────────────────────────────────────────────────
    def _build_recovery_tab(self):
        ph = self._placeholder("No Results Available",
            "Recovery loads compute automatically from chute config + flight data.\n"
            "Run Static Analysis to populate deployment shock loads.")
        st, content = self._stacked(ph)
        v = QVBoxLayout(content); v.setContentsMargins(8,8,8,8); v.setSpacing(8)
        g = QGroupBox("Recovery Deployment Loads"); g.setStyleSheet(_GRP)
        f = QFormLayout(); f.setSpacing(7)
        self.lbl_rec_drogue = self._metric(f, "Drogue Deployment Shock:")
        self.lbl_rec_main = self._metric(f, "Main Deployment Shock:")
        self.lbl_rec_harness = self._metric(f, "Harness Tension:")
        self.lbl_rec_nose = self._metric(f, "Nose Cone Separation:")
        self.lbl_rec_bulk = self._metric(f, "Bulkhead Load:")
        self.lbl_rec_eye = self._metric(f, "Eye Bolt Load:")
        self.lbl_rec_peak = self._metric(f, "Peak Deployment Force:")
        self.lbl_rec_sf = self._metric(f, "Recovery Safety Factor:", big=True)
        self.lbl_rec_status = QLabel("—")
        self.lbl_rec_status.setStyleSheet("font-weight:700;font-size:15px;padding:6px;")
        f.addRow("Status:", self.lbl_rec_status)
        g.setLayout(f); v.addWidget(g)
        self._rec_plot = self._new_plot("Force (N)", "")
        v.addWidget(self._rec_plot, 1)
        self._tab_recovery = st
        return st

    # ── Buckling tab ─────────────────────────────────────────────────────────
    def _build_buckling_tab(self):
        ph = self._placeholder("No Results Available",
            "Run Static Analysis to compute Euler, shell, panel and\n"
            "local crippling buckling margins (Applied vs Critical).")
        st, content = self._stacked(ph)
        v = QVBoxLayout(content); v.setContentsMargins(8,8,8,8); v.setSpacing(8)
        g = QGroupBox("Buckling Modes"); g.setStyleSheet(_GRP)
        f = QFormLayout(); f.setSpacing(7)
        self.lbl_buck_euler = self._metric(f, "Euler Column:")
        self.lbl_buck_shell = self._metric(f, "Shell Buckling:")
        self.lbl_buck_panel = self._metric(f, "Panel Buckling:")
        self.lbl_buck_crippling = self._metric(f, "Local Crippling:")
        self.lbl_buck_applied = self._metric(f, "Applied Axial Load:")
        self.lbl_buck_gov = self._metric(f, "Governing Margin:", big=True)
        self.lbl_buck_status = QLabel("—")
        self.lbl_buck_status.setStyleSheet("font-weight:700;font-size:15px;padding:6px;")
        f.addRow("Status:", self.lbl_buck_status)
        g.setLayout(f); v.addWidget(g)
        self._buck_plot = self._new_plot("Load / Stress (norm.)", "")
        v.addWidget(self._buck_plot, 1)
        self._tab_buckling = st
        return st

    # ── Load Paths tab ───────────────────────────────────────────────────────
    def _build_loadpath_tab(self):
        ph = self._placeholder("No Results Available",
            "Run Static Analysis to trace the compressive load path from\n"
            "the motor up through the airframe to the nose cone.")
        st, content = self._stacked(ph)
        v = QVBoxLayout(content); v.setContentsMargins(8,8,8,8)
        self._loadpath_plot = self._new_plot("", "")
        v.addWidget(self._loadpath_plot, 1)
        self._tab_loadpath = st
        return st

    # ── Failure Map tab ──────────────────────────────────────────────────────
    def _build_failuremap_tab(self):
        ph = self._placeholder("No Results Available",
            "Run Static Analysis to populate the subsystem failure map.\n"
            "Green = Safe · Yellow = Margin<1.5 · Orange = Margin<1.2 · Red = Failure")
        st, content = self._stacked(ph)
        v = QVBoxLayout(content); v.setContentsMargins(12,12,12,12); v.setSpacing(8)
        self._fail_grid = QGridLayout(); self._fail_grid.setSpacing(8)
        v.addLayout(self._fail_grid)
        self._fail_detail = QLabel("Click a component for detail.")
        self._fail_detail.setWordWrap(True)
        self._fail_detail.setStyleSheet(
            "color:#c9d1d9;background:#161b22;border:1px solid #21262d;"
            "border-radius:6px;padding:10px;font-size:12px;")
        v.addWidget(self._fail_detail)
        v.addStretch()
        self._fail_buttons = {}
        self._tab_failure = st
        return st

    # ── Mass Efficiency tab ──────────────────────────────────────────────────
    def _build_mass_tab(self):
        ph = self._placeholder("No Results Available",
            "Run Static Analysis to compare current structural mass against\n"
            "the minimum mass required to meet the target safety factor.")
        st, content = self._stacked(ph)
        v = QVBoxLayout(content); v.setContentsMargins(8,8,8,8); v.setSpacing(8)
        g = QGroupBox("Mass Efficiency"); g.setStyleSheet(_GRP)
        f = QFormLayout(); f.setSpacing(8)
        self.lbl_mass_cur = self._metric(f, "Current Structural Mass:")
        self.lbl_mass_req = self._metric(f, "Minimum Required Mass:")
        self.lbl_mass_over = self._metric(f, "Overbuilt:")
        self.lbl_mass_eff = self._metric(f, "Efficiency:", big=True)
        self.lbl_mass_opt = QLabel("—")
        self.lbl_mass_opt.setStyleSheet("font-weight:700;font-size:15px;padding:6px;")
        f.addRow("Optimization Potential:", self.lbl_mass_opt)
        g.setLayout(f); v.addWidget(g)
        v.addStretch()
        self._tab_mass = st
        return st

    # ── RIGHT: Results ───────────────────────────────────────────────────────
    def _build_right(self):
        sc = QScrollArea(); sc.setWidgetResizable(True); sc.setMaximumWidth(360)
        sc.setFrameShape(QFrame.Shape.NoFrame)
        w = QWidget(); lay = QVBoxLayout(w); lay.setContentsMargins(12,14,12,14); lay.setSpacing(12)
        t = QLabel("Results"); t.setStyleSheet("color:#58a6ff;font-size:15px;font-weight:700;padding:2px 0 6px 0;")
        lay.addWidget(t)

        # ── Structural Safety Assessment ──
        gsc = QGroupBox("Structural Safety Assessment"); gsc.setStyleSheet(_GRP)
        vsc = QVBoxLayout(); vsc.setSpacing(4)
        self.lbl_score = QLabel("RUN ANALYSIS")
        self.lbl_score.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_score.setStyleSheet("font-weight:800;font-size:16px;padding:10px;"
                                     "border-radius:6px;background:#161b22;color:#484f58;")
        vsc.addWidget(self.lbl_score)
        _struct_note = QLabel(
            "Note: This assessment is an estimated qualitative evaluation based on "
            "simulation outputs and predefined heuristics. It is intended for "
            "informational and comparative purposes only and should not be used for "
            "engineering certification, safety-critical decisions, or as a substitute "
            "for professional structural analysis and validation.")
        _struct_note.setWordWrap(True)
        _struct_note.setStyleSheet("color:#6e7681;font-size:9px;padding:4px 2px 2px 2px;")
        vsc.addWidget(_struct_note)
        gsc.setLayout(vsc); lay.addWidget(gsc)

        # ── Physics Consistency Checks ──
        gpc = QGroupBox("Physics Checks"); gpc.setStyleSheet(_GRP)
        vpc = QVBoxLayout(); vpc.setSpacing(4)
        self.lbl_warnings = QLabel("Run analysis to validate results.")
        self.lbl_warnings.setWordWrap(True)
        self.lbl_warnings.setStyleSheet("color:#8b949e;font-size:11px;padding:2px;")
        self.lbl_warnings.setTextFormat(Qt.TextFormat.RichText)
        vpc.addWidget(self.lbl_warnings)
        gpc.setLayout(vpc); lay.addWidget(gpc)

        # ── Airframe Modes (analytical beam) ──
        gam = QGroupBox("Airframe Modes (Analytical)"); gam.setStyleSheet(_GRP)
        fam = QFormLayout(); fam.setSpacing(6)
        self.lbl_modal_f1 = _vl(); fam.addRow("Mode 1 (bending):", self.lbl_modal_f1)
        self.lbl_modal_f2 = _vl(); fam.addRow("Mode 2:", self.lbl_modal_f2)
        self.lbl_modal_f3 = _vl(); fam.addRow("Mode 3:", self.lbl_modal_f3)
        self.lbl_modal_warn = QLabel("")
        self.lbl_modal_warn.setWordWrap(True)
        self.lbl_modal_warn.setStyleSheet("color:#8b949e;font-size:10px;padding:2px;")
        fam.addRow(self.lbl_modal_warn)
        gam.setLayout(fam); lay.addWidget(gam)

        # Stress
        gs = QGroupBox("Stress Analysis"); gs.setStyleSheet(_GRP)
        fs = QFormLayout(); fs.setSpacing(8)
        self.lbl_axial = _vl(); fs.addRow("Axial:", self.lbl_axial)
        self.lbl_hoop = _vl(); fs.addRow("Hoop:", self.lbl_hoop)
        self.lbl_bend = _vl(); fs.addRow("Bending:", self.lbl_bend)
        self.lbl_shear = _vl(); fs.addRow("Shear:", self.lbl_shear)
        self.lbl_therm = _vl(); fs.addRow("Thermal:", self.lbl_therm)
        self.lbl_vm = _vl(); fs.addRow("Von Mises:", self.lbl_vm)
        gs.setLayout(fs); lay.addWidget(gs)

        # Buckling
        gb = QGroupBox("Buckling"); gb.setStyleSheet(_GRP)
        fb = QFormLayout(); fb.setSpacing(8)
        self.lbl_buck = _vl(); fb.addRow("Euler Crit. Load:", self.lbl_buck)
        self.lbl_shell = _vl(); fb.addRow("Shell Buck. σ:", self.lbl_shell)
        gb.setLayout(fb); lay.addWidget(gb)

        # Safety
        gf = QGroupBox("Safety Assessment"); gf.setStyleSheet(_GRP)
        ff = QFormLayout(); ff.setSpacing(8)
        self.lbl_sf = _vl(); ff.addRow("Safety Factor:", self.lbl_sf)
        self.lbl_mos = _vl(); ff.addRow("Margin of Safety:", self.lbl_mos)
        self.lbl_util = _vl(); ff.addRow("Yield Utilization:", self.lbl_util)
        self.lbl_verdict = QLabel("—")
        self.lbl_verdict.setStyleSheet("font-weight:700;font-size:16px;padding:8px;")
        ff.addRow("Verdict:", self.lbl_verdict)
        gf.setLayout(ff); lay.addWidget(gf)

        # Modal results (expanded)
        gm = QGroupBox("Modal Analysis"); gm.setStyleSheet(_GRP)
        fm = QFormLayout(); fm.setSpacing(5)
        self.lbl_modes = []
        for i in range(10):
            l = _vl()
            l.setStyleSheet(l.styleSheet() + "font-size:11px;")
            fm.addRow(f"Mode {i+1}:", l)
            self.lbl_modes.append(l)
        gm.setLayout(fm); lay.addWidget(gm)

        # Resonance Assessment
        gr = QGroupBox("Resonance Assessment"); gr.setStyleSheet(_GRP)
        fr = QFormLayout(); fr.setSpacing(6)
        self.lbl_motor_1p = _vl(); fr.addRow("Motor 1P:", self.lbl_motor_1p)
        self.lbl_motor_2p = _vl(); fr.addRow("Motor 2P:", self.lbl_motor_2p)
        self.lbl_aero_buff = _vl(); fr.addRow("Aero Buffet:", self.lbl_aero_buff)
        self.lbl_resonance_status = QLabel("—")
        self.lbl_resonance_status.setWordWrap(True)
        self.lbl_resonance_status.setStyleSheet("color:#8b949e;font-size:11px;padding:4px;")
        fr.addRow("Warnings:", self.lbl_resonance_status)
        gr.setLayout(fr); lay.addWidget(gr)

        # Flutter Assessment (preliminary)
        gfl = QGroupBox("Fin Flutter (Preliminary)"); gfl.setStyleSheet(_GRP)
        ffl = QFormLayout(); ffl.setSpacing(6)
        self.lbl_flutter_speed = _vl(); ffl.addRow("V_flutter:", self.lbl_flutter_speed)
        self.lbl_flutter_margin = _vl(); ffl.addRow("Margin:", self.lbl_flutter_margin)
        self.lbl_flutter_verdict = QLabel("—")
        self.lbl_flutter_verdict.setStyleSheet("font-weight:600;font-size:12px;padding:4px;")
        ffl.addRow("Verdict:", self.lbl_flutter_verdict)
        self.lbl_flutter_method = QLabel("NACA empirical")
        self.lbl_flutter_method.setStyleSheet("color:#6e7681;font-size:10px;font-style:italic;")
        ffl.addRow("", self.lbl_flutter_method)
        gfl.setLayout(ffl); lay.addWidget(gfl)

        # Damping
        gd = QGroupBox("Structural Damping"); gd.setStyleSheet(_GRP)
        fd = QFormLayout(); fd.setSpacing(6)
        self.lbl_damping_source = _vl(); fd.addRow("Source:", self.lbl_damping_source)
        self.lbl_damping_range = _vl(); fd.addRow("ζ Range:", self.lbl_damping_range)
        gd.setLayout(fd); lay.addWidget(gd)

        # Thermal results
        gt = QGroupBox("Thermal"); gt.setStyleSheet(_GRP)
        ft = QFormLayout(); ft.setSpacing(6)
        self.lbl_tmax = _vl(); ft.addRow("Max Wall Temp:", self.lbl_tmax)
        self.lbl_tstag = _vl(); ft.addRow("Stagnation T:", self.lbl_tstag)
        self.lbl_tsig = _vl(); ft.addRow("Thermal Stress:", self.lbl_tsig)
        self.lbl_tlimit = _vl(); ft.addRow("Service Limit:", self.lbl_tlimit)
        gt.setLayout(ft); lay.addWidget(gt)

        lay.addStretch()
        sc.setWidget(w); return sc

    # ── Material/Geometry callbacks ───────────────────────────────────────────
    def _on_mat(self, name):
        mat = MATERIALS.get(name, MATERIALS["Aluminum 6061-T6"])
        self.engine.update(material_name=name, yield_strength=mat["yield"],
            elastic_modulus=mat["E"], material_density=mat["density"])
        self._update_mat_display(name)
        self._refresh()
        self._update_q()

    def _update_q(self):
        """Update dynamic pressure and load factor display.
        
        When 'Map Pressure from CFD' is checked AND CFD results have been
        injected into the engine, use the high-fidelity CFD values.
        Otherwise fall back to ISA-based estimation from Mach/Altitude.
        """
        s = self.engine.state

        # Use CFD data if available and checkbox is checked
        if self.chk_cfd_map.isChecked() and getattr(s, 'cfd_converged', False):
            q = s.cfd_dynamic_pressure
            self.lbl_q.setText(f"{q/1000:.1f} kPa  (CFD)")

            # Load factor: n = F_axial / (m * g)
            mass = s.total_mass() if callable(getattr(s, 'total_mass', None)) else 5.0
            if not isinstance(mass, (int, float)) or mass <= 0:
                mass = 5.0
            f_axial = s.cfd_force_axial
            load_g = f_axial / (mass * 9.81) if mass > 0 else 0.0
            self.lbl_lf.setText(f"{load_g:.1f} G  (CFD)")

            # Also sync Mach/Alt spinboxes to match CFD conditions
            if s.cfd_mach > 0:
                self.sp_mach.blockSignals(True)
                self.sp_mach.setValue(s.cfd_mach)
                self.sp_mach.blockSignals(False)
            return

        # Fallback: compute from ISA + Mach/Altitude spinboxes
        try:
            from cfd.solvers.base import isa_conditions
            import math
            mach = self.sp_mach.value()
            alt = self.sp_alt.value()
            P, T, rho = isa_conditions(alt)
            a = math.sqrt(1.4 * 287.05 * T)
            V = mach * a
            q = 0.5 * rho * V**2
            
            self.lbl_q.setText(f"{q/1000:.1f} kPa")

            # Load factor estimate from aero force / weight
            mass = s.total_mass() if callable(getattr(s, 'total_mass', None)) else 5.0
            if not isinstance(mass, (int, float)) or mass <= 0:
                mass = 5.0
            cd = s.cfd_cd if getattr(s, 'cfd_converged', False) else s.cd
            ref_area = math.pi * (s.diameter / 2)**2 if s.diameter > 0 else 0.01
            drag_force = q * cd * ref_area
            load_g = drag_force / (mass * 9.81) if mass > 0 else 0.0
            self.lbl_lf.setText(f"{load_g:.1f} G")
        except Exception as e:
            logger.error(f"Failed to update q: {e}")

    def _on_thick(self, val):
        self.engine.update(wall_thickness=val)
        self._refresh()

    def _on_fem_ref_changed(self, idx):
        """Show/hide custom FEM mesh controls based on refinement selection."""
        is_custom = (idx == 5)
        self._fem_custom_widget.setVisible(is_custom)

        if idx == 3:  # Very Fine
            self._lbl_fem_preset_warn.setText("Very Fine FEM mesh — slower solve times")
            self._lbl_fem_preset_warn.setVisible(True)
        elif idx == 4:  # Ultra Fine
            self._lbl_fem_preset_warn.setText("Ultra Fine FEM — may require significant memory and solve time")
            self._lbl_fem_preset_warn.setVisible(True)
        else:
            self._lbl_fem_preset_warn.setVisible(False)

        # Update custom controls when switching to custom
        if is_custom:
            circ = self._sp_circum.value()
            axial = self._sp_axial_cal.value()
            est = circ * axial * 4  # rough element estimate per caliber
            if est > 5000:
                self._lbl_fem_warn.setText(
                    f"High mesh density ({circ}×{axial}) — large models may be slow"
                )
                self._lbl_fem_warn.setVisible(True)
            else:
                self._lbl_fem_warn.setVisible(False)

    def _get_fem_custom_params(self):
        """Return (refinement_str, custom_circum, custom_axial_per_cal) from UI."""
        ref_map = {
            0: "coarse", 1: "medium", 2: "fine",
            3: "very_fine", 4: "ultra_fine", 5: "custom",
        }
        idx = self.ref_combo.currentIndex()
        refinement = ref_map.get(idx, "medium")
        custom_circum = None
        custom_axial = None
        if idx == 5:  # Custom
            custom_circum = self._sp_circum.value()
            custom_axial = self._sp_axial_cal.value()
        return refinement, custom_circum, custom_axial

    def _update_mat_display(self, name=None):
        name = name or self.mat_combo.currentText()
        from structures.solvers.base import STRUCTURAL_MATERIALS
        mat = STRUCTURAL_MATERIALS.get(name)
        if mat:
            self.lbl_yield.setText(f"{mat.yield_strength/1e6:.0f} MPa")
            self.lbl_E.setText(f"{mat.E/1e9:.1f} GPa")
            self.lbl_G.setText(f"{mat.G/1e9:.1f} GPa")
            self.lbl_dens.setText(f"{mat.density:.0f} kg/m³")
            self.lbl_cte.setText(f"{mat.cte*1e6:.1f} µm/m·K")

    # Representative flight point per load case. Each condition is dominated by a
    # different physics regime, so it must be evaluated at its own Mach/altitude
    # — otherwise (e.g.) Thermal at the transonic Max-Q Mach badly under-predicts
    # aero heating. Values are presets; the user can still edit them afterward.
    _CONDITION_FLIGHT = {
        "Max Thrust":     (0.25, 500.0),     # liftoff / early boost (thrust-dominated)
        "Max-Q":          (1.00, 8000.0),    # transonic peak dynamic pressure
        "Recovery Shock": (0.00, 500.0),     # subsonic descent (Mach unused here)
        "Thermal":        (3.00, 12000.0),   # sustained supersonic aero heating
    }

    def _on_condition_changed(self, condition):
        """Set the condition's flight point — from the last simulation when one
        exists (Max-Q point for Max-Q, peak-Mach point for Thermal, real peak
        thrust for Max Thrust), otherwise a representative preset — then
        re-analyse."""
        self._refresh_flight_indicator()
        fl = self._flight_loads
        use_sim = fl is not None and getattr(fl, "available", False)

        mach = alt = None
        if condition == "Max-Q" and use_sim and fl.maxq_mach > 0:
            mach, alt = fl.maxq_mach, fl.maxq_altitude
        elif condition == "Thermal" and use_sim and fl.max_mach > 0:
            # Peak aero heating ≈ peak Mach; use its altitude when captured.
            mach, alt = fl.max_mach, (fl.maxmach_altitude or fl.maxq_altitude)
        elif condition == "Max Thrust" and use_sim and fl.max_thrust > 0:
            # Peak thrust is early in the burn (low altitude, near the pad) —
            # use the actual altitude/Mach at that instant, not a fixed preset.
            mach, alt = fl.maxthrust_mach, fl.maxthrust_altitude
        # Recovery is axial-dominated (Mach-insensitive) → preset.

        if mach is None:
            fp = self._CONDITION_FLIGHT.get(condition)
            if fp:
                mach, alt = fp
        if mach is not None:
            # alt may legitimately be ~0 (peak thrust on the pad), so set it
            # directly rather than treating 0 as "no data".
            if alt is None:
                alt = self.sp_alt.value()
            for sp, val in ((self.sp_mach, mach), (self.sp_alt, alt)):
                sp.blockSignals(True)
                sp.setValue(val)
                sp.blockSignals(False)

        # Use the real peak thrust for the thrust-driven case when available.
        if condition == "Max Thrust" and use_sim and fl.max_thrust > 0:
            self.sp_force.setValue(fl.max_thrust)

        self._update_q()
        self._refresh()

    # ── Quick analytical refresh ─────────────────────────────────────────────
    def _refresh(self, state=None):
        s = state if state and not isinstance(state, str) else self.engine.state
        self._update_mat_display()
        condition = self.lc_combo.currentText()
        force = max(abs(s.net_force), abs(s.thrust), s.weight, self.sp_force.value())
        mach = self.sp_mach.value()
        alt = self.sp_alt.value()

        # AoA depends on condition
        aoa = 3.0 if condition == "Max-Q" else 2.0

        logger.info(f"_refresh: cond={condition}, d={s.diameter:.4f}, wt={s.wall_thickness:.4f}, "
                     f"L={s.length:.3f}, F={force:.1f}, M={mach}, alt={alt}, mat={s.material_name}")

        # Use condition-specific physics
        if condition in ("Max Thrust", "Max-Q", "Recovery Shock", "Thermal"):
            mass = s.total_mass() if callable(getattr(s, 'total_mass', None)) else getattr(s, 'total_mass', 5.0)
            arm = abs(getattr(s, "cp", 0.0) - getattr(s, "cg", 0.0))
            r = compute_for_condition(
                condition, s.diameter, s.wall_thickness, s.length, s.material_name,
                force=force, internal_pressure=self.sp_press.value(),
                delta_T=self.sp_dT.value(), mach=mach, altitude_m=alt,
                vehicle_mass_kg=mass,
                angle_of_attack_deg=aoa, moment_arm_m=arm,
            )
        else:
            r = compute_all(force, s.diameter, s.wall_thickness, s.length, s.material_name,
                            internal_pressure=self.sp_press.value(), delta_T=self.sp_dT.value())

        self.lbl_axial.setText(f"{r['axial']/1e6:.2f} MPa")
        self.lbl_hoop.setText(f"{r['hoop']/1e6:.2f} MPa")
        self.lbl_bend.setText(f"{r['bending']/1e6:.2f} MPa")
        self.lbl_shear.setText(f"{r['shear']/1e6:.2f} MPa")
        self.lbl_therm.setText(f"{r['thermal']/1e6:.2f} MPa")
        self.lbl_vm.setText(f"{r['von_mises']/1e6:.2f} MPa")
        self.lbl_buck.setText(f"{r['buckling']:.0f} N")
        self.lbl_shell.setText(f"{r['shell_buckling_stress']/1e6:.1f} MPa")
        sf = r["safety_factor"]
        self.lbl_sf.setText("∞" if sf > 1e6 else f"{sf:.2f}")
        mos = r["margin_of_safety"]
        self.lbl_mos.setText("∞" if mos > 1e6 else f"{mos:+.2f}")
        self.lbl_util.setText(f"{r['yield_utilization']*100:.1f}%")
        if sf >= 3.0:
            self.lbl_verdict.setText("✓ SAFE"); self.lbl_verdict.setStyleSheet("color:#7ee787;font-weight:700;font-size:16px;padding:8px;")
        elif sf >= 1.5:
            self.lbl_verdict.setText("ADEQUATE"); self.lbl_verdict.setStyleSheet("color:#d29922;font-weight:700;font-size:16px;padding:8px;")
        elif sf >= 1.0:
            self.lbl_verdict.setText("MARGINAL"); self.lbl_verdict.setStyleSheet("color:#f0883e;font-weight:700;font-size:16px;padding:8px;")
        else:
            self.lbl_verdict.setText("✕ FAILURE"); self.lbl_verdict.setStyleSheet("color:#f85149;font-weight:700;font-size:16px;padding:8px;")

        # Update status bar with condition name
        self._status.setText(f"Condition: {condition} — σ_vm={r['von_mises']/1e6:.1f} MPa, SF={sf:.2f}")

        # Store body condition for 3D viewer + run workstation suite
        self._last_bc = r
        self._run_workstation(body_condition=r)

    # ── FEM Analysis Runs ────────────────────────────────────────────────────
    def _get_assembly(self):
        # Try to get the assembly from the design workspace
        main = self.window()
        if hasattr(main, 'design_ws'):
            return getattr(main.design_ws, 'assembly', None)
        return None

    def _run_static(self):
        assembly = self._get_assembly()
        if not assembly:
            self._status.setText("No rocket assembly — import or design a rocket first.")
            return
        from structures.fem_interface import FEMInterface

        # Build condition-specific LoadCase
        condition = self.lc_combo.currentText()
        force = self.sp_force.value()
        mach = self.sp_mach.value()
        alt = self.sp_alt.value()

        if condition == "Max Thrust":
            lc = LoadCase.max_thrust(
                thrust=force, accel_g=5.0,
                internal_pressure=self.sp_press.value(),
                angle_of_attack_deg=2.0,
                mach=mach, altitude_m=alt,
            )
        elif condition == "Max-Q":
            try:
                from cfd.solvers.base import isa_conditions
                P, T, rho = isa_conditions(alt)
                a_s = math.sqrt(1.4 * 287.05 * T)
                V = mach * a_s
                q_dyn = 0.5 * rho * V ** 2
            except Exception:
                q_dyn = 50000.0
            lc = LoadCase.max_q(thrust=force, q_dyn=q_dyn, mach=mach, alt=alt, aoa=3.0)
        elif condition == "Recovery Shock":
            # Estimate vehicle mass from engine state
            s = self.engine.state
            mass = s.total_mass() if callable(getattr(s, 'total_mass', None)) else getattr(s, 'total_mass', 5.0)
            if not isinstance(mass, (int, float)) or mass <= 0:
                mass = 5.0
            lc = LoadCase.recovery(vehicle_mass_kg=mass, shock_g=15.0, daf=1.8, kt=2.5)
        elif condition == "Thermal":
            lc = LoadCase.thermal(mach=mach, alt=alt)
        else:
            # Custom
            lc = LoadCase(
                name="Custom",
                axial_force=force,
                internal_pressure=self.sp_press.value(),
                delta_T=self.sp_dT.value(),
                acceleration_g=5.0,
            )

        refinement, custom_circum, custom_axial = self._get_fem_custom_params()
        fem = FEMInterface()
        self._progress.setVisible(True); self.btn_static.setEnabled(False)

        cfd_path = None
        if self.chk_cfd_map.isChecked():
            from pathlib import Path
            s = self.engine.state

            # Override LoadCase with real CFD data if available
            if getattr(s, 'cfd_converged', False):
                lc.dynamic_pressure = s.cfd_dynamic_pressure
                lc.mach = s.cfd_mach if s.cfd_mach > 0 else lc.mach
                # Use CFD axial force if it's larger (more conservative)
                if s.cfd_force_axial > 0:
                    lc.axial_force = max(lc.axial_force, s.cfd_force_axial)
                if s.cfd_force_normal > 0:
                    lc.lateral_force = s.cfd_force_normal
                self._status.setText(
                    f"CFD mapped: q={s.cfd_dynamic_pressure/1000:.1f} kPa, "
                    f"Cd={s.cfd_cd:.4f}, F_ax={s.cfd_force_axial:.1f} N"
                )
            else:
                self._status.setText("CFD map enabled but no CFD results injected yet.")

            # Surface VTK for pressure-field mapping: prefer the path recorded
            # by the CFD run; fall back to the default work dir for older
            # sessions that predate cfd_surface_vtk in the state.
            candidates = []
            if getattr(s, 'cfd_surface_vtk', ''):
                candidates.append(Path(s.cfd_surface_vtk))
            from cfd.solvers.base import CFDConfig
            work = CFDConfig().work_dir
            candidates += [work / "surface_flow.vtu", work / "surface_flow.vtk"]
            for p in candidates:
                if p.is_file():
                    cfd_path = p
                    self._status.setText(self._status.text() + f" | VTK: {p.name}")
                    break

        self._thread = AnalysisThread(
            fem.analyze,
            (assembly, lc, self.mat_combo.currentText(), refinement, "static", cfd_path,
             custom_circum, custom_axial),
            "static"
        )
        self._thread.finished.connect(self._on_result)
        self._thread.errored.connect(self._on_error)
        self._thread.start()

    def _run_modal(self):
        assembly = self._get_assembly()
        if not assembly:
            self._status.setText("No rocket assembly available."); return
        from structures.fem_interface import FEMInterface
        refinement, custom_circum, custom_axial = self._get_fem_custom_params()
        fem = FEMInterface()
        self._progress.setVisible(True); self.btn_modal.setEnabled(False)
        self._thread = AnalysisThread(fem.modal_analysis,
            (assembly, self.mat_combo.currentText(), 10, refinement,
             custom_circum, custom_axial), "modal")
        self._thread.finished.connect(self._on_result)
        self._thread.errored.connect(self._on_error)
        self._thread.start()

    def _run_thermal(self):
        assembly = self._get_assembly()
        if not assembly:
            self._status.setText("No rocket assembly available."); return
        from structures.fem_interface import FEMInterface
        fem = FEMInterface()
        
        mach = self.sp_mach.value()
        alt = self.sp_alt.value()
        
        self._progress.setVisible(True); self.btn_thermal.setEnabled(False)
        self._thread = AnalysisThread(fem.thermal_analysis,
            (assembly, mach, alt, self.mat_combo.currentText()), "thermal")
        self._thread.finished.connect(self._on_result)
        self._thread.errored.connect(self._on_error)
        self._thread.start()

    def _on_result(self, result, rtype):
        self._progress.setVisible(False)
        self.btn_static.setEnabled(True); self.btn_modal.setEnabled(True); self.btn_thermal.setEnabled(True)

        if rtype == "static":
            self._has_run = True  # unlock workstation tabs/graphs
            self._fem_result = result
            self.lbl_vm.setText(f"{result.max_von_mises/1e6:.2f} MPa")
            self.lbl_axial.setText(f"{result.max_axial_stress/1e6:.2f} MPa")
            self.lbl_hoop.setText(f"{result.max_hoop_stress/1e6:.2f} MPa")
            self.lbl_bend.setText(f"{result.max_bending_stress/1e6:.2f} MPa")
            self.lbl_shear.setText(f"{result.max_shear_stress/1e6:.2f} MPa")
            self.lbl_therm.setText(f"{result.max_thermal_stress/1e6:.2f} MPa")
            sf = result.safety_factor
            self.lbl_sf.setText("∞" if sf > 1e6 else f"{sf:.2f}")
            self.lbl_mos.setText(f"{result.margin_of_safety:+.2f}" if result.margin_of_safety < 1e6 else "∞")
            self.lbl_util.setText(f"{result.yield_utilization*100:.1f}%")

            # Update verdict
            if sf >= 3.0:
                self.lbl_verdict.setText("✓ SAFE"); self.lbl_verdict.setStyleSheet("color:#7ee787;font-weight:700;font-size:16px;padding:8px;")
            elif sf >= 1.5:
                self.lbl_verdict.setText("ADEQUATE"); self.lbl_verdict.setStyleSheet("color:#d29922;font-weight:700;font-size:16px;padding:8px;")
            elif sf >= 1.0:
                self.lbl_verdict.setText("MARGINAL"); self.lbl_verdict.setStyleSheet("color:#f0883e;font-weight:700;font-size:16px;padding:8px;")
            else:
                self.lbl_verdict.setText("✕ FAILURE"); self.lbl_verdict.setStyleSheet("color:#f85149;font-weight:700;font-size:16px;padding:8px;")

            # Choose plot color and label based on condition
            lc_name = getattr(result, 'load_case_name', '') or self.lc_combo.currentText()
            color_map = {"Max Thrust": "#f0883e", "Recovery Shock": "#f85149", "Thermal": "#da3633"}
            plot_color = color_map.get(lc_name, "#f0883e")
            ylabel = "Thermal Stress (MPa)" if lc_name in ("Thermal", "Aerodynamic Heating") else "Von Mises Stress (MPa)"

            if result.element_stresses and hasattr(self._stress_plot, 'update_plot'):
                xs = [p[0] for p in result.element_stresses]
                ys = [p[1]/1e6 for p in result.element_stresses]
                self._stress_plot.update_plot(xs, ys, f"{lc_name} Stress", "Position (m)", ylabel, plot_color)

            # If thermal case produced temperature data, also update thermal plot
            if hasattr(result, 'station_temperatures') and result.station_temperatures:
                if hasattr(self._temp_plot, 'update_plot'):
                    txs = [p[0] for p in result.station_temperatures]
                    tys = [p[1] for p in result.station_temperatures]
                    self._temp_plot.update_plot(txs, tys, "Wall Temperature", "Position (m)", "T (K)", "#da3633")

            # Feed FEM peak stresses into the 3D stress viewer + workstation suite
            self._last_bc = {
                "axial": result.max_axial_stress, "hoop": result.max_hoop_stress,
                "bending": result.max_bending_stress, "shear": result.max_shear_stress,
                "thermal": result.max_thermal_stress, "von_mises": result.max_von_mises,
            }
            self._run_workstation(body_condition=self._last_bc)
            self._update_stress3d()

            self._center_tabs.setCurrentIndex(0)
            self._status.setText(f"{lc_name} analysis — \u03c3_vm={result.max_von_mises/1e6:.1f} MPa, SF={sf:.2f}")

        elif rtype == "modal":
            self._modal_result = result

            # Populate mode frequency labels with classification and eff. mass
            for i, lbl in enumerate(self.lbl_modes):
                if i < len(result.frequencies_hz):
                    freq = result.frequencies_hz[i]
                    desc = result.descriptions[i] if i < len(result.descriptions) else f"Mode {i+1}"
                    # Add effective modal mass if available
                    eff_str = ""
                    if i < len(result.effective_modal_mass):
                        em = result.effective_modal_mass[i]
                        dominant = max(em.values()) if em else 0
                        if dominant > 0:
                            eff_str = f" [{dominant:.0f}%]"
                    lbl.setText(f"{freq:.1f} Hz — {desc}{eff_str}")
                else:
                    lbl.setText("—")

            # Resonance assessment
            if result.motor_1p_hz > 0:
                self.lbl_motor_1p.setText(f"{result.motor_1p_hz:.0f} Hz")
                self.lbl_motor_2p.setText(f"{result.motor_2p_hz:.0f} Hz")
            buff = result.aero_buffet_band
            if buff[0] > 0:
                self.lbl_aero_buff.setText(f"{buff[0]:.0f}–{buff[1]:.0f} Hz")

            # Resonance warnings
            if result.resonance_warnings:
                warn_text = "\n".join(result.resonance_warnings)
                self.lbl_resonance_status.setText(warn_text)
                self.lbl_resonance_status.setStyleSheet(
                    "color:#d29922;font-size:11px;padding:4px;font-weight:600;"
                )
            else:
                self.lbl_resonance_status.setText("✓ No resonance concerns")
                self.lbl_resonance_status.setStyleSheet(
                    "color:#7ee787;font-size:11px;padding:4px;font-weight:600;"
                )

            # Flutter assessment
            fa = result.flutter_assessment
            if fa.get("critical_speed_m_s", 0) > 0:
                self.lbl_flutter_speed.setText(f"{fa['critical_speed_m_s']:.0f} m/s")
                margin = fa.get('flutter_margin', 0)
                self.lbl_flutter_margin.setText(f"{margin:.2f}×")
                verdict = fa.get('verdict', '—')
                self.lbl_flutter_verdict.setText(verdict)
                if '✓' in verdict or 'SAFE' in verdict:
                    self.lbl_flutter_verdict.setStyleSheet(
                        "color:#7ee787;font-weight:700;font-size:12px;padding:4px;"
                    )
                elif 'ADEQUATE' in verdict:
                    self.lbl_flutter_verdict.setStyleSheet(
                        "color:#d29922;font-weight:700;font-size:12px;padding:4px;"
                    )
                elif 'MARGINAL' in verdict:
                    self.lbl_flutter_verdict.setStyleSheet(
                        "color:#f0883e;font-weight:700;font-size:12px;padding:4px;"
                    )
                else:
                    self.lbl_flutter_verdict.setStyleSheet(
                        "color:#f85149;font-weight:700;font-size:12px;padding:4px;"
                    )
                self.lbl_flutter_method.setText(
                    f"{fa.get('method', 'NACA')} | AR={fa.get('fin_AR', '?')}, t/c={fa.get('fin_t_c', '?')}"
                )

            # Damping
            if result.damping_ratios:
                self.lbl_damping_source.setText(result.damping_source or "—")
                zmin = min(result.damping_ratios) * 100
                zmax = max(result.damping_ratios) * 100
                self.lbl_damping_range.setText(f"{zmin:.2f}% – {zmax:.2f}%")

            # Update ModeShapeViewer
            if hasattr(self, '_modal_plot') and hasattr(self._modal_plot, 'load_mesh'):
                import pathlib
                inp_path = pathlib.Path("fem_run/modal/structure_mesh.inp")
                if inp_path.is_file():
                    self._modal_plot.load_mesh(str(inp_path))

                self.mode_combo.clear()
                if result.frequencies_hz:
                    for i, freq in enumerate(result.frequencies_hz):
                        desc = result.descriptions[i] if i < len(result.descriptions) else f"Mode {i+1}"
                        cls_tag = ""
                        if i < len(result.mode_classifications):
                            cls_tag = f" [{result.mode_classifications[i]}]"
                        self.mode_combo.addItem(
                            f"Mode {i+1}: {freq:.1f} Hz — {desc}{cls_tag}",
                            userData=i
                        )
                    if result.mode_shapes:
                        desc0 = result.descriptions[0] if result.descriptions else "Mode 1"
                        freq0 = result.frequencies_hz[0]
                        self._modal_plot.set_mode_shape(
                            result.mode_shapes[0],
                            freq_hz=freq0,
                            description=desc0,
                            mode_index=1
                        )

            # Update mode info bar
            if hasattr(self, '_mode_info_bar') and result.frequencies_hz:
                info_parts = []
                info_parts.append(f"{result.num_modes} modes computed")
                if result.total_mass_kg > 0:
                    info_parts.append(f"Total mass: {result.total_mass_kg:.3f} kg")
                if result.damping_source:
                    info_parts.append(f"Damping: {result.damping_source}")
                self._mode_info_bar.setText(" │ ".join(info_parts))

            self._center_tabs.setCurrentIndex(1)
            status_parts = [f"Modal: {result.num_modes} modes"]
            if result.frequencies_hz:
                status_parts.append(f"f₁={result.frequencies_hz[0]:.1f} Hz")
            if result.resonance_warnings:
                status_parts.append(f"{len(result.resonance_warnings)} resonance warning(s)")
            fa = result.flutter_assessment
            if fa.get('flutter_margin', 0) > 0:
                status_parts.append(f"Flutter margin: {fa['flutter_margin']:.2f}×")
            self._status.setText(" — ".join(status_parts))

        elif rtype == "thermal":
            self._thermal_result = result
            self.lbl_tmax.setText(f"{result.max_wall_temp_K:.0f} K ({result.max_wall_temp_K-273.15:.0f} °C)")
            self.lbl_tstag.setText(f"{result.stagnation_temp_K:.0f} K")
            self.lbl_tsig.setText(f"{result.max_thermal_stress/1e6:.1f} MPa")
            limit_txt = f"{result.service_temp_limit_K:.0f} K"
            if result.exceeds_service_temp:
                limit_txt += " EXCEEDED"
                self.lbl_tlimit.setStyleSheet(_VAL + "color:#f85149;")
            else:
                limit_txt += " ✓ OK"
                self.lbl_tlimit.setStyleSheet(_VAL + "color:#7ee787;")
            self.lbl_tlimit.setText(limit_txt)
            if result.station_temps and hasattr(self._temp_plot, 'update_plot'):
                xs = [p[0] for p in result.station_temps]
                ys = [p[1] for p in result.station_temps]
                self._temp_plot.update_plot(xs, ys, "Wall Temperature", "Position (m)", "T (K)", "#f0883e")
            self._center_tabs.setCurrentIndex(2)
            self._status.setText(f"Thermal: T_max={result.max_wall_temp_K:.0f} K, σ_th={result.max_thermal_stress/1e6:.1f} MPa")

    def _on_mode_select(self, index):
        if not hasattr(self, '_modal_result') or not self._modal_result:
            return
        result = self._modal_result
        if not result.mode_shapes:
            return
        idx = self.mode_combo.currentData()
        if idx is None or idx < 0 or idx >= len(result.mode_shapes):
            return
        if not hasattr(self, '_modal_plot') or not hasattr(self._modal_plot, 'set_mode_shape'):
            return

        freq = result.frequencies_hz[idx] if idx < len(result.frequencies_hz) else 0
        desc = result.descriptions[idx] if idx < len(result.descriptions) else f"Mode {idx+1}"
        self._modal_plot.set_mode_shape(
            result.mode_shapes[idx],
            freq_hz=freq,
            description=desc,
            mode_index=idx + 1
        )

        # Update mode info bar with per-mode details
        if hasattr(self, '_mode_info_bar'):
            parts = [f"f = {freq:.1f} Hz"]
            if idx < len(result.mode_classifications):
                parts.append(f"Type: {result.mode_classifications[idx]}")
            if idx < len(result.effective_modal_mass):
                em = result.effective_modal_mass[idx]
                dominant_dir = max(em, key=em.get) if em else ""
                dominant_val = em.get(dominant_dir, 0) if em else 0
                if dominant_val > 0:
                    parts.append(f"Eff. mass: {dominant_val:.1f}% ({dominant_dir})")
            if idx < len(result.damping_ratios):
                parts.append(f"ζ = {result.damping_ratios[idx]*100:.3f}%")
            if idx < len(result.participation_factors):
                pf = result.participation_factors[idx]
                dominant_pf_dir = max(pf, key=pf.get) if pf else ""
                dominant_pf = pf.get(dominant_pf_dir, 0) if pf else 0
                if dominant_pf > 0:
                    parts.append(f"Γ_{dominant_pf_dir} = {dominant_pf:.4f}")
            self._mode_info_bar.setText(" │ ".join(parts))

    def _on_error(self, msg):
        self._progress.setVisible(False)
        self.btn_static.setEnabled(True); self.btn_modal.setEnabled(True); self.btn_thermal.setEnabled(True)
        self._status.setText(f"Error: {msg}")
        logger.error(f"Analysis error: {msg}")

    # ════════════════════════════════════════════════════════════════════════
    #  WORKSTATION INTEGRATION
    # ════════════════════════════════════════════════════════════════════════
    def _get_history(self):
        main = self.window()
        se = getattr(main, "sim_engine", None)
        hist = getattr(se, "history", None) if se else None
        return hist if (hist and len(hist) > 0) else None

    def reset_workspace(self):
        """Blank all structural results + plots (called on New Project)."""
        self._fem_result = None
        self._modal_result = None
        self._thermal_result = None
        self._wks_report = None
        self._last_bc = None
        self._has_run = False
        from ui.workspace_reset import clear_visuals
        clear_visuals(self)        # clears stress/temp/fin plots + 3D viewers
        try:
            self.lbl_score.setText("RUN ANALYSIS")
            self.lbl_score.setStyleSheet("font-weight:800;font-size:16px;padding:10px;"
                                         "border-radius:6px;background:#161b22;color:#484f58;")
        except Exception:
            pass
        for lbl in ("_defl_summary",):
            w = getattr(self, lbl, None)
            if w is not None:
                try:
                    w.setText("—")
                except Exception:
                    pass
        try:
            self._refresh_flight_indicator()
        except Exception:
            pass

    def _refresh_flight_indicator(self):
        """Update the flight-load import indicator from sim history/state."""
        hist = self._get_history()
        fl = wks.FlightLoads.from_history(hist, self.engine.state)
        self._flight_loads = fl
        if fl.available and fl.source == "simulation":
            self.lbl_flight_src.setText("✓ Using Last Simulation")
            self.lbl_flight_src.setStyleSheet("color:#7ee787;font-size:11px;font-weight:700;")
        elif fl.available:
            self.lbl_flight_src.setText("◐ Using State Maxima (no sim)")
            self.lbl_flight_src.setStyleSheet("color:#d29922;font-size:11px;font-weight:600;")
        else:
            self.lbl_flight_src.setText("○ No simulation data — manual loads")
            self.lbl_flight_src.setStyleSheet("color:#8b949e;font-size:11px;font-weight:600;")
        self.lbl_fl_v.setText(f"{fl.max_velocity:.0f} m/s" if fl.max_velocity else "—")
        self.lbl_fl_m.setText(f"{fl.max_mach:.2f}" if fl.max_mach else "—")
        self.lbl_fl_a.setText(f"{fl.max_accel_g:.1f} G" if fl.max_accel_g else "—")
        self.lbl_fl_q.setText(f"{fl.max_dynamic_pressure/1000:.1f} kPa" if fl.max_dynamic_pressure else "—")

    def _import_flight_loads(self):
        """Push imported flight loads into the manual spinboxes + refresh."""
        self._refresh_flight_indicator()
        fl = self._flight_loads
        if not fl.available:
            self._status.setText("No simulation data to import — run a flight simulation first.")
            return
        self.sp_mach.blockSignals(True); self.sp_mach.setValue(fl.maxq_mach or fl.max_mach)
        self.sp_mach.blockSignals(False)
        if fl.maxq_altitude > 0:
            self.sp_alt.blockSignals(True); self.sp_alt.setValue(fl.maxq_altitude)
            self.sp_alt.blockSignals(False)
        if fl.max_thrust > 0:
            self.sp_force.setValue(fl.max_thrust)
        self.lc_combo.setCurrentText("Max-Q")
        self._update_q()
        self._refresh()
        self._status.setText(
            f"Imported {fl.source}: V={fl.max_velocity:.0f} m/s, M={fl.max_mach:.2f}, "
            f"q={fl.max_dynamic_pressure/1000:.1f} kPa, n={fl.max_accel_g:.1f} G")

    def _on_worst_case(self):
        """Evaluate every major flight event and report the governing one."""
        hist = self._get_history()
        if hist is None:
            self.lbl_worst.setVisible(True)
            self.lbl_worst.setText("No flight data. Run a simulation, then search.")
            return
        self._status.setText("Searching worst-case condition…")
        res = wks.find_worst_case(self.engine.state, hist, self.mat_combo.currentText())
        if not res.available or res.critical_event is None:
            self.lbl_worst.setVisible(True)
            self.lbl_worst.setText("No events evaluated.")
            return
        c = res.critical_event
        hs = res.highest_stress_event
        hb = res.highest_buckling_event
        self.lbl_worst.setVisible(True)
        self.lbl_worst.setText(
            f"<b>Critical Event:</b> {c.name}<br>"
            f"<b>Time:</b> {c.time:.1f} s<br>"
            f"<b>Von Mises:</b> {c.von_mises/1e6:.0f} MPa<br>"
            f"<b>Safety Factor:</b> {c.safety_factor:.2f}<br>"
            f"<span style='color:#8b949e'>Peak stress: {hs.name} ({hs.von_mises/1e6:.0f} MPa) · "
            f"Peak buckling: {hb.name} ({hb.buckling_margin:.2f})</span>")
        self._status.setText(
            f"Worst case: {c.name} @ {c.time:.1f}s — σ_vm={c.von_mises/1e6:.0f} MPa, SF={c.safety_factor:.2f}")

    def _run_workstation(self, body_condition=None):
        """Run the full analytical workstation suite and populate all tabs.
        Fast (<50 ms); safe to call from _refresh."""
        # Keep all center tabs on their "No Results Available" placeholders and
        # graphs empty until the user explicitly runs an analysis.
        if not self._has_run:
            return
        try:
            s = self.engine.state
            if s.diameter <= 0 or s.length <= 0:
                return
            assembly = self._get_assembly()
            hist = self._get_history()
            mat = self.mat_combo.currentText()
            cond = self.lc_combo.currentText()
            cond = cond if cond in ("Max Thrust", "Max-Q", "Recovery Shock", "Thermal") else "Max-Q"
            rep = wks.full_analysis(s, assembly, hist, mat, cond)
            self._wks_report = rep
            if body_condition is not None:
                rep.body_condition = body_condition
            self._populate_workstation(rep)
        except Exception as e:
            logger.error(f"Workstation run failed: {e}", exc_info=True)

    def _populate_workstation(self, rep):
        self._refresh_flight_indicator()
        # ── Safety assessment (qualitative — no numeric score shown) ──
        sc = rep.score
        if sc.score >= 75:
            verdict = "✅ Good"
        elif sc.score >= 50:
            verdict = "⚠️ Marginal"
        else:
            verdict = "❌ Poor"
        self.lbl_score.setText(verdict)
        self.lbl_score.setStyleSheet(
            f"font-weight:800;font-size:16px;padding:10px;border-radius:6px;"
            f"background:#161b22;color:{sc.color};border:1px solid {sc.color}66;")

        self._populate_fin(rep.fin)
        self._populate_recovery(rep.recovery)
        self._populate_buckling(rep.buckling)
        self._populate_mass(rep.mass)
        self._populate_failuremap(rep.failure)
        self._populate_loadpath(rep.loads)
        self._populate_warnings(rep.warnings)
        self._populate_modal_estimate(rep.modal)
        self._populate_thermal_tab(rep.thermal)
        # pyvista-heavy views refreshed lazily when their tab is shown
        self._refresh_active_3d()

    def _populate_warnings(self, warnings):
        from structures.validation import severity_color
        if not warnings:
            self.lbl_warnings.setText("—")
            return
        icon = {"error": "✕", "warn": "!", "info": "✓"}
        lines = []
        for w in warnings:
            c = severity_color(w.severity)
            lines.append(f"<span style='color:{c}'>{icon.get(w.severity,'•')} {w.message}</span>")
        self.lbl_warnings.setText("<br>".join(lines))

    def _populate_modal_estimate(self, me):
        self.lbl_modal_f1.setText(f"{me.f1_hz:.0f} Hz" if me.f1_hz else "—")
        self.lbl_modal_f2.setText(f"{me.f2_hz:.0f} Hz" if me.f2_hz else "—")
        self.lbl_modal_f3.setText(f"{me.f3_hz:.0f} Hz" if me.f3_hz else "—")
        if me.low_freq:
            self.lbl_modal_f1.setStyleSheet(_VAL + "color:#d29922;")
            self.lbl_modal_warn.setText(f"{me.warning}")
            self.lbl_modal_warn.setStyleSheet("color:#d29922;font-size:10px;padding:2px;font-weight:600;")
        else:
            self.lbl_modal_f1.setStyleSheet(_VAL)
            self.lbl_modal_warn.setText(f"✓ {me.total_mass_kg:.2f} kg, cantilever beam estimate")
            self.lbl_modal_warn.setStyleSheet("color:#6e7681;font-size:10px;padding:2px;")

    def _populate_thermal_tab(self, tp):
        if hasattr(self._temp_plot, "update_plot") and tp.body_station_temps:
            xs = [x for x, _, _ in tp.body_station_temps]
            ys = [T for _, T, _ in tp.body_station_temps]
            self._temp_plot.update_plot(xs, ys, "Wall Temperature Along Body",
                                        "Position (nose=0 → tail=1)", "Temperature (K)", "#f0883e")

    # ── Per-tab populators ────────────────────────────────────────────────────
    def _populate_fin(self, fa):
        self.lbl_fin_bend.setText(f"{fa.root_bending_MPa:.1f} MPa")
        self.lbl_fin_shear.setText(f"{fa.root_shear_MPa:.1f} MPa")
        self.lbl_fin_defl.setText(f"{fa.tip_deflection_mm:.2f} mm")
        self.lbl_fin_freq.setText(f"{fa.natural_frequency_Hz:.0f} Hz")
        self.lbl_fin_flutter.setText(f"{fa.flutter_speed_m_s:.0f} m/s")
        self.lbl_fin_margin.setText("∞" if not (fa.flutter_margin < 1e6) else f"{fa.flutter_margin:.2f}×")
        self.lbl_fin_force.setText(f"{fa.fin_normal_force_N:.0f} N")
        self.lbl_fin_loaded.setText(fa.highest_loaded_fin)
        sf = fa.safety_factor
        self.lbl_fin_sf.setText("∞" if sf > 1e6 else f"{sf:.2f}")
        self.lbl_fin_sf.setStyleSheet(_VAL + f"font-size:18px;color:{fa.status_color};")
        if hasattr(self._fin_plot, "update_plot") and fa.deflection_profile:
            xs = [p[0] for p in fa.deflection_profile]
            ys = [p[1] for p in fa.deflection_profile]
            self._fin_plot.update_plot(xs, ys, "Fin Deflection", "Span Fraction", "Deflection (mm)", fa.status_color)
        self._tab_fin.setCurrentIndex(1)

    def _populate_recovery(self, rl):
        self.lbl_rec_drogue.setText(f"{rl.drogue_shock_N:.0f} N")
        self.lbl_rec_main.setText(f"{rl.main_shock_N:.0f} N")
        self.lbl_rec_harness.setText(f"{rl.harness_tension_N:.0f} N")
        self.lbl_rec_nose.setText(f"{rl.nosecone_separation_N:.0f} N")
        self.lbl_rec_bulk.setText(f"{rl.bulkhead_load_N:.0f} N")
        self.lbl_rec_eye.setText(f"{rl.eyebolt_load_N:.0f} N")
        self.lbl_rec_peak.setText(f"{rl.peak_force_N:.0f} N")
        sf = rl.safety_factor
        self.lbl_rec_sf.setText("∞" if sf > 1e6 else f"{sf:.2f}")
        self.lbl_rec_sf.setStyleSheet(_VAL + f"font-size:18px;color:{rl.status_color};")
        self.lbl_rec_status.setText(rl.status)
        self.lbl_rec_status.setStyleSheet(f"font-weight:700;font-size:15px;padding:6px;color:{rl.status_color};")
        if hasattr(self._rec_plot, "ax"):
            names = ["Drogue", "Main", "Harness", "Nose Sep", "Bulkhead", "Eyebolt"]
            vals = [rl.drogue_shock_N, rl.main_shock_N, rl.harness_tension_N,
                    rl.nosecone_separation_N, rl.bulkhead_load_N, rl.eyebolt_load_N]
            self._bar(self._rec_plot, names, vals, "Recovery Loads (N)", rl.status_color)
        self._tab_recovery.setCurrentIndex(1)

    def _populate_buckling(self, ba):
        if not ba.modes:
            return
        m = {mode.name: mode for mode in ba.modes}
        def fmt(mode):
            if mode is None: return "—"
            mu = "N" if mode.unit == "N" else "MPa"
            crit = mode.critical if mode.unit == "N" else mode.critical / 1e6
            return f"{crit:.0f} {mu}  (×{mode.margin:.1f})"
        self.lbl_buck_euler.setText(fmt(m.get("Euler Column")))
        self.lbl_buck_shell.setText(fmt(m.get("Shell Buckling")))
        self.lbl_buck_panel.setText(fmt(m.get("Panel Buckling")))
        self.lbl_buck_crippling.setText(fmt(m.get("Local Crippling")))
        self.lbl_buck_applied.setText(f"{ba.applied_axial_N:.0f} N")
        g = ba.governing
        self.lbl_buck_gov.setText(f"{g.name}: ×{g.margin:.2f}")
        self.lbl_buck_gov.setStyleSheet(_VAL + f"font-size:16px;color:{g.status_color};")
        self.lbl_buck_status.setText(ba.status)
        self.lbl_buck_status.setStyleSheet(f"font-weight:700;font-size:15px;padding:6px;color:{ba.status_color};")
        if hasattr(self._buck_plot, "ax"):
            names = [mode.name.split()[0] for mode in ba.modes]
            margins = [min(mode.margin, 10) for mode in ba.modes]
            self._bar(self._buck_plot, names, margins, "Buckling Margin (Critical/Applied)",
                      g.status_color, hline=1.0)
        self._tab_buckling.setCurrentIndex(1)

    def _populate_mass(self, me):
        self.lbl_mass_cur.setText(f"{me.current_mass_kg:.2f} kg")
        self.lbl_mass_req.setText(f"{me.required_mass_kg:.2f} kg")
        self.lbl_mass_over.setText(f"{me.overbuilt_pct:.0f} %")
        self.lbl_mass_eff.setText(f"{me.efficiency_pct:.0f} %")
        self.lbl_mass_opt.setText(me.optimization_potential)
        self.lbl_mass_opt.setStyleSheet(
            f"font-weight:700;font-size:15px;padding:6px;color:{me.potential_color};")
        self._tab_mass.setCurrentIndex(1)

    def _populate_failuremap(self, fm):
        # Clear old buttons
        while self._fail_grid.count():
            it = self._fail_grid.takeAt(0)
            wdg = it.widget()
            if wdg: wdg.deleteLater()
        self._fail_buttons = {}
        for i, comp in enumerate(fm.components):
            b = QPushButton(f"{comp.name}\nSF {comp.margin:.2f}" if comp.margin < 1e6 else f"{comp.name}\nSF ∞")
            b.setMinimumHeight(58)
            b.setStyleSheet(
                f"QPushButton{{background:{comp.color};color:#0d1117;font-weight:700;"
                f"font-size:12px;border:none;border-radius:8px;padding:6px;}}"
                f"QPushButton:hover{{border:2px solid #ffffff;}}")
            b.clicked.connect(lambda _=False, c=comp: self._on_fail_click(c))
            self._fail_grid.addWidget(b, i // 2, i % 2)
            self._fail_buttons[comp.name] = b
        if fm.weakest:
            self._fail_detail.setText(
                f"<b>Weakest component:</b> {fm.weakest.name} (SF {fm.weakest.margin:.2f}, {fm.weakest.status})"
                f"<br><span style='color:#8b949e'>Click any component for detail.</span>")
        self._tab_failure.setCurrentIndex(1)

    def _on_fail_click(self, comp):
        self._fail_detail.setText(
            f"<b style='font-size:14px;color:{comp.color}'>{comp.name} — {comp.status}</b><br>"
            f"Safety Factor: <b>{comp.margin:.2f}</b><br>"
            f"Governing check: {comp.detail}")

    def _populate_loadpath(self, lp):
        if not hasattr(self._loadpath_plot, "ax") or not lp.stations:
            return
        p = self._loadpath_plot
        p.ax.clear(); p._style_axis("Load Path — Force Flow", "", "")
        ax = p.ax
        n = len(lp.stations)
        peak = lp.peak_force_N or 1.0
        for i, st in enumerate(lp.stations):
            y = n - i
            frac = st.force_N / peak
            color = (0.95, 0.3 + 0.5*(1-frac), 0.2)
            ax.barh(y, frac, height=0.5, color=color, alpha=0.85, zorder=2)
            ax.text(0.02, y, f"{st.name}", va="center", ha="left",
                    color="#e6edf3", fontsize=9, fontweight="bold", zorder=3)
            ax.text(frac + 0.02, y, f"{st.force_N:.0f} N", va="center", ha="left",
                    color="#8b949e", fontsize=8, zorder=3)
            if i < n - 1:
                ax.annotate("", xy=(0.1, y-0.5), xytext=(0.1, y-0.0),
                            arrowprops=dict(arrowstyle="->", color="#58a6ff", lw=1.5))
        ax.set_xlim(0, 1.4); ax.set_ylim(0.2, n + 0.8)
        ax.set_yticks([]); ax.set_xticks([])
        p.figure.tight_layout(); p.canvas.draw()
        self._tab_loadpath.setCurrentIndex(1)

    def _populate_deformation(self, rep):
        s = self.engine.state
        # Max deflection: take the larger of the FEM peak nodal displacement and
        # the analytical Euler-Bernoulli lateral bend. FEM static for an axial
        # load case only captures the tiny axial shortening (~0.00 mm), which
        # would otherwise read as 0 next to the real lateral tip bend — so never
        # report a max below the analytical lateral deflection.
        beam_defl = rep.deflection.max_deflection_mm
        fem_defl = getattr(self._fem_result, "max_displacement_mm", 0.0) if self._fem_result else 0.0
        if fem_defl > beam_defl:
            max_defl = fem_defl
            loc = "FEM peak node"
        else:
            max_defl = beam_defl
            loc = rep.deflection.location
        tip_defl = rep.deflection.tip_deflection_mm or max_defl
        exag = {0: 1, 1: 10, 2: 50, 3: 100}.get(self._exag_combo.currentIndex(), 10)
        self._defl_summary.setText(
            f"Max Deflection: {max_defl:.2f} mm  ·  Tip: {tip_defl:.2f} mm  ·  "
            f"{loc}  ·  shape ×{exag} (auto-fit to view)")
        # Drive DeformationViewer (same assembly geometry as Design view)
        try:
            v = self._defo_view
            if hasattr(v, "set_deflection"):
                v.set_deflection(s, self._get_assembly(), max_defl, exaggeration=exag)
        except Exception as e:
            logger.error(f"Deformation view failed: {e}")
        self._tab_deform.setCurrentIndex(1)

    def _apply_exaggeration(self):
        # Re-render deformation with chosen scale note
        if hasattr(self, "_wks_report") and self._wks_report:
            self._populate_deformation(self._wks_report)

    def _refresh_active_3d(self):
        """Refresh whichever pyvista-heavy tab is currently visible."""
        idx = self._center_tabs.currentIndex()
        name = self._center_tabs.tabText(idx)
        if "3D Stress" in name:
            self._update_stress3d()
        elif "Deformation" in name and hasattr(self, "_wks_report") and self._wks_report:
            self._populate_deformation(self._wks_report)

    def _on_tab_changed(self, idx):
        self._refresh_active_3d()

    # ── 3D stress viewer ──────────────────────────────────────────────────────
    def _update_stress3d(self):
        if not hasattr(self, "_stress3d") or not hasattr(self._stress3d, "set_result"):
            return
        if not self._has_run:
            return
        bc = getattr(self, "_last_bc", None)
        if not bc:
            return
        from structures.solvers.base import get_structural_material
        mat = get_structural_material(self.mat_combo.currentText())
        try:
            self._stress3d.set_result(self.engine.state, self._get_assembly(),
                                      bc, mat.yield_strength)
        except Exception as e:
            logger.error(f"3D stress update failed: {e}")

    @staticmethod
    def _fig_to_image(fig):
        """Render a matplotlib figure to an RGB ndarray (or None if empty)."""
        try:
            import numpy as np
            if not fig.axes or not any(a.has_data() for a in fig.axes):
                return None
            fig.canvas.draw()
            buf = np.asarray(fig.canvas.buffer_rgba())
            return buf[:, :, :3].copy()
        except Exception:
            return None

    def _capture_stress_contours(self):
        """Capture all structural visuals for the report: 3D stress contours in
        every mode, the deformation and mode-shape views, and the fin /
        through-length stress / temperature plots."""
        images = []
        try:
            from PyQt6.QtWidgets import QApplication
        except Exception:
            QApplication = None

        # ── 3D stress contours (every mode) ──
        sv = getattr(self, "_stress3d", None)
        if sv is not None and getattr(sv, "plotter", None) is not None:
            try:
                from ui.widgets.stress_viewer import STRESS_MODES
                self._update_stress3d()
                orig = sv.mode_combo.currentText() if hasattr(sv, "mode_combo") else None
                for mode in STRESS_MODES:
                    try:
                        if hasattr(sv, "mode_combo"):
                            sv.mode_combo.setCurrentText(mode)
                        if QApplication: QApplication.processEvents()
                        sv.plotter.render()
                        img = sv.plotter.screenshot(return_img=True)
                        if img is not None:
                            images.append((f"Stress — {mode}", img))
                    except Exception as e:
                        logger.debug("stress contour %s skipped: %s", mode, e)
                if orig is not None:
                    try: sv.mode_combo.setCurrentText(orig)
                    except Exception: pass
            except Exception as e:
                logger.warning(f"Stress contour capture failed: {e}")

        # ── 3D pyvista viewers (deformation, mode shapes) ──
        for attr, caption in (("_defo_view", "Deformation"),
                              ("_modal_plot", "Mode Shape")):
            w = getattr(self, attr, None)
            plotter = getattr(w, "plotter", None)
            if plotter is not None:
                try:
                    if QApplication: QApplication.processEvents()
                    plotter.render()
                    img = plotter.screenshot(return_img=True)
                    if img is not None:
                        images.append((caption, img))
                except Exception as e:
                    logger.debug("%s screenshot skipped: %s", caption, e)

        # ── matplotlib plots (fin deflection, stress profile, temperature) ──
        for attr, caption in (("_fin_plot", "Fin Deflection"),
                              ("_stress_plot", "Von Mises along the airframe"),
                              ("_temp_plot", "Wall Temperature")):
            w = getattr(self, attr, None)
            fig = getattr(w, "figure", None)
            if fig is not None:
                img = self._fig_to_image(fig)
                if img is not None:
                    images.append((caption, img))
        return images

    def _export_report(self):
        """Export a full structural PDF report from the current analysis."""
        if not getattr(self, "_wks_report", None):
            self._run_workstation(self._last_bc)
        if not getattr(self, "_wks_report", None):
            self._status.setText("Run an analysis before exporting a report.")
            return
        from PyQt6.QtWidgets import QFileDialog
        name = getattr(self.engine.state, "name", "rocket") or "rocket"
        default = f"{name.replace(' ', '_')}_structural_report.pdf"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Structural Report", default, "PDF Files (*.pdf)")
        if not path:
            return
        try:
            from structures.report import generate_structural_report
            images = self._capture_stress_contours()
            generate_structural_report(
                path, self.engine.state, self._wks_report,
                self.mat_combo.currentText(), contour_images=images)
            self._status.setText(f"Report exported: {path}")
        except Exception as e:
            logger.error(f"Report export failed: {e}", exc_info=True)
            self._status.setText(f"Report export failed: {e}")

    def _bar(self, plot, names, vals, title, color, hline=None):
        if not hasattr(plot, "ax"):
            return
        plot.ax.clear(); plot._style_axis(title, "", "")
        ax = plot.ax
        x = range(len(names))
        ax.bar(x, vals, color=color, alpha=0.85)
        ax.set_xticks(list(x)); ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
        if hline is not None:
            ax.axhline(hline, color="#f85149", ls="--", lw=1.2)
        plot.figure.tight_layout(); plot.canvas.draw()
