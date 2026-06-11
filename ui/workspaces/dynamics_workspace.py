"""
K2 Aerospace — Dynamics Workspace
====================================
Flutter, vibration, aeroelastic + flight-envelope safety analysis UI.
3-panel layout: config | plots | engineering assessment.

Engineering features:
  - Flutter boundary with flight-trajectory overlay, safe/caution/unsafe
    shading, V markers, and margin-to-flutter verdict.
  - Frequency response with auto resonance identification (labelled peaks +
    resonance table + engineering summary).
  - Aeroelastic effectiveness with smooth control-reversal transition.
  - Mode-shape animation tab (reuses the FEM ModeShapeViewer).
  - Flight Envelope (V vs altitude) overlaying flutter + divergence boundaries
    and the actual trajectory — the primary flight-safety page.
  - Full engineering safety assessment panel + PDF/CSV/JSON report export.

All analyses run in worker threads (DynThread) so the UI never blocks.
"""
import logging, math, json, csv, datetime
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QGroupBox, QFormLayout, QLabel,
    QComboBox, QDoubleSpinBox, QSplitter, QFrame, QScrollArea,
    QPushButton, QProgressBar, QTabWidget, QTableWidget, QTableWidgetItem,
    QHeaderView, QFileDialog,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from ui.icons import icon

logger = logging.getLogger("K2.DynWS")

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

# Verdict colours
_C_SAFE = "#7ee787"
_C_CAUT = "#d29922"
_C_BAD  = "#f85149"
_C_INFO = "#58a6ff"


def _vl(t="—"):
    l = QLabel(t); l.setStyleSheet(_VAL); return l


def _verdict(margin_pct):
    """(text, colour) for a percentage safety margin."""
    if margin_pct >= 20.0:
        return "SAFE", _C_SAFE
    if margin_pct >= 10.0:
        return "CAUTION", _C_CAUT
    return "UNSAFE", _C_BAD


class DynThread(QThread):
    finished = pyqtSignal(object, str)
    errored = pyqtSignal(str)
    def __init__(self, func, args, rtype):
        super().__init__()
        self._f, self._a, self._t = func, args, rtype
    def run(self):
        try: self.finished.emit(self._f(*self._a), self._t)
        except Exception as e:
            import traceback; traceback.print_exc()
            self.errored.emit(str(e))


class DynamicsWorkspace(QWidget):
    def __init__(self, engine, parent=None):
        super().__init__(parent)
        self.engine = engine
        self._flutter_result = None
        self._vib_result = None
        self._aero_result = None
        self._modal_result = None
        self._thread = None
        self._setup_ui()

    # ===============================================================
    # UI
    # ===============================================================
    def _setup_ui(self):
        root = QHBoxLayout(self); root.setContentsMargins(0,0,0,0)
        sp = QSplitter(Qt.Orientation.Horizontal)
        sp.addWidget(self._build_left())
        sp.addWidget(self._build_center())
        sp.addWidget(self._build_right())
        sp.setSizes([320, 760, 360]); sp.setStretchFactor(1, 1)
        root.addWidget(sp)

    def _build_left(self):
        sc = QScrollArea(); sc.setWidgetResizable(True); sc.setMaximumWidth(340)
        sc.setFrameShape(QFrame.Shape.NoFrame)
        w = QWidget(); lay = QVBoxLayout(w); lay.setContentsMargins(12,14,12,14); lay.setSpacing(12)

        t = QLabel("Dynamic Analysis")
        t.setStyleSheet("color:#58a6ff;font-size:15px;font-weight:700;padding:2px 0 6px 0;")
        lay.addWidget(t)

        g1 = QGroupBox("Flight Envelope"); g1.setStyleSheet(_GRP)
        f1 = QFormLayout(); f1.setSpacing(8)
        self.sp_vmax = QDoubleSpinBox(); self.sp_vmax.setRange(10,2000); self.sp_vmax.setValue(300)
        self.sp_vmax.setSuffix(" m/s"); f1.addRow("Max Flight Speed:", self.sp_vmax)
        self.sp_mmax = QDoubleSpinBox(); self.sp_mmax.setRange(0.1,5); self.sp_mmax.setValue(1.0)
        self.sp_mmax.setDecimals(2); f1.addRow("Max Mach:", self.sp_mmax)
        self.btn_fromsim = QPushButton(icon("import"), "Use Last Simulation"); self.btn_fromsim.setStyleSheet(_BTN_S)
        self.btn_fromsim.clicked.connect(self._fill_from_sim); f1.addRow(self.btn_fromsim)
        self.lbl_simsrc = QLabel("Source: manual input")
        self.lbl_simsrc.setStyleSheet("color:#8b949e;font-size:10px;font-style:italic;")
        f1.addRow(self.lbl_simsrc)
        g1.setLayout(f1); lay.addWidget(g1)

        g2 = QGroupBox("Vibration Analysis"); g2.setStyleSheet(_GRP)
        f2 = QFormLayout(); f2.setSpacing(8)
        self.sp_damp = QDoubleSpinBox(); self.sp_damp.setRange(0.001,0.2); self.sp_damp.setValue(0.02)
        self.sp_damp.setDecimals(3); f2.addRow("Damping Ratio ζ:", self.sp_damp)
        self.sp_psd = QDoubleSpinBox(); self.sp_psd.setRange(0.001,1.0); self.sp_psd.setValue(0.04)
        self.sp_psd.setDecimals(3); self.sp_psd.setSuffix(" g²/Hz"); f2.addRow("Input PSD:", self.sp_psd)
        g2.setLayout(f2); lay.addWidget(g2)

        self.btn_all = QPushButton(icon("run", color="#fff"), "Run Full Assessment"); self.btn_all.setStyleSheet(_BTN_P)
        self.btn_all.clicked.connect(self._run_all); lay.addWidget(self.btn_all)
        self.btn_flutter = QPushButton(icon("flutter"), "Flutter"); self.btn_flutter.setStyleSheet(_BTN_S)
        self.btn_flutter.clicked.connect(self._run_flutter); lay.addWidget(self.btn_flutter)
        self.btn_vib = QPushButton(icon("vibration"), "Vibration"); self.btn_vib.setStyleSheet(_BTN_S)
        self.btn_vib.clicked.connect(self._run_vibration); lay.addWidget(self.btn_vib)
        self.btn_aero = QPushButton(icon("aeroelastic"), "Aeroelastic"); self.btn_aero.setStyleSheet(_BTN_S)
        self.btn_aero.clicked.connect(self._run_aeroelastic); lay.addWidget(self.btn_aero)
        self.btn_modal = QPushButton(icon("modal"), "Mode Shapes"); self.btn_modal.setStyleSheet(_BTN_S)
        self.btn_modal.clicked.connect(lambda: self._run_modal()); lay.addWidget(self.btn_modal)

        self._progress = QProgressBar(); self._progress.setRange(0,0); self._progress.setVisible(False)
        self._progress.setFixedHeight(6)
        self._progress.setStyleSheet("QProgressBar{background:#21262d;border-radius:3px;border:none;}"
                                      "QProgressBar::chunk{background:#1f6feb;border-radius:3px;}")
        lay.addWidget(self._progress)

        # Export
        ge = QGroupBox("Report Export"); ge.setStyleSheet(_GRP)
        fe = QVBoxLayout(); fe.setSpacing(6)
        rowx = QHBoxLayout()
        self.btn_json = QPushButton("JSON"); self.btn_json.setStyleSheet(_BTN_S); self.btn_json.clicked.connect(lambda: self._export("json"))
        self.btn_csv  = QPushButton("CSV");  self.btn_csv.setStyleSheet(_BTN_S);  self.btn_csv.clicked.connect(lambda: self._export("csv"))
        self.btn_pdf  = QPushButton("PDF");  self.btn_pdf.setStyleSheet(_BTN_S);  self.btn_pdf.clicked.connect(lambda: self._export("pdf"))
        rowx.addWidget(self.btn_json); rowx.addWidget(self.btn_csv); rowx.addWidget(self.btn_pdf)
        fe.addLayout(rowx); ge.setLayout(fe); lay.addWidget(ge)

        lay.addStretch()
        sc.setWidget(w); return sc

    def _build_center(self):
        w = QWidget(); lay = QVBoxLayout(w); lay.setContentsMargins(0,0,0,0); lay.setSpacing(0)
        bar = QWidget(); bar.setStyleSheet("background:#161b22; border-bottom:1px solid #21262d;")
        bar.setFixedHeight(44); bl = QHBoxLayout(bar); bl.setContentsMargins(12,0,8,0)
        lbl = QLabel("Dynamic Analysis Plots"); lbl.setStyleSheet("color:#58a6ff;font-weight:700;font-size:13px;")
        bl.addWidget(lbl); bl.addStretch(); lay.addWidget(bar)

        self._tabs = QTabWidget(); self._tabs.setDocumentMode(True)

        from ui.widgets.plot_widget import PlotWidget

        # Flutter boundary
        fw = QWidget(); fl = QVBoxLayout(fw); fl.setContentsMargins(8,8,8,8)
        self._flutter_plot = PlotWidget(title="", xlabel="Altitude (km)", ylabel="Speed (m/s)")
        self._flutter_plot.setMinimumHeight(300)
        fl.addWidget(self._flutter_plot)
        self._tabs.addTab(fw, "Flutter Boundary")

        # FRF + resonance table
        vw = QWidget(); vl = QVBoxLayout(vw); vl.setContentsMargins(8,8,8,8); vl.setSpacing(6)
        self._frf_plot = PlotWidget(title="", xlabel="Frequency (Hz)", ylabel="Magnitude (dB)")
        self._frf_plot.setMinimumHeight(220)
        vl.addWidget(self._frf_plot, 2)
        self._frf_summary = QLabel("Run vibration analysis to identify resonances.")
        self._frf_summary.setStyleSheet("color:#8b949e;font-size:11px;padding:2px 4px;")
        vl.addWidget(self._frf_summary)
        self._res_table = QTableWidget(0, 5)
        self._res_table.setHorizontalHeaderLabels(["Mode", "Freq (Hz)", "Mag (dB)", "Risk", "Driver (SR)"])
        self._res_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._res_table.setStyleSheet(
            "QTableWidget{background:#0d1117;color:#c9d1d9;gridline-color:#21262d;font-size:11px;}"
            "QHeaderView::section{background:#161b22;color:#8b949e;border:none;padding:4px;font-weight:600;}")
        self._res_table.setMaximumHeight(160)
        vl.addWidget(self._res_table, 1)
        self._tabs.addTab(vw, "Frequency Response")

        # Aeroelastic
        aw = QWidget(); al = QVBoxLayout(aw); al.setContentsMargins(8,8,8,8)
        self._aero_plot = PlotWidget(title="", xlabel="Mach", ylabel="Effectiveness η")
        self._aero_plot.setMinimumHeight(300)
        al.addWidget(self._aero_plot)
        self._tabs.addTab(aw, "Aeroelastic")

        # Mode Shapes (reuses FEM ModeShapeViewer)
        mw = QWidget(); ml = QVBoxLayout(mw); ml.setContentsMargins(8,8,8,8); ml.setSpacing(6)
        msel = QHBoxLayout()
        msel.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.setStyleSheet("QComboBox{background:#161b22;color:#e6edf3;border:1px solid #30363d;border-radius:4px;padding:3px 8px;}")
        self.mode_combo.currentIndexChanged.connect(self._on_mode_select)
        msel.addWidget(self.mode_combo, 1)
        ml.addLayout(msel)
        try:
            from ui.widgets.mode_shape_viewer import ModeShapeViewer
            self._modal_plot = ModeShapeViewer()
            self._modal_plot.setMinimumHeight(300)
        except Exception as e:
            logger.error(f"ModeShapeViewer load failed: {e}")
            self._modal_plot = QLabel("Mode-shape viewer unavailable")
        ml.addWidget(self._modal_plot, 1)
        self._mode_info = QLabel("Frequency: —   Damping: —   Type: —")
        self._mode_info.setStyleSheet(
            "color:#c9d1d9;font-family:'Cascadia Code',monospace;font-size:11px;"
            "background:#161b22;border:1px solid #21262d;border-radius:4px;padding:6px;")
        ml.addWidget(self._mode_info)
        self._tabs.addTab(mw, "〰 Mode Shapes")

        # Flight Envelope
        ew = QWidget(); el = QVBoxLayout(ew); el.setContentsMargins(8,8,8,8)
        self._env_plot = PlotWidget(title="", xlabel="Velocity (m/s)", ylabel="Altitude (km)")
        self._env_plot.setMinimumHeight(300)
        el.addWidget(self._env_plot)
        self._tabs.addTab(ew, "Flight Envelope")

        self._tabs.currentChanged.connect(self._on_tab_changed)
        lay.addWidget(self._tabs, 1)
        self._status = QLabel("Import a rocket design, then run dynamic analysis.")
        self._status.setStyleSheet("color:#8b949e;padding:5px 12px;font-size:11px;"
                                    "background:#161b22;border-top:1px solid #21262d;")
        self._status.setFixedHeight(28); lay.addWidget(self._status)
        return w

    def _build_right(self):
        sc = QScrollArea(); sc.setWidgetResizable(True); sc.setMaximumWidth(380)
        sc.setFrameShape(QFrame.Shape.NoFrame)
        w = QWidget(); lay = QVBoxLayout(w); lay.setContentsMargins(12,14,12,14); lay.setSpacing(10)
        t = QLabel("Flight Safety Assessment")
        t.setStyleSheet("color:#58a6ff;font-size:15px;font-weight:700;padding:2px 0 6px 0;")
        lay.addWidget(t)

        # Overall verdict banner
        self.lbl_overall = QLabel("RUN ASSESSMENT")
        self.lbl_overall.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_overall.setStyleSheet("font-weight:800;font-size:16px;padding:10px;"
                                       "border-radius:6px;background:#161b22;color:#484f58;")
        lay.addWidget(self.lbl_overall)

        # Flutter
        gf = QGroupBox("Flutter"); gf.setStyleSheet(_GRP)
        ff = QFormLayout(); ff.setSpacing(6)
        self.lbl_vmax_f = _vl();   ff.addRow("Max Velocity:", self.lbl_vmax_f)
        self.lbl_fspd = _vl();     ff.addRow("Flutter Velocity:", self.lbl_fspd)
        self.lbl_fmach = _vl();    ff.addRow("Flutter Mach:", self.lbl_fmach)
        self.lbl_fmargin = _vl();  ff.addRow("Margin:", self.lbl_fmargin)
        self.lbl_fverdict = QLabel("—"); self.lbl_fverdict.setStyleSheet("font-weight:700;font-size:14px;padding:4px;")
        ff.addRow("Status:", self.lbl_fverdict)
        gf.setLayout(ff); lay.addWidget(gf)

        # Divergence / aeroelastic
        ga = QGroupBox("Divergence / Aeroelastic"); ga.setStyleSheet(_GRP)
        fa = QFormLayout(); fa.setSpacing(6)
        self.lbl_divspd = _vl();    fa.addRow("Divergence Velocity:", self.lbl_divspd)
        self.lbl_divmach = _vl();   fa.addRow("Divergence Mach:", self.lbl_divmach)
        self.lbl_divmargin = _vl(); fa.addRow("Margin:", self.lbl_divmargin)
        self.lbl_divverdict = QLabel("—"); self.lbl_divverdict.setStyleSheet("font-weight:700;font-size:14px;padding:4px;")
        fa.addRow("Status:", self.lbl_divverdict)
        self.lbl_revmach = _vl();   fa.addRow("Reversal Mach:", self.lbl_revmach)
        self.lbl_effmax = _vl();    fa.addRow("η @ Max Mach:", self.lbl_effmax)
        self.lbl_deflect = _vl();   fa.addRow("Max Fin Deflect:", self.lbl_deflect)
        ga.setLayout(fa); lay.addWidget(ga)

        # Vibration
        gv = QGroupBox("Vibration Response"); gv.setStyleSheet(_GRP)
        fv = QFormLayout(); fv.setSpacing(6)
        self.lbl_grms = _vl(); fv.addRow("RMS Accel:", self.lbl_grms)
        self.lbl_gpk = _vl();  fv.addRow("Peak (3σ):", self.lbl_gpk)
        self.lbl_drms = _vl(); fv.addRow("RMS Disp:", self.lbl_drms)
        self.lbl_resfreq = _vl(); fv.addRow("Primary Resonance:", self.lbl_resfreq)
        gv.setLayout(fv); lay.addWidget(gv)

        # Flight loads (max-Q)
        gq = QGroupBox("Flight Loads (Max-Q)"); gq.setStyleSheet(_GRP)
        fq = QFormLayout(); fq.setSpacing(6)
        self.lbl_maxq = _vl();    fq.addRow("Max Dynamic Pressure:", self.lbl_maxq)
        self.lbl_qmach = _vl();   fq.addRow("Mach at Max-Q:", self.lbl_qmach)
        self.lbl_qalt = _vl();    fq.addRow("Altitude at Max-Q:", self.lbl_qalt)
        self.lbl_qverdict = QLabel("—"); self.lbl_qverdict.setStyleSheet("font-weight:700;font-size:13px;padding:4px;")
        fq.addRow("Status:", self.lbl_qverdict)
        gq.setLayout(fq); lay.addWidget(gq)

        # Consistency warnings
        gw = QGroupBox("Consistency Checks"); gw.setStyleSheet(_GRP)
        fw_ = QVBoxLayout(); fw_.setSpacing(3)
        self.lbl_warnings = QLabel("Run assessment to check consistency.")
        self.lbl_warnings.setWordWrap(True)
        self.lbl_warnings.setStyleSheet("color:#8b949e;font-size:11px;")
        fw_.addWidget(self.lbl_warnings)
        gw.setLayout(fw_); lay.addWidget(gw)

        lay.addStretch()
        sc.setWidget(w); return sc

    # ===============================================================
    # Data access
    # ===============================================================
    def _get_assembly(self):
        main = self.window()
        if hasattr(main, 'design_ws'):
            return getattr(main.design_ws, 'assembly', None)
        return None

    def _trajectory(self):
        """Pull (altitude, speed, mach, q) arrays from the last simulation.
        Returns dict or None if no flight data."""
        main = self.window()
        sim = getattr(main, 'sim_engine', None)
        if sim is None or not getattr(sim, 'history', None):
            return None
        h = sim.history
        try:
            alt = h.get_values("altitude")
            vel = [abs(v) for v in h.get_values("velocity")]
            mach = h.get_values("mach")
            q = h.get_values("dynamic_pressure")
        except Exception:
            return None
        if not alt or not vel:
            return None
        return {
            "alt": alt, "vel": vel, "mach": mach, "q": q,
            "vmax": max(vel) if vel else 0.0,
            "mmax": max(mach) if mach else 0.0,
            "qmax": max(q) if q else 0.0,
            "apogee": max(alt) if alt else 0.0,
        }

    def _fill_from_sim(self, silent=False):
        tr = self._trajectory()
        if not tr:
            self.lbl_simsrc.setText("Source: manual input (no simulation yet)")
            self.lbl_simsrc.setStyleSheet("color:#8b949e;font-size:10px;font-style:italic;")
            if not silent:
                self._status.setText("No simulation flight data available. Run a simulation first.")
            return False
        self.sp_vmax.setValue(tr["vmax"])
        self.sp_mmax.setValue(max(0.1, tr["mmax"]))
        self.lbl_simsrc.setText(
            f"✓ Using Last Simulation\nV={tr['vmax']:.0f} m/s  M={tr['mmax']:.2f}  "
            f"Q={tr['qmax']/1000:.1f} kPa  apogee={tr['apogee']:.0f} m")
        self.lbl_simsrc.setStyleSheet("color:#7ee787;font-size:10px;font-weight:600;")
        if not silent:
            self._status.setText(f"Loaded from sim: V_max={tr['vmax']:.0f} m/s, "
                                 f"M_max={tr['mmax']:.2f}, apogee={tr['apogee']:.0f} m")
        return True

    def showEvent(self, event):
        # Auto-import flight conditions from the last simulation — never require
        # manual re-entry.
        super().showEvent(event)
        self._fill_from_sim(silent=True)

    # ===============================================================
    # Run analyses (worker threads)
    # ===============================================================
    def _busy(self, on):
        self._progress.setVisible(on)
        for b in (self.btn_all, self.btn_flutter, self.btn_vib, self.btn_aero, self.btn_modal):
            b.setEnabled(not on)

    def _run_flutter(self):
        assembly = self._get_assembly()
        if not assembly:
            self._status.setText("No rocket assembly available."); return
        from dynamics.flutter_analysis import flutter_analysis
        self._busy(True)
        self._thread = DynThread(flutter_analysis,
            (assembly, self.sp_vmax.value(), self.sp_mmax.value()), "flutter")
        self._thread.finished.connect(self._on_result)
        self._thread.errored.connect(self._on_error)
        self._thread.start()

    def _run_vibration(self):
        assembly = self._get_assembly()
        if not assembly:
            self._status.setText("No rocket assembly available."); return
        from structures.fem_interface import FEMInterface
        from dynamics.vibration_analysis import random_vibration_response
        fem = FEMInterface()
        modal = fem.modal_analysis(assembly)
        freqs = modal.frequencies_hz if modal.frequencies_hz else [50, 120, 250, 400, 600]
        self._busy(True)
        self._thread = DynThread(random_vibration_response,
            (freqs, self.sp_damp.value(), self.sp_psd.value()), "vibration")
        self._thread.finished.connect(self._on_result)
        self._thread.errored.connect(self._on_error)
        self._thread.start()

    def _run_aeroelastic(self):
        assembly = self._get_assembly()
        if not assembly:
            self._status.setText("No rocket assembly available."); return
        from dynamics.aeroelastic import full_aeroelastic_analysis
        self._busy(True)
        self._thread = DynThread(full_aeroelastic_analysis,
            (assembly, self.sp_vmax.value(), self.sp_mmax.value()), "aeroelastic")
        self._thread.finished.connect(self._on_result)
        self._thread.errored.connect(self._on_error)
        self._thread.start()

    def _shared_modal_result(self):
        """Modal result already computed by the Structures workspace (same FEM
        feature) — reuse it instead of re-running CalculiX."""
        main = self.window()
        sws = getattr(main, "structures_ws", None)
        return getattr(sws, "_modal_result", None) if sws is not None else None

    def _run_modal(self, reuse=True):
        # Reuse the Structures-workspace modal result when available — no need
        # to re-run the solver for the identical analysis.
        if reuse and self._modal_result is None:
            shared = self._shared_modal_result()
            if shared is not None:
                self._modal_result = shared
                self._render_modal(shared)
                self._tabs.setCurrentIndex(3)
                self._update_assessment()
                self._status.setText("Modal: reusing Structures-workspace result")
                if getattr(self, "_pending", None):
                    self._run_next_in_chain()
                return
        assembly = self._get_assembly()
        if not assembly:
            self._status.setText("No rocket assembly available."); return
        from structures.fem_interface import FEMInterface
        fem = FEMInterface()
        self._busy(True)
        self._thread = DynThread(fem.modal_analysis, (assembly,), "modal")
        self._thread.finished.connect(self._on_result)
        self._thread.errored.connect(self._on_error)
        self._thread.start()

    def _on_tab_changed(self, idx):
        # Mode Shapes tab — auto-populate from a shared/own modal result so the
        # tab is never blank when modal data already exists.
        if idx == 3 and self._modal_result is None:
            shared = self._shared_modal_result()
            if shared is not None:
                self._modal_result = shared
                self._render_modal(shared)
                self._update_assessment()

    def _run_all(self):
        """Chain flutter -> aeroelastic -> vibration for a full assessment."""
        assembly = self._get_assembly()
        if not assembly:
            self._status.setText("No rocket assembly available."); return
        self._pending = ["flutter", "aeroelastic", "vibration", "modal"]
        self._run_next_in_chain()

    def _run_next_in_chain(self):
        if not getattr(self, "_pending", None):
            self._update_assessment()
            return
        step = self._pending.pop(0)
        {"flutter": self._run_flutter, "aeroelastic": self._run_aeroelastic,
         "vibration": self._run_vibration, "modal": self._run_modal}[step]()

    # ===============================================================
    # Result dispatch
    # ===============================================================
    def _on_result(self, result, rtype):
        self._busy(False)
        if rtype == "flutter":
            self._flutter_result = result
            self._render_flutter(result)
            self._tabs.setCurrentIndex(0)
        elif rtype == "vibration":
            self._vib_result = result
            self._render_frf(result)
            self._tabs.setCurrentIndex(1)
        elif rtype == "aeroelastic":
            self._aero_result = result
            self._render_aero(result)
            self._tabs.setCurrentIndex(2)
        elif rtype == "modal":
            self._modal_result = result
            self._render_modal(result)
            self._tabs.setCurrentIndex(3)

        self._update_assessment()
        self._render_envelope()

        # Continue a chained full-assessment run
        if getattr(self, "_pending", None):
            self._run_next_in_chain()

    def _on_error(self, msg):
        self._busy(False)
        self._pending = []
        self._status.setText(f"Error: {msg}")
        logger.error(f"Dynamics error: {msg}")

    # ===============================================================
    # Renders
    # ===============================================================
    def _render_flutter(self, r):
        self.lbl_fspd.setText(f"{r.flutter_speed_mps:.0f} m/s" if r.flutter_speed_mps < 1e6 else "∞")
        self.lbl_fmach.setText(f"{r.flutter_mach:.2f}" if r.flutter_mach < 100 else "∞")
        vmax = self.sp_vmax.value()
        self.lbl_vmax_f.setText(f"{vmax:.0f} m/s")
        if r.flutter_speed_mps < 1e6 and vmax > 0:
            pct = (r.flutter_speed_mps - vmax) / vmax * 100.0
            txt, col = _verdict(pct)
            self.lbl_fmargin.setText(f"{pct:+.1f}%")
            self.lbl_fverdict.setText(txt)
            self.lbl_fverdict.setStyleSheet(f"color:{col};font-weight:700;font-size:14px;padding:4px;")
        else:
            self.lbl_fmargin.setText("∞")
            self.lbl_fverdict.setText("SAFE")
            self.lbl_fverdict.setStyleSheet(f"color:{_C_SAFE};font-weight:700;font-size:14px;padding:4px;")

        ax = getattr(self._flutter_plot, "ax", None)
        if ax is None:
            return
        self._flutter_plot.clear()
        self._flutter_plot._style_axis("Flutter Boundary vs Flight Profile", "Altitude (km)", "Speed (m/s)")

        if r.altitude_sweep:
            alts = [p[0] / 1000.0 for p in r.altitude_sweep]
            vf = [p[1] for p in r.altitude_sweep]
            import numpy as np
            alts_a = np.array(alts); vf_a = np.array(vf)
            order = np.argsort(alts_a); alts_a = alts_a[order]; vf_a = vf_a[order]
            # Safe/caution/unsafe shading (relative to flutter boundary)
            ax.fill_between(alts_a, 0, vf_a / 1.2, color=_C_SAFE, alpha=0.10)
            ax.fill_between(alts_a, vf_a / 1.2, vf_a / 1.1, color=_C_CAUT, alpha=0.12)
            ax.fill_between(alts_a, vf_a / 1.1, vf_a, color=_C_BAD, alpha=0.12)
            ax.plot(alts_a, vf_a, color="#f0883e", linewidth=2.0, label="Flutter boundary")
            ax.axhline(min(vf), color=_C_BAD, linestyle=":", linewidth=1.2,
                       label=f"Flutter onset {min(vf):.0f} m/s")

        # Actual flight max-velocity profile vs altitude
        tr = self._trajectory()
        if tr:
            import numpy as np
            ta = np.array(tr["alt"]) / 1000.0; tv = np.array(tr["vel"])
            ax.plot(ta, tv, color=_C_INFO, linewidth=1.8, label="Flight velocity")
        ax.axhline(vmax, color="#c9d1d9", linestyle="--", linewidth=1.0,
                   label=f"Max flight {vmax:.0f} m/s")
        ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=8, loc="best")
        self._flutter_plot.figure.tight_layout(); self._flutter_plot.canvas.draw()
        self._status.setText(f"Flutter: V_f={r.flutter_speed_mps:.0f} m/s, "
                             f"margin={self.lbl_fmargin.text()}")

    def _render_frf(self, r):
        self.lbl_grms.setText(f"{r.rms_acceleration_g:.2f} g")
        self.lbl_gpk.setText(f"{r.peak_response_g:.2f} g")
        self.lbl_drms.setText(f"{r.rms_displacement_mm:.3f} mm")

        ax = getattr(self._frf_plot, "ax", None)
        if ax is not None and r.frf_data:
            self._frf_plot.clear()
            self._frf_plot._style_axis("Frequency Response + Resonances", "Frequency (Hz)", "Magnitude (dB)")
            xs = [p[0] for p in r.frf_data]; ys = [p[1] for p in r.frf_data]
            ax.plot(xs, ys, color=_C_INFO, linewidth=1.4)
            for i, (f, mag, name) in enumerate(r.modal_markers):
                ax.plot(f, mag, "o", color=_C_BAD, markersize=5)
                ax.annotate(f"M{i+1}\n{f:.0f}Hz", (f, mag), textcoords="offset points",
                            xytext=(0, 8), ha="center", fontsize=8, color="#c9d1d9")
            self._frf_plot.figure.tight_layout(); self._frf_plot.canvas.draw()

        # Resonance table — risk by proximity to excitation sources
        from PyQt6.QtGui import QColor
        sources = self._excitation_sources()
        peaks = sorted(r.modal_markers, key=lambda p: p[0])
        self._res_table.setRowCount(len(peaks))
        self._res_worst_sr = float('inf')
        n_high = 0
        for i, (f, mag, name) in enumerate(peaks):
            risk, sr, src = self._resonance_risk(f, sources)
            if sr < self._res_worst_sr:
                self._res_worst_sr = sr
            if risk == "High":
                n_high += 1
            driver = f"{src} (SR {sr:.2f})"
            cells = [f"Mode {i+1}", f"{f:.1f}", f"{mag:.1f}", risk, driver]
            for c, val in enumerate(cells):
                it = QTableWidgetItem(val)
                if c == 3:
                    it.setForeground(QColor({"Low": _C_SAFE, "Medium": _C_CAUT, "High": _C_BAD}[risk]))
                self._res_table.setItem(i, c, it)

        if peaks:
            primary = max(peaks, key=lambda p: p[1])
            self.lbl_resfreq.setText(f"{primary[0]:.0f} Hz")
            self._frf_summary.setText(
                f"Primary resonance at {primary[0]:.0f} Hz. {len(peaks)} modes; "
                f"{n_high} near an excitation source (SR<1.2)."
                if n_high else
                f"Primary resonance at {primary[0]:.0f} Hz. {len(peaks)} modes; "
                f"all well separated from excitation sources.")
        else:
            self._frf_summary.setText("No resonance peaks identified.")
        self._status.setText(f"Vibration: G_rms={r.rms_acceleration_g:.2f}g, "
                             f"{len(peaks)} resonances, {n_high} high-risk")

    def _excitation_sources(self):
        """Forcing frequencies (Hz) the structure can be driven at."""
        tr = self._trajectory()
        vmax = tr["vmax"] if tr else self.sp_vmax.value()
        asm = self._get_assembly()
        fin_t = 0.003
        body_len = 1.0
        if asm is not None:
            try:
                from core.components import TrapezoidalFinSet, BodyTube
                for stage in asm.stages:
                    for comp in stage.children:
                        body_len = max(body_len, getattr(comp, "length", 0.0) or 0.0)
                        kids = [comp] + list(getattr(comp, "children", []))
                        for k in kids:
                            if isinstance(k, TrapezoidalFinSet):
                                fin_t = getattr(k, "thickness", fin_t) or fin_t
            except Exception:
                pass
        St = 0.2  # Strouhal number
        return {
            "Launch rail": 25.0,                                  # rail/structure vibration
            "Motor combustion": 250.0,                           # combustion roughness (nominal)
            "Fin vortex shedding": St * vmax / max(fin_t, 1e-3),  # Strouhal shedding
            "Buffet": vmax / (2.0 * max(body_len, 0.3)),         # transonic buffet
        }

    def _resonance_risk(self, f_res, sources):
        """Risk from separation ratio to nearest excitation source.
        SR = max(f_res,f_exc)/min(f_res,f_exc) (>=1). Closer => higher risk."""
        best_sr, best_src = float('inf'), "—"
        for name, fe in sources.items():
            if fe <= 0:
                continue
            sr = max(f_res, fe) / min(f_res, fe)
            if sr < best_sr:
                best_sr, best_src = sr, name
        if best_sr < 1.2:
            risk = "High"
        elif best_sr < 1.5:
            risk = "Medium"
        else:
            risk = "Low"
        return risk, best_sr, best_src

    def _render_aero(self, r):
        self.lbl_divspd.setText(f"{r.divergence_speed_mps:.0f} m/s" if r.divergence_speed_mps < 1e6 else "∞")
        self.lbl_divmach.setText(f"{r.divergence_mach:.2f}" if r.divergence_mach < 100 else "∞")
        vmax = self.sp_vmax.value()
        if r.divergence_speed_mps < 1e6 and vmax > 0:
            pct = (r.divergence_speed_mps - vmax) / vmax * 100.0
            txt, col = _verdict(pct)
            self.lbl_divmargin.setText(f"{pct:+.1f}%")
            self.lbl_divverdict.setText(txt)
            self.lbl_divverdict.setStyleSheet(f"color:{col};font-weight:700;font-size:14px;padding:4px;")
        else:
            self.lbl_divmargin.setText("∞")
            self.lbl_divverdict.setText("SAFE")
            self.lbl_divverdict.setStyleSheet(f"color:{_C_SAFE};font-weight:700;font-size:14px;padding:4px;")
        rev = getattr(r, "reversal_mach", 0.0)
        self.lbl_revmach.setText(f"{rev:.2f}" if rev > 0 else "none")
        self.lbl_effmax.setText(f"{getattr(r, 'effectiveness_at_max_mach', 1.0):+.2f}")
        self.lbl_deflect.setText(f"{r.max_deflection_mm:.2f} mm / {r.max_deflection_deg:.2f}°")

        ax = getattr(self._aero_plot, "ax", None)
        if ax is not None and r.effectiveness_data:
            self._aero_plot.clear()
            self._aero_plot._style_axis("Aeroelastic Effectiveness (smooth reversal)", "Mach", "Effectiveness η")
            xs = [p[0] for p in r.effectiveness_data]; ys = [p[1] for p in r.effectiveness_data]
            # Authority regions (by effectiveness value)
            ax.axhspan(0.3, 1.2, color=_C_SAFE, alpha=0.10)    # positive authority
            ax.axhspan(-0.3, 0.3, color=_C_CAUT, alpha=0.12)   # neutral zone
            ax.axhspan(-1.2, -0.3, color=_C_BAD, alpha=0.12)   # control reversal
            ax.text(xs[0], 0.7, "AUTHORITY", color=_C_SAFE, fontsize=7, va="center")
            ax.text(xs[0], 0.0, "NEUTRAL", color=_C_CAUT, fontsize=7, va="center")
            ax.text(xs[0], -0.7, "REVERSAL", color=_C_BAD, fontsize=7, va="center")
            ax.plot(xs, ys, color="#e6edf3", linewidth=2.0)
            ax.axhline(0, color="#484f58", linewidth=0.8)
            if rev > 0:
                ax.axvline(rev, color=_C_BAD, linestyle="--", linewidth=1.4,
                           label=f"Reversal M={rev:.2f}")
            ax.axvline(self.sp_mmax.value(), color=_C_INFO, linestyle="-", linewidth=1.4,
                       label=f"Flight M={self.sp_mmax.value():.2f}")
            ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=8)
            self._aero_plot.figure.tight_layout(); self._aero_plot.canvas.draw()
        self._status.setText(f"Aeroelastic: V_div={r.divergence_speed_mps:.0f} m/s, "
                             f"margin={self.lbl_divmargin.text()}")

    def _render_modal(self, r):
        import pathlib, numpy as np
        if not getattr(self, '_modal_plot', None) or not hasattr(self._modal_plot, 'load_cylinder'):
            return
        self.mode_combo.blockSignals(True)
        self.mode_combo.clear()
        freqs = list(getattr(r, "frequencies_hz", []) or [])
        descs = list(getattr(r, "descriptions", []) or [])
        inp = pathlib.Path("fem_run/modal/structure_mesh.inp")
        have_fem = bool(getattr(r, "mode_shapes", None)) and inp.is_file()

        if have_fem:
            # Real FEM mesh + mode shapes
            self._modal_is_synth = False
            self._modal_plot.load_mesh(str(inp))
            for i, f in enumerate(freqs):
                d = descs[i] if i < len(descs) else f"Mode {i+1}"
                self.mode_combo.addItem(f"Mode {i+1}: {f:.1f} Hz — {d}", userData=i)
            self.mode_combo.blockSignals(False)
            if self.mode_combo.count():
                self.mode_combo.setCurrentIndex(0); self._on_mode_select(0)
            self._status.setText(f"Modal (FEM): {len(freqs)} modes"
                                 + (f", f1={freqs[0]:.1f} Hz" if freqs else ""))
            return

        # ---- Synthetic analytic fallback (no FEM shapes) ----
        self._modal_is_synth = True
        L, R = self._rocket_dims()
        self._modal_plot.load_cylinder(length=L, radius=R)
        # First 3 frequencies: use FEM freqs if present, else nominal estimates
        nominal = [52.0, 145.0, 270.0]
        f3 = [(freqs[i] if i < len(freqs) and freqs[i] > 0 else nominal[i]) for i in range(3)]
        self._synth_modes = [
            ("First Bending",  "bend1", f3[0]),
            ("Second Bending", "bend2", f3[1]),
            ("First Torsion",  "tors",  f3[2]),
        ]
        for i, (name, _, f) in enumerate(self._synth_modes):
            self.mode_combo.addItem(f"Mode {i+1} – {name} ({f:.0f} Hz)", userData=i)
        self.mode_combo.blockSignals(False)
        self.mode_combo.setCurrentIndex(0)
        self._apply_synthetic_mode(0)
        self._status.setText(f"Modal (analytic): 3 mode shapes, f1={f3[0]:.0f} Hz "
                             "(FEM shapes unavailable — synthetic visualization)")

    def _rocket_dims(self):
        asm = self._get_assembly()
        L, D = 1.0, 0.1
        if asm is not None:
            try:
                total = 0.0; dmax = 0.0
                for stage in asm.stages:
                    for comp in stage.children:
                        total += getattr(comp, "length", 0.0) or 0.0
                        dmax = max(dmax, getattr(comp, "diameter", 0.0) or 0.0)
                if total > 0: L = total
                if dmax > 0: D = dmax
            except Exception:
                pass
        return L, max(D / 2.0, 0.02)

    def _apply_synthetic_mode(self, idx):
        import numpy as np
        if not getattr(self, "_synth_modes", None):
            return
        if idx < 0 or idx >= len(self._synth_modes):
            return
        name, kind, freq = self._synth_modes[idx]
        pts = getattr(self._modal_plot, "base_points", None)
        if pts is None:
            return
        z = pts[:, 2]; L = z.max() if z.max() > 0 else 1.0
        zeta = z / L
        if kind == "bend1":
            phi = 1.0 - np.cos(np.pi * zeta / 2.0)
            vecs = np.column_stack([phi, np.zeros_like(phi), np.zeros_like(phi)])
        elif kind == "bend2":
            phi = 1.0 - np.cos(3.0 * np.pi * zeta / 2.0)
            vecs = np.column_stack([phi, np.zeros_like(phi), np.zeros_like(phi)])
        else:  # torsion: tangential rotation about axis
            th = np.sin(np.pi * zeta / 2.0)
            vecs = np.column_stack([-pts[:, 1] * th, pts[:, 0] * th, np.zeros_like(th)])
        disp = {i: tuple(vecs[i]) for i in range(len(pts))}
        self._modal_plot.set_mode_shape(disp, freq_hz=freq, description=name, mode_index=idx + 1)
        self._mode_info.setText(
            f"Frequency: {freq:.1f} Hz   Damping: {self.sp_damp.value()*100:.1f}%   Type: {name}")

    def _on_mode_select(self, _):
        if getattr(self, "_modal_is_synth", False):
            self._apply_synthetic_mode(self.mode_combo.currentData() or 0)
            return
        r = self._modal_result
        if not r or not getattr(r, "mode_shapes", None):
            return
        idx = self.mode_combo.currentData()
        if idx is None or idx >= len(r.mode_shapes):
            return
        freqs = r.frequencies_hz; descs = getattr(r, "descriptions", [])
        self._modal_plot.set_mode_shape(
            r.mode_shapes[idx], freq_hz=freqs[idx] if idx < len(freqs) else 0.0,
            description=descs[idx] if idx < len(descs) else f"Mode {idx+1}", mode_index=idx + 1)
        self._mode_info.setText(
            f"Frequency: {freqs[idx] if idx<len(freqs) else 0:.1f} Hz   "
            f"Damping: {self.sp_damp.value()*100:.1f}%   "
            f"Type: {descs[idx] if idx<len(descs) else 'Mode'}")

    def _render_envelope(self):
        ax = getattr(self._env_plot, "ax", None)
        if ax is None:
            return
        import numpy as np
        self._env_plot.clear()
        self._env_plot._style_axis("Flight Envelope — V vs Altitude", "Velocity (m/s)", "Altitude (km)")

        tr = self._trajectory()
        vf_sw = (sorted(self._flutter_result.altitude_sweep, key=lambda p: p[0])
                 if self._flutter_result and self._flutter_result.altitude_sweep else [])
        vdiv = (self._aero_result.divergence_speed_mps
                if self._aero_result and self._aero_result.divergence_speed_mps < 1e6 else None)

        verdict, vcol = "SAFE", _C_SAFE
        crosses = []

        # Boundaries
        xmax = self.sp_vmax.value() * 1.4
        if vf_sw:
            alts = np.array([p[0] / 1000.0 for p in vf_sw]); vf = np.array([p[1] for p in vf_sw])
            xmax = max(xmax, vf.max() * 1.1)
            ax.fill_betweenx(alts, vf, xmax, color=_C_BAD, alpha=0.10)        # unsafe (flutter)
            ax.fill_betweenx(alts, 0, vf, color=_C_SAFE, alpha=0.06)          # safe
            ax.plot(vf, alts, color="#f0883e", linewidth=2.0, label="Flutter boundary")
        if vdiv is not None:
            ax.axvline(vdiv, color="#bc8cff", linestyle="--", linewidth=1.4,
                       label=f"Divergence {vdiv:.0f} m/s")

        # Actual trajectory + crossing checks
        if tr:
            ta = np.array(tr["alt"]) / 1000.0; tv = np.array(tr["vel"])
            ax.plot(tv, ta, color=_C_INFO, linewidth=2.0, label="Actual trajectory")
            if vf_sw:
                vf_at = np.interp(tr["alt"], [p[0] for p in vf_sw], [p[1] for p in vf_sw])
                if np.any(np.array(tr["vel"]) >= vf_at):
                    crosses.append("flutter")
            if vdiv is not None and tr["vmax"] >= vdiv:
                crosses.append("divergence")
            if crosses:
                verdict, vcol = "UNSAFE", _C_BAD
        else:
            ax.text(0.5, 0.5, "No simulation trajectory.\nRun a simulation, then re-open Dynamics.",
                    transform=ax.transAxes, ha="center", va="center", color="#8b949e", fontsize=10)

        if tr or vf_sw or vdiv is not None:
            ax.set_title(
                f"Flight Envelope — {verdict}"
                + (f" (crosses {', '.join(crosses)})" if crosses else ""),
                color=vcol, fontsize=12, fontweight="bold", pad=10)
            ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=8, loc="best")
            self._env_plot.figure.tight_layout(); self._env_plot.canvas.draw()
        self._envelope_verdict = verdict

    # ===============================================================
    # Engineering assessment
    # ===============================================================
    _Q_LIMIT = 80_000.0  # nominal dynamic-pressure concern threshold (Pa)

    def _score_margin(self, pct):
        """Map a percentage margin to 0..100 (>=30%→100, <=-10%→0)."""
        return max(0.0, min(100.0, (pct + 10.0) / 40.0 * 100.0))

    def _update_assessment(self):
        vmax = self.sp_vmax.value()
        mmax = self.sp_mmax.value()

        # ---- Max-Q (Flight Loads) ----
        tr = self._trajectory()
        q_score = 100.0
        if tr and tr["q"]:
            import numpy as np
            q = np.array(tr["q"]); alt = np.array(tr["alt"]); mach = np.array(tr["mach"])
            iq = int(np.argmax(q)); qmax = float(q[iq])
            self.lbl_maxq.setText(f"{qmax/1000:.1f} kPa")
            self.lbl_qmach.setText(f"{mach[iq]:.2f}" if iq < len(mach) else "—")
            self.lbl_qalt.setText(f"{alt[iq]/1000:.2f} km" if iq < len(alt) else "—")
            q_pct = (self._Q_LIMIT - qmax) / self._Q_LIMIT * 100.0
            qv, qc = _verdict(q_pct)
            self.lbl_qverdict.setText(qv)
            self.lbl_qverdict.setStyleSheet(f"color:{qc};font-weight:700;font-size:13px;padding:4px;")
            q_score = self._score_margin(q_pct)
            self._maxq_info = {"qmax": qmax, "mach": float(mach[iq]) if iq < len(mach) else 0.0,
                               "alt": float(alt[iq]) if iq < len(alt) else 0.0, "verdict": qv}
        else:
            for l in (self.lbl_maxq, self.lbl_qmach, self.lbl_qalt):
                l.setText("—")
            self.lbl_qverdict.setText("no sim data")
            self.lbl_qverdict.setStyleSheet("color:#8b949e;font-weight:700;font-size:13px;padding:4px;")
            self._maxq_info = None

        # ---- Sub-scores ----
        flutter_score = div_score = res_score = rev_score = 100.0
        if self._flutter_result and self._flutter_result.flutter_speed_mps < 1e6 and vmax > 0:
            flutter_score = self._score_margin(
                (self._flutter_result.flutter_speed_mps - vmax) / vmax * 100.0)
        if self._aero_result and self._aero_result.divergence_speed_mps < 1e6 and vmax > 0:
            div_score = self._score_margin(
                (self._aero_result.divergence_speed_mps - vmax) / vmax * 100.0)
        worst_sr = getattr(self, "_res_worst_sr", float('inf'))
        if worst_sr < float('inf'):
            res_score = max(0.0, min(100.0, (worst_sr - 1.0) / 0.5 * 100.0))
        if self._aero_result:
            rm = getattr(self._aero_result, "reversal_margin", float('inf'))
            if rm < float('inf'):
                rev_score = max(0.0, min(100.0, (rm + 0.0) / 0.5 * 100.0))

        score = (0.30 * flutter_score + 0.25 * div_score + 0.20 * q_score
                 + 0.15 * res_score + 0.10 * rev_score)
        if score >= 90:
            rating, col = "EXCELLENT", _C_SAFE
        elif score >= 75:
            rating, col = "GOOD", _C_SAFE
        elif score >= 50:
            rating, col = "CAUTION", _C_CAUT
        else:
            rating, col = "UNSAFE", _C_BAD
        self._safety_score = score; self._safety_rating = rating
        self.lbl_overall.setText(f"SAFETY SCORE  {score:.0f}/100\n{rating}")
        self.lbl_overall.setStyleSheet(
            f"font-weight:800;font-size:16px;padding:10px;border-radius:6px;"
            f"background:#161b22;color:{col};border:1px solid {col}66;")

        self._update_warnings(vmax, mmax)

    def _update_warnings(self, vmax, mmax):
        warns = []
        fr, ar = self._flutter_result, self._aero_result
        if fr and fr.flutter_mach < 1e2 and fr.flutter_mach < mmax:
            warns.append(("UNSAFE", f"Flutter Mach {fr.flutter_mach:.2f} < max flight Mach {mmax:.2f}"))
        if fr and fr.flutter_speed_mps < 1e6 and vmax > 0 and fr.flutter_speed_mps < vmax:
            warns.append(("UNSAFE", "Negative flutter margin (V_flutter < V_max)"))
        if ar and ar.divergence_mach < 1e2 and ar.divergence_mach < mmax:
            warns.append(("UNSAFE", f"Divergence Mach {ar.divergence_mach:.2f} < max flight Mach {mmax:.2f}"))
        if getattr(self, "_modal_is_synth", False) and self._modal_result is not None:
            warns.append(("CAUTION", "FEM mode-shape extraction unavailable — using synthetic shapes"))
        if getattr(self, "_res_worst_sr", float('inf')) < 1.2:
            warns.append(("UNSAFE", "Resonance within SR<1.2 of an excitation source"))
        if self._maxq_info and self._maxq_info["verdict"] == "UNSAFE":
            warns.append(("UNSAFE", f"Max-Q {self._maxq_info['qmax']/1000:.1f} kPa exceeds limit"))
        if ar and getattr(ar, "reversal_mach", 0.0) > 0 and ar.reversal_mach < mmax:
            warns.append(("UNSAFE", f"Control reversal at M={ar.reversal_mach:.2f} below max Mach"))

        if not warns:
            self.lbl_warnings.setText("✓ All consistency checks passed.")
            self.lbl_warnings.setStyleSheet("color:#7ee787;font-size:11px;font-weight:600;")
            return
        has_bad = any(s == "UNSAFE" for s, _ in warns)
        lines = [("✗ " if s == "UNSAFE" else "! ") + msg for s, msg in warns]
        self.lbl_warnings.setText("\n".join(lines))
        self.lbl_warnings.setStyleSheet(
            f"color:{_C_BAD if has_bad else _C_CAUT};font-size:11px;font-weight:600;")

    # ===============================================================
    # Report export
    # ===============================================================
    def _assessment_dict(self):
        vmax = self.sp_vmax.value(); mmax = self.sp_mmax.value()
        d = {"timestamp": datetime.datetime.now().isoformat(),
             "max_flight_speed_mps": vmax, "max_mach": mmax,
             "safety_score": getattr(self, "_safety_score", None),
             "safety_rating": getattr(self, "_safety_rating", None),
             "envelope_verdict": getattr(self, "_envelope_verdict", None)}
        if getattr(self, "_maxq_info", None):
            d["max_q"] = {"q_pa": self._maxq_info["qmax"], "mach": self._maxq_info["mach"],
                          "altitude_m": self._maxq_info["alt"], "verdict": self._maxq_info["verdict"]}
        if self._flutter_result:
            r = self._flutter_result
            pct = ((r.flutter_speed_mps - vmax) / vmax * 100.0) if r.flutter_speed_mps < 1e6 and vmax > 0 else float('inf')
            d["flutter"] = {"flutter_speed_mps": r.flutter_speed_mps, "flutter_mach": r.flutter_mach,
                            "margin_pct": pct, "verdict": _verdict(pct)[0] if pct != float('inf') else "SAFE"}
        if self._aero_result:
            r = self._aero_result
            pct = ((r.divergence_speed_mps - vmax) / vmax * 100.0) if r.divergence_speed_mps < 1e6 and vmax > 0 else float('inf')
            d["aeroelastic"] = {"divergence_speed_mps": r.divergence_speed_mps,
                                "divergence_mach": r.divergence_mach, "margin_pct": pct,
                                "reversal_mach": getattr(r, "reversal_mach", 0.0),
                                "effectiveness_at_max_mach": getattr(r, "effectiveness_at_max_mach", 1.0),
                                "verdict": _verdict(pct)[0] if pct != float('inf') else "SAFE"}
        if self._vib_result:
            r = self._vib_result
            d["vibration"] = {"rms_accel_g": r.rms_acceleration_g, "peak_3sigma_g": r.peak_response_g,
                              "rms_disp_mm": r.rms_displacement_mm,
                              "resonances": [{"freq_hz": f, "mag_db": m, "mode": n}
                                             for f, m, n in r.modal_markers]}
        return d

    def _export(self, kind):
        if not (self._flutter_result or self._aero_result or self._vib_result):
            self._status.setText("Nothing to export — run an analysis first."); return
        exts = {"json": "JSON (*.json)", "csv": "CSV (*.csv)", "pdf": "PDF (*.pdf)"}
        path, _ = QFileDialog.getSaveFileName(self, f"Export {kind.upper()} report",
                                              f"dynamics_report.{kind}", exts[kind])
        if not path:
            return
        try:
            if kind == "json":
                with open(path, "w") as f:
                    json.dump(self._assessment_dict(), f, indent=2, default=str)
            elif kind == "csv":
                self._export_csv(path)
            elif kind == "pdf":
                self._export_pdf(path)
            self._status.setText(f"Report exported: {path}")
        except Exception as e:
            logger.error(f"Export failed: {e}")
            self._status.setText(f"Export failed: {e}")

    def _export_csv(self, path):
        d = self._assessment_dict()
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Section", "Metric", "Value"])
            for sec, vals in d.items():
                if isinstance(vals, dict):
                    for k, v in vals.items():
                        if not isinstance(v, list):
                            w.writerow([sec, k, v])
                else:
                    w.writerow(["", sec, vals])
            if self._vib_result and self._vib_result.modal_markers:
                w.writerow([]); w.writerow(["Resonance", "Freq (Hz)", "Mag (dB)", "Mode"])
                for fr, m, n in self._vib_result.modal_markers:
                    w.writerow(["", f"{fr:.1f}", f"{m:.1f}", n])

    def _export_pdf(self, path):
        from matplotlib.backends.backend_pdf import PdfPages
        from matplotlib.figure import Figure
        d = self._assessment_dict()
        with PdfPages(path) as pdf:
            # Summary page
            fig = Figure(figsize=(8.5, 11)); ax = fig.add_subplot(111); ax.axis("off")
            lines = ["K2 AEROSPACE — DYNAMICS FLIGHT SAFETY ASSESSMENT", "",
                     f"Generated: {d['timestamp']}",
                     f"Max Flight Speed: {d['max_flight_speed_mps']:.0f} m/s   Max Mach: {d['max_mach']:.2f}", ""]
            for sec in ("flutter", "aeroelastic", "vibration"):
                if sec in d:
                    lines.append(sec.upper())
                    for k, v in d[sec].items():
                        if not isinstance(v, list):
                            lines.append(f"   {k}: {v}")
                    lines.append("")
            ax.text(0.05, 0.95, "\n".join(lines), va="top", ha="left", fontsize=10, family="monospace")
            pdf.savefig(fig)
            # Plot pages
            for plot in (self._flutter_plot, self._frf_plot, self._aero_plot, self._env_plot):
                if hasattr(plot, "figure"):
                    pdf.savefig(plot.figure)
