"""
K2 AeroSim — Mode Shape Viewer
=================================
Professional displacement contour visualization for modal analysis.
Renders animated mode shapes with ANSYS/Abaqus-style color mapping.

Features:
  - Displacement magnitude contour mapping (turbo colormap)
  - Scalar bar with engineering units (mm)
  - Undeformed wireframe ghost overlay (reference shape)
  - Mode info text overlay
  - Sinusoidal animation at natural frequency
  - Optional amplification (kept at 0 by default per user preference)
"""
import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                              QLabel, QFrame)
from PyQt6.QtCore import Qt, QTimer

import logging
logger = logging.getLogger("K2.ModeShapeViewer")


class ModeShapeViewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.base_points = None
        self.mode_shape = None       # (N, 3) displacement array
        self.phase = 0.0
        self._freq_hz = 0.0          # Current mode natural frequency
        self._mode_desc = ""         # Current mode description
        self._mode_index = 0         # Current mode index (1-based)
        self._grid = None            # Main deformed grid
        self._ghost_grid = None      # Undeformed wireframe reference
        self._ghost_actor = None
        self._mesh_actor = None
        self._scalar_bar_actor = None
        self._text_actor = None
        self._info_text_actor = None
        self.id_to_idx = {}

        self._setup_ui()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._animate)
        self.timer.start(50)  # 20 FPS

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Plotter
        self.plotter = QtInteractor(self)
        self.plotter.set_background("#0d1117", top="#161b22")
        self.plotter.add_axes(interactive=False, line_width=2)
        layout.addWidget(self.plotter.interactor, 1)

        # Controls bar
        ctrl = QFrame()
        ctrl.setStyleSheet("background:#161b22; border-top:1px solid #30363d;")
        ctrl_layout = QHBoxLayout(ctrl)
        ctrl_layout.setContentsMargins(12, 6, 12, 6)
        ctrl_layout.setSpacing(10)

        self.btn_play = QPushButton("⏸ Pause")
        self.btn_play.setCheckable(True)
        self.btn_play.clicked.connect(self._toggle_play)
        self.btn_play.setStyleSheet(
            "background:#21262d; color:#c9d1d9; padding:5px 14px; "
            "border-radius:6px; font-weight:600; border:1px solid #30363d;"
        )
        ctrl_layout.addWidget(self.btn_play)

        # Mode info label
        self.lbl_mode_info = QLabel("No mode loaded")
        self.lbl_mode_info.setStyleSheet(
            "color:#8b949e; font-size:12px; font-family:'Segoe UI',sans-serif;"
        )
        ctrl_layout.addWidget(self.lbl_mode_info, 1)

        layout.addWidget(ctrl)

    def _toggle_play(self, checked):
        if checked:
            self.timer.stop()
            self.btn_play.setText("▶ Play")
        else:
            self.timer.start(50)
            self.btn_play.setText("⏸ Pause")

    def load_mesh(self, inp_path: str):
        """Load base mesh from CalculiX .inp file."""
        import pathlib
        path = pathlib.Path(inp_path)
        if not path.is_file():
            return

        nodes = {}
        elements = []
        in_nodes = False
        in_elems = False

        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("*NODE"):
                in_nodes = True; in_elems = False; continue
            if line.startswith("*ELEMENT"):
                in_elems = True; in_nodes = False; continue
            if line.startswith("*"):
                in_nodes = in_elems = False; continue

            parts = line.split(",")
            if in_nodes and len(parts) >= 4:
                nodes[int(parts[0])] = [float(parts[1]), float(parts[2]), float(parts[3])]
            elif in_elems and len(parts) >= 5:
                en = [int(n) for n in parts[1:5] if n.strip()]
                if len(en) == 4:
                    elements.append(en)

        if not nodes or not elements:
            return

        node_ids = sorted(nodes.keys())
        self.id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

        pts = np.array([nodes[nid] for nid in node_ids])
        self.base_points = pts.copy()

        cells = []
        cell_types = []
        for en in elements:
            cells.extend([4, self.id_to_idx[en[0]], self.id_to_idx[en[1]],
                          self.id_to_idx[en[2]], self.id_to_idx[en[3]]])
            cell_types.append(pv.CellType.QUAD)

        self._grid = pv.UnstructuredGrid(cells, np.array(cell_types), pts.copy())
        self._ghost_grid = pv.UnstructuredGrid(cells, np.array(cell_types), pts.copy())

        # Initial zero displacement contour
        self._grid["Displacement (mm)"] = np.zeros(len(pts))

        self.plotter.clear_actors()

        # Ghost wireframe (undeformed reference)
        self._ghost_actor = self.plotter.add_mesh(
            self._ghost_grid, style="wireframe",
            color="#30363d", line_width=1, opacity=0.3,
            label="_ghost"
        )

        # Deformed contour mesh
        self._mesh_actor = self.plotter.add_mesh(
            self._grid,
            scalars="Displacement (mm)",
            cmap="turbo",
            show_edges=True,
            edge_color="#1a1e24",
            line_width=0.5,
            clim=[0, 1],
            scalar_bar_args={
                "title": "Displacement (mm)",
                "title_font_size": 11,
                "label_font_size": 10,
                "color": "#c9d1d9",
                "position_x": 0.85,
                "position_y": 0.1,
                "width": 0.08,
                "height": 0.7,
                "fmt": "%.3f",
            },
        )

        self.plotter.reset_camera()
        logger.info(f"Mode shape viewer: loaded {len(nodes)} nodes, {len(elements)} elements")

    def load_cylinder(self, length=1.0, radius=0.05, n_z=26, n_theta=14):
        """Build a simple cylindrical rocket wireframe for analytic mode-shape
        animation (used when no FEM mesh/mode-shapes are available)."""
        zs = np.linspace(0.0, length, n_z)
        pts = []
        for z in zs:
            for j in range(n_theta):
                a = 2.0 * np.pi * j / n_theta
                pts.append([radius * np.cos(a), radius * np.sin(a), z])
        pts = np.array(pts, dtype=float)
        self.base_points = pts.copy()
        self.id_to_idx = {i: i for i in range(len(pts))}

        cells, ctypes = [], []
        for i in range(n_z - 1):
            for j in range(n_theta):
                j2 = (j + 1) % n_theta
                a = i * n_theta + j
                b = i * n_theta + j2
                c = (i + 1) * n_theta + j2
                d = (i + 1) * n_theta + j
                cells.extend([4, a, b, c, d]); ctypes.append(pv.CellType.QUAD)

        self._grid = pv.UnstructuredGrid(cells, np.array(ctypes), pts.copy())
        self._ghost_grid = pv.UnstructuredGrid(cells, np.array(ctypes), pts.copy())
        self._grid["Displacement (mm)"] = np.zeros(len(pts))

        self.plotter.clear_actors()
        self._ghost_actor = self.plotter.add_mesh(
            self._ghost_grid, style="wireframe", color="#484f58",
            line_width=1, opacity=0.35)
        self._mesh_actor = self.plotter.add_mesh(
            self._grid, scalars="Displacement (mm)", cmap="turbo",
            show_edges=True, edge_color="#1a1e24", line_width=0.5, clim=[0, 1],
            scalar_bar_args={"title": "Displacement", "title_font_size": 11,
                             "label_font_size": 10, "color": "#c9d1d9",
                             "position_x": 0.85, "position_y": 0.1,
                             "width": 0.08, "height": 0.7, "fmt": "%.2f"})
        self.plotter.reset_camera()
        logger.info(f"Mode shape viewer: synthetic cylinder {len(pts)} nodes")

    def set_mode_shape(self, mode_dict, freq_hz=0.0, description="", mode_index=0):
        """Set the active mode shape.

        Args:
            mode_dict: {node_id: (dx, dy, dz)} displacement mapping
            freq_hz: Natural frequency in Hz
            description: Mode description string
            mode_index: 1-based mode index
        """
        if self.base_points is None or not mode_dict:
            return

        # Build displacement array matching base_points order
        disp = np.zeros_like(self.base_points)
        for nid, vec in mode_dict.items():
            if nid in self.id_to_idx:
                disp[self.id_to_idx[nid]] = vec

        self.mode_shape = disp
        self.phase = 0.0
        self._freq_hz = freq_hz
        self._mode_desc = description
        self._mode_index = mode_index

        # Update info label
        if freq_hz > 0:
            self.lbl_mode_info.setText(
                f"Mode {mode_index}: {freq_hz:.1f} Hz — {description}"
            )
        else:
            self.lbl_mode_info.setText(f"Mode {mode_index}: {description}")

        # Force first frame render
        self._animate()

    def _animate(self):
        if self.base_points is None or self.mode_shape is None:
            return
        if self._grid is None:
            return

        self.phase += 0.15
        if self.phase > 2 * np.pi:
            self.phase -= 2 * np.pi

        # Normalize mode shape so max displacement = 1% of bounding box diagonal
        # This gives extremely subtle, minimal deformation — just enough
        # to perceive the mode character without distorting the rocket
        max_raw = np.linalg.norm(self.mode_shape, axis=1).max()
        if max_raw > 1e-15:
            bbox_diag = np.linalg.norm(
                self.base_points.max(axis=0) - self.base_points.min(axis=0)
            )
            # Target: 1% of bbox diagonal as peak visual displacement
            scale = 0.01 * bbox_diag / max_raw
        else:
            scale = 0.0

        sin_phase = np.sin(self.phase)
        disp_visual = self.mode_shape * sin_phase * scale

        # Update deformed geometry
        self._grid.points = self.base_points + disp_visual

        # Contour shows normalized displacement magnitude (0–100%)
        # so color mapping always shows the full shape pattern
        raw_mags = np.linalg.norm(self.mode_shape, axis=1)
        mags_normalized = (raw_mags / max_raw * abs(sin_phase)) if max_raw > 1e-15 else raw_mags
        self._grid["Displacement (mm)"] = mags_normalized * max_raw * 1000.0

        if max_raw > 0:
            self.plotter.update_scalar_bar_range([0, max_raw * 1000.0])

    def clear(self):
        """Reset the viewer to empty state."""
        self.base_points = None
        self.mode_shape = None
        self._grid = None
        self._ghost_grid = None
        self.plotter.clear_actors()
        self.lbl_mode_info.setText("No mode loaded")
