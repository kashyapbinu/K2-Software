"""
K2 Aerospace — 3D Structural Deformation Viewer
================================================
Shows the rocket airframe deforming under load. Uses the SAME assembly-based
geometry as the Design viewer and the 3D stress viewer (build_rocket_regions),
so the shape matches everywhere.

Renders a translucent undeformed "ghost" plus the deformed body coloured by
displacement magnitude (mm), with a user-selectable exaggeration factor.
The deflection shape is a cantilever bend (fixed at the aft end, growing
toward the nose) scaled to the analysed maximum displacement.
"""
from __future__ import annotations

import logging
import numpy as np

try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
    _PYVISTA = True
except Exception as e:  # pragma: no cover
    _PYVISTA = False
    _PV_ERR = str(e)

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt

from ui.widgets.stress_viewer import build_rocket_regions

logger = logging.getLogger("K2.DeformationViewer")


class DeformationViewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._base = None          # merged undeformed mesh
        self._total_len = 1.0
        self._max_defl_mm = 0.0
        self._exag = 10.0
        self._setup_ui()

    def _setup_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        if not _PYVISTA:
            l = QLabel(f"3D viewer unavailable: {_PV_ERR}")
            l.setAlignment(Qt.AlignmentFlag.AlignCenter)
            l.setStyleSheet("color:#f85149;padding:20px;")
            lay.addWidget(l)
            self.plotter = None
            return
        self.plotter = QtInteractor(self)
        self.plotter.set_background("#0d1117", top="#161b22")
        self.plotter.add_axes(interactive=False, line_width=2)
        lay.addWidget(self.plotter.interactor, 1)
        self._empty = QLabel("No Results Available\n\nRun Static Analysis to view deformation.")
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty.setStyleSheet("color:#6e7681;font-size:14px;background:transparent;")
        self._empty.setParent(self.plotter.interactor)
        self._empty.show()

    def _side_view(self):
        try:
            self.plotter.camera_position = "xz"
            self.plotter.camera.position = (0, 5, 0)
            self.plotter.camera.focal_point = (0, 0, 0)
            self.plotter.camera.up = (0, 0, 1)
        except Exception:
            pass

    def set_geometry(self, state, assembly):
        if not _PYVISTA or self.plotter is None:
            return
        regions, total_len = build_rocket_regions(state, assembly)
        self._total_len = max(total_len, 1e-6)
        meshes = list(regions.values())
        if not meshes:
            self._base = None
            return
        base = meshes[0]
        for m in meshes[1:]:
            base = base.merge(m)
        self._base = base

    def set_deflection(self, state, assembly, max_defl_mm, exaggeration=10.0):
        """Render the deformed shape. max_defl_mm = true peak deflection,
        exaggeration = visual amplification (1/10/50/100)."""
        if not _PYVISTA or self.plotter is None:
            return
        self.set_geometry(state, assembly)
        self._max_defl_mm = max_defl_mm
        self._exag = exaggeration
        if self._base is None:
            return
        if hasattr(self, "_empty"):
            self._empty.hide()
        self._render()

    def _render(self):
        if self._base is None:
            return
        self.plotter.clear()
        base = self._base
        pts = base.points
        L = self._total_len
        # Cantilever bend: fixed at aft (z=0), grows toward nose (z=L).
        zf = np.clip(pts[:, 2] / L, 0, 1)
        dmag = zf ** 2                                  # 0 at tail → 1 at nose

        # Visual amplitude = true deflection × exaggeration, but a stiff metal
        # airframe deflects <1 mm — invisible on a metre-scale model. So when a
        # non-zero deflection exists, floor the *visual* bend at ~6% of the model
        # length so the mode is always perceptible. The reported number stays the
        # true magnitude; only the on-screen amplitude is auto-fit.
        true_m = self._max_defl_mm / 1000.0
        amp_m = true_m * self._exag
        min_visible = 0.06 * L
        if true_m > 1e-9 and amp_m < min_visible:
            amp_m = min_visible
        dx = amp_m * dmag
        disp = np.zeros_like(pts)
        disp[:, 0] = dx

        deformed = base.copy()
        deformed.points = pts + disp
        deformed["Displacement (mm)"] = dmag * self._max_defl_mm

        # Ghost (undeformed) reference
        self.plotter.add_mesh(base, color="#30363d", opacity=0.25,
                              style="wireframe", line_width=1, name="ghost")
        # Deformed, coloured by displacement
        self.plotter.add_mesh(
            deformed, scalars="Displacement (mm)", cmap="turbo",
            smooth_shading=True, specular=0.3, name="deformed",
            scalar_bar_args=dict(title="Displacement (mm)", title_font_size=12,
                                 label_font_size=10, color="#c9d1d9",
                                 position_x=0.86, position_y=0.12,
                                 width=0.06, height=0.7, fmt="%.2f", n_labels=6))
        # Peak marker at the tip (max displacement)
        tip_idx = int(np.argmax(dmag))
        self.plotter.add_point_labels(
            [deformed.points[tip_idx]],
            [f"Max: {self._max_defl_mm:.2f} mm  (shape ×{self._exag:.0f}, auto-fit)"],
            font_size=11, text_color="#ffffff", point_color="#ff3b30",
            point_size=8, shape_color="#161b22", shape_opacity=0.7,
            always_visible=True, name="defl_label")

        self._side_view()
        self.plotter.reset_camera()
        self.plotter.render()
