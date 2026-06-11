"""
K2 Aerospace — 3D Structural Stress Viewer
==========================================
Interactive ANSYS-style stress visualization of the rocket airframe.

The rocket geometry is built from the SAME component assembly and the SAME
geometry helpers as the Design/Geometry workspace (visualization.viewer_3d),
so the shape shown here is identical to the design view — ogive nose, real
body tubes / transitions / fins / nozzle, correct stacking (nose tip at
z = total_length, aft end at z = 0).

A contour field for the selected stress measure is then painted on that mesh.
Because the analytical engine returns *peak* values per stress component, the
spatial distribution of those peaks is synthesized (bending peaks at the
critical section, thermal at the nose, axial roughly uniform, etc.) so the
contour reads like a real FE plot.

Features
--------
  • Stress modes: Von Mises / Axial / Hoop / Shear / Thermal / Safety Factor
  • Blue→Green→Yellow→Orange→Red contour (jet), engineering scalar bar
  • Maximum-stress marker (sphere) + floating peak label
  • Hover tooltips · rotate / pan / zoom + Reset View
  • Component isolation: Airframe / Fins / Motor Mount / Bulkheads /
    Recovery Bay / Entire Vehicle
"""
from __future__ import annotations

import logging
import math
import numpy as np

try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
    _PYVISTA = True
except Exception as e:  # pragma: no cover
    _PYVISTA = False
    _PV_ERR = str(e)

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QPushButton, QLabel, QFrame
)
from PyQt6.QtCore import Qt
from ui.icons import icon

logger = logging.getLogger("K2.StressViewer")

STRESS_MODES = [
    "Von Mises Stress", "Axial Stress", "Hoop Stress",
    "Shear Stress", "Thermal Stress", "Safety Factor",
]
COMPONENTS = [
    "Entire Vehicle", "Airframe", "Fins", "Motor Mount",
    "Bulkheads", "Recovery Bay",
]

_MODE_KEY = {
    "Von Mises Stress": "von_mises",
    "Axial Stress": "axial",
    "Hoop Stress": "hoop",
    "Shear Stress": "shear",
    "Thermal Stress": "thermal",
    "Safety Factor": "von_mises",
}

# Region → component-isolation group
_REGION_GROUP = {
    "nose": "Airframe", "airframe": "Airframe", "recovery": "Recovery Bay",
    "mount": "Motor Mount", "fins": "Fins", "bulkheads": "Bulkheads",
}
_REGION_LABEL = {
    "nose": "Nose Cone", "airframe": "Airframe", "recovery": "Recovery Bay",
    "mount": "Motor Mount", "fins": "Fins", "bulkheads": "Bulkheads",
}

_CTRL_BTN = (
    "QPushButton{background:#21262d;color:#c9d1d9;padding:5px 12px;"
    "border-radius:6px;font-weight:600;border:1px solid #30363d;}"
    "QPushButton:hover{background:#30363d;border-color:#8b949e;}"
)
_COMBO = (
    "QComboBox{background:#0d1117;color:#c9d1d9;border:1px solid #30363d;"
    "border-radius:6px;padding:4px 8px;}"
)


# ═════════════════════════════════════════════════════════════════════════════
#  SHARED ASSEMBLY → MESH BUILDER  (matches visualization.viewer_3d)
# ═════════════════════════════════════════════════════════════════════════════
def build_rocket_regions(state, assembly):
    """Build region meshes from the rocket assembly using the same geometry
    helpers as the Design viewer. Returns (regions, total_length) where
    ``regions`` is {region_name: pv.PolyData}. Falls back to a parametric
    build from the flat state when no assembly is present."""
    if not _PYVISTA:
        return {}, 0.0
    from visualization.viewer_3d import (
        _ogive_profile, _make_surface_of_revolution, _make_tube, _make_frustum,
    )
    from core.components import (
        NoseCone, BodyTube, Transition, TrapezoidalFinSet, InnerTube,
        CenteringRing, Parachute, Bulkhead, EngineBlock, Nozzle, Stage,
    )

    regions: dict[str, list] = {}

    def add(region, mesh):
        if mesh is None:
            return
        regions.setdefault(region, []).append(mesh)

    if assembly is None or not getattr(assembly, "stages", None):
        return _build_parametric(state), max(getattr(state, "length", 1.0), 0.2)

    try:
        total_len = assembly.total_length()
    except Exception:
        total_len = getattr(state, "length", 1.0)
    if total_len <= 0:
        total_len = getattr(state, "length", 1.0) or 1.0

    def comp_region(comp):
        if isinstance(comp, TrapezoidalFinSet):
            return "fins"
        if isinstance(comp, NoseCone):
            return "nose"
        if isinstance(comp, (Bulkhead, CenteringRing, EngineBlock)):
            return "bulkheads"
        if isinstance(comp, Parachute) or getattr(comp, "category", "") == "Recovery":
            return "recovery"
        if isinstance(comp, Nozzle) or getattr(comp, "is_motor_mount", False):
            return "mount"
        if isinstance(comp, InnerTube):
            return "mount"
        return "airframe"

    for comp in assembly.all_components():
        if isinstance(comp, Stage):
            continue
        try:
            z_top = total_len - getattr(comp, "position", 0.0)
            region = comp_region(comp)

            if isinstance(comp, NoseCone):
                r = comp.diameter / 2
                L_nose = comp.length
                L_sh = getattr(comp, "shoulder_length", 0.0)
                z_base = z_top - (L_nose + L_sh)
                if L_sh > 0:
                    r_sh = getattr(comp, "shoulder_diameter", comp.diameter) / 2 or r * 0.95
                    add(region, _make_tube(z_base, L_sh, r_sh))
                z_og = z_base + L_sh
                pz, pr = _ogive_profile(L_nose, r, n=40)
                add(region, _make_surface_of_revolution(pz + z_og, pr))

            elif isinstance(comp, BodyTube):
                r = comp.outer_diameter_val / 2
                L = comp.length
                z_base = z_top - L
                add(region, _make_tube(z_base, L, r))
                for child in comp.children:
                    if isinstance(child, TrapezoidalFinSet):
                        for m in _fin_meshes(child, r, z_base):
                            add("fins", m)

            elif isinstance(comp, Transition):
                L = comp.length
                z_base = z_top - L
                add(region, _make_frustum(z_base, L, comp.aft_diameter / 2,
                                          comp.fore_diameter / 2))

            elif isinstance(comp, TrapezoidalFinSet):
                # top-level fin set (parent not a body tube)
                pr = comp.parent.outer_diameter() / 2 if comp.parent else \
                    getattr(state, "diameter", 0.1) / 2
                z_base = z_top - comp.root_chord
                for m in _fin_meshes(comp, pr, z_base):
                    add("fins", m)

            elif isinstance(comp, Nozzle):
                L = comp.length
                z_base = z_top - L
                add(region, _make_frustum(z_base, L, comp.exit_diameter / 2,
                                          comp.inlet_diameter / 2))

            elif isinstance(comp, InnerTube):
                r = comp.outer_diameter_val / 2
                L = comp.length
                z_base = z_top - L
                add(region, _make_tube(z_base, L, r))

            elif isinstance(comp, (Bulkhead, CenteringRing, EngineBlock)):
                d = getattr(comp, "diameter", getattr(comp, "outer_diameter_val", 0.05))
                add(region, pv.Disc(inner=0.0, outer=max(d / 2, 1e-3),
                                    center=(0, 0, z_top), normal=(0, 0, 1),
                                    r_res=4, c_res=48))
        except Exception as e:
            logger.debug(f"region build skip {comp}: {e}")

    if not regions:
        return _build_parametric(state), total_len

    merged = {}
    for region, meshes in regions.items():
        m = meshes[0]
        for extra in meshes[1:]:
            m = m.merge(extra)
        merged[region] = m
    return merged, total_len


def _fin_meshes(comp, body_r, z_start):
    """Trapezoidal fins matching viewer_3d._create_fin (flat quad per fin)."""
    out = []
    n = getattr(comp, "fin_count", 3) or 3
    height = comp.height
    root = comp.root_chord
    tip = comp.tip_chord
    sweep = getattr(comp, "sweep_angle", 0.0)
    sweep_off = height * math.tan(math.radians(sweep)) if sweep > 0 else 0.0
    for i in range(n):
        ang = (2 * math.pi * i) / n
        pts = np.array([
            [body_r, 0, z_start],
            [body_r, 0, z_start + root],
            [body_r + height, 0, z_start + root - sweep_off],
            [body_r + height, 0, z_start + root - sweep_off - tip],
        ])
        fin = pv.PolyData(pts, np.array([4, 0, 1, 2, 3]))
        fin = fin.rotate_z(math.degrees(ang), point=(0, 0, 0))
        out.append(fin)
    return out


def _build_parametric(state):
    """Fallback parametric build (mirrors viewer_3d._build_simple shape)."""
    from visualization.viewer_3d import (
        _ogive_profile, _make_surface_of_revolution, _make_tube,
    )
    r = max(getattr(state, "diameter", 0.1) / 2, 0.01)
    L = max(getattr(state, "length", 1.0), 0.2)
    body_len = L * 0.8
    nose_len = L * 0.2
    regions = {}
    regions["airframe"] = _make_tube(0, body_len, r)
    pz, pr = _ogive_profile(nose_len, r, n=40)
    regions["nose"] = _make_surface_of_revolution(pz + body_len, pr)
    n_fins = int(getattr(state, "fin_count", 3) or 3)
    if n_fins > 0:
        fh = getattr(state, "diameter", 0.1) * 0.6
        rc = L * 0.1
        tc = rc * 0.5
        class _F:
            fin_count = n_fins; height = fh; root_chord = rc
            tip_chord = tc; sweep_angle = 30.0
        fins = _fin_meshes(_F(), r, 0.0)
        m = fins[0]
        for f in fins[1:]:
            m = m.merge(f)
        regions["fins"] = m
    return regions


class StressViewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._region_meshes: dict[str, object] = {}
        self._total_len = 1.0
        self._body_condition: dict = {}
        self._yield_pa = 276e6
        self._mode = "Von Mises Stress"
        self._component = "Entire Vehicle"
        self._setup_ui()

    # ── UI ────────────────────────────────────────────────────────────────
    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        bar = QFrame()
        bar.setStyleSheet("background:#161b22;border-bottom:1px solid #21262d;")
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(10, 6, 10, 6)
        bl.setSpacing(8)
        bl.addWidget(self._tag("Mode"))
        self.mode_combo = QComboBox(); self.mode_combo.addItems(STRESS_MODES)
        self.mode_combo.setStyleSheet(_COMBO)
        self.mode_combo.currentTextChanged.connect(self._on_mode)
        bl.addWidget(self.mode_combo)
        bl.addWidget(self._tag("Component"))
        self.comp_combo = QComboBox(); self.comp_combo.addItems(COMPONENTS)
        self.comp_combo.setStyleSheet(_COMBO)
        self.comp_combo.currentTextChanged.connect(self._on_component)
        bl.addWidget(self.comp_combo)
        bl.addStretch()
        self.btn_reset = QPushButton(icon("reset_view"), "Reset View")
        self.btn_reset.setStyleSheet(_CTRL_BTN)
        self.btn_reset.clicked.connect(self.reset_view)
        bl.addWidget(self.btn_reset)
        root.addWidget(bar)

        if not _PYVISTA:
            lbl = QLabel(f"3D viewer unavailable: {_PV_ERR}")
            lbl.setStyleSheet("color:#f85149;padding:20px;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            root.addWidget(lbl, 1)
            self.plotter = None
            return

        self.plotter = QtInteractor(self)
        self.plotter.set_background("#0d1117", top="#161b22")
        self.plotter.add_axes(interactive=False, line_width=2)
        root.addWidget(self.plotter.interactor, 1)

        self._empty = QLabel("No Results Available\n\nRun an analysis to view the stress field.")
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty.setStyleSheet("color:#6e7681;font-size:14px;background:transparent;")
        self._empty.setParent(self.plotter.interactor)
        self._empty.show()

    def _tag(self, txt):
        l = QLabel(txt)
        l.setStyleSheet("color:#8b949e;font-size:11px;font-weight:600;")
        return l

    def _side_view(self):
        try:
            self.plotter.camera_position = "xz"
            self.plotter.camera.position = (0, 5, 0)
            self.plotter.camera.focal_point = (0, 0, 0)
            self.plotter.camera.up = (0, 0, 1)
        except Exception:
            pass

    # ── Field synthesis ───────────────────────────────────────────────────
    def _field_for_mesh(self, mesh, region, mode, bc):
        pts = mesh.points
        L = max(self._total_len, 1e-9)
        z = pts[:, 2]
        zf = np.clip((L - z) / L, 0, 1)          # 0 at nose tip → 1 at aft
        theta = np.arctan2(pts[:, 1], pts[:, 0])
        peak = bc.get(_MODE_KEY[mode], 0.0)

        if mode == "Safety Factor":
            vm_local = self._vm_field(zf, theta, region, bc)
            sf = np.where(vm_local > 1e3, self._yield_pa / np.maximum(vm_local, 1e3), 10.0)
            return np.clip(sf, 0, 10)
        if mode == "Von Mises Stress":
            return self._vm_field(zf, theta, region, bc) / 1e6
        if mode == "Axial Stress":
            # Compressive load enters at the motor (aft) and is reacted by the
            # inertia of everything forward of each station, so axial stress is
            # highest at the base and tapers toward the nose tip.
            shape = 0.2 + 0.8 * zf
        elif mode == "Hoop Stress":
            shape = np.where((zf > 0.2) & (zf < 0.85), 1.0, 0.5)
        elif mode == "Shear Stress":
            shape = 0.4 + 0.6 * zf
        elif mode == "Thermal Stress":
            shape = np.exp(-3.0 * zf) * 0.8 + 0.2
        else:
            shape = np.ones_like(zf)
        field = peak * shape / 1e6
        if region == "fins":
            field = field * 1.4
        return field

    def _vm_field(self, zf, theta, region, bc):
        axial = bc.get("axial", 0.0); hoop = bc.get("hoop", 0.0)
        bend = bc.get("bending", 0.0); shear = bc.get("shear", 0.0)
        # Bending moment of the airframe as a free-free beam under the aero
        # normal force at the CP balanced by distributed inertia: the moment is
        # zero at the free ends (nose tip, tail) and peaks near mid-body. A
        # parabolic envelope captures that without the old gaussian that
        # zeroed the entire nose. The von Mises magnitude is the same on the
        # tension and compression fibres, so no cos(theta) lobing is applied
        # (that term only created spurious circumferential stripes).
        bshape = 4.0 * zf * (1.0 - zf)
        # Axial: low at the nose tip, accumulating toward the aft base.
        sx = axial * (0.2 + 0.8 * zf) + bend * bshape
        sy = hoop * np.where((zf > 0.2) & (zf < 0.85), 1.0, 0.5)
        tau = shear * (0.4 + 0.6 * zf)
        vm = np.sqrt(np.abs(sx ** 2 - sx * sy + sy ** 2 + 3 * tau ** 2))
        if region == "fins":
            vm = vm * 1.3   # root-bending concentration at the fin/body joint
        return vm

    # ── Public API ────────────────────────────────────────────────────────
    def update_geometry(self, state, assembly=None):
        if not _PYVISTA or self.plotter is None:
            return
        self._region_meshes, self._total_len = build_rocket_regions(state, assembly)

    def set_result(self, state, assembly, body_condition: dict, yield_pa: float):
        if not _PYVISTA or self.plotter is None:
            return
        self._region_meshes, self._total_len = build_rocket_regions(state, assembly)
        self._body_condition = body_condition or {}
        self._yield_pa = yield_pa or 276e6
        if hasattr(self, "_empty"):
            self._empty.hide()
        self._render()

    def _on_mode(self, mode):
        self._mode = mode
        if self._body_condition:
            self._render()

    def _on_component(self, comp):
        self._component = comp
        if self._body_condition:
            self._render()

    def _visible_regions(self):
        if self._component == "Entire Vehicle":
            return list(self._region_meshes.keys())
        return [r for r in self._region_meshes
                if _REGION_GROUP.get(r) == self._component]

    def _render(self):
        if not _PYVISTA or self.plotter is None or not self._region_meshes:
            return
        self.plotter.clear()
        bc = self._body_condition
        mode = self._mode
        visible = self._visible_regions()
        if not visible:
            visible = list(self._region_meshes.keys())
        title = mode + ("" if mode == "Safety Factor" else " (MPa)")

        region_fields, all_vals = {}, []
        for region in visible:
            mesh = self._region_meshes[region]
            f = self._field_for_mesh(mesh, region, mode, bc)
            region_fields[region] = f
            all_vals.append(f)
        if not all_vals:
            return
        cat = np.concatenate(all_vals)
        if mode == "Safety Factor":
            clim = [max(0.0, float(cat.min())), min(5.0, float(cat.max()) or 5.0)]
            cmap = "jet_r"
        else:
            clim = [0.0, float(cat.max()) or 1.0]
            cmap = "jet"

        peak_region, peak_pt, peak_val = None, None, None
        for region, f in region_fields.items():
            mesh = self._region_meshes[region]
            if mode == "Safety Factor":
                idx = int(np.argmin(f)); better = (peak_val is None or f[idx] < peak_val)
            else:
                idx = int(np.argmax(f)); better = (peak_val is None or f[idx] > peak_val)
            if better:
                peak_val = float(f[idx]); peak_pt = mesh.points[idx]; peak_region = region

        sbar = dict(title=title, title_font_size=12, label_font_size=10,
                    color="#c9d1d9", position_x=0.86, position_y=0.12,
                    width=0.06, height=0.7, fmt="%.1f", n_labels=6)
        first = next(iter(region_fields))
        for region, f in region_fields.items():
            mesh = self._region_meshes[region]
            mesh["stress"] = f
            is_first = (region == first)
            self.plotter.add_mesh(
                mesh, scalars="stress", cmap=cmap, clim=clim,
                show_edges=(region == "fins"), edge_color="#1a1e24",
                line_width=0.4, smooth_shading=True, specular=0.3,
                show_scalar_bar=is_first,
                scalar_bar_args=sbar if is_first else None)

        if peak_pt is not None:
            sphere = pv.Sphere(radius=self._total_len * 0.018, center=peak_pt)
            self.plotter.add_mesh(sphere, color="#ffffff", name="max_marker")
            if mode == "Safety Factor":
                txt = f"Min SF: {peak_val:.2f}\n{_REGION_LABEL.get(peak_region, peak_region)}"
            else:
                txt = f"Peak: {peak_val:.1f} MPa\n{_REGION_LABEL.get(peak_region, peak_region)}"
            self.plotter.add_point_labels(
                [peak_pt], [txt], font_size=12, text_color="#ffffff",
                point_color="#ff3b30", point_size=8, shape_color="#161b22",
                shape_opacity=0.7, always_visible=True, name="max_label")

        try:
            self.plotter.enable_point_picking(callback=self._on_pick,
                                              show_message=False, show_point=False)
        except Exception:
            pass
        self._side_view()
        self.plotter.reset_camera()
        self.plotter.render()

    def _on_pick(self, point):
        if point is None:
            return
        try:
            self.plotter.add_point_labels([point], [self._mode], font_size=10,
                                          text_color="#ffffff", name="hover_label",
                                          always_visible=True)
        except Exception:
            pass

    def reset_view(self):
        if _PYVISTA and self.plotter is not None:
            self._side_view()
            self.plotter.reset_camera()
            self.plotter.render()

    def show_empty(self):
        if hasattr(self, "_empty"):
            self._empty.show()
