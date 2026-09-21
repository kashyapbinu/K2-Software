"""
K2 AeroSim — 3D Structural Stress Viewer
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

from ui import theme

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
    f"QPushButton{{background:{theme.RAISED};color:{theme.TEXT};padding:5px 12px;"
    f"border-radius:6px;font-weight:600;border:1px solid {theme.LINE};}}"
    f"QPushButton:hover{{background:{theme.LINE};border-color:{theme.TEXT_DIM};}}"
)
_COMBO = (
    f"QComboBox{{background:{theme.BG};color:{theme.TEXT};border:1px solid {theme.LINE};"
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
        nose_profile, _make_surface_of_revolution, _make_tube, _make_frustum,
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
                pz, pr = nose_profile(
                    getattr(comp, "shape", "Ogive"), L_nose, r, n=40
                )
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
        self._fin_stress_pa = 0.0     # fin root bending from fin_analysis
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
        bar.setStyleSheet(f"background:{theme.PANEL};border-bottom:1px solid {theme.RAISED};")
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
        _basis = QLabel("⚠ Analytical estimate — smooth field, not nodal FEA")
        _basis.setStyleSheet(f"color:{theme.WARN};font-size:10px;font-weight:600;")
        _basis.setToolTip(
            "This contour is a smooth analytical reconstruction from the closed-form "
            "stress solution (axial / bending / hoop / shear distributed by beam-shape "
            "functions), not a mesh-based FEA field. It does NOT resolve geometric stress "
            "concentrations at fin roots, couplers, motor-mount or bulkhead interfaces — "
            "those are accounted for as scalar stress-concentration factors (Kt) in the "
            "reported safety factor, not shown spatially here.")
        bl.addWidget(_basis)
        bl.addSpacing(10)
        self.btn_reset = QPushButton(icon("reset_view"), "Reset View")
        self.btn_reset.setStyleSheet(_CTRL_BTN)
        self.btn_reset.clicked.connect(self.reset_view)
        bl.addWidget(self.btn_reset)
        root.addWidget(bar)

        if not _PYVISTA:
            lbl = QLabel(f"3D viewer unavailable: {_PV_ERR}")
            lbl.setStyleSheet(f"color:{theme.ERR};padding:20px;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            root.addWidget(lbl, 1)
            self.plotter = None
            return

        self.plotter = QtInteractor(self)
        self.plotter.set_background(theme.BG, top=theme.PANEL)
        # Orientation axes are added only once a result is rendered (see
        # _render). Showing them in the empty state put a stray gizmo + X/Y/Z
        # labels in the middle of the "no results" message.
        root.addWidget(self.plotter.interactor, 1)

        self._empty = QLabel("Please run a simulation to view results.")
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty.setStyleSheet(f"color:{theme.TEXT_FAINT};font-size:14px;background:transparent;")
        self._empty.setParent(self.plotter.interactor)
        self._empty.show()

    def _tag(self, txt):
        l = QLabel(txt)
        l.setStyleSheet(f"color:{theme.TEXT_DIM};font-size:11px;font-weight:600;")
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
    @staticmethod
    def _bc_amplification(bc):
        """Kt · DAF that the condition solver folded into its von Mises.

        compute_for_condition reports σ_vm = Kt·DAF·√(σx² − σx·σy + σy² + 3τ²)
        but only exposes the raw components. Rebuilding the field from the
        components alone dropped that factor, so the contour peak read up to
        2.3× below the number in the stress panel. Recover the factor from
        the ratio of reported σ_vm to the raw combination of the components."""
        sx = bc.get("axial", 0.0) + bc.get("longitudinal", 0.0) + bc.get("bending", 0.0) \
            + bc.get("thermal", 0.0)
        sy = bc.get("hoop", 0.0)
        tau = bc.get("shear", 0.0)
        raw = math.sqrt(max(sx * sx - sx * sy + sy * sy + 3 * tau * tau, 0.0))
        vm = bc.get("von_mises", 0.0)
        if raw <= 0.0 or vm <= 0.0:
            return 1.0
        return max(vm / raw, 1.0)

    def _fin_span_frac(self, mesh):
        """0 at the fin root (min radius) → 1 at the tip (max radius)."""
        pts = mesh.points
        rr = np.hypot(pts[:, 0], pts[:, 1])
        r0, r1 = float(rr.min()), float(rr.max())
        if r1 - r0 < 1e-9:
            return np.zeros_like(rr)
        return np.clip((rr - r0) / (r1 - r0), 0, 1)

    def _field_for_mesh(self, mesh, region, mode, bc):
        pts = mesh.points
        L = max(self._total_len, 1e-9)
        z = pts[:, 2]
        zf = np.clip((L - z) / L, 0, 1)          # 0 at nose tip → 1 at aft
        theta = np.arctan2(pts[:, 1], pts[:, 0])
        peak = bc.get(_MODE_KEY[mode], 0.0)

        if mode == "Safety Factor":
            vm_local = self._vm_field(zf, theta, region, bc, mesh)
            sf = np.where(vm_local > 1e3, self._yield_pa / np.maximum(vm_local, 1e3), 10.0)
            return np.clip(sf, 0, 10)
        if mode == "Von Mises Stress":
            return self._vm_field(zf, theta, region, bc, mesh) / 1e6
        if region == "fins":
            # Fins carry their own load path (root bending from fin_analysis);
            # the body's axial / hoop / thermal do not flow through them.
            if mode == "Thermal Stress":
                return np.full(len(zf), peak * 0.85 / 1e6)
            if mode == "Hoop Stress":
                return np.zeros(len(zf))
            sf_span = self._fin_span_frac(mesh)
            fin_peak = self._fin_stress_pa if self._fin_stress_pa > 0 else peak
            if mode == "Shear Stress":
                return fin_peak * 0.25 * (1.0 - sf_span) / 1e6
            return fin_peak * (1.0 - sf_span) ** 2 / 1e6
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
        return peak * shape / 1e6

    def _vm_field(self, zf, theta, region, bc, mesh=None):
        if region == "fins":
            # Fin von Mises = root bending from fin_analysis (uniaxial plate
            # bending, max at the root, ∝ (1 − η)² for a distributed load).
            # Was 1.3× the BODY field — an invented number that put the
            # "Peak" marker on the fins regardless of the fin result.
            fin_peak = self._fin_stress_pa
            if fin_peak <= 0 or mesh is None:
                fin_peak = bc.get("von_mises", 0.0) * 0.6
                return np.full(len(zf), fin_peak)
            return fin_peak * (1.0 - self._fin_span_frac(mesh)) ** 2

        axial = bc.get("axial", 0.0) + bc.get("longitudinal", 0.0) + bc.get("thermal", 0.0)
        hoop = bc.get("hoop", 0.0)
        bend = bc.get("bending", 0.0); shear = bc.get("shear", 0.0)
        amp = self._bc_amplification(bc)
        # Bending moment of the airframe as a free-free beam under the aero
        # normal force at the CP balanced by distributed inertia: zero at the
        # free ends (nose tip, tail), peak near mid-body → parabolic envelope.
        # Bending is a signed fibre stress: it ADDS to the compressive axial
        # on one side of the tube (cos θ = −1) and subtracts on the other, so
        # the compression-side fibre carries |σ_ax + σ_b| and the tension side
        # |σ_ax − σ_b|. Summing both as positive everywhere (the old field)
        # over-stated the tension side.
        bshape = 4.0 * zf * (1.0 - zf)
        # Axial compression: low at the nose tip, accumulating toward the aft
        # base where the thrust enters.
        sx = -axial * (0.2 + 0.8 * zf) - bend * bshape * np.cos(theta)
        sy = hoop * np.where((zf > 0.2) & (zf < 0.85), 1.0, 0.5)
        tau = shear * (0.4 + 0.6 * zf)
        vm = amp * np.sqrt(np.abs(sx ** 2 - sx * sy + sy ** 2 + 3 * tau ** 2))
        # The panel's σ_vm is the critical-section value with every component
        # at its peak. The shape functions never coincide (axial peaks aft,
        # bending mid-body), so pin the field maximum to the reported peak —
        # the contour then agrees with the number on screen.
        target = bc.get("von_mises", 0.0)
        vmax = float(vm.max()) if len(vm) else 0.0
        if target > 0 and vmax > 0:
            vm = vm * (target / vmax)
        return vm

    # ── Public API ────────────────────────────────────────────────────────
    def update_geometry(self, state, assembly=None):
        if not _PYVISTA or self.plotter is None:
            return
        self._region_meshes, self._total_len = build_rocket_regions(state, assembly)

    def set_result(self, state, assembly, body_condition: dict, yield_pa: float,
                   fin_stress_pa: float = 0.0):
        """``fin_stress_pa`` = fin root bending stress (Pa) from
        structures.workstation.fin_analysis; drives the fin region's field."""
        if not _PYVISTA or self.plotter is None:
            return
        self._region_meshes, self._total_len = build_rocket_regions(state, assembly)
        self._body_condition = body_condition or {}
        self._fin_stress_pa = float(fin_stress_pa or 0.0)
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
                    color=theme.TEXT, position_x=0.86, position_y=0.12,
                    width=0.06, height=0.7, fmt="%.1f", n_labels=6)
        first = next(iter(region_fields))
        for region, f in region_fields.items():
            mesh = self._region_meshes[region]
            mesh["stress"] = f
            is_first = (region == first)
            self.plotter.add_mesh(
                mesh, scalars="stress", cmap=cmap, clim=clim,
                show_edges=(region == "fins"), edge_color=theme.PANEL,
                line_width=0.4, smooth_shading=True, specular=0.3,
                show_scalar_bar=is_first,
                scalar_bar_args=sbar if is_first else None)

        if peak_pt is not None:
            sphere = pv.Sphere(radius=self._total_len * 0.018, center=peak_pt)
            self.plotter.add_mesh(sphere, color=theme.TEXT_BRIGHT, name="max_marker")
            if mode == "Safety Factor":
                txt = f"Min SF: {peak_val:.2f}\n{_REGION_LABEL.get(peak_region, peak_region)}"
            else:
                txt = f"Peak: {peak_val:.1f} MPa\n{_REGION_LABEL.get(peak_region, peak_region)}"
            self.plotter.add_point_labels(
                [peak_pt], [txt], font_size=12, text_color=theme.TEXT_BRIGHT,
                point_color="#ff3b30", point_size=8, shape_color=theme.PANEL,
                shape_opacity=0.7, always_visible=True, name="max_label")

        try:
            self.plotter.enable_point_picking(callback=self._on_pick,
                                              show_message=False, show_point=False)
        except Exception:
            pass
        # Orientation axes belong with an actual result, not the empty state.
        try:
            self.plotter.add_axes(interactive=False, line_width=2)
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
                                          text_color=theme.TEXT_BRIGHT, name="hover_label",
                                          always_visible=True)
        except Exception:
            pass

    def reset_view(self):
        if _PYVISTA and self.plotter is not None:
            self._side_view()
            self.plotter.reset_camera()
            self.plotter.render()

    def show_empty(self):
        # Clear any rendered mesh + the orientation gizmo so the empty state is
        # just the message — no leftover 3D clutter.
        if _PYVISTA and self.plotter is not None:
            try:
                self.plotter.clear()
                self.plotter.hide_axes()
            except Exception:
                pass
        if hasattr(self, "_empty"):
            self._empty.show()
