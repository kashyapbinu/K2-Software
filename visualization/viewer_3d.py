"""
K2 AeroSim — 3D Rocket Viewer (Component-Based)
Renders the assembled rocket from the component tree.

Coordinate system (matching OpenRocket):
  - Rocket lies along the Z axis.
  - Nose tip is at z = total_length (rightmost / topmost).
  - Nozzle/aft end is at z = 0.
  - Components are stacked top-down: nose first → body tubes → fins/nozzle.

Default camera shows a SIDE VIEW (nose to the right) so it matches ORK.
"""
import logging, math
import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QFrame, QPushButton
from PyQt6.QtCore import Qt, pyqtSignal
from core.components import (NoseCone, BodyTube, Transition, TrapezoidalFinSet,
    InnerTube, CenteringRing, Parachute, LaunchLug, Stage, Bulkhead, EngineBlock, Nozzle)

logger = logging.getLogger("K2.Viewer3D")

RES = 64  # mesh resolution


# ── Geometry helpers ─────────────────────────────────────────

def _ogive_profile(length, radius, n=50):
    """Generate an ogive nose cone profile (z vs r).
    Returns (zs, rs) where:
      zs[0]=0 (base, r=radius) → zs[-1]=length (tip, r≈0).
    """
    if radius <= 0 or length <= 0:
        return np.array([0, length]), np.array([radius, 0])
    rho = (radius**2 + length**2) / (2 * radius)
    zs = np.linspace(0, length, n)
    rs = np.sqrt(np.maximum(rho**2 - zs**2, 0)) - (rho - radius)
    rs = np.clip(rs, 0, radius)
    return zs, rs


def _make_surface_of_revolution(zs, rs, n_theta=RES):
    """Create a surface of revolution mesh from a z-r profile."""
    thetas = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    pts = []
    for z, r in zip(zs, rs):
        for t in thetas:
            pts.append([r * np.cos(t), r * np.sin(t), z])
    pts = np.array(pts)

    faces = []
    n_z = len(zs)
    for i in range(n_z - 1):
        for j in range(n_theta):
            j1 = (j + 1) % n_theta
            a = i * n_theta + j
            b = i * n_theta + j1
            c = (i + 1) * n_theta + j1
            d = (i + 1) * n_theta + j
            faces.extend([4, a, b, c, d])

    # Cap the tip if first or last radius ≈ 0
    if rs[-1] < 1e-6:
        tip_idx = len(pts)
        pts = np.vstack([pts, [[0, 0, zs[-1]]]])
        ring_start = (n_z - 1) * n_theta
        for j in range(n_theta):
            j1 = (j + 1) % n_theta
            faces.extend([3, tip_idx, ring_start + j, ring_start + j1])

    # Cap the base
    if rs[0] > 1e-6:
        base_idx = len(pts)
        pts = np.vstack([pts, [[0, 0, zs[0]]]])
        for j in range(n_theta):
            j1 = (j + 1) % n_theta
            faces.extend([3, base_idx, j1, j])

    return pv.PolyData(pts, faces=np.array(faces))


def _make_tube(z_base, length, radius, n_theta=RES):
    """Create a capped cylinder from z_base to z_base+length."""
    return pv.Cylinder(
        center=(0, 0, z_base + length / 2),
        direction=(0, 0, 1),
        radius=radius,
        height=length,
        resolution=n_theta,
        capping=True
    )


def _make_frustum(z_base, length, r_base, r_top, n=20, n_theta=RES):
    """Create a frustum (transition cone) from z_base to z_base+length.
    r_base is the radius at z_base, r_top is the radius at z_base+length.
    """
    zs = np.linspace(z_base, z_base + length, n)
    rs = np.linspace(r_base, r_top, n)
    return _make_surface_of_revolution(zs, rs, n_theta)


# ── Viewer widget ────────────────────────────────────────────

class Viewer3D(QWidget):
    BG_TOP = "#0d1117"
    BG_BOT = "#161b22"

    COLORS = {
        "nosecone": "#3a8fd6",
        "bodytube": "#2a6191",
        "transition": "#2d8aa5",
        "fins": "#d94f3b",
        "nozzle": "#444444",
        "lug": "#888888",
        "selected": "#fbc02d",  # Bright Gold/Yellow for selection
    }

    def __init__(self, engine, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.mesh_map = {}  # Map mesh object to component
        self.selected_component = None
        self._is_rebuilding = False
        self._setup_ui()
        self.engine.state_changed.connect(self._on_state_changed)

    def select_component(self, component):
        """Highlight the selected component in the 3D/2D view."""
        self.selected_component = component
        self._rebuild(reset_cam=False)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Top toolbar
        top_layout = QHBoxLayout()
        top_layout.setContentsMargins(10, 5, 10, 5)
        self.btn_wireframe = QPushButton("Toggle 2D Line / Wireframe Mode")
        self.btn_wireframe.setCheckable(True)
        self.btn_wireframe.setStyleSheet("background-color: #238636; color: white; padding: 5px 15px; border-radius: 4px; font-weight: bold;")
        self.btn_wireframe.clicked.connect(self._toggle_wireframe)
        top_layout.addWidget(self.btn_wireframe)
        top_layout.addStretch()
        layout.addLayout(top_layout)

        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.NoFrame)
        fl = QVBoxLayout(frame)
        fl.setContentsMargins(0, 0, 0, 0)

        self.plotter = QtInteractor(frame)
        fl.addWidget(self.plotter.interactor)
        layout.addWidget(frame)

        self.plotter.set_background(self.BG_TOP, top=self.BG_BOT)
        self.plotter.add_axes(interactive=False, line_width=2)
        # Default: side view — nose to the right, like OpenRocket
        self._set_side_view()

    def _set_side_view(self):
        """Position camera for side view: looking at the rocket from +Y, nose to the right."""
        self.plotter.camera_position = 'xz'
        self.plotter.camera.position = (0, 5, 0)
        self.plotter.camera.focal_point = (0, 0, 0)
        self.plotter.camera.up = (0, 0, 1)

    def _toggle_wireframe(self, checked):
        self._wireframe_mode = checked
        if checked:
            self.plotter.set_background("white")
            self.plotter.enable_parallel_projection()
            self.btn_wireframe.setText("Switch to 3D Solid Mode")
            self.btn_wireframe.setStyleSheet("background-color: #d73a49; color: white; padding: 5px 15px; border-radius: 4px; font-weight: bold;")
        else:
            self.plotter.set_background(self.BG_TOP, top=self.BG_BOT)
            self.plotter.disable_parallel_projection()
            self.btn_wireframe.setText("Toggle 2D Line / Wireframe Mode")
            self.btn_wireframe.setStyleSheet("background-color: #238636; color: white; padding: 5px 15px; border-radius: 4px; font-weight: bold;")
        
        self._rebuild()

    # ── Rebuild ──────────────────────────────────────────────

    def _on_state_changed(self, state):
        self._rebuild(reset_cam=True)

    def _rebuild(self, reset_cam=True):
        if getattr(self, '_is_rebuilding', False):
            return
        self._is_rebuilding = True
        try:
            self.plotter.clear_actors()
            
            if not getattr(self, '_wireframe_mode', False):
                # Only add axes if they don't exist, multiple calls can be unstable
                if not hasattr(self, '_axes_added'):
                    self.plotter.add_axes(interactive=False, line_width=2)
                    self._axes_added = True

            asm = getattr(self.engine, '_assembly', None)
            if asm is None:
                self._build_simple()
            else:
                if getattr(self, '_wireframe_mode', False):
                    self._build_from_assembly_2d(asm)
                else:
                    self._build_from_assembly(asm)

            if reset_cam:
                self.plotter.reset_camera()
        finally:
            self._is_rebuilding = False

    def _build_from_assembly(self, asm):
        """Build 3D from component assembly.
        
        Stacking convention:
          z_cursor starts at total_length (nose tip) and decreases to 0 (nozzle).
          Each body component occupies [z_cursor - length, z_cursor].
        """
        total_len = asm.total_length()
        if total_len <= 0:
            return

        z_cursor = total_len  # start at the nose tip
        idx = 0
        last_radius = 0.03  # fallback radius

        for stage in asm.stages:
            for comp in stage.children:
                z_cursor, last_radius, idx = self._render_component(
                    comp, z_cursor, last_radius, idx)

        # Simple nozzle cap only when no explicit Nozzle component exists
        has_nozzle = any(isinstance(c, Nozzle) for c in asm.all_components())
        if last_radius > 0.005 and not has_nozzle:
            nozzle = pv.Disc(inner=0, outer=last_radius, center=(0, 0, z_cursor))
            self.plotter.add_mesh(nozzle, color=self.COLORS["nozzle"], name=f'nozzle_{idx}')

    def _render_component(self, comp, z_top, parent_radius, idx):
        """Render one component. z_top is the TOP edge of this component.
        Returns (z_bottom, body_radius_at_bottom, next_idx).
        """
        name = f"c_{idx}"
        idx += 1
        
        is_sel = (comp == self.selected_component)
        sel_color = self.COLORS["selected"] if is_sel else None

        # ── Nose Cone ──
        if isinstance(comp, NoseCone):
            r = comp.diameter / 2
            L_nose = comp.length  # aerodynamic length only
            L_shoulder = getattr(comp, 'shoulder_length', 0.0)
            L_total = L_nose + L_shoulder
            z_base = z_top - L_total

            # Shoulder tube (below the ogive)
            if L_shoulder > 0:
                r_sh = getattr(comp, 'shoulder_diameter', comp.diameter) / 2
                if r_sh <= 0:
                    r_sh = r * 0.95
                sh_mesh = _make_tube(z_base, L_shoulder, r_sh)
                self.plotter.add_mesh(sh_mesh, color=sel_color or self.COLORS["nosecone"],
                                     smooth_shading=True, specular=0.5,
                                     specular_power=20, name=f"{name}_sh")

            # Ogive profile sits above the shoulder
            z_ogive_base = z_base + L_shoulder
            profile_z, profile_r = _ogive_profile(L_nose, r, n=50)
            profile_z = profile_z + z_ogive_base

            mesh = _make_surface_of_revolution(profile_z, profile_r)
            self.plotter.add_mesh(mesh, color=sel_color or self.COLORS["nosecone"],
                                 smooth_shading=True, specular=0.5,
                                 specular_power=20, name=name)
            return z_base, r, idx

        # ── Body Tube ──
        elif isinstance(comp, BodyTube):
            r = comp.outer_diameter_val / 2
            L = comp.length
            z_base = z_top - L

            mesh = _make_tube(z_base, L, r)
            self.plotter.add_mesh(mesh, color=sel_color or self.COLORS["bodytube"],
                                 smooth_shading=True, specular=0.5,
                                 specular_power=20, name=name)

            # Render child fin sets
            for child in comp.children:
                if isinstance(child, TrapezoidalFinSet):
                    idx = self._render_fins(child, z_base, r, idx, is_sel=is_sel)

            return z_base, r, idx

        # ── Transition ──
        elif isinstance(comp, Transition):
            L = comp.length
            z_base = z_top - L
            # Fore = top (connects to wider section above)
            r_top_val = comp.fore_diameter / 2
            # Aft = bottom (connects to narrower section below)
            r_bot_val = comp.aft_diameter / 2

            mesh = _make_frustum(z_base, L, r_bot_val, r_top_val)
            self.plotter.add_mesh(mesh, color=sel_color or self.COLORS["transition"],
                                 smooth_shading=True, specular=0.4, name=name)
            return z_base, r_bot_val, idx

        # ── Fin Set (top-level, outside body tube) ──
        elif isinstance(comp, TrapezoidalFinSet):
            idx = self._render_fins(comp, z_top, parent_radius, idx, is_sel=is_sel)
            return z_top, parent_radius, idx

        # ── Launch Lug ──
        elif isinstance(comp, LaunchLug):
            L = comp.length
            z_base = z_top - L
            mesh = pv.Cylinder(
                center=(parent_radius + 0.004, 0, z_base + L / 2),
                direction=(0, 0, 1),
                radius=comp.outer_diameter_val / 2,
                height=L, resolution=12
            )
            self.plotter.add_mesh(mesh, color=sel_color or self.COLORS["lug"], name=name)
            return z_base, parent_radius, idx

        # ── Nozzle ──
        elif isinstance(comp, Nozzle):
            L = comp.length
            z_base = z_top - L
            r_in = comp.inlet_diameter / 2
            r_th = comp.throat_diameter / 2
            r_ex = comp.exit_diameter / 2

            if comp.nozzle_type == "Boat-Tail":
                # Single tapered frustum
                mesh = _make_frustum(z_base, L, r_ex, r_in)
                self.plotter.add_mesh(mesh, color=sel_color or self.COLORS["nozzle"],
                                     smooth_shading=True, specular=0.6,
                                     specular_power=30, name=name)
            else:
                # Convergent section (top 40%): inlet radius → throat radius
                L_conv = L * 0.4
                L_div = L * 0.6
                z_throat = z_base + L_div  # throat is 60% from bottom

                # Divergent section (bottom 60%): throat → exit
                mesh_div = _make_frustum(z_base, L_div, r_ex, r_th)
                self.plotter.add_mesh(mesh_div, color=sel_color or self.COLORS["nozzle"],
                                     smooth_shading=True, specular=0.6,
                                     specular_power=30, name=f"{name}_div")

                # Convergent section (top 40%): inlet → throat
                mesh_conv = _make_frustum(z_throat, L_conv, r_th, r_in)
                self.plotter.add_mesh(mesh_conv, color=sel_color or self.COLORS["nozzle"],
                                     smooth_shading=True, specular=0.6,
                                     specular_power=30, name=f"{name}_conv")

            return z_base, r_ex, idx

        # Inner/recovery components don't affect the body line
        return z_top, parent_radius, idx

    def _render_fins(self, comp, body_z_base, body_radius, idx, is_sel=False):
        """Render a fin set at the AFT (bottom) end of its parent body tube.
        
        In OpenRocket, fin axialoffset with method='bottom' means measured from
        the bottom of the parent. We place fins starting from body_z_base (bottom).
        """
        for i in range(comp.fin_count):
            angle_rad = (2 * math.pi * i) / comp.fin_count
            fin = self._create_fin(
                body_radius, comp.height, comp.root_chord,
                comp.tip_chord, comp.sweep_angle, angle_rad,
                body_z_base
            )
            is_sel = (comp == self.selected_component)
            f_color = self.COLORS["selected"] if is_sel else self.COLORS["fins"]
            self.plotter.add_mesh(fin, color=f_color,
                                 smooth_shading=True, specular=0.3,
                                 name=f"fin_{idx}_{i}")
        return idx + 1

    def _create_fin(self, body_r, height, root_chord, tip_chord, sweep_deg, angle, z_start, is_sel=False):
        """Create a single trapezoidal fin as a solid hex mesh.
        
        Convention (matching OpenRocket):
          - Root leading edge at z_start (aft/bottom end of body tube)
          - Root trailing edge at z_start + root_chord (forward/upward)
          - Tip is swept back by sweep_angle from the root leading edge
          
        Wait — OpenRocket's fin convention:
          - root_chord starts at the axial position of the fin set
          - Leading edge = forward (toward nose) = higher z
          - Trailing edge = aft (toward nozzle) = lower z
          
        So root LE is at z_start + root_chord, root TE at z_start.
        """
        sweep_offset = height * math.tan(math.radians(sweep_deg)) if sweep_deg > 0 else 0

        # Fin in the XZ plane, then rotated around Z
        # z_start is the AFT end of the body tube (bottom of fin root)
        pts = np.array([
            [body_r,          0, z_start],                                      # root TE (aft)
            [body_r,          0, z_start + root_chord],                         # root LE (forward)
            [body_r + height, 0, z_start + root_chord - sweep_offset],          # tip LE
            [body_r + height, 0, z_start + root_chord - sweep_offset - tip_chord],  # tip TE
        ])

        # Add visual thickness
        thick = max(0.002, getattr(comp, 'thickness', 0.003) if 'comp' in dir() else 0.003)
        thick = 0.003  # visual minimum
        pts_inner = pts.copy()
        pts_outer = pts.copy()
        for p in pts_inner:
            p[1] -= thick / 2
        for p in pts_outer:
            p[1] += thick / 2

        all_pts = np.vstack([pts_outer, pts_inner])  # 0-3 outer, 4-7 inner
        faces = np.array([
            4, 0, 1, 2, 3,  # outer face
            4, 7, 6, 5, 4,  # inner face
            4, 0, 4, 5, 1,  # edge 1
            4, 1, 5, 6, 2,  # edge 2
            4, 2, 6, 7, 3,  # edge 3
            4, 3, 7, 4, 0,  # edge 4
        ])

        fin = pv.PolyData(all_pts, faces=faces)
        fin = fin.rotate_z(np.degrees(angle), point=(0, 0, 0))
        return fin

    # ── 2D Outline Rendering ─────────────────────────────────

    def _draw_loop(self, pts, color, line_width=2.0):
        if not pts: return
        poly = pv.PolyData(np.array(pts))
        n = len(pts)
        poly.lines = np.array([n+1] + list(range(n)) + [0])
        self.plotter.add_mesh(poly, color=color, line_width=line_width, render_lines_as_tubes=False)

    def _draw_lines(self, pts_list, color, line_width=2.0):
        if not pts_list: return
        pts = []
        lines = []
        idx = 0
        for p1, p2 in pts_list:
            pts.extend([p1, p2])
            lines.extend([2, idx, idx+1])
            idx += 2
        poly = pv.PolyData(np.array(pts))
        poly.lines = np.array(lines)
        self.plotter.add_mesh(poly, color=color, line_width=line_width, render_lines_as_tubes=False)

    def _build_from_assembly_2d(self, asm):
        total_len = asm.total_length()
        
        for comp in asm.all_components():
            if isinstance(comp, Stage): continue
            
            is_sel = (comp == self.selected_component)
            outer_color = "gold" if is_sel else "blue"
            inner_color = "gold" if is_sel else "red"
            l_width = 4.0 if is_sel else 2.0
            inner_width = 4.0 if is_sel else 1.0
            
            z_top = total_len - getattr(comp, '_position', 0.0)
            
            if isinstance(comp, NoseCone):
                r = comp.diameter / 2
                L_nose = comp.length
                L_shoulder = getattr(comp, 'shoulder_length', 0.0)
                z_base = z_top - (L_nose + L_shoulder)
                z_ogive_base = z_base + L_shoulder
                
                # Shoulder
                if L_shoulder > 0:
                    r_sh = getattr(comp, 'shoulder_diameter', comp.diameter) / 2
                    if r_sh <= 0: r_sh = r * 0.95
                    pts = [[r_sh, 0, z_base], [r_sh, 0, z_ogive_base],
                           [-r_sh, 0, z_ogive_base], [-r_sh, 0, z_base]]
                    self._draw_loop(pts, outer_color, line_width=l_width)
                
                # Ogive
                profile_z, profile_r = _ogive_profile(L_nose, r, n=50)
                profile_z = profile_z + z_ogive_base
                
                pts = []
                for pz, pr in zip(profile_z, profile_r):
                    pts.append([pr, 0, pz])
                for pz, pr in reversed(list(zip(profile_z, profile_r))):
                    pts.append([-pr, 0, pz])
                self._draw_loop(pts, outer_color, line_width=l_width)
                
            elif isinstance(comp, BodyTube):
                r = comp.outer_diameter_val / 2
                L = comp.length
                z_base = z_top - L
                pts = [[r, 0, z_base], [r, 0, z_top], [-r, 0, z_top], [-r, 0, z_base]]
                self._draw_loop(pts, outer_color, line_width=l_width)
                
            elif isinstance(comp, Transition):
                L = comp.length
                z_base = z_top - L
                r_top = comp.fore_diameter / 2
                r_bot = comp.aft_diameter / 2
                pts = [[r_bot, 0, z_base], [r_top, 0, z_top], [-r_top, 0, z_top], [-r_bot, 0, z_base]]
                self._draw_loop(pts, outer_color, line_width=l_width)
                
            elif isinstance(comp, TrapezoidalFinSet):
                r = 0.0
                if comp.parent:
                    r = comp.parent.outer_diameter() / 2
                
                Cr = comp.root_chord
                Ct = comp.tip_chord
                s = comp.height
                sweep_offset = s * math.tan(math.radians(comp.sweep_angle)) if comp.sweep_angle > 0 else 0
                
                # Match 3D _create_fin coordinate convention:
                # z_start = aft/trailing edge of fin root (bottom of parent body tube)
                # Root LE (forward) = z_start + Cr
                # Tip LE = z_start + Cr - sweep_offset  (swept aft from root LE)
                if comp.parent and isinstance(comp.parent, BodyTube):
                    parent_z_top = total_len - getattr(comp.parent, '_position', 0.0)
                    z_start = parent_z_top - comp.parent.component_length()
                else:
                    # Fallback: derive from fin's own _position
                    # _position is the LE, so z_start = LE - Cr = TE
                    z_start = z_top - Cr
                
                z_root_le = z_start + Cr       # root leading edge (forward)
                z_root_te = z_start             # root trailing edge (aft)
                z_tip_le  = z_start + Cr - sweep_offset
                z_tip_te  = z_start + Cr - sweep_offset - Ct
                
                pts_up = [
                    [r,     0, z_root_te],
                    [r,     0, z_root_le],
                    [r + s, 0, z_tip_le],
                    [r + s, 0, z_tip_te],
                ]
                self._draw_loop(pts_up, outer_color, line_width=l_width)
                
                pts_dn = [
                    [-r,     0, z_root_te],
                    [-r,     0, z_root_le],
                    [-r - s, 0, z_tip_le],
                    [-r - s, 0, z_tip_te],
                ]
                self._draw_loop(pts_dn, outer_color, line_width=l_width)
                
            elif isinstance(comp, InnerTube):
                r = comp.outer_diameter_val / 2
                L = comp.length
                z_base = z_top - L
                pts = [[r, 0, z_base], [r, 0, z_top], [-r, 0, z_top], [-r, 0, z_base]]
                self._draw_loop(pts, inner_color, line_width=inner_width)
                
            elif isinstance(comp, (Bulkhead, EngineBlock, CenteringRing)):
                L = getattr(comp, 'thickness', 0.005)
                d = getattr(comp, 'diameter', getattr(comp, 'outer_diameter_val', 0.05))
                r = d / 2
                z_base = z_top - L
                pts = [[r, 0, z_base], [r, 0, z_top], [-r, 0, z_top], [-r, 0, z_base]]
                self._draw_loop(pts, inner_color, line_width=inner_width)
                
            elif isinstance(comp, Parachute):
                L = 0.08
                r = 0.03
                if comp.parent: 
                    parent_diam = getattr(comp.parent, 'inner_diameter', getattr(comp.parent, 'outer_diameter_val', 0.06))
                    r = parent_diam / 2 * 0.8
                z_base = z_top - L
                pts = [[r, 0, z_base], [r, 0, z_top], [-r, 0, z_top], [-r, 0, z_base]]
                self._draw_loop(pts, inner_color, line_width=inner_width)
                self._draw_lines([ [[r, 0, z_base], [-r, 0, z_top]], [[-r, 0, z_base], [r, 0, z_top]] ], inner_color, inner_width)

            elif isinstance(comp, Nozzle):
                L = comp.length
                z_base = z_top - L
                r_in = comp.inlet_diameter / 2
                r_th = comp.throat_diameter / 2
                r_ex = comp.exit_diameter / 2

                if comp.nozzle_type == "Boat-Tail":
                    # Simple taper
                    pts = [[r_in, 0, z_top], [r_ex, 0, z_base],
                           [-r_ex, 0, z_base], [-r_in, 0, z_top]]
                    self._draw_loop(pts, outer_color, line_width=l_width)
                else:
                    # Convergent-Divergent profile
                    L_conv = L * 0.4
                    L_div = L * 0.6
                    z_throat = z_base + L_div
                    pts = [
                        [r_in, 0, z_top],
                        [r_th, 0, z_throat],
                        [r_ex, 0, z_base],
                        [-r_ex, 0, z_base],
                        [-r_th, 0, z_throat],
                        [-r_in, 0, z_top],
                    ]
                    self._draw_loop(pts, outer_color, line_width=l_width)
                    # Throat line
                    self._draw_lines([[[r_th, 0, z_throat], [-r_th, 0, z_throat]]], outer_color, inner_width)

        # Draw CG and CP markers
        cg_z = total_len - asm.compute_cg()
        cp_z = total_len - asm.compute_cp()
        
        R_m = 0.015
        self._draw_lines([ [[-R_m, 0, cg_z], [R_m, 0, cg_z]], [[0, 0, cg_z-R_m], [0, 0, cg_z+R_m]] ], "blue", 2)
        self._draw_lines([ [[-R_m, 0, cp_z], [R_m, 0, cp_z]], [[0, 0, cp_z-R_m], [0, 0, cp_z+R_m]] ], "red", 2)


    # ── Fallback simple render ───────────────────────────────

    def _build_simple(self):
        """Fallback: build from flat state fields when no assembly exists."""
        s = self.engine.state
        r = s.diameter / 2
        body_len = s.length * 0.8
        nose_len = s.length * 0.2

        if r <= 0 or s.length <= 0:
            return

        # Body
        body = _make_tube(0, body_len, r)
        self.plotter.add_mesh(body, color=self.COLORS["bodytube"],
                             smooth_shading=True, specular=0.5, name='body')

        # Nose
        zs, rs = _ogive_profile(nose_len, r)
        zs = body_len + zs  # shift above body
        mesh = _make_surface_of_revolution(zs, rs)
        self.plotter.add_mesh(mesh, color=self.COLORS["nosecone"],
                             smooth_shading=True, specular=0.5, name='nose')

        # Fins
        if s.fin_count > 0:
            fh = s.diameter * 0.6
            rc = s.length * 0.1
            tc = rc * 0.5
            for i in range(s.fin_count):
                angle = (2 * math.pi * i) / s.fin_count
                fin = self._create_fin(r, fh, rc, tc, 30, angle, 0)
                self.plotter.add_mesh(fin, color=self.COLORS["fins"],
                                     smooth_shading=True, name=f'fin_{i}')

        # Nozzle
        nozzle = pv.Cone(center=(0, 0, -0.015), direction=(0, 0, -1),
                        height=0.03, radius=r * 0.5, resolution=30)
        self.plotter.add_mesh(nozzle, color=self.COLORS["nozzle"],
                             smooth_shading=True, name='nozzle')

    # ── Trajectory Visualization ─────────────────────────────

    def render_trajectory(self, history):
        """
        Render the 3D flight trajectory from simulation history.
        Uses phase-colored segments with an apogee marker.

        Args:
            history: HistoryManager instance with recorded flight data.
        """
        if history is None or history.count < 2:
            return

        self.clear_trajectory()

        altitudes = history.get_values("altitude")
        phases = history.get_values("phase")
        velocities = history.get_values("velocity")

        # Phase → color mapping
        phase_colors = {
            "Ignition": "#f0883e",
            "Boost": "#f0883e",
            "Coast": "#d29922",
            "Apogee": "#58a6ff",
            "Drogue Descent": "#7ee787",
            "Main Descent": "#3fb950",
            "Landed": "#3fb950",
        }

        # Build trajectory line (vertical flight: x=0, y=0, z=altitude)
        # Offset to the side of the rocket for visibility
        x_offset = 0.3
        points = np.array([[x_offset, 0, alt] for alt in altitudes])

        # Add as a polyline
        n = len(points)
        lines = np.zeros((n - 1) * 3, dtype=int)
        for i in range(n - 1):
            lines[i * 3] = 2
            lines[i * 3 + 1] = i
            lines[i * 3 + 2] = i + 1

        poly = pv.PolyData(points, lines=lines)
        self.plotter.add_mesh(poly, color="#58a6ff", line_width=3,
                             render_lines_as_tubes=True,
                             name="trajectory_line")

        # Apogee marker
        max_alt = max(altitudes) if altitudes else 0
        apogee_idx = altitudes.index(max_alt) if max_alt > 0 else 0
        if max_alt > 0:
            marker = pv.Sphere(radius=max_alt * 0.015,
                              center=(x_offset, 0, max_alt))
            self.plotter.add_mesh(marker, color="#58a6ff",
                                smooth_shading=True, name="apogee_marker")
            self.plotter.add_point_labels(
                np.array([[x_offset + 0.05, 0, max_alt]]),
                [f"Apogee: {max_alt:.0f}m"],
                font_size=12, text_color="#58a6ff",
                point_size=0, name="apogee_label"
            )

        # Velocity vector arrows at key points (every 10% of flight)
        step = max(1, n // 10)
        arrow_pts = []
        arrow_dirs = []
        for i in range(0, n, step):
            if abs(velocities[i]) > 0.1:
                arrow_pts.append([x_offset + 0.1, 0, altitudes[i]])
                v_norm = velocities[i] / max(abs(max(velocities, key=abs)), 1)
                arrow_dirs.append([0, 0, v_norm * 0.2])

        if arrow_pts:
            arrows = pv.PolyData(np.array(arrow_pts))
            arrows["vectors"] = np.array(arrow_dirs)
            arrows.set_active_vectors("vectors")
            glyphs = arrows.glyph(orient="vectors", scale=True,
                                  factor=max_alt * 0.15)
            self.plotter.add_mesh(glyphs, color="#d29922",
                                name="velocity_vectors")

    def clear_trajectory(self):
        """Remove trajectory visualization actors."""
        for name in ["trajectory_line", "apogee_marker", "apogee_label",
                      "velocity_vectors"]:
            try:
                self.plotter.remove_actor(name)
            except:
                pass

    # ── Utilities ────────────────────────────────────────────

    def reset_camera(self):
        self._set_side_view()
        self.plotter.reset_camera()

    def closeEvent(self, event):
        try:
            self.plotter.close()
        except:
            pass
        super().closeEvent(event)
