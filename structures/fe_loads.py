"""
K2 AeroSim — Static load model for the CalculiX airframe
==========================================================
Turns a LoadCase into nodal forces (and nodal temperatures) on the shell
mesh from structures.meshing.

A rocket in flight is a free body. Thrust, drag and the aerodynamic normal
force are reacted by the acceleration of the rocket's own mass, not by a
support. The model therefore
  1. lumps the vehicle mass onto the mesh nodes — each shell component
     carries its own component mass, internal components (motor, payload,
     recovery gear, rings …) are spread over the rings they span;
  2. applies the external loads of the condition as nodal forces;
  3. adds the d'Alembert force −m_i (a + α × r_i) of the rigid-body motion
     those loads produce ("inertia relief"), so the load set sums to zero
     force AND zero moment;
  4. leaves the solver a statically determinate 3-2-1 support whose
     reactions are then zero to round-off.

The old deck applied only the shell's own weight (GRAV, toward the nose —
tension where thrust compresses), clamped the tail and also pinned the nose
tip; everything else was added afterwards by hand formulas, so CalculiX
carried under 1 % of the stress it was credited with.

Frame: mesh frame — z from the nose tip (0) aft. The aerodynamic normal
force acts along +y, i.e. perpendicular to fin 0: that fin carries its full
lifting load (the structural worst case for a fin root).
"""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("K2.FEM.Loads")

T_REF = 293.15          # K — stress-free temperature


# ── Mesh as read back from structure_mesh.inp ─────────────────────────────────

@dataclass
class FEMesh:
    ids: np.ndarray                 # (N,) node ids
    xyz: np.ndarray                 # (N, 3) coordinates, m
    index: dict                     # node id -> row
    elements: dict                  # element id -> (n1, n2, n3, n4) node ids
    elsets: dict                    # name -> [element ids]
    nsets: dict                     # name -> set(node ids)
    # per element (same order as elem_ids): area, outward unit normal, centroid
    elem_ids: np.ndarray = None
    conn: np.ndarray = None         # (E, 4) node ROWS
    area: np.ndarray = None
    normal: np.ndarray = None
    centroid: np.ndarray = None

    @classmethod
    def read(cls, path) -> "FEMesh":
        nodes, elements, elsets, nsets = {}, {}, {}, {}
        mode = cur = None
        for line in Path(path).read_text(encoding="ascii", errors="replace").splitlines():
            s = line.strip()
            if not s or s.startswith("**"):
                continue
            if s.startswith("*"):
                head = s.upper().replace(" ", "")
                mode = None
                if head.startswith("*NODE,") or head == "*NODE":
                    mode = "node"
                elif head.startswith("*ELEMENT"):
                    mode = "elem"
                elif head.startswith("*NSET") or head.startswith("*ELSET"):
                    key = "NSET" if head.startswith("*NSET") else "ELSET"
                    m = re.search(key + r"=([^,]+)", head)
                    if m:
                        cur = m.group(1)
                        mode = key.lower()
                        (nsets if mode == "nset" else elsets).setdefault(
                            cur, set() if mode == "nset" else [])
                continue
            parts = [p for p in s.split(",") if p.strip()]
            try:
                if mode == "node" and len(parts) >= 4:
                    nodes[int(parts[0])] = (float(parts[1]), float(parts[2]), float(parts[3]))
                elif mode == "elem" and len(parts) >= 5:
                    elements[int(parts[0])] = tuple(int(p) for p in parts[1:5])
                elif mode == "nset":
                    nsets[cur].update(int(p) for p in parts)
                elif mode == "elset":
                    elsets[cur].extend(int(p) for p in parts)
            except ValueError:
                continue
        ids = np.array(sorted(nodes), dtype=np.int64)
        mesh = cls(ids=ids, xyz=np.array([nodes[n] for n in ids], dtype=float),
                   index={int(n): k for k, n in enumerate(ids)},
                   elements=elements, elsets=elsets, nsets=nsets)
        mesh._element_geometry()
        return mesh

    def _element_geometry(self):
        self.elem_ids = np.array(sorted(self.elements), dtype=np.int64)
        self.conn = np.array([[self.index[n] for n in self.elements[e]] for e in self.elem_ids],
                             dtype=np.int64)
        p = self.xyz[self.conn]                          # (E, 4, 3)
        nrm = np.zeros((len(p), 3))
        for i in range(4):                               # Newell: |n| = 2·area
            a, b = p[:, i], p[:, (i + 1) % 4]
            nrm[:, 0] += (a[:, 1] - b[:, 1]) * (a[:, 2] + b[:, 2])
            nrm[:, 1] += (a[:, 2] - b[:, 2]) * (a[:, 0] + b[:, 0])
            nrm[:, 2] += (a[:, 0] - b[:, 0]) * (a[:, 1] + b[:, 1])
        mag = np.linalg.norm(nrm, axis=1)
        self.area = 0.5 * mag
        self.normal = nrm / np.maximum(mag, 1e-30)[:, None]
        self.centroid = p.mean(axis=1)

    def rows(self, node_ids):
        return np.array([self.index[n] for n in node_ids if n in self.index], dtype=np.int64)

    def elem_rows(self, elem_ids):
        pos = {int(e): k for k, e in enumerate(self.elem_ids)}
        return np.array([pos[e] for e in elem_ids if e in pos], dtype=np.int64)


@dataclass
class LoadModel:
    forces: np.ndarray                    # (N, 3) nodal forces incl. inertia relief, N
    temperatures: Optional[np.ndarray]    # (N,) K, or None
    support: tuple                        # (A, B, C) node ids of the 3-2-1 support
    masses: np.ndarray                    # (N,) lumped vehicle mass, kg
    summary: dict = field(default_factory=dict)


# ── Public entry point ────────────────────────────────────────────────────────

def build_static_loads(mesh: FEMesh, info, assembly, lc, density_of,
                       cfd_pressure: Optional[dict] = None) -> LoadModel:
    """Nodal loads of ``lc`` on ``mesh``, balanced by inertia relief.

    ``density_of(section) -> kg/m³`` gives the structural density used when a
    component reports no mass of its own. ``cfd_pressure`` maps element id →
    gauge surface pressure (Pa) from a CFD run (optional).
    """
    n = len(mesh.ids)
    ext = np.zeros((n, 3))
    s = {"condition": lc.name}
    stations = _Stations(mesh, info)
    masses = _nodal_masses(mesh, info, assembly, lc, density_of, stations, s)

    name = lc.name
    flight = name in ("Max Thrust", "Max-Q")
    q = _dynamic_pressure(lc) if flight else 0.0
    d_ref = assembly.get_reference_diameter()
    A_ref = math.pi * (d_ref / 2) ** 2
    s.update(q_Pa=q, ref_area_m2=A_ref)

    if name not in ("Recovery Shock", "Thermal"):          # thrust-carrying cases
        T = max(float(lc.axial_force), 0.0)
        if T > 0:
            ring = stations.ring_near(lc.motor_aft_m) if lc.motor_aft_m > 0 \
                else mesh.rows(info.thrust_ring)
            ext[ring, 2] -= T / len(ring)
        s["thrust_N"] = T
    if flight and q > 0:
        s.update(_drag(mesh, info, lc, q, A_ref, ext, stations, cfd_pressure))
        s.update(_aero_normal(mesh, info, lc, q, A_ref, ext, stations))
    if name == "Recovery Shock":
        F = max(float(lc.axial_force), 0.0) * max(lc.dynamic_amplification, 1.0)
        z_att = _attachment_z(assembly, stations)
        ring = stations.ring_near(z_att)
        ext[ring, 2] -= F / len(ring)
        s.update(recovery_force_N=F, attachment_z_m=float(stations.z[stations.nearest(z_att)]))
    if lc.internal_pressure:
        _internal_pressure(mesh, info, lc.internal_pressure, ext, stations)
        s["internal_pressure_Pa"] = float(lc.internal_pressure)
    if cfd_pressure:
        s["cfd_pressure_elements"] = _surface_pressure(mesh, cfd_pressure, ext)

    temps = _temperatures(mesh, info, lc, stations, s)
    forces = ext + _inertia_relief(mesh.xyz, masses, ext, s)
    support = _support_nodes(mesh, info)
    s["external_force_N"] = float(np.linalg.norm(ext.sum(axis=0)))
    return LoadModel(forces=forces, temperatures=temps, support=support,
                     masses=masses, summary=s)


# ── Airframe stations (rings) ─────────────────────────────────────────────────

class _Stations:
    """Airframe rings sorted by z, with their node rows."""

    def __init__(self, mesh, info):
        zs, rows = [], []
        for z, ids in info.body_stations:
            r = mesh.rows(ids)
            if len(r):
                zs.append(z)
                rows.append(r)
        self.z = np.array(zs)
        self.rows = rows
        self.radius = np.array([np.hypot(*mesh.xyz[r, :2].T).max() for r in rows])
        # tributary half-spacings, for spreading line loads / masses
        mids = (self.z[1:] + self.z[:-1]) / 2
        self.lo = np.concatenate([[-np.inf], mids])
        self.hi = np.concatenate([mids, [np.inf]])

    def nearest(self, z):
        return int(np.argmin(np.abs(self.z - z)))

    def ring_near(self, z):
        return self.rows[self.nearest(z)]

    def spread(self, total, za, zb):
        """[(ring rows, share)] of a quantity spread uniformly over [za, zb]."""
        if zb - za < 1e-9:
            return [(self.ring_near(0.5 * (za + zb)), total)]
        w = np.clip(np.minimum(self.hi, zb) - np.maximum(self.lo, za), 0, None)
        if w.sum() <= 0:
            return [(self.ring_near(0.5 * (za + zb)), total)]
        w = w / w.sum()
        return [(self.rows[k], total * w[k]) for k in np.nonzero(w)[0]]


def _add_to_rings(target, parts, value_fn):
    for rows, share in parts:
        target[rows] += value_fn(rows, share)


# ── Mass model ────────────────────────────────────────────────────────────────

def _own_mass(comp):
    try:
        return max(float(comp.computed_mass()), 0.0)
    except Exception:
        return 0.0


def _nodal_masses(mesh, info, assembly, lc, density_of, stations, s):
    from core.components import Stage
    m = np.zeros(len(mesh.ids))
    meshed = {id(sec.component) for sec in info.sections if sec.component is not None}
    shell = 0.0
    for sec in info.sections:
        er = mesh.elem_rows(mesh.elsets.get(sec.name, ()))
        if not len(er):
            continue
        geo = mesh.area[er] * sec.thickness
        own = _own_mass(sec.component) if (sec.component is not None and sec.kind != "step") else 0.0
        me = own * geo / geo.sum() if own > 0 and geo.sum() > 0 else geo * density_of(sec)
        np.add.at(m, mesh.conn[er].ravel(), np.repeat(me / 4.0, 4))
        shell += float(me.sum())

    internal = 0.0
    for c in assembly.all_components():
        if isinstance(c, Stage) or id(c) in meshed:
            continue
        p = getattr(c, "parent", None)
        if (p is not None and getattr(p, "_override_includes_children", False)
                and getattr(p, "override_mass", None) is not None):
            continue
        mc = _own_mass(c)
        if mc <= 0:
            continue
        za = float(c.position)
        _add_to_rings(m, stations.spread(mc, za, za + max(c.component_length(), 0.0)),
                      lambda rows, share: share / len(rows))
        internal += mc

    motor = max(float(getattr(lc, "vehicle_mass_kg", 0.0)) - shell - internal, 0.0)
    if motor > 1e-6:
        if lc.motor_aft_m > 0 and lc.motor_length_m > 0:
            za, zb = lc.motor_aft_m - lc.motor_length_m, lc.motor_aft_m
        else:
            zb = float(stations.z[-1])
            za = zb - 0.15 * (zb - float(stations.z[0]))
        _add_to_rings(m, stations.spread(motor, za, zb), lambda rows, share: share / len(rows))
    s.update(vehicle_mass_kg=float(m.sum()), shell_mass_kg=shell,
             internal_mass_kg=internal, motor_mass_kg=motor,
             cg_m=float((m * mesh.xyz[:, 2]).sum() / max(m.sum(), 1e-12)))
    return m


# ── External loads ────────────────────────────────────────────────────────────

def _dynamic_pressure(lc):
    if lc.dynamic_pressure > 0:
        return float(lc.dynamic_pressure)
    if lc.mach <= 0:
        return 0.0
    try:
        from cfd.solvers.base import isa_conditions
        P, T, rho = isa_conditions(lc.altitude_m)
        V = lc.mach * math.sqrt(1.4 * 287.05 * T)
        return 0.5 * rho * V * V
    except Exception:
        return 0.0


def _tributary_area(mesh, er):
    """Nodal share (rows, area) of the elements ``er`` (¼ of each element)."""
    rows = mesh.conn[er].ravel()
    return rows, np.repeat(mesh.area[er] / 4.0, 4)


def _drag(mesh, info, lc, q, A_ref, ext, stations, cfd_pressure):
    """Drag q·A·Cd: base drag on the aft ring, the rest (friction + pressure)
    spread over the whole wetted surface by area; both act aft (+z)."""
    from physics.drag_tables import base_cd
    cd = lc.drag_coefficient if lc.drag_coefficient > 0 else 0.5
    D = lc.drag_force_N if lc.drag_force_N > 0 else q * A_ref * cd
    D_base = min(D, q * A_ref * max(base_cd(lc.mach), 0.0))
    aft = mesh.rows(info.aft_ring)
    ext[aft, 2] += D_base / len(aft)
    rest = D - D_base
    if cfd_pressure:
        # the mapped CFD pressure already carries the pressure drag
        cfd_axial = sum(p * mesh.area[k] * -mesh.normal[k, 2]
                        for k, p in _cfd_rows(mesh, cfd_pressure))
        rest = max(rest - max(cfd_axial, 0.0), 0.0)
    er = np.arange(len(mesh.elem_ids))
    rows, a = _tributary_area(mesh, er)
    np.add.at(ext[:, 2], rows, rest * a / a.sum())
    return {"drag_N": D, "base_drag_N": D_base, "cd": cd}


def _lateral_on_ring(mesh, rows, F):
    """Nodal forces summing to F·ŷ on a ring, as a cos-distributed radial
    load (the slender-body pressure pattern: pushes in on the windward side,
    pulls out on the lee side, no ovalling)."""
    xy = mesh.xyz[rows, :2]
    r = np.maximum(np.hypot(xy[:, 0], xy[:, 1]), 1e-12)
    cos_t, sin_t = xy[:, 0] / r, xy[:, 1] / r
    w = (sin_t * sin_t).sum()
    if w <= 1e-12:
        return np.tile([0.0, F / len(rows), 0.0], (len(rows), 1))
    c = F / w
    return np.column_stack([c * sin_t * cos_t, c * sin_t * sin_t, np.zeros(len(rows))])


def _aero_normal(mesh, info, lc, q, A_ref, ext, stations):
    """Aerodynamic normal force at angle of attack, from the same Barrowman /
    Galejs / OpenRocket fin terms the flight model uses (physics.aerodynamics),
    distributed where it acts:
      * slender-body lift 2·q·sinα·dS/dz along the profile (nose, transitions,
        boat tails — zero on constant-diameter tubes);
      * Galejs body lift per component, uniform along it;
      * fin lift on the fin panels, each fin by cos²φ of its roll angle.
    ``dynamic_amplification`` (gust DAF for Max-Q) scales the load itself.
    A CFD normal force (``lateral_force``) larger than this total scales the
    whole distribution up to it — conservative, as the old handling was."""
    from physics.aerodynamics import compute_body_lift_cn, compute_fin_cn_alpha
    alpha = math.radians(lc.angle_of_attack_deg)
    if abs(alpha) < 1e-9:
        return {"normal_force_N": 0.0}
    daf = max(lc.dynamic_amplification, 1.0)
    sa = math.sin(alpha)
    parts = {}
    total_ext, ext = ext, np.zeros_like(ext)     # built apart, added at the end

    # slender-body term along the airframe profile
    S = math.pi * stations.radius ** 2
    dN = 2.0 * q * sa * daf * np.diff(S)
    for k, f in enumerate(dN):
        for kk in (k, k + 1):
            rows = stations.rows[kk]
            ext[rows] += _lateral_on_ring(mesh, rows, 0.5 * f)
    parts["slender_body_N"] = float(dN.sum())

    # Galejs body lift, per axisymmetric component
    body_lift = 0.0
    for sec in info.sections:
        if sec.kind in ("fin", "step"):
            continue
        zs = sorted({float(mesh.xyz[mesh.index[n], 2]) for n in mesh.nsets.get(sec.name, ())
                     if n in mesh.index})
        if len(zs) < 2:
            continue
        k0, k1 = stations.nearest(zs[0]), stations.nearest(zs[-1])
        R, Z = stations.radius[k0:k1 + 1], stations.z[k0:k1 + 1]
        planform = float(((R[:-1] + R[1:]) * np.diff(Z)).sum())      # ∫ 2r dz
        F = q * A_ref * compute_body_lift_cn(planform, A_ref, alpha, lc.mach) * daf
        for rows, share in stations.spread(F, zs[0], zs[-1]):
            ext[rows] += _lateral_on_ring(mesh, rows, share)
        body_lift += F
    parts["body_lift_N"] = body_lift

    # fins
    fin_total = 0.0
    for sec in info.fin_sections:
        fs = sec.component
        er = mesh.elem_rows(mesh.elsets.get(sec.name, ()))
        if fs is None or not len(er):
            continue
        rows_all = mesh.conn[er].ravel()
        body_r = float(np.hypot(*mesh.xyz[rows_all, :2].T).min())
        cna = compute_fin_cn_alpha(fs.fin_count, fs.height, fs.root_chord, fs.tip_chord,
                                   body_r, math.radians(fs.sweep_angle), lc.mach, alpha)
        N_set = q * math.pi * body_r ** 2 * cna * sa * daf
        phis = [2 * math.pi * k / fs.fin_count for k in range(fs.fin_count)]
        norm = sum(math.cos(p) ** 2 for p in phis)
        ang = np.arctan2(mesh.centroid[er, 1], mesh.centroid[er, 0])
        which = np.array([int(np.argmin([abs(math.remainder(a - p, 2 * math.pi)) for p in phis]))
                          for a in ang])
        for k, phi in enumerate(phis):
            ek = er[which == k]
            if not len(ek) or norm <= 0:
                continue
            Fk = N_set / norm * math.cos(phi)
            nk = np.array([-math.sin(phi), math.cos(phi), 0.0])
            rows, a = _tributary_area(mesh, ek)
            np.add.at(ext, rows, np.outer(Fk * a / a.sum(), nk))
        fin_total += N_set
    parts["fin_normal_N"] = fin_total
    total = float(parts["slender_body_N"] + body_lift + fin_total)
    scale = 1.0
    if lc.lateral_force > total > 0:
        scale = lc.lateral_force / total
        parts["cfd_normal_scale"] = scale
    total_ext += ext * scale
    parts["normal_force_N"] = total * scale
    parts.update(alpha_deg=lc.angle_of_attack_deg, daf=daf)
    return parts


def _attachment_z(assembly, stations):
    """Recovery harness attachment: the shock cord (else parachute) position,
    else 35 % of the airframe length."""
    from core.components import ShockCord, Parachute
    for kind in (ShockCord, Parachute):
        for c in assembly.all_components():
            if isinstance(c, kind):
                return float(c.position)
    return float(stations.z[0] + 0.35 * (stations.z[-1] - stations.z[0]))


def _internal_pressure(mesh, info, p, ext, stations):
    """Internal pressure on the airframe wall (outward, p·A·n̂) plus the
    end-closure load on the open aft ring — so the wall sees the axial
    pressure-vessel stress p·r/2t as well as the hoop p·r/t."""
    er = mesh.elem_rows([e for sec in info.sections if sec.kind != "fin"
                         for e in mesh.elsets.get(sec.name, ())])
    F = p * mesh.area[er, None] * mesh.normal[er]
    np.add.at(ext, mesh.conn[er].ravel(), np.repeat(F / 4.0, 4, axis=0))
    for ring, sign in ((mesh.rows(info.aft_ring), 1.0), (mesh.rows(info.nose_ring), -1.0)):
        if len(ring):
            r = float(np.hypot(*mesh.xyz[ring, :2].T).max())
            ext[ring, 2] += sign * p * math.pi * r * r / len(ring)


def _cfd_rows(mesh, cfd_pressure):
    pos = {int(e): k for k, e in enumerate(mesh.elem_ids)}
    return [(pos[e], p) for e, p in cfd_pressure.items() if e in pos]


def _surface_pressure(mesh, cfd_pressure, ext):
    """External (gauge) surface pressure pushes inward: −p·A·n̂."""
    rows = _cfd_rows(mesh, cfd_pressure)
    for k, p in rows:
        f = -p * mesh.area[k] * mesh.normal[k]
        ext[mesh.conn[k]] += f / 4.0
    return len(rows)


def _temperatures(mesh, info, lc, stations, s):
    """Nodal temperatures. Thermal case: stagnation temperature at the nose
    tip blending to the recovery temperature by 2 % of the length, recovery
    temperature along the body, +15 % of the rise over the motor section
    (same profile as the analytical thermal model). Otherwise a uniform
    lc.delta_T if set. Uniform heating of a free body is stress-free — only
    gradients and material (CTE) mismatch load it, as they should."""
    if lc.name == "Thermal":
        try:
            from cfd.solvers.base import isa_conditions
            _, T_amb, _ = isa_conditions(lc.altitude_m)
        except Exception:
            T_amb = 288.15
        mach = lc.mach if lc.mach > 0 else 0.0
        T_rec = T_amb * (1 + 0.89 * 0.2 * mach ** 2)
        T_stag = T_amb * (1 + 0.2 * mach ** 2)
        z0, z1 = float(stations.z[0]), float(stations.z[-1])
        f = np.clip((mesh.xyz[:, 2] - z0) / max(z1 - z0, 1e-9), 0.0, 1.0)
        T = np.full(len(f), T_rec)
        tip = f < 0.02
        T[tip] = T_stag + (T_rec - T_stag) * f[tip] / 0.02
        aft = f > 0.80
        T[aft] = T_rec + (T_rec - T_amb) * 0.15 * (f[aft] - 0.80) / 0.20
        s.update(T_recovery_K=T_rec, T_stagnation_K=T_stag, T_max_K=float(T.max()))
        return T
    if lc.delta_T:
        s["delta_T_K"] = float(lc.delta_T)
        return np.full(len(mesh.ids), T_REF + float(lc.delta_T))
    return None


# ── Inertia relief & support ──────────────────────────────────────────────────

def _inertia_relief(xyz, m, ext, s):
    """d'Alembert forces −m_i (a + α × d_i) of the rigid-body acceleration
    the external loads produce; with them the loads sum to zero force and
    zero moment about any point, exactly (same masses, same inertia)."""
    M = float(m.sum())
    if M <= 0:
        return np.zeros_like(ext)
    c = (m[:, None] * xyz).sum(axis=0) / M
    d = xyz - c
    R = ext.sum(axis=0)
    Mo = np.cross(d, ext).sum(axis=0)
    I = (m[:, None, None] * ((d * d).sum(axis=1)[:, None, None] * np.eye(3)
                             - d[:, :, None] * d[:, None, :])).sum(axis=0)
    a = R / M
    alpha = np.linalg.solve(I, Mo)
    s.update(accel_axial_g=float(-a[2] / 9.80665),          # + = forward (toward nose)
             accel_lateral_g=float(a[1] / 9.80665),
             angular_accel_rad_s2=float(np.linalg.norm(alpha)))
    return -m[:, None] * (a + np.cross(alpha, d))


def _support_nodes(mesh, info):
    """3-2-1 support on the aft airframe ring: A (x, y, z), B opposite A
    (y, z), C at 90° (z) — six constraints, no more: with balanced loads
    the reactions are round-off, so the support adds no stress."""
    ring = [n for n in info.aft_ring if n in mesh.index] or \
        [int(n) for n in mesh.ids[np.argsort(-mesh.xyz[:, 2])[:8]]]
    xy = np.array([mesh.xyz[mesh.index[n], :2] for n in ring])
    ang = np.arctan2(xy[:, 1], xy[:, 0])

    def pick(target, taken):
        order = np.argsort([abs(math.remainder(a - target, 2 * math.pi)) for a in ang])
        return next(ring[k] for k in order if ring[k] not in taken)

    A = pick(0.0, ())
    B = pick(math.pi, (A,))
    C = pick(math.pi / 2, (A, B))
    return A, B, C
