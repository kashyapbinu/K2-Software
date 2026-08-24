"""
ONERA M6 wing — surface pressure vs Schmitt & Charpin (AGARD AR-138, 1979).
============================================================================

The standard transonic CFD validation case: a swept, untwisted, symmetric-section
semi-span wing at M = 0.8395, alpha = 3.06 deg (Test 2308), with measured surface
pressure coefficients at seven span stations. The measurement uncertainty quoted
for this condition is +/-0.02 in Cp.

The experimental Cp files in ``validation/data/onera_m6`` are redistributed from
the NASA NPARC Alliance Validation Archive
(https://www.grc.nasa.gov/www/wind/valid/m6wing/m6wing.html), which digitised
them from the AGARD report. Columns are ``x/c, Cp, sigma, sigma``.

What this case tests that the rocket cases cannot
-------------------------------------------------
It drives K2's **external-CAD** path (``cfd.external_geometry``) rather than the
rocket component tree, on a geometry with a shock on it. A rocket template can
never exercise the CAD import, the frontal-area sizing or the transonic shock
capture at the same time; this does.

Geometry is generated here rather than shipped as a STEP file: the published
planform (semi-span 1.1963 m, MAC 0.64607 m, aspect ratio 3.8, taper 0.562,
leading-edge sweep 30 deg) determines the root and tip chords exactly, and the
ONERA D section coordinates ship with the Cp data. The generated planform is
checked against the published trailing-edge sweep of 15.8 deg as an assertion
that the reconstruction is right.
"""
from __future__ import annotations

import math
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "onera_m6"

# ── published case definition ─────────────────────────────────────────────────

MACH = 0.8395
ALPHA_DEG = 3.06
REYNOLDS_MAC = 11.72e6
CP_UNCERTAINTY = 0.02

SEMI_SPAN = 1.1963          # m
MAC = 0.64607               # m
TAPER = 0.562
LE_SWEEP_DEG = 30.0
TE_SWEEP_DEG_PUBLISHED = 15.8

# Span stations of the pressure sections, as fractions of the semi-span.
SECTIONS = {1: 0.20, 2: 0.44, 3: 0.65, 4: 0.80, 5: 0.90, 6: 0.95, 7: 0.99}


def root_chord() -> float:
    """Root chord implied by the published MAC and taper ratio.

    MAC = (2/3)*c_root*(1 + λ + λ²)/(1 + λ) for a straight-tapered wing.
    """
    lam = TAPER
    f = (2.0 / 3.0) * (1.0 + lam + lam ** 2) / (1.0 + lam)
    return MAC / f


def tip_chord() -> float:
    return TAPER * root_chord()


def chord_at(y: float) -> float:
    """Local chord at spanwise station y [m]."""
    return root_chord() + (tip_chord() - root_chord()) * (y / SEMI_SPAN)


def le_x_at(y: float) -> float:
    """Leading-edge x station at spanwise y [m]."""
    return y * math.tan(math.radians(LE_SWEEP_DEG))


def te_sweep_deg() -> float:
    """Trailing-edge sweep of the reconstructed planform, for self-checking."""
    dx = (le_x_at(SEMI_SPAN) + tip_chord()) - root_chord()
    return math.degrees(math.atan(dx / SEMI_SPAN))


# ── section coordinates ───────────────────────────────────────────────────────

def load_airfoil() -> list:
    """ONERA D section as [(x/c, z/c), ...] for the upper surface, tip-first.

    ``airfoil.txt`` from the NPARC archive lists the half-section from the
    leading edge to the trailing edge; the wing is symmetric, so the lower
    surface is the mirror image.
    """
    pts = []
    for line in (DATA_DIR / "airfoil.txt").read_text().splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            pts.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue
    if not pts:
        raise RuntimeError(f"no airfoil coordinates in {DATA_DIR/'airfoil.txt'}")
    return pts


def load_experimental_cp(section: int, upper: bool) -> list:
    """[(x/c, Cp), ...] measured at a span station.

    **The files store -Cp, not Cp**, following the plotting convention of the
    AGARD report and the NPARC archive (suction upward). The sign is flipped here
    so everything downstream is in Cp.

    This is not guesswork about the archive's intent — it is what the data says.
    At the first upper-surface tap (x/c = 0.0006, essentially the leading-edge
    stagnation point) the file reads -0.726, and by x/c = 0.05 it reads +1.19.
    Read as Cp that is impossible twice over: negative pressure at a stagnation
    point, and Cp > 1 on a suction surface at 3 deg. Negated it becomes +0.73 at
    the stagnation point and -1.19 at the suction peak, which is the M6 pressure
    distribution as published.
    """
    tag = "u" if upper else "l"
    path = DATA_DIR / f"cp{section}{tag}.ex"
    rows = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            rows.append((float(parts[0]), -float(parts[1])))
        except ValueError:
            continue
    rows.sort()
    return rows


# ── geometry generation ───────────────────────────────────────────────────────

def planform_area(mirrored: bool = True) -> float:
    """Wing planform area [m^2] — the reference area the Cp/force data uses.

    K2's CFD sizes its reference area from the *frontal* projection, which is the
    right default for a rocket and meaningless for a wing (it lands ~10x small
    here), so this is passed in as an explicit override.
    """
    semi = 0.5 * (root_chord() + tip_chord()) * SEMI_SPAN
    return 2.0 * semi if mirrored else semi


def build_m6_stl(out_path: Path, n_span: int = 60, mirror: bool = True) -> Path:
    """Write a closed STL of the M6 wing.

    Chord along +x, span along +y, thickness along z — so the CFD flow axis is
    x, which must be passed to ``analyze_cad`` explicitly: auto-detection picks
    the longest extent, and on a wing that is the span, not the chord.

    ``mirror`` builds both halves (span -b/2..+b/2) instead of the semi-span
    model, and defaults on. The published experiment is a half-model on a tunnel
    wall, i.e. the root sits on a reflection plane — but K2's external-CAD path
    has no symmetry boundary: it subtracts the imported solid from a wind tunnel
    and makes every face of it a wall. A semi-span STL therefore models a wing
    that simply *stops* at the root, so the flow relieves around that plane as if
    it were a second tip and the inboard loading comes out low. Mirroring restores
    the physics the reflection plane stands in for, at the cost of twice the mesh.
    Both tips are capped, and with ``mirror`` there is no root plane to cap.
    """
    import numpy as np
    import pyvista as pv

    foil = load_airfoil()
    # Full section loop: upper surface LE->TE, then lower surface TE->LE.
    upper = [(x, z) for x, z in foil]
    lower = [(x, -z) for x, z in reversed(foil[1:-1])]
    loop = upper + lower
    n_loop = len(loop)

    if mirror:
        half = np.linspace(0.0, SEMI_SPAN, n_span)
        ys = np.concatenate([-half[:0:-1], half])   # -b/2 .. +b/2, root once
    else:
        ys = np.linspace(0.0, SEMI_SPAN, n_span)
    n_span_total = len(ys)

    pts = []
    for y in ys:
        # Symmetric about the root: sweep and taper are functions of |y|.
        c, x0 = chord_at(abs(y)), le_x_at(abs(y))
        for xc, zc in loop:
            pts.append([x0 + xc * c, y, zc * c])
    pts = np.asarray(pts)

    faces = []
    for i in range(n_span_total - 1):
        for j in range(n_loop):
            j1 = (j + 1) % n_loop
            a = i * n_loop + j
            b = i * n_loop + j1
            cc = (i + 1) * n_loop + j1
            d = (i + 1) * n_loop + j
            faces.extend([3, a, b, cc])
            faces.extend([3, a, cc, d])

    # Cap the two end sections with a triangle fan: both tips when mirrored, the
    # root plane and the tip on a semi-span build.
    for i, flip in ((0, True), (n_span_total - 1, False)):
        base = i * n_loop
        cx = float(np.mean(pts[base:base + n_loop, 0]))
        cz = float(np.mean(pts[base:base + n_loop, 2]))
        centre = len(pts)
        pts = np.vstack([pts, [[cx, ys[i], cz]]])
        for j in range(n_loop):
            j1 = (j + 1) % n_loop
            tri = [centre, base + j, base + j1]
            if flip:
                tri = [centre, base + j1, base + j]
            faces.extend([3] + tri)

    mesh = pv.PolyData(pts, np.asarray(faces)).clean().triangulate()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.save(str(out_path))
    return out_path


# ── extracting Cp from an SU2 surface solution ────────────────────────────────

def span_frame(surface_vtk) -> tuple:
    """``(y_root, y_tip)`` of the wing *as solved*, read off the surface points.

    The station coordinates cannot be taken from the analytic planform: K2's CAD
    import aligns whatever it is given to the CFD frame and recentres it, so on a
    mirrored wing the root lands at y=0 with the tips at +/-b/2, and on a semi-span
    one the root lands at -b/2 — measured, not assumed. Working from the solved
    mesh's own extent makes the comparison immune to any rigid transform the
    import applies.

    The root is found as the thickest section, not as an end: thickness is 4% of
    a chord that tapers outboard, so maximum |z| locates it wherever it sits. It
    has to be found rather than assumed because the root is at one end on a
    semi-span import but at *mid-span* on the mirrored one, where both ends are
    tips. The tip returned is then the end furthest from the root — either one for
    a mirrored wing, which is symmetric about the root by construction.
    """
    import numpy as np
    import pyvista as pv

    mesh = surface_vtk if hasattr(surface_vtk, "points") else pv.read(str(surface_vtk))
    pts = np.asarray(mesh.points)
    ys, zs = pts[:, 1], np.abs(pts[:, 2])
    y_lo, y_hi = float(ys.min()), float(ys.max())

    n_bins = 24
    edges = np.linspace(y_lo, y_hi, n_bins + 1)
    idx = np.clip(np.digitize(ys, edges) - 1, 0, n_bins - 1)
    thickest, y_root = -1.0, y_lo
    for b in range(n_bins):
        sel = idx == b
        if not sel.any():
            continue
        t = float(zs[sel].max())
        if t > thickest:
            thickest, y_root = t, float(ys[sel].mean())

    y_tip = y_hi if abs(y_hi - y_root) >= abs(y_root - y_lo) else y_lo
    return y_root, y_tip


def surface_cp_at_section(surface_vtk, y_over_b: float, upper: bool,
                          x_over_c: list, span: tuple = None) -> list:
    """Interpolate solved surface Cp onto the experimental x/c stations.

    Takes a thin spanwise band around the section (rather than a zero-thickness
    slice) so an unstructured surface always yields points, then separates upper
    from lower by the sign of z relative to the local section mid-line.

    x/c comes from the band's own leading and trailing edge rather than from the
    analytic sweep line, for the same reason ``span_frame`` exists — and because
    it is what the experimentalists normalised by.
    """
    import numpy as np
    import pyvista as pv

    mesh = surface_vtk if hasattr(surface_vtk, "points") else pv.read(str(surface_vtk))
    if "Pressure_Coefficient" not in mesh.point_data:
        raise RuntimeError("surface solution has no Pressure_Coefficient field")

    y_root, y_tip = span if span is not None else span_frame(mesh)
    y_target = y_root + y_over_b * (y_tip - y_root)
    band = 0.01 * abs(y_tip - y_root)
    pts = np.asarray(mesh.points)
    cp = np.asarray(mesh.point_data["Pressure_Coefficient"])

    sel = np.abs(pts[:, 1] - y_target) <= band
    if sel.sum() < 10:
        band *= 4.0
        sel = np.abs(pts[:, 1] - y_target) <= band
    if sel.sum() < 4:
        raise RuntimeError(f"no surface points near y/b={y_over_b} "
                           f"(y={y_target:.4f} m, root={y_root:.4f}, tip={y_tip:.4f})")

    xs, zs, cps = pts[sel, 0], pts[sel, 2], cp[sel]
    x_le, x_te = float(xs.min()), float(xs.max())
    c = x_te - x_le
    if c <= 0.0:
        raise RuntimeError(f"degenerate chord at y/b={y_over_b}")
    xc = (xs - x_le) / c
    side = zs > 0.0 if upper else zs < 0.0
    xc, cps = xc[side], cps[side]
    if xc.size < 3:
        raise RuntimeError(f"too few {'upper' if upper else 'lower'} points "
                           f"at y/b={y_over_b}")

    order = np.argsort(xc)
    xc, cps = xc[order], cps[order]
    return [float(np.interp(x, xc, cps)) for x in x_over_c]


def _integrate_cp(rows) -> float:
    """Trapezoidal integral of Cp over x/c for one surface, [(x/c, Cp), ...]."""
    rows = sorted(rows)
    total = 0.0
    for (x0, c0), (x1, c1) in zip(rows, rows[1:]):
        total += 0.5 * (c0 + c1) * (x1 - x0)
    return total


def section_cn(surface_vtk, section: int, span: tuple = None) -> tuple:
    """``(cn_k2, cn_experiment)`` — sectional normal-force coefficient.

    ``cn = ∮ (Cp_lower - Cp_upper) d(x/c)``, both curves integrated on the
    experimental tap stations so the two sides see identical abscissae.

    This is the load the section actually carries, and it answers a question the
    pointwise RMSE cannot: an under-resolved solution smears the suction peak,
    which inflates RMSE without necessarily losing much lift, while a solution
    that has the *loading* wrong is wrong about the physics. Reporting both
    separates "the mesh is coarse" from "the answer is wrong".
    """
    exp_u = load_experimental_cp(section, True)
    exp_l = load_experimental_cp(section, False)
    xs_u = [x for x, _ in exp_u]
    xs_l = [x for x, _ in exp_l]

    k2_u = surface_cp_at_section(surface_vtk, SECTIONS[section], True, xs_u, span)
    k2_l = surface_cp_at_section(surface_vtk, SECTIONS[section], False, xs_l, span)

    cn_k2 = _integrate_cp(list(zip(xs_l, k2_l))) - _integrate_cp(list(zip(xs_u, k2_u)))
    cn_exp = _integrate_cp(exp_l) - _integrate_cp(exp_u)
    return cn_k2, cn_exp


def cp_rmse(surface_vtk, section: int, upper: bool, span: tuple = None) -> tuple:
    """(RMSE, n_points) of solved vs measured Cp at one section and surface."""
    exp = load_experimental_cp(section, upper)
    xs = [x for x, _ in exp]
    ref = [c for _, c in exp]
    got = surface_cp_at_section(surface_vtk, SECTIONS[section], upper, xs, span)
    n = len(ref)
    rmse = math.sqrt(sum((g - r) ** 2 for g, r in zip(got, ref)) / n)
    return rmse, n


# ── the solve ─────────────────────────────────────────────────────────────────

# Surface element sizes [m] for the refinement study. MAC is 0.646 m, so these
# are roughly 32 and 65 cells along the chord.
WALL_SIZES = (0.020, 0.010)


def run_su2(work_dir: Path, refinement: str = "fine",
            max_iterations: int = 4000, mirror: bool = True,
            wall_size: float | None = 0.010, resume: bool = True):
    """Run the M6 case through K2's external-CAD CFD path. Returns CFDResult.

    ``resume`` re-parses a completed solve in ``work_dir`` instead of repeating
    it — an hour of solver time across the two levels of the refinement study,
    which the report would otherwise pay on every regeneration. Each level has
    its own directory (keyed by wall size), and the completion test is the same
    one the sweep uses: the solver's exit banner plus a history that postdates
    the mesh it sits next to.

    ``wall_size`` sets the surface element size directly, because neither of the
    other two knobs can refine this case:

      * the refinement presets are calibrated on a rocket's frontal radius and
        land near 100k cells on this wing — about 17 cells along the chord, at
        which the suction peak comes out half its measured depth;
      * ``target_element_count`` spreads its estimate over the whole 24 x 48 x 48 m
        wind tunnel, so 900,000 elements asks for a *coarser* wall than the
        medium preset and gets clamped back to it.

    Under-resolution here is not a small error, and it is not distinguishable
    from a physics error in a single run — which is why the benchmark solves the
    case at two sizes and reports the trend.
    """
    from cfd.external_geometry import analyze_cad
    from cfd.solvers.base import CFDConfig
    from cfd.solvers.su2_solver import SU2Solver

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    stl = build_m6_stl(work_dir / "onera_m6.stl", mirror=mirror)

    # flow_axis must be forced: the span is the longest extent, so "auto" would
    # point the freestream along the wing.
    info = analyze_cad(stl, flow_axis="x", work_dir=work_dir, units="m")

    cfg = CFDConfig(
        mach=MACH, angle_of_attack_deg=ALPHA_DEG, altitude_m=0.0,
        mesh_refinement=refinement, work_dir=work_dir,
        turbulence_model="Euler", euler_analytic_friction=True,
        max_iterations=max_iterations,
        # Wing, not rocket: the frontal-area default is ~10x too small here and
        # would make CL/CD unreadable. Cp is unaffected either way.
        ref_area_override=planform_area(mirrored=mirror),
        ref_length_override=MAC,
        custom_wall_size=wall_size,
        # The default 20x lands the farfield 24 m from a 1.2 m wing and spends the
        # cell budget on empty air; 10x is still past the 6-span floor the mesher
        # enforces for a thin, wide body.
        domain_radius_scale=10.0,
    )
    # external_cad is the file the mesher meshes, not a flag — the parametric
    # rocket path is bypassed by it being set at all. cad_info travels as a plain
    # dict, which is what the mesher and the friction build-up both expect.
    aligned = Path(info.preview_stl or stl)
    cfg.external_cad = aligned
    cfg.cad_info = info.as_dict()
    cfg.geometry_stl = aligned
    cfg.flow_axis = "x"
    cfg.cad_units = "m"

    from cfd.sweep import point_is_complete

    # Fingerprint of the case, stored next to the solution. The completion test
    # proves a solve *finished*; it cannot tell what geometry it finished on, so
    # flipping `mirror` or the wall size would otherwise silently reuse the wrong
    # wing when the directory happens to be shared.
    key = _case_key(stl, mirror=mirror, wall_size=wall_size,
                    refinement=refinement)
    stamp = work_dir / "case_key.txt"
    mesh_path = work_dir / "rocket_mesh.su2"
    key_ok = (resume and stamp.is_file()
              and stamp.read_text(encoding="utf-8").strip() == key)

    # Two separate questions. The fine mesh takes ~12 minutes to build and is a
    # deterministic function of the fingerprint, so an interrupted *solve* should
    # not cost it: keep the mesh whenever the fingerprint matches, and re-solve
    # only when the previous solve did not finish.
    mesh_ready = key_ok and mesh_path.is_file()
    solve_ready = mesh_ready and point_is_complete(work_dir, mesh_path)

    solver = SU2Solver(cfg)
    if mesh_ready:
        solver._mesh_path = mesh_path
    else:
        solver.generate_mesh()
        stamp.write_text(key, encoding="utf-8")
    # Always regenerated: writing the .cfg is what populates the flow metadata
    # (Reynolds, q, reference area) that parse_results reads back.
    solver.generate_case()
    if not solve_ready:
        for _ in solver.run():
            pass
    return solver.parse_results()


def _case_key(stl: Path, mirror: bool, wall_size, refinement: str) -> str:
    """Hash of the geometry and the mesh settings a solved level belongs to.

    The mesher's own revision is part of it: a change to the size fields makes a
    different mesh from identical settings, and without this the resumed run
    reuses the old one and reproduces the old numbers exactly.
    """
    import hashlib

    from cfd.meshing import MESHER_REVISION

    h = hashlib.sha256()
    h.update(Path(stl).read_bytes())
    h.update(f"|{mirror}|{wall_size}|{refinement}|{MACH}|{ALPHA_DEG}"
             f"|{MESHER_REVISION}".encode())
    return h.hexdigest()[:32]
