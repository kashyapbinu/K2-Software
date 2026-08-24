"""
Guards on sweep resume: a point may only be reused if its numbers really came
from the current mesh.

A resumable sweep is worth an hour of solver time on the long validation cases,
but reuse is exactly the kind of shortcut that turns a validation run into a
comfortable lie — the failure mode is silent, plausible-looking numbers from a
superseded grid. These tests pin the three ways a point is rejected.
"""
import os
import time

from cfd.sweep import point_is_complete, staged_mesh_matches, sweep_point_dir

MESH_NAME = "rocket_mesh.su2"


def _mesh(tmp_path, body: bytes = b"NDIME= 3\n"):
    mesh = tmp_path / MESH_NAME
    mesh.write_bytes(body)
    return mesh


def _point(tmp_path, mesh, *, exit_ok=True, link=True, history_age=0.0):
    """Fabricate a solved sweep point folder."""
    wd = tmp_path / "sweep" / "aoa_p2_000"
    wd.mkdir(parents=True)
    (wd / "history.csv").write_text('"Inner_Iter","CL"\n0,0.1\n', encoding="utf-8")
    if history_age:
        t = time.time() + history_age
        os.utime(wd / "history.csv", (t, t))
    log = "Exit Success (SU2_CFD)\n" if exit_ok else "iteration 75 residual -4.7\n"
    (wd / "su2_run.log").write_text(log, encoding="utf-8")
    if link:
        try:
            os.link(mesh, wd / MESH_NAME)
        except OSError:                       # no hardlinks → copy keeps mtime
            import shutil
            shutil.copy2(mesh, wd / MESH_NAME)
    return wd


def test_point_dir_naming_matches_the_solver_convention(tmp_path):
    assert sweep_point_dir(tmp_path, "aoa", 2.0).name == "aoa_p2_000"
    assert sweep_point_dir(tmp_path, "aoa", -2.0).name == "aoa_m2_000"
    assert sweep_point_dir(tmp_path, "mach", 0.85).name == "mach_p0_850"


def test_completed_point_is_reusable(tmp_path):
    mesh = _mesh(tmp_path)
    wd = _point(tmp_path, mesh)
    assert staged_mesh_matches(mesh, wd)
    assert point_is_complete(wd, mesh)


def test_killed_point_is_not_reusable(tmp_path):
    """No exit banner: the history parses fine but stops mid-transient."""
    mesh = _mesh(tmp_path)
    wd = _point(tmp_path, mesh, exit_ok=False)
    assert not point_is_complete(wd, mesh)


def test_point_from_a_different_mesh_is_not_reusable(tmp_path):
    mesh = _mesh(tmp_path)
    wd = _point(tmp_path, mesh, link=False)
    (wd / MESH_NAME).write_bytes(b"NDIME= 3\nsome other much longer mesh\n")
    assert not staged_mesh_matches(mesh, wd)
    assert not point_is_complete(wd, mesh)


def test_results_older_than_the_mesh_are_not_reusable(tmp_path):
    """The hardlink case: re-meshing in place updates the point's copy too.

    Size and mtime then match — the staged file *is* the current mesh, because it
    is the same inode — so mesh identity alone cannot catch this. Only the
    history's own age shows the numbers predate the grid.
    """
    mesh = _mesh(tmp_path)
    wd = _point(tmp_path, mesh, history_age=-600.0)
    assert staged_mesh_matches(mesh, wd)
    assert not point_is_complete(wd, mesh)


def test_mesh_argument_is_optional(tmp_path):
    """Without a mesh to check against, completion is the log's word alone."""
    mesh = _mesh(tmp_path)
    wd = _point(tmp_path, mesh, history_age=-600.0)
    assert point_is_complete(wd)
