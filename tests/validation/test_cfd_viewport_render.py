"""
CFD 3D viewport repaint gates.

The interactor is built with ``QtInteractor(frame, auto_update=False)``, which
means pyvistaqt never starts its periodic render timer. VTK then keeps
presenting the last drawn frame until the user happens to click in the view,
so any scene change that does not end in an explicit ``render()`` is invisible
until the next mouse event — and the frame still on screen describes the
PREVIOUS state.

The user-visible symptom that motivated these tests: starting a solve cleared
the geometry actor, nothing repainted, and the rocket stayed on screen looking
fine until the first click made it vanish.

These use a stub plotter rather than a live Qt widget, so they are fast and
run headless.
"""
import sys
import types
from pathlib import Path

import pyvista as pv
import pytest

from ui.workspaces.cfd_workspace import CFDWorkspace


class FakePlotter:
    """Records the scene-mutation / repaint call order."""

    def __init__(self):
        self.calls = []
        self.meshes = []
        self.camera_position = [(1.0, 2.0, 3.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]

    def clear(self):
        self.calls.append("clear")
        self.meshes.clear()

    def add_mesh(self, mesh, **kw):
        self.calls.append("add_mesh")
        self.meshes.append(mesh)

    def add_axes(self):
        self.calls.append("add_axes")

    def reset_camera(self):
        self.calls.append("reset_camera")

    def render(self):
        self.calls.append("render")

    def disable_picking(self):
        self.calls.append("disable_picking")


class FakeLabel:
    def __init__(self):
        self.text = ""

    def setText(self, t):
        self.text = t


class FakeLogBox:
    def __init__(self):
        self.lines = []

    def append(self, msg):
        self.lines.append(msg)


@pytest.fixture
def stl(tmp_path):
    """A trivial watertight body on disk for the preview path to read."""
    p = tmp_path / "rocket.stl"
    pv.Sphere(radius=0.05).save(str(p))
    return p


class ViewportShim:
    """Borrows the real viewport methods onto a plain object.

    CFDWorkspace is a QWidget, so it cannot be instantiated without a QApp and
    an OpenGL context. The methods under test only touch _plotter / _status_lbl
    / _log_box / _current_stl, so binding the real functions to a plain class
    exercises the actual code with no Qt at all.
    """

    # Bound on demand rather than at class-creation time, so that a missing
    # method fails the one test that needs it instead of erroring collection
    # for the whole module.
    def __getattr__(self, name):
        fn = getattr(CFDWorkspace, name, None)
        if not callable(fn):
            raise AttributeError(name)
        return types.MethodType(fn, self)


def _workspace(stl_path):
    """A shim carrying only the attributes the viewport paths touch."""
    ws = ViewportShim()
    ws._plotter = FakePlotter()
    ws._status_lbl = FakeLabel()
    ws._log_box = FakeLogBox()
    ws._current_stl = stl_path
    ws._result = object()
    ws._volume_mesh = object()
    ws._surface_mesh = object()
    return ws


def test_clearing_results_repaints_the_view(stl):
    """A clear that never renders leaves the stale frame on screen.

    This is the actual reported bug: without the trailing render() the body
    is still visibly drawn until the user clicks, then disappears.
    """
    ws = _workspace(stl)
    ws._clear_results()

    assert "clear" in ws._plotter.calls, "results clear must drop the actors"
    assert ws._plotter.calls[-1] == "render", (
        "clear_results must end in an explicit repaint; auto_update=False means "
        f"nothing else will. Got call order: {ws._plotter.calls}"
    )


def test_run_start_keeps_the_body_on_screen(stl):
    """Starting a solve must not blank the geometry.

    The flow field is stale the moment Run is clicked, but the geometry is
    exactly what the run is solving, so it stays.
    """
    ws = _workspace(stl)
    ws._clear_results(keep_geometry=True)

    assert ws._plotter.meshes, (
        "geometry actor must be re-added when keep_geometry=True; "
        f"call order: {ws._plotter.calls}"
    )
    assert ws._plotter.calls[-1] == "render"
    # ...and the stale flow field is still gone.
    assert ws._volume_mesh is None
    assert ws._surface_mesh is None
    assert ws._result is None


def test_run_start_does_not_move_the_camera(stl):
    """Re-adding the body must not yank the view back to isometric.

    _preview() resets the camera, which is right for a freshly loaded
    geometry and wrong on every Run click.
    """
    ws = _workspace(stl)
    before = list(ws._plotter.camera_position)

    ws._clear_results(keep_geometry=True)

    assert "reset_camera" not in ws._plotter.calls, (
        "run start must preserve the user's camera"
    )
    assert list(ws._plotter.camera_position) == before


def test_new_project_leaves_the_view_blank(stl):
    """New Project drops the geometry too — nothing may be re-added."""
    ws = _workspace(stl)
    ws._clear_results()

    assert not ws._plotter.meshes, (
        "default clear must not resurrect the geometry"
    )


def test_clear_still_repaints_before_any_geometry_is_loaded():
    """_current_stl is unbound until geometry is exported or CAD is loaded.

    An unguarded attribute access here raises inside the try/except that wraps
    the whole block, which would swallow the render() as well — the exact
    failure the repaint is there to prevent.
    """
    ws = _workspace(None)
    del ws._current_stl

    ws._clear_results(keep_geometry=True)

    assert ws._plotter.calls[-1] == "render", (
        f"missing geometry must not cost the repaint; got {ws._plotter.calls}"
    )


def test_preview_repaints(stl):
    """Loading/exporting geometry must show up without a click."""
    ws = _workspace(stl)
    ws._preview(stl)

    assert ws._plotter.calls[-1] == "render", (
        f"preview must end in a repaint; got {ws._plotter.calls}"
    )
    assert "reset_camera" in ws._plotter.calls, (
        "a newly loaded body should be framed by the camera"
    )


def test_preview_can_keep_the_camera(stl):
    """keep_camera=True restores the pre-clear view instead of resetting."""
    ws = _workspace(stl)
    ws._preview(stl, keep_camera=True)

    assert "reset_camera" not in ws._plotter.calls
    assert ws._plotter.calls[-1] == "render"


def test_reset_camera_button_repaints(stl):
    """The Reset Camera button moved the camera but never redrew."""
    ws = _workspace(stl)
    ws._reset_camera()

    assert ws._plotter.calls == ["reset_camera", "render"]


def test_every_scene_mutation_is_followed_by_a_render():
    """Guard the root cause, not just today's symptoms.

    auto_update=False is a deliberate choice (no idle GPU cost on big meshes),
    but it makes an un-rendered mutation a silent display bug. If the
    interactor is ever switched back to a timer this test should be deleted,
    not muted.
    """
    src = Path(CFDWorkspace.__module__.replace(".", "/") + ".py")
    text = src.read_text(encoding="utf-8")
    assert "auto_update=False" in text, (
        "auto_update changed — revisit the explicit render() calls"
    )
    # _refresh_vis has many early returns; the repaint must be in a finally.
    assert "def _refresh_vis(self):" in text
    body = text.split("def _refresh_vis(self):", 1)[1].split("def _refresh_vis_impl", 1)[0]
    assert "finally" in body and "render()" in body, (
        "_refresh_vis must repaint on every exit path"
    )
