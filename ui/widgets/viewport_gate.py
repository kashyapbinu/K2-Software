"""
K2 AeroSim — stop hidden 3D viewports from drawing
=====================================================
pyvistaqt gives every QtInteractor a 5 Hz timer that redraws its render
window whether or not anyone can see it. K2 has seven such viewports and at
most one is on screen, so six kept redrawing behind hidden tabs. On a GPU
that is waste; with software OpenGL each hidden frame is slow enough to starve
the simulation timer, and a flight ran about a hundred times slower.

``gate_hidden_rendering(plotter)`` ties drawing to visibility:

* hidden  → pyvista's own ``suppress_rendering`` switch is set, so neither
  the auto-update timer nor a scene change (``add_mesh`` and friends call
  ``render``) reaches OpenGL;
* shown   → it is cleared and one frame is drawn, so changes made while the
  view was hidden appear at once.

The auto-update timer itself is left running. A suppressed tick costs
nothing, and a Hide also arrives while the window is being torn down at
exit: stopping the timer from there crashed the interpreter. The hide path
therefore makes no Qt call at all.

Screenshots need care. ``Plotter.screenshot`` does not draw: it copies the
last frame out of the window. The redraws being removed here are what kept
that frame current for a hidden view, and the PDF reports screenshot views
whose tab is not selected. A gated plotter therefore draws one real frame
before every screenshot it takes while hidden.
"""
import logging
import weakref

from PyQt6.QtCore import QEvent, QObject

logger = logging.getLogger("K2.ViewportGate")


class _ViewportGate(QObject):
    """Event filter on a QtInteractor: suppress its rendering while hidden."""

    def __init__(self, plotter):
        super().__init__(plotter)       # Qt-owned by the plotter, dies with it
        self.hidden = False

        # The wrapper lives on the plotter, so it may hold neither the plotter
        # nor this gate strongly: that cycle would keep an OpenGL widget alive
        # until the garbage collector ran.
        take = type(plotter).screenshot
        plotter_ref, gate_ref = weakref.ref(plotter), weakref.ref(self)

        def screenshot(*args, **kwargs):
            view, gate = plotter_ref(), gate_ref()
            if gate is None or not gate.hidden:
                return take(view, *args, **kwargs)
            # Copying the window would return the frame from before the view
            # was hidden. Draw the current scene first.
            view.suppress_rendering = False
            try:
                view.render()
                return take(view, *args, **kwargs)
            finally:
                view.suppress_rendering = True

        plotter.screenshot = screenshot
        plotter.installEventFilter(self)
        if not plotter.isVisible():
            self._suspend(plotter)

    def eventFilter(self, obj, event):
        kind = event.type()
        if kind == QEvent.Type.Hide:
            self._suspend(obj)
        elif kind == QEvent.Type.Show:
            self._resume(obj)
        return False

    def _suspend(self, plotter):
        # No Qt calls here: this also runs during teardown.
        self.hidden = True
        plotter.suppress_rendering = True

    def _resume(self, plotter):
        if not self.hidden:
            return
        self.hidden = False
        plotter.suppress_rendering = False
        plotter.render()


def gate_hidden_rendering(plotter):
    """Make *plotter* (a pyvistaqt QtInteractor) draw only while it is visible.

    Returns the gate, or None if *plotter* is not a live interactor: the
    viewers fall back to a plain label when OpenGL is unavailable, and a view
    that cannot be gated must still open.
    """
    try:
        return _ViewportGate(plotter)
    except Exception as exc:
        logger.debug("viewport gate not installed: %s", exc)
        return None
