"""
Helper to blank every visual a workspace holds when a new project is created.

Scans a workspace's attributes and clears each plot / 3D viewer it finds —
custom matplotlib PlotWidgets, raw matplotlib FigureCanvases, pyvista-based
viewers (StressViewer / DeformationViewer / ModeShapeViewer), so "New Project"
leaves no stale curves or contours from the previous rocket. Each reset_workspace
calls this, then clears its own cached result objects / summary labels.
"""

import logging

logger = logging.getLogger("K2.Reset")


def clear_visuals(ws):
    """Clear all plots and 3D viewers held as attributes of *ws*."""
    for w in list(vars(ws).values()):
        try:
            # Custom PlotWidget: has its own clear() + ax + canvas.
            if hasattr(w, "ax") and hasattr(w, "canvas") and hasattr(w, "clear"):
                w.clear()
                continue
            # Raw matplotlib FigureCanvas: clear each of its axes.
            fig = getattr(w, "figure", None)
            if fig is not None and hasattr(w, "draw") and hasattr(fig, "axes"):
                for a in fig.axes:
                    a.clear()
                try:
                    w.draw_idle()
                except Exception:
                    w.draw()
                continue
            # 3D viewers (Stress / Deformation / ModeShape): prefer the viewer's
            # OWN reset/clear/show_empty — these also reset internal state (e.g.
            # stop the mode-shape animation), so a running timer won't touch a
            # half-cleared plotter and crash. Only fall back to clearing the raw
            # plotter scene when the viewer exposes nothing.
            plotter = getattr(w, "plotter", None)
            if plotter is not None or hasattr(w, "show_empty"):
                for m in ("reset", "show_empty", "clear"):
                    fn = getattr(w, m, None)
                    if callable(fn):
                        try:
                            fn()
                        except Exception:
                            pass
                        break
                else:
                    if plotter is not None:
                        try:
                            plotter.clear()
                            plotter.render()
                        except Exception:
                            pass
                continue
        except Exception as e:
            logger.debug("clear_visuals skipped %r: %s", type(w).__name__, e)
