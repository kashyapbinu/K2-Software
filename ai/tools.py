"""
Tools the assistant may call, and the main-thread bridge that executes them.

The chat loop runs in a worker thread; every tool touches Qt objects and the
state engine, so calls are marshalled to the GUI thread through a queued
signal and the worker blocks on a threading.Event until the slot finishes.
"""

from __future__ import annotations

import json
import logging
import math
import threading

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from ai.context import summarize_state, last_flight_summary

logger = logging.getLogger("K2.AI.Tools")

# Parameters the model may write. name -> (state key, unit note, converter)
_SETTABLE = {
    "fin_span":            ("fin_span", "m", float),
    "fin_root_chord":      ("fin_root_chord", "m", float),
    "fin_tip_chord":       ("fin_tip_chord", "m", float),
    "fin_count":           ("fin_count", "", int),
    "fin_sweep_deg":       ("fin_sweep_angle", "deg", lambda v: math.radians(float(v))),
    "fin_thickness":       ("fin_thickness", "m", float),
    "fin_position":        ("fin_position", "m from nose", float),
    "nose_length":         ("nose_length", "m", float),
    "nose_type":           ("nose_type", "ogive|conical|elliptical|parabolic|haack", str),
    "length":              ("length", "m", float),
    "diameter":            ("diameter", "m", float),
    "dry_mass":            ("dry_mass", "kg", float),
    "launch_angle":        ("launch_angle", "deg from horizontal", float),
    "launch_rod_length":   ("launch_rod_length", "m", float),
    "wind_speed":          ("wind_speed", "m/s", float),
    "wind_direction":      ("wind_direction", "deg", float),
    "main_deploy_altitude": ("main_deploy_altitude", "m AGL", float),
    "drogue_deploy_delay": ("drogue_deploy_delay", "s", float),
    "drogue_cd_area":      ("drogue_cd_area", "m^2", float),
    "main_cd_area":        ("main_cd_area", "m^2", float),
    "wall_thickness":      ("wall_thickness", "m", float),
}

# In assembly mode these write to the component instead of the summary state.
_ASSEMBLY_ROUTE = {
    "fin_span":       ("fin", "height", float),
    "fin_root_chord": ("fin", "root_chord", float),
    "fin_tip_chord":  ("fin", "tip_chord", float),
    "fin_count":      ("fin", "fin_count", int),
    "fin_sweep_deg":  ("fin", "sweep_angle", float),
    "fin_thickness":  ("fin", "thickness", float),
    "nose_length":    ("nose", "length", float),
    "nose_type":      ("nose", "shape", lambda v: _NOSE_SHAPES.get(str(v).strip().lower(), str(v).strip().capitalize())),
}

# Model-facing nose names -> NoseCone.shape values used by the geometry code.
_NOSE_SHAPES = {
    "ogive": "Ogive", "conical": "Conical", "cone": "Conical",
    "elliptical": "Elliptical", "ellipsoid": "Elliptical",
    "parabolic": "Parabolic", "haack": "Haack (LD)", "haack (ld)": "Haack (LD)",
    "von karman": "Haack (LD)", "haack_ld": "Haack (LD)",
}

TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "add_mass",
        "description": "Add ballast (a mass component) to the nose cone or tail. The usual "
                       "fix for an unstable or marginal rocket is nose weight. Returns new CG/CP/margin.",
        "parameters": {"type": "object",
                       "properties": {"mass_kg": {"type": "number"},
                                      "location": {"type": "string", "enum": ["nose", "tail"], "default": "nose"}},
                       "required": ["mass_kg"]}}},
    {"type": "function", "function": {
        "name": "get_rocket_state",
        "description": "Full current design summary: geometry, mass, CG/CP, stability margin, "
                       "motor, launch conditions, recovery, structure and last flight if any.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "set_parameter",
        "description": "Change one design/launch parameter. Returns the new CG/CP/stability. "
                       "Parameters: " + ", ".join(f"{k} ({u})" if u else k
                                                  for k, (_, u, _) in _SETTABLE.items()),
        "parameters": {"type": "object",
                       "properties": {"name": {"type": "string", "enum": list(_SETTABLE)},
                                      "value": {"type": ["number", "string"]}},
                       "required": ["name", "value"]}}},
    {"type": "function", "function": {
        "name": "search_motors",
        "description": "Search the motor catalog (ThrustCurve.org data). Filter by impulse "
                       "class letter (e.g. 'H'), impulse range in N·s, max diameter in mm, "
                       "and min average thrust in N. Returns up to `limit` motors.",
        "parameters": {"type": "object", "properties": {
            "impulse_class": {"type": "string"},
            "min_impulse_Ns": {"type": "number"},
            "max_impulse_Ns": {"type": "number"},
            "max_diameter_mm": {"type": "number"},
            "min_avg_thrust_N": {"type": "number"},
            "manufacturer": {"type": "string"},
            "limit": {"type": "integer", "default": 10}}}}},
    {"type": "function", "function": {
        "name": "select_motor",
        "description": "Install a motor from the catalog by its designation (e.g. 'H128W'). "
                       "Updates propellant mass, thrust and stability.",
        "parameters": {"type": "object",
                       "properties": {"designation": {"type": "string"}},
                       "required": ["designation"]}}},
    {"type": "function", "function": {
        "name": "run_simulation",
        "description": "Run the flight simulation with the current design and wait for it to "
                       "finish. Returns apogee, max velocity/Mach/accel, descent, drift, and "
                       "any abort diagnosis.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "get_flight_result",
        "description": "Summary of the most recent flight simulation without re-running it.",
        "parameters": {"type": "object", "properties": {}}}},
]


class ToolBridge(QObject):
    """Executes tools on the GUI thread. Construct in the main thread."""

    _request = pyqtSignal(str, dict, object)   # name, args, _Pending

    SIM_TIMEOUT_S = 240.0
    CALL_TIMEOUT_S = 30.0
    SIM_FAST_FORWARD = 50.0   # sim seconds per wall second while a tool waits

    class _Pending:
        __slots__ = ("event", "result")

        def __init__(self):
            self.event = threading.Event()
            self.result = ""

    def __init__(self, main_window):
        super().__init__(main_window)
        self.mw = main_window
        self._sim_done = threading.Event()
        self._sim_failure = None
        self._request.connect(self._on_request)
        se = getattr(main_window, "sim_engine", None)
        if se is not None:
            se.sim_finished.connect(self._sim_done.set)
            se.sim_failed.connect(self._on_sim_failed)

    # -- worker-thread entry --------------------------------------------
    def call(self, name: str, args: dict) -> str:
        """Blocking call from any thread; returns JSON string."""
        p = self._Pending()
        self._request.emit(name, args or {}, p)
        timeout = self.SIM_TIMEOUT_S if name == "run_simulation" else self.CALL_TIMEOUT_S
        if not p.event.wait(timeout):
            return json.dumps({"error": f"{name} timed out after {timeout:.0f}s"})
        return p.result

    # -- GUI-thread dispatch --------------------------------------------
    @pyqtSlot(str, dict, object)
    def _on_request(self, name, args, pending):
        try:
            if name == "run_simulation":
                # Async: finish the pending when the sim signals completion.
                self._run_sim_async(pending)
                return
            fn = getattr(self, f"tool_{name}", None)
            if fn is None:
                res = {"error": f"unknown tool '{name}'"}
            else:
                res = fn(**args)
        except TypeError as e:
            res = {"error": f"bad arguments for {name}: {e}"}
        except Exception as e:
            logger.exception("Tool %s failed", name)
            res = {"error": f"{type(e).__name__}: {e}"}
        pending.result = json.dumps(res, default=str)
        pending.event.set()

    # -- tools -----------------------------------------------------------
    def tool_get_rocket_state(self):
        asm = None
        ws = getattr(self.mw, "design_ws", None)
        if ws is not None and hasattr(ws, "get_assembly"):
            asm = ws.get_assembly()
        return summarize_state(self.mw.engine.state, asm)

    def _assembly(self):
        ws = getattr(self.mw, "design_ws", None)
        asm = ws.get_assembly() if ws is not None and hasattr(ws, "get_assembly") else None
        if asm is None or getattr(self.mw.engine, "auto_estimate_properties", True):
            return None, ws
        from core.components import Stage
        if not any(not isinstance(c, Stage) for c in asm.all_components()):
            return None, ws   # empty tree: state-only mode
        return asm, ws

    def _stability(self) -> dict:
        s = self.mw.engine.state
        return {"cg_m": round(s.cg, 4), "cp_m": round(s.cp, 4),
                "stability_margin_cal": round(s.stability_margin, 3)}

    def tool_set_parameter(self, name, value):
        if name not in _SETTABLE:
            return {"error": f"'{name}' is not settable", "settable": list(_SETTABLE)}
        key, unit, conv = _SETTABLE[name]
        try:
            v = conv(value)
        except (TypeError, ValueError):
            return {"error": f"cannot convert {value!r} for {name} ({unit})"}

        asm, ws = self._assembly()
        target = _ASSEMBLY_ROUTE.get(name)
        if asm is not None and target is not None:
            # Geometry lives on components when a Design-tab assembly is active;
            # writing the summary state would be overwritten on the next sync.
            from core.components import TrapezoidalFinSet, NoseCone
            cls, attr, to_comp = target
            comp = next((c for c in asm.all_components()
                         if isinstance(c, {"fin": TrapezoidalFinSet, "nose": NoseCone}[cls])), None)
            if comp is None:
                return {"error": f"no {cls} component in the assembly to edit"}
            old = getattr(comp, attr)
            setattr(comp, attr, to_comp(value))
            ws._on_component_edited()
            return {"changed": name, "component": comp.name, "from": old,
                    "to": getattr(comp, attr), "unit": unit, **self._stability()}
        if asm is not None and name in ("length", "diameter", "fin_position", "dry_mass"):
            return {"error": f"'{name}' is derived from the component assembly; edit body "
                             f"tubes in the Design tab, or use add_mass for ballast."}

        old = getattr(self.mw.engine.state, key)
        self.mw.engine.update(**{key: v})
        if name == "fin_sweep_deg":   # state stores radians; report in the unit we advertise
            old, v = math.degrees(old), math.degrees(v)
        return {"changed": name, "from": old, "to": v, "unit": unit, **self._stability()}

    def tool_add_mass(self, mass_kg, location="nose"):
        """Ballast: the standard stability fix. Adds a MassComponent to the nose cone
        (or the aft-most body tube for 'tail')."""
        asm, ws = self._assembly()
        if asm is None:
            s = self.mw.engine.state
            m = float(mass_kg)
            # State-only mode: shift dry mass/CG analytically.
            pos = 0.05 * s.length if location == "nose" else 0.95 * s.length
            total = s.dry_mass + m
            new_cg = (s.dry_mass * s.dry_cg + m * pos) / total if total > 0 else s.dry_cg
            self.mw.engine.update(dry_mass=total, dry_cg=new_cg)
            return {"added_kg": m, "at_m_from_nose": round(pos, 3), **self._stability()}
        from core.components import MassComponent, NoseCone, BodyTube
        m = float(mass_kg)
        if m <= 0:
            return {"error": "mass_kg must be positive"}
        comps = list(asm.all_components())
        if location == "nose":
            parent = next((c for c in comps if isinstance(c, NoseCone)), None)
        else:
            parent = next((c for c in reversed(comps) if isinstance(c, BodyTube)), None)
        if parent is None:
            parent = next((c for c in comps if isinstance(c, BodyTube)), None)
        if parent is None:
            return {"error": "assembly has no nose cone or body tube to attach ballast to"}
        mc = MassComponent(f"Ballast ({m*1000:.0f} g)")
        mc.mass = m
        asm.add_component(parent, mc)
        ws._on_component_edited()
        return {"added_kg": m, "parent": parent.name,
                "at_m_from_nose": round(mc.cg_position(), 3), **self._stability()}

    def _catalog(self) -> list[dict]:
        ws = getattr(self.mw, "propulsion_ws", None)
        return list(getattr(ws, "_motors", []) or [])

    def tool_search_motors(self, impulse_class=None, min_impulse_Ns=None, max_impulse_Ns=None,
                           max_diameter_mm=None, min_avg_thrust_N=None, manufacturer=None,
                           limit=10):
        out = []
        for m in self._catalog():
            if impulse_class and m.get("class", "").upper() != str(impulse_class).upper():
                continue
            imp = m.get("total_impulse", 0.0)
            if min_impulse_Ns is not None and imp < float(min_impulse_Ns):
                continue
            if max_impulse_Ns is not None and imp > float(max_impulse_Ns):
                continue
            if max_diameter_mm is not None and m.get("diameter", 0) * 1000 > float(max_diameter_mm) + 1e-6:
                continue
            if min_avg_thrust_N is not None and m.get("avg_thrust", 0) < float(min_avg_thrust_N):
                continue
            if manufacturer and str(manufacturer).lower() not in str(m.get("manufacturer", "")).lower():
                continue
            out.append({
                "designation": m.get("designation"), "manufacturer": m.get("manufacturer"),
                "class": m.get("class"), "diameter_mm": round(m.get("diameter", 0) * 1000),
                "length_mm": round(m.get("length", 0) * 1000) if m.get("length") else None,
                "total_impulse_Ns": round(imp, 1), "avg_thrust_N": round(m.get("avg_thrust", 0), 1),
                "burn_time_s": round(m.get("burn_time", 0), 2),
                "total_mass_kg": round(m.get("total_mass", 0), 3),
                "availability": m.get("availability")})
        out.sort(key=lambda r: r["total_impulse_Ns"])
        n = len(out)
        return {"matches": n, "motors": out[: max(1, min(int(limit or 10), 30))]}

    def tool_select_motor(self, designation):
        ws = getattr(self.mw, "propulsion_ws", None)
        if ws is None:
            return {"error": "propulsion workspace unavailable"}
        want = str(designation).strip().upper()
        cat = self._catalog()
        m = next((x for x in cat if str(x.get("designation", "")).upper() == want), None)
        if m is None:
            m = next((x for x in cat if want in str(x.get("designation", "")).upper()), None)
        if m is None:
            return {"error": f"no motor '{designation}' in catalog"}
        # Prefer the combo path so the UI stays in sync; fall back to direct apply.
        idx = next((i for i, f in enumerate(ws._filtered) if f is m), None)
        if idx is not None:
            ws.motor_combo.setCurrentIndex(idx + 1)
        else:
            prop, dry = ws._sanitized_masses(m)
            ws._apply_motor(dict(
                motor_designation=m["designation"], motor_avg_thrust=m["avg_thrust"],
                motor_max_thrust=m.get("max_thrust", m["avg_thrust"] * 1.4),
                motor_total_impulse=m["total_impulse"], motor_burn_time=m["burn_time"],
                propellant_mass=prop, propellant_mass_initial=prop, motor_dry_mass=dry,
                motor_length=m.get("length", 0.0), custom_thrust_curve=[]))
            ws._load_real_curve(m.get("motor_id", ""), m["total_impulse"])
            ws._update_display()
        s = self.mw.engine.state
        liftoff = s.dry_mass + s.propellant_mass_initial
        return {"installed": m["designation"], "manufacturer": m.get("manufacturer"),
                "total_impulse_Ns": m["total_impulse"], "avg_thrust_N": m["avg_thrust"],
                "thrust_to_weight": round(s.motor_avg_thrust / (liftoff * 9.80665), 2) if liftoff else None,
                "cg_m": round(s.cg, 4), "cp_m": round(s.cp, 4),
                "stability_margin_cal": round(s.stability_margin, 3),
                "note": "motor not in the current filter view" if idx is None else ""}

    def tool_get_flight_result(self):
        s = self.mw.engine.state
        if not s.max_altitude:
            return {"error": "no flight has been simulated yet"}
        res = last_flight_summary(s)
        if self._sim_failure:
            res["abort"] = self._sim_failure
        return res

    def _on_sim_failed(self, title, detail):
        self._sim_failure = {"title": title, "detail": detail}

    def _run_sim_async(self, pending):
        se = self.mw.sim_engine
        if getattr(self.mw.engine.state, "sim_running", False):
            pending.result = json.dumps({"error": "a simulation is already running"})
            pending.event.set()
            return
        self._sim_done.clear()
        self._sim_failure = None
        # The engine animates in wall-clock time (QTimer, 1x = real time). A
        # tool call must not wait minutes: fast-forward, restore afterwards.
        prev_speed = self.mw.engine.state.sim_speed
        self.mw.engine.update(sim_speed=self.SIM_FAST_FORWARD, emit=False)

        def failed(title, detail):
            self._sim_failure = {"title": title, "detail": detail}
            finish()

        def cleanup():
            for sig, slot in ((se.sim_finished, finish), (se.sim_failed, failed)):
                try:
                    sig.disconnect(slot)
                except TypeError:
                    pass

        def finish():
            if pending.event.is_set():
                return
            cleanup()
            self.mw.engine.update(sim_speed=prev_speed, emit=False)
            res = self.tool_get_flight_result()
            pending.result = json.dumps(res, default=str)
            pending.event.set()

        se.sim_finished.connect(finish)
        se.sim_failed.connect(failed)
        try:
            se.start()
        except Exception as e:
            cleanup()
            pending.result = json.dumps({"error": f"could not start simulation: {e}"})
            pending.event.set()
            return
        # start() returns silently when there is no motor / empty geometry.
        if not pending.event.is_set() and not self.mw.engine.state.sim_running:
            cleanup()
            self.mw.engine.update(sim_speed=prev_speed, emit=False)
            s = self.mw.engine.state
            why = ("no motor selected" if s.motor_designation == "None" and not s.stages_config
                   else "rocket geometry is empty (diameter/length = 0)")
            pending.result = json.dumps({"error": f"simulation did not start: {why}"})
            pending.event.set()
