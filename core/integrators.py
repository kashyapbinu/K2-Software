"""
K2 Aerospace — Numerical Integrators
======================================
Pluggable integrators for trajectory simulation.

Supports:
    - Forward Euler (1st order, educational use only)
    - Classical RK4 (4th order, standard for rocketry)
    - RK45 Dormand-Prince (adaptive step, 4th/5th order embedded pair)

State vector: arbitrary length (e.g. [x, z, vx, vz, pitch, pitch_rate, mass])
"""

import math
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger("K2.Integrators")


class IntegratorBase(ABC):
    """Abstract base class for numerical integrators."""

    @abstractmethod
    def step(self, state_vec: list, t: float, dt: float,
             derivatives_fn) -> list:
        """
        Advance state_vec by one time step dt.

        Args:
            state_vec:       Current state vector.
            t:               Current time (s).
            dt:              Time step (s).
            derivatives_fn:  Callable(t, state) -> derivatives list.

        Returns:
            New state vector.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str: ...


# ── Forward Euler ─────────────────────────────────────────────────────────────

class EulerIntegrator(IntegratorBase):
    """Forward Euler — 1st order. Only for debugging / comparison."""

    @property
    def name(self) -> str:
        return "Euler"

    def step(self, state_vec, t, dt, derivatives_fn):
        derivs = derivatives_fn(t, state_vec)
        return [s + d * dt for s, d in zip(state_vec, derivs)]


# ── Classical RK4 ─────────────────────────────────────────────────────────────

class RK4Integrator(IntegratorBase):
    """
    Classical Runge-Kutta 4th-order method.
    Excellent balance of accuracy and computational cost for rocket trajectories.
    """

    @property
    def name(self) -> str:
        return "RK4"

    def step(self, state_vec, t, dt, derivatives_fn):
        n = len(state_vec)
        k1 = derivatives_fn(t, state_vec)
        s2 = [state_vec[i] + 0.5 * dt * k1[i] for i in range(n)]
        k2 = derivatives_fn(t + 0.5 * dt, s2)
        s3 = [state_vec[i] + 0.5 * dt * k2[i] for i in range(n)]
        k3 = derivatives_fn(t + 0.5 * dt, s3)
        s4 = [state_vec[i] + dt * k3[i] for i in range(n)]
        k4 = derivatives_fn(t + dt, s4)
        return [
            state_vec[i] + (dt / 6.0) * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i])
            for i in range(n)
        ]


# ── RK45 Dormand-Prince (Adaptive) ────────────────────────────────────────────

class RK45Integrator(IntegratorBase):
    """
    Runge-Kutta 4(5) — Dormand-Prince embedded pair.

    Adaptively controls step size to maintain a specified local error tolerance.
    Falls back to a fixed-step RK4 advance when called via the standard .step()
    interface. Use .step_adaptive() for full error control.

    Tableau (Dormand-Prince):
        c2=1/5, c3=3/10, c4=4/5, c5=8/9, c6=1, c7=1
    """

    # Dormand-Prince coefficients
    _C = [0, 1/5, 3/10, 4/5, 8/9, 1, 1]
    _A = [
        [],
        [1/5],
        [3/40, 9/40],
        [44/45, -56/15, 32/9],
        [19372/6561, -25360/2187, 64448/6561, -212/729],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656],
        [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84],
    ]
    # 4th order weights
    _B4 = [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40]
    # 5th order weights
    _B5 = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]

    def __init__(self, rtol: float = 1e-4, atol: float = 1e-6,
                 dt_min: float = 1e-5, dt_max: float = 0.05):
        self.rtol   = rtol
        self.atol   = atol
        self.dt_min = dt_min
        self.dt_max = dt_max
        self._dt_next = 0.01   # suggested next step

    @property
    def name(self) -> str:
        return "RK45"

    def step(self, state_vec, t, dt, derivatives_fn):
        """Standard fixed-step interface (propagates the 5th-order solution)."""
        _, new_state = self._rk45_core(state_vec, t, dt, derivatives_fn)
        return new_state

    def step_adaptive(self, state_vec, t, dt_max, derivatives_fn):
        """
        Adaptive step: automatically chooses dt to satisfy tolerance.

        Returns:
            (new_state, dt_used, dt_suggested_next)
        """
        dt = min(self._dt_next, dt_max)
        dt = max(dt, self.dt_min)

        for _ in range(20):   # safety limit on iterations
            y4, y5 = self._rk45_core(state_vec, t, dt, derivatives_fn)

            # Error estimate: norm of (y5 - y4)
            n = len(y4)
            err = math.sqrt(
                sum(((y5[i] - y4[i]) /
                     (self.atol + self.rtol * max(abs(state_vec[i]), abs(y4[i]), 1e-10))) ** 2
                    for i in range(n)) / n
            )

            if err <= 1.0:
                # Step accepted — propagate the 5th-order solution (local
                # extrapolation, standard Dormand-Prince practice); y5-y4 is
                # only the error estimate.
                factor = min(5.0, 0.9 * (1.0 / max(err, 1e-10)) ** 0.2)
                self._dt_next = min(dt * factor, self.dt_max)
                return y5, dt, self._dt_next
            else:
                # Reduce step
                factor = max(0.1, 0.9 * (1.0 / err) ** 0.25)
                dt = max(dt * factor, self.dt_min)

        # If we get here, use the last attempt anyway
        return y5, dt, self.dt_min

    def _rk45_core(self, state_vec, t, dt, fn):
        """Compute the 7 stage values and return (4th-order, 5th-order) estimates."""
        n = len(state_vec)
        k = [None] * 7
        k[0] = fn(t, state_vec)

        for stage in range(1, 7):
            c = self._C[stage]
            a = self._A[stage]
            s_i = [
                state_vec[j] + dt * sum(a[m] * k[m][j] for m in range(len(a)))
                for j in range(n)
            ]
            k[stage] = fn(t + c * dt, s_i)

        y4 = [state_vec[i] + dt * sum(self._B4[m] * k[m][i] for m in range(7)) for i in range(n)]
        y5 = [state_vec[i] + dt * sum(self._B5[m] * k[m][i] for m in range(7)) for i in range(n)]
        return y4, y5


# ── NaN Guard Wrapper ─────────────────────────────────────────────────────────

class NaNGuardIntegrator(IntegratorBase):
    """
    Wraps another integrator and validates the output for NaN / Inf.
    Raises ValueError immediately so the simulation engine can catch and
    stop cleanly instead of silently producing garbage data.
    """

    def __init__(self, inner: IntegratorBase):
        self._inner = inner

    @property
    def name(self) -> str:
        return f"{self._inner.name}+NaNGuard"

    def step(self, state_vec, t, dt, derivatives_fn):
        result = self._inner.step(state_vec, t, dt, derivatives_fn)
        for i, v in enumerate(result):
            if not math.isfinite(v):
                raise ValueError(
                    f"Non-finite value in state[{i}]={v} after integration at t={t:.4f}s"
                )
        return result


# ── Factory ───────────────────────────────────────────────────────────────────

_INTEGRATORS = {
    "euler": EulerIntegrator,
    "rk4":   RK4Integrator,
    "rk45":  RK45Integrator,
}


def get_integrator(name: str = "rk4", nan_guard: bool = True) -> IntegratorBase:
    """
    Get an integrator by name.

    Args:
        name:      'euler', 'rk4', or 'rk45'.
        nan_guard: Wrap with NaN validation (default True).

    Returns:
        IntegratorBase instance.
    """
    cls = _INTEGRATORS.get(name.lower())
    if cls is None:
        logger.warning(f"Unknown integrator '{name}', falling back to RK4")
        cls = RK4Integrator

    integrator = cls()
    if nan_guard:
        integrator = NaNGuardIntegrator(integrator)
    return integrator
