"""
K2 Aerospace — Propulsion Module
"""
import math, numpy as np, logging
from core.constants import G_EARTH
logger = logging.getLogger("K2.Propulsion")

def compute_isp(total_impulse, propellant_mass):
    return total_impulse / (propellant_mass * G_EARTH) if propellant_mass > 0 else 0.0

def compute_mass_flow_rate(propellant_mass, burn_time):
    return propellant_mass / burn_time if burn_time > 0 else 0.0

def estimate_chamber_pressure(avg_thrust, throat_diameter=0.01, cf=1.5):
    """Rough chamber-pressure estimate from F = cf · Pc · A_throat → Pc = F/(cf·At).

    ``cf`` is the nozzle thrust coefficient; 1.5 is a sea-level default for a
    moderately-expanded nozzle (real range ≈ 1.3–1.7, rising with expansion
    ratio and altitude). For an accurate value use the full nozzle solve in
    ``physics.internal_ballistics.MotorSimulator``; this is a quick first cut.
    """
    throat_area = math.pi * (throat_diameter / 2) ** 2
    return avg_thrust / (cf * throat_area) if throat_area > 0 else 0.0

def generate_thrust_curve(avg_thrust, max_thrust, burn_time, num_points=200):
    if burn_time <= 0 or avg_thrust <= 0:
        return np.array([0.0]), np.array([0.0])
    if max_thrust <= 0:
        max_thrust = avg_thrust * 1.4
    ramp = burn_time * 0.1
    t = np.array([0.0, ramp, burn_time - ramp, burn_time, burn_time * 1.05])
    f = np.array([0.0, max_thrust, avg_thrust, 0.0, 0.0])
    t_interp = np.linspace(0, burn_time * 1.05, num_points)
    f_interp = np.interp(t_interp, t, f)
    return t_interp, f_interp
