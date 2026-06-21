"""
K2 AeroSim — Internal Ballistics Engine
===========================================
Simulates the thermodynamics and burn geometry of solid rocket motors.
Supports BATES grains and calculates steady-state chamber pressure,
mass flow, and ideal thrust coefficients.
"""

import math
from scipy.optimize import fsolve
from typing import List

G0 = 9.80665
P_ATM = 101325.0


def nozzle_cf(pc: float, eps: float, gamma: float, ambient: float = P_ATM) -> tuple[float, float]:
    """Ideal nozzle thrust coefficient Cf and exit pressure Pe (Pa) for a given
    chamber pressure, area ratio and ambient, with a simple over-expansion
    (flow-separation) clamp. Mirrors the in-loop logic of MotorSimulator."""
    g = max(gamma, 1.01)
    if eps <= 1.0 or pc <= 0:
        return 0.0, pc
    # Pe/Pc from the area ratio (supersonic branch) via bisection.
    def area_ratio(pe_pc):
        t1 = ((g + 1) / 2.0) ** (1.0 / (g - 1.0))
        t2 = pe_pc ** (1.0 / g)
        t3 = math.sqrt(max((g + 1) / (g - 1) * (1 - pe_pc ** ((g - 1) / g)), 1e-12))
        return t1 / (t2 * t3)
    lo, hi = 1e-6, 0.999999
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        if area_ratio(mid) < eps:   # area_ratio decreases as pe_pc rises
            hi = mid
        else:
            lo = mid
    pe_pc = 0.5 * (lo + hi)
    pe = pe_pc * pc
    term1 = (2 * g * g) / (g - 1)
    term2 = (2 / (g + 1)) ** ((g + 1) / (g - 1))
    term3 = max(1 - pe_pc ** ((g - 1) / g), 0.0)
    cf_mom = math.sqrt(term1 * term2 * term3)
    if pe < ambient:   # over-expanded → separated, treat as perfectly expanded to Pa
        ideal_pe_pc = min(ambient / pc, 1.0)
        it3 = max(1 - ideal_pe_pc ** ((g - 1) / g), 0.0)
        return math.sqrt(term1 * term2 * it3), pe
    return cf_mom + (pe - ambient) / pc * eps, pe


def optimum_expansion_ratio(ambient: float, pc: float, gamma: float) -> float:
    """Area ratio that perfectly expands chamber pc to the given ambient."""
    if ambient <= 0:
        return 80.0
    if ambient >= pc:
        return 1.0
    g = max(gamma, 1.01)
    pe_pc = ambient / pc
    t1 = ((g + 1) / 2.0) ** (1.0 / (g - 1.0))
    t2 = pe_pc ** (1.0 / g)
    t3 = math.sqrt(max((g + 1) / (g - 1) * (1 - pe_pc ** ((g - 1) / g)), 1e-12))
    return max(1.5, min(200.0, t1 / (t2 * t3)))

class Propellant:
    def __init__(self, a: float, n: float, density: float, c_star: float, gamma: float = 1.2):
        self.a = a
        self.n = n
        self.density = density
        self.c_star = c_star
        self.gamma = gamma

class Grain:
    def get_burn_area(self, regression: float) -> float:
        raise NotImplementedError
    def get_web_thickness(self) -> float:
        raise NotImplementedError
    def get_volume(self) -> float:
        raise NotImplementedError
    def get_casing_volume(self) -> float:
        raise NotImplementedError
    def get_length(self) -> float:
        raise NotImplementedError

class BatesGrain(Grain):
    def __init__(self, length: float, outer_diameter: float, core_diameter: float,
                 inhibited_ends: bool = False):
        self.initial_length = length
        self.initial_od = outer_diameter
        self.initial_core = core_diameter
        self.inhibited_ends = inhibited_ends

    def get_burn_area(self, regression: float) -> float:
        x = regression
        L = self.initial_length
        if not self.inhibited_ends:
            L = self.initial_length - 2 * x
        
        d = self.initial_core + 2 * x
        D = self.initial_od
        
        if d >= D or L <= 0: return 0.0

        A_core = math.pi * d * L
        A_ends = 0.0 if self.inhibited_ends else 2 * (math.pi / 4.0) * (D**2 - d**2)
        return max(0.0, A_core + A_ends)

    def get_web_thickness(self) -> float:
        return (self.initial_od - self.initial_core) / 2.0
        
    def get_volume(self) -> float:
        return self.initial_length * (math.pi/4) * (self.initial_od**2 - self.initial_core**2)
        
    def get_casing_volume(self) -> float:
        return self.initial_length * (math.pi/4) * (self.initial_od**2)
        
    def get_length(self) -> float:
        return self.initial_length

class TubularGrain(Grain):
    """Core burning only. Ends and outer surface are inhibited."""
    def __init__(self, length: float, outer_diameter: float, core_diameter: float):
        self.length = length
        self.initial_od = outer_diameter
        self.initial_core = core_diameter

    def get_burn_area(self, regression: float) -> float:
        d = self.initial_core + 2 * regression
        if d >= self.initial_od: return 0.0
        return math.pi * d * self.length

    def get_web_thickness(self) -> float:
        return (self.initial_od - self.initial_core) / 2.0
        
    def get_volume(self) -> float:
        return self.length * (math.pi/4) * (self.initial_od**2 - self.initial_core**2)
        
    def get_casing_volume(self) -> float:
        return self.length * (math.pi/4) * (self.initial_od**2)
        
    def get_length(self) -> float:
        return self.length

class EndBurnerGrain(Grain):
    """Burns only from one end. Solid cylinder."""
    def __init__(self, length: float, diameter: float):
        self.initial_length = length
        self.diameter = diameter

    def get_burn_area(self, regression: float) -> float:
        if regression >= self.initial_length: return 0.0
        return (math.pi / 4.0) * (self.diameter**2)

    def get_web_thickness(self) -> float:
        return self.initial_length
        
    def get_volume(self) -> float:
        return self.initial_length * (math.pi/4) * (self.diameter**2)
        
    def get_casing_volume(self) -> float:
        return self.get_volume()
        
    def get_length(self) -> float:
        return self.initial_length

class StarGrain(Grain):
    """Analytical approximation of a Star Grain."""
    def __init__(self, length: float, outer_diameter: float, core_diameter: float, 
                 points: int, point_depth: float):
        self.length = length
        self.od = outer_diameter
        self.core = core_diameter
        self.points = points
        self.depth = point_depth
        
        # Calculate initial perimeter geometrically
        self.r_inner = self.core / 2.0
        self.r_outer = self.r_inner + self.depth
        if self.r_outer > self.od / 2.0:
            self.r_outer = self.od / 2.0
            
        alpha = math.pi / self.points
        # Length of one side of the star point
        self.L0 = math.sqrt(self.r_inner**2 + self.r_outer**2 - 2*self.r_inner*self.r_outer*math.cos(alpha))
        self.initial_perimeter = 2 * self.points * self.L0

    def get_burn_area(self, regression: float) -> float:
        x = regression
        max_x = (self.od / 2.0) - self.r_inner
        if x >= max_x: return 0.0
        
        # Simplified analytical model for star progression:
        # Phase 1: perimeter grows slightly due to inner rounding.
        # Phase 2: points hit the casing and perimeter decreases rapidly.
        # We approximate the perimeter P(x) based on geometric progression.
        
        # Time when the inner web hits the outer radius (if points didn't touch it)
        web_to_casing = (self.od/2.0) - self.r_inner
        
        # Distance outer point travels before hitting casing
        point_travel = (self.od/2.0) - self.r_outer
        
        if x < point_travel:
            # Phase 1: Neutral / slightly progressive
            # The perimeter increases by the arc generation at the inner vertices
            arc_growth = math.pi # Approximation of perimeter growth factor
            P = self.initial_perimeter + arc_growth * x
        else:
            # Phase 2: Regressive (points hit casing)
            # Linearly decay to 0 as regression goes from point_travel to max_x
            arc_growth = math.pi
            P_peak = self.initial_perimeter + arc_growth * point_travel
            progress = (x - point_travel) / (max_x - point_travel) if max_x > point_travel else 1.0
            P = P_peak * (1.0 - progress)
            
        return max(0.0, P * self.length)

    def get_web_thickness(self) -> float:
        return (self.od / 2.0) - self.r_inner
        
    def get_volume(self) -> float:
        # Star area approx
        A_circle = math.pi * self.r_inner**2
        A_points = self.points * self.r_inner * self.depth * math.sin(math.pi/self.points)
        return self.length * ((math.pi/4)*self.od**2 - (A_circle + A_points))
        
    def get_casing_volume(self) -> float:
        return self.length * (math.pi/4) * (self.od**2)
        
    def get_length(self) -> float:
        return self.length

class MotorSimulator:
    def __init__(self, propellant: Propellant, grains: list[Grain], 
                 throat_diameter: float, exit_diameter: float, 
                 ambient_pressure: float = 101325.0, efficiency: float = 0.95):
        self.propellant = propellant
        self.grains = grains
        self.throat_area = math.pi * (throat_diameter / 2.0)**2
        self.exit_area = math.pi * (exit_diameter / 2.0)**2
        self.eps = self.exit_area / self.throat_area if self.throat_area > 0 else 1.0
        self.ambient_pressure = ambient_pressure
        self.efficiency = efficiency

    def _calc_exit_pressure_ratio(self) -> float:
        """Finds Pe/Pc for the given expansion ratio."""
        if self.eps <= 1.0: return 1.0
        g = self.propellant.gamma
        def eq(pe_pc):
            if pe_pc <= 0 or pe_pc >= 1: return 1000
            term1 = ((g+1)/2.0)**(1.0/(g-1.0))
            term2 = pe_pc**(1.0/g)
            term3 = math.sqrt((g+1)/(g-1) * (1 - pe_pc**((g-1)/g)))
            if term2 == 0 or term3 == 0: return 1000
            return (term1 / (term2 * term3)) - self.eps
        try:
            res = fsolve(eq, 0.01)
            return res[0]
        except Exception:
            return 0.1 # Fallback

    def simulate(self, dt: float = 0.01) -> dict:
        times = [0.0]
        thrusts = [0.0]
        pressures = [101325.0]
        kns = [0.0]
        mass_fluxes = [0.0]
        regressions = [0.0]
        
        max_web = max(g.get_web_thickness() for g in self.grains) if self.grains else 0
        regression = 0.0
        time = 0.0
        pe_pc_ratio = self._calc_exit_pressure_ratio()
        
        rho = self.propellant.density
        a = self.propellant.a
        n = min(self.propellant.n, 0.99) # Prevent divide by zero in Pc calculation
        c_star = self.propellant.c_star
        gamma = max(self.propellant.gamma, 1.01) # Prevent divide by zero in Cf calculation

        # Initial metrics
        initial_port_area = 0.0
        if self.grains:
            g0 = self.grains[0]
            if hasattr(g0, 'initial_core'):
                initial_port_area = math.pi * (g0.initial_core / 2.0)**2
            elif hasattr(g0, 'r_inner'):
                initial_port_area = math.pi * (g0.r_inner)**2
            else:
                initial_port_area = 0.0
            
        port_to_throat = initial_port_area / self.throat_area if self.throat_area > 0 else 0.0
        
        prop_mass = 0.0
        prop_vol = 0.0
        casing_vol = 0.0
        prop_len = 0.0
        for g in self.grains:
            vol = g.get_volume()
            c_vol = g.get_casing_volume()
            prop_vol += vol
            casing_vol += c_vol
            prop_mass += vol * rho
            prop_len += g.get_length()
            
        vol_loading = (prop_vol / casing_vol) * 100 if casing_vol > 0 else 0.0

        while regression < max_web:
            A_b = sum(g.get_burn_area(regression) for g in self.grains)
            if A_b <= 0:
                break
                
            Kn = A_b / self.throat_area
            
            # Steady state chamber pressure
            try:
                base = Kn * rho * a * c_star
                Pc = base ** (1.0 / (1.0 - n))
            except Exception:
                Pc = 101325.0
                
            if Pc <= self.ambient_pressure:
                Pc = self.ambient_pressure
                
            # Burn rate and regression
            r = a * (Pc ** n)
            regression += r * dt
            time += dt
            
            # Mass Flow and Flux
            mass_flow = A_b * r * rho
            current_port_area = 0.0
            if self.grains:
                g0 = self.grains[0]
                if hasattr(g0, 'initial_core'):
                    current_port_area = math.pi * ((g0.initial_core + 2*regression) / 2.0)**2
                elif hasattr(g0, 'r_inner'):
                    # Star grain approximation
                    current_port_area = math.pi * ((g0.r_inner*2 + 2*regression) / 2.0)**2
            
            mass_flux = mass_flow / current_port_area if current_port_area > 0 else 0.0
            
            # Thrust calculation
            term1 = (2 * gamma**2) / (gamma - 1)
            term2 = (2 / (gamma + 1))**((gamma + 1)/(gamma - 1))
            term3 = 1 - (pe_pc_ratio)**((gamma - 1)/gamma)
            
            if term3 < 0: term3 = 0
            momentum_thrust_coeff = math.sqrt(term1 * term2 * term3)
            
            pe = Pc * pe_pc_ratio
            pressure_thrust_coeff = ((pe - self.ambient_pressure) / Pc) * self.eps
            
            # Flow separation check (Simplified Summerfield / Kalt-Bader)
            # If the nozzle is over-expanded such that Pe is significantly lower than Pa,
            # flow will separate. A simple approximation is to use the perfectly expanded Cf.
            if pe < self.ambient_pressure:
                # Flow separates. Calculate Cf as if nozzle expanded perfectly to Pa
                ideal_pe_pc = self.ambient_pressure / Pc if Pc > self.ambient_pressure else 1.0
                ideal_term3 = 1 - (ideal_pe_pc)**((gamma - 1)/gamma)
                if ideal_term3 < 0: ideal_term3 = 0
                cf = math.sqrt(term1 * term2 * ideal_term3)
            else:
                cf = momentum_thrust_coeff + pressure_thrust_coeff
                
            if cf < 0.0:
                cf = 0.0
            thrust = self.efficiency * cf * self.throat_area * Pc
            
            times.append(time)
            thrusts.append(thrust)
            pressures.append(Pc)
            kns.append(Kn)
            mass_fluxes.append(mass_flux)
            regressions.append(regression)

        total_impulse = 0.0
        for i in range(1, len(times)):
            dt_step = times[i] - times[i-1]
            avg_t = (thrusts[i] + thrusts[i-1]) / 2.0
            total_impulse += avg_t * dt_step
            
        delivered_isp = total_impulse / (prop_mass * 9.80665) if prop_mass > 0 else 0.0

        core_ld = 0.0
        if self.grains:
            g0 = self.grains[0]
            if hasattr(g0, 'initial_core') and g0.initial_core > 0:
                core_ld = g0.get_length() / g0.initial_core
            elif hasattr(g0, 'core') and g0.core > 0:
                core_ld = g0.get_length() / g0.core

        return {
            "time": times,
            "thrust": thrusts,
            "pressure": pressures,
            "kn": kns,
            "mass_flux": mass_fluxes,
            "regression": regressions,
            "prop_mass": prop_mass,
            "metrics": {
                "initial_kn": kns[1] if len(kns) > 1 else 0.0,
                "max_kn": max(kns),
                "max_pc": max(pressures),
                "vol_loading": vol_loading,
                "port_to_throat": port_to_throat,
                "throat_to_port": 1.0 / port_to_throat if port_to_throat > 0 else 0.0,
                "core_l_d": core_ld,
                "web": max_web,
                "prop_len": prop_len,
                "peak_mass_flux": max(mass_fluxes),
                "total_impulse": total_impulse,
                "delivered_isp": delivered_isp
            }
        }

    # ── conceptual nozzle / chamber / performance design summary ───────────
    def design_summary(self, res: dict, chamber_diameter: float,
                       rocket_total_mass: float = 0.0,
                       struct_frac: float = 0.5) -> dict:
        """First-order nozzle geometry, ideal-vs-delivered performance,
        chamber metrics, Δv / TWR and engineering validation warnings for a
        completed solid-motor simulation. `chamber_diameter` is the grain
        outer / casing inner diameter (m)."""
        g = max(self.propellant.gamma, 1.01)
        m = res["metrics"]
        pressures = [p for p in res["pressure"] if p > self.ambient_pressure * 1.01]
        pc_mean = sum(pressures) / len(pressures) if pressures else m["max_pc"]
        thrusts = res["thrust"]
        max_thrust = max(thrusts) if thrusts else 0.0
        burn_time = res["time"][-1] if res["time"] else 0.0
        avg_thrust = m["total_impulse"] / burn_time if burn_time > 0 else 0.0

        throat_d = 2.0 * math.sqrt(self.throat_area / math.pi)
        exit_d = 2.0 * math.sqrt(self.exit_area / math.pi)

        cf_sl, pe = nozzle_cf(pc_mean, self.eps, g, P_ATM)
        cf_vac, _ = nozzle_cf(pc_mean, self.eps, g, 0.0)
        cstar = self.propellant.c_star
        isp_ideal = cf_sl * cstar / G0
        isp_vac = cf_vac * cstar / G0 * self.efficiency
        isp_delivered = m["delivered_isp"]
        efficiency = isp_delivered / isp_ideal if isp_ideal > 0 else 0.0

        # Geometry / chamber metrics.
        chamber_area = math.pi * (chamber_diameter / 2.0) ** 2
        chamber_len = m["prop_len"]
        chamber_vol = chamber_area * chamber_len
        contraction = chamber_area / self.throat_area if self.throat_area > 0 else 0.0
        l_star = chamber_vol / self.throat_area if self.throat_area > 0 else 0.0
        nozzle_len = (exit_d - throat_d) / 2.0 / math.tan(math.radians(15.0))
        residence = l_star / cstar if cstar > 0 else 0.0   # t_s ≈ L*/c*
        opt_eps = optimum_expansion_ratio(self.ambient_pressure, pc_mean, g)

        # Masses / Δv / TWR.
        prop_mass = res["prop_mass"]
        dry_mass = prop_mass * struct_frac
        wet_mass = prop_mass + dry_mass
        mass_ratio = wet_mass / dry_mass if dry_mass > 0 else 0.0
        delta_v = isp_delivered * G0 * math.log(mass_ratio) if mass_ratio > 1 else 0.0
        twr_engine = max_thrust / (wet_mass * G0) if wet_mass > 0 else 0.0
        twr_liftoff = twr_burnout = 0.0
        if rocket_total_mass > 0:
            twr_liftoff = max_thrust / (rocket_total_mass * G0)
            burnout = max(rocket_total_mass - prop_mass, 1e-6)
            twr_burnout = max_thrust / (burnout * G0)

        warnings = []
        ptt = m["port_to_throat"]
        if 0 < ptt < 2.0:
            warnings.append(f"Port-to-throat {ptt:.2f} < 2 — erosive burning / pressure spike risk at ignition. Use a larger core or smaller throat.")
        if m["peak_mass_flux"] > 1400.0:
            warnings.append(f"Peak mass flux {m['peak_mass_flux']:.0f} kg/m²s exceeds ~1400 — erosive burning likely.")
        if m["max_pc"] / 1e5 > 100.0:
            warnings.append(f"Max chamber pressure {m['max_pc']/1e5:.0f} bar is high — verify case & closure strength (MEOP margin).")
        if m["max_pc"] / 1e5 < 10.0:
            warnings.append(f"Max chamber pressure {m['max_pc']/1e5:.1f} bar is low — burn rate exponent regime may be unstable.")
        if self.ambient_pressure > 0 and pe < 0.35 * self.ambient_pressure:
            warnings.append(f"Overexpanded: Pe={pe/1e5:.2f} bar ≪ ambient — flow separation at sea level. Reduce exit diameter (ε≈{opt_eps:.1f}).")
        if self.ambient_pressure > 0 and pe > 4.0 * self.ambient_pressure:
            warnings.append(f"Underexpanded — a larger exit (ε≈{opt_eps:.1f}) would recover thrust.")
        if m["core_l_d"] > 8.0:
            warnings.append(f"Core L/D {m['core_l_d']:.1f} is very high — risk of erosive burning down the bore.")
        if m["vol_loading"] > 95.0:
            warnings.append(f"Volumetric loading {m['vol_loading']:.0f}% leaves little port area — high initial Kn.")
        if efficiency > 1.0:
            warnings.append("Delivered Isp exceeds ideal — check c*/efficiency inputs.")

        return {
            "throat_diameter": throat_d, "exit_diameter": exit_d,
            "chamber_diameter": chamber_diameter, "chamber_length": chamber_len,
            "expansion_ratio": self.eps, "contraction_ratio": contraction,
            "l_star": l_star, "nozzle_length": nozzle_len,
            "residence_time": residence, "exit_pressure": pe,
            "pc_mean": pc_mean, "opt_expansion": opt_eps,
            "isp_ideal": isp_ideal, "isp_delivered": isp_delivered, "isp_vac": isp_vac,
            "cf_sl": cf_sl, "cf_vac": cf_vac, "efficiency": efficiency,
            "thrust_ideal": max_thrust / self.efficiency if self.efficiency > 0 else max_thrust,
            "thrust_delivered": max_thrust, "avg_thrust": avg_thrust,
            "prop_mass": prop_mass, "dry_mass": dry_mass, "wet_mass": wet_mass,
            "mass_ratio": mass_ratio, "delta_v": delta_v,
            "twr_engine": twr_engine, "twr_liftoff": twr_liftoff, "twr_burnout": twr_burnout,
            "warnings": warnings,
        }
