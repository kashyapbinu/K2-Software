"""
K2 Aerospace — Internal Ballistics Engine
===========================================
Simulates the thermodynamics and burn geometry of solid rocket motors.
Supports BATES grains and calculates steady-state chamber pressure,
mass flow, and ideal thrust coefficients.
"""

import math
from scipy.optimize import fsolve
from typing import List

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
