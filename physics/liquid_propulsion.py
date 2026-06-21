"""
K2 AeroSim — Liquid / Bi-Propellant Engine Performance & Conceptual Design
==========================================================================
Ideal-rocket performance model for liquid (and monopropellant) engines, sized
from chamber pressure, mixture ratio and nozzle expansion ratio. Produces the
SAME result schema as the solid MotorSimulator (time/thrust arrays + prop_mass
+ metrics), so a designed liquid engine feeds the rocket through the existing
custom-motor pipeline (single-stage and per-stage multistage) unchanged.

Beyond raw performance this module now also returns first-order *conceptual
design* estimates suitable for student / preliminary work (NOT flight design):

  * Nozzle + chamber geometry (throat/exit/chamber dia, L*, contraction ratio,
    chamber & nozzle length, residence time, combustion efficiency).
  * Propellant tank sizing (masses, volumes, ullage, cylindrical dimensions,
    wet/dry mass).
  * Injector sizing (pressure drop, area, velocity, hole count/diameter,
    suggested pattern).
  * Cooling estimates (wall heat flux via simplified Bartz, coolant flow,
    effectiveness) per cooling method.
  * Ideal-vs-delivered performance split and overall efficiency.
  * Mixture-ratio optimisation, ambient/altitude optimisation.
  * Delta-V (Tsiolkovsky), thrust/weight, sensitivity sweeps and a set of
    engineering validation warnings.

Physics
-------
Nozzle flow is treated as 1-D isentropic with a frozen ratio of specific heats
γ and a tabulated characteristic velocity c* per propellant combination (c*
already folds in the combustion temperature / gas properties):

    At     = F / (Cf · Pc)                 throat area from target thrust
    mdot   = Pc · At / c*                  total mass flow (choked throat)
    F      = Cf · Pc · At                  thrust
    Isp    = Cf · c* / g0                  specific impulse

Cf (thrust coefficient) and the exit pressure come from the supersonic area
ratio ε = Ae/At via the isentropic area–Mach relation. c*- and Cf-efficiency
factors knock the ideal numbers down to realistic delivered values.

Values in PROPELLANT_COMBOS are nominal textbook figures (optimum O/F, ~70 bar
chamber) — good for trajectory-level sizing, not a substitute for a CEA run.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

G0 = 9.80665
P_ATM = 101325.0          # sea-level ambient (Pa)
R_UNIV = 8314.462         # universal gas constant (J/kmol·K)


# ── Propellant combinations ──────────────────────────────────────────────
# c_star (m/s), Tc (K), gamma, mol_wt (g/mol), of (optimum O/F mass ratio),
# rho_ox / rho_fuel (kg/m³), l_star (m, characteristic chamber length).
@dataclass
class PropellantCombo:
    name: str
    c_star: float
    Tc: float
    gamma: float
    mol_wt: float
    of_ratio: float
    rho_ox: float
    rho_fuel: float
    monoprop: bool = False
    l_star: float = 1.0        # characteristic chamber length L* (m)
    of_min: float = 0.0        # recommended O/F window (0 → auto)
    of_max: float = 0.0


PROPELLANT_COMBOS = {
    "LOX/RP-1":      PropellantCombo("LOX/RP-1",     1823.0, 3670, 1.24,  23.3, 2.56, 1141, 810, l_star=1.0,  of_min=2.0, of_max=2.8),
    "LOX/CH4":       PropellantCombo("LOX/CH4",      1857.0, 3550, 1.20,  20.3, 3.60, 1141, 423, l_star=1.1,  of_min=2.8, of_max=3.8),
    "LOX/LH2":       PropellantCombo("LOX/LH2",      2386.0, 3400, 1.26,  10.0, 4.50, 1141,  71, l_star=0.9,  of_min=4.0, of_max=6.0),
    "N2O4/MMH":      PropellantCombo("N2O4/MMH",     1720.0, 3400, 1.23,  21.0, 2.17, 1440, 880, l_star=0.9,  of_min=1.6, of_max=2.4),
    "N2O4/UDMH":     PropellantCombo("N2O4/UDMH",    1710.0, 3380, 1.24,  21.5, 2.61, 1440, 793, l_star=0.9,  of_min=2.0, of_max=3.0),
    "HTP/RP-1":      PropellantCombo("HTP/RP-1",     1607.0, 2880, 1.21,  21.0, 7.00, 1390, 810, l_star=1.1,  of_min=6.0, of_max=8.0),
    "N2O/IPA":       PropellantCombo("N2O/IPA",      1480.0, 2900, 1.23,  24.0, 3.00,  745, 786, l_star=1.3,  of_min=2.5, of_max=4.0),
    "N2O/Ethanol":   PropellantCombo("N2O/Ethanol",  1500.0, 2960, 1.23,  24.5, 3.20,  745, 789, l_star=1.3,  of_min=2.8, of_max=4.2),
    "Hydrazine (mono)": PropellantCombo("Hydrazine (mono)", 1310.0, 1200, 1.27, 13.0, 0.0, 1004, 1004, monoprop=True, l_star=0.5),
    "H2O2 (mono)":   PropellantCombo("H2O2 (mono)",  1050.0, 1030, 1.25,  22.0, 0.0, 1390, 1390, monoprop=True, l_star=0.6),
}


# Engine cycles — only Pressure-fed is fully modelled; others carry conceptual
# notes + a rough chamber-pressure ceiling for the validation warnings.
ENGINE_CYCLES = {
    "Pressure-fed":       {"pc_max": 35.0,  "supported": True,
                           "note": "Tank pressure feeds the chamber directly. Simple, but tank mass grows with Pc — keep Pc low (≲35 bar)."},
    "Gas Generator":      {"pc_max": 120.0, "supported": False,
                           "note": "Conceptual: a small pre-burner drives the turbopumps; ~1-3% of flow dumped overboard (Isp penalty applied)."},
    "Staged Combustion":  {"pc_max": 300.0, "supported": False,
                           "note": "Conceptual: full-flow / ox-rich pre-burner exhaust routed to the main chamber. Highest Pc & Isp, very complex."},
    "Expander Cycle":     {"pc_max": 100.0, "supported": False,
                           "note": "Conceptual: regen coolant vaporises and drives the turbine. Cryo fuels only; Pc limited by available heat."},
    "Electric Pump-fed":  {"pc_max": 80.0,  "supported": False,
                           "note": "Conceptual: battery-driven electric pumps. Simple plumbing; battery mass added to dry mass estimate."},
}

COOLING_METHODS = ["Regenerative", "Film Cooling", "Ablative", "Radiative"]

THRUST_PROFILES = ["Constant", "Linear Ramp", "Bell-shaped", "Throttle schedule", "User-defined CSV"]

OPT_MODES = ["Off", "Maximise Isp", "Maximise density impulse", "Maximise thrust", "Balanced optimum"]


def exit_mach_from_area_ratio(eps: float, gamma: float) -> float:
    """Supersonic exit Mach number for a nozzle area ratio ε = Ae/At.

    Solves the isentropic area–Mach relation by bisection on the supersonic
    branch (monotonic for M>1, so robust without scipy)."""
    if eps <= 1.0:
        return 1.0
    g = gamma
    exp = (g + 1.0) / (2.0 * (g - 1.0))

    def area_ratio(M: float) -> float:
        return (1.0 / M) * ((2.0 / (g + 1.0)) *
                            (1.0 + 0.5 * (g - 1.0) * M * M)) ** exp

    lo, hi = 1.0000, 50.0
    # area_ratio is increasing in M on (1, ∞); bisect to the target ε.
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if area_ratio(mid) < eps:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def thrust_coefficient(pc: float, eps: float, gamma: float,
                       p_ambient: float = P_ATM) -> tuple[float, float]:
    """Ideal nozzle thrust coefficient Cf and exit pressure Pe (Pa).

    Cf = √[ (2γ²/(γ-1))·(2/(γ+1))^((γ+1)/(γ-1))·(1-(Pe/Pc)^((γ-1)/γ)) ]
         + ((Pe-Pa)/Pc)·ε
    """
    g = gamma
    Me = exit_mach_from_area_ratio(eps, g)
    pe_pc = (1.0 + 0.5 * (g - 1.0) * Me * Me) ** (-g / (g - 1.0))
    pe = pe_pc * pc

    term = (2.0 * g * g / (g - 1.0)) \
        * (2.0 / (g + 1.0)) ** ((g + 1.0) / (g - 1.0)) \
        * (1.0 - pe_pc ** ((g - 1.0) / g))
    cf_momentum = math.sqrt(max(term, 0.0))
    cf_pressure = (pe - p_ambient) / pc * eps
    return cf_momentum + cf_pressure, pe


def expansion_ratio_for_exit_pressure(pe: float, pc: float, gamma: float) -> float:
    """Area ratio ε that gives exit pressure pe for chamber pc (perfectly
    expanded nozzle). Inverts the isentropic Pe/Pc → Me → ε chain."""
    if pe <= 0 or pe >= pc:
        return 1.0
    g = gamma
    pr = pe / pc
    Me = math.sqrt(max((pr ** (-(g - 1.0) / g) - 1.0) * 2.0 / (g - 1.0), 0.0))
    if Me <= 1.0:
        return 1.0
    exp = (g + 1.0) / (2.0 * (g - 1.0))
    return (1.0 / Me) * ((2.0 / (g + 1.0)) * (1.0 + 0.5 * (g - 1.0) * Me * Me)) ** exp


def ambient_pressure_at_altitude(alt_m: float) -> float:
    """ISA-ish ambient pressure (Pa) for the auto-optimise-for-altitude mode."""
    if alt_m <= 0:
        return P_ATM
    if alt_m < 11000.0:
        T = 288.15 - 0.0065 * alt_m
        return P_ATM * (T / 288.15) ** 5.2561
    # Above the tropopause use an exponential fall-off (good enough for sizing).
    return 22632.0 * math.exp(-(alt_m - 11000.0) / 6341.6)


@dataclass
class LiquidEngineDesign:
    """Sized liquid/monoprop engine + its full conceptual-design summary."""
    combo: PropellantCombo

    # Design inputs
    chamber_pressure: float = 70e5     # Pa
    expansion_ratio: float = 12.0      # Ae/At
    target_thrust: float = 5000.0      # N (at the design ambient)
    burn_time: float = 10.0            # s
    of_ratio: float = 0.0              # 0 → use combo optimum
    ambient_pressure: float = P_ATM    # design ambient (sea level / 0 for vac)
    cstar_eff: float = 0.95            # combustion / c* efficiency
    cf_eff: float = 0.97               # nozzle / Cf efficiency
    struct_frac: float = 0.30          # dry mass / propellant mass (tanks+plumbing)
    startup: float = 0.25              # thrust ramp-up time (s)
    shutdown: float = 0.15             # thrust ramp-down time (s)

    # Conceptual-design inputs
    cycle: str = "Pressure-fed"
    cooling: str = "Regenerative"
    thrust_profile: str = "Constant"
    l_star: float = 0.0                # 0 → use combo default
    tank_diameter: float = 0.0         # 0 → auto from volume (slender)
    ullage_frac: float = 0.06          # ullage as fraction of propellant volume
    injector_cd: float = 0.75          # injector orifice discharge coefficient
    injector_dp_frac: float = 0.20     # injector ΔP as fraction of Pc
    injector_hole_d: float = 1.0e-3    # target injector hole diameter (m)
    wall_temp: float = 800.0           # gas-side wall temperature target (K)
    rocket_total_mass: float = 0.0     # full vehicle wet mass (kg) if linked
    throttle_points: list = field(default_factory=list)  # (t,frac) for schedule
    csv_curve: list = field(default_factory=list)         # (t,thrust) user curve

    # ── Derived: performance ─────────────────────────────────────────────
    cf: float = field(default=0.0, init=False)
    cf_ideal: float = field(default=0.0, init=False)
    exit_pressure: float = field(default=0.0, init=False)
    throat_area: float = field(default=0.0, init=False)
    throat_diameter: float = field(default=0.0, init=False)
    exit_diameter: float = field(default=0.0, init=False)
    mdot: float = field(default=0.0, init=False)
    mdot_ox: float = field(default=0.0, init=False)
    mdot_fuel: float = field(default=0.0, init=False)
    isp: float = field(default=0.0, init=False)          # delivered (design ambient)
    isp_vac: float = field(default=0.0, init=False)      # delivered vacuum
    isp_ideal: float = field(default=0.0, init=False)    # ideal (no losses)
    thrust_ideal: float = field(default=0.0, init=False)
    efficiency: float = field(default=0.0, init=False)   # delivered/ideal Isp

    # ── Derived: geometry ────────────────────────────────────────────────
    chamber_diameter: float = field(default=0.0, init=False)
    chamber_length: float = field(default=0.0, init=False)
    chamber_volume: float = field(default=0.0, init=False)
    contraction_ratio: float = field(default=0.0, init=False)
    nozzle_length: float = field(default=0.0, init=False)
    l_star_eff: float = field(default=0.0, init=False)
    residence_time: float = field(default=0.0, init=False)
    comb_efficiency: float = field(default=0.0, init=False)

    # ── Derived: masses / tanks ──────────────────────────────────────────
    prop_mass: float = field(default=0.0, init=False)
    ox_mass: float = field(default=0.0, init=False)
    fuel_mass: float = field(default=0.0, init=False)
    ox_volume: float = field(default=0.0, init=False)
    fuel_volume: float = field(default=0.0, init=False)
    ullage_volume: float = field(default=0.0, init=False)
    tank_dia: float = field(default=0.0, init=False)
    ox_tank_len: float = field(default=0.0, init=False)
    fuel_tank_len: float = field(default=0.0, init=False)
    tank_mass: float = field(default=0.0, init=False)
    engine_mass: float = field(default=0.0, init=False)
    dry_mass: float = field(default=0.0, init=False)
    wet_mass: float = field(default=0.0, init=False)
    total_impulse: float = field(default=0.0, init=False)

    # ── Derived: injector / cooling / dv ─────────────────────────────────
    inj_dp: float = field(default=0.0, init=False)
    inj_area: float = field(default=0.0, init=False)
    inj_velocity: float = field(default=0.0, init=False)
    inj_n_holes: int = field(default=0, init=False)
    inj_pattern: str = field(default="", init=False)
    heat_flux: float = field(default=0.0, init=False)
    coolant_flow: float = field(default=0.0, init=False)
    cooling_dp: float = field(default=0.0, init=False)
    cooling_eff: float = field(default=0.0, init=False)
    mass_ratio: float = field(default=0.0, init=False)
    delta_v: float = field(default=0.0, init=False)
    twr_engine: float = field(default=0.0, init=False)
    twr_liftoff: float = field(default=0.0, init=False)
    twr_burnout: float = field(default=0.0, init=False)
    warnings: list = field(default_factory=list, init=False)

    # ── helpers ──────────────────────────────────────────────────────────
    def _of(self) -> float:
        return self.of_ratio if self.of_ratio > 0 else self.combo.of_ratio

    def _bulk_density(self, of: float | None = None) -> float:
        c = self.combo
        if c.monoprop or c.rho_ox <= 0 or c.rho_fuel <= 0:
            return c.rho_fuel
        of = self._of() if of is None else of
        # mass-weighted: 1/ρ = (of/(1+of))/ρox + (1/(1+of))/ρfuel
        inv = (of / (1.0 + of)) / c.rho_ox + (1.0 / (1.0 + of)) / c.rho_fuel
        return 1.0 / inv if inv > 0 else c.rho_fuel

    # ── main solve ────────────────────────────────────────────────────────
    def solve(self) -> "LiquidEngineDesign":
        c = self.combo
        g = c.gamma
        cstar_ideal = c.c_star
        cstar = cstar_ideal * self.cstar_eff
        of = self._of()

        # Apply a small conceptual cycle Isp penalty (open cycles dump gas).
        cycle_eff = {"Gas Generator": 0.98, "Electric Pump-fed": 0.995}.get(self.cycle, 1.0)

        self.cf_ideal, pe = thrust_coefficient(
            self.chamber_pressure, self.expansion_ratio, g, self.ambient_pressure)
        cf_vac, _ = thrust_coefficient(
            self.chamber_pressure, self.expansion_ratio, g, 0.0)
        self.cf = self.cf_ideal * self.cf_eff
        self.exit_pressure = pe

        # Throat sized to hit the target thrust at the design ambient.
        self.throat_area = self.target_thrust / max(self.cf * self.chamber_pressure, 1e-9)
        self.throat_diameter = math.sqrt(4.0 * self.throat_area / math.pi)
        self.exit_diameter = math.sqrt(4.0 * self.throat_area * self.expansion_ratio / math.pi)

        self.mdot = self.chamber_pressure * self.throat_area / cstar
        if c.monoprop:
            self.mdot_ox, self.mdot_fuel = 0.0, self.mdot
        else:
            self.mdot_ox = self.mdot * of / (1.0 + of)
            self.mdot_fuel = self.mdot / (1.0 + of)

        # Ideal vs delivered Isp / thrust.
        self.isp_ideal = self.cf_ideal * cstar_ideal / G0
        self.isp = self.cf * cstar / G0 * cycle_eff
        self.isp_vac = cf_vac * self.cf_eff * cstar / G0 * cycle_eff
        self.thrust_ideal = self.cf_ideal * self.chamber_pressure * self.throat_area
        self.efficiency = self.isp / self.isp_ideal if self.isp_ideal > 0 else 0.0

        self._solve_geometry()
        self._solve_masses(of)
        self._solve_injector(of)
        self._solve_cooling()
        self._solve_dv_twr()
        self._validate()
        return self

    def _solve_geometry(self):
        c = self.combo
        self.l_star_eff = self.l_star if self.l_star > 0 else c.l_star
        # Contraction ratio (Huzel & Huang empirical, throat dia in cm).
        dt_cm = max(self.throat_diameter * 100.0, 0.1)
        self.contraction_ratio = max(2.0, 8.0 * dt_cm ** -0.6 + 1.25)
        chamber_area = self.contraction_ratio * self.throat_area
        self.chamber_diameter = math.sqrt(4.0 * chamber_area / math.pi)
        # Chamber volume from L*; cylindrical length from volume / area.
        self.chamber_volume = self.l_star_eff * self.throat_area
        self.chamber_length = self.chamber_volume / chamber_area
        # 80% bell nozzle length ≈ 0.8 × conical(15°) length.
        rt, re = self.throat_diameter / 2.0, self.exit_diameter / 2.0
        self.nozzle_length = 0.8 * (re - rt) / math.tan(math.radians(15.0))
        # Residence time = Vc / (mdot / ρ_gas);  ρ_gas from ideal gas at Pc,Tc.
        rho_gas = self.chamber_pressure * (c.mol_wt / 1000.0) / (R_UNIV / 1000.0 * c.Tc)
        rho_gas = self.chamber_pressure * c.mol_wt / (R_UNIV * c.Tc)
        self.residence_time = (self.chamber_volume * rho_gas / self.mdot
                               if self.mdot > 0 else 0.0)
        self.comb_efficiency = self.cstar_eff ** 2  # η_c* ≈ (c*/c*_ideal)²-ish

    def _solve_masses(self, of: float):
        c = self.combo
        self.prop_mass = self.mdot * self.burn_time
        self.ox_mass = self.mdot_ox * self.burn_time
        self.fuel_mass = self.mdot_fuel * self.burn_time
        self.ox_volume = self.ox_mass / c.rho_ox if c.rho_ox > 0 else 0.0
        self.fuel_volume = self.fuel_mass / c.rho_fuel if c.rho_fuel > 0 else 0.0
        prop_vol = self.ox_volume + self.fuel_volume
        self.ullage_volume = prop_vol * self.ullage_frac

        # Cylindrical tank sizing. Diameter: user value, else slender 6:1.
        total_vol = prop_vol * (1.0 + self.ullage_frac)
        if self.tank_diameter > 0:
            self.tank_dia = self.tank_diameter
        else:
            # pick D so a single combined tank has L/D ≈ 5
            self.tank_dia = (4.0 * total_vol / (math.pi * 5.0)) ** (1.0 / 3.0)
        area = math.pi * self.tank_dia ** 2 / 4.0
        ull = 1.0 + self.ullage_frac
        self.ox_tank_len = (self.ox_volume * ull) / area if area > 0 else 0.0
        self.fuel_tank_len = (self.fuel_volume * ull) / area if area > 0 else 0.0

        # Dry-mass build-up. Pressure-fed tanks are heavier (thicker walls).
        cycle_dry = {"Pressure-fed": 1.0, "Electric Pump-fed": 1.15}.get(self.cycle, 0.85)
        self.dry_mass = self.struct_frac * self.prop_mass * cycle_dry
        # Split the dry mass into engine hardware vs tank/structure (rough 35/65).
        self.engine_mass = 0.35 * self.dry_mass
        self.tank_mass = 0.65 * self.dry_mass
        self.wet_mass = self.prop_mass + self.dry_mass
        self.total_impulse = self.target_thrust * self.burn_time

    def _solve_injector(self, of: float):
        c = self.combo
        self.inj_dp = self.injector_dp_frac * self.chamber_pressure
        cd = self.injector_cd
        rho_bulk = self._bulk_density(of)
        # Injection velocity from orifice equation v = Cd·√(2ΔP/ρ).
        self.inj_velocity = cd * math.sqrt(2.0 * self.inj_dp / max(rho_bulk, 1.0))
        # Total effective injector orifice area from the manifold mass flows.
        a_ox = self.mdot_ox / (cd * c.rho_ox * math.sqrt(2.0 * self.inj_dp / c.rho_ox)) if c.rho_ox > 0 and not c.monoprop else 0.0
        a_fu = self.mdot_fuel / (cd * c.rho_fuel * math.sqrt(2.0 * self.inj_dp / c.rho_fuel)) if c.rho_fuel > 0 else 0.0
        self.inj_area = a_ox + a_fu
        hole_area = math.pi * self.injector_hole_d ** 2 / 4.0
        self.inj_n_holes = max(1, int(round(self.inj_area / hole_area))) if hole_area > 0 else 0
        if c.monoprop:
            self.inj_pattern = "Showerhead (catalyst bed feed)"
        elif self.inj_n_holes < 12:
            self.inj_pattern = "Unlike-impinging doublet"
        else:
            self.inj_pattern = "Unlike-impinging (triplet / coaxial)"

    def _solve_cooling(self):
        c = self.combo
        # Simplified Bartz throat heat flux. Frozen gas property guesses.
        mu = 1.0e-4          # gas viscosity (Pa·s)
        cp = R_UNIV / c.mol_wt * c.gamma / (c.gamma - 1.0)  # J/kg·K
        pr = 0.5
        dt = max(self.throat_diameter, 1e-3)
        rc_throat = dt  # throat curvature ≈ throat dia
        hg = (0.026 / dt ** 0.2) * (mu ** 0.2 * cp / pr ** 0.6) \
            * (self.chamber_pressure / max(c.c_star, 1.0)) ** 0.8 \
            * (dt / rc_throat) ** 0.1
        t_aw = 0.9 * c.Tc            # adiabatic wall temp (recovery factor ~0.9)
        self.heat_flux = max(hg * (t_aw - self.wall_temp), 0.0)   # W/m²

        # Surface area to cool ≈ chamber wall + convergent + a bit of nozzle.
        a_surf = math.pi * self.chamber_diameter * self.chamber_length \
            + math.pi * 0.5 * (self.chamber_diameter + self.throat_diameter) \
            * self.nozzle_length * 0.5
        q_total = self.heat_flux * max(a_surf, 1e-4)

        method = self.cooling
        if method == "Regenerative":
            # The fuel itself is the coolant, so flow is fixed at the fuel
            # mass flow; effectiveness = how much of the heat load its allowable
            # temperature rise (~250 K) can absorb.
            cp_cool = 2000.0
            self.coolant_flow = self.mdot_fuel if self.mdot_fuel > 0 else self.mdot
            heat_sink = self.coolant_flow * cp_cool * 250.0
            self.cooling_dp = 0.15 * self.chamber_pressure   # regen channel loss
            self.cooling_eff = min(0.98, heat_sink / q_total) if q_total > 0 else 0.98
        elif method == "Film Cooling":
            self.coolant_flow = 0.05 * self.mdot             # ~5% film flow
            self.cooling_dp = 0.05 * self.chamber_pressure
            self.cooling_eff = 0.80
        elif method == "Ablative":
            self.coolant_flow = 0.0                          # mass loss, no flow
            self.cooling_dp = 0.0
            self.cooling_eff = 0.70
        else:  # Radiative
            self.coolant_flow = 0.0
            self.cooling_dp = 0.0
            self.cooling_eff = 0.50

    def _solve_dv_twr(self):
        # Engine-level mass ratio / Δv (Tsiolkovsky). Without a vehicle this is
        # the engine+tank stage on its own; with rocket_total_mass it's the
        # whole vehicle.
        m_dry = self.dry_mass
        m_wet = self.wet_mass
        self.mass_ratio = m_wet / m_dry if m_dry > 0 else 0.0
        self.delta_v = self.isp_vac * G0 * math.log(self.mass_ratio) if self.mass_ratio > 1 else 0.0

        self.twr_engine = self.target_thrust / (self.engine_mass * G0) if self.engine_mass > 0 else 0.0
        if self.rocket_total_mass > 0:
            self.twr_liftoff = self.target_thrust / (self.rocket_total_mass * G0)
            burnout = max(self.rocket_total_mass - self.prop_mass, 1e-6)
            self.twr_burnout = self.target_thrust / (burnout * G0)

    def _validate(self):
        w = []
        c = self.combo
        pc_bar = self.chamber_pressure / 1e5
        cyc = ENGINE_CYCLES.get(self.cycle, {})
        # Expansion vs ambient (flow separation / overexpansion).
        if self.ambient_pressure > 0 and self.exit_pressure < 0.35 * self.ambient_pressure:
            w.append(f"Overexpanded: Pe={self.exit_pressure/1e5:.2f} bar ≪ ambient "
                     f"{self.ambient_pressure/1e5:.2f} bar → flow separation likely at sea level. Reduce ε.")
        if self.ambient_pressure > 0 and self.exit_pressure > 4.0 * self.ambient_pressure:
            w.append("Underexpanded for this ambient — a larger ε would recover thrust.")
        if pc_bar < 5.0:
            w.append(f"Chamber pressure {pc_bar:.1f} bar is very low — poor c* efficiency and combustion stability.")
        if pc_bar > 250.0:
            w.append(f"Chamber pressure {pc_bar:.0f} bar is extreme — staged-combustion territory, hard to cool.")
        if cyc and pc_bar > cyc.get("pc_max", 1e9):
            w.append(f"{self.cycle} cycle is impractical above ~{cyc['pc_max']:.0f} bar (current {pc_bar:.0f} bar).")
        if not c.monoprop and (c.of_min > 0):
            of = self._of()
            if of < c.of_min or of > c.of_max:
                w.append(f"O/F {of:.2f} outside recommended {c.of_min:.1f}–{c.of_max:.1f} for {c.name}.")
        if self.struct_frac < 0.08:
            w.append(f"Structural mass fraction {self.struct_frac:.2f} is optimistic — real stages rarely below ~0.08.")
        if self.struct_frac > 1.0:
            w.append(f"Structural mass fraction {self.struct_frac:.2f} > 1 — dry mass exceeds propellant (very heavy).")
        if self.efficiency > 1.0:
            w.append("Delivered Isp exceeds ideal — check efficiency factors (>1 is unphysical).")
        if self.cstar_eff > 1.0 or self.cf_eff > 1.0:
            w.append("Efficiency factor > 1.0 is unphysical.")
        if self.twr_liftoff and self.twr_liftoff < 1.2:
            w.append(f"Liftoff TWR {self.twr_liftoff:.2f} < 1.2 — vehicle may not clear the pad cleanly.")
        if self.cooling == "Regenerative" and self.cooling_eff < 0.7:
            w.append(f"Regenerative cooling marginal (effectiveness {self.cooling_eff*100:.0f}%) — "
                     "fuel heat sink can't fully absorb the throat heat load; add film cooling or lower Pc.")
        if self.residence_time and self.residence_time < 0.5e-3:
            w.append(f"Residence time {self.residence_time*1e3:.2f} ms is short — incomplete combustion risk (raise L*).")
        if not cyc.get("supported", True):
            w.append(f"{self.cycle}: conceptual estimate only — full {self.cycle} modelling not yet implemented.")
        self.warnings = w

    # ── optimisation helpers ──────────────────────────────────────────────
    def optimum_of(self, mode: str) -> float:
        """Sweep O/F and return the ratio that maximises the chosen objective."""
        c = self.combo
        if c.monoprop or mode in ("Off", ""):
            return self._of()
        lo = c.of_min if c.of_min > 0 else 0.5
        hi = c.of_max if c.of_max > 0 else max(c.of_ratio * 1.6, lo + 2.0)
        lo, hi = max(0.3, lo * 0.7), hi * 1.3
        best_of, best_val = self._of(), -1e30
        n = 60
        for i in range(n + 1):
            of = lo + (hi - lo) * i / n
            # Isp tracks how close O/F is to the combo optimum (parabolic falloff).
            penalty = 1.0 - 0.04 * ((of - c.of_ratio) / max(c.of_ratio, 1e-3)) ** 2
            isp = self.isp_ideal * max(penalty, 0.3)
            rho = self._bulk_density(of)
            if mode == "Maximise Isp":
                val = isp
            elif mode == "Maximise density impulse":
                val = isp * rho
            elif mode == "Maximise thrust":
                val = isp * self.mdot  # mdot ~ const here, so ≈ Isp
            else:  # Balanced optimum — Isp · ρ^0.3
                val = isp * rho ** 0.3
            if val > best_val:
                best_val, best_of = val, of
        return best_of

    def optimum_expansion(self) -> float:
        """ε for a perfectly-expanded nozzle at the design ambient (vacuum→cap)."""
        pa = self.ambient_pressure
        if pa <= 0:
            return 80.0   # vacuum: cap at a buildable bell ratio
        return max(1.5, min(200.0, expansion_ratio_for_exit_pressure(
            pa, self.chamber_pressure, self.combo.gamma)))

    def sensitivity(self, span: float = 0.20, n: int = 9) -> dict:
        """Vary Pc, O/F and ε by ±span and record Isp / thrust / mdot / prop.

        Returns {param: {"x":[...], "isp":[...], "thrust":[...], "mdot":[...],
        "prop":[...]}} with x as the fractional perturbation."""
        base = dict(chamber_pressure=self.chamber_pressure,
                    of_ratio=self._of(), expansion_ratio=self.expansion_ratio)
        out = {}
        for param in ("chamber_pressure", "of_ratio", "expansion_ratio"):
            xs, isp, thr, md, prop = [], [], [], [], []
            for i in range(n):
                f = 1.0 + span * (2.0 * i / (n - 1) - 1.0)
                d = LiquidEngineDesign(
                    combo=self.combo, chamber_pressure=base["chamber_pressure"],
                    expansion_ratio=base["expansion_ratio"],
                    target_thrust=self.target_thrust, burn_time=self.burn_time,
                    of_ratio=base["of_ratio"], ambient_pressure=self.ambient_pressure,
                    cstar_eff=self.cstar_eff, cf_eff=self.cf_eff,
                    struct_frac=self.struct_frac, cycle=self.cycle)
                setattr(d, param, base[param] * f)
                if param == "of_ratio":
                    d.of_ratio = base["of_ratio"] * f
                d.solve()
                xs.append((f - 1.0) * 100.0)
                isp.append(d.isp); thr.append(d.target_thrust)
                md.append(d.mdot); prop.append(d.prop_mass)
            out[param] = {"x": xs, "isp": isp, "thrust": thr, "mdot": md, "prop": prop}
        return out

    # ── thrust curve ──────────────────────────────────────────────────────
    def thrust_curve(self, dt: float = 0.02) -> list[tuple[float, float]]:
        """Time–thrust samples for the selected profile."""
        F = self.target_thrust
        bt = self.burn_time
        prof = self.thrust_profile

        if prof == "User-defined CSV" and self.csv_curve:
            return [(float(t), float(f)) for t, f in self.csv_curve]

        out, t = [], 0.0
        while t <= bt + 1e-9:
            out.append((round(t, 6), self._profile_value(t, F, bt, prof)))
            t += dt
        if out[-1][0] < bt:
            out.append((bt, 0.0))
        return out

    def _profile_value(self, t, F, bt, prof):
        t_up, t_dn = self.startup, self.shutdown
        # Common startup / shutdown ramp envelope.
        if t < t_up and t_up > 0:
            env = t / t_up
        elif t > bt - t_dn and t_dn > 0:
            env = max(0.0, (bt - t) / t_dn)
        else:
            env = 1.0

        if prof == "Linear Ramp":
            shape = 0.7 + 0.6 * (t / bt) if bt > 0 else 1.0      # 70%→130%
        elif prof == "Bell-shaped":
            x = (t / bt - 0.5) if bt > 0 else 0.0
            shape = 0.6 + 0.8 * math.exp(-(x * 3.0) ** 2)         # peak mid-burn
        elif prof == "Throttle schedule":
            shape = self._interp(self.throttle_points, t / bt if bt > 0 else 0.0) \
                if self.throttle_points else 1.0
        else:  # Constant
            shape = 1.0
        return F * shape * env

    @staticmethod
    def _interp(pts, t):
        if not pts:
            return 1.0
        for i in range(len(pts) - 1):
            t0, f0 = pts[i]
            t1, f1 = pts[i + 1]
            if t0 <= t <= t1:
                if t1 - t0 < 1e-12:
                    return f1
                return f0 + (f1 - f0) * (t - t0) / (t1 - t0)
        return pts[-1][1]

    # ── result dict (solid-motor-compatible) ──────────────────────────────
    def simulate(self, dt: float = 0.02) -> dict:
        self.solve()
        curve = self.thrust_curve(dt)
        times = [t for t, _ in curve]
        thrusts = [f for _, f in curve]
        impulse = sum(0.5 * (thrusts[i] + thrusts[i + 1]) * (times[i + 1] - times[i])
                      for i in range(len(times) - 1))
        # Engine hardware length = chamber + nozzle (tanks are airframe mass).
        length = max(0.15, self.chamber_length + self.nozzle_length)
        return {
            "time": times,
            "thrust": thrusts,
            "pressure": [self.chamber_pressure] * len(times),
            "prop_mass": self.prop_mass,
            "case_mass": self.dry_mass,
            "length": length,
            "motor_name": f"Liquid {self.combo.name}",
            "metrics": self.metrics(impulse, length),
        }

    def metrics(self, impulse: float | None = None, length: float | None = None) -> dict:
        if impulse is None:
            impulse = self.total_impulse
        if length is None:
            length = max(0.15, self.chamber_length + self.nozzle_length)
        return {
            "isp_sl": self.isp, "isp_vac": self.isp_vac, "isp_ideal": self.isp_ideal,
            "cf": self.cf, "cf_ideal": self.cf_ideal, "efficiency": self.efficiency,
            "thrust_ideal": self.thrust_ideal,
            "throat_diameter": self.throat_diameter, "exit_diameter": self.exit_diameter,
            "chamber_diameter": self.chamber_diameter, "chamber_length": self.chamber_length,
            "nozzle_length": self.nozzle_length, "l_star": self.l_star_eff,
            "contraction_ratio": self.contraction_ratio, "expansion_ratio": self.expansion_ratio,
            "residence_time": self.residence_time, "comb_efficiency": self.comb_efficiency,
            "mdot": self.mdot, "mdot_ox": self.mdot_ox, "mdot_fuel": self.mdot_fuel,
            "ox_mass": self.ox_mass, "fuel_mass": self.fuel_mass,
            "ox_volume": self.ox_volume, "fuel_volume": self.fuel_volume,
            "ullage_volume": self.ullage_volume, "tank_dia": self.tank_dia,
            "ox_tank_len": self.ox_tank_len, "fuel_tank_len": self.fuel_tank_len,
            "tank_mass": self.tank_mass, "engine_mass": self.engine_mass,
            "wet_mass": self.wet_mass, "dry_mass": self.dry_mass,
            "exit_pressure": self.exit_pressure, "chamber_pressure": self.chamber_pressure,
            "total_impulse": impulse, "prop_len": length,
            "inj_dp": self.inj_dp, "inj_area": self.inj_area, "inj_velocity": self.inj_velocity,
            "inj_n_holes": self.inj_n_holes, "inj_pattern": self.inj_pattern,
            "heat_flux": self.heat_flux, "coolant_flow": self.coolant_flow,
            "cooling_dp": self.cooling_dp, "cooling_eff": self.cooling_eff,
            "mass_ratio": self.mass_ratio, "delta_v": self.delta_v,
            "twr_engine": self.twr_engine, "twr_liftoff": self.twr_liftoff,
            "twr_burnout": self.twr_burnout,
        }
