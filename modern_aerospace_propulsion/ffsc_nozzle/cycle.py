"""Full-flow staged combustion (FFSC) cycle model with regen-cooled nozzle.

Note: Requires Cantera and CoolProp to be installed.

Module Summary:
- Functions:
    - ``saturated_liquid_density(fluid_name, T)``: Saturated liquid density (Q=0) via CoolProp.
    - ``pump_power(m_dot, rho, dp, eta_pump)``: Incompressible pump shaft power.
    - ``turbine_power(m_dot, cp_g, T_in, T_out, eta_turb)``: Turbine power from enthalpy drop.
    - ``solve_preburner_T_for_power(...)``: Bisection solver for preburner inlet temperature to meet power.
    - ``ffsc_full_flow_cycle(...)``: Integrated FFSC engine point analysis (chamber, cooling, preburners, pumps/turbines).
    - ``optimize_ffsc_design(...)``: Optimize FFSC cycle parameters to maximize Isp with constraints.
"""

import math
from scipy.optimize import minimize, NonlinearConstraint

from .regen import simple_conical_nozzle, regen_nozzle_1D, make_coolprop_liquid
from .thermo import cantera_chamber_state, cantera_preburner_state
from typing import Tuple, Dict, Optional
import numpy as np
from CoolProp.CoolProp import PropsSI


def saturated_liquid_density(fluid_name: str, T: float) -> float:
    """Compute saturated liquid density using CoolProp.
    
    Calculates the density of a saturated liquid (quality = 0) at the
    specified temperature using CoolProp thermophysical property database.
    
    Args:
        fluid_name: CoolProp fluid name (e.g., 'Methane', 'Oxygen', 'Hydrogen')
        T: Temperature (K)
        
    Returns:
        Saturated liquid density (kg/m³)
        
    Note:
        - Uses CoolProp's high-accuracy equation of state
        - Quality Q=0 ensures saturated liquid state (no vapor)
        - Valid between triple point and critical point temperatures
        
    Example:
        >>> rho_LCH4 = saturated_liquid_density('Methane', 110.0)  # ~422 kg/m³
        >>> rho_LOX = saturated_liquid_density('Oxygen', 90.0)    # ~1141 kg/m³
    """
    # Query CoolProp for saturated liquid density at specified temperature
    # Q=0 specifies saturated liquid (quality = 0, all liquid, no vapor)
    return PropsSI('D', 'T', T, 'Q', 0, fluid_name)


def pump_power(m_dot: float, rho: float, dp: float, eta_pump: float) -> float:
    """Compute pump shaft power required for incompressible liquid pumping.
    
    Uses the incompressible flow pump power equation accounting for
    pump isentropic efficiency. Power is the work rate needed to
    increase fluid pressure.
    
    Args:
        m_dot: Mass flow rate (kg/s)
        rho: Liquid density (kg/m³)
        dp: Pressure rise across pump (Pa)
        eta_pump: Pump isentropic efficiency (dimensionless, 0 < η < 1)
        
    Returns:
        Pump shaft power (W)
        
    Note:
        Formula: P = (m_dot * dp) / (rho * eta_pump)
        This assumes incompressible flow (valid for liquids).
    """
    # Ideal hydraulic power: P_ideal = m_dot * dp / rho = (volume flow) * dp
    # Actual shaft power accounts for pump inefficiency: P_shaft = P_ideal / eta
    return m_dot*dp/(rho*eta_pump)


def turbine_power(m_dot: float, cp_g: float, T_in: float, T_out: float, eta_turb: float) -> float:
    """Compute turbine shaft power output from expanding hot gas.
    
    Calculates mechanical power extracted by turbine as hot gas expands
    and cools. Assumes constant specific heat and accounts for turbine
    isentropic efficiency.
    
    Args:
        m_dot: Mass flow rate through turbine (kg/s)
        cp_g: Specific heat at constant pressure (J/kg/K)
        T_in: Turbine inlet temperature (K)
        T_out: Turbine exit temperature (K)
        eta_turb: Turbine isentropic efficiency (dimensionless, 0 < η < 1)
        
    Returns:
        Turbine shaft power output (W)
        Returns 0.0 if T_in <= T_out (no expansion possible)
        
    Note:
        Formula: P = m_dot * cp * (T_in - T_out) * eta_turb
        Power is positive for T_in > T_out (turbine produces work).
    """
    # Check for valid expansion (inlet hotter than exit)
    if T_in <= T_out:
        return 0.0
    
    # Turbine power from enthalpy drop: P = m_dot * cp * ΔT * eta_turb
    # Efficiency accounts for irreversibilities in expansion process
    return m_dot*cp_g*(T_in - T_out)*eta_turb


def solve_preburner_T_for_power(
        P_req: float,
        m_dot: float,
        cp_g: float,
        T_out: float,
        T_min: float,
        T_max: float,
        eta_turb: float = 0.9,
        tol: float = 1e-3,
        max_iter: int = 60
) -> Tuple[float, float]:
    """Solve for preburner temperature that delivers required turbine power.
    
    Uses bisection method to find the turbine inlet temperature (preburner
    exit temperature) that produces exactly the required shaft power for
    driving the turbopump. This is used for cycle balance in staged
    combustion engines.
    
    Args:
        P_req: Required turbine power output (W)
        m_dot: Mass flow rate through turbine (kg/s)
        cp_g: Gas specific heat at constant pressure (J/kg/K)
        T_out: Turbine exit temperature (K)
        T_min: Minimum search bound for inlet temperature (K)
        T_max: Maximum search bound for inlet temperature (K)
        eta_turb: Turbine isentropic efficiency (dimensionless), default 0.9
        tol: Relative convergence tolerance (dimensionless), default 1e-3
        max_iter: Maximum number of bisection iterations, default 60
        
    Returns:
        Tuple of (T_inlet, P_actual):
            - T_inlet: Solved turbine inlet temperature (K)
            - P_actual: Actual power at solved temperature (W)
            
    Note:
        - Uses bisection method: robust but requires bracketing solution
        - Assumes turbine_power is monotonic in T_in
        - If not converged after max_iter, returns last midpoint estimate
        - For FFSC cycles, this determines preburner operating temperature
    """
    # ==================== Bisection Method Setup ====================
    # Initialize lower and upper bounds for inlet temperature
    lo, hi = T_min, T_max
    
    # ==================== Bisection Iteration ====================
    for _ in range(max_iter):
        # Evaluate midpoint of current bracket
        mid = 0.5*(lo + hi)
        
        # Compute turbine power at midpoint temperature
        P_mid = turbine_power(m_dot, cp_g, mid, T_out, eta_turb)
        
        # Check convergence: relative error within tolerance
        if abs(P_mid - P_req)/max(P_req, 1.0) < tol:
            return mid, P_mid
        
        # Update bracket based on whether power is too low or too high
        if P_mid < P_req:
            # Need higher temperature to get more power
            lo = mid
        else:
            # Need lower temperature, already have too much power
            hi = mid
    
    # ==================== Return Last Estimate if Not Converged ====================
    # If max iterations reached, return final midpoint estimate
    mid = 0.5*(lo + hi)
    return mid, turbine_power(m_dot, cp_g, mid, T_out, eta_turb)


def ffsc_full_flow_cycle(
        F_vac: float,
        p0: float,
        OF: float,
        r_t: float,
        eps: float,
        L_noz: float,
        fuel_name: str = "CH4",
        ox_name: str = "O2",
        fuel_coolprop: str = "Methane",
        ox_coolprop: str = "Oxygen",
        T_fuel_tank: float = 110.0,
        T_ox_tank: float = 90.0,
        n_channels: int = 120,
        w_channel: float = 0.0015,
        wall_thickness: float = 0.0020,
        roughness: float = 5e-6,
        p_cool_in: float = 18e6,
        T_wall_limit: float = 900.0,
        dp_limit: float = 5e6,
        injector_dp_frac: float = 0.2,
        feed_line_dp: float = 1.0e6,
        eta_pump_fuel: float = 0.70,
        eta_pump_ox: float = 0.70,
        eta_turb_fuel: float = 0.90,
        eta_turb_ox: float = 0.90,
        eta_mech: float = 0.98,
        T_turb_out: Optional[float] = None,
        coolant_fraction_override: Optional[float] = None,
        g0: float = 9.80665,
) -> Dict[str, any]:
    """Evaluate a full-flow staged combustion (FFSC) rocket engine design point.
    
    Performs integrated analysis of an FFSC methane-oxygen engine including:
    - Main chamber equilibrium thermodynamics (Cantera)
    - Regenerative nozzle cooling with CoolProp properties
    - Fuel-rich and oxygen-rich preburner states
    - Turbopump power balance and cycle feasibility
    
    FFSC Architecture:
    In FFSC, ALL propellant flows through preburners before entering the main
    chamber. The fuel-rich preburner drives the fuel turbopump, and the
    oxygen-rich preburner drives the oxidizer turbopump. Hot turbine exhaust
    gases from both sides are mixed in the main combustion chamber.
    
    Args:
        F_vac: Target vacuum thrust (N)
        p0: Main chamber pressure (Pa)
        OF: Main chamber oxidizer-to-fuel mass ratio (dimensionless)
        r_t: Nozzle throat radius (m)
        eps: Nozzle expansion ratio A_exit/A_throat (dimensionless)
        L_noz: Nozzle length from throat to exit (m)
        fuel_name: Cantera species name for fuel (e.g., "CH4", "H2")
        ox_name: Cantera species name for oxidizer (e.g., "O2")
        fuel_coolprop: CoolProp fluid name for fuel (e.g., "Methane", "Hydrogen")
        ox_coolprop: CoolProp fluid name for oxidizer (e.g., "Oxygen")
        T_fuel_tank: Fuel tank temperature (K), used for saturated liquid density
        T_ox_tank: Oxidizer tank temperature (K), used for saturated liquid density
        n_channels: Number of parallel regenerative cooling channels (dimensionless)
        w_channel: Cooling channel width (m)
        wall_thickness: Nozzle wall thickness (m)
        roughness: Channel surface roughness (m)
        p_cool_in: Coolant inlet pressure (Pa)
        T_wall_limit: Maximum allowable wall temperature (K) - not enforced, for reference
        dp_limit: Maximum allowable pressure drop (Pa) - not enforced, for reference
        injector_dp_frac: Injector pressure drop as fraction of p0 (dimensionless)
        feed_line_dp: Feed line pressure drop (Pa)
        eta_pump_fuel: Fuel pump isentropic efficiency (dimensionless, 0 < η < 1)
        eta_pump_ox: Oxidizer pump isentropic efficiency (dimensionless, 0 < η < 1)
        eta_turb_fuel: Fuel turbine isentropic efficiency (dimensionless, 0 < η < 1)
        eta_turb_ox: Oxidizer turbine isentropic efficiency (dimensionless, 0 < η < 1)
        eta_mech: Mechanical transmission efficiency (dimensionless, 0 < η < 1)
        T_turb_out: Turbine exit temperature (K). If None, defaults to T0
        coolant_fraction_override: If specified, use this coolant fraction instead of default 0.2
        g0: Standard gravity for Isp calculation (m/s²), default 9.80665
        
    Returns:
        Dictionary containing complete cycle analysis results:
            Performance:
                - F_vac: Vacuum thrust (N)
                - Isp_vac_ideal: Ideal vacuum specific impulse (s)
                - Isp_vac_eff: Effective vacuum specific impulse with 95% efficiency (s)
                - m_dot_total: Total propellant mass flow (kg/s)
                - m_dot_fuel: Fuel mass flow (kg/s)
                - m_dot_ox: Oxidizer mass flow (kg/s)
            
            Chamber State:
                - T0: Chamber stagnation temperature (K)
                - gamma: Chamber gas ratio of specific heats (dimensionless)
                - R_g: Chamber gas constant (J/kg/K)
                - gas_chamber: Cantera gas object at chamber state
            
            Cooling:
                - m_dot_cool: Coolant mass flow (kg/s)
                - coolant_fraction: Coolant flow as fraction of fuel flow (dimensionless)
                - T_wall_max: Maximum inner wall temperature (K)
                - dp_cool: Coolant pressure drop through regen channels (Pa)
                - nozzle_results: Full regen_nozzle_1D output dictionary
            
            Turbopump System:
                - dp_fuel: Total fuel pump pressure rise (Pa)
                - dp_ox: Total oxidizer pump pressure rise (Pa)
                - P_pump_fuel: Fuel pump shaft power (W)
                - P_pump_ox: Oxidizer pump shaft power (W)
                - P_turb_req_fuel: Required fuel turbine power including mech losses (W)
                - P_turb_req_ox: Required ox turbine power including mech losses (W)
            
            Preburners:
                - T_pb_fuel: Fuel-rich preburner temperature (K)
                - T_pb_ox: Oxygen-rich preburner temperature (K)
                - P_turb_avail_fuel: Available fuel turbine power (W)
                - P_turb_avail_ox: Available oxidizer turbine power (W)
            
            Feasibility:
                - fuel_side_ok: True if fuel turbine power >= 95% of required (bool)
                - ox_side_ok: True if ox turbine power >= 95% of required (bool)
                
    Note:
        - Requires Cantera for equilibrium calculations
        - Requires CoolProp for liquid densities and coolant properties
        - Coolant is always the fuel (fuel flows through regenerative cooling channels)
        - Tank densities computed as saturated liquid (quality = 0) at tank temperatures
        - Uses fixed equivalence ratios: φ_fuel=1.3 (fuel-rich), φ_ox=0.7 (ox-rich)
        - Assumes 95% nozzle efficiency for Isp calculation
        - Assumes 20% of fuel flow as coolant (coolant_fraction_guess)
    """
    # ==================== Main Chamber Equilibrium State ====================
    # Compute equilibrium thermodynamics using Cantera
    # This determines chamber temperature, gamma, and gas constant for given O/F ratio
    T0, gamma, R_g, gas_ch = cantera_chamber_state(
        OF=OF,
        p0=p0,
        T_fuel=T_fuel_tank,  # Use fuel tank temperature for inlet state
        T_ox=T_ox_tank,      # Use oxidizer tank temperature for inlet state
        fuel_species=fuel_name,
        ox_species=ox_name,
        mech="gri30.yaml",
    )

    # ==================== Set Turbine Exit Temperature ====================
    # Default to chamber temperature if not specified
    # Typically set lower to create temperature drop for power extraction
    if T_turb_out is None:
        T_turb_out = T0

    # ==================== Generate Nozzle Geometry ====================
    # Create simple conical nozzle contour from throat to exit
    x_noz, r_noz, A_noz, At = simple_conical_nozzle(r_t, eps, L_noz, N=250)

    # ==================== Initialize Coolant Properties ====================
    # Create CoolProp property function for fuel coolant
    # Note: Coolant is always the fuel in regenerative cooling
    coolant_props = make_coolprop_liquid(fuel_coolprop)

    # ==================== Compute Ideal Performance ====================
    from .sweep import ideal_vacuum_isp_from_eps
    # Calculate theoretical specific impulse assuming isentropic expansion
    Isp_vac_ideal, _, T_e, M_e = ideal_vacuum_isp_from_eps(p0, T0, gamma, R_g, eps, g0)
    
    # Account for nozzle losses (boundary layer, divergence, etc.)
    eta_Isp = 0.95  # Typical nozzle efficiency
    Isp_vac_eff = eta_Isp*Isp_vac_ideal
    
    # ==================== Compute Required Mass Flows ====================
    # From thrust equation: F = Isp * m_dot * g0
    m_dot_total = F_vac/(Isp_vac_eff*g0)  # Total propellant flow
    
    # Split into fuel and oxidizer based on O/F ratio
    m_dot_fuel = m_dot_total/(1.0 + OF)  # Fuel mass flow
    m_dot_ox = m_dot_total - m_dot_fuel  # Oxidizer mass flow

    # ==================== Estimate Coolant Flow ====================
    # Assume fraction of fuel flow is diverted for regenerative cooling
    if coolant_fraction_override is not None:
        coolant_fraction_guess = coolant_fraction_override  # Use specified fraction
    else:
        coolant_fraction_guess = 0.2  # 20% of fuel as coolant (typical default)
    m_dot_cool = coolant_fraction_guess*m_dot_fuel

    # ==================== Regenerative Cooling Analysis ====================
    # Run 1D conjugate heat transfer model for nozzle cooling
    # Computes wall temperatures, coolant temperature rise, and pressure drop
    # Coolant enters at fuel tank temperature
    res_noz = regen_nozzle_1D(
        x=x_noz,
        r=r_noz,
        A=A_noz,
        At=At,
        p0=p0,
        T0=T0,
        gamma=gamma,
        R_g=R_g,
        coolant=fuel_coolprop,  # Use fuel as coolant
        coolant_props=coolant_props,
        m_dot_cool=m_dot_cool,
        n_channels=n_channels,
        w_channel=w_channel,
        h_channel=0.0020,  # Fixed channel height
        wall_thickness=wall_thickness,
        roughness=roughness,
        T_cool_in=T_fuel_tank,  # Coolant enters at fuel tank temperature
        p_cool_in=p_cool_in,
        L_regen=x_noz[-1],  # Cool entire nozzle
        T_env=3.0,  # Space environment temperature
    )

    # Extract key cooling results
    T_wall_max = res_noz["T_wall_inner"].max()  # Maximum inner wall temperature
    dp_cool = res_noz["p_cool"][0] - res_noz["p_cool"][-1]  # Coolant pressure drop

    # ==================== Compute Total Pressure Rises ====================
    # Injector pressure drop (fraction of chamber pressure)
    dp_injector = injector_dp_frac*p0
    
    # Fuel side: must overcome cooling circuit, injector, feed lines, and reach chamber
    dp_fuel = dp_cool + dp_injector + feed_line_dp + (p0 - 0.1e6)
    
    # Oxidizer side: must overcome injector, feed lines, and reach chamber
    # (no cooling circuit for oxidizer)
    dp_ox = dp_injector + feed_line_dp + (p0 - 0.1e6)

    # ==================== Compute Pump Power Requirements ====================
    # Get saturated liquid densities at tank temperatures using CoolProp
    # Assumes quality Q=0 (saturated liquid, no vapor)
    rho_fuel = saturated_liquid_density(fuel_coolprop, T_fuel_tank)  # Fuel density at tank temp
    rho_ox = saturated_liquid_density(ox_coolprop, T_ox_tank)  # Oxidizer density at tank temp

    # Calculate pump shaft power from incompressible flow equation
    P_pump_fuel = pump_power(m_dot_fuel, rho_fuel, dp_fuel, eta_pump_fuel)
    P_pump_ox = pump_power(m_dot_ox, rho_ox, dp_ox, eta_pump_ox)

    # ==================== Compute Required Turbine Power ====================
    # Account for mechanical transmission losses (gearbox, bearings, seals)
    P_turb_req_fuel = P_pump_fuel/eta_mech  # Required fuel turbine power
    P_turb_req_ox = P_pump_ox/eta_mech  # Required oxidizer turbine power

    # ==================== Preburner Equilibrium States ====================
    # Set equivalence ratios for each preburner
    # Fuel-rich: excess fuel keeps temperature manageable, provides reducing environment
    phi_fuel = 1.3  # Fuel-rich preburner (φ > 1)
    # Oxygen-rich: excess oxidizer keeps temperature manageable, provides oxidizing environment  
    phi_ox = 0.7    # Oxygen-rich preburner (φ < 1)
    
    # Compute fuel-rich preburner state
    # Hot fuel-rich gas drives fuel turbopump, then enters main chamber
    T_ad_fuel, cp_fuel_pb, gas_fb = cantera_preburner_state(
        fuel_name=fuel_name,
        ox_name=ox_name,
        mech="gri30.yaml",
        p_pb=p0,
        T_inlet=T_fuel_tank,  # Fuel preburner fed by fuel from tank (heated by regen cooling)
        phi=phi_fuel,
    )
    
    # Compute oxygen-rich preburner state
    # Hot oxygen-rich gas drives oxidizer turbopump, then enters main chamber
    T_ad_ox, cp_ox_pb, gas_ob = cantera_preburner_state(
        fuel_name=fuel_name,
        ox_name=ox_name,
        mech="gri30.yaml",
        p_pb=p0,
        T_inlet=T_ox_tank,  # Oxidizer preburner fed by oxidizer from tank
        phi=phi_ox,
    )
    
    # ==================== Compute Available Turbine Power ====================
    # Calculate power extracted from gas expansion through turbines
    # Power = m_dot * cp * (T_in - T_out) * eta_turb
    P_turb_avail_fuel = turbine_power(m_dot_fuel, cp_fuel_pb, T_ad_fuel, T_turb_out, eta_turb_fuel)
    P_turb_avail_ox = turbine_power(m_dot_ox, cp_ox_pb, T_ad_ox, T_turb_out, eta_turb_ox)
    
    # Store preburner temperatures for output
    T_pb_fuel = T_ad_fuel
    T_pb_ox = T_ad_ox

    # ==================== Check Power Balance (Feasibility) ====================
    # Each turbopump must be self-sustaining: turbine power >= pump power requirement
    # Use 95% threshold to allow small margin
    fuel_side_ok = P_turb_avail_fuel >= 0.95*P_turb_req_fuel
    ox_side_ok = P_turb_avail_ox >= 0.95*P_turb_req_ox

    return dict(
        F_vac=F_vac,
        Isp_vac_ideal=Isp_vac_ideal,
        Isp_vac_eff=Isp_vac_eff,
        m_dot_total=m_dot_total,
        m_dot_fuel=m_dot_fuel,
        m_dot_ox=m_dot_ox,
        T0=T0,
        gamma=gamma,
        R_g=R_g,
        gas_chamber=gas_ch,
        m_dot_cool=m_dot_cool,
        coolant_fraction=m_dot_cool/m_dot_fuel,
        T_wall_max=T_wall_max,
        dp_cool=dp_cool,
        nozzle_results=res_noz,
        dp_fuel=dp_fuel,
        dp_ox=dp_ox,
        P_pump_fuel=P_pump_fuel,
        P_pump_ox=P_pump_ox,
        P_turb_req_fuel=P_turb_req_fuel,
        P_turb_req_ox=P_turb_req_ox,
        T_pb_fuel=T_pb_fuel,
        T_pb_ox=T_pb_ox,
        P_turb_avail_fuel=P_turb_avail_fuel,
        P_turb_avail_ox=P_turb_avail_ox,
        fuel_side_ok=fuel_side_ok,
        ox_side_ok=ox_side_ok,
    )


def optimize_ffsc_design(
        F_vac: float,
        fuel_name: str = "CH4",
        ox_name: str = "O2",
        fuel_coolprop: str = "Methane",
        ox_coolprop: str = "Oxygen",
        T_fuel_tank: float = 110.0,
        T_ox_tank: float = 90.0,
        p_amb: float = 0.0,
        T_wall_max_limit: float = 900.0,
        eta_pump_fuel: float = 0.70,
        eta_pump_ox: float = 0.70,
        eta_turb_fuel: float = 0.90,
        eta_turb_ox: float = 0.90,
        eta_mech: float = 0.98,
        p0_bounds: Tuple[float, float] = (10e6, 35e6),
        OF_bounds: Tuple[float, float] = (2.0, 4.5),
        r_t_bounds: Tuple[float, float] = (0.05, 0.20),
        eps_bounds: Tuple[float, float] = (10.0, 30.0),
        L_noz_bounds: Tuple[float, float] = (0.5, 2.5),
        coolant_frac_bounds: Tuple[float, float] = (0.10, 0.60),
        initial_guess: Optional[Dict[str, float]] = None,
        optimizer_method: str = "SLSQP",
        max_iterations: int = 100,
        verbose: bool = True,
        g0: float = 9.80665,
) -> Dict[str, any]:
    """Optimize FFSC engine design parameters to maximize specific impulse.
    
    Uses constrained nonlinear optimization (scipy.optimize.minimize) to find
    the optimal combination of chamber pressure, O/F ratio, nozzle throat radius,
    expansion ratio, and nozzle length that maximizes Isp while satisfying:
    
    Constraints:
    1. Turbopump power balance: Each turbine must provide >= required pump power
    2. Wall temperature limit: Maximum wall temperature <= T_wall_max_limit
    3. Feasibility: Both fuel and oxidizer turbopump systems must be viable
    
    Optimization Variables (6 DOF):
    - p0: Chamber pressure (Pa)
    - OF: Oxidizer-to-fuel mass ratio (dimensionless)
    - r_t: Nozzle throat radius (m)
    - eps: Nozzle expansion ratio (dimensionless)
    - L_noz: Nozzle length (m)
    - coolant_frac: Coolant flow as fraction of fuel flow (dimensionless)
    
    The optimizer adjusts these parameters to maximize Isp while respecting all
    constraints. This allows finding optimal performance-feasibility trade-offs
    for a given propellant combination and operating environment.
    
    Args:
        F_vac: Target vacuum thrust (N) - fixed requirement
        fuel_name: Cantera species name for fuel (e.g., "CH4", "H2")
        ox_name: Cantera species name for oxidizer (e.g., "O2")
        fuel_coolprop: CoolProp fluid name for fuel (e.g., "Methane", "Hydrogen")
        ox_coolprop: CoolProp fluid name for oxidizer (e.g., "Oxygen")
        T_fuel_tank: Fuel tank temperature (K)
        T_ox_tank: Oxidizer tank temperature (K)
        p_amb: Ambient pressure for Isp calculation (Pa), 0 for vacuum
        T_wall_max_limit: Maximum allowable wall temperature (K), default 900 K
        eta_pump_fuel: Fuel pump efficiency (dimensionless, 0-1), default 0.70
        eta_pump_ox: Oxidizer pump efficiency (dimensionless, 0-1), default 0.70
        eta_turb_fuel: Fuel turbine efficiency (dimensionless, 0-1), default 0.90
        eta_turb_ox: Oxidizer turbine efficiency (dimensionless, 0-1), default 0.90
        eta_mech: Mechanical transmission efficiency (dimensionless, 0-1), default 0.98
        p0_bounds: Bounds for chamber pressure (Pa), default (10-35 MPa)
        OF_bounds: Bounds for O/F ratio (dimensionless), default (2.0-4.5)
        r_t_bounds: Bounds for throat radius (m), default (0.05-0.20 m)
        eps_bounds: Bounds for expansion ratio (dimensionless), default (10-30)
        L_noz_bounds: Bounds for nozzle length (m), default (0.5-2.5 m)
        coolant_frac_bounds: Bounds for coolant fraction (dimensionless), default (0.10-0.60)
        initial_guess: Optional dict with keys ['p0', 'OF', 'r_t', 'eps', 'L_noz', 'coolant_frac']
                      If None, uses midpoint of bounds
        optimizer_method: Scipy optimizer method, default "SLSQP" (Sequential Least Squares)
                         Other options: "trust-constr", "COBYLA"
        max_iterations: Maximum optimizer iterations, default 100
        verbose: Print optimization progress, default True
        g0: Standard gravity (m/s²), default 9.80665
        
    Returns:
        Dictionary containing optimization results:
            Optimal Parameters:
                - p0_opt: Optimal chamber pressure (Pa)
                - OF_opt: Optimal O/F ratio (dimensionless)
                - r_t_opt: Optimal throat radius (m)
                - eps_opt: Optimal expansion ratio (dimensionless)
                - L_noz_opt: Optimal nozzle length (m)
                - coolant_frac_opt: Optimal coolant fraction (dimensionless)
            
            Performance:
                - Isp_opt: Maximum achievable Isp (s)
                - F_vac: Vacuum thrust (N)
                - m_dot_total: Total propellant flow (kg/s)
            
            Optimization Results:
                - success: True if optimizer converged to solution
                - message: Optimizer termination message
                - n_iterations: Number of iterations performed
                - n_evaluations: Number of objective function evaluations
            
            Complete Cycle:
                - cycle_results: Full ffsc_full_flow_cycle output dict at optimum
                
    Note:
        - Uses SLSQP (Sequential Least Squares Programming) by default - good for
          smooth, differentiable objectives with equality/inequality constraints
        - Alternative: "trust-constr" for tighter constraint handling
        - Optimization typically takes 30-80 function evaluations
        - Each evaluation runs complete cycle analysis (~0.5-2 seconds)
        - Total optimization time: 30-150 seconds depending on problem complexity
        
    Example:
        >>> # Optimize CH4/O2 engine for 500 kN vacuum thrust
        >>> result = optimize_ffsc_design(
        ...     F_vac=500e3,
        ...     fuel_name="CH4", ox_name="O2",
        ...     fuel_coolprop="Methane", ox_coolprop="Oxygen",
        ...     T_fuel_tank=110.0, T_ox_tank=90.0,
        ...     p_amb=0.0,  # Vacuum operation
        ...     T_wall_max_limit=900.0,
        ...     verbose=True
        ... )
        >>> print(f"Optimal Isp: {result['Isp_opt']:.1f} s")
        >>> print(f"Optimal p0: {result['p0_opt']/1e6:.1f} MPa")
        >>> print(f"Optimal O/F: {result['OF_opt']:.2f}")
        
        >>> # Optimize H2/O2 for sea-level operation
        >>> result_sl = optimize_ffsc_design(
        ...     F_vac=1000e3,
        ...     fuel_name="H2", ox_name="O2",
        ...     fuel_coolprop="Hydrogen", ox_coolprop="Oxygen",
        ...     T_fuel_tank=20.0, T_ox_tank=90.0,
        ...     p_amb=101325.0,  # Sea level
        ...     OF_bounds=(4.0, 8.0),  # H2/O2 typical range
        ...     verbose=True
        ... )
    """
    # ==================== Setup Initial Guess ====================
    if initial_guess is None:
        # Use midpoint of bounds as initial guess
        x0 = np.array([
            0.5*(p0_bounds[0] + p0_bounds[1]),  # Chamber pressure
            0.5*(OF_bounds[0] + OF_bounds[1]),  # O/F ratio
            0.5*(r_t_bounds[0] + r_t_bounds[1]),  # Throat radius
            0.5*(eps_bounds[0] + eps_bounds[1]),  # Expansion ratio
            0.5*(L_noz_bounds[0] + L_noz_bounds[1]),  # Nozzle length
            0.5*(coolant_frac_bounds[0] + coolant_frac_bounds[1]),  # Coolant fraction
        ])
    else:
        # Use user-provided initial guess
        x0 = np.array([
            initial_guess.get('p0', 0.5*(p0_bounds[0] + p0_bounds[1])),
            initial_guess.get('OF', 0.5*(OF_bounds[0] + OF_bounds[1])),
            initial_guess.get('r_t', 0.5*(r_t_bounds[0] + r_t_bounds[1])),
            initial_guess.get('eps', 0.5*(eps_bounds[0] + eps_bounds[1])),
            initial_guess.get('L_noz', 0.5*(L_noz_bounds[0] + L_noz_bounds[1])),
            initial_guess.get('coolant_frac', 0.5*(coolant_frac_bounds[0] + coolant_frac_bounds[1])),
        ])
    
    # ==================== Setup Optimization Bounds ====================
    from scipy.optimize import Bounds
    bounds = Bounds(
        lb=[p0_bounds[0], OF_bounds[0], r_t_bounds[0], eps_bounds[0], L_noz_bounds[0], coolant_frac_bounds[0]],
        ub=[p0_bounds[1], OF_bounds[1], r_t_bounds[1], eps_bounds[1], L_noz_bounds[1], coolant_frac_bounds[1]],
    )
    
    # ==================== Tracking Variables ====================
    # Track number of function evaluations and best result so far
    eval_count = {'n': 0}
    best_result = {'Isp': -np.inf, 'x': None, 'cycle': None}
    error_log = {'exceptions': [], 'error_params': []}
    
    # ==================== Define Objective Function ====================
    def objective(x):
        """Objective function to minimize (negative Isp for maximization)."""
        eval_count['n'] += 1
        p0, OF, r_t, eps, L_noz, coolant_frac = x
        
        try:
            # Run complete FFSC cycle analysis with specified coolant fraction
            cycle_result = ffsc_full_flow_cycle(
                F_vac=F_vac,
                p0=p0,
                OF=OF,
                r_t=r_t,
                eps=eps,
                L_noz=L_noz,
                fuel_name=fuel_name,
                ox_name=ox_name,
                fuel_coolprop=fuel_coolprop,
                ox_coolprop=ox_coolprop,
                T_fuel_tank=T_fuel_tank,
                T_ox_tank=T_ox_tank,
                eta_pump_fuel=eta_pump_fuel,
                eta_pump_ox=eta_pump_ox,
                eta_turb_fuel=eta_turb_fuel,
                eta_turb_ox=eta_turb_ox,
                eta_mech=eta_mech,
                coolant_fraction_override=coolant_frac,
                g0=g0,
            )
            
            # Extract Isp (use effective Isp accounting for nozzle efficiency)
            if p_amb > 0:
                # For atmospheric operation, compute Isp with pressure thrust
                from .sweep import ideal_atmospheric_isp_from_eps
                Isp_atm, _, _, _ = ideal_atmospheric_isp_from_eps(
                    p0=cycle_result['T0'],
                    T0=cycle_result['T0'],
                    gamma=cycle_result['gamma'],
                    R_g=cycle_result['R_g'],
                    eps=eps,
                    p_amb=p_amb,
                    g0=g0,
                )
                # Apply nozzle efficiency
                Isp = 0.95 * Isp_atm
            else:
                # Vacuum operation
                Isp = cycle_result['Isp_vac_eff']
            
            # Check for invalid Isp
            if not np.isfinite(Isp) or Isp < 0:
                if verbose:
                    print(f"  Eval {eval_count['n']:3d}: ⚠ Invalid Isp={Isp} from cycle - "
                          f"Isp_vac_eff={cycle_result.get('Isp_vac_eff', 'N/A')}")
                return 1e6
            
            # Track best result for reporting
            if Isp > best_result['Isp']:
                best_result['Isp'] = Isp
                best_result['x'] = x.copy()
                best_result['cycle'] = cycle_result
            
            # Print progress if verbose
            if verbose and eval_count['n'] % 1 == 0:
                print(f"  Eval {eval_count['n']:3d}: p0={p0/1e6:.1f} MPa, OF={OF:.2f}, "
                      f"r_t={r_t*1000:.1f} mm, eps={eps:.1f}, L={L_noz:.2f} m, "
                      f"cool={coolant_frac*100:.1f}% → Isp={Isp:.2f} s")
            
            # Return negative Isp (optimizer minimizes)
            return -Isp
            
        except Exception as e:
            # Log detailed exception information (limit to first 10 unique errors)
            error_key = f"{type(e).__name__}: {str(e)[:80]}"
            if len(error_log['exceptions']) < 10 and error_key not in error_log['exceptions']:
                error_log['exceptions'].append(error_key)
                error_log['error_params'].append({
                    'eval': eval_count['n'],
                    'p0_MPa': p0/1e6,
                    'OF': OF,
                    'r_t_mm': r_t*1000,
                    'eps': eps,
                    'L_noz': L_noz,
                    'coolant_frac': coolant_frac,
                })
            
            # If cycle analysis fails, return large penalty
            if verbose:
                print(f"  Eval {eval_count['n']:3d}: ✗ FAILED - {type(e).__name__}: {str(e)[:60]}")
            return 1e6
    
    # ==================== Define Constraint Functions ====================
    def constraint_turbopump_balance(x):
        """Constraint: Both turbopumps must have adequate power.
        
        Returns array of constraint values that should be >= 0:
        - Fuel turbine power margin (should be >= 0)
        - Ox turbine power margin (should be >= 0)
        """
        p0, OF, r_t, eps, L_noz, coolant_frac = x
        try:
            cycle_result = ffsc_full_flow_cycle(
                F_vac=F_vac, p0=p0, OF=OF, r_t=r_t, eps=eps, L_noz=L_noz,
                fuel_name=fuel_name, ox_name=ox_name,
                fuel_coolprop=fuel_coolprop, ox_coolprop=ox_coolprop,
                T_fuel_tank=T_fuel_tank, T_ox_tank=T_ox_tank,
                eta_pump_fuel=eta_pump_fuel, eta_pump_ox=eta_pump_ox,
                eta_turb_fuel=eta_turb_fuel, eta_turb_ox=eta_turb_ox,
                eta_mech=eta_mech, coolant_fraction_override=coolant_frac, g0=g0,
            )
            # Return power margins (positive = feasible)
            fuel_margin = cycle_result['P_turb_avail_fuel'] - cycle_result['P_turb_req_fuel']
            ox_margin = cycle_result['P_turb_avail_ox'] - cycle_result['P_turb_req_ox']
            return np.array([fuel_margin, ox_margin])
        except:
            return np.array([-1e9, -1e9])  # Large negative = infeasible
    
    def constraint_wall_temperature(x):
        """Constraint: Maximum wall temperature <= limit.
        
        Returns T_wall_max_limit - T_wall_max (should be >= 0).
        """
        p0, OF, r_t, eps, L_noz, coolant_frac = x
        try:
            cycle_result = ffsc_full_flow_cycle(
                F_vac=F_vac, p0=p0, OF=OF, r_t=r_t, eps=eps, L_noz=L_noz,
                fuel_name=fuel_name, ox_name=ox_name,
                fuel_coolprop=fuel_coolprop, ox_coolprop=ox_coolprop,
                T_fuel_tank=T_fuel_tank, T_ox_tank=T_ox_tank,
                eta_pump_fuel=eta_pump_fuel, eta_pump_ox=eta_pump_ox,
                eta_turb_fuel=eta_turb_fuel, eta_turb_ox=eta_turb_ox,
                eta_mech=eta_mech, coolant_fraction_override=coolant_frac, g0=g0,
            )
            # Return temperature margin (positive = feasible)
            return T_wall_max_limit - cycle_result['T_wall_max']
        except:
            return -1e9  # Large negative = infeasible
    
    # ==================== Setup Constraints for Optimizer ====================
    constraints = [
        NonlinearConstraint(constraint_turbopump_balance, lb=[0.0, 0.0], ub=[np.inf, np.inf]),
        NonlinearConstraint(constraint_wall_temperature, lb=0.0, ub=np.inf),
    ]
    
    # ==================== Pre-flight Checks ====================
    if verbose:
        print(f"\n{'='*70}")
        print(f"Starting FFSC Design Optimization")
        print(f"{'='*70}")
        print(f"Target thrust: {F_vac/1e3:.1f} kN")
        print(f"Propellants: {fuel_name}/{ox_name}")
        print(f"Tank temps: T_fuel={T_fuel_tank:.1f} K, T_ox={T_ox_tank:.1f} K")
        print(f"Ambient pressure: {p_amb/1e3:.1f} kPa")
        print(f"Wall temp limit: {T_wall_max_limit:.1f} K")
        print(f"\nBounds:")
        print(f"  p0:           {p0_bounds[0]/1e6:.1f} - {p0_bounds[1]/1e6:.1f} MPa")
        print(f"  OF:           {OF_bounds[0]:.2f} - {OF_bounds[1]:.2f}")
        print(f"  r_t:          {r_t_bounds[0]*1000:.1f} - {r_t_bounds[1]*1000:.1f} mm")
        print(f"  eps:          {eps_bounds[0]:.1f} - {eps_bounds[1]:.1f}")
        print(f"  L_noz:        {L_noz_bounds[0]:.2f} - {L_noz_bounds[1]:.2f} m")
        print(f"  coolant_frac: {coolant_frac_bounds[0]*100:.1f}% - {coolant_frac_bounds[1]*100:.1f}%")
        print(f"\nInitial guess:")
        print(f"  p0={x0[0]/1e6:.1f} MPa, OF={x0[1]:.2f}, r_t={x0[2]*1000:.1f} mm, "
              f"eps={x0[3]:.1f}, L_noz={x0[4]:.2f} m, coolant_frac={x0[5]*100:.1f}%")
        print(f"\n--- Testing initial guess ---")
    
    # Test initial guess before optimization
    try:
        initial_obj_value = objective(x0)
        if np.isfinite(initial_obj_value) and initial_obj_value < 1e5:
            if verbose:
                print(f"✓ Initial guess valid: Isp = {-initial_obj_value:.2f} s")
        else:
            if verbose:
                print(f"⚠ WARNING: Initial guess produced invalid result (obj={initial_obj_value:.2e})")
                print(f"  This may indicate a problem with input parameters or bounds.")
                print(f"  Attempting optimization anyway...")
    except Exception as e:
        if verbose:
            print(f"✗ ERROR: Initial guess failed completely!")
            print(f"  Exception: {type(e).__name__}: {str(e)}")
            print(f"  Cannot proceed with optimization.")
        # Return failure immediately
        return dict(
            p0_opt=x0[0], OF_opt=x0[1], r_t_opt=x0[2], eps_opt=x0[3], 
            L_noz_opt=x0[4], coolant_frac_opt=x0[5],
            Isp_opt=-np.inf, F_vac=F_vac, m_dot_total=np.nan,
            success=False,
            message=f"Initial guess evaluation failed: {type(e).__name__}: {str(e)}",
            n_iterations=0, n_evaluations=0, cycle_results=None,
        )
    
    if verbose:
        print(f"\n{'='*70}\n")
    
    # ==================== Run Optimization ====================
    result = minimize(
        objective,
        x0=x0,
        method=optimizer_method,
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': max_iterations, 'disp': verbose},
    )
    
    # ==================== Extract Optimal Results ====================
    if result.success:
        p0_opt, OF_opt, r_t_opt, eps_opt, L_noz_opt, coolant_frac_opt = result.x
        
        # Run final cycle analysis at optimum
        cycle_opt = ffsc_full_flow_cycle(
            F_vac=F_vac,
            p0=p0_opt,
            OF=OF_opt,
            r_t=r_t_opt,
            eps=eps_opt,
            L_noz=L_noz_opt,
            fuel_name=fuel_name,
            ox_name=ox_name,
            fuel_coolprop=fuel_coolprop,
            ox_coolprop=ox_coolprop,
            T_fuel_tank=T_fuel_tank,
            T_ox_tank=T_ox_tank,
            eta_pump_fuel=eta_pump_fuel,
            eta_pump_ox=eta_pump_ox,
            eta_turb_fuel=eta_turb_fuel,
            eta_turb_ox=eta_turb_ox,
            eta_mech=eta_mech,
            coolant_fraction_override=coolant_frac_opt,
            g0=g0,
        )
        
        # Compute final Isp
        if p_amb > 0:
            from .sweep import ideal_atmospheric_isp_from_eps
            Isp_atm, _, _, _ = ideal_atmospheric_isp_from_eps(
                p0=cycle_opt['T0'],
                T0=cycle_opt['T0'],
                gamma=cycle_opt['gamma'],
                R_g=cycle_opt['R_g'],
                eps=eps_opt,
                p_amb=p_amb,
                g0=g0,
            )
            Isp_opt = 0.95 * Isp_atm
        else:
            Isp_opt = cycle_opt['Isp_vac_eff']
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Optimization Complete - SUCCESS")
            print(f"{'='*70}")
            print(f"\nOptimal Design:")
            print(f"  Chamber pressure: {p0_opt/1e6:.2f} MPa")
            print(f"  O/F ratio:        {OF_opt:.3f}")
            print(f"  Throat radius:    {r_t_opt*1000:.2f} mm")
            print(f"  Expansion ratio:  {eps_opt:.2f}")
            print(f"  Nozzle length:    {L_noz_opt:.3f} m")
            print(f"  Coolant fraction: {coolant_frac_opt*100:.1f}%")
            print(f"\nOptimal Performance:")
            print(f"  Isp:              {Isp_opt:.2f} s")
            print(f"  Total mass flow:  {cycle_opt['m_dot_total']:.2f} kg/s")
            print(f"  Fuel flow:        {cycle_opt['m_dot_fuel']:.2f} kg/s")
            print(f"  Oxidizer flow:    {cycle_opt['m_dot_ox']:.2f} kg/s")
            print(f"\nFeasibility Check:")
            print(f"  Fuel turbopump:   {'✓' if cycle_opt['fuel_side_ok'] else '✗'}")
            print(f"  Ox turbopump:     {'✓' if cycle_opt['ox_side_ok'] else '✗'}")
            print(f"  Max wall temp:    {cycle_opt['T_wall_max']:.1f} K "
                  f"({'✓' if cycle_opt['T_wall_max'] <= T_wall_max_limit else '✗'})")
            print(f"  Coolant fraction: {cycle_opt['coolant_fraction']*100:.1f}% "
                  f"({'✓' if cycle_opt['coolant_fraction'] <= 0.6 else '✗'})")
            print(f"\nOptimizer Statistics:")
            print(f"  Iterations:       {result.nit}")
            print(f"  Function evals:   {result.nfev}")
            print(f"  Message:          {result.message}")
            print(f"{'='*70}\n")
        
        # Log any errors that occurred during optimization
        if len(error_log['exceptions']) > 0 and verbose:
            print(f"\nNote: {len(error_log['exceptions'])} unique errors occurred during optimization:")
            for i, (err, params) in enumerate(zip(error_log['exceptions'], error_log['error_params'])):
                print(f"  {i+1}. {err}")
                print(f"     at p0={params['p0_MPa']:.1f} MPa, OF={params['OF']:.2f}, eps={params['eps']:.1f}")
        
        return dict(
            # Optimal parameters
            p0_opt=p0_opt,
            OF_opt=OF_opt,
            r_t_opt=r_t_opt,
            eps_opt=eps_opt,
            L_noz_opt=L_noz_opt,
            coolant_frac_opt=coolant_frac_opt,
            # Performance
            Isp_opt=Isp_opt,
            F_vac=F_vac,
            m_dot_total=cycle_opt['m_dot_total'],
            # Optimization results
            success=True,
            message=result.message,
            n_iterations=result.nit,
            n_evaluations=result.nfev,
            # Complete cycle at optimum
            cycle_results=cycle_opt,
        )
    
    else:
        # Optimization failed - return best result found so far
        if verbose:
            print(f"\n{'='*70}")
            print(f"Optimization Complete - FAILED TO CONVERGE")
            print(f"{'='*70}")
            print(f"\nMessage: {result.message}")
            print(f"Function evaluations: {result.nfev}")
            print(f"Iterations: {result.nit if hasattr(result, 'nit') else 'N/A'}")
            
            # Show detailed error diagnostics
            if len(error_log['exceptions']) > 0:
                print(f"\n⚠ ERRORS ENCOUNTERED ({len(error_log['exceptions'])} unique types):")
                for i, (err, params) in enumerate(zip(error_log['exceptions'], error_log['error_params'])):
                    print(f"\n  Error {i+1} (eval {params['eval']}):")
                    print(f"    {err}")
                    print(f"    Parameters: p0={params['p0_MPa']:.2f} MPa, OF={params['OF']:.2f}, "
                          f"r_t={params['r_t_mm']:.1f} mm, eps={params['eps']:.1f}, "
                          f"L={params['L_noz']:.2f} m, cool={params['coolant_frac']*100:.1f}%")
                print(f"\n  Common causes:")
                print(f"    - Thermodynamic properties out of range (check tank temperatures)")
                print(f"    - CoolProp fluid name mismatch (fuel_coolprop/ox_coolprop)")
                print(f"    - Cantera species name mismatch (fuel_name/ox_name)")
                print(f"    - Extreme parameter combinations causing numerical issues")
            
            print(f"\nBest result found during search:")
            if best_result['Isp'] > -np.inf and best_result['x'] is not None:
                print(f"  Best Isp: {best_result['Isp']:.2f} s")
                p0_b, OF_b, r_t_b, eps_b, L_b, cool_b = best_result['x']
                print(f"  At: p0={p0_b/1e6:.2f} MPa, OF={OF_b:.3f}, "
                      f"r_t={r_t_b*1000:.1f} mm, eps={eps_b:.2f}, "
                      f"L={L_b:.3f} m, coolant={cool_b*100:.1f}%")
            else:
                print(f"  ✗ No valid solutions found - ALL evaluations failed!")
                print(f"\n  DIAGNOSIS:")
                print(f"    ALL function evaluations returned errors or invalid Isp.")
                print(f"    This typically means:")
                print(f"      1. Input parameters are fundamentally incompatible")
                print(f"      2. Bounds are outside feasible design space")
                print(f"      3. Tank temperatures don't match propellant expectations")
                print(f"      4. CoolProp/Cantera species names are incorrect")
                print(f"\n  SUGGESTIONS:")
                print(f"    - Verify fuel_coolprop='{fuel_coolprop}' is valid for CoolProp")
                print(f"    - Verify ox_coolprop='{ox_coolprop}' is valid for CoolProp")
                print(f"    - Verify fuel_name='{fuel_name}' exists in Cantera gas.yaml")
                print(f"    - Verify ox_name='{ox_name}' exists in Cantera gas.yaml")
                print(f"    - Check tank temps: T_fuel={T_fuel_tank}K, T_ox={T_ox_tank}K")
                print(f"    - Try wider bounds or different initial guess")
            print(f"{'='*70}\n")
        
        return dict(
            p0_opt=best_result['x'][0] if best_result['x'] is not None else x0[0],
            OF_opt=best_result['x'][1] if best_result['x'] is not None else x0[1],
            r_t_opt=best_result['x'][2] if best_result['x'] is not None else x0[2],
            eps_opt=best_result['x'][3] if best_result['x'] is not None else x0[3],
            L_noz_opt=best_result['x'][4] if best_result['x'] is not None else x0[4],
            coolant_frac_opt=best_result['x'][5] if best_result['x'] is not None else x0[5],
            Isp_opt=best_result['Isp'],
            F_vac=F_vac,
            m_dot_total=best_result['cycle']['m_dot_total'] if best_result['cycle'] else None,
            success=False,
            message=result.message,
            n_iterations=result.nit,
            n_evaluations=result.nfev,
            cycle_results=best_result['cycle'],
        )
