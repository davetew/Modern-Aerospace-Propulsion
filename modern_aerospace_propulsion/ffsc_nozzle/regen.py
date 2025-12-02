"""1D regenerative nozzle cooling model.

This module provides a simple axisymmetric, quasi-1D model of a
regeneratively cooled rocket nozzle. The model integrates heat transfer
from hot combustion gases through the nozzle wall to the regenerative
coolant flowing in channels along the nozzle exterior.

Key features:
- Bartz correlation for gas-side heat transfer
- Gnielinski correlation for coolant-side heat transfer
- Conjugate heat transfer with wall conduction
- 1D axial integration of coolant temperature and pressure

Module Summary:
- Functions:
    - ``make_coolprop_liquid(fluid_name)``: Returns CoolProp-based liquid property callable.
    - ``k_wall_copper(T)``: Approximate thermal conductivity for copper alloy.
    - ``isentropic_M_from_area_ratio(Ar, gamma)``: Supersonic Mach from area ratio (Newton–Raphson).
    - ``gas_state_from_area(At, A, p0, T0, gamma, R_g)``: Static p,T,M from stagnation and area.
    - ``bartz_h(...)``: Gas-side heat transfer coefficient and local state via Bartz.
    - ``friction_factor_haaland(Re, eps_over_D)``: Darcy friction factor approximation.
    - ``Nu_gnielinski(Re, Pr, f)``: Nusselt number for turbulent internal flow.
    - ``simple_conical_nozzle(r_t, expansion_ratio, L, N)``: Conical nozzle geometry arrays.
    - ``regen_nozzle_1D(...)``: Axial conjugate cooling solution (returns arrays of state/HTC).
"""

import numpy as np
import math
from typing import Callable, Dict, Tuple, Optional

from CoolProp.CoolProp import PropsSI

def make_coolprop_liquid(fluid_name: str) -> Callable[[float, float], Tuple[float, float, float, float]]:
    """Create a CoolProp-based property function for a given fluid.
    
    Args:
        fluid_name: Name of the fluid as recognized by CoolProp (e.g., 'Methane', 'Nitrogen', 'Oxygen')
        
    Returns:
        A callable that takes temperature (K) and pressure (Pa) and returns:
            - rho: Density (kg/m³)
            - cp: Specific heat capacity (J/kg/K)
            - k: Thermal conductivity (W/m/K)
            - mu: Dynamic viscosity (Pa·s)
            
    Example:
        >>> props = make_coolprop_liquid('Methane')
        >>> rho, cp, k, mu = props(T=110.0, p=30e6)
    """
    def props(T: float, p: float) -> Tuple[float, float, float, float]:
        # Query CoolProp for thermophysical properties at given state
        rho = PropsSI("D", "T", T, "P", p, fluid_name)  # Density
        cp  = PropsSI("C", "T", T, "P", p, fluid_name)  # Specific heat at constant pressure
        k   = PropsSI("L", "T", T, "P", p, fluid_name)  # Thermal conductivity
        mu  = PropsSI("V", "T", T, "P", p, fluid_name)  # Dynamic viscosity
        return rho, cp, k, mu
    return props


def k_wall_copper(T: float) -> float:
    """Approximate thermal conductivity for a copper-based alloy.
    
    Provides a simple linear correlation for copper alloy thermal conductivity
    as a function of temperature, typical for regenerative cooling jackets.
    
    Args:
        T: Wall temperature (K)
        
    Returns:
        Thermal conductivity (W/m/K)
        
    Note:
        - Below 300 K: returns 350 W/m/K
        - Above 900 K: returns 250 W/m/K
        - Between: linear interpolation
    """
    # Copper conductivity decreases with temperature
    # Use constant values at extremes to avoid extrapolation issues
    if T < 300.0:
        return 350.0  # Cold limit (high conductivity)
    elif T > 900.0:
        return 250.0  # Hot limit (reduced conductivity)
    else:
        # Linear interpolation: decreases 100 W/m/K over 600 K range
        return 350.0 - (T-300.0)*(100.0/600.0)


def isentropic_M_from_area_ratio(Ar: float, gamma: float) -> float:
    """Solve for supersonic Mach number from area ratio using Newton-Raphson.
    
    Uses the isentropic area-Mach relation for a converging-diverging nozzle
    to find the supersonic Mach number corresponding to a given area ratio.
    
    Args:
        Ar: Area ratio A/A*, where A* is the throat area (dimensionless)
        gamma: Ratio of specific heats (dimensionless)
        
    Returns:
        Supersonic Mach number (M > 1.0)
        
    Note:
        Uses Newton-Raphson iteration starting from M=2.0. Assumes supersonic
        branch of the area-Mach relation. Converges to within 1e-6 or returns
        result after 50 iterations.
    """
    # Initial guess for supersonic Mach number
    M = 2.0
    
    # Newton-Raphson iteration to solve: (A/A*)(M, gamma) - Ar = 0
    for _ in range(50):
        # Evaluate area-Mach relation at current M
        term = (2.0/(gamma+1.0))*(1.0 + 0.5*(gamma-1.0)*M**2)
        f = (1.0/M)*(term**((gamma+1.0)/(2.0*(gamma-1.0)))) - Ar
        
        # Numerical derivative using finite difference
        dM = 1e-4
        term2 = (2.0/(gamma+1.0))*(1.0 + 0.5*(gamma-1.0)*(M+dM)**2)
        f2 = (1.0/(M+dM))*(term2**((gamma+1.0)/(2.0*(gamma-1.0)))) - Ar
        df = (f2-f)/dM
        
        # Newton-Raphson update: M_new = M - f/f'
        M_new = M - f/df
        
        # Enforce supersonic constraint (M > 1)
        if M_new < 1.01:
            M_new = 1.01
        
        # Check convergence
        if abs(M_new - M) < 1e-6:
            break
        M = M_new
    
    return M


def gas_state_from_area(At: float, A: float, p0: float, T0: float, 
                        gamma: float, R_g: float) -> Tuple[float, float, float]:
    """Compute static gas properties from stagnation conditions and local area.
    
    Applies isentropic flow relations to find local static pressure, temperature,
    and Mach number at a given axial station in the nozzle.
    
    Args:
        At: Throat area (m²)
        A: Local cross-sectional area (m²)
        p0: Stagnation (chamber) pressure (Pa)
        T0: Stagnation (chamber) temperature (K)
        gamma: Ratio of specific heats (dimensionless)
        R_g: Specific gas constant (J/kg/K)
        
    Returns:
        Tuple of (p, T, M):
            - p: Local static pressure (Pa)
            - T: Local static temperature (K)
            - M: Local Mach number (dimensionless)
    """
    # Compute area ratio relative to throat
    Ar = A/At
    if Ar < 1.0:  # Ensure area ratio is at least 1 (throat condition)
        Ar = 1.0
    
    # Solve for Mach number from area ratio (supersonic branch)
    M = isentropic_M_from_area_ratio(Ar, gamma)
    
    # Compute static temperature from stagnation using isentropic relation
    # T/T0 = 1 / (1 + (gamma-1)/2 * M^2)
    T = T0 / (1.0 + 0.5*(gamma-1.0)*M**2)
    
    # Compute static pressure using isentropic relation
    # p/p0 = (T/T0)^(gamma/(gamma-1))
    p = p0 * (T/T0)**(gamma/(gamma-1.0))
    
    return p, T, M


def bartz_h(p0: float, T0: float, T_wall: float, r_t: float, At: float, A: float,
           gamma: float, R_g: float, mu_g: float = 3e-5, Pr_g: float = 0.9,
           ) -> Tuple[float, float, float, float]:
    """Compute gas-side heat transfer coefficient using Bartz correlation.
    
    Implements a simplified form of the Bartz equation for convective heat
    transfer from hot combustion gases to the nozzle wall. Accounts for
    compressibility effects and area ratio variations.
    
    Args:
        p0: Stagnation (chamber) pressure (Pa)
        T0: Stagnation (chamber) temperature (K)
        r_t: Throat radius (m)
        At: Throat area (m²)
        A: Local cross-sectional area (m²)
        gamma: Ratio of specific heats (dimensionless)
        R_g: Specific gas constant (J/kg/K)
        mu_g: Gas dynamic viscosity (Pa·s), default 3e-5
        Pr_g: Gas Prandtl number (dimensionless), default 0.9
        T_wall: Wall temperature (K)

    Returns:
        Tuple of (h_g, p, T, M):
            - h_g: Gas-side heat transfer coefficient (W/m²/K), minimum 1e3
            - p: Local static pressure (Pa)
            - T: Local static temperature (K)
            - M: Local Mach number (dimensionless)
            
    Reference:
        Bartz, D. R. (1957). "A Simple Equation for Rapid Estimation of Rocket
        Nozzle Convective Heat Transfer Coefficients." Jet Propulsion, 27(1).
    """
    # Compute specific heat at constant pressure from ideal gas relations
    cp_g = gamma*R_g/(gamma-1.0)

    # Total to static temperature ratio
    Tt_T = lambda M, ga: 1 + 0.5*(ga-1.0)*M**2

    # Get local gas state (pressure, temperature, Mach number)
    p, T, M = gas_state_from_area(At, A, p0, T0, gamma, R_g)
    
    # Bartz correlation: h = C / d_t^0.2 * mu^0.2 * cp / Pr^0.6 * 
    #                         (p0/c*)^0.8 * (At/A)^0.9 * sigma
    C = 0.026  # Empirical constant
    d_t = 2.0*r_t  # Throat diameter (m)
    sigma = (0.5*T_wall/T0*Tt_T(M, gamma) + 0.5)**(-0.68)*Tt_T(M, gamma)**(-0.12)  # Correction factor

    # Characteristic velocity
    c_star = math.sqrt(R_g*T0/gamma) * (2.0/(gamma+1.0))**((gamma+1.0)/(2.0*(gamma-1.0)))

    # Compute heat transfer coefficient
    h_g = C/d_t**0.2 * mu_g**0.2 * cp_g / Pr_g**0.6 * \
          (p0/c_star)**0.8 * (At/A)**0.9 * sigma
    
    # Enforce minimum heat transfer coefficient to avoid numerical issues
    return max(h_g, 1e3), p, T, M


def friction_factor_haaland(Re: float, eps_over_D: float) -> float:
    """Compute Darcy friction factor using Haaland approximation.
    
    The Haaland equation is an explicit approximation to the implicit
    Colebrook-White equation for turbulent pipe flow. For laminar flow
    (Re < 2300), returns the exact laminar result f = 64/Re.
    
    Args:
        Re: Reynolds number (dimensionless)
        eps_over_D: Relative roughness ε/D (dimensionless)
        
    Returns:
        Darcy friction factor (dimensionless)
        
    Reference:
        Haaland, S. E. (1983). "Simple and Explicit Formulas for the Friction
        Factor in Turbulent Pipe Flow." Journal of Fluids Engineering, 105(1).
    """
    # Handle edge case of zero or negative Reynolds number
    if Re <= 0.0:
        return 0.0
    
    # Laminar flow: use exact Hagen-Poiseuille result
    if Re < 2300.0:
        return 64.0/Re
    
    # Turbulent flow: use Haaland explicit approximation to Colebrook-White
    # f = 1 / (-1.8 * log10((ε/D/3.7)^1.11 + 6.9/Re))^2
    term = (eps_over_D/3.7)**1.11 + 6.9/Re
    return 1.0/(-1.8*math.log10(term))**2


def Nu_gnielinski(Re: float, Pr: float, f: float) -> float:
    """Compute Nusselt number for turbulent internal flow using Gnielinski correlation.
    
    The Gnielinski equation is valid for turbulent flow in smooth and rough pipes.
    For laminar flow (Re < 2300), returns the fully-developed constant-wall-temperature
    value Nu = 4.36.
    
    Args:
        Re: Reynolds number (dimensionless)
        Pr: Prandtl number (dimensionless)
        f: Darcy friction factor (dimensionless)
        
    Returns:
        Nusselt number (dimensionless)
        
    Note:
        Valid for 2300 < Re < 5×10⁶ and 0.5 < Pr < 2000.
        
    Reference:
        Gnielinski, V. (1976). "New Equations for Heat and Mass Transfer in
        Turbulent Pipe and Channel Flow." International Chemical Engineering, 16(2).
    """
    # Laminar flow: use fully-developed circular pipe value for constant wall temperature
    if Re < 2300.0:
        return 4.36
    
    # Turbulent flow: Gnielinski correlation
    # Nu = [(f/8) * (Re - 1000) * Pr] / [1 + 12.7 * sqrt(f/8) * (Pr^(2/3) - 1)]
    return ((f/8.0)*(Re-1000.0)*Pr) / (1.0 + 12.7*math.sqrt(f/8.0)*(Pr**(2.0/3.0)-1.0))


def simple_conical_nozzle(r_t: float, expansion_ratio: float, L: float, 
                          N: int = 200) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Generate simple conical nozzle geometry from throat to exit.
    
    Creates a straight-walled conical diverging nozzle with linearly
    increasing radius from throat to exit.
    
    Args:
        r_t: Throat radius (m)
        expansion_ratio: Area ratio A_exit/A_throat (dimensionless)
        L: Nozzle length from throat to exit (m)
        N: Number of axial stations (default 200)
        
    Returns:
        Tuple of (x, r, A, At):
            - x: Axial coordinate array (m), shape (N,)
            - r: Local radius array (m), shape (N,)
            - A: Local area array (m²), shape (N,)
            - At: Throat area (m²), scalar
            
    Note:
        This is a simplified geometry. Real rocket nozzles typically use
        bell-shaped contours for better performance.
    """
    # Compute exit radius from expansion ratio: A_e/A_t = (r_e/r_t)^2
    r_e = r_t*math.sqrt(expansion_ratio)
    
    # Create uniform axial grid from throat (x=0) to exit (x=L)
    x = np.linspace(0.0, L, N)
    
    # Linear radius variation (conical wall)
    r = r_t + (r_e - r_t)*(x/L)
    
    # Compute local cross-sectional area
    A = math.pi*r**2
    
    # Throat area (minimum area)
    At = math.pi*r_t**2
    
    return x, r, A, At


def regen_nozzle_1D(
        x: np.ndarray,
        r: np.ndarray,
        A: np.ndarray,
        At: float,
        p0: float,
        T0: float,
        gamma: float,
        R_g: float,
        coolant: str = "LCH4",
        coolant_props: Optional[Callable[[float, float], Tuple[float, float, float, float]]] = None,
        m_dot_cool: float = 5.0,
        n_channels: int = 100,
        w_channel: float = 0.0015,
        h_channel: float = 0.0020,
        wall_thickness: float = 0.0020,
        roughness: float = 5e-6,
        T_cool_in: float = 110.0,
        p_cool_in: float = 1.0e7,
        L_regen: Optional[float] = None,
        emissivity_ext: float = 0.8,
        T_env: float = 3.0,
        wall_k_func: Callable[[float], float] = k_wall_copper,
        max_iter_wall: int = 10,
    ) -> Dict[str, np.ndarray]:
    """Integrate 1D regenerative cooling solution along nozzle axis.
    
    Solves conjugate heat transfer problem for regenerative nozzle cooling,
    marching axially from throat to exit. Accounts for:
    - Gas-side convection (Bartz correlation)
    - Wall conduction (1D radial)
    - Coolant-side convection (Gnielinski correlation with Sieder-Tate correction)
    - Coolant pressure drop (Darcy-Weisbach)
    - Coolant temperature rise (energy balance)
    
    Args:
        x: Axial coordinate array (m), shape (N,)
        r: Local radius array (m), shape (N,)
        A: Local area array (m²), shape (N,)
        At: Throat area (m²)
        p0: Stagnation (chamber) pressure (Pa)
        T0: Stagnation (chamber) temperature (K)
        gamma: Ratio of specific heats (dimensionless)
        R_g: Specific gas constant (J/kg/K)
        coolant: Coolant name (informational only, not used)
        coolant_props: Function returning (rho, cp, k, mu) given (T, p).
                       Must be provided; use make_coolprop_liquid(fluid_name)
        m_dot_cool: Total coolant mass flow rate (kg/s)
        n_channels: Number of parallel coolant channels (dimensionless)
        w_channel: Channel width (m)
        h_channel: Channel height (m)
        wall_thickness: Nozzle wall thickness (m)
        roughness: Channel surface roughness (m)
        T_cool_in: Coolant inlet temperature (K)
        p_cool_in: Coolant inlet pressure (Pa)
        L_regen: Regenerative cooling length (m). If None, cools entire nozzle
        emissivity_ext: External surface emissivity for radiation (dimensionless)
        T_env: Environment temperature for radiation (K)
        wall_k_func: Function returning wall thermal conductivity (W/m/K) given T (K)
        max_iter_wall: Maximum iterations for wall temperature convergence
        
    Returns:
        Dictionary containing solution arrays (all shape (N,) except D_h):
            - x: Axial coordinates (m)
            - r: Local radius (m)
            - A: Local area (m²)
            - At: Throat area (m²), scalar
            - T_cool: Coolant temperature (K)
            - p_cool: Coolant pressure (Pa)
            - T_wall_inner: Inner wall temperature (K)
            - T_wall_outer: Outer wall temperature (K)
            - qpp: Heat flux (W/m²)
            - h_cool: Coolant-side heat transfer coefficient (W/m²/K)
            - h_gas: Gas-side heat transfer coefficient (W/m²/K)
            - p_g: Gas static pressure (Pa)
            - T_g: Gas static temperature (K)
            - M_g: Gas Mach number (dimensionless)
            - D_h: Hydraulic diameter of coolant channels (m), scalar
            
    Raises:
        ValueError: If coolant_props is None
        
    Note:
        - Solution marches axially assuming uniform conditions at each station
        - Wall temperature is iterated to convergence at each station using
          Sieder-Tate viscosity correction
        - Beyond L_regen, radiative cooling is assumed
    """
    # ==================== Initialization ====================
    N = len(x)  # Number of axial stations
    dx = x[1] - x[0]  # Axial step size (assuming uniform grid)
    
    # Set regenerative cooling length (default to full nozzle)
    if L_regen is None:
        L_regen = x[-1]

    # Verify coolant properties function is provided
    if coolant_props is None:
        raise ValueError(f"coolant_props must be provided. Use make_coolprop_liquid(fluid_name).")
    props_cool = coolant_props

    # ==================== Channel Geometry ====================
    # Compute hydraulic diameter for rectangular channels
    A_ch = w_channel*h_channel  # Channel cross-sectional area
    P_wet = 2.0*(w_channel + h_channel)  # Wetted perimeter
    D_h = 4.0*A_ch/P_wet  # Hydraulic diameter: D_h = 4A/P

    # ==================== Allocate Solution Arrays ====================
    T_cool = np.zeros(N)  # Coolant temperature
    p_cool = np.zeros(N)  # Coolant pressure
    T_wall_inner = np.zeros(N)  # Inner (gas-side) wall temperature
    T_wall_outer = np.zeros(N)  # Outer (coolant-side) wall temperature
    qpp = np.zeros(N)  # Heat flux (W/m²)
    h_cool = np.zeros(N)  # Coolant-side heat transfer coefficient
    h_gas = np.zeros(N)  # Gas-side heat transfer coefficient
    p_g = np.zeros(N)  # Gas static pressure
    T_g = np.zeros(N)  # Gas static temperature
    M_g = np.zeros(N)  # Gas Mach number

    # ==================== Inlet Boundary Conditions ====================
    T_cool[0] = T_cool_in
    p_cool[0] = p_cool_in

    # ==================== Axial Integration Loop ====================
    for i in range(N):
        # Local geometry at station i
        r_i = r[i]  # Local nozzle radius
        A_i = A[i]  # Local cross-sectional area
        perim_inner = 2.0*math.pi*r_i  # Inner perimeter for heat transfer area
        A_seg_inner = perim_inner*dx  # Heat transfer area for this segment
        
        # Check if regenerative cooling is active at this station
        regen_active = (x[i] <= L_regen)

        # -------------------- Gas-Side Conditions --------------------
        # Compute gas-side heat transfer coefficient and local flow state
        hg_i, pgi, Tgi, Mi = bartz_h(
            p0=p0, T0=T0, T_wall=T_wall_inner[i], r_t=math.sqrt(At/math.pi),
            At=At, A=A_i,
            gamma=gamma, R_g=R_g, 
        )
        h_gas[i] = hg_i  # Store gas-side heat transfer coefficient
        p_g[i] = pgi  # Store gas static pressure
        T_g[i] = Tgi  # Store gas static temperature
        M_g[i] = Mi  # Store Mach number

        # -------------------- Radiative Cooling Region --------------------
        # Beyond L_regen, assume no regenerative cooling (radiative cooling only)
        if not regen_active:
            sigma = 5.670374419e-8  # Stefan-Boltzmann constant (W/m²/K⁴)
            T_wall_inner[i] = 0.7*T_g[i]  # Approximate wall temperature
            # Radiative heat loss to environment
            qpp[i] = emissivity_ext*sigma*(T_wall_inner[i]**4 - T_env**4)
            T_wall_outer[i] = T_wall_inner[i]  # No conduction gradient
            
            # Coolant state remains constant beyond regenerative region
            if i < N-1:
                T_cool[i+1] = T_cool[i]
                p_cool[i+1] = p_cool[i]
            continue

        # -------------------- Coolant Properties & Flow Parameters --------------------
        # Get coolant thermophysical properties at current state
        rho_c, cp_c, k_c, mu_c = props_cool(T_cool[i], p_cool[i])
        
        # Compute mass flux and velocity in each channel
        G = (m_dot_cool/n_channels)/A_ch  # Mass flux per channel (kg/m²/s)
        v_c = G/rho_c  # Coolant velocity (m/s)
        
        # Dimensionless numbers for coolant flow
        Re = G*D_h/mu_c  # Reynolds number
        Pr = cp_c*mu_c/k_c  # Prandtl number
        
        # Compute friction factor for pressure drop
        eps_over_D = roughness/max(D_h, 1e-9)  # Relative roughness
        f = friction_factor_haaland(Re, eps_over_D)  # Darcy friction factor
        
        # Base Nusselt number (without viscosity correction)
        Nu_base = Nu_gnielinski(Re, Pr, f)

        # -------------------- Conjugate Heat Transfer (Iterative Solution) --------------------
        # Initial guess for inner wall temperature
        T_wall_guess = T_cool[i] + 50.0
        
        # Iterate to find wall temperature with Sieder-Tate viscosity correction
        for _ in range(max_iter_wall):
            # Evaluate coolant viscosity at wall temperature for Sieder-Tate correction
            _, _, _, mu_w = props_cool(T_wall_guess, p_cool[i])
            
            # Apply Sieder-Tate correction: Nu = Nu_base * (mu_bulk/mu_wall)^0.14
            Nu_ST = Nu_base*(mu_c/mu_w)**0.14
            h_c = Nu_ST*k_c/D_h  # Coolant-side heat transfer coefficient
            
            # Wall thermal conductivity at current wall temperature
            k_w = wall_k_func(T_wall_guess)
            
            # Total thermal resistance: gas convection + wall conduction + coolant convection
            R_tot = 1.0/hg_i + wall_thickness/k_w + 1.0/h_c
            
            # Heat flux from overall temperature difference and total resistance
            qpp_i = (T_g[i] - T_cool[i])/R_tot
            
            # Updated inner wall temperature from gas-side energy balance
            T_wall_new = T_g[i] - qpp_i*(1.0/hg_i)
            
            # Check convergence
            if abs(T_wall_new - T_wall_guess) < 1e-3:
                T_wall_guess = T_wall_new
                break
            T_wall_guess = T_wall_new

        # -------------------- Store Converged Results --------------------
        h_cool[i] = h_c  # Coolant-side heat transfer coefficient
        qpp[i] = qpp_i  # Heat flux
        T_wall_inner[i] = T_wall_guess  # Inner wall temperature
        # Outer wall temperature from conduction through wall
        T_wall_outer[i] = T_wall_inner[i] - qpp_i*(wall_thickness/wall_k_func(T_wall_inner[i]))

        # -------------------- Update Coolant State --------------------
        # Total heat transfer into coolant for this segment
        Q_seg = qpp_i*A_seg_inner
        
        # Temperature rise from energy balance: Q = m_dot * cp * dT
        dT = Q_seg/(m_dot_cool*cp_c)
        if i < N-1:
            T_cool[i+1] = T_cool[i] + dT

        # Pressure drop from Darcy-Weisbach equation: dp = f * (L/D) * (1/2) * rho * v^2
        dp_seg = f*(dx/D_h)*0.5*rho_c*v_c**2
        if i < N-1:
            p_cool[i+1] = p_cool[i] - dp_seg

    # ==================== Return Complete Solution ====================
    return dict(
        # Geometry
        x=x,  # Axial coordinates (m)
        r=r,  # Local radius (m)
        A=A,  # Local area (m²)
        At=At,  # Throat area (m²)
        # Coolant state
        T_cool=T_cool,  # Coolant temperature (K)
        p_cool=p_cool,  # Coolant pressure (Pa)
        # Wall temperatures
        T_wall_inner=T_wall_inner,  # Inner wall temperature (K)
        T_wall_outer=T_wall_outer,  # Outer wall temperature (K)
        # Heat transfer
        qpp=qpp,  # Heat flux (W/m²)
        h_cool=h_cool,  # Coolant-side HTC (W/m²/K)
        h_gas=h_gas,  # Gas-side HTC (W/m²/K)
        # Gas state
        p_g=p_g,  # Gas static pressure (Pa)
        T_g=T_g,  # Gas static temperature (K)
        M_g=M_g,  # Gas Mach number
        # Channel geometry
        D_h=D_h,  # Hydraulic diameter (m)
    )
