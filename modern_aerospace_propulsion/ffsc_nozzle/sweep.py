"""Parameter sweeps and feasibility mapping for FFSC designs.

Note: Requires Cantera and CoolProp to be installed.

Module Summary:
- Functions:
    - ``ideal_vacuum_isp_from_eps(p0, T0, gamma, R_g, eps, g0=9.80665)``:
        Ideal vacuum specific impulse and exit conditions from expansion ratio.
    - ``ideal_atmospheric_isp_from_eps(p0, T0, gamma, R_g, eps, p_amb, g0=9.80665)``:
        Ideal specific impulse at finite ambient pressure, including pressure thrust.
    - ``sweep_ffsc_feasibility(F_vac, r_t, eps, L_noz, p0_array, OF_array, g0=9.80665)``:
        2D feasibility map over chamber pressure and O/F using the FFSC cycle.
"""

import numpy as np
import math
from typing import Dict, Tuple

from .cycle import ffsc_full_flow_cycle


def ideal_vacuum_isp_from_eps(
        p0: float,
        T0: float,
        gamma: float,
        R_g: float,
        eps: float,
        g0: float = 9.80665
) -> Tuple[float, None, float, float]:
    """Compute ideal vacuum specific impulse from nozzle expansion ratio.
    
    Calculates theoretical maximum specific impulse assuming:
    - Isentropic expansion from chamber to exit
    - Perfectly expanded nozzle (exit pressure = 0 for vacuum)
    - Frozen flow composition (no further reactions in nozzle)
    - One-dimensional flow
    
    Uses isentropic relations to find exit Mach number from area ratio,
    then computes exit velocity and specific impulse.
    
    Args:
        p0: Chamber stagnation pressure (Pa)
        T0: Chamber stagnation temperature (K)
        gamma: Ratio of specific heats cp/cv (dimensionless)
        R_g: Specific gas constant (J/kg/K)
        eps: Nozzle expansion ratio A_exit/A_throat (dimensionless)
        g0: Standard gravity (m/s²), default 9.80665
        
    Returns:
        Tuple of (Isp_vac, None, T_e, M_e):
            - Isp_vac: Ideal vacuum specific impulse (s)
            - None: Placeholder for consistency with other functions
            - T_e: Exit static temperature (K)
            - M_e: Exit Mach number (dimensionless)
            
    Note:
        - Assumes area ratio At = 1.0 for calculation (arbitrary scale)
        - Exit velocity: V_e = M_e * sqrt(gamma * R_g * T_e)
        - Specific impulse: Isp = V_e / g0
        - Real nozzles achieve 94-98% of ideal Isp due to losses
        - Typical CH4/O2 at eps=15: Isp_ideal ~ 320 s, Isp_eff ~ 305 s
    """
    # ==================== Define Nozzle Areas ====================
    # Use unit throat area (actual scale doesn't matter for area ratio)
    At = 1.0
    Ae = eps*At  # Exit area from expansion ratio

    # ==================== Solve for Exit Mach Number ====================
    def isentropic_M_from_area_ratio(Ar, gamma):
        """Solve isentropic area-Mach relation for supersonic Mach number."""
        M = 2.0  # Initial guess (supersonic)
        
        # Newton-Raphson iteration
        for _ in range(50):
            # Evaluate area-Mach function: f(M) = (A/A*)(M) - Ar
            term = (2.0/(gamma+1.0))*(1.0 + 0.5*(gamma-1.0)*M**2)
            f = (1.0/M)*(term**((gamma+1.0)/(2.0*(gamma-1.0)))) - Ar
            
            # Numerical derivative (finite difference)
            dM = 1e-4
            term2 = (2.0/(gamma+1.0))*(1.0 + 0.5*(gamma-1.0)*(M+dM)**2)
            f2 = (1.0/(M+dM))*(term2**((gamma+1.0)/(2.0*(gamma-1.0)))) - Ar
            df = (f2-f)/dM
            
            # Newton-Raphson update
            M_new = M - f/df
            
            # Enforce supersonic constraint
            if M_new < 1.01:
                M_new = 1.01
            
            # Check convergence
            if abs(M_new - M) < 1e-6:
                break
            M = M_new
        return M

    # Compute area ratio and solve for exit Mach number
    Ar = Ae/At
    M_e = isentropic_M_from_area_ratio(Ar, gamma)
    
    # ==================== Compute Exit Conditions ====================
    # Exit static temperature from isentropic relation
    T_e = T0/(1.0 + 0.5*(gamma-1.0)*M_e**2)
    
    # Exit velocity: V = M * a, where a = sqrt(gamma * R * T)
    V_e = M_e*math.sqrt(gamma*R_g*T_e)
    
    # ==================== Convert to Specific Impulse ====================
    # Isp = V_e / g0 (seconds)
    return V_e/g0, None, T_e, M_e


def ideal_atmospheric_isp_from_eps(
        p0: float,
        T0: float,
        gamma: float,
        R_g: float,
        eps: float,
        p_amb: float,
        g0: float = 9.80665,
) -> Tuple[float, float, float, float]:
    """Compute ideal specific impulse with finite ambient pressure.
    
    Extends the vacuum Isp calculation by adding the pressure thrust term
    at the nozzle exit for an engine operating in an atmosphere with ambient
    pressure ``p_amb``. Assumes isentropic expansion and frozen flow.
    
    Thrust per unit mass flow:
    - F/ṁ = V_e + (p_e - p_amb) * A_e / ṁ
    - Isp = (F/ṁ)/g0 = V_e/g0 + (p_e - p_amb) * A_e / (ṁ g0)
    
    The mass flow rate ṁ is obtained from the choked mass flow relation at
    the throat assuming unit throat area (A_t = 1 m²), which makes the result
    independent of actual scale for a given expansion ratio ``eps``.
    
    Args:
        p0: Chamber stagnation pressure (Pa)
        T0: Chamber stagnation temperature (K)
        gamma: Ratio of specific heats (dimensionless)
        R_g: Specific gas constant (J/kg/K)
        eps: Nozzle expansion ratio A_e/A_t (dimensionless)
        p_amb: Ambient (back) pressure (Pa)
        g0: Standard gravity (m/s²), default 9.80665
    
    Returns:
        Tuple of (Isp_atm, p_e, T_e, M_e):
            - Isp_atm: Ideal Isp accounting for ambient pressure (s)
            - p_e: Exit static pressure (Pa)
            - T_e: Exit static temperature (K)
            - M_e: Exit Mach number (dimensionless)
    
    Note:
        - Uses unit throat area (A_t = 1 m²); A_e = eps
        - Choked mass flow per unit throat area:
          ṁ = (p0/√T0) * √(gamma/R_g) * (2/(gamma+1))^((gamma+1)/(2(gamma-1)))
        - Pressure thrust term reduces Isp when p_e < p_amb and increases when p_e > p_amb
        - If ``p_amb`` → 0, this reduces to the vacuum case
    """
    # ==================== Define Nozzle Areas ====================
    At = 1.0
    Ae = eps*At

    # ==================== Exit Mach, Temperature, Pressure ====================
    def isentropic_M_from_area_ratio(Ar, gamma):
        M = 2.0
        for _ in range(50):
            term = (2.0/(gamma+1.0))*(1.0 + 0.5*(gamma-1.0)*M**2)
            f = (1.0/M)*(term**((gamma+1.0)/(2.0*(gamma-1.0)))) - Ar
            dM = 1e-4
            term2 = (2.0/(gamma+1.0))*(1.0 + 0.5*(gamma-1.0)*(M+dM)**2)
            f2 = (1.0/(M+dM))*(term2**((gamma+1.0)/(2.0*(gamma-1.0)))) - Ar
            df = (f2-f)/dM
            M_new = M - f/df
            if M_new < 1.01:
                M_new = 1.01
            if abs(M_new - M) < 1e-6:
                break
            M = M_new
        return M

    Ar = Ae/At
    M_e = isentropic_M_from_area_ratio(Ar, gamma)
    T_e = T0/(1.0 + 0.5*(gamma-1.0)*M_e**2)
    p_e = p0 * (T_e/T0)**(gamma/(gamma-1.0))

    # Exit velocity
    V_e = M_e*math.sqrt(gamma*R_g*T_e)

    # ==================== Choked Mass Flow (Unit Throat Area) ====================
    mdot_per_At = (p0/math.sqrt(T0)) * math.sqrt(gamma/R_g) * (
        (2.0/(gamma+1.0)) ** ((gamma+1.0)/(2.0*(gamma-1.0)))
    )

    # Pressure thrust contribution per unit mass flow
    pressure_term = (p_e - p_amb) * (Ae/mdot_per_At)

    # Specific impulse
    Isp_atm = (V_e + pressure_term)/g0
    return Isp_atm, p_e, T_e, M_e


def sweep_ffsc_feasibility(
        F_vac: float,
        r_t: float,
        eps: float,
        L_noz: float,
        p0_array: np.ndarray,
        OF_array: np.ndarray,
        g0: float = 9.80665,
) -> Dict[str, np.ndarray]:
    """Perform 2D parameter sweep over chamber pressure and O/F ratio for FFSC cycle feasibility.
    
    Evaluates FFSC engine design at multiple operating points to map out
    the feasible design space. For each (p0, O/F) combination, performs
    complete cycle analysis and checks turbopump power balance, wall
    temperature limits, and coolant fraction constraints.
    
    Feasibility Criteria:
    1. Fuel turbine provides sufficient power for fuel pump
    2. Oxidizer turbine provides sufficient power for oxidizer pump
    3. Coolant fraction < 60% of fuel flow (reasonable cooling requirement)
    4. Maximum wall temperature ≤ 900 K (material limit)
    
    Use Cases:
    - Finding optimal chamber pressure and O/F ratio
    - Understanding design space boundaries
    - Identifying trade-offs between performance and feasibility
    - Sensitivity analysis for engine parameters

    Args:
        F_vac: Target vacuum thrust (N)
        r_t: Nozzle throat radius (m)
        eps: Nozzle expansion ratio A_exit/A_throat (dimensionless)
        L_noz: Nozzle length from throat to exit (m)
        p0_array: Array of chamber pressures to sweep (Pa)
        OF_array: Array of O/F ratios to sweep (dimensionless)
        g0: Standard gravity (m/s²), default 9.80665

    Returns:
        Dictionary containing 2D gridded results:
            - p0_grid: Meshgrid of chamber pressures (Pa), shape (len(p0_array), len(OF_array))
            - OF_grid: Meshgrid of O/F ratios (dimensionless), shape (len(p0_array), len(OF_array))
            - feasible: Boolean array indicating feasible designs, shape (len(p0_array), len(OF_array))
            - coolant_fraction: Coolant flow as fraction of fuel, shape (len(p0_array), len(OF_array))
            - T_wall_max: Maximum wall temperature (K), shape (len(p0_array), len(OF_array))
            - fuel_side_ok: Boolean array for fuel pump power balance, shape (len(p0_array), len(OF_array))
            - ox_side_ok: Boolean array for ox pump power balance, shape (len(p0_array), len(OF_array))
            - Isp_vac_eff: Effective vacuum specific impulse (s), shape (len(p0_array), len(OF_array))
            
    Note:
        - Failed evaluations (exceptions) are marked as infeasible with NaN values
        - Uses Cantera for chemistry (required)
        - Uses CoolProp for coolant properties (required)
        - Can be time-consuming for large grids (each point runs full cycle analysis)
        - Typical sweep: 5-10 pressure points × 7-10 O/F points
        
    Example:
        >>> p0_array = np.linspace(10e6, 30e6, 5)  # 10-30 MPa
        >>> OF_array = np.linspace(2.5, 4.0, 7)   # O/F 2.5-4.0
        >>> results = sweep_ffsc_feasibility(
        ...     F_vac=500e3, r_t=0.10, eps=15.0, L_noz=1.2,
        ...     p0_array=p0_array, OF_array=OF_array
        ... )
        >>> # Plot feasibility map
        >>> import matplotlib.pyplot as plt
        >>> plt.contourf(results['OF_grid'], results['p0_grid']/1e6,
        ...              results['feasible'].astype(float))
        >>> plt.xlabel('O/F'); plt.ylabel('Chamber Pressure (MPa)')
    """
    # ==================== Setup Parameter Grids ====================
    # Convert input arrays to numpy arrays
    p0_array = np.array(p0_array, dtype=float)
    OF_array = np.array(OF_array, dtype=float)
    
    # Create 2D meshgrid for all parameter combinations
    # indexing="ij" gives shape (len(p0), len(OF))
    P0, OFG = np.meshgrid(p0_array, OF_array, indexing="ij")

    # ==================== Allocate Result Arrays ====================
    # Initialize arrays to store results for each design point
    feasible = np.zeros_like(P0, dtype=bool)  # Overall feasibility flag
    coolant_frac_grid = np.zeros_like(P0, dtype=float)  # Coolant/fuel ratio
    T_wall_max_grid = np.zeros_like(P0, dtype=float)  # Maximum wall temperature
    fuel_ok = np.zeros_like(P0, dtype=bool)  # Fuel turbopump power balance
    ox_ok = np.zeros_like(P0, dtype=bool)  # Ox turbopump power balance
    Isp_eff_grid = np.zeros_like(P0, dtype=float)  # Effective specific impulse

    # ==================== Loop Over All Design Points ====================
    for i, p0 in enumerate(p0_array):
        for j, OF in enumerate(OF_array):
            # Try to evaluate this design point
            try:
                # Run complete FFSC cycle analysis
                summary = ffsc_full_flow_cycle(
                    F_vac=F_vac,
                    p0=p0,
                    OF=OF,
                    r_t=r_t,
                    eps=eps,
                    L_noz=L_noz,
                )
            except Exception:
                # If analysis fails (numerical issues, convergence failure, etc.),
                # mark as infeasible and continue to next point
                feasible[i, j] = False
                coolant_frac_grid[i, j] = np.nan
                T_wall_max_grid[i, j] = np.nan
                fuel_ok[i, j] = False
                ox_ok[i, j] = False
                Isp_eff_grid[i, j] = np.nan
                continue

            # ==================== Extract Results ====================
            # Store individual metrics from cycle analysis
            coolant_frac_grid[i, j] = summary["coolant_fraction"]
            T_wall_max_grid[i, j] = summary["T_wall_max"]
            fuel_ok[i, j] = summary["fuel_side_ok"]
            ox_ok[i, j] = summary["ox_side_ok"]
            Isp_eff_grid[i, j] = summary["Isp_vac_eff"]

            # ==================== Evaluate Feasibility ====================
            # Design point is feasible if ALL criteria are met:
            feasible[i, j] = (
                summary["fuel_side_ok"]  # 1. Fuel turbine provides enough power
                and summary["ox_side_ok"]  # 2. Ox turbine provides enough power
                and (summary["coolant_fraction"] < 0.6)  # 3. Reasonable coolant requirement
                and (summary["T_wall_max"] <= 900.0)  # 4. Wall temperature within material limits
            )

    # ==================== Return Complete Results ====================
    # Package all results into dictionary for plotting and analysis
    return dict(
        p0_grid=P0,  # Meshgrid of chamber pressures
        OF_grid=OFG,  # Meshgrid of O/F ratios
        feasible=feasible,  # Boolean feasibility map
        coolant_fraction=coolant_frac_grid,  # Coolant flow requirements
        T_wall_max=T_wall_max_grid,  # Wall temperature distribution
        fuel_side_ok=fuel_ok,  # Fuel pump power balance results
        ox_side_ok=ox_ok,  # Ox pump power balance results
        Isp_vac_eff=Isp_eff_grid,  # Performance map
    )
