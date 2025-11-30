"""Thermodynamic and combustion helpers for the ffsc_nozzle package.

This module provides:
- Cantera-based routines for computing equilibrium main-chamber states.
- Cantera-based routines for computing preburner (fuel-rich / ox-rich)
    adiabatic flame temperatures and approximate cp values.

Note: Cantera is required for this module to function.

Module Summary:
- Functions:
    - ``cantera_chamber_state(OF, p0, T_fuel=110.0, T_ox=90.0, fuel_species='CH4', ox_species='O2', mech='gri30.yaml')``:
        Equilibrium chamber state (T0, gamma, R_g) and Cantera gas object.
    - ``cantera_preburner_state(fuel_name, ox_name, mech, p_pb, T_inlet, phi)``:
        Preburner adiabatic flame temperature, midpoint cp, and Cantera gas object.
"""

import math
import cantera as ct
from typing import Tuple


def cantera_chamber_state(
        OF: float,
        p0: float,
        T_fuel: float = 110.0,
        T_ox: float = 90.0,
        fuel_species: str = "CH4",
        ox_species: str = "O2",
        mech: str = "gri30.yaml",
) -> Tuple[float, float, float, ct.Solution]:
    """Compute equilibrium main combustion chamber state using Cantera.
    
    Performs chemical equilibrium calculation for rocket combustion chamber
    by mixing fuel and oxidizer streams at specified temperatures and O/F
    ratio. Uses constant enthalpy-pressure (HP) equilibrium to account
    for mixing and combustion energy release.
    
    Process:
    1. Convert mass-based O/F ratio to molar composition
    2. Mix streams at mass-weighted average temperature
    3. Equilibrate at constant enthalpy and pressure
    4. Extract thermodynamic properties (T, gamma, R_g)

    Args:
        OF: Oxidizer-to-fuel mass ratio (dimensionless, kg_ox/kg_fuel)
        p0: Chamber pressure (Pa)
        T_fuel: Fuel inlet temperature (K), default 110.0 (LCH4 storage temp)
        T_ox: Oxidizer inlet temperature (K), default 90.0 (LOX storage temp)
        fuel_species: Cantera species name for fuel, default "CH4"
        ox_species: Cantera species name for oxidizer, default "O2"
        mech: Cantera mechanism file path, default "gri30.yaml"

    Returns:
        Tuple of (T0, gamma, R_g, gas):
            - T0: Equilibrium chamber stagnation temperature (K)
            - gamma: Ratio of specific heats cp/cv (dimensionless)
            - R_g: Specific gas constant for mixture (J/kg/K)
            - gas: Cantera Solution object at equilibrium state
            
    Note:
        - Uses GRI-Mech 3.0 by default (53 species, 325 reactions)
        - HP equilibrium accounts for enthalpy of mixing and reaction
        - gamma and R_g are computed at equilibrium composition
        - Typical CH4/O2 results: T0 ~ 3400-3700 K, gamma ~ 1.2-1.25
    
    Example:
        >>> T0, gamma, R_g, gas = cantera_chamber_state(
        ...     OF=3.2, p0=20e6, T_fuel=110.0, T_ox=90.0
        ... )
        >>> print(f"Chamber: {T0:.0f} K, gamma={gamma:.3f}")
    """
    # Load Cantera mechanism (chemical species and reactions database)
    gas = ct.Solution(mech)

    # ==================== Convert Mass Ratio to Molar Composition ====================
    # Get molecular weights for fuel and oxidizer species
    MW_f = gas.molecular_weights[gas.species_index(fuel_species)]  # kg/kmol
    MW_ox = gas.molecular_weights[gas.species_index(ox_species)]   # kg/kmol

    # Use unit mass of fuel as basis for calculation
    m_f = 1.0  # kg fuel
    m_ox = OF * m_f  # kg oxidizer (from O/F ratio)
    
    # Convert masses to moles: n = m / MW
    n_f = m_f / MW_f    # kmol fuel
    n_ox = m_ox / MW_ox  # kmol oxidizer

    # Create molar composition dictionary for Cantera
    comp = {fuel_species: n_f, ox_species: n_ox}

    # ==================== Compute Mixed Inlet Temperature ====================
    # Mass-weighted average of fuel and oxidizer inlet temperatures
    T_mix = (m_f*T_fuel + m_ox*T_ox) / (m_f + m_ox)

    # ==================== Equilibrate at Chamber Conditions ====================
    # Set initial state: Temperature, Pressure, and mole fractions (X)
    gas.TPX = T_mix, p0, comp
    
    # Equilibrate at constant enthalpy (H) and pressure (P)
    # This accounts for heat release from combustion
    gas.equilibrate("HP")

    # ==================== Extract Equilibrium Properties ====================
    T0 = gas.T  # Equilibrium temperature (K)
    cp = gas.cp_mass  # Specific heat at constant pressure (J/kg/K)
    cv = gas.cv_mass  # Specific heat at constant volume (J/kg/K)
    gamma = cp / cv  # Ratio of specific heats (dimensionless)
    
    # Specific gas constant: R_g = R_universal / MW_mixture
    R_g = ct.gas_constant / gas.mean_molecular_weight  # J/kg/K

    return T0, gamma, R_g, gas


def cantera_preburner_state(
        fuel_name: str,
        ox_name: str,
        mech: str,
        p_pb: float,
        T_inlet: float,
        phi: float,
) -> Tuple[float, float, ct.Solution]:
    """Compute preburner adiabatic flame temperature using Cantera.
    
    Calculates equilibrium combustion products for fuel-rich or oxidizer-rich
    preburners used in staged combustion rocket cycles. Preburners operate
    off-stoichiometric to limit temperature and provide turbine working fluid.
    
    Equivalence Ratio Definition:
    - φ = 1.0: Stoichiometric combustion (complete reaction)
    - φ > 1.0: Fuel-rich (excess fuel, cooler flame, reducing environment)
    - φ < 1.0: Oxidizer-rich (excess oxidizer, cooler flame, oxidizing environment)
    
    FFSC Application:
    - Fuel preburner: φ ~ 1.3 (fuel-rich) drives fuel turbopump
    - Oxidizer preburner: φ ~ 0.7 (ox-rich) drives oxidizer turbopump
    - Hot gases from both preburners mix in main chamber

    Args:
        fuel_name: Cantera species name for fuel (e.g., "CH4")
        ox_name: Cantera species name for oxidizer (e.g., "O2")
        mech: Cantera mechanism file path (e.g., "gri30.yaml")
        p_pb: Preburner pressure (Pa)
        T_inlet: Inlet temperature for both streams (K)
        phi: Equivalence ratio (dimensionless)
            - phi > 1.0 for fuel-rich preburner
            - phi < 1.0 for oxidizer-rich preburner

    Returns:
        Tuple of (T_ad, cp_mid, gas):
            - T_ad: Adiabatic flame temperature at equilibrium (K)
            - cp_mid: Specific heat at constant pressure at T_mid = (T_inlet + T_ad)/2 (J/kg/K)
            - gas: Cantera Solution object at equilibrium state
            
    Note:
        - Uses HP (constant enthalpy-pressure) equilibrium
        - cp_mid is evaluated at midpoint temperature for turbine power calculation
        - For CH4/O2 at φ=1.3: T_ad ~ 2700 K (fuel-rich)
        - For CH4/O2 at φ=0.7: T_ad ~ 2600 K (oxidizer-rich)
        - Stoichiometry assumes CH4 + 2 O2 -> CO2 + 2 H2O
        
    Example:
        >>> T_ad, cp, gas = cantera_preburner_state(
        ...     fuel_name="CH4", ox_name="O2", mech="gri30.yaml",
        ...     p_pb=20e6, T_inlet=110.0, phi=1.3
        ... )
        >>> print(f"Fuel-rich preburner: {T_ad:.0f} K")
    """
    # Load Cantera mechanism
    gas = ct.Solution(mech)

    # ==================== Compute Molar Composition from Equivalence Ratio ====================
    # Equivalence ratio: φ = (F/O)_actual / (F/O)_stoichiometric
    
    if fuel_name == "CH4" and ox_name == "O2":
        # Stoichiometry: CH4 + 2 O2 -> CO2 + 2 H2O
        # For stoichiometric: n_CH4 = 1.0, n_O2 = 2.0
        if phi >= 1.0:
            # Fuel-rich: excess fuel, use stoichiometric O2
            n_ox = 2.0
            n_fuel = phi * 1.0  # Scale up fuel by φ
        else:
            # Oxidizer-rich: excess oxidizer, use stoichiometric fuel
            n_ox = 2.0 / phi  # Scale up oxidizer by 1/φ
            n_fuel = 1.0
    else:
        # Generic fuel/oxidizer pair: assume 1:1 stoichiometry
        if phi >= 1.0:
            # Fuel-rich
            n_ox = 1.0
            n_fuel = phi * 1.0
        else:
            # Oxidizer-rich
            n_ox = 1.0 / phi
            n_fuel = 1.0

    # Create molar composition dictionary
    comp = {fuel_name: n_fuel, ox_name: n_ox}
    
    # ==================== Equilibrate Preburner ====================
    # Set initial state at inlet temperature and preburner pressure
    gas.TPX = T_inlet, p_pb, comp
    
    # Equilibrate at constant enthalpy-pressure (adiabatic combustion)
    gas.equilibrate("HP")

    # ==================== Extract Adiabatic Flame Temperature ====================
    T_ad = gas.T  # Equilibrium temperature (K)
    
    # ==================== Compute cp at Midpoint Temperature ====================
    # For turbine power calculation, evaluate properties at average temperature
    # between inlet and adiabatic flame temperature
    T_mid = 0.5*(T_inlet + T_ad)
    gas.TP = T_mid, p_pb  # Set state to midpoint
    cp_mid = gas.cp_mass  # Specific heat at midpoint (J/kg/K)

    return T_ad, cp_mid, gas
