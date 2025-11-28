"""Thermodynamic and combustion helpers for the ffsc_nozzle package.

This module provides:
- A simple ideal-gas fallback for main-chamber properties as a function
  of mixture ratio (O/F) and chamber pressure.
- Cantera-based routines for computing equilibrium main-chamber states.
- Cantera-based routines for computing preburner (fuel-rich / ox-rich)
  adiabatic flame temperatures and approximate cp values.

The functions are written so that the rest of the package can operate
in two modes:
- High‑fidelity mode (with Cantera installed)
- Lightweight / educational mode (without Cantera)
"""

import math

try:
    import cantera as ct
    HAVE_CANTERA = True
except ImportError:
    HAVE_CANTERA = False


def ideal_gas_chamber_state(OF, p0, fuel="CH4", ox="O2"):
    """Return a crude ideal‑gas estimate of main chamber properties.

    Parameters
    ----------
    OF : float
        Oxidizer-to-fuel mass ratio (O/F).
    p0 : float
        Chamber pressure in Pa.
    fuel : str, optional
        Fuel name (for documentation only). Defaults to "CH4".
    ox : str, optional
        Oxidizer name (for documentation only). Defaults to "O2".

    Returns
    -------
    T0 : float
        Approximate chamber temperature [K].
    gamma : float
        Ratio of specific heats cp/cv.
    R_g : float
        Effective gas constant [J/kg/K].
    """
    OF_stoich = 3.3
    T_peak = 3600.0
    T_min = 2900.0
    dOF = (OF - OF_stoich) / OF_stoich
    T0 = max(T_min, T_peak * (1.0 - 0.4*dOF*dOF))
    gamma = 1.25
    R_g = 355.0
    return T0, gamma, R_g


def cantera_chamber_state(
        OF,
        p0,
        T_fuel=110.0,
        T_ox=90.0,
        fuel_species="CH4",
        ox_species="O2",
        mech="gri30.yaml",
):
    """Compute an equilibrium main-chamber state using Cantera.

    Parameters
    ----------
    OF : float
        Oxidizer-to-fuel mass ratio (O/F) by mass.
    p0 : float
        Chamber pressure [Pa].
    T_fuel : float, optional
        Inlet temperature of the fuel stream [K].
    T_ox : float, optional
        Inlet temperature of the oxidizer stream [K].
    fuel_species : str, optional
        Cantera name of the fuel species (e.g., "CH4").
    ox_species : str, optional
        Cantera name of the oxidizer species (e.g., "O2").
    mech : str, optional
        Path to the Cantera mechanism file, such as "gri30.yaml".

    Returns
    -------
    T0 : float
        Equilibrium chamber temperature [K].
    gamma : float
        Ratio of specific heats cp/cv at the equilibrium state.
    R_g : float
        Effective gas constant [J/kg/K] for the mixture.
    gas : cantera.Solution
        Cantera gas object in the final equilibrium state.

    Raises
    ------
    RuntimeError
        If Cantera is not available.
    """
    if not HAVE_CANTERA:
        raise RuntimeError("Cantera not available; cannot compute chamber state.")

    gas = ct.Solution(mech)

    MW_f = gas.molecular_weights[gas.species_index(fuel_species)]
    MW_ox = gas.molecular_weights[gas.species_index(ox_species)]

    m_f = 1.0
    m_ox = OF * m_f
    n_f = m_f / MW_f
    n_ox = m_ox / MW_ox

    comp = {fuel_species: n_f, ox_species: n_ox}

    T_mix = (m_f*T_fuel + m_ox*T_ox) / (m_f + m_ox)

    gas.TPX = T_mix, p0, comp
    gas.equilibrate("HP")

    T0 = gas.T
    cp = gas.cp_mass
    cv = gas.cv_mass
    gamma = cp / cv
    R_g = ct.gas_constant / gas.mean_molecular_weight

    return T0, gamma, R_g, gas


def cantera_preburner_state(
        fuel_name: str,
        ox_name: str,
        mech: str,
        p_pb: float,
        T_inlet: float,
        phi: float,
):
    """Compute a preburner adiabatic state using Cantera.

    Parameters
    ----------
    fuel_name : str
        Cantera species name for the fuel (e.g., "CH4").
    ox_name : str
        Cantera species name for the oxidizer (e.g., "O2").
    mech : str
        Path to the Cantera mechanism file (e.g., "gri30.yaml").
    p_pb : float
        Preburner pressure [Pa].
    T_inlet : float
        Inlet temperature of both fluid streams [K].
    phi : float
        Equivalence ratio (>1 fuel-rich, <1 oxidizer-rich).

    Returns
    -------
    T_ad : float
        Adiabatic flame temperature [K].
    cp_mid : float
        Approximate cp at the midpoint temperature [J/kg/K].
    gas : cantera.Solution
        Gas object at the final equilibrium state.

    Raises
    ------
    RuntimeError
        If Cantera is not available.
    """
    if not HAVE_CANTERA:
        raise RuntimeError("Cantera not available. Install cantera to use this function.")

    gas = ct.Solution(mech)

    if fuel_name == "CH4" and ox_name == "O2":
        if phi >= 1.0:
            n_ox = 2.0
            n_fuel = phi * 1.0
        else:
            n_ox = 2.0 / phi
            n_fuel = 1.0
    else:
        if phi >= 1.0:
            n_ox = 1.0
            n_fuel = phi * 1.0
        else:
            n_ox = 1.0 / phi
            n_fuel = 1.0

    comp = {fuel_name: n_fuel, ox_name: n_ox}
    gas.TPX = T_inlet, p_pb, comp
    gas.equilibrate("HP")

    T_ad = gas.T
    T_mid = 0.5*(T_inlet + T_ad)
    gas.TP = T_mid, p_pb
    cp_mid = gas.cp_mass

    return T_ad, cp_mid, gas
