import math

try:
    import cantera as ct
    HAVE_CANTERA = True
except ImportError:
    HAVE_CANTERA = False


def ideal_gas_chamber_state(OF, p0, fuel="CH4", ox="O2"):
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
    R_g = gas.gas_constant / gas.mean_molecular_weight

    return T0, gamma, R_g, gas


def cantera_preburner_state(
        fuel_name: str,
        ox_name: str,
        mech: str,
        p_pb: float,
        T_inlet: float,
        phi: float,
):
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
