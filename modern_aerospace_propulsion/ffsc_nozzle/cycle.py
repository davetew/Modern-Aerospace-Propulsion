"""Full-flow staged combustion (FFSC) cycle model with regen-cooled nozzle."""

import math

from .regen import simple_conical_nozzle, regen_nozzle_1D, make_coolprop_liquid
from .thermo import HAVE_CANTERA, ideal_gas_chamber_state, cantera_chamber_state, cantera_preburner_state

def rho_LCH4(T=110.0):
    """Return a representative density for liquid methane [kg/m^3]."""
    return 420.0

def rho_LOX(T=90.0):
    """Return a representative density for liquid oxygen [kg/m^3]."""
    return 1140.0


def pump_power(m_dot, rho, dp, eta_pump):
    """Compute pump shaft power [W]."""
    return m_dot*dp/(rho*eta_pump)


def turbine_power(m_dot, cp_g, T_in, T_out, eta_turb):
    """Compute turbine shaft power [W] for an expanding gas."""
    if T_in <= T_out:
        return 0.0
    return m_dot*cp_g*(T_in - T_out)*eta_turb


def solve_preburner_T_for_power(
        P_req, m_dot, cp_g, T_out,
        T_min, T_max, eta_turb=0.9,
        tol=1e-3, max_iter=60):
    """Solve for turbine inlet T that delivers a target power."""
    lo, hi = T_min, T_max
    for _ in range(max_iter):
        mid = 0.5*(lo + hi)
        P_mid = turbine_power(m_dot, cp_g, mid, T_out, eta_turb)
        if abs(P_mid - P_req)/max(P_req, 1.0) < tol:
            return mid, P_mid
        if P_mid < P_req:
            lo = mid
        else:
            hi = mid
    mid = 0.5*(lo + hi)
    return mid, turbine_power(m_dot, cp_g, mid, T_out, eta_turb)


def ffsc_full_flow_cycle(
        F_vac,
        p0,
        OF,
        r_t,
        eps,
        L_noz,
        coolant="LCH4",
        use_coolprop=True,
        n_channels=120,
        w_channel=0.0015,
        wall_thickness=0.0020,
        roughness=5e-6,
        T_cool_in=110.0,
        p_cool_in=18e6,
        T_wall_limit=900.0,
        dp_limit=5e6,
        injector_dp_frac=0.2,
        feed_line_dp=1.0e6,
        eta_pump_fuel=0.70,
        eta_pump_ox=0.70,
        eta_turb_fuel=0.90,
        eta_turb_ox=0.90,
        eta_mech=0.98,
        cp_pb_fuel=4000.0,
        cp_pb_ox=2500.0,
        T_pb_min=900.0,
        T_pb_max=3500.0,
        T_turb_out=None,
        use_cantera_chamber=True,
        g0=9.80665,
):
    """Evaluate a single FFSC engine design point."""
    if HAVE_CANTERA and use_cantera_chamber:
        T0, gamma, R_g, gas_ch = cantera_chamber_state(
            OF=OF,
            p0=p0,
            T_fuel=T_cool_in,
            T_ox=90.0,
            fuel_species="CH4",
            ox_species="O2",
            mech="gri30.yaml",
        )
    else:
        T0, gamma, R_g = ideal_gas_chamber_state(OF=OF, p0=p0)
        gas_ch = None

    if T_turb_out is None:
        T_turb_out = T0

    x_noz, r_noz, A_noz, At = simple_conical_nozzle(r_t, eps, L_noz, N=250)

    if use_coolprop:
        try:
            coolant_props = make_coolprop_liquid("Methane")
        except RuntimeError:
            coolant_props = None
    else:
        coolant_props = None

    from .sweep import ideal_vacuum_isp_from_eps
    Isp_vac_ideal, _, T_e, M_e = ideal_vacuum_isp_from_eps(p0, T0, gamma, R_g, eps, g0)
    eta_Isp = 0.95
    Isp_vac_eff = eta_Isp*Isp_vac_ideal
    m_dot_total = F_vac/(Isp_vac_eff*g0)
    m_dot_fuel = m_dot_total/(1.0 + OF)
    m_dot_ox = m_dot_total - m_dot_fuel

    coolant_fraction_guess = 0.2
    m_dot_cool = coolant_fraction_guess*m_dot_fuel

    res_noz = regen_nozzle_1D(
        x=x_noz,
        r=r_noz,
        A=A_noz,
        At=At,
        p0=p0,
        T0=T0,
        gamma=gamma,
        R_g=R_g,
        coolant=coolant,
        coolant_props=coolant_props,
        m_dot_cool=m_dot_cool,
        n_channels=n_channels,
        w_channel=w_channel,
        h_channel=0.0020,
        wall_thickness=wall_thickness,
        roughness=roughness,
        T_cool_in=T_cool_in,
        p_cool_in=p_cool_in,
        L_regen=x_noz[-1],
        T_env=3.0,
    )

    T_wall_max = res_noz["T_wall_inner"].max()
    dp_cool = res_noz["p_cool"][0] - res_noz["p_cool"][-1]

    dp_injector = injector_dp_frac*p0
    dp_fuel = dp_cool + dp_injector + feed_line_dp + (p0 - 0.1e6)
    dp_ox = dp_injector + feed_line_dp + (p0 - 0.1e6)

    rho_fuel = rho_LCH4(T_cool_in)
    rho_ox = rho_LOX()

    P_pump_fuel = pump_power(m_dot_fuel, rho_fuel, dp_fuel, eta_pump_fuel)
    P_pump_ox = pump_power(m_dot_ox, rho_ox, dp_ox, eta_pump_ox)

    P_turb_req_fuel = P_pump_fuel/eta_mech
    P_turb_req_ox = P_pump_ox/eta_mech

    if HAVE_CANTERA:
        phi_fuel = 1.3
        phi_ox = 0.7
        T_ad_fuel, cp_fuel_pb, gas_fb = cantera_preburner_state(
            fuel_name="CH4",
            ox_name="O2",
            mech="gri30.yaml",
            p_pb=p0,
            T_inlet=T_cool_in,
            phi=phi_fuel,
        )
        T_ad_ox, cp_ox_pb, gas_ob = cantera_preburner_state(
            fuel_name="CH4",
            ox_name="O2",
            mech="gri30.yaml",
            p_pb=p0,
            T_inlet=90.0,
            phi=phi_ox,
        )
        P_turb_avail_fuel = turbine_power(m_dot_fuel, cp_fuel_pb, T_ad_fuel, T_turb_out, eta_turb_fuel)
        P_turb_avail_ox = turbine_power(m_dot_ox, cp_ox_pb, T_ad_ox, T_turb_out, eta_turb_ox)
        T_pb_fuel = T_ad_fuel
        T_pb_ox = T_ad_ox
    else:
        T_pb_fuel, P_turb_avail_fuel = solve_preburner_T_for_power(
            P_req=P_turb_req_fuel,
            m_dot=m_dot_fuel,
            cp_g=cp_pb_fuel,
            T_out=T_turb_out,
            T_min=T_pb_min,
            T_max=T_pb_max,
            eta_turb=eta_turb_fuel,
        )
        T_pb_ox, P_turb_avail_ox = solve_preburner_T_for_power(
            P_req=P_turb_req_ox,
            m_dot=m_dot_ox,
            cp_g=cp_pb_ox,
            T_out=T_turb_out,
            T_min=T_pb_min,
            T_max=T_pb_max,
            eta_turb=eta_turb_ox,
        )

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
