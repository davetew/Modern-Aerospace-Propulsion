"""1D regenerative nozzle cooling model.

This module provides a simple axisymmetric, quasi-1D model of a
regeneratively cooled rocket nozzle.
"""

import numpy as np
import math

from .thermo import HAVE_CANTERA  # currently unused, but kept for context

def props_LCH4_stub(T, p=1e7):
    """Return crude thermophysical properties for liquid methane."""
    rho = 420.0
    cp  = 3500.0
    k   = 0.19
    mu  = 1.3e-4
    return rho, cp, k, mu

COOLANTS_STUB = {
    "LCH4": props_LCH4_stub,
}

try:
    from CoolProp.CoolProp import PropsSI
    HAVE_COOLPROP = True
except ImportError:
    HAVE_COOLPROP = False


def make_coolprop_liquid(fluid_name):
    """Return a CoolProp-based property function for a given fluid."""
    if not HAVE_COOLPROP:
        raise RuntimeError("CoolProp not installed.")
    def props(T, p):
        rho = PropsSI("D", "T", T, "P", p, fluid_name)
        cp  = PropsSI("C", "T", T, "P", p, fluid_name)
        k   = PropsSI("L", "T", T, "P", p, fluid_name)
        mu  = PropsSI("V", "T", T, "P", p, fluid_name)
        return rho, cp, k, mu
    return props


def k_wall_copper(T):
    """Approximate thermal conductivity for a Cu-based alloy."""
    if T < 300.0:
        return 350.0
    elif T > 900.0:
        return 250.0
    else:
        return 350.0 - (T-300.0)*(100.0/600.0)


def isentropic_M_from_area_ratio(Ar, gamma):
    """Solve the supersonic Mach number from an area ratio A/A*."""
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


def gas_state_from_area(At, A, p0, T0, gamma, R_g):
    """Return static gas properties from chamber conditions and area."""
    Ar = A/At
    if Ar < 1.0:
        Ar = 1.0
    M = isentropic_M_from_area_ratio(Ar, gamma)
    T = T0 / (1.0 + 0.5*(gamma-1.0)*M**2)
    p = p0 * (T/T0)**(gamma/(gamma-1.0))
    return p, T, M


def bartz_h(p0, T0, r_t, r, At, A, gamma, R_g, mu_g=3e-5, Pr_g=0.9):
    """Compute a simplified Bartz-like gas-side heat transfer coefficient."""
    cp_g = gamma*R_g/(gamma-1.0)
    p, T, M = gas_state_from_area(At, A, p0, T0, gamma, R_g)
    C = 0.026
    factor = C * (mu_g**0.2) * (cp_g**0.6) / (Pr_g**0.6)
    h_g = factor * (p0**0.8) * (r_t**0.1) * (At/A)**0.9 * (T0/T)**0.55
    return max(h_g, 1e3), p, T, M


def friction_factor_haaland(Re, eps_over_D):
    """Return Darcy friction factor using the Haaland correlation."""
    if Re <= 0.0:
        return 0.0
    if Re < 2300.0:
        return 64.0/Re
    term = (eps_over_D/3.7)**1.11 + 6.9/Re
    return 1.0/(-1.8*math.log10(term))**2


def Nu_gnielinski(Re, Pr, f):
    """Return the Gnielinski Nusselt number for internal flow."""
    if Re < 2300.0:
        return 4.36
    return ((f/8.0)*(Re-1000.0)*Pr) / (1.0 + 12.7*math.sqrt(f/8.0)*(Pr**(2.0/3.0)-1.0))


def simple_conical_nozzle(r_t, expansion_ratio, L, N=200):
    """Build a simple conical nozzle geometry (throat to exit)."""
    r_e = r_t*math.sqrt(expansion_ratio)
    x = np.linspace(0.0, L, N)
    r = r_t + (r_e - r_t)*(x/L)
    A = math.pi*r**2
    At = math.pi*r_t**2
    return x, r, A, At


def regen_nozzle_1D(
        x, r, A, At,
        p0, T0, gamma, R_g,
        coolant="LCH4",
        coolant_props=None,
        m_dot_cool=5.0,
        n_channels=100,
        w_channel=0.0015,
        h_channel=0.0020,
        wall_thickness=0.0020,
        roughness=5e-6,
        T_cool_in=110.0,
        p_cool_in=1.0e7,
        L_regen=None,
        emissivity_ext=0.8,
        T_env=3.0,
        wall_k_func=k_wall_copper,
        max_iter_wall=10,
    ):
    """Integrate a 1D regenerative cooling solution along a nozzle."""
    N = len(x)
    dx = x[1] - x[0]
    if L_regen is None:
        L_regen = x[-1]

    if coolant_props is None:
        if coolant not in COOLANTS_STUB:
            raise ValueError(f"Unknown coolant {coolant} and no coolant_props.")
        props_cool = COOLANTS_STUB[coolant]
    else:
        props_cool = coolant_props

    A_ch = w_channel*h_channel
    P_wet = 2.0*(w_channel + h_channel)
    D_h = 4.0*A_ch/P_wet

    T_cool = np.zeros(N)
    p_cool = np.zeros(N)
    T_wall_inner = np.zeros(N)
    T_wall_outer = np.zeros(N)
    qpp = np.zeros(N)
    h_cool = np.zeros(N)
    h_gas = np.zeros(N)
    p_g = np.zeros(N)
    T_g = np.zeros(N)
    M_g = np.zeros(N)

    T_cool[0] = T_cool_in
    p_cool[0] = p_cool_in

    for i in range(N):
        r_i = r[i]
        A_i = A[i]
        perim_inner = 2.0*math.pi*r_i
        A_seg_inner = perim_inner*dx
        regen_active = (x[i] <= L_regen)

        hg_i, pgi, Tgi, Mi = bartz_h(
            p0=p0, T0=T0, r_t=math.sqrt(At/math.pi),
            r=r_i, At=At, A=A_i,
            gamma=gamma, R_g=R_g
        )
        h_gas[i] = hg_i
        p_g[i] = pgi
        T_g[i] = Tgi
        M_g[i] = Mi

        if not regen_active:
            sigma = 5.670374419e-8
            T_wall_inner[i] = 0.7*T_g[i]
            qpp[i] = emissivity_ext*sigma*(T_wall_inner[i]**4 - T_env**4)
            T_wall_outer[i] = T_wall_inner[i]
            if i < N-1:
                T_cool[i+1] = T_cool[i]
                p_cool[i+1] = p_cool[i]
            continue

        rho_c, cp_c, k_c, mu_c = props_cool(T_cool[i], p_cool[i])
        G = (m_dot_cool/n_channels)/A_ch
        v_c = G/rho_c
        Re = G*D_h/mu_c
        Pr = cp_c*mu_c/k_c
        eps_over_D = roughness/max(D_h, 1e-9)
        f = friction_factor_haaland(Re, eps_over_D)
        Nu_base = Nu_gnielinski(Re, Pr, f)

        T_wall_guess = T_cool[i] + 50.0
        for _ in range(max_iter_wall):
            _, _, _, mu_w = props_cool(T_wall_guess, p_cool[i])
            Nu_ST = Nu_base*(mu_c/mu_w)**0.14
            h_c = Nu_ST*k_c/D_h
            k_w = wall_k_func(T_wall_guess)
            R_tot = 1.0/hg_i + wall_thickness/k_w + 1.0/h_c
            qpp_i = (T_g[i] - T_cool[i])/R_tot
            T_wall_new = T_g[i] - qpp_i*(1.0/hg_i)
            if abs(T_wall_new - T_wall_guess) < 1e-3:
                T_wall_guess = T_wall_new
                break
            T_wall_guess = T_wall_new

        h_cool[i] = h_c
        qpp[i] = qpp_i
        T_wall_inner[i] = T_wall_guess
        T_wall_outer[i] = T_wall_inner[i] - qpp_i*(wall_thickness/wall_k_func(T_wall_inner[i]))

        Q_seg = qpp_i*A_seg_inner
        dT = Q_seg/(m_dot_cool*cp_c)
        if i < N-1:
            T_cool[i+1] = T_cool[i] + dT

        dp_seg = f*(dx/D_h)*0.5*rho_c*v_c**2
        if i < N-1:
            p_cool[i+1] = p_cool[i] - dp_seg

    return dict(
        x=x, r=r, A=A, At=At,
        T_cool=T_cool, p_cool=p_cool,
        T_wall_inner=T_wall_inner, T_wall_outer=T_wall_outer,
        qpp=qpp, h_cool=h_cool, h_gas=h_gas,
        p_g=p_g, T_g=T_g, M_g=M_g,
        D_h=D_h,
    )
