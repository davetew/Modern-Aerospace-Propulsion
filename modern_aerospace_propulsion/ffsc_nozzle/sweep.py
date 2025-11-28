"""Parameter sweeps and feasibility mapping for FFSC designs."""

import numpy as np
import math

from .cycle import ffsc_full_flow_cycle


def ideal_vacuum_isp_from_eps(p0, T0, gamma, R_g, eps, g0=9.80665):
    """Compute an ideal vacuum Isp from expansion ratio and chamber state."""
    At = 1.0
    Ae = eps*At

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
    V_e = M_e*math.sqrt(gamma*R_g*T_e)
    return V_e/g0, None, T_e, M_e


def sweep_ffsc_feasibility(
        F_vac,
        r_t,
        eps,
        L_noz,
        p0_array,
        OF_array,
        g0=9.80665,
):
    """Sweep over chamber pressure and mixture ratio for FFSC feasibility."""
    p0_array = np.array(p0_array, dtype=float)
    OF_array = np.array(OF_array, dtype=float)
    P0, OFG = np.meshgrid(p0_array, OF_array, indexing="ij")

    feasible = np.zeros_like(P0, dtype=bool)
    coolant_frac_grid = np.zeros_like(P0, dtype=float)
    T_wall_max_grid = np.zeros_like(P0, dtype=float)
    fuel_ok = np.zeros_like(P0, dtype=bool)
    ox_ok = np.zeros_like(P0, dtype=bool)
    Isp_eff_grid = np.zeros_like(P0, dtype=float)

    for i, p0 in enumerate(p0_array):
        for j, OF in enumerate(OF_array):
            try:
                summary = ffsc_full_flow_cycle(
                    F_vac=F_vac,
                    p0=p0,
                    OF=OF,
                    r_t=r_t,
                    eps=eps,
                    L_noz=L_noz,
                    use_cantera_chamber=True,
                )
            except Exception:
                feasible[i, j] = False
                coolant_frac_grid[i, j] = np.nan
                T_wall_max_grid[i, j] = np.nan
                fuel_ok[i, j] = False
                ox_ok[i, j] = False
                Isp_eff_grid[i, j] = np.nan
                continue

            coolant_frac_grid[i, j] = summary["coolant_fraction"]
            T_wall_max_grid[i, j] = summary["T_wall_max"]
            fuel_ok[i, j] = summary["fuel_side_ok"]
            ox_ok[i, j] = summary["ox_side_ok"]
            Isp_eff_grid[i, j] = summary["Isp_vac_eff"]

            feasible[i, j] = (
                summary["fuel_side_ok"]
                and summary["ox_side_ok"]
                and (summary["coolant_fraction"] < 0.6)
                and (summary["T_wall_max"] <= 900.0)
            )

    return dict(
        p0_grid=P0,
        OF_grid=OFG,
        feasible=feasible,
        coolant_fraction=coolant_frac_grid,
        T_wall_max=T_wall_max_grid,
        fuel_side_ok=fuel_ok,
        ox_side_ok=ox_ok,
        Isp_vac_eff=Isp_eff_grid,
    )
