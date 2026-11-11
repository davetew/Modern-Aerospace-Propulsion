"""US Standard Atmosphere 1976, as a smooth table for jax.

0-86 km: the seven analytic layers (geopotential altitude, piecewise-linear temperature).
86-1000 km: log-linear interpolation of the published US76 densities / pressures.
Everything is exposed as a table (h, log rho, log p, T) and evaluated with jnp.interp,
so it is C0 in log(rho) and differentiable everywhere jax needs it.  The notebook used
the `ambiance` package (valid to 81 km); this reproduces it to <0.1 % below 81 km and
extends it.  Drag above ~100 km is negligible for ascent (q < 10 Pa) but the table keeps
the dynamics smooth to orbit instead of clipping.
"""
import numpy as np
import jax.numpy as jnp
from .constants import G0, R_AIR, GAMMA_AIR, P_SL

R0_GEOPOT = 6356766.0  # m, US76 effective Earth radius for geopotential altitude

# base geopotential altitudes [m], lapse rates [K/m], base temperatures [K]
_HB = np.array([0.0, 11000.0, 20000.0, 32000.0, 47000.0, 51000.0, 71000.0, 84852.0])
_LB = np.array([-6.5e-3, 0.0, 1.0e-3, 2.8e-3, 0.0, -2.8e-3, -2.0e-3])
_T0 = 288.15


def _layers():
    T = [_T0]
    p = [P_SL]
    for i in range(7):
        dH = _HB[i + 1] - _HB[i]
        if _LB[i] == 0.0:
            T.append(T[i])
            p.append(p[i] * np.exp(-G0 * dH / (R_AIR * T[i])))
        else:
            T.append(T[i] + _LB[i] * dH)
            p.append(p[i] * (T[i + 1] / T[i]) ** (-G0 / (R_AIR * _LB[i])))
    return np.array(T), np.array(p)


_TB, _PB = _layers()


def us76_below_86km(h_geom):
    """Vectorised analytic US76 for geometric altitude h [m] in [0, 86 km]. Returns T, p, rho."""
    h = np.asarray(h_geom, dtype=float)
    H = R0_GEOPOT * h / (R0_GEOPOT + h)
    idx = np.clip(np.searchsorted(_HB, H, side="right") - 1, 0, 6)
    Tb, pb, Lb, Hb = _TB[idx], _PB[idx], _LB[idx], _HB[idx]
    T = Tb + Lb * (H - Hb)
    with np.errstate(divide="ignore", invalid="ignore"):
        p_iso = pb * np.exp(-G0 * (H - Hb) / (R_AIR * Tb))
        p_grad = pb * np.where(Lb != 0.0, (T / Tb) ** (-G0 / (R_AIR * np.where(Lb != 0.0, Lb, 1.0))), 1.0)
    p = np.where(Lb == 0.0, p_iso, p_grad)
    rho = p / (R_AIR * T)
    return T, p, rho


# Published US76 values above 86 km (geometric km : rho kg/m^3, p Pa, T K)
_UPPER = np.array([
    # h_km   rho          p          T
    [90.0,  3.416e-6,  1.836e-1, 186.87],
    [100.0, 5.604e-7,  3.201e-2, 195.08],
    [110.0, 9.708e-8,  7.104e-3, 240.00],
    [120.0, 2.222e-8,  2.538e-3, 360.00],
    [130.0, 8.152e-9,  1.250e-3, 469.27],
    [140.0, 3.831e-9,  7.203e-4, 559.63],
    [150.0, 2.076e-9,  4.542e-4, 634.39],
    [160.0, 1.233e-9,  3.040e-4, 696.29],
    [180.0, 5.194e-10, 1.527e-4, 790.07],
    [200.0, 2.541e-10, 8.474e-5, 854.56],
    [250.0, 6.073e-11, 2.476e-5, 941.33],
    [300.0, 1.916e-11, 8.770e-6, 976.01],
    [400.0, 2.803e-12, 1.452e-6, 995.83],
    [500.0, 5.215e-13, 3.024e-7, 999.24],
    [700.0, 3.070e-14, 2.000e-8, 999.97],
    [1000.0, 3.561e-15, 7.514e-9, 1000.00],
])


def build_table(dh_low=250.0):
    """Return (h [m], log rho, log p, T) arrays spanning -1 km to 1000 km."""
    h_lo = np.arange(-1000.0, 86000.0 + 1e-6, dh_low)
    T, p, rho = us76_below_86km(np.clip(h_lo, 0.0, 86000.0))
    # extend below sea level with the first layer (valid)
    neg = h_lo < 0
    if neg.any():
        H = h_lo[neg]
        T[neg] = _T0 + _LB[0] * H
        p[neg] = P_SL * (T[neg] / _T0) ** (-G0 / (R_AIR * _LB[0]))
        rho[neg] = p[neg] / (R_AIR * T[neg])
    h_hi = _UPPER[:, 0] * 1000.0
    h = np.concatenate([h_lo, h_hi])
    lrho = np.concatenate([np.log(rho), np.log(_UPPER[:, 1])])
    lp = np.concatenate([np.log(p), np.log(_UPPER[:, 2])])
    Tt = np.concatenate([T, _UPPER[:, 3]])
    return h, lrho, lp, Tt


class Atmosphere:
    """Table-based US76 usable from numpy (guess / verification) and jax (ODE)."""

    def __init__(self):
        self.h, self.lrho, self.lp, self.T = build_table()
        self._jh = jnp.asarray(self.h)
        self._jlrho = jnp.asarray(self.lrho)
        self._jlp = jnp.asarray(self.lp)
        self._jT = jnp.asarray(self.T)

    # ---- jax (for the ODE) ----
    def rho_j(self, h):
        return jnp.exp(jnp.interp(h, self._jh, self._jlrho))

    def p_j(self, h):
        return jnp.exp(jnp.interp(h, self._jh, self._jlp))

    def a_j(self, h):
        T = jnp.interp(h, self._jh, self._jT)
        return jnp.sqrt(GAMMA_AIR * R_AIR * T)

    # ---- numpy (for guess / verify) ----
    def rho(self, h):
        return np.exp(np.interp(h, self.h, self.lrho))

    def p(self, h):
        return np.exp(np.interp(h, self.h, self.lp))

    def a(self, h):
        return np.sqrt(GAMMA_AIR * R_AIR * np.interp(h, self.h, self.T))


ATM = Atmosphere()
