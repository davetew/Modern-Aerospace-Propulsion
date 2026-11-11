"""Earth-Moon patched-conic transfers.

Two levels of fidelity:

1. ``lunar_hohmann_estimate`` -- coplanar Hohmann ellipse from the parking
   orbit to the Moon's orbital radius, Moon treated as a point at apogee; the
   lunar-orbit-insertion (LOI) burn uses the v_inf = |v_moon - v_apogee|.
   Gives the textbook TLI ~3.1 km/s and LOI ~0.8-0.9 km/s numbers.

2. ``lunar_patched_conic`` -- the Bate-Mueller-White (§7.4) patched-conic
   method: geocentric conic from injection (r0, v0, phi0) to the lunar sphere
   of influence, patched to a selenocentric hyperbola.  The arrival geometry
   is set by lambda1 (angle at the Moon between the Moon-Earth line and the
   arrival point on the SOI).  Implemented vectorially (no sign ambiguities);
   the Moon's orbit is taken circular with radius D = 384,400 km and the
   sidereal mean motion.  ``design_lunar_transfer`` solves for lambda1 (given
   v0) or v0 (given lambda1) that produces a requested perilune altitude.

Units: km, s; angles in radians internally (``*_deg`` arguments in degrees).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import brentq

from .bodies import EARTH, MOON
from .kepler import (circular_velocity, vis_viva, period, coe_from_rv, time_since_periapsis,
                     hyperbolic_periapsis_speed)
from .patched_conics import capture_dv, CaptureResult

D_MOON = MOON.a_mean                               # km
OMEGA_MOON = 2.0 * math.pi / MOON.rotation_period  # rad/s, sidereal mean motion (27.3217 d)
V_MOON = OMEGA_MOON * D_MOON                       # km/s (~1.023)
R_SOI_MOON = MOON.soi                              # km (~66,200)


# ---------------------------------------------------------------------------
# Level 1: Hohmann-to-the-Moon estimate
# ---------------------------------------------------------------------------
@dataclass
class LunarHohmannEstimate:
    r0: float
    v_park: float
    v_tli: float
    dv_tli: float
    tof: float
    v_apogee: float
    vinf_moon: float
    loi: CaptureResult

    @property
    def dv_total(self) -> float:
        return self.dv_tli + self.loi.dv

    def summary(self) -> str:
        return (f"Lunar Hohmann estimate: parking r0={self.r0:.0f} km, TLI dv={self.dv_tli:.4f} km/s "
                f"(v0={self.v_tli:.4f} km/s), TOF={self.tof/86400:.2f} d, v_inf at Moon={self.vinf_moon:.4f} km/s, "
                f"LOI dv={self.loi.dv:.4f} km/s (into {self.loi.rp-MOON.radius:.0f}"
                + (f" x {self.loi.ra-MOON.radius:.0f}" if self.loi.ra != self.loi.rp else "")
                + f" km), total={self.dv_total:.4f} km/s")


def lunar_hohmann_estimate(park_alt: float = 185.0, lunar_peri_alt: float = 100.0,
                           lunar_apo_alt: Optional[float] = None,
                           r_moon: float = D_MOON) -> LunarHohmannEstimate:
    r0 = EARTH.radius + park_alt
    a_t = 0.5 * (r0 + r_moon)
    v_park = circular_velocity(EARTH.mu, r0)
    v_tli = vis_viva(EARTH.mu, r0, a_t)
    v_apo = vis_viva(EARTH.mu, r_moon, a_t)
    v_m = math.sqrt(EARTH.mu / r_moon) if r_moon != D_MOON else V_MOON
    vinf = abs(v_m - v_apo)
    loi = capture_dv(MOON, vinf, lunar_peri_alt, lunar_apo_alt)
    return LunarHohmannEstimate(r0, v_park, v_tli, v_tli - v_park, 0.5 * period(EARTH.mu, a_t),
                                v_apo, vinf, loi)


# ---------------------------------------------------------------------------
# Level 2: BMW patched conic
# ---------------------------------------------------------------------------
@dataclass
class LunarPatchedConic:
    # inputs
    r0: float
    v0: float
    phi0: float
    lambda1: float
    # geocentric transfer
    a_geo: float
    e_geo: float
    nu0: float
    nu1: float
    r1: float
    v1: float
    phi1: float
    gamma1: float
    tof: float
    gamma0: float             # Moon lead angle at injection [rad]
    # selenocentric hyperbola
    v2: np.ndarray            # km/s, selenocentric arrival velocity vector (x along Moon->Earth? see geometry)
    vinf: float               # |v2| at SOI (used as v_inf of the hyperbola, BMW convention)
    energy2: float
    h2: float                 # signed: >0 counterclockwise about the Moon (same sense as the Moon's orbit)
    e2: float
    rp: float                 # km, perilune radius
    vp: float                 # km/s, perilune speed
    hits_moon: bool
    tof_soi: float = 0.0      # s, time from SOI entry to perilune on the selenocentric hyperbola

    @property
    def tof_total(self) -> float:
        """Injection to perilune: geocentric leg + selenocentric leg [s]."""
        return self.tof + self.tof_soi

    @property
    def perilune_alt(self) -> float:
        return self.rp - MOON.radius

    @property
    def dv_tli(self) -> float:
        """Injection dv from a circular parking orbit at r0 (only exact if phi0 = 0)."""
        v_c = circular_velocity(EARTH.mu, self.r0)
        return math.sqrt(self.v0 ** 2 + v_c ** 2 - 2 * self.v0 * v_c * math.cos(self.phi0))

    @property
    def direction(self) -> str:
        return "prograde (counterclockwise) about the Moon" if self.h2 > 0 else "retrograde (clockwise) about the Moon"

    def loi_dv(self, peri_alt: Optional[float] = None, apo_alt: Optional[float] = None) -> float:
        """LOI burn at perilune into an orbit with the given apoapsis altitude
        (None -> circular at the perilune altitude)."""
        rp = self.rp if peri_alt is None else MOON.radius + peri_alt
        ra = rp if apo_alt is None else MOON.radius + apo_alt
        v_orb = vis_viva(MOON.mu, rp, 0.5 * (rp + ra))
        v_hyp = math.sqrt(2.0 * (self.energy2 + MOON.mu / rp))
        return v_hyp - v_orb

    def summary(self) -> str:
        return (f"Lunar patched conic: r0={self.r0:.0f} km, v0={self.v0:.4f} km/s, phi0={math.degrees(self.phi0):.2f} deg, "
                f"lambda1={math.degrees(self.lambda1):.2f} deg\n"
                f"  geocentric: a={self.a_geo:.0f} km, e={self.e_geo:.4f}, TOF to SOI={self.tof/3600:.2f} h "
                f"(+{self.tof_soi/3600:.2f} h SOI->perilune = {self.tof_total/3600:.2f} h), "
                f"arrival r1={self.r1:.0f} km, v1={self.v1:.4f} km/s, phi1={math.degrees(self.phi1):.2f} deg, "
                f"Moon lead angle at TLI={math.degrees(self.gamma0):.2f} deg\n"
                f"  selenocentric: v_inf={self.vinf:.4f} km/s, e={self.e2:.3f}, perilune alt={self.perilune_alt:.1f} km, "
                f"v_perilune={self.vp:.4f} km/s, {self.direction}"
                + ("  ** IMPACT **" if self.hits_moon else "")
                + f"\n  TLI dv={self.dv_tli:.4f} km/s, LOI (circular at perilune)={self.loi_dv():.4f} km/s")


def lunar_patched_conic(r0: float, v0: float, phi0: float, lambda1: float,
                        r_soi: float = R_SOI_MOON) -> LunarPatchedConic:
    """BMW patched-conic lunar transfer (vector implementation).

    Geometry (plane of the Moon's orbit, Earth at origin): the Moon is at
    (D, 0) at arrival moving in +y; the arrival point on the SOI is
    P = Moon + R_soi*(-cos lambda1, sin lambda1), i.e. lambda1 > 0 puts the
    arrival point on the Moon's leading side.  The spacecraft moves
    counterclockwise (prograde) about the Earth.

    r0 [km], v0 [km/s], phi0 [rad] flight-path angle at injection, lambda1 [rad].
    """
    mu = EARTH.mu
    energy = 0.5 * v0 * v0 - mu / r0
    if energy >= 0.0:
        raise ValueError("injection energy >= 0: geocentric transfer is not an ellipse (unsupported here)")
    h = r0 * v0 * math.cos(phi0)
    # arrival point
    P = np.array([D_MOON - r_soi * math.cos(lambda1), r_soi * math.sin(lambda1), 0.0])
    r1 = float(np.linalg.norm(P))
    v1_sq = 2.0 * (energy + mu / r1)
    if v1_sq <= 0.0:
        raise ValueError("transfer orbit does not reach the lunar sphere of influence (apogee too low)")
    v1 = math.sqrt(v1_sq)
    cphi1 = h / (r1 * v1)
    if abs(cphi1) > 1.0:
        raise ValueError("inconsistent geometry (|cos phi1| > 1)")
    phi1 = math.acos(cphi1)          # outbound: positive flight-path angle
    gamma1 = math.atan2(P[1], P[0])
    rhat = P / r1
    that = np.array([-rhat[1], rhat[0], 0.0])   # counterclockwise tangential
    v1_vec = v1 * (math.sin(phi1) * rhat + math.cos(phi1) * that)
    v_moon = np.array([0.0, V_MOON, 0.0])
    # selenocentric
    rho = P - np.array([D_MOON, 0.0, 0.0])
    v2 = v1_vec - v_moon
    vinf = float(np.linalg.norm(v2))
    energy2 = 0.5 * vinf * vinf - MOON.mu / r_soi
    h2 = float(np.cross(rho, v2)[2])
    p2 = h2 * h2 / MOON.mu
    e2 = math.sqrt(1.0 + 2.0 * energy2 * h2 * h2 / MOON.mu ** 2)
    rp = p2 / (1.0 + e2)
    vp = math.sqrt(2.0 * (energy2 + MOON.mu / rp))
    # geocentric conic details and time of flight
    r0_vec = np.array([r0, 0.0, 0.0])
    v0_vec = v0 * np.array([math.sin(phi0), math.cos(phi0), 0.0])
    coe = coe_from_rv(r0_vec, v0_vec, mu)
    a, e = coe.a, coe.e
    nu0 = coe.nu if coe.nu <= math.pi else coe.nu - 2 * math.pi
    p = a * (1 - e * e)
    cnu1 = (p / r1 - 1.0) / e
    nu1 = math.acos(max(-1.0, min(1.0, cnu1)))       # outbound branch
    tof = time_since_periapsis(mu, a, e, nu1) - time_since_periapsis(mu, a, e, nu0)
    gamma0 = (nu1 - nu0) - gamma1 - OMEGA_MOON * tof
    # selenocentric time from SOI entry to perilune (hyperbolic Kepler equation)
    a_h = MOON.mu / (2.0 * energy2)                      # |a| of the hyperbola
    cosh_F = (1.0 + r_soi / a_h) / e2
    F = math.acosh(max(1.0, cosh_F))
    tof_soi = (e2 * math.sinh(F) - F) / math.sqrt(MOON.mu / a_h ** 3)
    return LunarPatchedConic(r0, v0, phi0, lambda1, a, e, nu0, nu1, r1, v1, phi1, gamma1, tof,
                             gamma0, v2, vinf, energy2, h2, e2, rp, vp, rp < MOON.radius, tof_soi)


def _rp_signed(r0, v0, phi0, lam):
    s = lunar_patched_conic(r0, v0, phi0, lam)
    return s.rp if s.h2 > 0 else -s.rp   # sign encodes the direction about the Moon


def design_lunar_transfer(park_alt: float = 185.0, perilune_alt: float = 100.0,
                          v0: Optional[float] = None, lambda1_deg: Optional[float] = None,
                          phi0_deg: float = 0.0, direction: str = "either",
                          lambda_range_deg: tuple[float, float] = (0.5, 89.5),
                          n_scan: int = 360) -> list[LunarPatchedConic]:
    """Find patched-conic transfers reaching a given perilune altitude.

    Give either ``v0`` (injection speed, km/s; lambda1 is solved) or
    ``lambda1_deg`` (v0 is solved).  ``direction`` filters solutions by the
    sense of motion about the Moon: 'prograde', 'retrograde' or 'either'.
    Returns all solutions found (usually one per direction).
    """
    r0 = EARTH.radius + park_alt
    phi0 = math.radians(phi0_deg)
    target = MOON.radius + perilune_alt
    sols: list[LunarPatchedConic] = []

    def accept(s: LunarPatchedConic) -> bool:
        if direction == "prograde":
            return s.h2 > 0
        if direction == "retrograde":
            return s.h2 < 0
        return True

    if v0 is not None and lambda1_deg is None:
        lams = np.radians(np.linspace(*lambda_range_deg, n_scan))
        vals = []
        for lam in lams:
            try:
                vals.append(_rp_signed(r0, v0, phi0, float(lam)))
            except ValueError:
                vals.append(math.nan)
        vals = np.array(vals)
        for sign in (+1.0, -1.0):
            f = sign * vals - target
            for k in range(len(lams) - 1):
                if np.isnan(f[k]) or np.isnan(f[k + 1]):
                    continue
                if f[k] == 0.0 or f[k] * f[k + 1] < 0.0:
                    try:
                        lam_root = brentq(lambda L: sign * _rp_signed(r0, v0, phi0, L) - target,
                                          float(lams[k]), float(lams[k + 1]), xtol=1e-10)
                    except ValueError:
                        continue
                    s = lunar_patched_conic(r0, v0, phi0, lam_root)
                    if accept(s):
                        sols.append(s)
    elif lambda1_deg is not None and v0 is None:
        lam = math.radians(lambda1_deg)
        v_min = _min_injection_speed(r0, phi0, lam)
        v_max = math.sqrt(2.0 * EARTH.mu / r0) * 0.9999
        vs = np.linspace(v_min * 1.0001, v_max, n_scan)
        vals = []
        for v in vs:
            try:
                vals.append(_rp_signed(r0, float(v), phi0, lam))
            except ValueError:
                vals.append(math.nan)
        vals = np.array(vals)
        for sign in (+1.0, -1.0):
            f = sign * vals - target
            for k in range(len(vs) - 1):
                if np.isnan(f[k]) or np.isnan(f[k + 1]):
                    continue
                if f[k] == 0.0 or f[k] * f[k + 1] < 0.0:
                    try:
                        v_root = brentq(lambda V: sign * _rp_signed(r0, V, phi0, lam) - target,
                                        float(vs[k]), float(vs[k + 1]), xtol=1e-10)
                    except ValueError:
                        continue
                    s = lunar_patched_conic(r0, v_root, phi0, lam)
                    if accept(s):
                        sols.append(s)
    else:
        raise ValueError("specify exactly one of v0 or lambda1_deg")
    # sort: shortest TOF first
    sols.sort(key=lambda s: s.tof)
    return sols


def _min_injection_speed(r0: float, phi0: float, lam: float) -> float:
    """Smallest v0 whose geocentric conic just reaches the SOI arrival point."""
    P = np.array([D_MOON - R_SOI_MOON * math.cos(lam), R_SOI_MOON * math.sin(lam)])
    r1 = float(np.linalg.norm(P))
    # need energy + mu/r1 >= 0 and h consistent: for phi0 = 0 the apogee must reach r1
    # general: solve v0 such that v1 = h/(r1) i.e. cos(phi1)=1  ->  2(e + mu/r1) = (r0 v0 cos phi0 / r1)^2
    mu = EARTH.mu
    c = math.cos(phi0)
    # v0^2 (1 - (r0 c / r1)^2) = 2 mu (1/r0 - 1/r1)
    return math.sqrt(2.0 * mu * (1.0 / r0 - 1.0 / r1) / (1.0 - (r0 * c / r1) ** 2))


def tof_for_injection_speed(park_alt: float, v0: float, lambda1_deg: float = 30.0) -> float:
    """Convenience: time of flight [s] to the lunar SOI for a given injection speed."""
    r0 = EARTH.radius + park_alt
    return lunar_patched_conic(r0, v0, 0.0, math.radians(lambda1_deg)).tof
