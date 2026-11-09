"""Patched-conic interplanetary mission design.

* planetocentric departure (parking orbit -> hyperbola with given v_inf)
* planetocentric capture (hyperbola -> circular/elliptical orbit)
* unpowered and powered gravity-assist flybys
* heliocentric legs via Lambert between ephemeris positions
* multi-leg sequences (Earth -> [flybys] -> target) with a Delta-v budget

Units: km, s, rad unless a parameter name says otherwise (``*_alt`` in km,
``*_deg`` in degrees, epochs as datetime/ISO/JD).

References: Sutton Ch. 4; Curtis Ch. 8; Bate, Mueller & White Ch. 8.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy.optimize import brentq

from .bodies import Body, get_body, MU_SUN
from .ephemeris import body_state, to_jd, jd_to_iso, Epoch
from .kepler import (hyperbolic_periapsis_speed, hyperbolic_eccentricity, turn_angle,
                     periapsis_for_turn_angle, circular_velocity, vis_viva, angle_between,
                     unit, aiming_radius)
from .lambert import lambert, LambertSolution, LambertError


# ---------------------------------------------------------------------------
# Departure
# ---------------------------------------------------------------------------
@dataclass
class DepartureResult:
    body: str
    r_park: float          # km, radius of the parking-orbit periapsis / burn point
    v_park: float          # km/s, speed in parking orbit at the burn point
    vinf: float            # km/s
    c3: float              # km^2/s^2
    v_hyp: float           # km/s, hyperbolic periapsis speed
    dv: float              # km/s
    e_hyp: float
    beta: float            # rad, angle from periapsis to the asymptote direction

    def summary(self) -> str:
        return (f"Departure from {self.body}: v_inf={self.vinf:.3f} km/s (C3={self.c3:.2f} km2/s2), "
                f"parking r={self.r_park:.0f} km, dv={self.dv:.4f} km/s")


def departure_dv(body, vinf: float, park_alt: float, park_apo_alt: Optional[float] = None) -> DepartureResult:
    """Delta-v to inject from a parking orbit (circular at ``park_alt`` km, or
    elliptical with periapsis ``park_alt`` and apoapsis ``park_apo_alt``) onto
    a departure hyperbola with hyperbolic excess speed ``vinf`` [km/s].

    The burn is tangential at periapsis (the optimum).
    """
    b = get_body(body)
    rp = b.radius + park_alt
    if park_apo_alt is None:
        v_park = circular_velocity(b.mu, rp)
    else:
        ra = b.radius + park_apo_alt
        v_park = vis_viva(b.mu, rp, 0.5 * (rp + ra))
    v_hyp = hyperbolic_periapsis_speed(b.mu, rp, vinf)
    e = hyperbolic_eccentricity(b.mu, rp, vinf)
    beta = math.acos(1.0 / e)
    return DepartureResult(b.name, rp, v_park, vinf, vinf * vinf, v_hyp, v_hyp - v_park, e, beta)


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------
@dataclass
class CaptureResult:
    body: str
    vinf: float
    rp: float              # km, periapsis radius of the capture orbit
    ra: float              # km, apoapsis radius (== rp for circular)
    v_hyp: float
    v_orbit: float         # km/s, capture-orbit speed at periapsis
    dv: float
    e_capture: float
    period: Optional[float]   # s

    def summary(self) -> str:
        orb = "circular" if abs(self.ra - self.rp) < 1e-6 else f"{self.rp:.0f} x {self.ra:.0f} km"
        return (f"Capture at {self.body}: v_inf={self.vinf:.3f} km/s into {orb} orbit, "
                f"dv={self.dv:.4f} km/s")


def capture_dv(body, vinf: float, peri_alt: float, apo_alt: Optional[float] = None) -> CaptureResult:
    """Delta-v at periapsis to capture from a hyperbola (``vinf``) into an orbit
    with periapsis altitude ``peri_alt`` and apoapsis altitude ``apo_alt``
    (None -> circular).  Altitudes in km above the body's mean radius."""
    b = get_body(body)
    rp = b.radius + peri_alt
    ra = rp if apo_alt is None else b.radius + apo_alt
    v_hyp = hyperbolic_periapsis_speed(b.mu, rp, vinf)
    a = 0.5 * (rp + ra)
    v_orb = vis_viva(b.mu, rp, a)
    from .kepler import period as _period
    return CaptureResult(b.name, vinf, rp, ra, v_hyp, v_orb, v_hyp - v_orb,
                         (ra - rp) / (ra + rp), _period(b.mu, a))


def optimal_capture_periapsis(body, vinf: float, e_capture: float) -> float:
    """Periapsis radius minimizing capture dv for a fixed capture eccentricity:
    r_p* = 2 mu (1 - e) / v_inf^2   (Curtis eq. 8.67)."""
    b = get_body(body)
    return 2.0 * b.mu * (1.0 - e_capture) / (vinf * vinf)


# ---------------------------------------------------------------------------
# Flybys (gravity assists)
# ---------------------------------------------------------------------------
@dataclass
class FlybyResult:
    body: str
    vinf_in: np.ndarray
    vinf_out: np.ndarray
    rp: float                  # km, periapsis radius
    alt: float                 # km, periapsis altitude
    turn_angle: float          # rad, actual bending between v_inf vectors
    max_turn_angle: float      # rad, achievable at this rp with |vinf_in|
    dv_powered: float          # km/s, impulsive burn at periapsis (0 if unpowered)
    v_helio_in: np.ndarray
    v_helio_out: np.ndarray
    delta_v_helio: float       # km/s, |v_out - v_in| heliocentric (the "free" dv)
    feasible: bool

    def summary(self) -> str:
        return (f"Flyby of {self.body}: |v_inf| in={np.linalg.norm(self.vinf_in):.3f} "
                f"out={np.linalg.norm(self.vinf_out):.3f} km/s, turn={math.degrees(self.turn_angle):.1f} deg, "
                f"periapsis alt={self.alt:.0f} km, powered dv={self.dv_powered:.4f} km/s, "
                f"heliocentric |dv|={self.delta_v_helio:.3f} km/s"
                + ("" if self.feasible else "  ** INFEASIBLE (periapsis below minimum) **"))


def flyby_unpowered(body, vinf_in: np.ndarray, v_body: np.ndarray, rp: float,
                    side: str = "leading") -> FlybyResult:
    """Unpowered planar flyby: rotate v_inf_in by the turn angle for periapsis
    radius ``rp``.  ``side`` = 'leading' (pass ahead of the planet, spacecraft
    slows down heliocentrically) or 'trailing' (pass behind, speeds up).
    The rotation is about the axis normal to the plane of v_inf_in and v_body.
    """
    b = get_body(body)
    vinf_in = np.asarray(vinf_in, float)
    v_body = np.asarray(v_body, float)
    vinf = float(np.linalg.norm(vinf_in))
    delta = turn_angle(b.mu, rp, vinf)
    n = np.cross(v_body, vinf_in)
    if np.linalg.norm(n) < 1e-9:
        n = np.array([0.0, 0.0, 1.0])
    n = unit(n)
    sgn = 1.0 if side == "trailing" else -1.0
    vinf_out = _rotate(vinf_in, n, sgn * delta)
    v_in = v_body + vinf_in
    v_out = v_body + vinf_out
    return FlybyResult(b.name, vinf_in, vinf_out, rp, rp - b.radius, delta, delta, 0.0,
                       v_in, v_out, float(np.linalg.norm(v_out - v_in)), rp >= b.radius)


def _rotate(v: np.ndarray, axis: np.ndarray, ang: float) -> np.ndarray:
    """Rodrigues rotation of v about unit ``axis`` by ``ang`` [rad]."""
    c, s = math.cos(ang), math.sin(ang)
    return v * c + np.cross(axis, v) * s + axis * float(np.dot(axis, v)) * (1.0 - c)


def flyby_powered(body, vinf_in: np.ndarray, vinf_out: np.ndarray, v_body: np.ndarray,
                  min_alt: float = 200.0) -> FlybyResult:
    """Powered flyby matching given incoming and outgoing v_inf vectors.

    The two hyperbolas share a periapsis radius r_p chosen so that
    delta_in/2 + delta_out/2 equals the angle between the v_inf vectors; the
    speed mismatch is made up by a tangential burn at periapsis:
    dv = |sqrt(vinf_out^2 + 2mu/rp) - sqrt(vinf_in^2 + 2mu/rp)|.
    If the required turning cannot be achieved above ``min_alt`` the flyby is
    flagged infeasible and r_p is clamped to the minimum.
    (Standard MGA powered-flyby model, e.g. Izzo 2010.)
    """
    b = get_body(body)
    vinf_in = np.asarray(vinf_in, float)
    vinf_out = np.asarray(vinf_out, float)
    v_body = np.asarray(v_body, float)
    vi, vo = float(np.linalg.norm(vinf_in)), float(np.linalg.norm(vinf_out))
    delta_req = angle_between(vinf_in, vinf_out)
    rp_min = b.radius + min_alt

    def turn_total(rp: float) -> float:
        return 0.5 * turn_angle(b.mu, rp, vi) + 0.5 * turn_angle(b.mu, rp, vo)

    feasible = True
    if turn_total(rp_min) < delta_req:
        rp = rp_min
        feasible = False
    else:
        # turn_total decreases monotonically with rp; find rp with turn_total = delta_req
        rp_hi = rp_min
        while turn_total(rp_hi) > delta_req and rp_hi < 1e9:
            rp_hi *= 2.0
        rp = brentq(lambda r: turn_total(r) - delta_req, rp_min, rp_hi, xtol=1e-6) \
            if delta_req > 1e-9 else math.inf
    if math.isinf(rp):
        dv = abs(vo - vi)
    else:
        dv = abs(hyperbolic_periapsis_speed(b.mu, rp, vo) - hyperbolic_periapsis_speed(b.mu, rp, vi))
    v_in = v_body + vinf_in
    v_out = v_body + vinf_out
    return FlybyResult(b.name, vinf_in, vinf_out, rp, (rp - b.radius) if not math.isinf(rp) else math.inf,
                       delta_req, turn_total(rp_min), dv, v_in, v_out,
                       float(np.linalg.norm(v_out - v_in)), feasible)


def flyby_max_heliocentric_dv(body, vinf: float, alt: float) -> float:
    """Largest heliocentric speed change an unpowered flyby can give:
    |dv| = 2 v_inf sin(delta/2) = 2 v_inf / e."""
    b = get_body(body)
    return 2.0 * vinf * math.sin(0.5 * turn_angle(b.mu, b.radius + alt, vinf))


# ---------------------------------------------------------------------------
# Heliocentric legs
# ---------------------------------------------------------------------------
@dataclass
class Leg:
    origin: str
    target: str
    jd_dep: float
    jd_arr: float
    r1: np.ndarray
    r2: np.ndarray
    v_origin: np.ndarray      # heliocentric velocity of origin body at departure
    v_target: np.ndarray
    lambert: LambertSolution
    vinf_dep: np.ndarray      # v_sc - v_origin at departure
    vinf_arr: np.ndarray      # v_sc - v_target at arrival

    @property
    def tof_days(self) -> float:
        return self.jd_arr - self.jd_dep

    @property
    def c3(self) -> float:
        return float(np.dot(self.vinf_dep, self.vinf_dep))

    @property
    def vinf_dep_mag(self) -> float:
        return float(np.linalg.norm(self.vinf_dep))

    @property
    def vinf_arr_mag(self) -> float:
        return float(np.linalg.norm(self.vinf_arr))

    @property
    def dla_deg(self) -> float:
        """Declination of the departure asymptote relative to the ecliptic [deg]
        (use frame='equatorial' states for the Earth-equatorial DLA)."""
        v = self.vinf_dep
        return math.degrees(math.asin(v[2] / np.linalg.norm(v)))

    def summary(self) -> str:
        return (f"Leg {self.origin} -> {self.target}: depart {jd_to_iso(self.jd_dep, 'hours')} "
                f"arrive {jd_to_iso(self.jd_arr, 'hours')} (TOF {self.tof_days:.1f} d), "
                f"C3={self.c3:.2f} km2/s2, v_inf,arr={self.vinf_arr_mag:.3f} km/s, "
                f"transfer angle {math.degrees(self.lambert.dtheta):.1f} deg")


def heliocentric_leg(origin, target, depart: Epoch, arrive: Epoch, prograde: bool = True,
                     frame: str = "ecliptic") -> Leg:
    """Lambert arc between two bodies at the given epochs."""
    o, t = get_body(origin), get_body(target)
    jd1, jd2 = to_jd(depart), to_jd(arrive)
    tof = (jd2 - jd1) * 86400.0
    r1, v1 = body_state(o.name, jd1, frame)
    r2, v2 = body_state(t.name, jd2, frame)
    sol = lambert(r1, r2, tof, MU_SUN, prograde=prograde)
    return Leg(o.name, t.name, jd1, jd2, r1, r2, v1, v2, sol, sol.v1 - v1, sol.v2 - v2)


# ---------------------------------------------------------------------------
# Multi-leg missions
# ---------------------------------------------------------------------------
@dataclass
class InterplanetaryMission:
    bodies: list[str]
    epochs: list[float]                 # JD
    legs: list[Leg]
    departure: DepartureResult
    flybys: list[FlybyResult]
    capture: Optional[CaptureResult]
    aerocapture: bool

    @property
    def dv_departure(self) -> float:
        return self.departure.dv

    @property
    def dv_flybys(self) -> float:
        return sum(f.dv_powered for f in self.flybys)

    @property
    def dv_capture(self) -> float:
        return 0.0 if (self.capture is None or self.aerocapture) else self.capture.dv

    @property
    def dv_total(self) -> float:
        return self.dv_departure + self.dv_flybys + self.dv_capture

    @property
    def tof_days(self) -> float:
        return self.epochs[-1] - self.epochs[0]

    @property
    def feasible(self) -> bool:
        return all(f.feasible for f in self.flybys)

    def summary(self) -> str:
        lines = [f"Mission {' -> '.join(self.bodies)}  (TOF {self.tof_days:.1f} d = {self.tof_days/365.25:.2f} yr)"]
        for leg in self.legs:
            lines.append("  " + leg.summary())
        lines.append("  " + self.departure.summary())
        for f in self.flybys:
            lines.append("  " + f.summary())
        if self.capture is not None:
            lines.append("  " + self.capture.summary() + ("  [aerocapture: propulsive dv not charged]" if self.aerocapture else ""))
        lines.append(f"  TOTAL propulsive dv (post-launch) = {self.dv_total:.4f} km/s"
                     + ("" if self.feasible else "   ** contains an infeasible flyby **"))
        return "\n".join(lines)


def interplanetary_mission(bodies: Sequence[str], epochs: Sequence[Epoch], park_alt: float = 300.0,
                           capture_peri_alt: Optional[float] = 300.0, capture_apo_alt: Optional[float] = None,
                           flyby_min_alt: float = 300.0, aerocapture: bool = False,
                           prograde: bool = True) -> InterplanetaryMission:
    """Patched-conic mission through a sequence of bodies at given epochs.

    bodies = [origin, flyby_1, ..., target]; epochs = matching encounter times.
    Departure from a circular parking orbit at ``park_alt`` km about the origin;
    powered flybys at intermediate bodies (dv charged if v_inf magnitudes
    mismatch); capture at the target into ``capture_peri_alt`` x
    ``capture_apo_alt`` km (None apo -> circular; peri None -> flyby only).
    """
    if len(bodies) != len(epochs) or len(bodies) < 2:
        raise ValueError("need len(bodies) == len(epochs) >= 2")
    jds = [to_jd(e) for e in epochs]
    legs = [heliocentric_leg(bodies[k], bodies[k + 1], jds[k], jds[k + 1], prograde)
            for k in range(len(bodies) - 1)]
    dep = departure_dv(bodies[0], legs[0].vinf_dep_mag, park_alt)
    flybys = []
    for k in range(1, len(bodies) - 1):
        flybys.append(flyby_powered(bodies[k], legs[k - 1].vinf_arr, legs[k].vinf_dep,
                                    legs[k - 1].v_target, flyby_min_alt))
    cap = None
    if capture_peri_alt is not None:
        cap = capture_dv(bodies[-1], legs[-1].vinf_arr_mag, capture_peri_alt, capture_apo_alt)
    return InterplanetaryMission([get_body(b).name for b in bodies], jds, legs, dep, flybys, cap, aerocapture)


# ---------------------------------------------------------------------------
# Hohmann-based quick estimate (circular coplanar planetary orbits)
# ---------------------------------------------------------------------------
@dataclass
class HohmannInterplanetary:
    origin: str
    target: str
    vinf_dep: float
    vinf_arr: float
    departure: DepartureResult
    capture: Optional[CaptureResult]
    tof: float            # s
    phase_angle: float    # rad, target lead angle at departure
    synodic_period: float # s

    @property
    def dv_total(self) -> float:
        return self.departure.dv + (self.capture.dv if self.capture else 0.0)

    def summary(self) -> str:
        s = (f"Hohmann {self.origin} -> {self.target}: v_inf dep={self.vinf_dep:.3f}, arr={self.vinf_arr:.3f} km/s; "
             f"TOF={self.tof/86400:.1f} d; phase angle={math.degrees(self.phase_angle):.1f} deg; "
             f"synodic period={self.synodic_period/86400/365.25:.2f} yr\n  {self.departure.summary()}")
        if self.capture:
            s += f"\n  {self.capture.summary()}\n  total dv={self.dv_total:.4f} km/s"
        return s


def hohmann_interplanetary(origin, target, park_alt: float = 300.0,
                           capture_peri_alt: Optional[float] = 300.0,
                           capture_apo_alt: Optional[float] = None) -> HohmannInterplanetary:
    """Textbook estimate using circular, coplanar planetary orbits at the mean
    semi-major axes (Curtis §8.4-8.7)."""
    from .transfers import hohmann, hohmann_phase_angle, synodic_period
    from .kepler import period as _period
    o, t = get_body(origin), get_body(target)
    h = hohmann(MU_SUN, o.a_mean, t.a_mean)
    vinf_dep = abs(h.vt1 - h.v1)
    vinf_arr = abs(h.vt2 - h.v2)
    dep = departure_dv(o, vinf_dep, park_alt)
    cap = capture_dv(t, vinf_arr, capture_peri_alt, capture_apo_alt) if capture_peri_alt is not None else None
    return HohmannInterplanetary(o.name, t.name, vinf_dep, vinf_arr, dep, cap, h.tof,
                                 hohmann_phase_angle(MU_SUN, o.a_mean, t.a_mean),
                                 synodic_period(_period(MU_SUN, o.a_mean), _period(MU_SUN, t.a_mean)))
