"""Impulsive transfers between coplanar/inclined circular orbits about one body.

Hohmann, bi-elliptic, plane changes, combined maneuvers and phasing.
Units: km, s, rad (angles in radians; helper wrappers in degrees are noted).

References: Sutton & Biblarz Ch. 4; Curtis Ch. 6; Vallado Ch. 6.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from scipy.optimize import minimize_scalar

from .kepler import vis_viva, circular_velocity, period, apsides_from_rp_ra, TWO_PI


@dataclass
class HohmannResult:
    r1: float
    r2: float
    a_t: float
    e_t: float
    v1: float          # circular speed at r1
    v2: float          # circular speed at r2
    vt1: float         # transfer-orbit speed at r1
    vt2: float         # transfer-orbit speed at r2
    dv1: float         # |vt1 - v1|
    dv2: float         # |v2 - vt2|
    tof: float         # s, half the transfer period

    @property
    def dv_total(self) -> float:
        return self.dv1 + self.dv2

    def summary(self, label: str = "Hohmann") -> str:
        return (f"{label}: r1={self.r1:.1f} km -> r2={self.r2:.1f} km | "
                f"dv1={self.dv1:.4f} dv2={self.dv2:.4f} total={self.dv_total:.4f} km/s | "
                f"TOF={self.tof/86400:.3f} d")


def hohmann(mu: float, r1: float, r2: float) -> HohmannResult:
    """Two-impulse Hohmann transfer between circular orbits of radii r1, r2."""
    a_t, e_t = apsides_from_rp_ra(min(r1, r2), max(r1, r2))
    v1 = circular_velocity(mu, r1)
    v2 = circular_velocity(mu, r2)
    vt1 = vis_viva(mu, r1, a_t)
    vt2 = vis_viva(mu, r2, a_t)
    return HohmannResult(r1, r2, a_t, e_t, v1, v2, vt1, vt2,
                         abs(vt1 - v1), abs(v2 - vt2), 0.5 * period(mu, a_t))


@dataclass
class BiellipticResult:
    r1: float
    r2: float
    rb: float
    dv1: float
    dv2: float
    dv3: float
    tof: float

    @property
    def dv_total(self) -> float:
        return self.dv1 + self.dv2 + self.dv3


def bielliptic(mu: float, r1: float, r2: float, rb: float) -> BiellipticResult:
    """Three-impulse bi-elliptic transfer via intermediate apoapsis rb (> r1, r2)."""
    a1 = 0.5 * (r1 + rb)
    a2 = 0.5 * (rb + r2)
    dv1 = abs(vis_viva(mu, r1, a1) - circular_velocity(mu, r1))
    dv2 = abs(vis_viva(mu, rb, a2) - vis_viva(mu, rb, a1))
    dv3 = abs(circular_velocity(mu, r2) - vis_viva(mu, r2, a2))
    tof = 0.5 * period(mu, a1) + 0.5 * period(mu, a2)
    return BiellipticResult(r1, r2, rb, dv1, dv2, dv3, tof)


def plane_change_dv(v: float, di: float) -> float:
    """Pure plane change of angle ``di`` [rad] at speed ``v``: dv = 2 v sin(di/2)."""
    return 2.0 * v * math.sin(0.5 * di)


def combined_dv(v_before: float, v_after: float, di: float) -> float:
    """Single impulse that changes speed and rotates the velocity by ``di`` [rad]."""
    return math.sqrt(v_before ** 2 + v_after ** 2 - 2.0 * v_before * v_after * math.cos(di))


@dataclass
class HohmannPlaneChangeResult:
    hohmann: HohmannResult
    di_total: float
    di1: float       # plane change performed at the first burn
    di2: float       # at the second burn
    dv1: float
    dv2: float

    @property
    def dv_total(self) -> float:
        return self.dv1 + self.dv2


def hohmann_with_plane_change(mu: float, r1: float, r2: float, di: float,
                              split: float | None = None) -> HohmannPlaneChangeResult:
    """Hohmann transfer with total plane change ``di`` [rad] shared between the
    two burns.  ``split`` = fraction of the plane change done at burn 1; if None
    the split minimizing total dv is found (normally most of it at the slower,
    outer burn)."""
    h = hohmann(mu, r1, r2)

    def total(s: float) -> float:
        d1 = combined_dv(h.v1, h.vt1, s * di)
        d2 = combined_dv(h.vt2, h.v2, (1.0 - s) * di)
        return d1 + d2

    if split is None:
        res = minimize_scalar(total, bounds=(0.0, 1.0), method="bounded",
                              options={"xatol": 1e-8})
        split = float(res.x)
    dv1 = combined_dv(h.v1, h.vt1, split * di)
    dv2 = combined_dv(h.vt2, h.v2, (1.0 - split) * di)
    return HohmannPlaneChangeResult(h, di, split * di, (1.0 - split) * di, dv1, dv2)


# ---------------------------------------------------------------------------
# Phasing / launch windows for circular coplanar orbits
# ---------------------------------------------------------------------------
def synodic_period(T1: float, T2: float) -> float:
    """Synodic period of two bodies with periods T1, T2 [same units]."""
    return abs(1.0 / (1.0 / T1 - 1.0 / T2))


def hohmann_phase_angle(mu: float, r1: float, r2: float) -> float:
    """Required lead angle of the target at departure [rad, (-pi, pi]].

    Positive = target ahead of the departure body.
    """
    h = hohmann(mu, r1, r2)
    n2 = math.sqrt(mu / r2 ** 3)
    phi = math.pi - n2 * h.tof
    return (phi + math.pi) % TWO_PI - math.pi


def wait_for_phase(mu: float, r1: float, r2: float, current_phase: float) -> float:
    """Time [s] until the phase angle (target minus origin, rad) next equals the
    Hohmann requirement, for circular coplanar orbits."""
    n1 = math.sqrt(mu / r1 ** 3)
    n2 = math.sqrt(mu / r2 ** 3)
    target = hohmann_phase_angle(mu, r1, r2)
    rate = n2 - n1                   # d(phase)/dt
    dphi = (target - current_phase)
    if rate > 0:
        dphi %= TWO_PI
    else:
        dphi = -((-dphi) % TWO_PI)
    return dphi / rate


def hohmann_round_trip(mu: float, r1: float, r2: float) -> dict:
    """Outbound Hohmann, stay time for the return window, and inbound Hohmann
    (circular coplanar).  Returns a dict with leg times [s] and dv [km/s]."""
    h = hohmann(mu, r1, r2)
    n1 = math.sqrt(mu / r1 ** 3)
    n2 = math.sqrt(mu / r2 ** 3)
    # phase angle at arrival: origin leads target by (n1 - n2)*tof - pi
    phi_arr = (n1 - n2) * h.tof - math.pi
    # need origin to be at required phase for return: -phase_angle(r2 -> r1)
    phi_req = -hohmann_phase_angle(mu, r2, r1)  # from target's viewpoint
    # phase of origin relative to target evolves at (n1 - n2)
    rate = n1 - n2
    dphi = (phi_req - phi_arr)
    if rate > 0:
        dphi %= TWO_PI
    else:
        dphi = -((-dphi) % TWO_PI)
    stay = dphi / rate
    return dict(outbound_tof=h.tof, stay=stay, inbound_tof=h.tof,
                total=2 * h.tof + stay, dv_outbound=h.dv_total, dv_inbound=h.dv_total,
                dv_total=2 * h.dv_total)
