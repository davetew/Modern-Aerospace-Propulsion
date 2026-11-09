"""Lambert's problem: the conic joining two position vectors in a given time.

Universal-variable formulation (Bate, Mueller & White §5.3; Curtis Algorithm
5.2) with a bracketed root-finder in z = alpha*chi^2, which is robust for
single-revolution transfers of either direction (prograde/retrograde) and any
conic type (ellipse, parabola, hyperbola).  Multi-revolution solutions are
not supported in this version.

Units: km, s, rad.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from .kepler import stumpff_C, stumpff_S, coe_from_rv, ClassicalElements


class LambertError(RuntimeError):
    pass


@dataclass
class LambertSolution:
    v1: np.ndarray          # km/s, velocity at r1
    v2: np.ndarray          # km/s, velocity at r2
    dtheta: float           # rad, transfer angle
    z: float                # universal variable at convergence
    a: float                # km, semi-major axis of the transfer conic
    tof: float              # s
    mu: float

    @property
    def elements(self) -> ClassicalElements:
        return coe_from_rv(self._r1, self.v1, self.mu)

    def with_positions(self, r1, r2):
        self._r1 = np.asarray(r1, float)
        self._r2 = np.asarray(r2, float)
        return self


def _y(z: float, r1: float, r2: float, A: float) -> float:
    C = stumpff_C(z)
    S = stumpff_S(z)
    return r1 + r2 + A * (z * S - 1.0) / math.sqrt(C)


def lambert(r1, r2, tof: float, mu: float, prograde: bool = True,
            z_tol: float = 1e-10) -> LambertSolution:
    """Solve Lambert's problem for the single-revolution transfer.

    Parameters
    ----------
    r1, r2 : array-like (3,)  position vectors [km]
    tof    : time of flight [s] (> 0)
    mu     : gravitational parameter of the central body [km^3/s^2]
    prograde : True for a transfer with angular momentum along +z (counter-
        clockwise seen from +z, i.e. the direction the planets move in the
        ecliptic frame); False for retrograde.

    Returns
    -------
    LambertSolution with v1, v2 [km/s].
    """
    if tof <= 0.0:
        raise ValueError("time of flight must be positive")
    r1 = np.asarray(r1, float)
    r2 = np.asarray(r2, float)
    r1n = float(np.linalg.norm(r1))
    r2n = float(np.linalg.norm(r2))
    cross = np.cross(r1, r2)
    cosdt = float(np.dot(r1, r2)) / (r1n * r2n)
    cosdt = max(-1.0, min(1.0, cosdt))
    dtheta = math.acos(cosdt)
    # resolve the transfer direction from the sign of the z-component of r1 x r2
    if prograde:
        if cross[2] < 0.0:
            dtheta = 2.0 * math.pi - dtheta
    else:
        if cross[2] >= 0.0:
            dtheta = 2.0 * math.pi - dtheta

    sindt = math.sin(dtheta)
    if abs(sindt) < 1e-12 or abs(1.0 - cosdt) < 1e-14:
        raise LambertError("transfer angle is 0 or 180 deg: transfer plane undefined "
                           "(perturb the geometry or the dates slightly)")
    A = sindt * math.sqrt(r1n * r2n / (1.0 - cosdt))

    def F(z: float) -> float:
        y = _y(z, r1n, r2n, A)
        if y < 0.0:
            return -math.inf
        C = stumpff_C(z)
        S = stumpff_S(z)
        return (y / C) ** 1.5 * S + A * math.sqrt(y) - math.sqrt(mu) * tof

    # ---- bracket the root ------------------------------------------------
    # Upper bound: single-revolution ellipses have z < (2*pi)^2 where F -> +inf.
    z_hi = 4.0 * math.pi ** 2 - 1e-6
    # Lower bound: move into the hyperbolic regime (z < 0) until F < 0.  Where
    # y(z) < 0 (only possible for A < 0, i.e. dtheta > 180 deg) F is undefined,
    # so first find z_min with y(z_min) = 0 and start just above it.
    if A < 0.0:
        # y is increasing in z; find where it crosses zero
        z_lo = 0.0
        if _y(z_lo, r1n, r2n, A) < 0.0:
            # y(0) < 0 -> root of y lies at positive z
            z_a, z_b = 0.0, z_hi
            z_lo = brentq(lambda z: _y(z, r1n, r2n, A), z_a, z_b, xtol=1e-12) + 1e-9
        else:
            # y(0) > 0 -> find negative z where y just stays positive
            z_lo = -1.0
            while _y(z_lo, r1n, r2n, A) > 0.0 and F(z_lo) > 0.0 and z_lo > -1e6:
                z_lo *= 2.0
            if _y(z_lo, r1n, r2n, A) <= 0.0:
                z_lo = brentq(lambda z: _y(z, r1n, r2n, A), z_lo, 0.0, xtol=1e-12) + 1e-9
    else:
        z_lo = -1.0
        while F(z_lo) > 0.0 and z_lo > -1e8:
            z_lo *= 4.0
    if F(z_lo) > 0.0:
        raise LambertError("could not bracket the Lambert root (time of flight too short?)")
    if F(z_hi) < 0.0:
        raise LambertError("time of flight exceeds the single-revolution limit")

    z = brentq(F, z_lo, z_hi, xtol=z_tol, maxiter=200)
    y = _y(z, r1n, r2n, A)
    f = 1.0 - y / r1n
    g = A * math.sqrt(y / mu)
    gdot = 1.0 - y / r2n
    v1 = (r2 - f * r1) / g
    v2 = (gdot * r2 - r1) / g
    energy = 0.5 * float(np.dot(v1, v1)) - mu / r1n
    a = math.inf if abs(energy) < 1e-14 else -mu / (2.0 * energy)
    sol = LambertSolution(v1=v1, v2=v2, dtheta=dtheta, z=z, a=a, tof=tof, mu=mu)
    return sol.with_positions(r1, r2)


def lambert_both_directions(r1, r2, tof: float, mu: float) -> dict[str, LambertSolution]:
    """Convenience: both the prograde and retrograde solutions."""
    out = {}
    for name, pro in (("prograde", True), ("retrograde", False)):
        try:
            out[name] = lambert(r1, r2, tof, mu, prograde=pro)
        except LambertError:
            pass
    return out
