"""Two-body (Keplerian) orbital mechanics utilities.

Units: km, s, rad.  Vectors are numpy arrays of shape (3,).

References: Curtis, *Orbital Mechanics for Engineering Students* (3rd ed.)
Algorithms 3.3/3.4 (universal Kepler equation and Lagrange coefficients),
4.1/4.2 (elements <-> state); Bate, Mueller & White, *Fundamentals of
Astrodynamics* Ch. 4.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

TWO_PI = 2.0 * math.pi


# ---------------------------------------------------------------------------
# Scalar energy / speed relations
# ---------------------------------------------------------------------------
def vis_viva(mu: float, r: float, a: float) -> float:
    """Speed on a conic of semi-major axis ``a`` at radius ``r``.

    v^2 = mu (2/r - 1/a).  ``a`` < 0 for hyperbolas, inf for parabolas.
    """
    if math.isinf(a):
        return math.sqrt(2.0 * mu / r)
    return math.sqrt(mu * (2.0 / r - 1.0 / a))


def circular_velocity(mu: float, r: float) -> float:
    return math.sqrt(mu / r)


def escape_velocity(mu: float, r: float) -> float:
    return math.sqrt(2.0 * mu / r)


def specific_energy(mu: float, r: float, v: float) -> float:
    """epsilon = v^2/2 - mu/r  [km^2/s^2]"""
    return 0.5 * v * v - mu / r


def semi_major_axis_from_energy(mu: float, energy: float) -> float:
    if energy == 0.0:
        return math.inf
    return -mu / (2.0 * energy)


def period(mu: float, a: float) -> float:
    """Orbital period of an ellipse [s]."""
    if a <= 0:
        raise ValueError("period defined only for elliptical orbits (a > 0)")
    return TWO_PI * math.sqrt(a ** 3 / mu)


def mean_motion(mu: float, a: float) -> float:
    return math.sqrt(mu / abs(a) ** 3)


def apsides_from_rp_ra(rp: float, ra: float) -> tuple[float, float]:
    """(a, e) from periapsis and apoapsis radii."""
    a = 0.5 * (rp + ra)
    e = (ra - rp) / (ra + rp)
    return a, e


def periapsis_speed(mu: float, rp: float, ra: float) -> float:
    """Speed at periapsis of an ellipse with apsides rp, ra."""
    a = 0.5 * (rp + ra)
    return vis_viva(mu, rp, a)


def apoapsis_speed(mu: float, rp: float, ra: float) -> float:
    a = 0.5 * (rp + ra)
    return vis_viva(mu, ra, a)


# ---------------------------------------------------------------------------
# Hyperbolic relations (used by patched conics)
# ---------------------------------------------------------------------------
def hyperbolic_periapsis_speed(mu: float, rp: float, vinf: float) -> float:
    """v_p = sqrt(v_inf^2 + 2 mu / r_p)"""
    return math.sqrt(vinf * vinf + 2.0 * mu / rp)


def hyperbolic_eccentricity(mu: float, rp: float, vinf: float) -> float:
    """e = 1 + r_p v_inf^2 / mu"""
    return 1.0 + rp * vinf * vinf / mu


def turn_angle(mu: float, rp: float, vinf: float) -> float:
    """Asymptotic turning angle delta = 2 asin(1/e) [rad]."""
    e = hyperbolic_eccentricity(mu, rp, vinf)
    return 2.0 * math.asin(1.0 / e)


def periapsis_for_turn_angle(mu: float, vinf: float, delta: float) -> float:
    """r_p that produces turning angle ``delta`` for a given v_inf."""
    e = 1.0 / math.sin(0.5 * delta)
    return (e - 1.0) * mu / (vinf * vinf)


def aiming_radius(mu: float, rp: float, vinf: float) -> float:
    """Impact parameter (B-plane magnitude) Delta = r_p sqrt(1 + 2 mu/(r_p v_inf^2))."""
    return rp * math.sqrt(1.0 + 2.0 * mu / (rp * vinf * vinf))


# ---------------------------------------------------------------------------
# Kepler's equation
# ---------------------------------------------------------------------------
def kepler_E(M: float, e: float, tol: float = 1e-12, max_iter: int = 60) -> float:
    """Solve M = E - e sin E for E (elliptic), Newton iteration."""
    M = math.fmod(M, TWO_PI)
    if M > math.pi:
        M -= TWO_PI
    elif M < -math.pi:
        M += TWO_PI
    # Danby's starting value; Newton converges from it for all 0 <= e < 1
    E = M + 0.85 * e * (1.0 if math.sin(M) >= 0 else -1.0)
    for _ in range(max_iter):
        f = E - e * math.sin(E) - M
        fp = 1.0 - e * math.cos(E)
        dE = -f / fp
        E += dE
        if abs(dE) < tol:
            return E
    # fall back to bisection on the monotonic function f(E) over [-pi, pi]
    lo, hi = -math.pi, math.pi
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid - e * math.sin(mid) - M > 0:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            return 0.5 * (lo + hi)
    raise RuntimeError("kepler_E did not converge")


def kepler_H(M: float, e: float, tol: float = 1e-12, max_iter: int = 100) -> float:
    """Solve M = e sinh F - F for F (hyperbolic)."""
    F = math.asinh(M / e) if e > 1.0 else M
    for _ in range(max_iter):
        f = e * math.sinh(F) - F - M
        fp = e * math.cosh(F) - 1.0
        dF = -f / fp
        F += dF
        if abs(dF) < tol:
            return F
    raise RuntimeError("kepler_H did not converge")


def true_anomaly_from_E(E: float, e: float) -> float:
    return 2.0 * math.atan2(math.sqrt(1.0 + e) * math.sin(0.5 * E),
                            math.sqrt(1.0 - e) * math.cos(0.5 * E))


def E_from_true_anomaly(nu: float, e: float) -> float:
    return 2.0 * math.atan2(math.sqrt(1.0 - e) * math.sin(0.5 * nu),
                            math.sqrt(1.0 + e) * math.cos(0.5 * nu))


def time_since_periapsis(mu: float, a: float, e: float, nu: float) -> float:
    """Time from periapsis to true anomaly ``nu`` (elliptic or hyperbolic)."""
    if e < 1.0:
        E = E_from_true_anomaly(nu, e)
        M = E - e * math.sin(E)
        return M / mean_motion(mu, a)
    else:
        F = 2.0 * math.atanh(math.sqrt((e - 1.0) / (e + 1.0)) * math.tan(0.5 * nu))
        M = e * math.sinh(F) - F
        return M / mean_motion(mu, a)


# ---------------------------------------------------------------------------
# Classical orbital elements
# ---------------------------------------------------------------------------
@dataclass
class ClassicalElements:
    a: float        # km (negative for hyperbola)
    e: float
    i: float        # rad
    raan: float     # rad, longitude of ascending node
    argp: float     # rad, argument of periapsis
    nu: float       # rad, true anomaly
    mu: float

    @property
    def p(self) -> float:
        return self.a * (1.0 - self.e ** 2)

    @property
    def h(self) -> float:
        return math.sqrt(self.mu * self.p)

    @property
    def rp(self) -> float:
        return self.a * (1.0 - self.e)

    @property
    def ra(self) -> float:
        return self.a * (1.0 + self.e) if self.e < 1.0 else math.inf

    @property
    def energy(self) -> float:
        return -self.mu / (2.0 * self.a)

    @property
    def period(self) -> float:
        return period(self.mu, self.a)

    def as_degrees(self) -> dict:
        return dict(a_km=self.a, e=self.e, i_deg=math.degrees(self.i),
                    raan_deg=math.degrees(self.raan), argp_deg=math.degrees(self.argp),
                    nu_deg=math.degrees(self.nu))


def coe_from_rv(r: np.ndarray, v: np.ndarray, mu: float) -> ClassicalElements:
    """Classical elements from state vector (Curtis Algorithm 4.2)."""
    r = np.asarray(r, float)
    v = np.asarray(v, float)
    rn = np.linalg.norm(r)
    vn = np.linalg.norm(v)
    vr = float(np.dot(r, v)) / rn
    hvec = np.cross(r, v)
    h = np.linalg.norm(hvec)
    inc = math.acos(max(-1.0, min(1.0, hvec[2] / h)))
    K = np.array([0.0, 0.0, 1.0])
    N = np.cross(K, hvec)
    Nn = np.linalg.norm(N)
    if Nn > 1e-12:
        raan = math.acos(max(-1.0, min(1.0, N[0] / Nn)))
        if N[1] < 0:
            raan = TWO_PI - raan
    else:
        raan = 0.0
    evec = ((vn * vn - mu / rn) * r - rn * vr * v) / mu
    e = np.linalg.norm(evec)
    if Nn > 1e-12 and e > 1e-12:
        argp = math.acos(max(-1.0, min(1.0, float(np.dot(N, evec)) / (Nn * e))))
        if evec[2] < 0:
            argp = TWO_PI - argp
    elif e > 1e-12:            # equatorial: measure argp from x-axis
        argp = math.atan2(evec[1], evec[0]) % TWO_PI
    else:
        argp = 0.0
    if e > 1e-12:
        nu = math.acos(max(-1.0, min(1.0, float(np.dot(evec, r)) / (e * rn))))
        if vr < 0:
            nu = TWO_PI - nu
    else:                      # circular: true anomaly from node (or x-axis)
        ref = N if Nn > 1e-12 else np.array([1.0, 0.0, 0.0])
        nu = math.acos(max(-1.0, min(1.0, float(np.dot(ref, r)) / (np.linalg.norm(ref) * rn))))
        if r[2] < 0 if Nn > 1e-12 else r[1] < 0:
            nu = TWO_PI - nu
    a = (h * h / mu) / (1.0 - e * e) if abs(1.0 - e) > 1e-12 else math.inf
    return ClassicalElements(a, e, inc, raan, argp, nu, mu)


def perifocal_to_inertial(raan: float, inc: float, argp: float) -> np.ndarray:
    """Rotation matrix Q from perifocal (p,q,w) to inertial (I,J,K)."""
    cO, sO = math.cos(raan), math.sin(raan)
    ci, si = math.cos(inc), math.sin(inc)
    cw, sw = math.cos(argp), math.sin(argp)
    return np.array([
        [cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci, sO * si],
        [sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci, -cO * si],
        [sw * si, cw * si, ci],
    ])


def rv_from_coe(coe: ClassicalElements) -> tuple[np.ndarray, np.ndarray]:
    """State vector from classical elements (Curtis Algorithm 4.5)."""
    mu, e, nu = coe.mu, coe.e, coe.nu
    p = coe.p
    r_pf = (p / (1.0 + e * math.cos(nu))) * np.array([math.cos(nu), math.sin(nu), 0.0])
    v_pf = math.sqrt(mu / p) * np.array([-math.sin(nu), e + math.cos(nu), 0.0])
    Q = perifocal_to_inertial(coe.raan, coe.i, coe.argp)
    return Q @ r_pf, Q @ v_pf


def rv_from_elements(mu: float, a: float, e: float, i: float, raan: float,
                     argp: float, nu: float) -> tuple[np.ndarray, np.ndarray]:
    return rv_from_coe(ClassicalElements(a, e, i, raan, argp, nu, mu))


# ---------------------------------------------------------------------------
# Universal-variable propagation
# ---------------------------------------------------------------------------
def stumpff_C(z: float) -> float:
    if z > 1e-8:
        sz = math.sqrt(z)
        return (1.0 - math.cos(sz)) / z
    if z < -1e-8:
        sz = math.sqrt(-z)
        return (math.cosh(sz) - 1.0) / (-z)
    return 0.5 - z / 24.0 + z * z / 720.0


def stumpff_S(z: float) -> float:
    if z > 1e-8:
        sz = math.sqrt(z)
        return (sz - math.sin(sz)) / (sz ** 3)
    if z < -1e-8:
        sz = math.sqrt(-z)
        return (math.sinh(sz) - sz) / (sz ** 3)
    return 1.0 / 6.0 - z / 120.0 + z * z / 5040.0


def universal_anomaly(mu: float, dt: float, r0: float, vr0: float, alpha: float,
                      tol: float = 1e-10, max_iter: int = 200) -> float:
    """Solve the universal Kepler equation for chi (Curtis Algorithm 3.3).

    alpha = 1/a (0 for parabola, <0 for hyperbola).
    """
    smu = math.sqrt(mu)
    chi = smu * abs(alpha) * dt if alpha != 0.0 else smu * dt / r0  # initial guess
    for _ in range(max_iter):
        z = alpha * chi * chi
        C, S = stumpff_C(z), stumpff_S(z)
        F = (r0 * vr0 / smu * chi * chi * C + (1.0 - alpha * r0) * chi ** 3 * S
             + r0 * chi - smu * dt)
        dF = (r0 * vr0 / smu * chi * (1.0 - alpha * chi * chi * S)
              + (1.0 - alpha * r0) * chi * chi * C + r0)
        ratio = F / dF
        chi -= ratio
        if abs(ratio) < tol:
            return chi
    raise RuntimeError("universal_anomaly did not converge")


def propagate(r0: np.ndarray, v0: np.ndarray, dt: float, mu: float) -> tuple[np.ndarray, np.ndarray]:
    """Propagate a two-body state by ``dt`` seconds (Curtis Algorithm 3.4)."""
    r0 = np.asarray(r0, float)
    v0 = np.asarray(v0, float)
    if dt == 0.0:
        return r0.copy(), v0.copy()
    r0n = np.linalg.norm(r0)
    v0n = np.linalg.norm(v0)
    vr0 = float(np.dot(r0, v0)) / r0n
    alpha = 2.0 / r0n - v0n * v0n / mu
    chi = universal_anomaly(mu, dt, r0n, vr0, alpha)
    z = alpha * chi * chi
    C, S = stumpff_C(z), stumpff_S(z)
    f = 1.0 - chi * chi / r0n * C
    g = dt - chi ** 3 / math.sqrt(mu) * S
    r = f * r0 + g * v0
    rn = np.linalg.norm(r)
    fdot = math.sqrt(mu) / (rn * r0n) * (alpha * chi ** 3 * S - chi)
    gdot = 1.0 - chi * chi / rn * C
    v = fdot * r0 + gdot * v0
    return r, v


def sample_orbit(r0: np.ndarray, v0: np.ndarray, mu: float, t_end: float,
                 n: int = 400) -> np.ndarray:
    """Positions along a trajectory at n uniformly spaced times in [0, t_end]."""
    ts = np.linspace(0.0, t_end, n)
    out = np.empty((n, 3))
    for k, t in enumerate(ts):
        out[k], _ = propagate(r0, v0, float(t), mu)
    return out


# ---------------------------------------------------------------------------
# Small vector helpers
# ---------------------------------------------------------------------------
def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    c = float(np.dot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))
    return math.acos(max(-1.0, min(1.0, c)))


def rotate_z(v: np.ndarray, angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return R @ np.asarray(v, float)
