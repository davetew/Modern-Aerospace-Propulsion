"""Analytic (mean-element) ephemerides for the planets and the Moon.

Planets
-------
JPL "Keplerian Elements for Approximate Positions of the Major Planets"
(E. M. Standish, JPL Solar System Dynamics,
https://ssd.jpl.nasa.gov/planets/approx_pos.html):

* Table 1 -- valid 1800 AD to 2050 AD (default when the epoch is inside it)
* Table 2a/2b -- valid 3000 BC to 3000 AD (used automatically outside Table 1)

Elements are referred to the mean ecliptic and equinox of J2000.  Stated
accuracy over 1800-2050 is of order 10-100 arcsec in longitude for the inner
planets and a few hundred arcsec for the outer planets -- ample for
preliminary mission design (porkchop plots, Delta-v budgets), not for
navigation.  "Earth" returns the Earth-Moon barycenter unless
``earth_state(..., correct_for_moon=True)`` is used.

Moon
----
Truncated Meeus (*Astronomical Algorithms*, 2nd ed., Ch. 47) series:
~30 longitude/distance terms and ~20 latitude terms, giving roughly 0.02 deg
in position and ~10 km in distance.  Returned in the J2000 ecliptic frame
(precession in longitude removed) as a geocentric state.

Time
----
Epochs may be ``datetime`` (naive = UTC), ISO-8601 strings, or Julian Date
floats.  TT-UTC = 69.184 s is applied (TDB ~ TT); the effect is negligible at
the accuracy of these models.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Union

import numpy as np

from .bodies import AU_KM, MU_SUN, EARTH, MOON, get_body
from .kepler import kepler_E, rv_from_elements, rotate_z

Epoch = Union[datetime, str, float, int]

JD_J2000 = 2451545.0
JD_UNIX_EPOCH = 2440587.5
OBLIQUITY_J2000 = math.radians(23.43928)
TT_MINUS_UTC_S = 69.184     # 37 leap seconds + 32.184 s (valid 2017 onward)


# ---------------------------------------------------------------------------
# Time utilities
# ---------------------------------------------------------------------------
def to_jd(epoch: Epoch) -> float:
    """Convert an epoch to Julian Date (UTC scale)."""
    if isinstance(epoch, (float, int)) and not isinstance(epoch, bool):
        return float(epoch)
    if isinstance(epoch, str):
        s = epoch.strip().replace("Z", "+00:00")
        try:
            epoch = datetime.fromisoformat(s)
        except ValueError:
            epoch = datetime.strptime(s, "%Y-%m-%d")
    if isinstance(epoch, datetime):
        if epoch.tzinfo is not None:
            epoch = epoch.astimezone(timezone.utc).replace(tzinfo=None)
        delta = epoch - datetime(1970, 1, 1)
        return JD_UNIX_EPOCH + delta.total_seconds() / 86400.0
    raise TypeError(f"Unsupported epoch type {type(epoch)}")


def jd_to_datetime(jd: float) -> datetime:
    return datetime(1970, 1, 1) + timedelta(days=jd - JD_UNIX_EPOCH)


def jd_to_iso(jd: float, precision: str = "minutes") -> str:
    return jd_to_datetime(jd).isoformat(timespec=precision)


def centuries_since_j2000(epoch: Epoch) -> float:
    jd_tt = to_jd(epoch) + TT_MINUS_UTC_S / 86400.0
    return (jd_tt - JD_J2000) / 36525.0


# ---------------------------------------------------------------------------
# JPL approximate planetary elements
#   columns: a [au], e, I [deg], L [deg], long.peri [deg], long.node [deg]
#   row 1: value at J2000, row 2: rate per Julian century
# ---------------------------------------------------------------------------
_TABLE1 = {
    "mercury": ((0.38709927, 0.20563593, 7.00497902, 252.25032350, 77.45779628, 48.33076593),
                (0.00000037, 0.00001906, -0.00594749, 149472.67411175, 0.16047689, -0.12534081)),
    "venus":   ((0.72333566, 0.00677672, 3.39467605, 181.97909950, 131.60246718, 76.67984255),
                (0.00000390, -0.00004107, -0.00078890, 58517.81538729, 0.00268329, -0.27769418)),
    "earth":   ((1.00000261, 0.01671123, -0.00001531, 100.46457166, 102.93768193, 0.0),
                (0.00000562, -0.00004392, -0.01294668, 35999.37244981, 0.32327364, 0.0)),
    "mars":    ((1.52371034, 0.09339410, 1.84969142, -4.55343205, -23.94362959, 49.55953891),
                (0.00001847, 0.00007882, -0.00813131, 19140.30268499, 0.44441088, -0.29257343)),
    "jupiter": ((5.20288700, 0.04838624, 1.30439695, 34.39644051, 14.72847983, 100.47390909),
                (-0.00011607, -0.00013253, -0.00183714, 3034.74612775, 0.21252668, 0.20469106)),
    "saturn":  ((9.53667594, 0.05386179, 2.48599187, 49.95424423, 92.59887831, 113.66242448),
                (-0.00125060, -0.00050991, 0.00193609, 1222.49362201, -0.41897216, -0.28867794)),
    "uranus":  ((19.18916464, 0.04725744, 0.77263783, 313.23810451, 170.95427630, 74.01692503),
                (-0.00196176, -0.00004397, -0.00242939, 428.48202785, 0.40805281, 0.04240589)),
    "neptune": ((30.06992276, 0.00859048, 1.77004347, -55.12002969, 44.96476227, 131.78422574),
                (0.00026291, 0.00005105, 0.00035372, 218.45945325, -0.32241464, -0.00508664)),
}

_TABLE2 = {
    "mercury": ((0.38709843, 0.20563661, 7.00559432, 252.25166724, 77.45771895, 48.33961819),
                (0.00000000, 0.00002123, -0.00590158, 149472.67486623, 0.15940013, -0.12214182)),
    "venus":   ((0.72332102, 0.00676399, 3.39777545, 181.97970850, 131.76755713, 76.67261496),
                (-0.00000026, -0.00005107, 0.00043494, 58517.81560260, 0.05679648, -0.27274174)),
    "earth":   ((1.00000018, 0.01673163, -0.00054346, 100.46691572, 102.93005885, -5.11260389),
                (-0.00000003, -0.00003661, -0.01337178, 35999.37306329, 0.31795260, -0.24123856)),
    "mars":    ((1.52371243, 0.09336511, 1.85181869, -4.56813164, -23.91744784, 49.71320984),
                (0.00000097, 0.00009149, -0.00724757, 19140.29934243, 0.45223625, -0.26852431)),
    "jupiter": ((5.20248019, 0.04853590, 1.29861416, 34.33479152, 14.27495244, 100.29282654),
                (-0.00002864, 0.00018026, -0.00322699, 3034.90371757, 0.18199196, 0.13024619)),
    "saturn":  ((9.54149883, 0.05550825, 2.49424102, 50.07571329, 92.86136063, 113.63998702),
                (-0.00003065, -0.00032044, 0.00451969, 1222.11494724, 0.54179478, -0.25015002)),
    "uranus":  ((19.18797948, 0.04685740, 0.77298127, 314.20276625, 172.43404441, 73.96250215),
                (-0.00020455, -0.00001550, -0.00180155, 428.49512595, 0.09266985, 0.05739699)),
    "neptune": ((30.06952752, 0.00895439, 1.77005520, 304.22289287, 46.68158724, 131.78635853),
                (0.00006447, 0.00000818, 0.00022400, 218.46515314, 0.01009938, -0.00606302)),
}
# Table 2b: extra terms b, c, s, f for the mean anomaly of the outer planets
_TABLE2B = {
    "jupiter": (-0.00012452, 0.06064060, -0.35635438, 38.35125000),
    "saturn":  (0.00025899, -0.13434469, 0.87320147, 38.35125000),
    "uranus":  (0.00058331, -0.97731848, 0.17689245, 7.67025000),
    "neptune": (-0.00041348, 0.68346318, -0.10162547, 7.67025000),
}

_JD_1800 = 2378496.5
_JD_2050 = 2469807.5


def _normalize_deg(x: float) -> float:
    x = math.fmod(x, 360.0)
    return x + 360.0 if x < 0 else x


def planet_elements(name: str, epoch: Epoch, table: str = "auto") -> dict:
    """Osculating-mean elements of a planet at ``epoch`` (J2000 ecliptic).

    Returns a dict with a [km], e, i, raan, argp, M, nu, L, lon_peri [rad].
    """
    key = name.strip().lower()
    if key in ("em bary", "emb", "earth-moon barycenter"):
        key = "earth"
    jd = to_jd(epoch)
    if table == "auto":
        table = "1" if _JD_1800 <= jd <= _JD_2050 else "2"
    tab = _TABLE1 if table == "1" else _TABLE2
    if key not in tab:
        raise KeyError(f"No analytic ephemeris for '{name}'. Available: {sorted(tab)}")
    T = centuries_since_j2000(epoch)
    base, rate = tab[key]
    a, e, I, L, lon_peri, lon_node = (b + r * T for b, r in zip(base, rate))
    argp = lon_peri - lon_node
    M = L - lon_peri
    if table == "2" and key in _TABLE2B:
        b, c, s, f = _TABLE2B[key]
        fT = math.radians(f * T)
        M += b * T * T + c * math.cos(fT) + s * math.sin(fT)
    M = math.radians(_normalize_deg(M))
    E = kepler_E(M, e)
    nu = 2.0 * math.atan2(math.sqrt(1 + e) * math.sin(E / 2), math.sqrt(1 - e) * math.cos(E / 2))
    return dict(a=a * AU_KM, e=e, i=math.radians(I), raan=math.radians(_normalize_deg(lon_node)),
                argp=math.radians(_normalize_deg(argp)), M=M, nu=nu % (2 * math.pi),
                L=math.radians(_normalize_deg(L)), lon_peri=math.radians(_normalize_deg(lon_peri)),
                table=table)


def planet_state(name: str, epoch: Epoch, frame: str = "ecliptic",
                 table: str = "auto") -> tuple[np.ndarray, np.ndarray]:
    """Heliocentric position [km] and velocity [km/s] of a planet.

    frame = 'ecliptic' (J2000 ecliptic, default) or 'equatorial' (J2000 ICRF-like).
    Velocity is the two-body velocity on the instantaneous mean orbit (the
    secular element rates contribute < 1e-6 km/s and are neglected).
    """
    el = planet_elements(name, epoch, table)
    r, v = rv_from_elements(MU_SUN, el["a"], el["e"], el["i"], el["raan"], el["argp"], el["nu"])
    if frame == "equatorial":
        r, v = ecliptic_to_equatorial(r), ecliptic_to_equatorial(v)
    elif frame != "ecliptic":
        raise ValueError("frame must be 'ecliptic' or 'equatorial'")
    return r, v


def ecliptic_to_equatorial(vec: np.ndarray) -> np.ndarray:
    c, s = math.cos(OBLIQUITY_J2000), math.sin(OBLIQUITY_J2000)
    R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return R @ np.asarray(vec, float)


def equatorial_to_ecliptic(vec: np.ndarray) -> np.ndarray:
    c, s = math.cos(OBLIQUITY_J2000), math.sin(OBLIQUITY_J2000)
    R = np.array([[1, 0, 0], [0, c, s], [0, -s, c]])
    return R @ np.asarray(vec, float)


def body_state(name, epoch: Epoch, frame: str = "ecliptic") -> tuple[np.ndarray, np.ndarray]:
    """Heliocentric state of a planet, or of the Moon (via Earth + geocentric Moon)."""
    body = get_body(name)
    if body.name == "Moon":
        re, ve = earth_state(epoch, frame=frame, correct_for_moon=True)
        rm, vm = moon_state_geocentric(epoch, frame=frame)
        return re + rm, ve + vm
    if body.name == "Sun":
        return np.zeros(3), np.zeros(3)
    return planet_state(body.name, epoch, frame=frame)


def earth_state(epoch: Epoch, frame: str = "ecliptic",
                correct_for_moon: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Heliocentric Earth state.  By default this is the Earth-Moon barycenter;
    with ``correct_for_moon`` the Moon's pull on the barycenter is removed
    (shift of up to ~4700 km)."""
    r, v = planet_state("earth", epoch, frame=frame)
    if correct_for_moon:
        rm, vm = moon_state_geocentric(epoch, frame=frame)
        f = MOON.mu / (EARTH.mu + MOON.mu)
        r = r - f * rm
        v = v - f * vm
    return r, v


# ---------------------------------------------------------------------------
# Moon (truncated Meeus Ch. 47)
# ---------------------------------------------------------------------------
# (D, M, M', F, sigma_l [1e-6 deg], sigma_r [1e-3 km])
_MOON_LR = [
    (0, 0, 1, 0, 6288774, -20905355),
    (2, 0, -1, 0, 1274027, -3699111),
    (2, 0, 0, 0, 658314, -2955968),
    (0, 0, 2, 0, 213618, -569925),
    (0, 1, 0, 0, -185116, 48888),
    (0, 0, 0, 2, -114332, -3149),
    (2, 0, -2, 0, 58793, 246158),
    (2, -1, -1, 0, 57066, -152138),
    (2, 0, 1, 0, 53322, -170733),
    (2, -1, 0, 0, 45758, -204586),
    (0, 1, -1, 0, -40923, -129620),
    (1, 0, 0, 0, -34720, 108743),
    (0, 1, 1, 0, -30383, 104755),
    (2, 0, 0, -2, 15327, 10321),
    (0, 0, 1, 2, -12528, 0),
    (0, 0, 1, -2, 10980, 79661),
    (4, 0, -1, 0, 10675, -34782),
    (0, 0, 3, 0, 10034, -23210),
    (4, 0, -2, 0, 8548, -21636),
    (2, 1, -1, 0, -7888, 24208),
    (2, 1, 0, 0, -6766, 30824),
    (1, 0, -1, 0, -5163, -8379),
    (1, 1, 0, 0, 4987, -16675),
    (2, -1, 1, 0, 4036, -12831),
    (2, 0, 2, 0, 3994, -10445),
    (4, 0, 0, 0, 3861, -11650),
    (2, 0, -3, 0, 3665, 14403),
    (0, 1, -2, 0, -2689, -7003),
    (2, 0, -1, 2, -2602, 0),
    (2, -1, -2, 0, 2390, 10056),
    (1, 0, 1, 0, -2348, 6322),
    (2, -2, 0, 0, 2236, -9884),
]
# (D, M, M', F, sigma_b [1e-6 deg])
_MOON_B = [
    (0, 0, 0, 1, 5128122),
    (0, 0, 1, 1, 280602),
    (0, 0, 1, -1, 277693),
    (2, 0, 0, -1, 173237),
    (2, 0, -1, 1, 55413),
    (2, 0, -1, -1, 46271),
    (2, 0, 0, 1, 32573),
    (0, 0, 2, 1, 17198),
    (2, 0, 1, -1, 9266),
    (0, 0, 2, -1, 8822),
    (2, -1, 0, -1, 8216),
    (2, 0, -2, -1, 4324),
    (2, 0, 1, 1, 4200),
    (2, 1, 0, -1, -3359),
    (2, -1, -1, 1, 2463),
    (2, -1, 0, 1, 2211),
    (2, -1, -1, -1, 2065),
    (0, 1, -1, -1, -1870),
    (4, 0, -1, -1, 1828),
    (0, 1, 0, 1, -1794),
]
GENERAL_PRECESSION_DEG_PER_CY = 1.3969713   # 5029.0966"/cy, to bring ecliptic-of-date to J2000


def moon_ecliptic_of_date(epoch: Epoch) -> tuple[float, float, float]:
    """Geocentric ecliptic longitude, latitude [rad] (mean equinox of date) and distance [km]."""
    T = centuries_since_j2000(epoch)
    Lp = 218.3164477 + 481267.88123421 * T - 0.0015786 * T**2 + T**3 / 538841 - T**4 / 65194000
    D = 297.8501921 + 445267.1114034 * T - 0.0018819 * T**2 + T**3 / 545868 - T**4 / 113065000
    M = 357.5291092 + 35999.0502909 * T - 0.0001536 * T**2 + T**3 / 24490000
    Mp = 134.9633964 + 477198.8675055 * T + 0.0087414 * T**2 + T**3 / 69699 - T**4 / 14712000
    F = 93.2720950 + 483202.0175233 * T - 0.0036539 * T**2 - T**3 / 3526000 + T**4 / 863310000
    A1 = 119.75 + 131.849 * T
    A2 = 53.09 + 479264.290 * T
    A3 = 313.45 + 481266.484 * T
    Ecc = 1.0 - 0.002516 * T - 0.0000074 * T**2
    rad = math.radians
    Lp_r, D_r, M_r, Mp_r, F_r = rad(Lp), rad(D), rad(M), rad(Mp), rad(F)

    sl = sr = 0.0
    for d, m, mp, f, cl, cr in _MOON_LR:
        arg = d * D_r + m * M_r + mp * Mp_r + f * F_r
        w = Ecc ** abs(m)
        sl += w * cl * math.sin(arg)
        sr += w * cr * math.cos(arg)
    sl += 3958 * math.sin(rad(A1)) + 1962 * math.sin(Lp_r - F_r) + 318 * math.sin(rad(A2))
    sb = 0.0
    for d, m, mp, f, cb in _MOON_B:
        arg = d * D_r + m * M_r + mp * Mp_r + f * F_r
        sb += (Ecc ** abs(m)) * cb * math.sin(arg)
    sb += (-2235 * math.sin(Lp_r) + 382 * math.sin(rad(A3)) + 175 * math.sin(rad(A1) - F_r)
           + 175 * math.sin(rad(A1) + F_r) + 127 * math.sin(Lp_r - Mp_r) - 115 * math.sin(Lp_r + Mp_r))
    lon = rad(_normalize_deg(Lp + sl / 1e6))
    lat = rad(sb / 1e6)
    dist = 385000.56 + sr / 1e3
    return lon, lat, dist


def moon_position_geocentric(epoch: Epoch, frame: str = "ecliptic") -> np.ndarray:
    """Geocentric Moon position [km], J2000 ecliptic (default) or equatorial frame."""
    lon, lat, dist = moon_ecliptic_of_date(epoch)
    T = centuries_since_j2000(epoch)
    lon_j2000 = lon - math.radians(GENERAL_PRECESSION_DEG_PER_CY * T)
    r = dist * np.array([math.cos(lat) * math.cos(lon_j2000),
                         math.cos(lat) * math.sin(lon_j2000),
                         math.sin(lat)])
    if frame == "equatorial":
        r = ecliptic_to_equatorial(r)
    return r


def moon_state_geocentric(epoch: Epoch, frame: str = "ecliptic",
                          dt_s: float = 60.0) -> tuple[np.ndarray, np.ndarray]:
    """Geocentric Moon position [km] and velocity [km/s] (central-difference velocity)."""
    jd = to_jd(epoch)
    h = dt_s / 86400.0
    r = moon_position_geocentric(jd, frame)
    rp = moon_position_geocentric(jd + h, frame)
    rm = moon_position_geocentric(jd - h, frame)
    v = (rp - rm) / (2.0 * dt_s)
    return r, v


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------
def heliocentric_longitude(name: str, epoch: Epoch) -> float:
    """Ecliptic longitude of a planet [deg, 0-360)."""
    r, _ = planet_state(name, epoch)
    return math.degrees(math.atan2(r[1], r[0])) % 360.0


def phase_angle(origin: str, target: str, epoch: Epoch) -> float:
    """Heliocentric angle from ``origin`` to ``target`` measured in the direction
    of orbital motion [deg, 0-360) -- the 'lead angle' for launch windows."""
    return (heliocentric_longitude(target, epoch) - heliocentric_longitude(origin, epoch)) % 360.0


def orbit_samples(name: str, epoch: Epoch, n: int = 361, frame: str = "ecliptic") -> np.ndarray:
    """Positions around one full orbit of a body starting at ``epoch`` (for plotting)."""
    from .kepler import period as _period
    body = get_body(name)
    if body.name == "Moon":
        jd0 = to_jd(epoch)
        ts = jd0 + np.linspace(0.0, 27.321661, n)
        return np.array([moon_position_geocentric(float(t), frame) for t in ts])
    el = planet_elements(body.name, epoch)
    P = _period(MU_SUN, el["a"])
    jd0 = to_jd(epoch)
    ts = jd0 + np.linspace(0.0, P / 86400.0, n)
    return np.array([planet_state(body.name, float(t), frame)[0] for t in ts])
