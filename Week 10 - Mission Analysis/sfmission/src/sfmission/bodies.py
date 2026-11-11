"""Physical and orbital constants for solar-system bodies.

Units throughout the package: km, s, kg, rad (low-level) -- see README.

Sources
-------
* GM values: JPL planetary ephemeris DE440 (Park et al. 2021), as listed at
  https://ssd.jpl.nasa.gov/astro_par.html  (planet-system GMs for the outer
  planets, i.e. planet + satellites, which is what patched-conic flybys need).
* Mean radii: IAU 2015 nominal values / NASA planetary fact sheets.
* Atmosphere models: single-scale-height exponential fits, adequate for
  first-order aerobraking/aerocapture estimates only (see aerobraking.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

AU_KM = 149_597_870.7          # km, IAU 2012 definition
G0 = 9.80665e-3                # km/s^2, standard gravity (Isp definition)
DAY_S = 86_400.0
YEAR_S = 365.25 * DAY_S
MU_SUN = 1.32712440041279419e11  # km^3/s^2 (DE440)


@dataclass(frozen=True)
class Atmosphere:
    """Exponential atmosphere rho(h) = rho0 * exp(-(h - h0)/H).

    rho0 [kg/m^3] at reference altitude h0 [km]; scale height H [km].
    The fits are tuned to the aerobraking altitude band (roughly 90-150 km
    for Earth and Mars), not to the surface.
    """
    rho0: float
    h0: float
    H: float

    def density(self, h_km: float) -> float:
        import math
        return self.rho0 * math.exp(-(h_km - self.h0) / self.H)


@dataclass(frozen=True)
class Body:
    name: str
    mu: float                              # km^3/s^2
    radius: float                          # km, mean equatorial
    parent: Optional[str] = None           # body it orbits
    a_mean: Optional[float] = None         # km, mean orbital semi-major axis about parent
    mass_ratio_to_parent: Optional[float] = None  # m/M for SOI
    rotation_period: Optional[float] = None       # s, sidereal
    atmosphere: Optional[Atmosphere] = None
    atmosphere_interface_alt: Optional[float] = None  # km, "entry interface" altitude

    # ---- derived quantities --------------------------------------------
    @property
    def soi(self) -> Optional[float]:
        """Laplace sphere-of-influence radius r_SOI = a (m/M)^(2/5) [km]."""
        if self.a_mean is None or self.mass_ratio_to_parent is None:
            return None
        return self.a_mean * self.mass_ratio_to_parent ** 0.4

    @property
    def omega(self) -> Optional[float]:
        """Sidereal rotation rate [rad/s]."""
        import math
        if self.rotation_period is None:
            return None
        return 2.0 * math.pi / self.rotation_period

    def v_circ(self, r_km: float) -> float:
        import math
        return math.sqrt(self.mu / r_km)

    def v_esc(self, r_km: float) -> float:
        import math
        return math.sqrt(2.0 * self.mu / r_km)

    def surface_gravity(self) -> float:
        """[km/s^2]"""
        return self.mu / self.radius ** 2

    def equatorial_rotation_speed(self) -> Optional[float]:
        """Surface speed at the equator [km/s]."""
        if self.omega is None:
            return None
        return self.omega * self.radius


# ---------------------------------------------------------------------------
# Registry.  GMs in km^3/s^2 (DE440).  Outer planets use the *system* GM.
# ---------------------------------------------------------------------------
_SUN = Body("Sun", MU_SUN, 695_700.0, rotation_period=25.38 * DAY_S)

_MERCURY = Body("Mercury", 22_031.868551, 2_439.7, "Sun", 0.38709927 * AU_KM,
                22_031.868551 / MU_SUN, 58.646 * DAY_S)
_VENUS = Body("Venus", 324_858.592, 6_051.8, "Sun", 0.72333566 * AU_KM,
              324_858.592 / MU_SUN, -243.025 * DAY_S,
              Atmosphere(rho0=65.0, h0=0.0, H=15.9), 200.0)
_EARTH = Body("Earth", 398_600.435507, 6_378.137, "Sun", 1.00000261 * AU_KM,
              (398_600.435507 + 4_902.800118) / MU_SUN, 86_164.0905,
              # fit through US Standard Atmosphere 1976 at 100-150 km
              # (rho(100 km)=5.6e-7, rho(120 km)=2.2e-8 kg/m^3 -> H~7.3 km)
              Atmosphere(rho0=5.6e-7, h0=100.0, H=7.3), 122.0)
_MOON = Body("Moon", 4_902.800118, 1_737.4, "Earth", 384_400.0,
             4_902.800118 / 398_600.435507, 27.321661 * DAY_S)
_MARS = Body("Mars", 42_828.375816, 3_396.19, "Sun", 1.52371034 * AU_KM,
             42_828.375816 / MU_SUN, 88_642.66,
             # Mars aerobraking band: rho(100 km) ~ 1.5e-7, rho(130 km) ~ 9e-9 kg/m^3
             Atmosphere(rho0=1.5e-7, h0=100.0, H=10.5), 125.0)
_JUPITER = Body("Jupiter", 126_712_764.1, 71_492.0, "Sun", 5.20288700 * AU_KM,
                126_712_764.1 / MU_SUN, 9.925 * 3600.0)
_SATURN = Body("Saturn", 37_940_584.8418, 60_268.0, "Sun", 9.53667594 * AU_KM,
               37_940_584.8418 / MU_SUN, 10.656 * 3600.0)
_URANUS = Body("Uranus", 5_794_556.4, 25_559.0, "Sun", 19.18916464 * AU_KM,
               5_794_556.4 / MU_SUN, -17.24 * 3600.0)
_NEPTUNE = Body("Neptune", 6_836_527.10058, 24_764.0, "Sun", 30.06992276 * AU_KM,
                6_836_527.10058 / MU_SUN, 16.11 * 3600.0)
_PLUTO = Body("Pluto", 975.5, 1_188.3, "Sun", 39.48 * AU_KM, 975.5 / MU_SUN,
              -6.3872 * DAY_S)

BODIES: dict[str, Body] = {
    b.name.lower(): b for b in
    (_SUN, _MERCURY, _VENUS, _EARTH, _MOON, _MARS, _JUPITER, _SATURN,
     _URANUS, _NEPTUNE, _PLUTO)
}

PLANETS = ("Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn",
           "Uranus", "Neptune")


def get_body(name: str | Body) -> Body:
    if isinstance(name, Body):
        return name
    key = name.strip().lower()
    if key in ("em bary", "emb", "earth-moon barycenter"):
        key = "earth"
    try:
        return BODIES[key]
    except KeyError:
        raise KeyError(f"Unknown body '{name}'. Known: {sorted(BODIES)}") from None


# Convenience module-level handles
SUN, MERCURY, VENUS, EARTH, MOON, MARS = _SUN, _MERCURY, _VENUS, _EARTH, _MOON, _MARS
JUPITER, SATURN, URANUS, NEPTUNE, PLUTO = _JUPITER, _SATURN, _URANUS, _NEPTUNE, _PLUTO
