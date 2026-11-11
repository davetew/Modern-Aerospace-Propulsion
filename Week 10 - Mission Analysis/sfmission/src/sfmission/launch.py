"""Launch Delta-v to orbit: ideal velocity, launch-site rotation credit, losses.

  dv_launch = |v_orbit_vec - v_site_vec| + dv_gravity + dv_drag + dv_steering

The vector difference accounts for the launch-site inertial velocity
(omega R cos(lat), eastward) and the launch azimuth needed to reach the target
inclination (cos i = cos(lat) sin(Az)).  For a due-east launch (i = lat) it
reduces to v_circ - omega R cos(lat).  Losses are given as a band (defaults:
gravity 1.0-1.5, drag 0.05-0.15, steering 0.1-0.4 km/s -> total ~1.15-2.05
km/s for a LEO launcher; Sutton §4.3, typical values).

Sign convention (correcting Week-9 slide 32):
  dv_LEO = v_orbital + dv_losses - v_site   ~ 7.8 + (1.5..2.0) - 0.41 ~ 8.9-9.4 km/s
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .bodies import Body, get_body, EARTH
from .kepler import circular_velocity, vis_viva


@dataclass
class LossBand:
    gravity: tuple[float, float] = (1.0, 1.5)
    drag: tuple[float, float] = (0.05, 0.15)
    steering: tuple[float, float] = (0.1, 0.4)

    @property
    def low(self) -> float:
        return self.gravity[0] + self.drag[0] + self.steering[0]

    @property
    def high(self) -> float:
        return self.gravity[1] + self.drag[1] + self.steering[1]


@dataclass
class LaunchResult:
    body: str
    altitude: float          # km
    inclination: float       # rad
    latitude: float          # rad
    azimuth: float           # rad, launch azimuth (from north, clockwise)
    v_orbit: float           # km/s, required inertial speed at insertion
    v_site: float            # km/s, inertial speed of the launch site
    dv_rotation_credit: float  # km/s, v_orbit - |v_orbit_vec - v_site_vec|
    dv_ideal: float          # km/s, |v_orbit_vec - v_site_vec|
    losses: LossBand
    retrograde: bool = False

    @property
    def dv_low(self) -> float:
        return self.dv_ideal + self.losses.low

    @property
    def dv_high(self) -> float:
        return self.dv_ideal + self.losses.high

    @property
    def dv_mid(self) -> float:
        return 0.5 * (self.dv_low + self.dv_high)

    def summary(self) -> str:
        return (f"Launch from {self.body}, lat {math.degrees(self.latitude):.2f} deg to {self.altitude:.0f} km, "
                f"i={math.degrees(self.inclination):.1f} deg (Az={math.degrees(self.azimuth):.1f} deg): "
                f"v_orbit={self.v_orbit:.3f}, site credit={self.dv_rotation_credit:.3f}, ideal dv={self.dv_ideal:.3f}, "
                f"losses {self.losses.low:.2f}-{self.losses.high:.2f} -> dv={self.dv_low:.2f}-{self.dv_high:.2f} km/s")


def launch_to_orbit(altitude: float = 300.0, inclination_deg: float | None = None,
                    latitude_deg: float = 28.5, body: Body | str = EARTH,
                    losses: LossBand | None = None, apoapsis_alt: float | None = None,
                    retrograde: bool = False) -> LaunchResult:
    """Delta-v to reach a circular orbit (or ``altitude`` x ``apoapsis_alt``
    ellipse, insertion at periapsis) from a launch site at ``latitude_deg``.

    inclination_deg defaults to the latitude (due-east launch, minimum dv).
    Must satisfy i >= |lat| (direct) or i <= 180 - |lat| (retrograde).
    """
    b = get_body(body)
    losses = losses or LossBand()
    lat = math.radians(latitude_deg)
    inc = lat if inclination_deg is None else math.radians(inclination_deg)
    if retrograde:
        inc = math.pi - inc if inc < math.pi / 2 else inc
    cos_lat = math.cos(lat)
    if abs(math.cos(inc)) > cos_lat + 1e-12:
        raise ValueError(f"inclination {math.degrees(inc):.1f} deg not reachable from latitude "
                         f"{latitude_deg:.1f} deg without a plane change (need i >= |lat|)")
    # launch azimuth: cos i = cos(lat) sin(Az)  (Az from north, eastward positive)
    sin_az = math.cos(inc) / cos_lat
    az = math.asin(max(-1.0, min(1.0, sin_az)))        # northerly launch branch
    r = b.radius + altitude
    if apoapsis_alt is None:
        v_orb = circular_velocity(b.mu, r)
    else:
        v_orb = vis_viva(b.mu, r, 0.5 * (r + b.radius + apoapsis_alt))
    omega = b.omega or 0.0
    v_site = omega * b.radius * cos_lat
    # local ENU frame: orbital velocity direction (horizontal) at azimuth Az
    v_orb_vec = v_orb * np.array([math.sin(az), math.cos(az), 0.0])   # (east, north, up)
    v_site_vec = np.array([v_site, 0.0, 0.0])
    dv_ideal = float(np.linalg.norm(v_orb_vec - v_site_vec))
    return LaunchResult(b.name, altitude, inc, lat, az, v_orb, v_site, v_orb - dv_ideal,
                        dv_ideal, losses, retrograde)


def dv_to_escape_from_orbit(body: Body | str, altitude: float, vinf: float = 0.0) -> float:
    """Delta-v from a circular orbit at ``altitude`` to a hyperbola with ``vinf``."""
    b = get_body(body)
    r = b.radius + altitude
    return math.sqrt(vinf ** 2 + 2 * b.mu / r) - circular_velocity(b.mu, r)


def gravity_loss_estimate(burn_time: float, mean_pitch_deg: float = 60.0, g: float = 9.80665e-3) -> float:
    """Crude gravity-loss estimate dv_g ~ g * t_burn * <sin(pitch)> [km/s]
    (pitch measured from horizontal; a vertical ascent gives g*t).  burn_time in s."""
    return g * burn_time * math.sin(math.radians(mean_pitch_deg))
