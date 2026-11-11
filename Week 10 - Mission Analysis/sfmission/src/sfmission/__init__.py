"""sfmission -- patched-conic mission analysis for lunar and interplanetary design.

MECE 4305 (Columbia), Rocket Module Week 10; lives in "Week 10 - Mission Analysis/sfmission/" of the course repository.

Quick start
-----------
>>> from sfmission import hohmann_interplanetary, porkchop, interplanetary_mission
>>> print(hohmann_interplanetary("Earth", "Mars", park_alt=300, capture_peri_alt=300).summary())
>>> pc = porkchop("Earth", "Mars", "2026-09-01", "2027-02-01", 120, 400)
>>> print(pc.summary())

Modules: bodies, kepler, ephemeris, lambert, transfers, patched_conics, lunar,
launch, aerobraking, staging, budget, porkchop, plotting, mcp_server, cli.
Units: km, s, kg; radians in low-level functions, degrees where the argument
name ends in ``_deg``; epochs as datetime / ISO string / Julian Date.
"""
from .bodies import (AU_KM, G0, MU_SUN, BODIES, PLANETS, Body, get_body,
                     SUN, MERCURY, VENUS, EARTH, MOON, MARS, JUPITER, SATURN, URANUS, NEPTUNE, PLUTO)
from .kepler import (vis_viva, circular_velocity, escape_velocity, period, coe_from_rv, rv_from_coe,
                     propagate, turn_angle, hyperbolic_periapsis_speed)
from .ephemeris import planet_state, body_state, earth_state, moon_state_geocentric, to_jd, jd_to_iso, phase_angle
from .lambert import lambert, LambertError
from .transfers import (hohmann, bielliptic, plane_change_dv, hohmann_with_plane_change, synodic_period,
                        hohmann_phase_angle, wait_for_phase, hohmann_round_trip)
from .patched_conics import (departure_dv, capture_dv, optimal_capture_periapsis, flyby_unpowered, flyby_powered,
                             heliocentric_leg, interplanetary_mission, hohmann_interplanetary,
                             flyby_max_heliocentric_dv)
from .lunar import lunar_hohmann_estimate, lunar_patched_conic, design_lunar_transfer
from .launch import launch_to_orbit, LossBand, dv_to_escape_from_orbit
from .aerobraking import aerocapture_credit, aerobraking_pass_dv, aerobraking_campaign, aerobraking_pass_dv_numeric
from .staging import (tsiolkovsky_dv, mass_ratio_for_dv, propellant_fraction_for_dv, required_isp, Stage, Vehicle,
                      size_vehicle, optimal_dv_split, design_vehicle, Burn, size_burn_sequence)
from .budget import DeltaVBudget, DeltaVLeg, interplanetary_budget, lunar_budget
from .porkchop import porkchop, next_launch_windows, Porkchop

__version__ = "0.1.0"
