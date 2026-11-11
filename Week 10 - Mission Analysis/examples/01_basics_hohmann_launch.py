"""Basics for the Week-10 lecture: dv to LEO, Hohmann estimates, phase angles.

Run:  python examples/01_basics_hohmann_launch.py
"""
import math
from sfmission import *

print("=== Delta-v to LEO (correcting Week-9 slide 32: dv = v_orb + losses - v_site) ===")
for lat, inc, label in ((28.5, None, "KSC due east"), (28.5, 51.6, "KSC to ISS inclination"),
                        (5.2, None, "Kourou due east"), (45.9, None, "Baikonur due east"), (28.5, 90.0, "KSC polar")):
    print(launch_to_orbit(300, inc, lat).summary() + f"   [{label}]")

print("\n=== Hohmann between circular orbits about Earth ===")
h = hohmann(EARTH.mu, EARTH.radius + 300, 42164)
print(h.summary("LEO 300 km -> GEO"))
pc = hohmann_with_plane_change(EARTH.mu, EARTH.radius + 300, 42164, math.radians(28.5))
print(f"  with 28.5 deg plane change: dv1={pc.dv1:.3f} (di1={math.degrees(pc.di1):.2f} deg) "
      f"dv2={pc.dv2:.3f} (di2={math.degrees(pc.di2):.2f} deg) total={pc.dv_total:.3f} km/s")

print("\n=== Interplanetary Hohmann estimates (circular coplanar planet orbits) ===")
for target in ("Venus", "Mars", "Jupiter", "Saturn"):
    r = hohmann_interplanetary("Earth", target, park_alt=300, capture_peri_alt=None)
    print(f"{target:8s}: v_inf dep {r.vinf_dep:5.2f} km/s (C3 {r.vinf_dep**2:6.1f}), inj dv {r.departure.dv:5.2f} km/s, "
          f"arr v_inf {r.vinf_arr:5.2f}, TOF {r.tof/86400:6.0f} d, phase {math.degrees(r.phase_angle):6.1f} deg, "
          f"synodic {r.synodic_period/86400/365.25:4.2f} yr")

print("\n=== Current Earth-Mars phase angle vs Hohmann requirement ===")
print(f"2026-11-12: phase angle {phase_angle('Earth','Mars','2026-11-12'):.1f} deg; "
      f"Hohmann needs {math.degrees(hohmann_phase_angle(MU_SUN, EARTH.a_mean, MARS.a_mean)):.1f} deg")

print("\n=== Rocket equation / staging (design-roadmap steps 2-4) ===")
dv = 5.7
for isp in (320, 450, 900, 3000):
    print(f"dv {dv} km/s at Isp {isp:5d} s -> propellant fraction {propellant_fraction_for_dv(isp, dv):.3f}")
print(f"Isp required for 9.3 km/s with zeta=0.90 (SSTO): {required_isp(9.3, 0.90):.0f} s")
print(design_vehicle(9.3, 10000, [300, 450], [0.08, 0.10], ["booster", "upper"]).table())
