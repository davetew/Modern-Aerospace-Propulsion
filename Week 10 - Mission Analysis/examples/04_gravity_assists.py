"""Gravity assists: Europa Clipper (Earth-Mars-Earth-Jupiter) and Voyager 2 (Earth-Jupiter-Saturn).

Run:  python examples/04_gravity_assists.py [outdir]
"""
import sys, os, math
from sfmission import interplanetary_mission, hohmann_interplanetary, flyby_max_heliocentric_dv
from sfmission.plotting import plot_transfer

out = sys.argv[1] if len(sys.argv) > 1 else "figures"
os.makedirs(out, exist_ok=True)

print("Europa Clipper: launch 2024-10-14, Mars flyby 2025-03-01, Earth flyby 2026-12-03, Jupiter arrival 2030-04-11")
m = interplanetary_mission(["Earth", "Mars", "Earth", "Jupiter"], ["2024-10-14", "2025-03-01", "2026-12-03", "2030-04-11"],
                           park_alt=300, capture_peri_alt=None, flyby_min_alt=300)
print(m.summary())
direct = hohmann_interplanetary("Earth", "Jupiter", 300, None)
print(f"\nCompare direct Hohmann to Jupiter: C3 {direct.vinf_dep**2:.1f} km2/s2, injection dv {direct.departure.dv:.3f} km/s, "
      f"TOF {direct.tof/86400/365.25:.2f} yr  vs  MEGA C3 {m.departure.c3:.1f}, dv {m.departure.dv:.3f} km/s, TOF {m.tof_days/365.25:.2f} yr")
plot_transfer(m.legs, title="Europa Clipper Mars-Earth gravity assist (patched conic)").savefig(
    f"{out}/transfer_europa_clipper_mega.png", dpi=200)

print("\nVoyager 2: launch 1977-08-20, Jupiter 1979-07-09, Saturn 1981-08-25")
v = interplanetary_mission(["Earth", "Jupiter", "Saturn"], ["1977-08-20", "1979-07-09", "1981-08-25"],
                           park_alt=185, capture_peri_alt=None, flyby_min_alt=100000)
print(v.summary())
plot_transfer(v.legs, title="Voyager 2 Earth-Jupiter-Saturn (patched conic)").savefig(f"{out}/transfer_voyager2.png", dpi=200)

print("\nMaximum heliocentric dv from an unpowered flyby (v_inf = 8 km/s, 1000 km altitude):")
for b in ("Venus", "Earth", "Mars", "Jupiter", "Saturn"):
    print(f"  {b:8s} {flyby_max_heliocentric_dv(b, 8.0, 1000.0):6.2f} km/s")
print("figures written to", out)
