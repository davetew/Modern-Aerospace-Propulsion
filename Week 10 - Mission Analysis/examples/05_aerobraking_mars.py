"""Aerocapture / aerobraking at Mars as dv-budget levers; analytic vs numerical pass.

Run:  python examples/05_aerobraking_mars.py
"""
import math
from sfmission import *

vinf = 2.65   # km/s, typical Mars arrival
print(capture_dv("Mars", vinf, 400).summary())
print(aerocapture_credit("Mars", vinf, 400).summary())
for beta in (30, 50, 100):
    print(aerobraking_campaign("Mars", vinf, 110, 40000, 400, beta=beta).summary())

print("\nOne-pass check (periapsis 110 km, apoapsis 40,000 km, beta 50 kg/m2):")
num = aerobraking_pass_dv_numeric("Mars", 110, 40000, 50)
rp = MARS.radius + 110
Vp = vis_viva(MARS.mu, rp, 0.5 * (rp + MARS.radius + 40000))
ana = aerobraking_pass_dv(num["rho_p"], Vp, MARS.atmosphere.H, rp, 50, MARS.mu)
sl = aerobraking_pass_dv(num["rho_p"], Vp, MARS.atmosphere.H, rp, 50)
print(f"  rho_p = {num['rho_p']:.2e} kg/m3, V_p = {Vp:.3f} km/s, peak deceleration {num['peak_decel_g']:.3f} g")
print(f"  numerical integration : dv = {num['dv_pass']*1000:.3f} m/s, apoapsis {num['apo_alt_before']:.0f} -> {num['apo_alt_after']:.0f} km")
print(f"  analytic, gravity-aware: dv = {ana*1000:.3f} m/s  (rddot_p = V_p^2/r_p - mu/r_p^2)")
print(f"  analytic, straight-line: dv = {sl*1000:.3f} m/s  (rddot_p = V_p^2/r_p; Week-11 slide form) -> low by factor "
      f"{ana/sl:.3f} = sqrt(V_p^2/(V_p^2 - mu/r_p))")
