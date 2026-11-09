"""Earth-Moon transfers: Hohmann estimate vs BMW patched conic; Apollo 11 check.

Run:  python examples/03_lunar_mission.py [outdir]
"""
import sys, os, math
from modern_aerospace_propulsion.mission_analysis import lunar_hohmann_estimate, design_lunar_transfer, lunar_budget
from modern_aerospace_propulsion.mission_analysis.plotting import plot_lunar_transfer, plot_budget

out = sys.argv[1] if len(sys.argv) > 1 else "figures"
os.makedirs(out, exist_ok=True)

print(lunar_hohmann_estimate(185, 100).summary())

print("\nPatched-conic transfers to a 100 km perilune for several injection speeds (185 km parking orbit):")
print(f"{'v0 [km/s]':>9s} {'dir':>5s} {'lambda1':>8s} {'TOF geo':>8s} {'TOF tot':>8s} {'TLI dv':>7s} {'LOI dv':>7s} {'v_inf':>6s}")
for v0 in (10.93, 10.94, 10.96, 11.00, 11.05):
    for s in design_lunar_transfer(185, 100, v0=v0, direction="either"):
        print(f"{v0:9.3f} {'retro' if s.h2 < 0 else 'pro':>5s} {math.degrees(s.lambda1):8.1f} {s.tof/3600:8.1f} "
              f"{s.tof_total/3600:8.1f} {s.dv_tli:7.3f} {s.loi_dv():7.3f} {s.vinf:6.3f}")

print("\nApollo 11: TLI 10.832 km/s at 336 km, FPA ~7 deg; actual TLI->LOI 73.0 h, LOI-1 0.891 km/s")
s = design_lunar_transfer(336.3, 110, v0=10.832, phi0_deg=7.2, direction="retrograde")[0]
print(s.summary())
plot_lunar_transfer(s).savefig(f"{out}/lunar_patched_conic_apollo11.png", dpi=200)

print("\nBudget: Earth (185 km) -> 100 km lunar orbit -> land -> ascend -> return, 3-day transfer")
b, d = lunar_budget(185, 100, tof_hours=72, land=True, ascend=True, return_to_earth=True)
print(b.table())
plot_budget(b).savefig(f"{out}/budget_lunar_landing.png", dpi=200)
print("figures written to", out)
