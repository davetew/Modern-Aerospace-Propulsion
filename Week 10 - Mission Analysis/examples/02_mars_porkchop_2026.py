"""Earth -> Mars launch window opening now (Oct 2026 - Jan 2027): porkchop plots.

Run:  python examples/02_mars_porkchop_2026.py  [outdir]
"""
import sys, os
from modern_aerospace_propulsion.mission_analysis import porkchop, heliocentric_leg, next_launch_windows
from modern_aerospace_propulsion.mission_analysis.plotting import plot_porkchop, plot_transfer

out = sys.argv[1] if len(sys.argv) > 1 else "figures"
os.makedirs(out, exist_ok=True)

pc = porkchop("Earth", "Mars", "2026-09-01", "2027-02-15", 120, 450, n_dep=110, n_tof=110,
              park_alt=300, capture_peri_alt=400)
print(pc.summary())
plot_porkchop(pc, "dv_total").savefig(f"{out}/porkchop_earth_mars_2026_dvtotal.png", dpi=200)
plot_porkchop(pc, "c3", y="tof").savefig(f"{out}/porkchop_earth_mars_2026_c3.png", dpi=200)
plot_porkchop(pc, "vinf_arr").savefig(f"{out}/porkchop_earth_mars_2026_vinf.png", dpi=200)

o = pc.optimum("dv_total")
leg = heliocentric_leg("Earth", "Mars", o["depart"], o["arrive"])
print(leg.summary())
plot_transfer(leg, title=f"Earth → Mars, minimum-Δv transfer of the 2026 window").savefig(
    f"{out}/transfer_earth_mars_2026.png", dpi=200)

# the type-I alternative (fast transfer) in the same window
pc1 = porkchop("Earth", "Mars", "2026-10-01", "2027-01-15", 150, 260, n_dep=60, n_tof=60, park_alt=300, capture_peri_alt=400)
o1 = pc1.optimum("dv_total")
print(f"Type-I optimum: depart {o1['depart'][:10]}, TOF {o1['tof_days']:.0f} d, C3 {o1['c3']:.2f}, "
      f"v_inf,arr {o1['vinf_arr']:.3f}, total dv {o1['dv_total']:.3f} km/s")

print("\nUpcoming Earth->Mars opportunities (type I and II minima):")
for w in next_launch_windows("Earth", "Mars", "2026-01-01", years=9, park_alt=300, capture_peri_alt=400):
    print(f"  {w['depart']} type {w['type']:2s} TOF {w['tof_days']:5.0f} d  C3 {w['c3']:5.2f}  v_inf,arr {w['vinf_arr']:.2f}  dv {w['dv_total']:.3f}")
print("figures written to", out)
