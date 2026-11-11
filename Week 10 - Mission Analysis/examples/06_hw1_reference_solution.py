"""HW 1 reference solution -- Mission dv budget and staging (design-roadmap steps 1-4).

Problem (as proposed in the module review):
  For your team's mission, (a) estimate dv to orbit with a loss band and the
  launch-site credit; (b) compute the interplanetary (or lunar) legs with the
  patched-conic model -- Hohmann estimate, then the dated Lambert solution --
  including an aerocapture option where applicable; (c) from the post-launch
  dv and an assumed propellant mass fraction find the required Isp; (d) size
  a two-stage launcher for the injected mass with the optimum dv split.

Worked here for a Mars orbiter launched in the 2026/27 window from KSC.
Run:  python examples/06_hw1_reference_solution.py [outdir]
"""
import sys, os, math
from sfmission import *
from sfmission.plotting import plot_budget, plot_staging_tradeoff

out = sys.argv[1] if len(sys.argv) > 1 else "figures"
os.makedirs(out, exist_ok=True)

# (a) launch
L = launch_to_orbit(300, None, 28.5)
print("(a)", L.summary())

# (b) Hohmann estimate, then the dated solution at the window optimum
h = hohmann_interplanetary("Earth", "Mars", 300, 400)
print("(b) Hohmann estimate:\n   " + h.summary().replace("\n", "\n   "))
pc = porkchop("Earth", "Mars", "2026-09-15", "2027-02-01", 150, 450, n_dep=80, n_tof=80, park_alt=300, capture_peri_alt=400)
o = pc.optimum("dv_total")
print(f"    Porkchop optimum: depart {o['depart'][:10]}, arrive {o['arrive'][:10]} (TOF {o['tof_days']:.0f} d), "
      f"C3 {o['c3']:.2f} km2/s2, v_inf,arr {o['vinf_arr']:.3f} km/s")

budgets = {}
for aero in (False, True):
    b, m = interplanetary_budget("Earth", "Mars", o["depart"], o["arrive"], park_alt=300, launch_lat_deg=28.5,
                                 capture_peri_alt=400, aerocapture=aero, margin=0.05)
    budgets[aero] = b
    print(f"\n    --- {'with aerocapture' if aero else 'propulsive capture'} ---")
    print("    " + b.table().replace("\n", "\n    "))
    plot_budget(b).savefig(f"{out}/hw1_budget_{'aero' if aero else 'prop'}.png", dpi=200)

# (c) required Isp for the post-launch dv at an assumed propellant fraction
for aero, b in budgets.items():
    dv_post = b.post_launch_propulsive()
    for zeta in (0.5, 0.6, 0.7):
        print(f"(c) {'aero' if aero else 'prop'}: post-launch dv {dv_post:.3f} km/s, zeta={zeta:.1f} -> Isp required {b.required_isp(zeta):.0f} s")

# propellant for a 1000 kg final mass spacecraft with a 320 s storable engine
rows = budgets[False].size_propellant(1000.0, 320.0)
print("\n    Spacecraft propellant (final mass 1000 kg, Isp 320 s, propulsive capture):")
for r in rows:
    print(f"      {r['name']:35s} dv {r['dv']:.3f}  m_before {r['m_before']:8.1f} kg  m_prop {r['m_prop']:7.1f} kg")
m_inj = rows[0]["m_before"]
print(f"    injected mass on the escape hyperbola = {m_inj:.0f} kg")

# (d) two-stage launcher delivering m_inj onto C3 = o['c3'] from KSC
dv_launcher = L.dv_mid + budgets[False].legs[1].dv   # launch to LEO (mid band) + injection burn
print(f"\n(d) launcher dv = {L.dv_mid:.2f} (to LEO) + {budgets[False].legs[1].dv:.3f} (injection) = {dv_launcher:.3f} km/s")
v = design_vehicle(dv_launcher, m_inj, [300, 450], [0.07, 0.10], ["kerolox booster", "hydrolox upper"])
print(v.table())
plot_staging_tradeoff(dv_launcher, 350, 0.08).savefig(f"{out}/hw1_staging_tradeoff.png", dpi=200)
print("figures written to", out)
