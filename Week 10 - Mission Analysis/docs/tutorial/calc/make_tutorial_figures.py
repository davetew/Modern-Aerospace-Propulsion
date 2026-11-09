"""Quantitative figures for the orbital-mechanics tutorial (figs/*.png).

Run from the repository root:
    PYTHONPATH=. python "Week 10 - Mission Analysis/docs/tutorial/calc/make_tutorial_figures.py"
"""
from __future__ import annotations

import math
import os
import shutil

import numpy as np
import matplotlib
matplotlib.use("Agg")

from modern_aerospace_propulsion.mission_analysis import *
from modern_aerospace_propulsion.mission_analysis.kepler import (circular_velocity, escape_velocity, vis_viva,
                                                                   apsides_from_rp_ra, kepler_E, true_anomaly_from_E,
                                                                   stumpff_C, stumpff_S, turn_angle)
from modern_aerospace_propulsion.mission_analysis.plotting import _mpl, SERIES, INK, INK2, MUTED, AXIS

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(os.path.dirname(HERE), "figs")
SHARED = os.path.join(os.path.dirname(os.path.dirname(HERE)), "figures")
os.makedirs(FIGS, exist_ok=True)
plt, _ = _mpl()
mu, R = EARTH.mu, EARTH.radius


def save(fig, name):
    fig.savefig(os.path.join(FIGS, name), dpi=200)
    plt.close(fig)


# 1. vis-viva: speed vs radius ------------------------------------------------
r = np.linspace(R + 200, 60000, 400)
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(r / 1e3, np.sqrt(mu / r), color=SERIES[0], lw=2, label=r"circular $\sqrt{\mu/r}$")
ax.plot(r / 1e3, np.sqrt(2 * mu / r), color=SERIES[1], lw=2, label=r"escape $\sqrt{2\mu/r}$")
rp, ra = R + 300, R + 35786
a_t, e_t = apsides_from_rp_ra(rp, ra)
rr = np.linspace(rp, ra, 300)
ax.plot(rr / 1e3, np.sqrt(mu * (2 / rr - 1 / a_t)), color=SERIES[2], lw=2, label=r"GTO ellipse, $a=%.0f$ km" % a_t)
ax.plot([rp / 1e3, ra / 1e3], [vis_viva(mu, rp, a_t), vis_viva(mu, ra, a_t)], "o", color=SERIES[2], ms=6)
vinf = 3.0
ax.plot(r / 1e3, np.sqrt(vinf ** 2 + 2 * mu / r), color=SERIES[6], lw=2, ls="--", label=r"hyperbola, $v_\infty=3$ km/s")
ax.axhline(vinf, color=SERIES[6], lw=0.8, ls=":")
ax.text(50, vinf + 0.15, r"$v_\infty$", color=SERIES[6], fontsize=9)
ax.set_xlabel("radius [10³ km]"); ax.set_ylabel("speed [km/s]"); ax.set_ylim(0, 12)
ax.legend(fontsize=8.5); ax.set_title("Speed on conics about Earth (vis-viva)", loc="left", fontsize=10.5)
fig.tight_layout(); save(fig, "tut_visviva.png")

# 2. anomalies vs time --------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.2))
M = np.linspace(0, 2 * math.pi, 400)
for k, e in enumerate((0.0, 0.3, 0.7306)):
    E = np.array([kepler_E(m, e) % (2 * math.pi) for m in M])
    nu = np.array([true_anomaly_from_E(En, e) % (2 * math.pi) for En in E])
    ax.plot(np.degrees(M), np.degrees(nu), color=SERIES[k], lw=2, label=f"$\\nu$, e = {e:.2f}" + (" (GTO)" if k == 2 else ""))
    if k == 2:
        ax.plot(np.degrees(M), np.degrees(E), color=SERIES[k], lw=1.2, ls="--", label=f"$E$, e = {e:.2f}")
ax.set_xlabel("mean anomaly $M = n\\,(t - t_p)$ [deg]  (∝ time since periapsis)")
ax.set_ylabel("true anomaly $\\nu$ (and $E$) [deg]")
ax.set_xticks(range(0, 361, 60)); ax.set_yticks(range(0, 361, 60))
ax.legend(fontsize=8.5, loc="lower right"); ax.set_title("Where you are vs. when you are: Kepler's equation", loc="left", fontsize=10.5)
fig.tight_layout(); save(fig, "tut_anomalies.png")

# 3. normalized Hohmann ---------------------------------------------------------
ratios = np.logspace(0.01, 2.5, 600)
tot = np.array([hohmann(1.0, 1.0, x).dv_total for x in ratios])
dv1 = np.array([hohmann(1.0, 1.0, x).dv1 for x in ratios])
dv2 = np.array([hohmann(1.0, 1.0, x).dv2 for x in ratios])
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.semilogx(ratios, tot, color=INK, lw=2.2, label=r"total $(\Delta v_1+\Delta v_2)/v_{c,1}$")
ax.semilogx(ratios, dv1, color=SERIES[0], lw=1.5, label=r"$\Delta v_1/v_{c,1}$")
ax.semilogx(ratios, dv2, color=SERIES[1], lw=1.5, label=r"$\Delta v_2/v_{c,1}$")
ax.axhline(math.sqrt(2) - 1, color=MUTED, lw=0.8, ls="--"); ax.text(1.1, math.sqrt(2) - 1 + 0.01, r"escape: $\sqrt{2}-1$", color=MUTED, fontsize=8.5)
k = int(np.argmax(tot)); ax.plot(ratios[k], tot[k], "o", color=INK, ms=6)
ax.annotate(f"max {tot[k]:.3f} at r₂/r₁ = {ratios[k]:.1f}", (ratios[k], tot[k]), xytext=(0, 12), textcoords="offset points", ha="center", fontsize=8.5)
for lab, x in (("LEO→GEO", 42164 / 6678), ("LEO→Moon", 384400 / 6563)):
    y = hohmann(1.0, 1.0, x).dv_total; ax.plot(x, y, "s", color=SERIES[2], ms=6); ax.annotate(lab, (x, y), xytext=(4, -14), textcoords="offset points", fontsize=8.5, color=INK2)
ax.set_xlabel(r"$r_2 / r_1$"); ax.set_ylabel(r"$\Delta v / v_{c,1}$"); ax.set_ylim(0, 0.7)
ax.legend(fontsize=8.5, loc="center right"); ax.set_title("Hohmann transfer cost between circular orbits", loc="left", fontsize=10.5)
fig.tight_layout(); save(fig, "tut_hohmann_norm.png")

# 4. Oberth: departure dv vs v_inf ------------------------------------------------
vinfs = np.linspace(0, 12, 200)
fig, ax = plt.subplots(figsize=(7, 4.2))
for k, alt in enumerate((300.0, 1000.0, 5000.0)):
    rp_ = R + alt
    ax.plot(vinfs, np.sqrt(vinfs ** 2 + 2 * mu / rp_) - circular_velocity(mu, rp_), color=SERIES[k], lw=2, label=f"single burn from {alt:.0f} km circular orbit")
rp_ = R + 300
ax.plot(vinfs, (escape_velocity(mu, rp_) - circular_velocity(mu, rp_)) + vinfs, color=MUTED, lw=1.5, ls="--",
        label="escape from 300 km, then accelerate far away")
ax.set_xlabel(r"hyperbolic excess speed $v_\infty$ [km/s]"); ax.set_ylabel(r"departure $\Delta v$ [km/s]")
ax.legend(fontsize=8.5); ax.set_title("The Oberth effect: burn deep in the well", loc="left", fontsize=10.5)
fig.tight_layout(); save(fig, "tut_oberth.png")

# 5. capture dv vs periapsis altitude at Mars -----------------------------------------
vinf_m = 2.649
alts = np.linspace(200, 40000, 400)
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(alts / 1e3, [capture_dv("Mars", vinf_m, h).dv for h in alts], color=SERIES[0], lw=2, label="into circular orbit at that altitude")
for k, apo in enumerate((20000.0, 40000.0, 100000.0)):
    ax.plot(alts[alts < apo] / 1e3, [capture_dv("Mars", vinf_m, h, apo).dv for h in alts[alts < apo]], color=SERIES[k + 1], lw=1.6,
            label=f"into ellipse with apoapsis {apo/1e3:.0f}×10³ km")
ropt = 2 * MARS.mu / vinf_m ** 2 - MARS.radius
ax.axvline(ropt / 1e3, color=MUTED, lw=0.8, ls=":"); ax.text(ropt / 1e3 + 0.5, 0.55, f"circular optimum\n$r_p=2\\mu/v_\\infty^2$\n({ropt/1e3:.1f}×10³ km)", fontsize=8.5, color=INK2)
ax.set_xlabel("periapsis altitude [10³ km]"); ax.set_ylabel(r"capture $\Delta v$ [km/s]"); ax.set_ylim(0, 2.3)
ax.legend(fontsize=8.5); ax.set_title(r"Capture at Mars, $v_\infty=2.65$ km/s", loc="left", fontsize=10.5)
fig.tight_layout(); save(fig, "tut_capture.png")

# 6. gravity assist: heliocentric dv vs v_inf ------------------------------------------
vinfs = np.linspace(0.5, 45, 300)
fig, ax = plt.subplots(figsize=(7, 4.2))
for k, (b, alt) in enumerate((("Venus", 300.0), ("Earth", 300.0), ("Mars", 300.0), ("Jupiter", 1000.0), ("Saturn", 1000.0))):
    bb = get_body(b); rp_ = bb.radius + alt
    dv = 2 * vinfs * np.sin(0.5 * np.array([turn_angle(bb.mu, rp_, v) for v in vinfs]))
    ax.plot(vinfs, dv, color=SERIES[k], lw=2, label=f"{b}, periapsis alt {alt:.0f} km")
    vstar = math.sqrt(bb.mu / rp_); ax.plot(vstar, 2 * vstar * math.sin(0.5 * turn_angle(bb.mu, rp_, vstar)), "o", color=SERIES[k], ms=5)
ax.set_xlabel(r"$v_\infty$ relative to the planet [km/s]"); ax.set_ylabel(r"max heliocentric $|\Delta v|$ [km/s]")
ax.legend(fontsize=8.5); ax.set_title(r"Free $\Delta v$ from an unpowered flyby: $2 v_\infty \sin(\delta/2)$; dots at $v_\infty=\sqrt{\mu/r_p}$", loc="left", fontsize=10)
fig.tight_layout(); save(fig, "tut_gravity_assist.png")

# 7. Lambert: time of flight vs the universal variable z -----------------------------------
r1n, r2n = AU_KM, 1.524 * AU_KM
fig, ax = plt.subplots(figsize=(7, 4.2))
for k, dth in enumerate((75.0, 120.0, 165.0, 240.0)):
    th = math.radians(dth)
    A = math.sin(th) * math.sqrt(r1n * r2n / (1 - math.cos(th)))
    zs = np.linspace(-40, 4 * math.pi ** 2 - 0.3, 800)
    tofs = []
    for z in zs:
        C, S = stumpff_C(z), stumpff_S(z)
        y = r1n + r2n + A * (z * S - 1) / math.sqrt(C)
        tofs.append(((y / C) ** 1.5 * S + A * math.sqrt(y)) / math.sqrt(MU_SUN) / 86400 if y > 0 else np.nan)
    ax.plot(zs, tofs, color=SERIES[k], lw=2, label=f"transfer angle {dth:.0f}°")
ax.axvline(0, color=MUTED, lw=0.8, ls=":"); ax.text(0.5, 20, "parabolic\n(z = 0)", fontsize=8.5, color=INK2)
ax.text(-38, 60, "hyperbolic  ←", fontsize=8.5, color=INK2); ax.text(8, 60, "→  elliptic", fontsize=8.5, color=INK2)
ax.set_xlabel(r"universal variable $z = \alpha\chi^2$  ($\alpha = 1/a$)"); ax.set_ylabel("time of flight [days]"); ax.set_ylim(0, 700)
ax.legend(fontsize=8.5, loc="upper left"); ax.set_title("Lambert's problem: time of flight is monotonic in z (Earth→Mars radii)", loc="left", fontsize=10)
fig.tight_layout(); save(fig, "tut_lambert_tof.png")

# 8. copies of shared figures ---------------------------------------------------------------
for name in ("porkchop_earth_mars_2026_dvtotal.png", "transfer_earth_mars_2026.png", "lunar_patched_conic_apollo11.png",
             "aerobraking_pass_comparison.png", "hw1_budget_aero.png", "hw1_budget_prop.png", "transfer_voyager2.png"):
    shutil.copy(os.path.join(SHARED, name), os.path.join(FIGS, name))
print("figures written to", FIGS)
