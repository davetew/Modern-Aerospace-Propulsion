"""Compute every number quoted in the orbital-mechanics tutorial with the
sfmission package and write them as LaTeX macros (tutorial_numbers.tex).

Run (with sfmission installed):
    python "Week 10 - Mission Analysis/docs/tutorial/calc/make_tutorial_numbers.py"
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

from sfmission import *
from sfmission.kepler import (
    circular_velocity, escape_velocity, period, vis_viva, apsides_from_rp_ra, E_from_true_anomaly,
    time_since_periapsis, turn_angle, specific_energy, propagate)
from sfmission.ephemeris import (
    heliocentric_longitude, planet_state, moon_state_geocentric, body_state)
from sfmission.lunar import D_MOON, R_SOI_MOON, V_MOON
from sfmission.transfers import hohmann_phase_angle, synodic_period
from sfmission.patched_conics import optimal_capture_periapsis
from sfmission.staging import exhaust_velocity

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tutorial_numbers.tex")
macros: dict[str, str] = {}


_DIGIT_WORDS = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five",
                "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}


def put(name: str, value, fmt: str = "{:.3f}"):
    """Store a macro; LaTeX control sequences cannot contain digits, so spell them out."""
    name = "".join(_DIGIT_WORDS.get(ch, ch) for ch in name)
    macros[name] = fmt.format(value) if not isinstance(value, str) else value


def sig(x, n=4):
    """n significant figures as a plain decimal string."""
    if x == 0:
        return "0"
    d = n - int(math.floor(math.log10(abs(x)))) - 1
    return f"{round(x, d):.{max(d,0)}f}"


# ---------------------------------------------------------------- Ch.1 two-body
mu, R = EARTH.mu, EARTH.radius
put("muEarth", mu, "{:.1f}")
put("REarth", R, "{:.1f}")
put("muSun", MU_SUN, "{:.4e}")
put("muMoon", MOON.mu, "{:.1f}")
put("muMars", MARS.mu, "{:.1f}")
put("muJupiter", JUPITER.mu, "{:.4e}")
r_iss = R + 420.0
put("ISSvc", circular_velocity(mu, r_iss))
put("ISSperiodmin", period(mu, r_iss) / 60, "{:.1f}")
put("ISSvesc", escape_velocity(mu, r_iss))
put("ISSenergy", specific_energy(mu, r_iss, circular_velocity(mu, r_iss)), "{:.2f}")
a_geo = (mu * (86164.0905 / (2 * math.pi)) ** 2) ** (1 / 3)
put("GEOa", a_geo, "{:.0f}")
put("GEOalt", a_geo - R, "{:.0f}")
put("GEOvc", circular_velocity(mu, a_geo))
# GTO 300 x 35786 km
rp, ra = R + 300, R + 35786
a_t, e_t = apsides_from_rp_ra(rp, ra)
put("GTOa", a_t, "{:.0f}")
put("GTOe", e_t, "{:.4f}")
put("GTOvp", vis_viva(mu, rp, a_t))
put("GTOva", vis_viva(mu, ra, a_t))
put("GTOperiodh", period(mu, a_t) / 3600, "{:.2f}")
put("GTOenergy", -mu / (2 * a_t), "{:.3f}")
put("GTOh", math.sqrt(mu * a_t * (1 - e_t ** 2)), "{:.0f}")
put("GTOp", a_t * (1 - e_t ** 2), "{:.0f}")
# time from perigee to nu = 90 deg on the GTO
t90 = time_since_periapsis(mu, a_t, e_t, math.radians(90))
E90 = E_from_true_anomaly(math.radians(90), e_t)
put("GTOEninety", math.degrees(E90), "{:.2f}")
put("GTOMninety", math.degrees(E90 - e_t * math.sin(E90)), "{:.2f}")
put("GTOtninetymin", t90 / 60, "{:.1f}")
put("GTOhalfperiodmin", period(mu, a_t) / 120, "{:.1f}")

# ---------------------------------------------------------------- Ch.2 maneuvers
h = hohmann(mu, R + 300, a_geo)
put("HohLEOGEOdvone", h.dv1); put("HohLEOGEOdvtwo", h.dv2); put("HohLEOGEOtot", h.dv_total)
put("HohLEOGEOtofh", h.tof / 3600, "{:.2f}")
put("HohLEOGEOvtp", h.vt1); put("HohLEOGEOvta", h.vt2); put("HohLEOGEOvcone", h.v1); put("HohLEOGEOvctwo", h.v2)
pc = hohmann_with_plane_change(mu, R + 300, a_geo, math.radians(28.5))
put("HohPCdvone", pc.dv1); put("HohPCdvtwo", pc.dv2); put("HohPCtot", pc.dv_total)
put("HohPCdione", math.degrees(pc.di1), "{:.2f}"); put("HohPCditwo", math.degrees(pc.di2), "{:.2f}")
put("PCatLEO", plane_change_dv(h.v1, math.radians(28.5)))
put("PCatGEO", plane_change_dv(h.v2, math.radians(28.5)))
# normalized Hohmann curve peak
ratios = np.linspace(2, 100, 200001)
tot = np.array([hohmann(1.0, 1.0, r).dv_total for r in ratios[::50]])
rr = ratios[::50]
k = int(np.argmax(tot))
put("HohPeakRatio", rr[k], "{:.2f}"); put("HohPeakValue", tot[k], "{:.4f}")
put("HohEscapeLimit", math.sqrt(2) - 1, "{:.4f}")
be = bielliptic(mu, 7000.0, 15 * 7000.0, 60 * 7000.0)
hh = hohmann(mu, 7000.0, 15 * 7000.0)
put("BiEllTot", be.dv_total); put("BiEllHoh", hh.dv_total); put("BiEllTOFd", be.tof / 86400, "{:.1f}"); put("BiEllHohTOFh", hh.tof / 3600, "{:.1f}")
put("SynEM", synodic_period(period(MU_SUN, EARTH.a_mean), period(MU_SUN, MARS.a_mean)) / 86400, "{:.1f}")
put("PhaseEM", math.degrees(hohmann_phase_angle(MU_SUN, EARTH.a_mean, MARS.a_mean)), "{:.1f}")

# ---------------------------------------------------------------- Ch.3 patched conics
for b in (EARTH, MOON, MARS, JUPITER, VENUS, SATURN):
    put(f"SOI{b.name}", b.soi / 1e3, "{:.0f}")     # 10^3 km
put("SOIEarthAU", EARTH.soi / AU_KM, "{:.4f}")
hm = hohmann_interplanetary("Earth", "Mars", 300, 400)
put("EMvtp", hohmann(MU_SUN, EARTH.a_mean, MARS.a_mean).vt1); put("EMvta", hohmann(MU_SUN, EARTH.a_mean, MARS.a_mean).vt2)
put("EMvEarth", hohmann(MU_SUN, EARTH.a_mean, MARS.a_mean).v1); put("EMvMars", hohmann(MU_SUN, EARTH.a_mean, MARS.a_mean).v2)
put("EMvinfdep", hm.vinf_dep); put("EMvinfarr", hm.vinf_arr); put("EMCthree", hm.vinf_dep ** 2, "{:.2f}")
put("EMtofd", hm.tof / 86400, "{:.1f}"); put("EMdvdep", hm.departure.dv); put("EMvhyp", hm.departure.v_hyp)
put("EMvpark", hm.departure.v_park); put("EMehyp", hm.departure.e_hyp, "{:.3f}")
put("EMturnhalf", math.degrees(math.acos(1 / hm.departure.e_hyp)), "{:.1f}")
cap_c = capture_dv("Mars", hm.vinf_arr, 400); cap_e = capture_dv("Mars", hm.vinf_arr, 400, 40000)
put("EMcapcirc", cap_c.dv); put("EMcapell", cap_e.dv); put("EMcapvhyp", cap_c.v_hyp); put("EMcapvc", cap_c.v_orbit)
put("EMcapvpell", cap_e.v_orbit)
put("EMcapOptRp", optimal_capture_periapsis("Mars", hm.vinf_arr, 0.0), "{:.0f}")
put("EMcapOptAlt", optimal_capture_periapsis("Mars", hm.vinf_arr, 0.0) - MARS.radius, "{:.0f}")
# Oberth: dv to reach v_inf = 3 km/s from 300 km vs from far away
vinf = 3.0
d = departure_dv("Earth", vinf, 300)
put("Obvinf", vinf, "{:.1f}"); put("Obdv", d.dv); put("Obesc", escape_velocity(mu, R + 300) - circular_velocity(mu, R + 300))
put("ObvescLEO", escape_velocity(mu, R + 300))
# planets table
rows = []
for t in ("Venus", "Mars", "Jupiter", "Saturn"):
    r = hohmann_interplanetary("Earth", t, 300, None)
    rows.append((t, r.vinf_dep, r.vinf_dep ** 2, r.departure.dv, r.vinf_arr, r.tof / 86400, math.degrees(r.phase_angle),
                 r.synodic_period / 86400 / 365.25))
    put(f"Pl{t}vinfdep", r.vinf_dep, "{:.2f}"); put(f"Pl{t}Cthree", r.vinf_dep ** 2, "{:.1f}"); put(f"Pl{t}dvdep", r.departure.dv, "{:.2f}")
    put(f"Pl{t}vinfarr", r.vinf_arr, "{:.2f}"); put(f"Pl{t}tofd", r.tof / 86400, "{:.0f}"); put(f"Pl{t}phase", math.degrees(r.phase_angle), "{:.1f}")
    put(f"Pl{t}syn", r.synodic_period / 86400 / 365.25, "{:.2f}")
# gravity assist
for b, alt in (("Jupiter", 1000.0), ("Venus", 300.0), ("Earth", 300.0)):
    put(f"GA{b}turn", math.degrees(turn_angle(get_body(b).mu, get_body(b).radius + alt, 8.0)), "{:.1f}")
    put(f"GA{b}dv", flyby_max_heliocentric_dv(b, 8.0, alt), "{:.2f}")
v2 = interplanetary_mission(["Earth", "Jupiter", "Saturn"], ["1977-08-20", "1979-07-09", "1981-08-25"], 185, None, flyby_min_alt=100000)
f = v2.flybys[0]
put("VgCthree", v2.departure.c3, "{:.1f}"); put("Vgvinfin", np.linalg.norm(f.vinf_in)); put("Vgvinfout", np.linalg.norm(f.vinf_out))
put("Vgturn", math.degrees(f.turn_angle), "{:.1f}"); put("Vgrp", f.rp / 1e3, "{:.0f}"); put("Vghelio", f.delta_v_helio, "{:.2f}")
put("Vgvin", np.linalg.norm(f.v_helio_in), "{:.2f}"); put("Vgvout", np.linalg.norm(f.v_helio_out), "{:.2f}"); put("Vgdvpow", f.dv_powered)

# ---------------------------------------------------------------- Ch.4 ephemerides
d0 = "2026-11-12"
lE, lM = heliocentric_longitude("earth", d0), heliocentric_longitude("mars", d0)
put("EphLonEarth", lE, "{:.1f}"); put("EphLonMars", lM, "{:.1f}"); put("EphPhase", (lM - lE) % 360, "{:.1f}")
rE, vE = planet_state("earth", d0); rM, vM = planet_state("mars", d0)
put("EphrEarth", np.linalg.norm(rE) / AU_KM, "{:.4f}"); put("EphrMars", np.linalg.norm(rM) / AU_KM, "{:.4f}")
put("EphvEarth", np.linalg.norm(vE), "{:.3f}"); put("EphvMars", np.linalg.norm(vM), "{:.3f}")
put("EphdistEM", np.linalg.norm(rM - rE) / AU_KM, "{:.3f}")
rm, vm = moon_state_geocentric(d0)
put("EphMoonDist", np.linalg.norm(rm), "{:.0f}"); put("EphMoonSpeed", np.linalg.norm(vm))
from sfmission.ephemeris import planet_elements, centuries_since_j2000
el = planet_elements("mars", d0)
put("EphT", centuries_since_j2000(d0), "{:.6f}")
put("EphMarsa", el["a"] / AU_KM, "{:.6f}"); put("EphMarse", el["e"], "{:.6f}"); put("EphMarsM", math.degrees(el["M"]), "{:.2f}")
put("EphMarsnu", math.degrees(el["nu"]), "{:.2f}")

# ---------------------------------------------------------------- Ch.5 Lambert / porkchop
leg = heliocentric_leg("Earth", "Mars", "2026-10-30", "2027-09-06")
put("LamCthree", leg.c3, "{:.2f}"); put("Lamvinfdep", leg.vinf_dep_mag); put("Lamvinfarr", leg.vinf_arr_mag)
put("Lamtofd", leg.tof_days, "{:.0f}"); put("Lamdtheta", math.degrees(leg.lambert.dtheta), "{:.1f}")
put("Lama", leg.lambert.a / AU_KM, "{:.4f}")
put("Lamdvdep", departure_dv("Earth", leg.vinf_dep_mag, 300).dv); put("Lamdvcap", capture_dv("Mars", leg.vinf_arr_mag, 400).dv)
put("Lamr1AU", np.linalg.norm(leg.r1) / AU_KM, "{:.4f}"); put("Lamr2AU", np.linalg.norm(leg.r2) / AU_KM, "{:.4f}")
put("Lamv1", np.linalg.norm(leg.lambert.v1)); put("LamvE", np.linalg.norm(leg.v_origin))
pcm = porkchop("Earth", "Mars", "2026-09-01", "2027-02-15", 120, 450, n_dep=110, n_tof=110, park_alt=300, capture_peri_alt=400)
o = pcm.optimum("dv_total"); c = pcm.optimum("c3"); w = pcm.window("dv_total")
put("PkOptDep", o["depart"][:10], "{}"); put("PkOptTOF", o["tof_days"], "{:.0f}"); put("PkOptCthree", o["c3"], "{:.2f}")
put("PkOptvinfarr", o["vinf_arr"]); put("PkOptdv", o["dv_total"]); put("PkOptAngle", o["transfer_angle_deg"], "{:.0f}")
put("PkMinCthree", c["c3"], "{:.2f}"); put("PkMinCthreeDep", c["depart"][:10], "{}"); put("PkMinCthreeTOF", c["tof_days"], "{:.0f}")
put("PkWinOpen", w["open"][:10], "{}"); put("PkWinClose", w["close"][:10], "{}"); put("PkWinDays", w["days"], "{:.0f}")
pc1 = porkchop("Earth", "Mars", "2026-10-01", "2027-01-15", 150, 260, n_dep=60, n_tof=60, park_alt=300, capture_peri_alt=400)
o1 = pc1.optimum("dv_total")
put("PkTypeIDep", o1["depart"][:10], "{}"); put("PkTypeITOF", o1["tof_days"], "{:.0f}"); put("PkTypeICthree", o1["c3"], "{:.2f}")
put("PkTypeIvinf", o1["vinf_arr"]); put("PkTypeIdv", o1["dv_total"])
m20 = heliocentric_leg("Earth", "Mars", "2020-07-30T11:50", "2021-02-18T20:37")
put("MtwCthree", m20.c3, "{:.2f}"); put("Mtwvinfarr", m20.vinf_arr_mag); put("Mtwtof", m20.tof_days, "{:.0f}")
put("Mtwdtheta", math.degrees(m20.lambert.dtheta), "{:.1f}")
L, prev = 0.0, m20.r1
for t in np.linspace(0, m20.tof_days * 86400, 6000)[1:]:
    rr_, _ = propagate(m20.r1, m20.lambert.v1, float(t), MU_SUN); L += np.linalg.norm(rr_ - prev); prev = rr_
put("MtwPath", L / 1e6, "{:.1f}")
put("MtwEntry", math.sqrt(m20.vinf_arr_mag ** 2 + 2 * MARS.mu / (MARS.radius + 125)), "{:.2f}")

# ---------------------------------------------------------------- Ch.6 Moon
put("MoonD", D_MOON, "{:.0f}"); put("MoonSOI", R_SOI_MOON, "{:.0f}"); put("MoonV", V_MOON, "{:.4f}")
put("MoonSOIfrac", R_SOI_MOON / D_MOON * 100, "{:.0f}")
est = lunar_hohmann_estimate(185, 100)
put("LunTLIv", est.v_tli, "{:.4f}"); put("LunTLIdv", est.dv_tli); put("Lunvpark", est.v_park); put("LunTOFd", est.tof / 86400, "{:.2f}")
put("Lunvapo", est.v_apogee, "{:.4f}"); put("Lunvinf", est.vinf_moon, "{:.4f}"); put("LunLOI", est.loi.dv); put("LunLOIvhyp", est.loi.v_hyp)
put("LunLOIvc", est.loi.v_orbit); put("Luntot", est.dv_total)
put("Lunat", 0.5 * (EARTH.radius + 185 + D_MOON), "{:.0f}")
sols = design_lunar_transfer(185, 100, v0=10.95, direction="either")
sr = [s for s in sols if s.h2 < 0][0]; sp = [s for s in sols if s.h2 > 0][0]
for tag, s in (("R", sr), ("P", sp)):
    put(f"Lun{tag}lam", math.degrees(s.lambda1), "{:.1f}"); put(f"Lun{tag}tofgeo", s.tof / 3600, "{:.1f}"); put(f"Lun{tag}tofsoi", s.tof_soi / 3600, "{:.1f}")
    put(f"Lun{tag}toftot", s.tof_total / 3600, "{:.1f}"); put(f"Lun{tag}vinf", s.vinf, "{:.3f}"); put(f"Lun{tag}LOI", s.loi_dv()); put(f"Lun{tag}TLI", s.dv_tli)
    put(f"Lun{tag}rone", s.r1 / 1e3, "{:.0f}"); put(f"Lun{tag}vone", s.v1, "{:.3f}"); put(f"Lun{tag}phione", math.degrees(s.phi1), "{:.1f}")
    put(f"Lun{tag}gammaone", math.degrees(s.gamma1), "{:.2f}"); put(f"Lun{tag}lead", math.degrees(s.gamma0), "{:.1f}"); put(f"Lun{tag}etwo", s.e2, "{:.3f}")
    put(f"Lun{tag}a", s.a_geo / 1e3, "{:.0f}"); put(f"Lun{tag}e", s.e_geo, "{:.4f}")
ap = design_lunar_transfer(336.3, 110, v0=10.832, phi0_deg=7.2, direction="retrograde")[0]
put("ApTOF", ap.tof_total / 3600, "{:.1f}"); put("ApLOI", ap.loi_dv()); put("Aplam", math.degrees(ap.lambda1), "{:.1f}")
ap2 = design_lunar_transfer(322.7, 110, v0=10.840, phi0_deg=7.2, direction="retrograde")[0]
put("ApTOFtwo", ap2.tof_total / 3600, "{:.1f}"); put("ApLOItwo", ap2.loi_dv())

# ---------------------------------------------------------------- Ch.7 launch
L0 = launch_to_orbit(300, None, 28.5)
put("LchVorb", L0.v_orbit); put("LchVsite", L0.v_site); put("LchIdeal", L0.dv_ideal); put("LchLow", L0.dv_low, "{:.2f}"); put("LchHigh", L0.dv_high, "{:.2f}")
put("LchOmega", EARTH.omega, "{:.4e}"); put("LchVeq", EARTH.equatorial_rotation_speed())
L1 = launch_to_orbit(300, 51.6, 28.5); put("LchISSaz", math.degrees(L1.azimuth), "{:.1f}"); put("LchISSsite", L1.v_site - (L1.v_orbit - L1.dv_ideal) + L1.v_site, "{:.3f}")
put("LchISScredit", L1.dv_rotation_credit); put("LchISSideal", L1.dv_ideal)
L2 = launch_to_orbit(300, 90.0, 28.5); put("LchPolarIdeal", L2.dv_ideal); put("LchPolarCredit", L2.dv_rotation_credit)
L3 = launch_to_orbit(300, None, 5.2); put("LchKourouCredit", L3.v_site)

# ---------------------------------------------------------------- Ch.8 aero
ac = aerocapture_credit("Mars", hm.vinf_arr, 400)
put("AcEntry", ac.entry_speed); put("AcSaved", ac.dv_saved); put("AcRaise", ac.dv_periapsis_raise, "{:.4f}"); put("AcNet", ac.dv_net_saving)
camp = aerobraking_campaign("Mars", hm.vinf_arr, 110, 40000, 400, beta=50)
put("AbPasses", camp.n_passes, "{:d}"); put("AbDays", camp.duration / 86400, "{:.0f}"); put("AbDrag", camp.dv_drag_total); put("AbCapHigh", camp.dv_capture_high)
put("AbRaise", camp.dv_periapsis_raise); put("AbDirect", camp.dv_direct_capture); put("AbSaved", camp.dv_saved); put("AbFirst", camp.pass_dv[0] * 1e3, "{:.2f}")
num = aerobraking_pass_dv_numeric("Mars", 110, 40000, 50)
rp_ = MARS.radius + 110; Vp = vis_viva(MARS.mu, rp_, 0.5 * (rp_ + MARS.radius + 40000))
put("AbRho", num["rho_p"], "{:.2e}"); put("AbVp", Vp); put("AbNum", num["dv_pass"] * 1e3); put("AbAna", aerobraking_pass_dv(num["rho_p"], Vp, MARS.atmosphere.H, rp_, 50, MARS.mu) * 1e3)
put("AbStraight", aerobraking_pass_dv(num["rho_p"], Vp, MARS.atmosphere.H, rp_, 50) * 1e3)
put("AbFactor", math.sqrt(Vp ** 2 / (Vp ** 2 - MARS.mu / rp_))); put("AbH", MARS.atmosphere.H, "{:.1f}")
put("AbRddot", (Vp ** 2 / rp_ - MARS.mu / rp_ ** 2) * 1e3, "{:.4f}")   # m/s^2

# ---------------------------------------------------------------- Ch.9 rocket equation / staging / HW1 chain
put("gzero", G0 * 1e3, "{:.5f}")
for isp in (320, 450, 900, 3000):
    put(f"Zeta{isp}", propellant_fraction_for_dv(isp, 5.7), "{:.3f}")
put("IspSSTO", required_isp(9.3, 0.90), "{:.0f}")
b_prop, m_prop = interplanetary_budget("Earth", "Mars", o["depart"], o["arrive"], 300, 28.5, 400, None, False, True, 0.05)
b_aero, m_aero = interplanetary_budget("Earth", "Mars", o["depart"], o["arrive"], 300, 28.5, 400, None, True, True, 0.05)
put("HWdepart", o["depart"][:10], "{}"); put("HWarrive", o["arrive"][:10], "{}")
put("HWlaunchMid", b_prop.legs[0].dv); put("HWinj", b_prop.legs[1].dv); put("HWcap", b_prop.legs[2].dv); put("HWmargin", b_prop.legs[3].dv)
put("HWtotProp", b_prop.total_propulsive()); put("HWtotAero", b_aero.total_propulsive())
put("HWpostProp", b_prop.post_launch_propulsive()); put("HWpostAero", b_aero.post_launch_propulsive())
for z in (0.5, 0.6, 0.7):
    put(f"HWIspProp{int(z*10)}", b_prop.required_isp(z), "{:.0f}"); put(f"HWIspAero{int(z*10)}", b_aero.required_isp(z), "{:.0f}")
rows_p = b_prop.size_propellant(1000.0, 320.0)
put("HWminj", rows_p[0]["m_before"], "{:.0f}"); put("HWmpropInj", rows_p[0]["m_prop"], "{:.0f}"); put("HWmpropCap", rows_p[1]["m_prop"], "{:.0f}")
put("HWmpropMargin", rows_p[2]["m_prop"], "{:.0f}")
dv_l = b_prop.legs[0].dv + b_prop.legs[1].dv
veh = design_vehicle(dv_l, rows_p[0]["m_before"], [300, 450], [0.07, 0.10], ["booster", "upper"])
put("HWdvLauncher", dv_l); put("HWm0", veh.m0 / 1e3, "{:.0f}"); put("HWpf", veh.payload_fraction * 100, "{:.2f}")
put("HWdvB", veh.stages[0].dv); put("HWdvU", veh.stages[1].dv); put("HWmB", veh.stages[0].m_stage / 1e3, "{:.0f}"); put("HWmU", veh.stages[1].m_stage / 1e3, "{:.1f}")
veh_eq = design_vehicle(dv_l, rows_p[0]["m_before"], [300, 450], [0.07, 0.10], split=[dv_l / 2, dv_l / 2])
put("HWm0eq", veh_eq.m0 / 1e3, "{:.0f}")
put("HWcLow", exhaust_velocity(300)); put("HWcUp", exhaust_velocity(450))


# ---------------------------------------------------------------- exercise answers
r5 = MARS.radius + 500
put("ExMarsVc", circular_velocity(MARS.mu, r5)); put("ExMarsTmin", period(MARS.mu, r5) / 60, "{:.1f}"); put("ExMarsVesc", escape_velocity(MARS.mu, r5))
hl = hohmann(mu, R + 400, D_MOON)
put("ExLunHohdvone", hl.dv1); put("ExLunHohdvtwo", hl.dv2); put("ExLunHohtot", hl.dv_total); put("ExLunHohtofd", hl.tof / 86400, "{:.2f}")
put("ExLunHohvta", hl.vt2); put("ExLunHohvc2", hl.v2)
hj = hohmann_interplanetary("Earth", "Jupiter", 300, None)
cj_c = capture_dv("Jupiter", hj.vinf_arr, 1000); cj_e = capture_dv("Jupiter", hj.vinf_arr, 1000, 2.0e6)
put("ExJupvinf", hj.vinf_arr); put("ExJupCapCirc", cj_c.dv); put("ExJupCapEll", cj_e.dv); put("ExJupVhyp", cj_c.v_hyp); put("ExJupVc", cj_c.v_orbit)
put("ExJupOptAlt", (optimal_capture_periapsis("Jupiter", hj.vinf_arr, 0.0) - JUPITER.radius) / 1e3, "{:.0f}")
dV = turn_angle(VENUS.mu, VENUS.radius + 300, 5.0)
put("ExVenTurn", math.degrees(dV), "{:.1f}"); put("ExVenDv", 2 * 5.0 * math.sin(dV / 2)); put("ExVenE", 1 + (VENUS.radius + 300) * 25 / VENUS.mu, "{:.3f}")
Lb = launch_to_orbit(300, None, 45.9); Lb2 = launch_to_orbit(300, 51.6, 45.9)
put("ExBaikSite", Lb.v_site); put("ExBaikIdeal", Lb.dv_ideal); put("ExBaikISSideal", Lb2.dv_ideal); put("ExBaikISSaz", math.degrees(Lb2.azimuth), "{:.1f}")
c100 = aerobraking_campaign("Mars", hm.vinf_arr, 110, 40000, 400, beta=100)
put("ExAbPasses", c100.n_passes, "{:d}"); put("ExAbDays", c100.duration / 86400, "{:.0f}")
put("ExIspReq", required_isp(4.31, 0.65), "{:.0f}"); put("ExZeta", propellant_fraction_for_dv(320, 4.31), "{:.3f}")
s11 = design_lunar_transfer(185, 100, v0=11.0, direction="retrograde")[0]
put("ExLunTOF", s11.tof_total / 3600, "{:.1f}"); put("ExLunLOI", s11.loi_dv()); put("ExLunTLI", s11.dv_tli); put("ExLunLam", math.degrees(s11.lambda1), "{:.1f}")
legI = heliocentric_leg("Earth", "Mars", "2026-11-16", "2027-08-03")
put("ExLegCthree", legI.c3, "{:.2f}"); put("ExLegvinf", legI.vinf_arr_mag); put("ExLegAngle", math.degrees(legI.lambert.dtheta), "{:.0f}"); put("ExLegTOF", legI.tof_days, "{:.0f}")
# generic: hyperbolic excess from LEO escape burn of 3.5 km/s
dv_b = 3.5; vp_b = circular_velocity(mu, R + 300) + dv_b
put("ExVinfFromBurn", math.sqrt(vp_b ** 2 - 2 * mu / (R + 300))); put("ExCthreeFromBurn", vp_b ** 2 - 2 * mu / (R + 300), "{:.2f}")

# ---------------------------------------------------------------- write
with open(OUT, "w") as fh:
    fh.write("% AUTO-GENERATED by calc/make_tutorial_numbers.py -- do not edit by hand\n")
    for k, v in macros.items():
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(f"wrote {len(macros)} macros to {OUT}")
