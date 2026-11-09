"""Validation against textbook examples and flown missions."""
import math
import numpy as np
import pytest

from modern_aerospace_propulsion.mission_analysis import *                                   # noqa: F401,F403
from modern_aerospace_propulsion.mission_analysis.bodies import EARTH, MARS, MOON, MU_SUN
from modern_aerospace_propulsion.mission_analysis.kepler import propagate
from modern_aerospace_propulsion.mission_analysis.lunar import lunar_patched_conic, D_MOON, R_SOI_MOON, V_MOON


# ---------------------------------------------------------------------------
# Transfers
# ---------------------------------------------------------------------------
def test_hohmann_leo_to_geo():
    # Curtis Example 6.1-style: 300 km -> GEO (42164 km): total 3.893 km/s, TOF 5.27 h
    h = hohmann(EARTH.mu, EARTH.radius + 300, 42164.0)
    assert h.dv_total == pytest.approx(3.893, abs=0.005)
    assert h.tof / 3600 == pytest.approx(5.27, abs=0.02)


def test_bielliptic_vs_hohmann_crossover():
    # bi-elliptic beats Hohmann for r2/r1 > ~11.94 (with rb -> inf); check ratio 15 with rb = 60 r1
    r1 = 7000.0
    assert bielliptic(EARTH.mu, r1, 15 * r1, 60 * r1).dv_total < hohmann(EARTH.mu, r1, 15 * r1).dv_total
    assert bielliptic(EARTH.mu, r1, 5 * r1, 60 * r1).dv_total > hohmann(EARTH.mu, r1, 5 * r1).dv_total


def test_plane_change_60deg_equals_speed():
    assert plane_change_dv(7.5, math.radians(60)) == pytest.approx(7.5)


def test_hohmann_plane_change_prefers_outer_burn():
    res = hohmann_with_plane_change(EARTH.mu, EARTH.radius + 300, 42164.0, math.radians(28.5))
    assert res.di2 > res.di1
    assert res.dv_total == pytest.approx(4.26, abs=0.03)   # classic 28.5 deg GTO -> GEO combined result


# ---------------------------------------------------------------------------
# Interplanetary (Curtis Ch. 8 examples, circular coplanar)
# ---------------------------------------------------------------------------
def test_earth_mars_hohmann_textbook():
    h = hohmann_interplanetary("Earth", "Mars", park_alt=300, capture_peri_alt=300)
    assert h.vinf_dep == pytest.approx(2.943, abs=0.01)        # Curtis Ex 8.4: 2.943 km/s
    assert h.vinf_arr == pytest.approx(2.648, abs=0.01)
    assert h.departure.dv == pytest.approx(3.590, abs=0.01)    # Curtis Ex 8.5: 3.590 km/s from 300 km
    assert h.tof / 86400 == pytest.approx(258.8, abs=1.0)
    assert math.degrees(h.phase_angle) == pytest.approx(44.3, abs=0.3)
    assert h.synodic_period / 86400 == pytest.approx(779.9, abs=1.0)   # 2.135 yr


def test_capture_optimum_periapsis_formula():
    rp = optimal_capture_periapsis("Mars", 2.648, 0.0)
    # Curtis Ex 8.6: circular capture optimum r_p = 2 mu / v_inf^2
    assert rp == pytest.approx(2 * MARS.mu / 2.648 ** 2)


# ---------------------------------------------------------------------------
# Flown missions (ephemeris + Lambert)
# ---------------------------------------------------------------------------
def test_mars_2020_reconstruction():
    """Perseverance: launch 2020-07-30 11:50 UTC, entry 2021-02-18 20:37 UTC.
    JPL press kit: 203-day cruise, ~471 million km traveled."""
    leg = heliocentric_leg("Earth", "Mars", "2020-07-30T11:50", "2021-02-18T20:37")
    assert leg.tof_days == pytest.approx(203.4, abs=0.1)
    assert 13.0 < leg.c3 < 16.0                      # Atlas V 541 class C3 for this window
    assert 2.3 < leg.vinf_arr_mag < 2.9
    # path length
    n = 4000
    L, prev = 0.0, leg.r1
    for t in np.linspace(0, leg.tof_days * 86400, n)[1:]:
        r, _ = propagate(leg.r1, leg.lambert.v1, float(t), MU_SUN)
        L += np.linalg.norm(r - prev)
        prev = r
    assert L / 1e6 == pytest.approx(471.0, rel=0.01)
    # entry speed at 125 km vs press-kit 19,500 km/h (5.42 km/s, atmosphere-relative)
    v_entry = math.sqrt(leg.vinf_arr_mag ** 2 + 2 * MARS.mu / (MARS.radius + 125))
    assert 5.3 < v_entry < 5.7


def test_mars_2020_porkchop_optimum_near_actual_launch():
    pc = porkchop("Earth", "Mars", "2020-06-15", "2020-09-15", 150, 350, n_dep=30, n_tof=30, park_alt=185)
    o = pc.optimum("c3")
    assert o["depart"][:7] == "2020-07"
    assert o["c3"] < 15.0


def test_voyager2_jupiter_flyby_geometry():
    """Voyager 2: launch 1977-08-20, Jupiter 1979-07-09, Saturn 1981-08-25.
    Patched conic should need only a small powered correction at Jupiter."""
    m = interplanetary_mission(["Earth", "Jupiter", "Saturn"], ["1977-08-20", "1979-07-09", "1981-08-25"],
                               park_alt=185, capture_peri_alt=None, flyby_min_alt=100000)
    fb = m.flybys[0]
    assert fb.feasible
    assert fb.dv_powered < 0.2
    assert 500_000 < fb.alt < 800_000          # Voyager 2 closest approach ~570,000 km above the cloud tops
    assert 90 < math.degrees(fb.turn_angle) < 110
    assert 95 < m.departure.c3 < 115            # Titan IIIE-Centaur C3 ~102-106 km2/s2


def test_powered_flyby_zero_when_matched():
    vinf_in = np.array([3.0, 4.0, 0.0])
    fb = flyby_unpowered("Jupiter", vinf_in, np.array([0.0, 13.0, 0.0]), 1_000_000.0)
    pf = flyby_powered("Jupiter", vinf_in, fb.vinf_out, np.array([0.0, 13.0, 0.0]), min_alt=1000)
    assert pf.dv_powered == pytest.approx(0.0, abs=1e-6)
    assert pf.rp == pytest.approx(1_000_000.0, rel=1e-5)


# ---------------------------------------------------------------------------
# Lunar
# ---------------------------------------------------------------------------
def test_lunar_hohmann_estimate_apollo_class():
    est = lunar_hohmann_estimate(185, 100)
    assert est.dv_tli == pytest.approx(3.135, abs=0.01)
    assert 0.78 < est.loi.dv < 0.90
    assert est.tof / 86400 == pytest.approx(4.98, abs=0.05)


def test_patched_conic_matches_bmw_trig_formulas():
    """Vector implementation vs the textbook scalar relations (BMW 7.4)."""
    r0, v0, phi0, lam = 6700.0, 10.90, math.radians(2.0), math.radians(45.0)
    s = lunar_patched_conic(r0, v0, phi0, lam)
    mu = EARTH.mu
    energy = 0.5 * v0 ** 2 - mu / r0
    h = r0 * v0 * math.cos(phi0)
    r1 = math.sqrt(D_MOON ** 2 + R_SOI_MOON ** 2 - 2 * D_MOON * R_SOI_MOON * math.cos(lam))
    v1 = math.sqrt(2 * (energy + mu / r1))
    phi1 = math.acos(h / (r1 * v1))
    gamma1 = math.asin(R_SOI_MOON / r1 * math.sin(lam))
    v2 = math.sqrt(v1 ** 2 + V_MOON ** 2 - 2 * v1 * V_MOON * math.cos(phi1 - gamma1))
    eps2 = math.asin(V_MOON / v2 * math.cos(lam) - v1 / v2 * math.cos(lam + gamma1 - phi1))
    h2 = R_SOI_MOON * v2 * math.sin(eps2)
    assert s.r1 == pytest.approx(r1)
    assert s.v1 == pytest.approx(v1)
    assert s.phi1 == pytest.approx(phi1)
    assert s.gamma1 == pytest.approx(gamma1)
    assert s.vinf == pytest.approx(v2)
    assert abs(s.h2) == pytest.approx(abs(h2), rel=1e-9)


def test_design_lunar_transfer_hits_target_perilune():
    sols = design_lunar_transfer(185, 110, v0=10.95, direction="either")
    assert len(sols) >= 2            # one prograde, one retrograde solution
    for s in sols:
        assert s.perilune_alt == pytest.approx(110.0, abs=1e-3)
        assert 50 < s.tof_total / 3600 < 130
        assert 3.10 < s.dv_tli < 3.20
        assert 0.70 < s.loi_dv() < 1.0


def test_apollo11_tli_to_loi_time():
    """Apollo 11 trajectory table (apollomaniacs.com, from the Mission Report):
    S-IVB cutoff 02:50:03 GET at 1,058,809 ft (322.7 km), 35,562.9 ft/s (10.840 km/s);
    'TLI' 02:50:13 GET at 1,103,215 ft (336.3 km), 35,538.5 ft/s (10.832 km/s);
    LOI-1 at 75:49:50 GET -> TLI-to-LOI = 73.0 h.
    The patched conic (circular Moon orbit, Earth radius 6378 km as altitude datum)
    gives 68-73 h for those two states: a 0.1 km^2/s^2 change in injection energy
    moves the lunar TOF by ~5 h, so the result is as good as the altitude datum."""
    tofs = []
    for alt, v0 in ((322.7, 10.840), (336.3, 10.832)):
        sols = design_lunar_transfer(alt, 110, v0=v0, phi0_deg=7.2, direction="retrograde")
        assert sols, "no retrograde solution found"
        s = sols[0]
        tofs.append(s.tof_total / 3600)
        assert s.tof_soi > 5 * 3600          # the SOI transit is a sizeable part of the trip
        assert 0.85 < s.loi_dv() < 0.95      # Apollo 11 LOI-1 was 2924 ft/s = 0.891 km/s
    assert min(tofs) > 62 and max(tofs) < 78


# ---------------------------------------------------------------------------
# Launch, aerobraking, staging
# ---------------------------------------------------------------------------
def test_launch_site_credit_ksc():
    L = launch_to_orbit(300, None, 28.5)
    assert L.v_site == pytest.approx(0.409, abs=0.002)
    assert L.dv_ideal == pytest.approx(L.v_orbit - L.v_site, abs=1e-9)
    assert 8.4 < L.dv_low < L.dv_high < 9.5
    with pytest.raises(ValueError):
        launch_to_orbit(300, 10.0, 28.5)


def test_aerobraking_analytic_vs_numeric():
    beta, peri, apo = 50.0, 110.0, 40000.0
    num = aerobraking_pass_dv_numeric("Mars", peri, apo, beta)
    rp = MARS.radius + peri
    Vp = vis_viva(MARS.mu, rp, 0.5 * (rp + MARS.radius + apo))
    ana = aerobraking_pass_dv(num["rho_p"], Vp, MARS.atmosphere.H, rp, beta, MARS.mu)
    straight = aerobraking_pass_dv(num["rho_p"], Vp, MARS.atmosphere.H, rp, beta, None)
    assert ana == pytest.approx(num["dv_pass"], rel=0.02)
    # the straight-line (slide) form is low by sqrt(V^2/(V^2 - mu/r)) ~ sqrt(2) near parabolic
    assert ana / straight == pytest.approx(math.sqrt(Vp ** 2 / (Vp ** 2 - MARS.mu / rp)), rel=1e-6)


def test_aerocapture_credit_mars():
    ac = aerocapture_credit("Mars", 2.65, 400)
    assert 2.0 < ac.dv_saved < 2.3
    assert ac.dv_periapsis_raise < 0.1


def test_staging_identities():
    v = design_vehicle(9.3, 10000, [300, 450], [0.08, 0.10])
    s1, s2 = v.stages
    assert s1.m0 / s1.mf == pytest.approx(s1.mass_ratio)
    assert s2.m0 / s2.mf == pytest.approx(s2.mass_ratio)
    assert s1.m_payload_above == pytest.approx(s2.m0)
    assert tsiolkovsky_dv(300, s1.m0, s1.mf) + tsiolkovsky_dv(450, s2.m0, s2.mf) == pytest.approx(9.3)
    # the optimal split should beat an equal split
    v_eq = design_vehicle(9.3, 10000, [300, 450], [0.08, 0.10], split=[4.65, 4.65])
    assert v.m0 < v_eq.m0


def test_optimal_split_equal_for_identical_stages():
    assert optimal_dv_split(9.0, [350, 350, 350], [0.1, 0.1, 0.1]) == pytest.approx([3.0, 3.0, 3.0])


def test_required_isp_roundtrip():
    isp = required_isp(4.0, 0.6)
    assert propellant_fraction_for_dv(isp, 4.0) == pytest.approx(0.6)


def test_budget_composition():
    b, m = interplanetary_budget("Earth", "Mars", "2026-11-20", "2027-08-15", margin=0.05)
    assert b.total_propulsive() > 8.0
    rows = b.size_propellant(1000.0, 320.0)
    assert rows[0]["m_before"] > rows[-1]["m_after"]
    d = b.to_dict()
    assert d["total_propulsive"] == pytest.approx(b.total_propulsive())
