import math
import numpy as np
import pytest

from sfmission.kepler import (vis_viva, circular_velocity, period, coe_from_rv, rv_from_coe, ClassicalElements,
                              propagate, kepler_E, stumpff_C, stumpff_S, turn_angle, periapsis_for_turn_angle)
from sfmission.lambert import lambert, LambertError
from sfmission.bodies import EARTH, MU_SUN, AU_KM

MU = EARTH.mu


def test_vis_viva_circular_and_escape():
    r = 7000.0
    assert vis_viva(MU, r, r) == pytest.approx(circular_velocity(MU, r))
    assert vis_viva(MU, r, math.inf) == pytest.approx(math.sqrt(2 * MU / r))


def test_period_leo():
    # 400 km circular orbit: ~92.6 min
    assert period(MU, EARTH.radius + 400) / 60 == pytest.approx(92.56, abs=0.05)


def test_kepler_equation():
    for e in (0.0, 0.3, 0.9, 0.99):
        for M in np.linspace(-3, 9, 13):
            E = kepler_E(M, e)
            resid = (E - e * math.sin(E) - M + math.pi) % (2 * math.pi) - math.pi
            assert resid == pytest.approx(0.0, abs=1e-10)


def test_stumpff_continuity():
    assert stumpff_C(1e-9) == pytest.approx(0.5, abs=1e-8)
    assert stumpff_S(1e-9) == pytest.approx(1 / 6, abs=1e-8)
    assert stumpff_C(-1e-9) == pytest.approx(0.5, abs=1e-8)


def test_elements_roundtrip():
    # Curtis Example 4.3: r = (-6045, -3490, 2500), v = (-3.457, 6.618, 2.533)
    r = np.array([-6045.0, -3490.0, 2500.0])
    v = np.array([-3.457, 6.618, 2.533])
    coe = coe_from_rv(r, v, 398600.0)
    assert coe.h == pytest.approx(58310, rel=1e-3)
    assert coe.e == pytest.approx(0.1712, abs=2e-4)
    assert math.degrees(coe.i) == pytest.approx(153.2, abs=0.1)
    assert math.degrees(coe.raan) == pytest.approx(255.3, abs=0.1)
    assert math.degrees(coe.argp) == pytest.approx(20.07, abs=0.05)
    assert math.degrees(coe.nu) == pytest.approx(28.45, abs=0.05)
    r2, v2 = rv_from_coe(coe)
    assert np.allclose(r2, r, atol=1e-6)
    assert np.allclose(v2, v, atol=1e-9)


def test_propagation_closes_after_one_period():
    coe = ClassicalElements(a=12000.0, e=0.4, i=0.5, raan=1.0, argp=0.3, nu=0.7, mu=MU)
    r0, v0 = rv_from_coe(coe)
    r1, v1 = propagate(r0, v0, coe.period, MU)
    assert np.allclose(r1, r0, atol=1e-5)
    assert np.allclose(v1, v0, atol=1e-9)


def test_propagation_hyperbolic():
    r0 = np.array([7000.0, 0.0, 0.0])
    v0 = np.array([0.0, 12.0, 0.0])  # > escape speed
    r, v = propagate(r0, v0, 5000.0, MU)
    e0 = 0.5 * 144 - MU / 7000
    e1 = 0.5 * float(np.dot(v, v)) - MU / np.linalg.norm(r)
    assert e1 == pytest.approx(e0, rel=1e-10)


def test_turn_angle_inverse():
    vinf, rp = 5.0, 10000.0
    d = turn_angle(MU, rp, vinf)
    assert periapsis_for_turn_angle(MU, vinf, d) == pytest.approx(rp)


# ---------------------------------------------------------------------------
def test_lambert_curtis_example_5_2():
    r1 = np.array([5000.0, 10000.0, 2100.0])
    r2 = np.array([-14600.0, 2500.0, 7000.0])
    s = lambert(r1, r2, 3600.0, 398600.0, prograde=True)
    assert np.allclose(s.v1, [-5.9925, 1.9254, 3.2456], atol=1e-4)
    assert np.allclose(s.v2, [-3.3125, -4.1966, -0.38529], atol=1e-4)


@pytest.mark.parametrize("prograde", [True, False])
@pytest.mark.parametrize("dtheta_deg,tof_days", [(60, 120), (150, 250), (250, 320), (330, 500), (120, 40)])
def test_lambert_closure(prograde, dtheta_deg, tof_days):
    r1 = np.array([AU_KM, 0.0, 0.0])
    th = math.radians(dtheta_deg)
    r2 = 1.5 * AU_KM * np.array([math.cos(th), math.sin(th), 0.05])
    s = lambert(r1, r2, tof_days * 86400.0, MU_SUN, prograde=prograde)
    r2p, v2p = propagate(r1, s.v1, tof_days * 86400.0, MU_SUN)
    assert np.allclose(r2p, r2, atol=1e-3)
    assert np.allclose(v2p, s.v2, atol=1e-9)


def test_lambert_recovers_hohmann():
    a_t = 0.5 * (EARTH.a_mean + 1.52371034 * AU_KM)
    tof = math.pi * math.sqrt(a_t ** 3 / MU_SUN)
    r1 = np.array([EARTH.a_mean, 0.0, 0.0])
    th = math.radians(179.95)
    r2 = 1.52371034 * AU_KM * np.array([math.cos(th), math.sin(th), 0.0])
    s = lambert(r1, r2, tof, MU_SUN)
    assert np.linalg.norm(s.v1) == pytest.approx(vis_viva(MU_SUN, EARTH.a_mean, a_t), rel=1e-4)
    assert s.a == pytest.approx(a_t, rel=1e-4)


def test_lambert_rejects_180deg():
    r1 = np.array([AU_KM, 0.0, 0.0])
    r2 = np.array([-1.5 * AU_KM, 0.0, 0.0])
    with pytest.raises(LambertError):
        lambert(r1, r2, 200 * 86400.0, MU_SUN)
