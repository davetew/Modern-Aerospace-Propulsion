import math
import numpy as np
import pytest

from modern_aerospace_propulsion.mission_analysis import ephemeris as eph
from modern_aerospace_propulsion.mission_analysis.bodies import AU_KM

astropy = pytest.importorskip("astropy")
from astropy.time import Time                                   # noqa: E402
from astropy.coordinates import get_body_barycentric_posvel, get_body, solar_system_ephemeris  # noqa: E402
import astropy.units as u                                       # noqa: E402

solar_system_ephemeris.set("builtin")


def _astropy_helio_ecl(name, iso):
    t = Time(iso + " 00:00:00", scale="utc")
    rb, vb = get_body_barycentric_posvel(name, t)
    rs, vs = get_body_barycentric_posvel("sun", t)
    r = (rb - rs).xyz.to(u.km).value
    v = (vb - vs).xyz.to(u.km / u.s).value
    return eph.equatorial_to_ecliptic(r), eph.equatorial_to_ecliptic(v)


def _ang_arcsec(a, b):
    c = float(np.dot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))
    return math.degrees(math.acos(max(-1.0, min(1.0, c)))) * 3600


# JPL's own quoted maximum errors for Table 1 (1800-2050), arcsec, with margin
TOL_ARCSEC = {"mercury": 60, "venus": 60, "earth": 40, "mars": 100, "jupiter": 600, "saturn": 1200,
              "uranus": 400, "neptune": 300}


@pytest.mark.parametrize("name", list(TOL_ARCSEC))
@pytest.mark.parametrize("date", ["2000-01-01", "2020-07-30", "2026-11-12", "2045-01-01"])
def test_planets_vs_astropy_builtin(name, date):
    ra, va = _astropy_helio_ecl("earth-moon-barycenter" if name == "earth" else name, date)
    ro, vo = eph.planet_state(name, date)
    assert _ang_arcsec(ro, ra) < TOL_ARCSEC[name]
    rel_tol = 3e-4 if name in ("mercury", "venus", "earth", "mars") else 2e-3
    assert abs(np.linalg.norm(ro) - np.linalg.norm(ra)) / np.linalg.norm(ra) < rel_tol
    assert np.linalg.norm(vo - va) < 0.05  # km/s


@pytest.mark.parametrize("date", ["2000-01-01", "2020-07-30", "2026-11-12", "2026-11-25", "2040-03-01"])
def test_moon_vs_astropy_builtin(date):
    t = Time(date + " 00:00:00", scale="utc")
    m = get_body("moon", t)
    rm = eph.equatorial_to_ecliptic(m.cartesian.xyz.to(u.km).value)
    ro = eph.moon_position_geocentric(date)
    assert _ang_arcsec(ro, rm) < 120
    assert abs(np.linalg.norm(ro) - np.linalg.norm(rm)) < 100


def test_moon_speed_and_distance_range():
    r, v = eph.moon_state_geocentric("2026-11-12")
    assert 0.95 < np.linalg.norm(v) < 1.09
    assert 356000 < np.linalg.norm(r) < 407000


@pytest.mark.parametrize("date", ["2022-12-08", "2025-01-16", "2027-02-19"])
def test_mars_oppositions(date):
    """At opposition Earth and Mars share heliocentric longitude (within ~1 deg
    because of orbit inclinations / ecliptic-longitude definitions)."""
    dl = (eph.heliocentric_longitude("mars", date) - eph.heliocentric_longitude("earth", date) + 180) % 360 - 180
    assert abs(dl) < 1.5


def test_table2_matches_table1_in_overlap():
    for name in ("earth", "mars", "jupiter"):
        r1, _ = eph.planet_state(name, "2010-06-01", table="1")
        r2, _ = eph.planet_state(name, "2010-06-01", table="2")
        assert _ang_arcsec(r1, r2) < 2000   # tables agree to ~0.5 deg (Table 2 is coarser)


def test_jd_conversion():
    assert eph.to_jd("2000-01-01T12:00:00") == pytest.approx(2451545.0)
    assert eph.jd_to_iso(2451545.0)[:16] == "2000-01-01T12:00"
