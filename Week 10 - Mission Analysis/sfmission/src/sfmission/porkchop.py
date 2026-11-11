"""Porkchop plots: launch-window maps of C3, arrival v_inf and total Delta-v.

For a grid of departure dates x times of flight the heliocentric Lambert arc
between the ephemeris positions is solved and the planetocentric departure /
capture burns are evaluated with the patched-conic relations.  Both type-I
(<180 deg) and type-II (>180 deg) transfers appear naturally; the Lambert
solver is single-revolution.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .bodies import get_body, MU_SUN
from .ephemeris import body_state, to_jd, jd_to_iso, Epoch
from .lambert import lambert, LambertError
from .kepler import hyperbolic_periapsis_speed, circular_velocity, vis_viva


@dataclass
class Porkchop:
    origin: str
    target: str
    jd_dep: np.ndarray        # (n_dep,)
    tof_days: np.ndarray      # (n_tof,)
    c3: np.ndarray            # (n_tof, n_dep) km^2/s^2
    vinf_arr: np.ndarray      # km/s
    dv_dep: np.ndarray        # km/s, from parking orbit
    dv_arr: np.ndarray        # km/s, capture (0 if not requested)
    dtheta: np.ndarray        # rad, transfer angle
    park_alt: float
    capture_peri_alt: Optional[float]
    capture_apo_alt: Optional[float]

    @property
    def dv_total(self) -> np.ndarray:
        return self.dv_dep + self.dv_arr

    @property
    def jd_arr(self) -> np.ndarray:
        return self.jd_dep[None, :] + self.tof_days[:, None]

    def _argmin(self, field: np.ndarray) -> tuple[int, int]:
        f = np.where(np.isnan(field), np.inf, field)
        k = int(np.argmin(f))
        return np.unravel_index(k, field.shape)  # (i_tof, j_dep)

    def optimum(self, quantity: str = "dv_total") -> dict:
        """Grid minimum of 'c3', 'vinf_arr', 'dv_dep', 'dv_arr' or 'dv_total'."""
        field = getattr(self, quantity)
        i, j = self._argmin(field)
        return dict(quantity=quantity, value=float(field[i, j]), depart=jd_to_iso(float(self.jd_dep[j]), "hours"),
                    arrive=jd_to_iso(float(self.jd_dep[j] + self.tof_days[i]), "hours"), tof_days=float(self.tof_days[i]),
                    c3=float(self.c3[i, j]), vinf_arr=float(self.vinf_arr[i, j]), dv_dep=float(self.dv_dep[i, j]),
                    dv_arr=float(self.dv_arr[i, j]), dv_total=float(self.dv_total[i, j]),
                    transfer_angle_deg=math.degrees(float(self.dtheta[i, j])))

    def window(self, quantity: str = "dv_total", threshold: float | None = None, margin: float = 0.10) -> dict:
        """Departure-date span over which min_tof(quantity) <= threshold
        (default: (1+margin) x grid optimum)."""
        field = getattr(self, quantity)
        best_per_day = np.nanmin(field, axis=0)
        if threshold is None:
            threshold = (1.0 + margin) * float(np.nanmin(best_per_day))
        ok = np.where(best_per_day <= threshold)[0]
        if len(ok) == 0:
            return dict(open=None, close=None, days=0.0, threshold=threshold)
        return dict(open=jd_to_iso(float(self.jd_dep[ok[0]]), "hours"), close=jd_to_iso(float(self.jd_dep[ok[-1]]), "hours"),
                    days=float(self.jd_dep[ok[-1]] - self.jd_dep[ok[0]]), threshold=threshold)

    def summary(self) -> str:
        o = self.optimum("dv_total")
        c = self.optimum("c3")
        w = self.window("dv_total")
        return (f"Porkchop {self.origin} -> {self.target}: departures {jd_to_iso(float(self.jd_dep[0]),'hours')[:10]} .. "
                f"{jd_to_iso(float(self.jd_dep[-1]),'hours')[:10]}, TOF {self.tof_days[0]:.0f}-{self.tof_days[-1]:.0f} d\n"
                f"  min total dv {o['dv_total']:.3f} km/s: depart {o['depart'][:10]}, TOF {o['tof_days']:.0f} d, "
                f"C3 {o['c3']:.2f} km2/s2, v_inf,arr {o['vinf_arr']:.3f} km/s, type {'I' if o['transfer_angle_deg']<180 else 'II'}\n"
                f"  min C3 {c['c3']:.2f} km2/s2: depart {c['depart'][:10]}, TOF {c['tof_days']:.0f} d\n"
                f"  window (dv_total within 10% of optimum): {w['open'][:10] if w['open'] else '-'} .. "
                f"{w['close'][:10] if w['close'] else '-'} ({w['days']:.0f} d)")


def porkchop(origin, target, depart_start: Epoch, depart_end: Epoch, tof_min_days: float, tof_max_days: float,
             n_dep: int = 80, n_tof: int = 80, park_alt: float = 300.0, capture_peri_alt: Optional[float] = 300.0,
             capture_apo_alt: Optional[float] = None, prograde: bool = True) -> Porkchop:
    """Compute a porkchop grid.  Failed Lambert solutions (e.g., ~180 deg
    transfers) are NaN."""
    o, t = get_body(origin), get_body(target)
    jd0, jd1 = to_jd(depart_start), to_jd(depart_end)
    jd_dep = np.linspace(jd0, jd1, n_dep)
    tofs = np.linspace(tof_min_days, tof_max_days, n_tof)
    shape = (n_tof, n_dep)
    c3 = np.full(shape, np.nan)
    vinf_a = np.full(shape, np.nan)
    dth = np.full(shape, np.nan)
    # cache body states
    dep_states = [body_state(o.name, float(j)) for j in jd_dep]
    arr_cache: dict[float, tuple] = {}
    rp_o = o.radius + park_alt
    v_park = circular_velocity(o.mu, rp_o)
    for j, jd in enumerate(jd_dep):
        r1, v1 = dep_states[j]
        for i, tof in enumerate(tofs):
            jda = float(jd + tof)
            key = round(jda, 6)
            if key not in arr_cache:
                arr_cache[key] = body_state(t.name, jda)
            r2, v2 = arr_cache[key]
            try:
                s = lambert(r1, r2, tof * 86400.0, MU_SUN, prograde=prograde)
            except LambertError:
                continue
            vinf_dep = s.v1 - v1
            vinf_arr = s.v2 - v2
            c3[i, j] = float(np.dot(vinf_dep, vinf_dep))
            vinf_a[i, j] = float(np.linalg.norm(vinf_arr))
            dth[i, j] = s.dtheta
    dv_dep = np.sqrt(c3 + 2 * o.mu / rp_o) - v_park
    if capture_peri_alt is not None:
        rp_t = t.radius + capture_peri_alt
        ra_t = rp_t if capture_apo_alt is None else t.radius + capture_apo_alt
        v_orb = vis_viva(t.mu, rp_t, 0.5 * (rp_t + ra_t))
        dv_arr = np.sqrt(vinf_a ** 2 + 2 * t.mu / rp_t) - v_orb
    else:
        dv_arr = np.zeros(shape)
    return Porkchop(o.name, t.name, jd_dep, tofs, c3, vinf_a, dv_dep, dv_arr, dth, park_alt,
                    capture_peri_alt, capture_apo_alt)


def next_launch_windows(origin, target, start: Epoch, years: float = 20.0, park_alt: float = 300.0,
                        capture_peri_alt: Optional[float] = 300.0, tof_range_days: tuple[float, float] | None = None,
                        n_dep_per_year: int = 73, n_tof: int = 60) -> list[dict]:
    """Coarse scan over ``years`` listing the local minima of the best-per-day
    total dv.  Each synodic opportunity usually yields two entries -- the type-I
    (<180 deg) and type-II (>180 deg) transfer minima -- a few months apart."""
    from .transfers import hohmann
    from .kepler import period
    o, t = get_body(origin), get_body(target)
    h = hohmann(MU_SUN, o.a_mean, t.a_mean)
    if tof_range_days is None:
        tof_range_days = (0.5 * h.tof / 86400.0, 1.8 * h.tof / 86400.0)
    pc = porkchop(o, t, start, to_jd(start) + years * 365.25, tof_range_days[0], tof_range_days[1],
                  int(n_dep_per_year * years), n_tof, park_alt, capture_peri_alt)
    best = np.nanmin(pc.dv_total, axis=0)
    # local minima of the best-per-day curve
    out = []
    for j in range(1, len(best) - 1):
        if best[j] <= best[j - 1] and best[j] <= best[j + 1] and best[j] < 1.5 * np.nanmin(best):
            i = int(np.nanargmin(pc.dv_total[:, j]))
            out.append(dict(depart=jd_to_iso(float(pc.jd_dep[j]), "hours")[:10], tof_days=float(pc.tof_days[i]),
                            c3=float(pc.c3[i, j]), vinf_arr=float(pc.vinf_arr[i, j]), dv_total=float(best[j]),
                            type="I" if pc.dtheta[i, j] < math.pi else "II"))
    return out
