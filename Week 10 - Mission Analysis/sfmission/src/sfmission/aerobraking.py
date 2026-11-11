"""Aerobraking and aerocapture as Delta-v-budget levers.

Aerocapture: the propulsive capture burn is replaced by one atmospheric pass;
the budget credit is the full capture dv (``aerocapture_credit``), at the cost
of a heat-shield mass fraction and entry-corridor constraints.

Aerobraking: capture propulsively into a high ellipse, then lower the apoapsis
over many passes with periapsis inside the upper atmosphere.  The analytic
per-pass dv (exponential atmosphere, flight through periapsis approximated by
a parabola in time) is

    dv_pass = rho_p V_p^2 / (2 beta) * sqrt(2 pi H / rddot_p),
    rddot_p = V_p^2 / r_p - mu / r_p^2     (radial acceleration at periapsis)

where beta = m / (C_D A) is the ballistic coefficient [kg/m^2].  Taking
rddot_p = V_p^2 / r_p (straight-line fly-through, gravity neglected) gives the
compact form rho_p V_p sqrt(2 pi H r_p) / (2 beta) used on the Week-11 slide;
the gravity-aware form is larger by sqrt(V_p^2/(V_p^2 - mu/r_p)) (-> sqrt(2) for
a parabolic pass).  ``aerobraking_pass_dv_numeric`` integrates the pass to
check either closed form.

Units: km, s, kg; rho in kg/m^3; beta in kg/m^2.  Conversions are internal.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.integrate import solve_ivp

from .bodies import Body, get_body, Atmosphere
from .kepler import vis_viva, circular_velocity, period, hyperbolic_periapsis_speed
from .patched_conics import capture_dv


# ---------------------------------------------------------------------------
# Aerocapture
# ---------------------------------------------------------------------------
@dataclass
class AerocaptureResult:
    body: str
    vinf: float
    entry_speed: float      # km/s at the atmospheric interface
    dv_saved: float         # km/s = propulsive capture dv replaced
    dv_periapsis_raise: float   # km/s, small burn at apoapsis to lift periapsis out of the atmosphere
    dv_net_saving: float

    def summary(self) -> str:
        return (f"Aerocapture at {self.body}: entry speed {self.entry_speed:.3f} km/s, replaces capture "
                f"dv={self.dv_saved:.4f} km/s; periapsis-raise burn {self.dv_periapsis_raise:.4f} km/s; "
                f"net saving {self.dv_net_saving:.4f} km/s")


def aerocapture_credit(body, vinf: float, orbit_peri_alt: float, orbit_apo_alt: Optional[float] = None,
                       interface_alt: Optional[float] = None) -> AerocaptureResult:
    """Delta-v saved by aerocapturing into (orbit_peri_alt x orbit_apo_alt) instead of
    burning at periapsis; includes the post-pass periapsis-raise burn from the
    atmospheric interface altitude up to ``orbit_peri_alt``."""
    b = get_body(body)
    hi = interface_alt if interface_alt is not None else (b.atmosphere_interface_alt or 120.0)
    cap = capture_dv(b, vinf, orbit_peri_alt, orbit_apo_alt)
    r_int = b.radius + hi
    v_entry = hyperbolic_periapsis_speed(b.mu, r_int, vinf)
    # after the pass the orbit has periapsis ~ r_int and apoapsis = target apoapsis
    ra = cap.ra
    rp_new = cap.rp
    v_apo_before = vis_viva(b.mu, ra, 0.5 * (r_int + ra))
    v_apo_after = vis_viva(b.mu, ra, 0.5 * (rp_new + ra))
    dv_raise = abs(v_apo_after - v_apo_before)
    return AerocaptureResult(b.name, vinf, v_entry, cap.dv, dv_raise, cap.dv - dv_raise)


# ---------------------------------------------------------------------------
# Aerobraking per-pass dv
# ---------------------------------------------------------------------------
def aerobraking_pass_dv(rho_p: float, V_p: float, H: float, r_p: float, beta: float,
                        mu: Optional[float] = None) -> float:
    """Analytic dv per pass [km/s].

    rho_p [kg/m^3], V_p [km/s], H [km], r_p [km], beta [kg/m^2].
    mu=None -> straight-line form (slide formula); mu given -> gravity-aware form.
    """
    V_p_m = V_p * 1e3
    H_m = H * 1e3
    r_p_m = r_p * 1e3
    if mu is None:
        rddot = V_p_m ** 2 / r_p_m
    else:
        rddot = V_p_m ** 2 / r_p_m - mu * 1e9 / r_p_m ** 2
        if rddot <= 0.0:
            raise ValueError("periapsis radial acceleration <= 0: orbit is circular/sub-circular at periapsis")
    dv = rho_p * V_p_m ** 2 / (2.0 * beta) * math.sqrt(2.0 * math.pi * H_m / rddot)
    return dv / 1e3


def aerobraking_pass_dv_numeric(body, peri_alt: float, apo_alt: float, beta: float,
                                atmosphere: Optional[Atmosphere] = None, n_scale_heights: float = 12.0) -> dict:
    """Integrate one atmospheric pass (planar 2-body + drag, exponential
    atmosphere) and return the speed loss, apoapsis change and peak
    deceleration.  Used to verify the closed-form estimates."""
    b = get_body(body)
    atm = atmosphere or b.atmosphere
    if atm is None:
        raise ValueError(f"{b.name} has no atmosphere model")
    mu = b.mu
    rp = b.radius + peri_alt
    ra = b.radius + apo_alt
    a = 0.5 * (rp + ra)
    e = (ra - rp) / (ra + rp)
    # start at the altitude where density is negligible (n scale heights above periapsis)
    h_start = peri_alt + n_scale_heights * atm.H
    r_start = b.radius + h_start
    if r_start >= ra:
        r_start = 0.5 * (rp + ra)
    cos_nu = (a * (1 - e * e) / r_start - 1.0) / e
    nu0 = -math.acos(max(-1.0, min(1.0, cos_nu)))        # inbound
    p = a * (1 - e * e)
    h = math.sqrt(mu * p)
    vr = mu / h * e * math.sin(nu0)
    vt = h / r_start
    state0 = [r_start * math.cos(nu0), r_start * math.sin(nu0),
              vr * math.cos(nu0) - vt * math.sin(nu0), vr * math.sin(nu0) + vt * math.cos(nu0)]
    peak = [0.0]

    def rhs(t, s):
        x, y, vx, vy = s
        r = math.hypot(x, y)
        v = math.hypot(vx, vy)
        rho = atm.density(r - b.radius)
        # drag acceleration [km/s^2]: 0.5 rho v^2 / beta with v in m/s -> m/s^2 -> /1e3
        ad = 0.5 * rho * (v * 1e3) ** 2 / beta / 1e3
        peak[0] = max(peak[0], ad)
        ax = -mu * x / r ** 3 - ad * vx / v
        ay = -mu * y / r ** 3 - ad * vy / v
        return [vx, vy, ax, ay]

    def exit_event(t, s):
        return math.hypot(s[0], s[1]) - r_start
    exit_event.terminal = True
    exit_event.direction = 1
    T = period(mu, a)
    sol = solve_ivp(rhs, (0.0, T), state0, events=exit_event, rtol=1e-10, atol=1e-12, max_step=5.0)
    x, y, vx, vy = sol.y[:, -1]
    r = math.hypot(x, y)
    v = math.hypot(vx, vy)
    energy = 0.5 * v * v - mu / r
    a_new = -mu / (2 * energy)
    hvec = x * vy - y * vx
    e_new = math.sqrt(max(0.0, 1 - hvec * hvec / (mu * a_new)))
    v0 = math.hypot(state0[2], state0[3])
    # compare speeds at the same radius (energy loss expressed as dv at periapsis)
    dE = energy - (0.5 * v0 * v0 - mu / r_start)
    vp_before = vis_viva(mu, rp, a)
    vp_after = math.sqrt(max(0.0, vp_before ** 2 + 2 * dE))
    return dict(dv_pass=vp_before - vp_after, apo_alt_before=apo_alt,
                apo_alt_after=a_new * (1 + e_new) - b.radius, peak_decel_g=peak[0] / 9.80665e-3,
                rho_p=atm.density(peri_alt), V_p=vp_before)


@dataclass
class AerobrakingCampaign:
    body: str
    peri_alt: float
    apo_alt_start: float
    apo_alt_end: float
    beta: float
    n_passes: int
    duration: float             # s
    dv_drag_total: float        # km/s removed by the atmosphere
    dv_periapsis_raise: float   # km/s, final burn to lift periapsis to apo_alt_end (circularize)
    dv_direct_capture: float    # km/s, propulsive capture straight into the final circular orbit
    dv_capture_high: float      # km/s, propulsive capture into the initial ellipse
    pass_dv: list = field(default_factory=list)
    pass_apo_alt: list = field(default_factory=list)

    @property
    def dv_saved(self) -> float:
        return self.dv_direct_capture - (self.dv_capture_high + self.dv_periapsis_raise)

    def summary(self) -> str:
        return (f"Aerobraking at {self.body}: periapsis {self.peri_alt:.0f} km, apoapsis {self.apo_alt_start:.0f} -> "
                f"{self.apo_alt_end:.0f} km, beta={self.beta:.0f} kg/m2: {self.n_passes} passes over "
                f"{self.duration/86400:.1f} d, drag removed {self.dv_drag_total:.3f} km/s; "
                f"propulsive: capture-high {self.dv_capture_high:.3f} + periapsis raise {self.dv_periapsis_raise:.3f} "
                f"vs direct capture {self.dv_direct_capture:.3f} km/s -> saving {self.dv_saved:.3f} km/s")


def aerobraking_campaign(body, vinf: float, peri_alt: float, apo_alt_start: float, apo_alt_end: float,
                         beta: float, atmosphere: Optional[Atmosphere] = None, rho_p: Optional[float] = None,
                         gravity_aware: bool = True, max_passes: int = 5000) -> AerobrakingCampaign:
    """Pass-by-pass aerobraking from (peri_alt x apo_alt_start) down to a circular
    orbit at apo_alt_end, with a final periapsis-raise burn.  Density at
    periapsis from the body's exponential atmosphere unless ``rho_p`` is given."""
    b = get_body(body)
    atm = atmosphere or b.atmosphere
    if rho_p is None:
        if atm is None:
            raise ValueError("need an atmosphere model or rho_p")
        rho_p = atm.density(peri_alt)
    H = (atm.H if atm else 10.0)
    rp = b.radius + peri_alt
    ra = b.radius + apo_alt_start
    ra_end = b.radius + apo_alt_end
    cap_high = capture_dv(b, vinf, peri_alt, apo_alt_start).dv
    cap_direct = capture_dv(b, vinf, apo_alt_end).dv
    passes, apos, t = [], [], 0.0
    n = 0
    while ra > ra_end and n < max_passes:
        a = 0.5 * (rp + ra)
        Vp = vis_viva(b.mu, rp, a)
        dv = aerobraking_pass_dv(rho_p, Vp, H, rp, beta, b.mu if gravity_aware else None)
        Vp_new = Vp - dv
        energy = 0.5 * Vp_new ** 2 - b.mu / rp
        if energy >= 0:
            raise RuntimeError("energy became non-negative; check inputs")
        a_new = -b.mu / (2 * energy)
        ra = 2 * a_new - rp
        t += period(b.mu, a)
        passes.append(dv)
        apos.append(ra - b.radius)
        n += 1
    # final: at apoapsis (now ~ra_end), raise periapsis to ra_end -> circular
    a_f = 0.5 * (rp + ra)
    v_apo = vis_viva(b.mu, ra, a_f)
    dv_raise = circular_velocity(b.mu, ra) - v_apo
    return AerobrakingCampaign(b.name, peri_alt, apo_alt_start, apo_alt_end, beta, n, t, sum(passes), dv_raise,
                               cap_direct, cap_high, passes, apos)
