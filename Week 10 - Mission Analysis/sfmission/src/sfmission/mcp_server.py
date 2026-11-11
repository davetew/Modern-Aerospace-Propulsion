"""MCP server exposing the mission-analysis tools to agents.

Run:  sfmission-mcp            (stdio transport, for Claude Desktop / Code, Antigravity, Codex)
      python -m sfmission.mcp_server --http 8765   (streamable HTTP on a port)

Works with both the 1.x SDK (``mcp.server.fastmcp.FastMCP``) and the 2.x SDK
(``mcp.server.mcpserver.MCPServer``).  All tools return JSON-serialisable
dicts; angles in degrees, altitudes in km, speeds in km/s, epochs ISO-8601.

Claude Desktop config example:
  "sfmission": {"command": "sfmission-mcp"}
"""
from __future__ import annotations

import json
import math
import os
import sys
from typing import Optional

import numpy as np

from . import __version__
from .bodies import get_body, BODIES, PLANETS
from .ephemeris import body_state, to_jd, jd_to_iso, phase_angle as _phase_angle
from .transfers import hohmann as _hohmann, bielliptic as _bielliptic, hohmann_with_plane_change, synodic_period
from .patched_conics import (departure_dv as _departure_dv, capture_dv as _capture_dv, heliocentric_leg,
                             interplanetary_mission, hohmann_interplanetary, flyby_max_heliocentric_dv)
from .lunar import lunar_hohmann_estimate, design_lunar_transfer
from .launch import launch_to_orbit, LossBand
from .aerobraking import aerocapture_credit, aerobraking_campaign
from .staging import design_vehicle, required_isp as _required_isp, propellant_fraction_for_dv, Burn, size_burn_sequence
from .budget import interplanetary_budget, lunar_budget
from .porkchop import porkchop as _porkchop, next_launch_windows

try:                       # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server
except Exception:          # mcp 2.x
    from mcp.server.mcpserver import MCPServer as _Server

mcp = _Server("sfmission", instructions=(
    "Patched-conic mission analysis for lunar and interplanetary design: Hohmann and Lambert transfers, "
    "porkchop launch windows, departure/capture burns, gravity assists, Earth-Moon patched conics, launch "
    "dv to orbit, aerobraking/aerocapture credits, Tsiolkovsky staging and dv budgets. Units: km, km/s, "
    "seconds unless stated; angles in degrees; epochs as ISO-8601 strings."))

FIG_DIR = os.environ.get("SFMISSION_FIG_DIR", os.path.join(os.getcwd(), "sfmission_figures"))


def _vec(v) -> list[float]:
    return [float(x) for x in np.asarray(v).ravel()]


def _savefig(fig, name: str) -> str:
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=180)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
@mcp.tool()
def list_bodies() -> dict:
    """List the bodies known to the tool with mu [km^3/s^2], mean radius [km] and sphere of influence [km]."""
    return {b.name: dict(mu=b.mu, radius=b.radius, soi=b.soi, parent=b.parent, a_mean=b.a_mean,
                         has_atmosphere_model=b.atmosphere is not None) for b in BODIES.values()}


@mcp.tool()
def planet_state(body: str, epoch: str, frame: str = "ecliptic") -> dict:
    """Heliocentric position [km] and velocity [km/s] of a planet (or the Moon) at an ISO epoch,
    from the JPL mean-element ephemeris (J2000 ecliptic or equatorial frame)."""
    r, v = body_state(body, epoch, frame)
    return dict(body=get_body(body).name, epoch=epoch, frame=frame, r_km=_vec(r), v_kms=_vec(v),
                r_mag_au=float(np.linalg.norm(r)) / 149597870.7, speed_kms=float(np.linalg.norm(v)))


@mcp.tool()
def phase_angle(origin: str, target: str, epoch: str) -> dict:
    """Heliocentric lead angle of target over origin [deg] at an epoch, and the Hohmann requirement."""
    from .transfers import hohmann_phase_angle
    from .bodies import MU_SUN
    o, t = get_body(origin), get_body(target)
    return dict(epoch=epoch, phase_angle_deg=_phase_angle(o.name, t.name, epoch),
                hohmann_required_deg=math.degrees(hohmann_phase_angle(MU_SUN, o.a_mean, t.a_mean)),
                synodic_period_days=synodic_period(2 * math.pi * math.sqrt(o.a_mean ** 3 / MU_SUN),
                                                   2 * math.pi * math.sqrt(t.a_mean ** 3 / MU_SUN)) / 86400)


@mcp.tool()
def hohmann_transfer(central_body: str, r1_km: float, r2_km: float, plane_change_deg: float = 0.0) -> dict:
    """Hohmann transfer between circular orbits of radii r1, r2 [km] about a body, with optional
    total plane change split optimally between the two burns.  Returns dv1, dv2, total [km/s], TOF [s]."""
    b = get_body(central_body)
    if plane_change_deg:
        r = hohmann_with_plane_change(b.mu, r1_km, r2_km, math.radians(plane_change_deg))
        return dict(dv1=r.dv1, dv2=r.dv2, dv_total=r.dv_total, tof_s=r.hohmann.tof, tof_days=r.hohmann.tof / 86400,
                    plane_change_at_burn1_deg=math.degrees(r.di1), plane_change_at_burn2_deg=math.degrees(r.di2),
                    transfer_a_km=r.hohmann.a_t, transfer_e=r.hohmann.e_t)
    h = _hohmann(b.mu, r1_km, r2_km)
    return dict(dv1=h.dv1, dv2=h.dv2, dv_total=h.dv_total, tof_s=h.tof, tof_days=h.tof / 86400, v1_circ=h.v1,
                v2_circ=h.v2, vt1=h.vt1, vt2=h.vt2, transfer_a_km=h.a_t, transfer_e=h.e_t)


@mcp.tool()
def bielliptic_transfer(central_body: str, r1_km: float, r2_km: float, rb_km: float) -> dict:
    """Bi-elliptic transfer via intermediate apoapsis rb; compare with Hohmann."""
    b = get_body(central_body)
    be = _bielliptic(b.mu, r1_km, r2_km, rb_km)
    h = _hohmann(b.mu, r1_km, r2_km)
    return dict(dv1=be.dv1, dv2=be.dv2, dv3=be.dv3, dv_total=be.dv_total, tof_days=be.tof / 86400,
                hohmann_dv_total=h.dv_total, hohmann_tof_days=h.tof / 86400)


@mcp.tool()
def interplanetary_hohmann(origin: str, target: str, park_alt_km: float = 300.0,
                           capture_peri_alt_km: Optional[float] = 300.0, capture_apo_alt_km: Optional[float] = None) -> dict:
    """Textbook circular-coplanar estimate: departure/arrival v_inf, injection dv from a parking orbit,
    capture dv, TOF, required phase angle and synodic period."""
    h = hohmann_interplanetary(origin, target, park_alt_km, capture_peri_alt_km, capture_apo_alt_km)
    return dict(vinf_dep=h.vinf_dep, vinf_arr=h.vinf_arr, c3=h.vinf_dep ** 2, dv_departure=h.departure.dv,
                dv_capture=(h.capture.dv if h.capture else None), dv_total=h.dv_total, tof_days=h.tof / 86400,
                phase_angle_deg=math.degrees(h.phase_angle), synodic_period_days=h.synodic_period / 86400,
                summary=h.summary())


@mcp.tool()
def lambert_leg(origin: str, target: str, depart: str, arrive: str, park_alt_km: float = 300.0,
                capture_peri_alt_km: Optional[float] = 300.0, capture_apo_alt_km: Optional[float] = None) -> dict:
    """Ephemeris-based transfer between two bodies on given ISO dates (Lambert arc): C3, v_inf at both
    ends, departure and capture dv, transfer angle, declination of the launch asymptote."""
    leg = heliocentric_leg(origin, target, depart, arrive)
    leg_eq = heliocentric_leg(origin, target, depart, arrive, frame="equatorial")
    dep = _departure_dv(origin, leg.vinf_dep_mag, park_alt_km)
    out = dict(origin=leg.origin, target=leg.target, depart=depart, arrive=arrive, tof_days=leg.tof_days,
               c3=leg.c3, vinf_dep=leg.vinf_dep_mag, vinf_arr=leg.vinf_arr_mag, dv_departure=dep.dv,
               transfer_angle_deg=math.degrees(leg.lambert.dtheta), transfer_type="I" if leg.lambert.dtheta < math.pi else "II",
               dla_equatorial_deg=leg_eq.dla_deg, vinf_dep_vec_ecl=_vec(leg.vinf_dep), vinf_arr_vec_ecl=_vec(leg.vinf_arr))
    if capture_peri_alt_km is not None:
        cap = _capture_dv(target, leg.vinf_arr_mag, capture_peri_alt_km, capture_apo_alt_km)
        out.update(dv_capture=cap.dv, dv_total=dep.dv + cap.dv)
    return out


@mcp.tool()
def porkchop(origin: str, target: str, depart_start: str, depart_end: str, tof_min_days: float, tof_max_days: float,
             park_alt_km: float = 300.0, capture_peri_alt_km: Optional[float] = 300.0, capture_apo_alt_km: Optional[float] = None,
             n_dep: int = 60, n_tof: int = 60, make_plot: bool = True, quantity: str = "dv_total") -> dict:
    """Launch-window map: grid of departure dates x times of flight -> C3, arrival v_inf, departure and
    capture dv.  Returns the optima (min total dv, min C3), the window span, and optionally saves a
    porkchop PNG (path returned)."""
    pc = _porkchop(origin, target, depart_start, depart_end, tof_min_days, tof_max_days, n_dep, n_tof,
                   park_alt_km, capture_peri_alt_km, capture_apo_alt_km)
    out = dict(summary=pc.summary(), optimum_dv_total=pc.optimum("dv_total"), optimum_c3=pc.optimum("c3"),
               optimum_vinf_arr=pc.optimum("vinf_arr"), window_10pct=pc.window("dv_total"))
    if make_plot:
        from .plotting import plot_porkchop
        fig = plot_porkchop(pc, quantity=quantity)
        out["figure"] = _savefig(fig, f"porkchop_{pc.origin}_{pc.target}_{depart_start[:10]}.png")
    return out


@mcp.tool()
def launch_windows(origin: str, target: str, start: str, years: float = 10.0, park_alt_km: float = 300.0,
                   capture_peri_alt_km: Optional[float] = 300.0) -> dict:
    """List successive launch opportunities (one per synodic period) over a span of years with their
    optimum total dv, C3 and TOF."""
    return dict(windows=next_launch_windows(origin, target, start, years, park_alt_km, capture_peri_alt_km))


@mcp.tool()
def departure_dv(body: str, vinf_kms: float, park_alt_km: float = 300.0, park_apo_alt_km: Optional[float] = None) -> dict:
    """Injection dv from a parking orbit onto an escape hyperbola with the given v_inf."""
    d = _departure_dv(body, vinf_kms, park_alt_km, park_apo_alt_km)
    return dict(body=d.body, vinf=d.vinf, c3=d.c3, dv=d.dv, v_park=d.v_park, v_hyperbolic_periapsis=d.v_hyp,
                e_hyperbola=d.e_hyp, summary=d.summary())


@mcp.tool()
def capture_dv(body: str, vinf_kms: float, peri_alt_km: float, apo_alt_km: Optional[float] = None) -> dict:
    """Capture dv at periapsis from a hyperbola into a circular (apo None) or elliptical orbit."""
    c = _capture_dv(body, vinf_kms, peri_alt_km, apo_alt_km)
    return dict(body=c.body, vinf=c.vinf, dv=c.dv, v_hyperbolic_periapsis=c.v_hyp, v_orbit=c.v_orbit,
                period_hours=c.period / 3600 if c.period else None, summary=c.summary())


@mcp.tool()
def gravity_assist_mission(bodies: list[str], epochs: list[str], park_alt_km: float = 300.0,
                           capture_peri_alt_km: Optional[float] = 300.0, capture_apo_alt_km: Optional[float] = None,
                           flyby_min_alt_km: float = 300.0, aerocapture: bool = False) -> dict:
    """Multi-leg patched-conic mission (e.g., ["Earth","Venus","Mars"] with matching ISO epochs):
    Lambert legs, departure dv, powered-flyby dv at each intermediate body (with periapsis altitude and
    feasibility), capture dv, totals."""
    m = interplanetary_mission(bodies, epochs, park_alt_km, capture_peri_alt_km, capture_apo_alt_km,
                               flyby_min_alt_km, aerocapture)
    return dict(summary=m.summary(), dv_departure=m.dv_departure, dv_flybys=m.dv_flybys, dv_capture=m.dv_capture,
                dv_total=m.dv_total, tof_days=m.tof_days, feasible=m.feasible,
                legs=[dict(origin=l.origin, target=l.target, tof_days=l.tof_days, c3=l.c3, vinf_arr=l.vinf_arr_mag,
                           transfer_angle_deg=math.degrees(l.lambert.dtheta)) for l in m.legs],
                flybys=[dict(body=f.body, vinf_in=float(np.linalg.norm(f.vinf_in)), vinf_out=float(np.linalg.norm(f.vinf_out)),
                             turn_angle_deg=math.degrees(f.turn_angle), periapsis_alt_km=f.alt, dv_powered=f.dv_powered,
                             heliocentric_dv=f.delta_v_helio, feasible=f.feasible) for f in m.flybys])


@mcp.tool()
def flyby_capability(body: str, vinf_kms: float, periapsis_alt_km: float) -> dict:
    """Maximum turning angle and heliocentric dv an unpowered flyby can deliver for a given v_inf and
    closest-approach altitude."""
    from .kepler import turn_angle
    b = get_body(body)
    d = turn_angle(b.mu, b.radius + periapsis_alt_km, vinf_kms)
    return dict(body=b.name, vinf=vinf_kms, turn_angle_deg=math.degrees(d),
                max_heliocentric_dv=flyby_max_heliocentric_dv(b, vinf_kms, periapsis_alt_km))


@mcp.tool()
def lunar_transfer(park_alt_km: float = 185.0, perilune_alt_km: float = 100.0, tof_hours: Optional[float] = None,
                   v0_kms: Optional[float] = None, lambda1_deg: Optional[float] = None, direction: str = "retrograde") -> dict:
    """Earth-Moon transfer.  With no v0/lambda1/tof: Hohmann estimate (TLI, LOI, TOF).  With v0 (injection
    speed) or lambda1 (SOI arrival angle) or a target TOF in hours: BMW patched-conic solution(s) reaching
    the requested perilune, with TLI dv, LOI dv, TOF to SOI and to perilune, Moon lead angle at TLI."""
    est = lunar_hohmann_estimate(park_alt_km, perilune_alt_km)
    out = dict(hohmann_estimate=dict(dv_tli=est.dv_tli, v0=est.v_tli, dv_loi=est.loi.dv, tof_days=est.tof / 86400,
                                     vinf_moon=est.vinf_moon, dv_total=est.dv_total))
    if tof_hours is not None:
        b, pc = lunar_budget(park_alt_km, perilune_alt_km, tof_hours=tof_hours, include_launch=False, margin=0.0)
        sols = [pc]
    elif v0_kms is not None or lambda1_deg is not None:
        sols = design_lunar_transfer(park_alt_km, perilune_alt_km, v0=v0_kms, lambda1_deg=lambda1_deg, direction=direction)
    else:
        sols = []
    out["patched_conic_solutions"] = [
        dict(v0=s.v0, lambda1_deg=math.degrees(s.lambda1), dv_tli=s.dv_tli, dv_loi_circular=s.loi_dv(),
             tof_to_soi_hours=s.tof / 3600, tof_soi_to_perilune_hours=s.tof_soi / 3600, tof_total_hours=s.tof_total / 3600,
             vinf_moon=s.vinf, perilune_alt_km=s.perilune_alt, direction=s.direction,
             moon_lead_angle_at_tli_deg=math.degrees(s.gamma0), summary=s.summary()) for s in sols]
    return out


@mcp.tool()
def launch_dv(altitude_km: float = 300.0, inclination_deg: Optional[float] = None, latitude_deg: float = 28.5,
              body: str = "Earth", loss_low_kms: float = 1.15, loss_high_kms: float = 2.05) -> dict:
    """Delta-v to reach a circular orbit: ideal (with launch-site rotation credit and azimuth for the
    inclination) plus a loss band.  Default band 1.15-2.05 km/s (gravity+drag+steering)."""
    lb = LossBand((loss_low_kms, loss_high_kms), (0.0, 0.0), (0.0, 0.0))
    L = launch_to_orbit(altitude_km, inclination_deg, latitude_deg, body, lb)
    return dict(v_orbit=L.v_orbit, v_site=L.v_site, azimuth_deg=math.degrees(L.azimuth), dv_ideal=L.dv_ideal,
                dv_low=L.dv_low, dv_high=L.dv_high, dv_mid=L.dv_mid, summary=L.summary())


@mcp.tool()
def aerocapture(body: str, vinf_kms: float, orbit_peri_alt_km: float, orbit_apo_alt_km: Optional[float] = None) -> dict:
    """Delta-v credit from aerocapture vs propulsive capture, entry speed at the atmospheric interface,
    and the post-pass periapsis-raise burn."""
    ac = aerocapture_credit(body, vinf_kms, orbit_peri_alt_km, orbit_apo_alt_km)
    return dict(body=ac.body, entry_speed=ac.entry_speed, dv_saved=ac.dv_saved, dv_periapsis_raise=ac.dv_periapsis_raise,
                dv_net_saving=ac.dv_net_saving, summary=ac.summary())


@mcp.tool()
def aerobraking(body: str, vinf_kms: float, peri_alt_km: float, apo_alt_start_km: float, apo_alt_end_km: float,
                ballistic_coefficient: float = 50.0, rho_periapsis: Optional[float] = None) -> dict:
    """Pass-by-pass aerobraking campaign: number of passes, duration, dv removed by drag, propulsive dv
    (capture-high + final periapsis raise) vs direct capture, net saving.  beta = m/(C_D A) [kg/m^2];
    rho_periapsis [kg/m^3] overrides the body's exponential atmosphere."""
    c = aerobraking_campaign(body, vinf_kms, peri_alt_km, apo_alt_start_km, apo_alt_end_km, ballistic_coefficient,
                             rho_p=rho_periapsis)
    return dict(body=c.body, n_passes=c.n_passes, duration_days=c.duration / 86400, dv_drag_total=c.dv_drag_total,
                dv_capture_high=c.dv_capture_high, dv_periapsis_raise=c.dv_periapsis_raise,
                dv_direct_capture=c.dv_direct_capture, dv_saved=c.dv_saved, first_pass_dv=c.pass_dv[0] if c.pass_dv else None,
                summary=c.summary())


@mcp.tool()
def stage_vehicle(dv_total_kms: float, payload_kg: float, isps_s: list[float], structural_fractions: list[float],
                  dv_split_kms: Optional[list[float]] = None, names: Optional[list[str]] = None) -> dict:
    """Size an N-stage vehicle for a total dv (optimal dv split unless given): stage propellant and
    structural masses, gross mass, payload fraction.  Structural fraction sigma = m_s/(m_s+m_p)."""
    v = design_vehicle(dv_total_kms, payload_kg, isps_s, structural_fractions, names, dv_split_kms)
    return dict(m0=v.m0, payload_fraction=v.payload_fraction, dv_total=v.dv_total,
                stages=[dict(name=s.name, isp=s.isp, sigma=s.structural_fraction, dv=s.dv, mass_ratio=s.mass_ratio,
                             m_prop=s.m_prop, m_struct=s.m_struct, m0=s.m0, payload_ratio=s.payload_ratio) for s in v.stages],
                table=v.table())


@mcp.tool()
def rocket_equation(dv_kms: Optional[float] = None, isp_s: Optional[float] = None, propellant_fraction: Optional[float] = None) -> dict:
    """Solve the rocket equation for the missing quantity among dv, Isp, propellant fraction
    zeta = m_p/m_0 (give two of the three)."""
    if dv_kms is not None and isp_s is not None:
        z = propellant_fraction_for_dv(isp_s, dv_kms)
        return dict(dv=dv_kms, isp=isp_s, propellant_fraction=z, mass_ratio=1 / (1 - z))
    if dv_kms is not None and propellant_fraction is not None:
        return dict(dv=dv_kms, propellant_fraction=propellant_fraction, isp=_required_isp(dv_kms, propellant_fraction),
                    mass_ratio=1 / (1 - propellant_fraction))
    if isp_s is not None and propellant_fraction is not None:
        from .staging import exhaust_velocity
        return dict(isp=isp_s, propellant_fraction=propellant_fraction,
                    dv=exhaust_velocity(isp_s) * math.log(1 / (1 - propellant_fraction)), mass_ratio=1 / (1 - propellant_fraction))
    raise ValueError("give two of dv_kms, isp_s, propellant_fraction")


@mcp.tool()
def burn_sequence_propellant(burns: list[dict], final_mass_kg: float) -> dict:
    """Backward propellant sizing for a sequence of burns [{name, dv_kms, isp_s, mass_dropped_after_kg}]
    ending at final_mass_kg (payload + dry mass after the last burn)."""
    bl = [Burn(b["name"], b["dv_kms"], b["isp_s"], b.get("mass_dropped_after_kg", 0.0)) for b in burns]
    rows = size_burn_sequence(bl, final_mass_kg)
    return dict(initial_mass=rows[0]["m_before"] if rows else final_mass_kg, total_propellant=sum(r["m_prop"] for r in rows), burns=rows)


@mcp.tool()
def mission_budget(origin: str, target: str, depart: str, arrive: str, park_alt_km: float = 300.0,
                   launch_lat_deg: float = 28.5, capture_peri_alt_km: Optional[float] = 300.0,
                   capture_apo_alt_km: Optional[float] = None, aerocapture: bool = False, include_launch: bool = True,
                   margin: float = 0.05, flybys: Optional[list[str]] = None, flyby_epochs: Optional[list[str]] = None,
                   landing_dv_kms: Optional[float] = None, stage_isp_s: Optional[float] = None,
                   final_mass_kg: Optional[float] = None, make_plot: bool = False) -> dict:
    """End-to-end dv budget for an interplanetary mission on given dates: launch (with loss band),
    injection, powered flybys, capture or aerocapture, landing, margin.  Optionally sizes propellant for a
    single-stage spacecraft (stage_isp_s, final_mass_kg) and saves a budget bar chart."""
    b, m = interplanetary_budget(origin, target, depart, arrive, park_alt_km, launch_lat_deg, capture_peri_alt_km,
                                 capture_apo_alt_km, aerocapture, include_launch, margin, flybys or (), flyby_epochs or (),
                                 landing_dv_kms)
    out = dict(budget=b.to_dict(), table=b.table(), mission_summary=m.summary())
    if stage_isp_s is not None and final_mass_kg is not None:
        out["propellant"] = b.size_propellant(final_mass_kg, stage_isp_s)
        out["propellant_fraction_post_launch"] = b.propellant_fraction(stage_isp_s)
    if make_plot:
        from .plotting import plot_budget
        out["figure"] = _savefig(plot_budget(b), f"budget_{get_body(origin).name}_{get_body(target).name}_{depart[:10]}.png")
    return out


@mcp.tool()
def lunar_mission_budget(park_alt_km: float = 185.0, lunar_orbit_alt_km: float = 100.0, launch_lat_deg: float = 28.5,
                         tof_hours: Optional[float] = None, land: bool = False, ascend: bool = False,
                         return_to_earth: bool = False, margin: float = 0.05) -> dict:
    """Earth -> lunar orbit [-> surface -> orbit -> Earth] dv budget (launch band, TLI, LOI, Apollo-reference
    descent/ascent, TEI, margin)."""
    b, d = lunar_budget(park_alt_km, lunar_orbit_alt_km, launch_lat_deg, tof_hours, True, land, ascend, return_to_earth, margin)
    return dict(budget=b.to_dict(), table=b.table(), detail=d.summary())


@mcp.tool()
def version() -> dict:
    return dict(sfmission=__version__)


def main():
    if "--http" in sys.argv:
        port = int(sys.argv[sys.argv.index("--http") + 1]) if len(sys.argv) > sys.argv.index("--http") + 1 else 8765
        try:
            mcp.settings.port = port
        except Exception:
            pass
        mcp.run(transport="streamable-http")
    else:
        mcp.run()


if __name__ == "__main__":
    main()
