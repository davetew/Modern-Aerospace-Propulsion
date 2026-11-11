"""Gravity-turn forward simulation (the slide-33 notebook's guidance law, in ECI) used as
the collocation initial guess and as an independent integrator for verification.

Guidance: vertical for t_rise, an instantaneous pitch kick of `kick_deg` toward the launch
azimuth, then thrust along the air-relative velocity (pure gravity turn) at full throttle.
Stages burn to their usable propellant and separate instantly.  `kick_deg` can be tuned by
`tune_kick` so that the vehicle's flight-path angle is ~0 when propellant runs out, which
gives a guess close to an orbital insertion.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from .constants import MU_E, R_E, OMEGA_E
from .atmosphere import ATM


def launch_state(lat_deg, lon_deg=0.0, R=R_E, omega=OMEGA_E):
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    r0 = R * np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    v0 = np.cross([0.0, 0.0, omega], r0)
    return r0, v0


def local_frame(r):
    """Up, East, North unit vectors at position r."""
    up = r / np.linalg.norm(r)
    east = np.cross([0.0, 0.0, 1.0], up)
    east /= np.linalg.norm(east)
    north = np.cross(up, east)
    return up, east, north


def gravity_turn_guess(vehicle, lat_deg, kick_deg=2.0, t_rise=10.0, azimuth_deg=90.0, lon_deg=0.0,
                       t_kick_dur=5.0, max_time=2000.0, stop_on_energy=None, dense_dt=1.0, a_max=None):
    """Integrate the staged gravity turn.  Returns dict of per-phase time histories.

    Phases: 'rise' (stage 0 vertical), 'stage0', 'stage1', ...  Each has t, r, v, m_prop, u.
    stop_on_energy: if given (m^2/s^2), stop the last stage when v^2/2 - mu/r reaches it.
    """
    az = np.radians(azimuth_deg)
    phases = {}
    r0, v0 = launch_state(lat_deg, lon_deg)
    ns = len(vehicle.stages)

    def throttle(stage, m, T_full):
        """Throttle to hold a_max (if given), within the stage's throttle range."""
        if a_max is None:
            return 1.0
        return float(np.clip(a_max * m / T_full, stage.throttle_min, 1.0))

    def make_rhs(stage, m_struct, mode):
        T_vac, A_e, mdot_full = stage.T_vac, stage.A_e, stage.mdot_max

        def rhs(t, y):
            r, v, mp = y[0:3], y[3:6], y[6]
            rmag = np.linalg.norm(r)
            h = rmag - R_E
            m = mp + m_struct
            v_rel = v - np.cross([0, 0, OMEGA_E], r)
            vr = np.linalg.norm(v_rel)
            pa, rho = ATM.p(h), ATM.rho(h)
            T_full = T_vac - pa * A_e
            tau = throttle(stage, m, T_full)
            T = tau * T_full
            mdot = tau * mdot_full
            D = 0.5 * rho * vr ** 2 * vehicle.cd * vehicle.A_ref
            up, east, north = local_frame(r)
            if mode == "vertical":
                u = up
            elif mode == "kick":
                k = np.radians(kick_deg)
                horiz = np.cos(az) * north + np.sin(az) * east
                u = np.cos(k) * up + np.sin(k) * horiz
            else:
                u = v_rel / max(vr, 1e-6)
            a = -MU_E * r / rmag ** 3 + T / m * u - D / m * v_rel / max(vr, 1e-6)
            return np.concatenate([v, a, [-mdot]])
        return rhs

    t_now = 0.0
    y = np.concatenate([r0, v0, [vehicle.stages[0].propellant]])
    for k, st in enumerate(vehicle.stages):
        m_struct = st.dry_mass + vehicle.mass_above(k) + vehicle.payload
        burn = st.usable_propellant / st.mdot_max
        if a_max is not None:  # throttled burns last longer; integrate until propellant is gone instead
            burn = 10.0 * burn
        segs = []
        if k == 0:
            segs.append(("rise", "vertical", t_rise))
            segs.append(("stage0", "kick", t_kick_dur))
            segs.append(("stage0", "turn", burn - t_rise - t_kick_dur))
        else:
            segs.append((f"stage{k}", "turn", burn))
        for name, mode, dur in segs:
            rhs = make_rhs(st, m_struct, mode)
            usable_floor = st.reserve

            def ground(t, yy):
                return np.linalg.norm(yy[0:3]) - R_E
            ground.terminal, ground.direction = True, -1

            def depleted(t, yy):
                return yy[6] - usable_floor
            depleted.terminal, depleted.direction = True, -1

            def energy_reached(t, yy):
                return 0.5 * np.dot(yy[3:6], yy[3:6]) - MU_E / np.linalg.norm(yy[0:3]) - (stop_on_energy if stop_on_energy is not None else 1e30)
            energy_reached.terminal, energy_reached.direction = True, 1

            events = [ground, depleted, energy_reached]  # fixed order: 0 ground, 1 depleted, 2 energy
            t_eval = np.arange(t_now, t_now + dur + 1e-9, dense_dt)
            if t_eval[-1] < t_now + dur - 1e-9:
                t_eval = np.append(t_eval, t_now + dur)
            sol = solve_ivp(rhs, (t_now, t_now + dur), y, t_eval=t_eval, events=events, rtol=1e-9, atol=1e-6, max_step=2.0)
            rr, vv, mp = sol.y[0:3].T, sol.y[3:6].T, sol.y[6]
            # thrust direction & throttle history for the guess
            uu = np.zeros_like(rr)
            tt = np.ones(len(sol.t))
            for i in range(len(sol.t)):
                up, east, north = local_frame(rr[i])
                if mode == "vertical":
                    uu[i] = up
                elif mode == "kick":
                    kk = np.radians(kick_deg)
                    uu[i] = np.cos(kk) * up + np.sin(kk) * (np.cos(az) * north + np.sin(az) * east)
                else:
                    vrel = vv[i] - np.cross([0, 0, OMEGA_E], rr[i])
                    uu[i] = vrel / np.linalg.norm(vrel)
                hh = np.linalg.norm(rr[i]) - R_E
                tt[i] = throttle(st, mp[i] + m_struct, st.T_vac - ATM.p(hh) * st.A_e)
            ph = phases.setdefault(name, dict(t=[], r=[], v=[], m_prop=[], u=[], tau=[]))
            ph["t"].append(sol.t); ph["r"].append(rr); ph["v"].append(vv); ph["m_prop"].append(mp); ph["u"].append(uu); ph["tau"].append(tt)
            y = sol.y[:, -1].copy()
            t_now = sol.t[-1]
            hit_ground = len(sol.t_events[0]) > 0
            hit_depleted = len(sol.t_events[1]) > 0
            hit_energy = len(sol.t_events[2]) > 0
            stopped = hit_ground or hit_energy
            if stopped or hit_depleted:
                break  # depleted -> go to next stage; ground/energy -> stop everything
        if stopped:
            break
        if k < ns - 1:
            y[6] = vehicle.stages[k + 1].propellant  # staging: new propellant state

    for name, ph in phases.items():
        for key in ("t", "r", "v", "m_prop", "u", "tau"):
            ph[key] = np.concatenate(ph[key])
    return phases


def final_gamma(phases):
    last = phases[sorted(phases)[-1]]
    r, v = last["r"][-1], last["v"][-1]
    return np.degrees(np.arcsin(np.dot(r, v) / (np.linalg.norm(r) * np.linalg.norm(v))))


def tune_kick(vehicle, lat_deg, lo=0.3, hi=85.0, **kw):
    """Find the pitch-kick angle for which the burnout flight-path angle is zero."""
    f = lambda k: final_gamma(gravity_turn_guess(vehicle, lat_deg, kick_deg=k, **kw))
    try:
        return brentq(f, lo, hi, xtol=1e-3)
    except ValueError:
        # no sign change: return the end with the smaller |gamma|
        return lo if abs(f(lo)) < abs(f(hi)) else hi
