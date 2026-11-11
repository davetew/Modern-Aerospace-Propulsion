"""Post-processing: time histories, 3-D path, loss breakdown, control profiles.

Reads a Dymos solution record (dymos_solution.db) and, optionally, the gravity-turn guess
saved by scripts/run_case.py (guess.npz) so the optimized trajectory can be drawn against the
prescribed gravity-turn guidance law on the same axes.
"""
import glob
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import openmdao.api as om

from .constants import R_E, MU_E, OMEGA_E, G0

plt.rcParams.update({"font.size": 11, "axes.labelsize": 11, "legend.fontsize": 9, "figure.dpi": 150})
OPT_C, GUESS_C, BAR2_C = "#C8102E", "#4D4D4D", "#2E5090"


def load_solution(run_dir):
    db = sorted(glob.glob(os.path.join(run_dir, "**", "dymos_solution.db"), recursive=True))
    if not db:
        raise FileNotFoundError(f"no dymos_solution.db under {run_dir}")
    case = om.CaseReader(db[0]).get_case("final")
    names = sorted({k.split(".")[1] for k in case.outputs.keys() if k.startswith("traj.") and ".timeseries." in k},
                   key=lambda n: (n != "rise", n))
    out = {}
    for n in names:
        ts = f"traj.{n}.timeseries."
        d = {}
        for k in case.outputs.keys():
            if k.startswith(ts):
                d[k[len(ts):]] = np.asarray(case.get_val(k))
        out[n] = d
    pay = float(case.get_val("traj.parameter_vals:m_pay")[0, 0]) if "traj.parameter_vals:m_pay" in case.outputs else None
    return out, pay


def load_guess(run_dir):
    f = os.path.join(run_dir, "guess.npz")
    if not os.path.exists(f):
        return None
    z = np.load(f)
    g = {}
    for k in z.files:
        if "__" in k:
            n, var = k.split("__")
            g.setdefault(n, {})[var] = z[k]
    g["kick_deg"] = float(z["kick_deg"])
    return g


def _concat(sol, var, fill=np.nan):
    """Concatenate a timeseries variable over phases; phases lacking it (e.g. tau in the
    vertical-rise phase, where it is a parameter) are filled with `fill`."""
    t = np.concatenate([sol[n]["time"][:, 0] for n in sol])
    parts = []
    width = max(sol[n][var].reshape(len(sol[n]["time"]), -1).shape[1] for n in sol if var in sol[n])
    for n in sol:
        if var in sol[n]:
            parts.append(sol[n][var].reshape(len(sol[n]["time"]), -1))
        else:
            parts.append(np.full((len(sol[n]["time"]), width), fill))
    return t, np.concatenate(parts)


def _guess_series(g):
    names = [n for n in sorted(g) if n != "kick_deg"]
    names.sort(key=lambda n: (n != "rise", n))
    t = np.concatenate([g[n]["t"] for n in names])
    r = np.concatenate([g[n]["r"] for n in names])
    v = np.concatenate([g[n]["v"] for n in names])
    mp = np.concatenate([g[n]["m_prop"] for n in names])
    return t, r, v, mp, names


def _site_frame(r0):
    up = r0 / np.linalg.norm(r0)
    east = np.cross([0, 0, 1.0], up); east /= np.linalg.norm(east)
    north = np.cross(up, east)
    return up, east, north


def plot_histories(sol, pay, guess=None, title="", fname="histories.png", vehicle=None):
    """Altitude / speed & components / propellant + throttle, q, accel."""
    t, r = _concat(sol, "r"); _, v = _concat(sol, "v"); _, mp = _concat(sol, "m_prop")
    _, h = _concat(sol, "h"); _, tau = _concat(sol, "tau", fill=1.0)
    _, q = _concat(sol, "q"); _, acc = _concat(sol, "accel"); _, gam = _concat(sol, "gamma")
    v_site = v - np.cross([0, 0, OMEGA_E], r)  # Earth-relative velocity
    # local (radial / east / north) components at each point
    up = r / np.linalg.norm(r, axis=1)[:, None]
    east = np.cross(np.array([0.0, 0.0, 1.0])[None, :], up); east /= np.linalg.norm(east, axis=1)[:, None]
    north = np.cross(up, east)
    v_up = np.sum(v_site * up, axis=1); v_east = np.sum(v_site * east, axis=1); v_north = np.sum(v_site * north, axis=1)
    fig, axs = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
    ax = axs[0, 0]; ax.plot(t, h[:, 0] / 1e3, color=OPT_C, lw=2, label="optimized")
    if guess is not None:
        gt, gr, gv, gmp, _ = _guess_series(guess)
        ax.plot(gt, (np.linalg.norm(gr, axis=1) - R_E) / 1e3, color=GUESS_C, ls="--", lw=1.5,
                label=f"prescribed gravity turn (kick {guess['kick_deg']:.1f}°)")
    ax.set_ylabel("Altitude [km]"); ax.grid(alpha=.3); ax.legend(); ax.set_title("Altitude")
    ax = axs[1, 0]
    ax.plot(t, np.linalg.norm(v_site, axis=1) / 1e3, color=OPT_C, lw=2, label="Earth-relative speed")
    ax.plot(t, v_up / 1e3, color="green", ls="--", label="radial")
    ax.plot(t, v_east / 1e3, color="blue", ls="--", label="east (azimuthal)")
    ax.plot(t, v_north / 1e3, color="cyan", ls="--", label="north (polar)")
    if guess is not None:
        gvs = gv - np.cross([0, 0, OMEGA_E], gr)
        ax.plot(gt, np.linalg.norm(gvs, axis=1) / 1e3, color=GUESS_C, ls="--", lw=1.5, label="gravity turn speed")
    ax.set_ylabel("Velocity [km/s]"); ax.grid(alpha=.3); ax.legend(ncol=2); ax.set_title("Earth-relative velocity, local radial/east/north components")
    ax = axs[2, 0]; ax.plot(t, mp[:, 0] / 1e3, color=OPT_C, lw=2)
    if guess is not None:
        ax.plot(gt, gmp / 1e3, color=GUESS_C, ls="--", lw=1.5)
    ax.set_ylabel("Stage propellant [t]"); ax.set_xlabel("Time [s]"); ax.grid(alpha=.3); ax.set_title("Propellant (per active stage)")
    ax = axs[0, 1]; ax.plot(t, tau[:, 0], color=OPT_C, lw=2); ax.set_ylim(0, 1.05); ax.set_ylabel("Throttle τ [-]"); ax.grid(alpha=.3); ax.set_title("Throttle (optimized control)")
    ax = axs[1, 1]; ax.plot(t, q[:, 0] / 1e3, color=OPT_C, lw=2, label="q"); ax.set_ylabel("Dynamic pressure [kPa]"); ax.grid(alpha=.3)
    ax2 = ax.twinx(); ax2.plot(t, acc[:, 0] / G0, color="tab:orange", lw=1.5, label="T/m"); ax2.set_ylabel("Thrust acceleration [g]")
    ax.set_title("Path constraints: max-q and acceleration"); ax.legend(loc="upper left"); ax2.legend(loc="upper right")
    ax = axs[2, 1]; ax.plot(t, np.degrees(gam[:, 0]), color=OPT_C, lw=2, label="flight-path angle γ (inertial)")
    _, alpha = _concat(sol, "alpha"); ax.plot(t, np.degrees(alpha[:, 0]), color="tab:purple", lw=1.2, label="thrust–velocity angle α (air-relative)")
    ax.set_ylabel("[deg]"); ax.set_xlabel("Time [s]"); ax.grid(alpha=.3); ax.legend(); ax.set_title("Steering")
    for n in list(sol)[1:]:
        ts = sol[n]["time"][0, 0]
        for a in axs.flat:
            a.axvline(ts, color="k", lw=0.6, alpha=0.4)
    sup = title + (f"   payload = {pay/1e3:.1f} t" if pay is not None else "")
    fig.suptitle(sup, fontsize=13); fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(fname); plt.close(fig)
    return fname


def plot_3d(sol, guess=None, fname="path3d.png", title="3-D trajectory (ECI)"):
    """Left: whole-globe ECI view. Right: downrange-altitude profile with staging."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    t, r = _concat(sol, "r")
    fig = plt.figure(figsize=(14, 6.5))
    ax = fig.add_subplot(121, projection="3d")
    u_, v_ = np.mgrid[0:2 * np.pi:80j, 0:np.pi:40j]
    ax.plot_surface(R_E * np.cos(u_) * np.sin(v_), R_E * np.sin(u_) * np.sin(v_), R_E * np.cos(v_), color="royalblue", alpha=0.18, linewidth=0)
    for lon in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        ph = np.linspace(0, np.pi, 100)
        ax.plot(R_E * np.sin(ph) * np.cos(lon), R_E * np.sin(ph) * np.sin(lon), R_E * np.cos(ph), "k:", lw=0.4, alpha=0.6)
    for lat in np.radians([-60, -30, 0, 30, 60]):
        th = np.linspace(0, 2 * np.pi, 100)
        ax.plot(R_E * np.cos(lat) * np.cos(th), R_E * np.cos(lat) * np.sin(th), R_E * np.sin(lat) * np.ones_like(th), "k:", lw=0.4, alpha=0.6)
    if guess is not None:
        gt, gr, gv, gmp, _ = _guess_series(guess)
        ax.plot(gr[:, 0], gr[:, 1], gr[:, 2], color=GUESS_C, ls="--", lw=1.5, label="prescribed gravity turn")
    ax.plot(r[:, 0], r[:, 1], r[:, 2], color=OPT_C, lw=2.5, label="optimized")
    lim = 1.08 * R_E
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    lon0 = np.degrees(np.arctan2(r[0, 1], r[0, 0])); lat0 = np.degrees(np.arcsin(r[0, 2] / np.linalg.norm(r[0])))
    ax.view_init(elev=lat0 + 10, azim=lon0 + 25)
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m]"); ax.set_title("ECI path", fontsize=11); ax.legend(loc="upper left", fontsize=9)
    ax.set_box_aspect([1, 1, 1])

    ax2 = fig.add_subplot(122)
    r0 = r[0]
    def downrange(rr):
        c = np.clip(np.sum(rr * r0, axis=1) / (np.linalg.norm(rr, axis=1) * np.linalg.norm(r0)), -1, 1)
        return R_E * np.arccos(c) / 1e3
    _, h = _concat(sol, "h")
    ax2.plot(downrange(r), h[:, 0] / 1e3, color=OPT_C, lw=2.5, label="optimized")
    if guess is not None:
        ax2.plot(downrange(gr), (np.linalg.norm(gr, axis=1) - R_E) / 1e3, color=GUESS_C, ls="--", lw=1.5, label="prescribed gravity turn")
    for n in list(sol)[1:]:
        rs = sol[n]["r"][0]; hs = sol[n]["h"][0, 0]
        ax2.plot(downrange(rs[None, :]), hs / 1e3, "ko", ms=5)
        ax2.annotate(f"{n} start\nt={sol[n]['time'][0,0]:.0f} s, h={hs/1e3:.0f} km", (downrange(rs[None, :])[0], hs / 1e3),
                     textcoords="offset points", xytext=(8, -18), fontsize=8)
    ax2.set_xlabel("Downrange (great-circle) [km]"); ax2.set_ylabel("Altitude [km]"); ax2.grid(alpha=.3)
    ax2.set_title("Ascent profile", fontsize=11); ax2.legend(fontsize=9); ax2.set_ylim(bottom=0)
    fig.suptitle(title, fontsize=12); fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(fname); plt.close(fig)
    return fname


def plot_losses(sol, fname="losses.png", title="Δv budget closure"):
    last = sol[list(sol)[-1]]; first = sol[list(sol)[0]]
    L = {k: float(last[k][-1, 0]) for k in ("dv_vac", "L_grav", "L_drag", "L_steer", "L_bp")}
    v0 = float(first["vmag"][0, 0]); vf = float(last["vmag"][-1, 0])
    labels = ["Δv_ideal (vacuum)", "orbital speed gained\n|v_f| − |v_0|", "+ gravity loss", "+ steering loss", "+ back-pressure loss", "+ drag loss"]
    vals = [L["dv_vac"], vf - v0, L["L_grav"], L["L_steer"], L["L_bp"], L["L_drag"]]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    cols = [OPT_C, BAR2_C, "#7A7A7A", "#9A9A9A", "#BABABA", "#DADADA"]
    bottoms = [0, 0, vals[1], vals[1] + vals[2], vals[1] + vals[2] + vals[3], vals[1] + vals[2] + vals[3] + vals[4]]
    for i, (lab, val, b, c) in enumerate(zip(labels, vals, bottoms, cols)):
        ax.bar(i, val, bottom=b, color=c, edgecolor="k", lw=0.5)
        ax.text(i, b + val + 60, f"{val:,.0f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=8.5); ax.set_ylabel("m/s")
    resid = L["dv_vac"] - (vf - v0) - L["L_grav"] - L["L_steer"] - L["L_bp"] - L["L_drag"]
    ax.set_title(f"{title}\nΔv_ideal = |v_f| − |v_0| + Σ losses ;  v_0 = {v0:.0f} m/s (Earth rotation) ; residual {resid:+.3f} m/s", fontsize=10)
    ax.grid(axis="y", alpha=.3); fig.tight_layout(); fig.savefig(fname); plt.close(fig)
    return fname, L, v0, vf


def plot_sweep(results, fname="sweep.png"):
    """results: list of dicts with keys tag, payload_t, dry_masses_t, reserves_t."""
    import textwrap
    fig, ax = plt.subplots(figsize=(11, 5.2))
    tags = [textwrap.fill(r["tag"] or "", 20) for r in results]; pays = [r["payload_t"] for r in results]
    cols = [OPT_C if "baseline" in (t or "") else BAR2_C for t in tags]
    ax.bar(range(len(tags)), pays, color=cols, edgecolor="k", lw=0.5)
    for i, (p_, r) in enumerate(zip(pays, results)):
        ax.text(i, p_ + 3, f"{p_:.0f} t", ha="center", fontsize=9)
    ax.axhline(100, color="k", ls="--", lw=1); ax.text(len(tags) - 0.5, 103, "SpaceX: 100 t reusable", ha="right", fontsize=8)
    ax.set_xticks(range(len(tags))); ax.set_xticklabels(tags, fontsize=8, rotation=0)
    ax.set_ylabel("Max payload to 275 km × 26° [t]")
    ax.set_title("Starship Block 3 payload vs. dry mass and return-propellant reserves (Raptor 3: 275 tf vac, Isp 350/380 s; q ≤ 35 kPa, a ≤ 4 g)", fontsize=10)
    ax.grid(axis="y", alpha=.3); fig.tight_layout(); fig.savefig(fname); plt.close(fig)
    return fname
