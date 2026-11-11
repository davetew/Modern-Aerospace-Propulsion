"""Figures for the technical report and Beamer deck (vector PDF + PNG)."""
import os, sys, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import openmdao.api as om

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from ascentopt.post import load_solution, load_guess, plot_histories, plot_3d, plot_losses, plot_sweep, _concat, _guess_series
from ascentopt.atmosphere import ATM
from ascentopt.vehicle import load_vehicle
from ascentopt.constants import R_E, MU_E, OMEGA_E, G0, P_SL

FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
DATA = os.path.join(HERE, "data")


def _prepare_data():
    """Lay out the recorded solutions from ../results as <case>/dymos_solution.db etc. (what
    post.load_solution expects).  Idempotent; run from the repo without any manual setup."""
    import shutil
    res = os.path.join(os.path.dirname(HERE), "results")
    layout = {"starship16": {"starship16_dymos_solution.db": "dymos_solution.db",
                             "starship16_dymos_simulation.db": "dymos_simulation.db",
                             "starship16_guess.npz": "guess.npz"},
              "sw_reserves": {"sw_reserves_dymos_solution.db": "dymos_solution.db",
                              "sw_reserves_guess.npz": "guess.npz"}}
    for case, files in layout.items():
        d = os.path.join(DATA, case); os.makedirs(d, exist_ok=True)
        for src, dst in files.items():
            if not os.path.exists(os.path.join(d, dst)):
                shutil.copy(os.path.join(res, src), os.path.join(d, dst))
    for f in glob.glob(os.path.join(res, "result_*.json")):
        if not os.path.exists(os.path.join(DATA, os.path.basename(f))):
            shutil.copy(f, DATA)


_prepare_data()
plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "legend.fontsize": 8.5, "figure.dpi": 150,
                     "axes.titlesize": 10.5, "pdf.fonttype": 42})
# Columbia palette (beamercolorthemeColumbia.sty): PMS7635C magenta, PMS280C dark blue, PMS284C light blue,
# PMS2461C turquoise, PMS2612C purple
RED, BLUE, GREY, ORANGE, GREEN, PURPLE = "#D4285B", "#012169", "#5A5A5A", "#E07B00", "#259B9A", "#7B2A8D"
LIGHTBLUE = "#6CACE4"
import ascentopt.post as _post
_post.OPT_C, _post.GUESS_C, _post.BAR2_C = RED, "#5A5A5A", BLUE


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------------------ data
sol, pay = load_solution(os.path.join(DATA, "starship16"))
guess = load_guess(os.path.join(DATA, "starship16"))
sol_r, pay_r = load_solution(os.path.join(DATA, "sw_reserves"))
sim = om.CaseReader(os.path.join(DATA, "starship16", "dymos_simulation.db")).get_case("final")
veh = load_vehicle(os.path.join(os.path.dirname(HERE), "vehicles", "starship_v3_baseline.yaml"))
results = {os.path.basename(f)[7:-5]: json.load(open(f)) for f in glob.glob(os.path.join(DATA, "result_*.json"))}

# ------------------------------------------------------------------ 1. atmosphere
h = np.linspace(0, 300e3, 1200)
fig, axs = plt.subplots(1, 3, figsize=(10, 3.2))
axs[0].plot(ATM.rho(h), h / 1e3, color=BLUE); axs[0].set_xlabel("density ρ [kg/m³]")
axs[1].plot(ATM.p(h), h / 1e3, color=BLUE); axs[1].set_xlabel("pressure p [Pa]")
axs[2].plot(ATM.a(h), h / 1e3, color=BLUE); axs[2].set_xlabel("speed of sound a [m/s]")
for ax in axs:
    ax.set_ylim(0, 300); ax.grid(alpha=.3)
    ax.axhline(60.4, color=RED, lw=0.8, ls="--"); ax.axhline(275, color=GREEN, lw=0.8, ls="--")
    ax.axhline(86, color=GREY, lw=0.6, ls=":")
axs[0].set_ylabel("altitude [km]")
axs[2].text(305, 64, "staging (60 km)", color=RED, fontsize=7.5); axs[2].text(305, 279, "insertion (275 km)", color=GREEN, fontsize=7.5)
axs[2].text(305, 90, "86 km: analytic layers → tabulated US76", color=GREY, fontsize=7)
for ax in axs[:2]:
    ax.set_xscale("log")
fig.suptitle("US Standard Atmosphere 1976 as implemented (smooth log-interpolant; jax-differentiable)", fontsize=10)
fig.tight_layout(); save(fig, "fig_atmosphere")

# ------------------------------------------------------------------ 2. engine model
hh = np.linspace(0, 100e3, 400)
pa = ATM.p(hh)
fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
for es, col, lab in ((veh.stages[0].engines[0].engine, RED, "Raptor 3 (SL nozzle, $A_e$ = %.2f m²)" % veh.stages[0].engines[0].engine.A_e),
                     (veh.stages[1].engines[1].engine, BLUE, "Raptor Vacuum ($A_e$ = %.2f m²)" % veh.stages[1].engines[1].engine.A_e)):
    T = es.thrust(pa) / 1e6; isp = es.thrust(pa) / (es.mdot_max * G0)
    axs[0].plot(hh / 1e3, T, color=col, label=lab); axs[1].plot(hh / 1e3, isp, color=col, label=lab)
axs[0].set_ylabel("thrust per engine [MN]"); axs[1].set_ylabel("$I_{sp}$ [s]")
for ax in axs:
    ax.set_xlabel("altitude [km]"); ax.grid(alpha=.3); ax.legend(loc="lower right")
axs[0].set_title(r"$T(h)=\tau\,(T_{vac}-p_a(h)\,A_e)$"); axs[1].set_title(r"$I_{sp}(h)=T(h)/(\dot m\,g_0)$,  $\dot m = T_{vac}/(I_{sp,vac}\,g_0)$")
fig.suptitle("Engine model: one anchor ($T_{vac}$, $I_{sp,vac}$) plus exit area; sea-level values follow from back-pressure", fontsize=10)
fig.tight_layout(); save(fig, "fig_engine_model")

# ------------------------------------------------------------------ 3. histories / 3d / losses / sweep (vector versions)
title = "Starship Block 3 → 275 km × 26° (Starbase), expendable trajectory, dry 200/100 t"
plot_histories(sol, pay, guess, title=title, fname=os.path.join(FIG, "fig_histories.pdf"))
plot_histories(sol, pay, guess, title=title, fname=os.path.join(FIG, "fig_histories.png"))
plot_3d(sol, guess, fname=os.path.join(FIG, "fig_path3d.pdf"), title=title)
plot_3d(sol, guess, fname=os.path.join(FIG, "fig_path3d.png"), title=title)
plot_losses(sol, fname=os.path.join(FIG, "fig_losses.pdf"))
plot_losses(sol, fname=os.path.join(FIG, "fig_losses.png"))
order = ["sw_light", "starship16", "sw_heavy", "sw_shipdry", "sw_reserves", "sw_heavy_res", "sw_res15"]
plot_sweep([results[k] for k in order], fname=os.path.join(FIG, "fig_sweep.pdf"))
plot_sweep([results[k] for k in order], fname=os.path.join(FIG, "fig_sweep.png"))
print("wrote histories/path3d/losses/sweep")

# ------------------------------------------------------------------ 4. thrust / Isp / mass / Mach vs time
t, thr = _concat(sol, "thrust"); _, isp = _concat(sol, "isp"); _, m = _concat(sol, "m"); _, mach = _concat(sol, "mach"); _, h_ = _concat(sol, "h")
fig, axs = plt.subplots(2, 2, figsize=(10, 5.6), sharex=True)
axs[0, 0].plot(t, thr[:, 0] / 1e6, color=RED); axs[0, 0].set_ylabel("total thrust [MN]"); axs[0, 0].set_title("Thrust (33 → 6 engines; throttle and back-pressure)")
axs[0, 1].plot(t, isp[:, 0], color=RED); axs[0, 1].set_ylabel("effective $I_{sp}$ [s]"); axs[0, 1].set_title("Effective $I_{sp}$ of the active stage")
axs[1, 0].plot(t, m[:, 0] / 1e3, color=RED); axs[1, 0].set_ylabel("vehicle mass [t]"); axs[1, 0].set_title("Mass (payload 293.3 t carried throughout)")
axs[1, 1].plot(t, mach[:, 0], color=RED); axs[1, 1].set_ylabel("Mach (air-relative)"); axs[1, 1].set_title("Mach number")
for n in list(sol)[1:]:
    ts = sol[n]["time"][0, 0]
    for a in axs.flat:
        a.axvline(ts, color="k", lw=0.6, alpha=0.4)
for a in axs.flat:
    a.grid(alpha=.3)
for a in axs[1]:
    a.set_xlabel("time [s]")
fig.tight_layout(); save(fig, "fig_thrust_isp_mass")

# ------------------------------------------------------------------ 5. verification: collocation vs explicit simulation
names = list(sol)
tc = np.concatenate([sol[n]["time"][:, 0] for n in names]); hc = np.concatenate([sol[n]["h"][:, 0] for n in names]); vc = np.concatenate([sol[n]["vmag"][:, 0] for n in names]); mc = np.concatenate([sol[n]["m"][:, 0] for n in names])
ts_ = np.concatenate([sim.get_val(f"traj.{n}.timeseries.time")[:, 0] for n in names]); hs = np.concatenate([sim.get_val(f"traj.{n}.timeseries.h")[:, 0] for n in names]); vs = np.concatenate([sim.get_val(f"traj.{n}.timeseries.vmag")[:, 0] for n in names]); ms = np.concatenate([sim.get_val(f"traj.{n}.timeseries.m")[:, 0] for n in names])
fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.2))
axs[0].plot(tc, hc / 1e3, color=RED, lw=2, label="collocation solution"); axs[0].plot(ts_, hs / 1e3, "k--", lw=1, label="explicit RK re-integration"); axs[0].set_ylabel("altitude [km]")
axs[1].plot(tc, vc / 1e3, color=RED, lw=2); axs[1].plot(ts_, vs / 1e3, "k--", lw=1); axs[1].set_ylabel("inertial speed [km/s]")
hi = np.interp(tc, ts_, hs); vi = np.interp(tc, ts_, vs)
axs[2].plot(tc, hi - hc, color=BLUE, label="Δh [m]"); axs[2].set_ylabel("Δh [m]  (sim − colloc)", color=BLUE)
ax2 = axs[2].twinx(); ax2.plot(tc, vi - vc, color=ORANGE, label="Δv [m/s]"); ax2.set_ylabel("Δ|v| [m/s]", color=ORANGE)
axs[0].legend(fontsize=8); axs[2].set_title("difference (final: +211 m, +4.5 m/s)")
for a in axs:
    a.set_xlabel("time [s]"); a.grid(alpha=.3)
fig.suptitle("Verification V1 — optimized controls re-integrated with an explicit RK45 integrator (Dymos simulate), 16 Radau segments / phase", fontsize=10)
fig.tight_layout(); save(fig, "fig_verification")

# ------------------------------------------------------------------ 6. sensitivity (linear model from the sweep)
# payload vs stage dry masses: fit PL = a + b1*m1 + b2*m2 on the four no-reserve cases
pts = [(200, 100, results["starship16"]["payload_t"]), (160, 85, results["sw_light"]["payload_t"]),
       (250, 120, results["sw_heavy"]["payload_t"]), (200, 150, results["sw_shipdry"]["payload_t"])]
A = np.array([[1, p[0], p[1]] for p in pts]); y = np.array([p[2] for p in pts])
coef, *_ = np.linalg.lstsq(A, y, rcond=None)
pred = A @ coef
print("linear fit PL = %.1f %+.3f*m_dry1 %+.3f*m_dry2 ; max resid %.2f t" % (coef[0], coef[1], coef[2], np.max(np.abs(pred - y))))
fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
m2 = np.linspace(80, 160, 50)
for m1, col in ((160, GREEN), (200, RED), (250, BLUE)):
    axs[0].plot(m2, coef[0] + coef[1] * m1 + coef[2] * m2, color=col, lw=1.2, label=f"booster dry {m1} t (linear fit)")
for p, lab in zip(pts, ("baseline", "light", "heavy", "ship 150 t")):
    axs[0].plot(p[1], p[2], "ko", ms=5); axs[0].annotate(f"{lab}\n{p[2]:.0f} t", (p[1], p[2]), textcoords="offset points", xytext=(6, 4), fontsize=7.5)
axs[0].set_xlabel("ship dry mass [t]"); axs[0].set_ylabel("max payload [t]"); axs[0].grid(alpha=.3); axs[0].legend(fontsize=8)
axs[0].set_title(f"∂PL/∂m_dry: ship {coef[2]:+.2f} t/t, booster {coef[1]:+.2f} t/t (fit residual < {np.max(np.abs(pred-y)):.1f} t)")
res_pts = [(0, 0, results["starship16"]["payload_t"]), (292, 30, results["sw_reserves"]["payload_t"]), (548, 30, results["sw_res15"]["payload_t"])]
axs[1].plot([p[0] for p in res_pts], [p[2] for p in res_pts], "o-", color=RED, lw=1.5)
for p in res_pts:
    axs[1].annotate(f"{p[2]:.0f} t\n(ship reserve {p[1]} t)", (p[0], p[2]), textcoords="offset points", xytext=(8, -4), fontsize=7.5)
axs[1].set_xlabel("booster propellant reserved for boost-back / landing [t]"); axs[1].set_ylabel("max payload [t]"); axs[1].grid(alpha=.3)
axs[1].set_title("Reserves: −0.28 t/t then −0.16 t/t (staging moves earlier)")
fig.suptitle("Sensitivities extracted from the sweep (dry 200/100 t baseline, 275 km × 26°)", fontsize=10)
fig.tight_layout(); save(fig, "fig_sensitivity")

# ------------------------------------------------------------------ 7. baseline vs 8 % reserve trajectories
fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.3))
for s_, col, lab in ((sol, RED, "baseline, no reserves (293 t)"), (sol_r, BLUE, "booster 292 t + ship 30 t reserved (205 t)")):
    t_, h__ = _concat(s_, "h"); _, v__ = _concat(s_, "vmag"); _, tau__ = _concat(s_, "tau", fill=1.0); _, gam__ = _concat(s_, "gamma")
    axs[0].plot(t_, h__[:, 0] / 1e3, color=col, label=lab)
    axs[1].plot(t_, v__[:, 0] / 1e3, color=col)
    axs[2].plot(t_, tau__[:, 0], color=col)
    for n in list(s_)[2:]:
        axs[0].plot(s_[n]["time"][0, 0], s_[n]["h"][0, 0] / 1e3, "o", color=col, ms=5)
        axs[1].plot(s_[n]["time"][0, 0], s_[n]["vmag"][0, 0] / 1e3, "o", color=col, ms=5)
axs[0].set_ylabel("altitude [km]"); axs[1].set_ylabel("inertial speed [km/s]"); axs[2].set_ylabel("throttle τ"); axs[2].set_ylim(0, 1.05)
axs[0].legend(fontsize=7.5, loc="upper left"); axs[0].set_title("Staging point moves: 145 s/60 km → 133 s/53 km")
axs[1].set_title("Staging speed 2.61 → 2.29 km/s"); axs[2].set_title("Throttle: same max-q bucket, different ship burn")
for a in axs:
    a.set_xlabel("time [s]"); a.grid(alpha=.3)
fig.suptitle("Effect of return-propellant reserves on the optimal ascent (markers = staging)", fontsize=10)
fig.tight_layout(); save(fig, "fig_reserve_compare")

# ------------------------------------------------------------------ 8. tsiolkovsky vs optimizer comparison data (for a table) + control vector
_, u = _concat(sol, "u", fill=np.nan); _, alpha = _concat(sol, "alpha"); _, r = _concat(sol, "r"); _, v = _concat(sol, "v")
# pitch angle of thrust relative to local horizontal, and thrust azimuth deviation from east
up = r / np.linalg.norm(r, axis=1)[:, None]
east = np.cross(np.array([0, 0, 1.0])[None, :], up); east /= np.linalg.norm(east, axis=1)[:, None]
north = np.cross(up, east)
uu = u / np.maximum(np.linalg.norm(u, axis=1), 1e-9)[:, None]
pitch = np.degrees(np.arcsin(np.clip(np.sum(uu * up, axis=1), -1, 1)))
yaw = np.degrees(np.arctan2(np.sum(uu * north, axis=1), np.sum(uu * east, axis=1)))
fig, axs = plt.subplots(1, 2, figsize=(10, 3.3))
ok = ~np.isnan(u[:, 0])
axs[0].plot(t[ok], pitch[ok], color=RED, label="thrust pitch above local horizontal"); axs[0].plot(t, np.degrees(_concat(sol, "gamma")[1][:, 0]), color=GREY, ls="--", label="flight-path angle γ (inertial)")
axs[0].set_ylabel("[deg]"); axs[0].legend(fontsize=8); axs[0].set_title("Optimized pitch program vs flight-path angle")
hvec = np.cross(r, v); hhat = hvec / np.linalg.norm(hvec, axis=1)[:, None]
oop = np.degrees(np.arcsin(np.clip(np.sum(uu * hhat, axis=1), -1, 1)))
axs[1].plot(t[ok], oop[ok], color=BLUE, label="thrust out of the osculating orbit plane")
axs[1].plot(t[ok], yaw[ok], color=GREY, ls="--", lw=1, label="thrust azimuth from local east (in-plane drift along the great circle)")
axs[1].set_ylabel("[deg]"); axs[1].legend(fontsize=7.5, loc="lower left"); axs[1].set_title("Out-of-plane steering ≈ 0 → planar problem; azimuth drifts with the ground track")
for a in axs:
    a.set_xlabel("time [s]"); a.grid(alpha=.3)
    for n in list(sol)[1:]:
        a.axvline(sol[n]["time"][0, 0], color="k", lw=0.6, alpha=0.4)
fig.tight_layout(); save(fig, "fig_steering")

# dump key numbers for the LaTeX tables
r0 = results["starship16"]
tab = dict(payload=pay / 1e3, t_final=r0["t_final_s"], losses=r0["losses_ms"], staging=dict(t=r0["stage0_t_end_s"], h=r0["stage0_h_end_km"], v=r0["stage0_v_end_ms"]),
           q_max=r0["q_max_kPa"], a_max=r0["a_max_g"], fit=dict(a=coef[0], b1=coef[1], b2=coef[2]),
           cases={k: dict(tag=results[k]["tag"], payload=results[k]["payload_t"], t_stage=results[k]["stage0_t_end_s"], h_stage=results[k]["stage0_h_end_km"],
                          v_stage=results[k]["stage0_v_end_ms"], grav=results[k]["losses_ms"]["grav"], steer=results[k]["losses_ms"]["steer"], iters=results[k]["iterations"]) for k in order})
json.dump(tab, open(os.path.join(HERE, "numbers.json"), "w"), indent=1)
print(json.dumps(tab["staging"]), tab["q_max"], tab["a_max"])


# ------------------------------------------------------------------ 9. split history figures for slides
def split_histories():
    from ascentopt.post import OPT_C, GUESS_C
    t, r = _concat(sol, "r"); _, v = _concat(sol, "v"); _, mp = _concat(sol, "m_prop"); _, hh_ = _concat(sol, "h")
    _, tau = _concat(sol, "tau", fill=1.0); _, q = _concat(sol, "q"); _, acc = _concat(sol, "accel"); _, gam = _concat(sol, "gamma"); _, al = _concat(sol, "alpha")
    v_site = v - np.cross([0, 0, OMEGA_E], r)
    up = r / np.linalg.norm(r, axis=1)[:, None]; east = np.cross(np.array([0.0, 0.0, 1.0])[None, :], up); east /= np.linalg.norm(east, axis=1)[:, None]; north = np.cross(up, east)
    gt, gr, gv, gmp, _ = _guess_series(guess); gvs = gv - np.cross([0, 0, OMEGA_E], gr)
    # left: slide-33 layout
    fig, axs = plt.subplots(3, 1, figsize=(7.2, 7.6), sharex=True)
    axs[0].plot(t, hh_[:, 0] / 1e3, color=OPT_C, lw=2, label="optimized"); axs[0].plot(gt, (np.linalg.norm(gr, axis=1) - R_E) / 1e3, color=GUESS_C, ls="--", lw=1.5, label=f"gravity turn (slide-33 law, kick {guess['kick_deg']:.1f}°)")
    axs[0].set_ylabel("altitude [km]"); axs[0].legend(); axs[0].set_title("Altitude")
    axs[1].plot(t, np.linalg.norm(v_site, axis=1) / 1e3, color=OPT_C, lw=2, label="Earth-relative speed")
    axs[1].plot(t, np.sum(v_site * up, axis=1) / 1e3, color="green", ls="--", label="radial"); axs[1].plot(t, np.sum(v_site * east, axis=1) / 1e3, color="blue", ls="--", label="east"); axs[1].plot(t, np.sum(v_site * north, axis=1) / 1e3, color="cyan", ls="--", label="north")
    axs[1].plot(gt, np.linalg.norm(gvs, axis=1) / 1e3, color=GUESS_C, ls="--", lw=1.5, label="gravity turn speed"); axs[1].set_ylabel("velocity [km/s]"); axs[1].legend(ncol=2); axs[1].set_title("Earth-relative velocity, local components")
    axs[2].plot(t, mp[:, 0] / 1e3, color=OPT_C, lw=2); axs[2].plot(gt, gmp / 1e3, color=GUESS_C, ls="--", lw=1.5); axs[2].set_ylabel("stage propellant [t]"); axs[2].set_xlabel("time [s]"); axs[2].set_title("Propellant of the active stage")
    for n in list(sol)[1:]:
        for a in axs: a.axvline(sol[n]["time"][0, 0], color="k", lw=0.6, alpha=0.4)
    for a in axs: a.grid(alpha=.3)
    fig.tight_layout(); save(fig, "fig_hist_left")
    # right: controls and constraints
    fig, axs = plt.subplots(3, 1, figsize=(7.2, 7.6), sharex=True)
    axs[0].plot(t, tau[:, 0], color=OPT_C, lw=2); axs[0].set_ylim(0, 1.05); axs[0].set_ylabel("throttle τ"); axs[0].set_title("Throttle (optimized control): max-q bucket, then 4 g limit, then 40 % floor")
    axs[1].plot(t, q[:, 0] / 1e3, color=OPT_C, lw=2, label="q"); axs[1].set_ylabel("dynamic pressure [kPa]"); ax2 = axs[1].twinx(); ax2.plot(t, acc[:, 0] / G0, color="tab:orange", lw=1.5, label="T/m"); ax2.set_ylabel("thrust acceleration [g]")
    axs[1].axhline(35, color=OPT_C, lw=0.7, ls=":"); ax2.axhline(4, color="tab:orange", lw=0.7, ls=":"); axs[1].legend(loc="upper left"); ax2.legend(loc="upper right"); axs[1].set_title("Path constraints: q ≤ 35 kPa and T/m ≤ 4 g (both active)")
    axs[2].plot(t, np.degrees(gam[:, 0]), color=OPT_C, lw=2, label="flight-path angle γ (inertial)"); axs[2].plot(t, np.degrees(al[:, 0]), color="tab:purple", lw=1.2, label="thrust–velocity angle α (air-relative)"); axs[2].set_ylabel("[deg]"); axs[2].set_xlabel("time [s]"); axs[2].legend(); axs[2].set_title("Steering")
    for n in list(sol)[1:]:
        for a in axs: a.axvline(sol[n]["time"][0, 0], color="k", lw=0.6, alpha=0.4)
    for a in axs: a.grid(alpha=.3)
    fig.tight_layout(); save(fig, "fig_hist_right")

split_histories()
