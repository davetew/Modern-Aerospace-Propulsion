#!/usr/bin/env python
"""Run (or resume) an ascent optimization case.

    python scripts/run_case.py --vehicle vehicles/starship_v3_baseline.yaml --out runs/starship \
        --h-target 275e3 --lat 25.997 --q-max 35e3 --a-max 39.2 --segments 15 --max-iter 120 --restart

Re-running with --restart warm-starts from the previous dymos_solution.db in the output folder
(useful when the job must be split into time-limited chunks).  --simulate re-integrates the
optimized controls explicitly (Dymos simulate) and reports the terminal mismatch.
"""
import argparse, glob, os, sys, time, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import openmdao.api as om
from ascentopt import load_vehicle
from ascentopt.problem import build_problem, run, Mission, Options
from ascentopt.constants import G0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vehicle", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--h-target", type=float, default=275e3)
    ap.add_argument("--lat", type=float, default=25.997)
    ap.add_argument("--inc", type=float, default=None)
    ap.add_argument("--q-max", type=float, default=None)
    ap.add_argument("--a-max", type=float, default=None, help="m/s^2")
    ap.add_argument("--t-rise", type=float, default=10.0)
    ap.add_argument("--fixed-throttle", action="store_true")
    ap.add_argument("--objective", default="max_payload")
    ap.add_argument("--segments", type=int, default=15)
    ap.add_argument("--order", type=int, default=3)
    ap.add_argument("--optimizer", default="IPOPT")
    ap.add_argument("--max-iter", type=int, default=150)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--kick", type=float, default=None)
    ap.add_argument("--restart", action="store_true")
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--refine", type=int, default=0)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VAL",
                    help="override deck values, e.g. stage0.dry_mass=250e3 stage1.reserve=30e3 payload=100e3")
    ap.add_argument("--warm-from", default=None, help="dymos_solution.db of another case to warm-start from")
    ap.add_argument("--tag", default=None, help="case tag used in result.json")
    a = ap.parse_args()

    vpath = os.path.abspath(a.vehicle)
    if a.warm_from:
        a.warm_from = os.path.abspath(a.warm_from)
        if not os.path.exists(a.warm_from):
            sys.exit(f"--warm-from file not found: {a.warm_from}")
    os.makedirs(a.out, exist_ok=True)
    os.chdir(a.out)
    v = load_vehicle(vpath)
    for kv in a.set:
        key, val = kv.split("=")
        val = float(val)
        if key.startswith("stage"):
            idx, attr = key.split(".")
            setattr(v.stages[int(idx[5:])], attr, val)
        else:
            setattr(v, key, val)
        print(f"override {key} = {val}")
    print(v.summary())
    mis = Mission(h_target=a.h_target, lat_deg=a.lat, inc_deg=a.inc, q_max=a.q_max, a_max=a.a_max, t_rise=a.t_rise,
                  free_throttle=not a.fixed_throttle, objective=a.objective)
    opts = Options(num_segments=a.segments, order=a.order, optimizer=a.optimizer, max_iter=a.max_iter, tol=a.tol,
                   simulate=a.simulate, kick_deg=a.kick, verbose=False, refine_iteration_limit=a.refine)
    t0 = time.time()
    p = build_problem(v, mis, opts)
    outdir = str(p.get_outputs_dir())
    prev = os.path.join(outdir, "dymos_solution.db")
    if a.restart and os.path.exists(prev):
        opts.restart = prev
        print(f"warm start from {prev}")
    elif a.warm_from and os.path.exists(a.warm_from):
        opts.restart = os.path.abspath(a.warm_from)
        print(f"warm start from {opts.restart}")
    print(f"kick guess {p._ascentopt_guess['kick_deg']:.2f} deg; setup {time.time()-t0:.1f}s; outputs -> {outdir}")

    gz = {f"{n}__{k}": arr for n, ph in p._ascentopt_guess["phases"].items() for k, arr in ph.items()}
    np.savez("guess.npz", kick_deg=p._ascentopt_guess["kick_deg"], **gz)
    t0 = time.time()
    run(p, opts)
    wall = time.time() - t0

    # ---- report ----
    ipopt_out = os.path.join(outdir, "IPOPT.out")
    exit_line, n_iter = "n/a", "n/a"
    if os.path.exists(ipopt_out):
        for line in open(ipopt_out):
            if line.startswith("EXIT:"):
                exit_line = line.strip()
            if line.startswith("Number of Iterations"):
                n_iter = line.split(":")[-1].strip()
    names = ["rise"] + [f"stage{k}" for k in range(len(v.stages))]
    last = names[-1]
    g = lambda ph, n: p.get_val(f"traj.{ph}.timeseries.{n}")
    v0 = g("rise", "vmag")[0, 0]
    res = dict(tag=a.tag, vehicle=v.name, exit=exit_line, iterations=n_iter, wall_s=round(wall, 1),
               overrides=a.set, dry_masses_t=[s.dry_mass / 1e3 for s in v.stages], reserves_t=[s.reserve / 1e3 for s in v.stages],
               payload_t=float(p.get_val("traj.parameter_vals:m_pay")[0, 0] / 1e3),
               t_final_s=float(g(last, "time")[-1, 0]), h_final_km=float(g(last, "h")[-1, 0] / 1e3),
               v_final_ms=float(g(last, "vmag")[-1, 0]), gamma_final_deg=float(np.degrees(g(last, "gamma")[-1, 0])),
               inc_final_deg=float(np.degrees(g(last, "inc")[-1, 0])),
               q_max_kPa=float(max(g(n, "q").max() for n in names) / 1e3),
               a_max_g=float(max(g(n, "accel").max() for n in names) / G0),
               losses_ms=dict(dv_vac=float(g(last, "dv_vac")[-1, 0]), grav=float(g(last, "L_grav")[-1, 0]),
                              drag=float(g(last, "L_drag")[-1, 0]), steer=float(g(last, "L_steer")[-1, 0]),
                              bp=float(g(last, "L_bp")[-1, 0]), v_gain=float(g(last, "vmag")[-1, 0] - v0)))
    L = res["losses_ms"]
    res["identity_residual_ms"] = L["dv_vac"] - L["v_gain"] - L["grav"] - L["drag"] - L["steer"] - L["bp"]
    for k, n in enumerate(names[1:]):
        res[f"{n}_t_start_s"] = float(g(n, "time")[0, 0]); res[f"{n}_t_end_s"] = float(g(n, "time")[-1, 0])
        res[f"{n}_h_end_km"] = float(g(n, "h")[-1, 0] / 1e3); res[f"{n}_v_end_ms"] = float(g(n, "vmag")[-1, 0])
        res[f"{n}_mprop_end_t"] = float(g(n, "m_prop")[-1, 0] / 1e3)
        res[f"{n}_tau_min"] = float(g(n, "tau").min()) if not a.fixed_throttle else 1.0
    if a.simulate:
        simdb = glob.glob(os.path.join(outdir, "**", "dymos_simulation.db"), recursive=True)
        if simdb:
            sim = om.CaseReader(simdb[0]).get_case("final")
            res["simulate_check"] = {}
            for n in ("h", "vmag", "gamma", "m"):
                ac = g(last, n)[-1, 0]; sm = sim.get_val(f"traj.{last}.timeseries.{n}")[-1, 0]
                res["simulate_check"][n] = dict(collocation=float(ac), simulate=float(sm), diff=float(sm - ac))
    print(json.dumps(res, indent=1))
    with open("result.json", "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
