#!/usr/bin/env python
"""Make the demo figures for one or more run directories.

    python scripts/make_figures.py runs/starship --title "Starship Block 3, 275 km x 26 deg"
    python scripts/make_figures.py --sweep runs/starship runs/sw_* --out runs/figs
"""
import argparse, glob, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from ascentopt.post import load_solution, load_guess, plot_histories, plot_3d, plot_losses, plot_sweep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--title", default="")
    ap.add_argument("--out", default=None)
    ap.add_argument("--sweep", action="store_true")
    a = ap.parse_args()
    out = a.out or a.runs[0]
    os.makedirs(out, exist_ok=True)
    if a.sweep:
        res = []
        for rd in a.runs:
            f = os.path.join(rd, "result.json")
            if os.path.exists(f):
                res.append(json.load(open(f)))
        res.sort(key=lambda r: -r["payload_t"])
        print(plot_sweep(res, os.path.join(out, "sweep_payload.png")))
        for r in res:
            print(f"  {r.get('tag'):28s} payload {r['payload_t']:7.1f} t  dry {r['dry_masses_t']}  reserves {r['reserves_t']}  {r['exit']}")
        return
    rd = a.runs[0]
    sol, pay = load_solution(rd)
    guess = load_guess(rd)
    print(plot_histories(sol, pay, guess, title=a.title, fname=os.path.join(out, "histories.png")))
    print(plot_3d(sol, guess, fname=os.path.join(out, "path3d.png"), title=a.title + " — 3-D path (ECI)"))
    f, L, v0, vf = plot_losses(sol, fname=os.path.join(out, "losses.png"))
    print(f); print("losses:", {k: round(x) for k, x in L.items()}, "v0", round(v0), "vf", round(vf))


if __name__ == "__main__":
    main()
