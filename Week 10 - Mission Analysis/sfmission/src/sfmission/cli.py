"""Command-line interface.

  sfmission hohmann Earth Mars [--park 300 --capture 300]
  sfmission leg Earth Mars 2026-11-20 2027-08-15 [--park 300 --capture 300 --plot out.png]
  sfmission porkchop Earth Mars 2026-09-01 2027-03-01 --tof 120 420 [--plot pork.png --quantity c3]
  sfmission windows Earth Mars 2026-01-01 --years 10
  sfmission mission Earth Mars Earth Jupiter --epochs 2024-10-14 2025-03-01 2026-12-03 2030-04-11
  sfmission lunar [--park 185 --perilune 100 --v0 10.95 | --lambda1 60 | --tof 72]
  sfmission launch [--alt 300 --inc 51.6 --lat 28.5]
  sfmission stage 9.3 10000 --isp 300 450 --sigma 0.08 0.10
  sfmission budget Earth Mars 2026-11-20 2027-08-15 [--aerocapture --plot budget.png]
"""
from __future__ import annotations

import argparse
import json
import math
import sys


def main(argv=None):
    p = argparse.ArgumentParser(prog="sfmission", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("hohmann"); s.add_argument("origin"); s.add_argument("target")
    s.add_argument("--park", type=float, default=300); s.add_argument("--capture", type=float, default=300)

    s = sub.add_parser("leg"); s.add_argument("origin"); s.add_argument("target"); s.add_argument("depart"); s.add_argument("arrive")
    s.add_argument("--park", type=float, default=300); s.add_argument("--capture", type=float, default=300); s.add_argument("--plot")

    s = sub.add_parser("porkchop"); s.add_argument("origin"); s.add_argument("target"); s.add_argument("start"); s.add_argument("end")
    s.add_argument("--tof", type=float, nargs=2, required=True); s.add_argument("--park", type=float, default=300)
    s.add_argument("--capture", type=float, default=300); s.add_argument("--n", type=int, default=80)
    s.add_argument("--plot"); s.add_argument("--quantity", default="dv_total"); s.add_argument("--y", default="arrival")

    s = sub.add_parser("windows"); s.add_argument("origin"); s.add_argument("target"); s.add_argument("start")
    s.add_argument("--years", type=float, default=10); s.add_argument("--park", type=float, default=300); s.add_argument("--capture", type=float, default=300)

    s = sub.add_parser("mission"); s.add_argument("bodies", nargs="+"); s.add_argument("--epochs", nargs="+", required=True)
    s.add_argument("--park", type=float, default=300); s.add_argument("--capture", type=float, default=None)
    s.add_argument("--flyby-min-alt", type=float, default=300); s.add_argument("--plot")

    s = sub.add_parser("lunar"); s.add_argument("--park", type=float, default=185); s.add_argument("--perilune", type=float, default=100)
    s.add_argument("--v0", type=float); s.add_argument("--lambda1", type=float); s.add_argument("--tof", type=float, help="hours")
    s.add_argument("--direction", default="either"); s.add_argument("--plot")

    s = sub.add_parser("launch"); s.add_argument("--alt", type=float, default=300); s.add_argument("--inc", type=float)
    s.add_argument("--lat", type=float, default=28.5); s.add_argument("--body", default="Earth")

    s = sub.add_parser("stage"); s.add_argument("dv", type=float); s.add_argument("payload", type=float)
    s.add_argument("--isp", type=float, nargs="+", required=True); s.add_argument("--sigma", type=float, nargs="+", required=True)
    s.add_argument("--split", type=float, nargs="+")

    s = sub.add_parser("budget"); s.add_argument("origin"); s.add_argument("target"); s.add_argument("depart"); s.add_argument("arrive")
    s.add_argument("--park", type=float, default=300); s.add_argument("--lat", type=float, default=28.5)
    s.add_argument("--capture", type=float, default=300); s.add_argument("--capture-apo", type=float)
    s.add_argument("--aerocapture", action="store_true"); s.add_argument("--no-launch", action="store_true")
    s.add_argument("--margin", type=float, default=0.05); s.add_argument("--isp", type=float); s.add_argument("--final-mass", type=float)
    s.add_argument("--plot"); s.add_argument("--json", action="store_true")

    sub.add_parser("mcp", help="run the MCP server (stdio)")

    a = p.parse_args(argv)
    from . import (hohmann_interplanetary, heliocentric_leg, departure_dv, capture_dv, porkchop, next_launch_windows,
                   interplanetary_mission, lunar_hohmann_estimate, design_lunar_transfer, launch_to_orbit, design_vehicle,
                   interplanetary_budget, lunar_budget)

    if a.cmd == "hohmann":
        print(hohmann_interplanetary(a.origin, a.target, a.park, a.capture).summary())
    elif a.cmd == "leg":
        leg = heliocentric_leg(a.origin, a.target, a.depart, a.arrive)
        print(leg.summary())
        print(departure_dv(a.origin, leg.vinf_dep_mag, a.park).summary())
        if a.capture is not None:
            print(capture_dv(a.target, leg.vinf_arr_mag, a.capture).summary())
        if a.plot:
            from .plotting import plot_transfer
            plot_transfer(leg).savefig(a.plot, dpi=180); print("saved", a.plot)
    elif a.cmd == "porkchop":
        pc = porkchop(a.origin, a.target, a.start, a.end, a.tof[0], a.tof[1], a.n, a.n, a.park, a.capture)
        print(pc.summary())
        if a.plot:
            from .plotting import plot_porkchop
            plot_porkchop(pc, quantity=a.quantity, y=a.y).savefig(a.plot, dpi=180); print("saved", a.plot)
    elif a.cmd == "windows":
        for w in next_launch_windows(a.origin, a.target, a.start, a.years, a.park, a.capture):
            print(f"{w['depart']}  type {w['type']:2s} TOF {w['tof_days']:6.1f} d  C3 {w['c3']:6.2f}  v_inf,arr {w['vinf_arr']:.3f}  dv_total {w['dv_total']:.3f} km/s")
    elif a.cmd == "mission":
        m = interplanetary_mission(a.bodies, a.epochs, a.park, a.capture, None, a.flyby_min_alt)
        print(m.summary())
        if a.plot:
            from .plotting import plot_transfer
            plot_transfer(m.legs, title=" -> ".join(m.bodies)).savefig(a.plot, dpi=180); print("saved", a.plot)
    elif a.cmd == "lunar":
        print(lunar_hohmann_estimate(a.park, a.perilune).summary())
        sols = []
        if a.tof is not None:
            b, pc = lunar_budget(a.park, a.perilune, tof_hours=a.tof, include_launch=False, margin=0)
            sols = [pc]
        elif a.v0 is not None or a.lambda1 is not None:
            sols = design_lunar_transfer(a.park, a.perilune, v0=a.v0, lambda1_deg=a.lambda1, direction=a.direction)
        for s_ in sols:
            print(); print(s_.summary())
        if a.plot and sols:
            from .plotting import plot_lunar_transfer
            plot_lunar_transfer(sols[0]).savefig(a.plot, dpi=180); print("saved", a.plot)
    elif a.cmd == "launch":
        print(launch_to_orbit(a.alt, a.inc, a.lat, a.body).summary())
    elif a.cmd == "stage":
        print(design_vehicle(a.dv, a.payload, a.isp, a.sigma, split=a.split).table())
    elif a.cmd == "budget":
        b, m = interplanetary_budget(a.origin, a.target, a.depart, a.arrive, a.park, a.lat, a.capture, a.capture_apo,
                                     a.aerocapture, not a.no_launch, a.margin)
        print(m.summary()); print()
        print(json.dumps(b.to_dict(), indent=1) if a.json else b.table())
        if a.isp and a.final_mass:
            print(); print(f"propellant sizing (Isp {a.isp:.0f} s, final mass {a.final_mass:.0f} kg):")
            for r in b.size_propellant(a.final_mass, a.isp):
                print(f"  {r['name']:40s} dv {r['dv']:.3f}  m_before {r['m_before']:10.1f}  m_prop {r['m_prop']:9.1f}")
        if a.plot:
            from .plotting import plot_budget
            plot_budget(b).savefig(a.plot, dpi=180); print("saved", a.plot)
    elif a.cmd == "mcp":
        from .mcp_server import main as mcp_main
        mcp_main()


if __name__ == "__main__":
    main()
