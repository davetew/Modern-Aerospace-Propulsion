"""Matplotlib figures for lecture slides and reports.

Style: a restrained palette (categorical hues in a fixed, CVD-safe order; a
single-hue sequential ramp for magnitudes; recessive grid/axes).  All
functions return the matplotlib Figure so callers can save or tweak.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

from .bodies import AU_KM, get_body, MU_SUN
from .ephemeris import orbit_samples, body_state, jd_to_datetime, jd_to_iso
from .kepler import sample_orbit

# ---- palette ---------------------------------------------------------------
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
            "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"


def _mpl():
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "grid.color": GRID, "grid.linewidth": 0.6, "axes.grid": True,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
        "axes.prop_cycle": matplotlib.cycler(color=SERIES), "legend.frameon": False,
        "font.family": "sans-serif",
    })
    cmap = LinearSegmentedColormap.from_list("seqblue", SEQ_BLUE)
    return plt, cmap


# ---------------------------------------------------------------------------
def plot_porkchop(pc, quantity: str = "dv_total", y: str = "arrival", levels: Optional[Sequence[float]] = None,
                  c3_levels: Optional[Sequence[float]] = None, tof_levels: Optional[Sequence[float]] = None,
                  title: Optional[str] = None, mark_optimum: bool = True, ax=None):
    """Porkchop plot.  Filled contours of ``quantity`` ('dv_total', 'dv_dep',
    'c3', 'vinf_arr'), C3 contour lines (orange, if quantity != c3), and
    time-of-flight lines (gray, when y='arrival')."""
    import matplotlib.dates as mdates
    plt, cmap = _mpl()
    field = getattr(pc, quantity)
    labels = {"dv_total": "total $\\Delta v$ (departure + capture) [km/s]",
              "dv_dep": "departure $\\Delta v$ from parking orbit [km/s]",
              "c3": "launch energy $C_3$ [km$^2$/s$^2$]", "vinf_arr": "arrival $v_\\infty$ [km/s]"}
    X = np.array([jd_to_datetime(float(j)) for j in pc.jd_dep])
    if y == "arrival":
        Y = pc.jd_arr   # (n_tof, n_dep)
        Yd = np.vectorize(jd_to_datetime)(Y)
        Xg = np.broadcast_to(X[None, :], Y.shape)
        Xn = mdates.date2num(Xg)
        Yn = mdates.date2num(Yd)
    else:
        Xn = np.broadcast_to(mdates.date2num(X)[None, :], field.shape)
        Yn = np.broadcast_to(pc.tof_days[:, None], field.shape)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8.5, 6.2))
    else:
        fig = ax.figure
    fmin = np.nanmin(field)
    if levels is None:
        top = min(np.nanpercentile(field, 85), fmin * 2.5 if fmin > 0 else np.nanmax(field))
        levels = np.linspace(fmin, top, 14)
    cf = ax.contourf(Xn, Yn, np.clip(field, levels[0], levels[-1]), levels=levels, cmap=cmap, extend="max")
    cb = fig.colorbar(cf, ax=ax, pad=0.02, format="%.1f")
    cb.set_label(labels.get(quantity, quantity), color=INK2)
    cb.outline.set_edgecolor(AXIS)
    if quantity != "c3":
        if c3_levels is None:
            c3min = np.nanmin(pc.c3)
            c3_levels = [round(v, 0) for v in np.linspace(c3min * 1.05, min(np.nanpercentile(pc.c3, 70), c3min * 4), 6)]
        cs = ax.contour(Xn, Yn, pc.c3, levels=c3_levels, colors=SERIES[1], linewidths=0.9)
        ax.clabel(cs, fmt="C3=%.0f", fontsize=7.5, colors=SERIES[1])
    if y == "arrival":
        if tof_levels is None:
            tof_levels = np.arange(math.ceil(pc.tof_days[0] / 50) * 50, pc.tof_days[-1], 50)
        T = np.broadcast_to(pc.tof_days[:, None], field.shape)
        ct = ax.contour(Xn, Yn, T, levels=tof_levels, colors=MUTED, linewidths=0.6, linestyles="dashed")
        ax.clabel(ct, fmt="%.0f d", fontsize=7, colors=MUTED)
        ax.yaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.set_ylabel("arrival date")
    else:
        ax.set_ylabel("time of flight [days]")
    if mark_optimum:
        o = pc.optimum(quantity)
        i, j = pc._argmin(field)
        xo = Xn[i, j]
        yo = Yn[i, j]
        ax.plot(xo, yo, marker="o", ms=7, mfc="white", mec=INK, mew=1.5, zorder=5)
        ax.annotate(f"min {o['value']:.2f}\n{o['depart'][:10]}, {o['tof_days']:.0f} d", (xo, yo),
                    xytext=(10, 10), textcoords="offset points", fontsize=8, color=INK,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=AXIS, lw=0.6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.set_xlabel("departure date")
    for lab in ax.get_xticklabels():
        lab.set_rotation(30)
        lab.set_ha("right")
    ax.grid(False)
    ax.set_title(title or f"{pc.origin} → {pc.target}: {labels.get(quantity, quantity).split('[')[0].strip()}"
                 f"  (parking {pc.park_alt:.0f} km"
                 + (f", capture {pc.capture_peri_alt:.0f} km" if pc.capture_peri_alt is not None else "") + ")",
                 color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
def plot_transfer(legs, title: Optional[str] = None, show_orbits: Sequence[str] | None = None, ax=None,
                  units: str = "AU"):
    """Heliocentric plan view (ecliptic x-y) of one or more Lambert legs."""
    plt, _ = _mpl()
    if not isinstance(legs, (list, tuple)):
        legs = [legs]
    scale = AU_KM if units == "AU" else 1.0
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    else:
        fig = ax.figure
    bodies = []
    for leg in legs:
        for b in (leg.origin, leg.target):
            if b not in bodies:
                bodies.append(b)
    if show_orbits is not None:
        for b in show_orbits:
            if get_body(b).name not in bodies:
                bodies.append(get_body(b).name)
    # planet orbits (muted) and positions at the encounters (categorical)
    for k, b in enumerate(bodies):
        orb = orbit_samples(b, legs[0].jd_dep)
        ax.plot(orb[:, 0] / scale, orb[:, 1] / scale, color=AXIS, lw=0.8, zorder=1)
    color_of = {b: SERIES[k % len(SERIES)] for k, b in enumerate(bodies)}
    for n, leg in enumerate(legs):
        pts = sample_orbit(leg.r1, leg.lambert.v1, MU_SUN, leg.tof_days * 86400.0, 300)
        ax.plot(pts[:, 0] / scale, pts[:, 1] / scale, color=INK, lw=1.8, zorder=3,
                label=f"transfer {leg.origin}→{leg.target}, {leg.tof_days:.0f} d" if n == 0 else
                f"{leg.origin}→{leg.target}, {leg.tof_days:.0f} d")
        ax.plot(leg.r1[0] / scale, leg.r1[1] / scale, "o", color=color_of[leg.origin], ms=8, zorder=4)
        ax.plot(leg.r2[0] / scale, leg.r2[1] / scale, "o", color=color_of[leg.target], ms=8, zorder=4)
        ax.annotate(f"{leg.origin}\n{jd_to_iso(leg.jd_dep)[:10]}", (leg.r1[0] / scale, leg.r1[1] / scale),
                    xytext=(6, 6), textcoords="offset points", fontsize=8, color=INK2)
        ax.annotate(f"{leg.target}\n{jd_to_iso(leg.jd_arr)[:10]}", (leg.r2[0] / scale, leg.r2[1] / scale),
                    xytext=(6, -14), textcoords="offset points", fontsize=8, color=INK2)
        # where the origin body is at arrival and target at departure (faint)
        ro, _ = body_state(leg.origin, leg.jd_arr)
        rt, _ = body_state(leg.target, leg.jd_dep)
        ax.plot(ro[0] / scale, ro[1] / scale, "o", color=color_of[leg.origin], ms=5, alpha=0.35, zorder=2)
        ax.plot(rt[0] / scale, rt[1] / scale, "o", color=color_of[leg.target], ms=5, alpha=0.35, zorder=2)
    ax.plot(0, 0, "o", color=SERIES[3], ms=10, zorder=4)
    ax.set_aspect("equal")
    ax.set_xlabel(f"x [{units}] (J2000 ecliptic)")
    ax.set_ylabel(f"y [{units}]")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(title or "Heliocentric transfer geometry", loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
def plot_budget(budget, title: Optional[str] = None, ax=None):
    """Horizontal bars of the Delta-v legs with low/high bands; aero legs hatched."""
    plt, _ = _mpl()
    legs = budget.legs
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 0.5 * len(legs) + 1.6))
    else:
        fig = ax.figure
    ys = np.arange(len(legs))[::-1]
    for y, l in zip(ys, legs):
        color = {"propulsive": SERIES[0], "margin": MUTED, "aero": SERIES[2], "credit": SERIES[5]}.get(l.kind, SERIES[0])
        ax.barh(y, l.dv, color=color, height=0.62, hatch="//" if l.kind == "aero" else None,
                edgecolor=SURFACE if l.kind != "aero" else color, linewidth=1.0, alpha=0.95 if l.kind != "aero" else 0.5)
        if l.dv_low != l.dv or l.dv_high != l.dv:
            ax.plot([l.dv_low, l.dv_high], [y, y], color=INK, lw=1.4)
            ax.plot([l.dv_low, l.dv_low], [y - 0.15, y + 0.15], color=INK, lw=1.2)
            ax.plot([l.dv_high, l.dv_high], [y - 0.15, y + 0.15], color=INK, lw=1.2)
            ax.text(l.dv_high + 0.08, y, f"{l.dv:.2f} ({l.dv_low:.2f}–{l.dv_high:.2f})", va="center", fontsize=8.5, color=INK2)
        else:
            ax.text(l.dv + 0.08, y, f"{l.dv:.3f}", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(ys)
    ax.set_yticklabels([l.name for l in legs], fontsize=9)
    ax.set_xlabel("$\\Delta v$ [km/s]")
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    xmax = max(max(l.dv_high for l in legs), 1e-3)
    ax.set_xlim(0, xmax * 1.45)
    tot = budget.total_propulsive()
    ax.set_title(title or f"{budget.title}\npropulsive total {tot:.2f} km/s "
                 f"({budget.total_propulsive('low'):.2f}–{budget.total_propulsive('high'):.2f})"
                 + (f"\naero-assisted {budget.total_aero():.2f} km/s not charged" if budget.total_aero() else ""),
                 loc="left", fontsize=10.5, color=INK)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
def plot_staging_tradeoff(dv_total: float, isp: float, sigma: float, max_stages: int = 5, ax=None):
    """Payload fraction vs number of (identical) stages for a given total dv."""
    from .staging import design_vehicle
    plt, _ = _mpl()
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
    else:
        fig = ax.figure
    ns, pf = [], []
    for n in range(1, max_stages + 1):
        try:
            v = design_vehicle(dv_total, 1.0, [isp] * n, [sigma] * n)
            ns.append(n)
            pf.append(v.payload_fraction * 100)
        except ValueError:
            ns.append(n)
            pf.append(0.0)
    ax.bar(ns, pf, color=SERIES[0], width=0.6, edgecolor=SURFACE)
    for n, p in zip(ns, pf):
        ax.text(n, p + 0.05 * max(pf), f"{p:.2f}%" if p > 0 else "infeasible", ha="center", fontsize=8.5, color=INK2)
    ax.set_xlabel("number of stages (identical Isp, σ)")
    ax.set_ylabel("payload fraction [%]")
    ax.set_xticks(ns)
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
    ax.set_title(f"Δv = {dv_total:.2f} km/s, Isp = {isp:.0f} s, σ = {sigma:.2f}", loc="left", fontsize=10.5)
    fig.tight_layout()
    return fig


def plot_lunar_transfer(sol, ax=None):
    """Plan view of the BMW patched-conic lunar transfer in the Earth-Moon plane
    (Moon fixed at its arrival position; the geocentric arc is drawn in the
    inertial frame)."""
    from .bodies import EARTH, MOON
    from .lunar import D_MOON, R_SOI_MOON
    from .kepler import propagate
    plt, _ = _mpl()
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
    else:
        fig = ax.figure
    # geocentric arc: rotate so that the Earth-Moon line at arrival is the x axis
    # injection point is at angle (gamma1 - (nu1-nu0)) from that line
    ang_inj = sol.gamma1 - (sol.nu1 - sol.nu0)
    r0 = sol.r0 * np.array([math.cos(ang_inj), math.sin(ang_inj), 0.0])
    v0 = sol.v0 * np.array([math.sin(sol.phi0) * math.cos(ang_inj) - math.cos(sol.phi0) * math.sin(ang_inj),
                            math.sin(sol.phi0) * math.sin(ang_inj) + math.cos(sol.phi0) * math.cos(ang_inj), 0.0])
    pts = sample_orbit(r0, v0, EARTH.mu, sol.tof, 400)
    ax.plot(pts[:, 0] / 1e3, pts[:, 1] / 1e3, color=INK, lw=1.8,
            label=f"geocentric arc to SOI: {sol.tof/3600:.1f} h (+{sol.tof_soi/3600:.1f} h to perilune)")
    th = np.linspace(0, 2 * math.pi, 200)
    ax.plot(D_MOON * np.cos(th) / 1e3, D_MOON * np.sin(th) / 1e3, color=AXIS, lw=0.8)
    ax.plot((D_MOON + R_SOI_MOON * np.cos(th)) / 1e3, R_SOI_MOON * np.sin(th) / 1e3, color=SERIES[1], lw=0.8, ls="--",
            label="lunar SOI at arrival")
    ax.plot(0, 0, "o", color=SERIES[0], ms=9)
    ax.plot(D_MOON / 1e3, 0, "o", color=MUTED, ms=6, label="Moon at arrival")
    moon_inj = D_MOON * np.array([math.cos(-sol.tof * 2 * math.pi / MOON.rotation_period),
                                  math.sin(-sol.tof * 2 * math.pi / MOON.rotation_period)])
    ax.plot(moon_inj[0] / 1e3, moon_inj[1] / 1e3, "o", color=MUTED, ms=6, alpha=0.4, label="Moon at TLI")
    ax.plot(r0[0] / 1e3, r0[1] / 1e3, "o", color=SERIES[2], ms=6, label="TLI")
    ax.set_aspect("equal")
    ax.set_xlabel("x [10³ km]")
    ax.set_ylabel("y [10³ km]")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title(f"Patched-conic lunar transfer: v0={sol.v0:.3f} km/s, λ1={math.degrees(sol.lambda1):.1f}°, "
                 f"perilune {sol.perilune_alt:.0f} km ({'retro' if sol.h2 < 0 else 'pro'}grade)", loc="left", fontsize=10)
    fig.tight_layout()
    return fig
