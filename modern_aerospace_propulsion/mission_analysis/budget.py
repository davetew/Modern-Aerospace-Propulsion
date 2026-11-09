"""Delta-v budgets and end-to-end mission composition.

A ``DeltaVBudget`` is an ordered list of legs (launch, injection, flybys,
capture, plane changes, landing, margins ...).  Each leg may carry a low/high
band (e.g., launch losses) and the Isp of the stage that performs it, so the
budget can be turned straight into propellant masses (design-roadmap steps
1-4: mission -> dv -> propellant fraction -> required Isp -> staging).

Reference values (for quick budgets; cite before using in a report):
  Apollo LM powered descent ~2.07 km/s (incl. ~100 s hover), ascent ~1.86 km/s
  (Apollo 11 Mission Report, NASA SP-238 tables); Mars EDL is aero-dominated,
  propulsive terminal descent ~0.5-0.7 km/s for MSL-class landers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

from .bodies import get_body, EARTH, G0
from .staging import mass_ratio_for_dv, required_isp, propellant_fraction_for_dv


@dataclass
class DeltaVLeg:
    name: str
    dv: float                      # km/s, nominal
    dv_low: Optional[float] = None
    dv_high: Optional[float] = None
    kind: str = "propulsive"       # propulsive | aero | credit | margin | reference
    isp: Optional[float] = None    # s, stage performing the burn (for propellant sizing)
    note: str = ""

    def __post_init__(self):
        if self.dv_low is None:
            self.dv_low = self.dv
        if self.dv_high is None:
            self.dv_high = self.dv


@dataclass
class DeltaVBudget:
    title: str
    legs: list[DeltaVLeg] = field(default_factory=list)

    def add(self, name: str, dv: float, **kw) -> "DeltaVLeg":
        leg = DeltaVLeg(name, dv, **kw)
        self.legs.append(leg)
        return leg

    def add_margin(self, fraction: float = 0.05, name: str = "margin") -> DeltaVLeg:
        base = self.total_propulsive()
        return self.add(name, fraction * base, kind="margin", note=f"{fraction:.0%} of propulsive dv")

    # ---- totals --------------------------------------------------------
    def total_propulsive(self, which: str = "nominal") -> float:
        key = {"nominal": "dv", "low": "dv_low", "high": "dv_high"}[which]
        return sum(getattr(l, key) for l in self.legs if l.kind in ("propulsive", "margin"))

    def total_aero(self) -> float:
        return sum(l.dv for l in self.legs if l.kind == "aero")

    def post_launch_propulsive(self) -> float:
        return sum(l.dv for l in self.legs if l.kind in ("propulsive", "margin") and "launch" not in l.name.lower())

    # ---- propellant sizing ------------------------------------------------
    def propellant_fraction(self, isp: float, exclude_launch: bool = True) -> float:
        dv = self.post_launch_propulsive() if exclude_launch else self.total_propulsive()
        return propellant_fraction_for_dv(isp, dv)

    def required_isp(self, propellant_fraction: float, exclude_launch: bool = True) -> float:
        dv = self.post_launch_propulsive() if exclude_launch else self.total_propulsive()
        return required_isp(dv, propellant_fraction)

    def size_propellant(self, final_mass: float, default_isp: float, exclude_launch: bool = True) -> list[dict]:
        """Backward propellant sizing through the propulsive legs (single stage
        unless legs carry their own Isp)."""
        rows = []
        m_after = final_mass
        legs = [l for l in self.legs if l.kind in ("propulsive", "margin")]
        if exclude_launch:
            legs = [l for l in legs if "launch" not in l.name.lower()]
        for l in reversed(legs):
            isp = l.isp or default_isp
            m_before = m_after * mass_ratio_for_dv(isp, l.dv)
            rows.append(dict(name=l.name, dv=l.dv, isp=isp, m_before=m_before, m_after=m_after, m_prop=m_before - m_after))
            m_after = m_before
        rows.reverse()
        return rows

    # ---- presentation -----------------------------------------------------
    def table(self, markdown: bool = False) -> str:
        if markdown:
            lines = [f"**{self.title}**", "", "| leg | dv [km/s] | band [km/s] | type | note |", "|---|---:|:---:|---|---|"]
            for l in self.legs:
                band = "" if (l.dv_low == l.dv and l.dv_high == l.dv) else f"{l.dv_low:.3f}-{l.dv_high:.3f}"
                lines.append(f"| {l.name} | {l.dv:.3f} | {band} | {l.kind} | {l.note} |")
            lines.append(f"| **total propulsive** | **{self.total_propulsive():.3f}** | "
                         f"{self.total_propulsive('low'):.3f}-{self.total_propulsive('high'):.3f} | | |")
            if self.total_aero():
                lines.append(f"| aero-assisted (not propulsive) | {self.total_aero():.3f} | | aero | |")
            return "\n".join(lines)
        w = max(len(l.name) for l in self.legs) if self.legs else 10
        lines = [self.title, "-" * len(self.title),
                 f"{'leg':{w}s} {'dv[km/s]':>9s} {'band':>14s}  type        note"]
        for l in self.legs:
            band = "" if (l.dv_low == l.dv and l.dv_high == l.dv) else f"{l.dv_low:.3f}-{l.dv_high:.3f}"
            lines.append(f"{l.name:{w}s} {l.dv:9.3f} {band:>14s}  {l.kind:10s}  {l.note}")
        lines.append(f"{'TOTAL propulsive':{w}s} {self.total_propulsive():9.3f} "
                     f"{self.total_propulsive('low'):.3f}-{self.total_propulsive('high'):.3f}")
        if self.total_aero():
            lines.append(f"{'aero-assisted':{w}s} {self.total_aero():9.3f}  (replaced by atmosphere, not charged)")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return dict(title=self.title,
                    legs=[dict(name=l.name, dv=l.dv, dv_low=l.dv_low, dv_high=l.dv_high, kind=l.kind,
                               isp=l.isp, note=l.note) for l in self.legs],
                    total_propulsive=self.total_propulsive(),
                    total_propulsive_low=self.total_propulsive("low"),
                    total_propulsive_high=self.total_propulsive("high"),
                    total_aero=self.total_aero())


# ---------------------------------------------------------------------------
# Mission archetypes
# ---------------------------------------------------------------------------
APOLLO_LM_DESCENT_DV = 2.07   # km/s, includes hover/redesignation
APOLLO_LM_ASCENT_DV = 1.86    # km/s


def lunar_landing_dv_estimate(orbit_alt: float = 100.0, hover_time_s: float = 60.0,
                              gravity_loss_factor: float = 0.15) -> float:
    """Powered descent from a circular lunar orbit: orbital speed + gravity/steering
    losses (fraction of orbital speed) + hover at 1 lunar g."""
    from .bodies import MOON
    from .kepler import circular_velocity
    v = circular_velocity(MOON.mu, MOON.radius + orbit_alt)
    g_m = MOON.surface_gravity()
    return v * (1.0 + gravity_loss_factor) + g_m * hover_time_s


def interplanetary_budget(origin, target, depart, arrive, park_alt: float = 300.0, launch_lat_deg: float = 28.5,
                          capture_peri_alt: Optional[float] = 300.0, capture_apo_alt: Optional[float] = None,
                          aerocapture: bool = False, include_launch: bool = True, margin: float = 0.05,
                          flybys: Sequence[str] = (), flyby_epochs: Sequence = (),
                          landing_dv: Optional[float] = None) -> tuple[DeltaVBudget, object]:
    """Compose a Delta-v budget for an interplanetary mission on given dates.

    Returns (budget, InterplanetaryMission) so the detailed leg results are available.
    """
    from .patched_conics import interplanetary_mission
    from .launch import launch_to_orbit
    from .aerobraking import aerocapture_credit
    bodies = [origin, *flybys, target]
    epochs = [depart, *flyby_epochs, arrive]
    m = interplanetary_mission(bodies, epochs, park_alt, capture_peri_alt, capture_apo_alt, aerocapture=aerocapture)
    b = DeltaVBudget(f"{get_body(origin).name} -> {' -> '.join(get_body(f).name for f in flybys) + ' -> ' if flybys else ''}"
                     f"{get_body(target).name} ({str(depart)[:10]} -> {str(arrive)[:10]})")
    if include_launch:
        L = launch_to_orbit(park_alt, None, launch_lat_deg, origin)
        b.add(f"launch to {park_alt:.0f} km LEO", L.dv_mid, dv_low=L.dv_low, dv_high=L.dv_high,
              note=f"ideal {L.dv_ideal:.2f} + losses {L.losses.low:.2f}-{L.losses.high:.2f}")
    b.add(f"{get_body(origin).name} departure injection", m.departure.dv,
          note=f"C3={m.departure.c3:.2f} km2/s2, v_inf={m.departure.vinf:.3f} km/s")
    for f in m.flybys:
        b.add(f"powered flyby at {f.body}", f.dv_powered,
              note=f"turn {math.degrees(f.turn_angle):.1f} deg, periapsis alt {f.alt:.0f} km"
                   + ("" if f.feasible else " INFEASIBLE"))
    if m.capture is not None:
        if aerocapture:
            ac = aerocapture_credit(target, m.capture.vinf, capture_peri_alt, capture_apo_alt)
            b.add(f"aerocapture at {get_body(target).name}", ac.dv_saved, kind="aero",
                  note=f"entry {ac.entry_speed:.2f} km/s; replaces propulsive capture")
            b.add("periapsis raise after aerocapture", ac.dv_periapsis_raise)
        else:
            b.add(f"capture at {get_body(target).name}", m.capture.dv,
                  note=f"v_inf={m.capture.vinf:.3f} km/s into {m.capture.rp-get_body(target).radius:.0f}"
                       + (f"x{m.capture.ra-get_body(target).radius:.0f}" if m.capture.ra != m.capture.rp else "") + " km")
    if landing_dv is not None:
        b.add(f"landing at {get_body(target).name}", landing_dv, note="user-supplied")
    if margin:
        b.add_margin(margin)
    return b, m


def lunar_budget(park_alt: float = 185.0, lunar_orbit_alt: float = 100.0, launch_lat_deg: float = 28.5,
                 tof_hours: Optional[float] = None, include_launch: bool = True, land: bool = False,
                 ascend: bool = False, return_to_earth: bool = False, margin: float = 0.05) -> tuple[DeltaVBudget, object]:
    """Earth -> lunar orbit [-> surface -> orbit -> Earth] budget.  Uses the
    patched-conic transfer (retrograde lunar approach, Apollo-like) if a
    time of flight is given, otherwise the Hohmann estimate."""
    from .lunar import lunar_hohmann_estimate, design_lunar_transfer, lunar_patched_conic
    from .launch import launch_to_orbit
    from .bodies import MOON
    b = DeltaVBudget(f"Earth ({park_alt:.0f} km) -> Moon ({lunar_orbit_alt:.0f} km orbit)")
    if include_launch:
        L = launch_to_orbit(park_alt, None, launch_lat_deg, EARTH)
        b.add(f"launch to {park_alt:.0f} km LEO", L.dv_mid, dv_low=L.dv_low, dv_high=L.dv_high,
              note=f"ideal {L.dv_ideal:.2f} + losses {L.losses.low:.2f}-{L.losses.high:.2f}")
    detail: object
    if tof_hours is None:
        est = lunar_hohmann_estimate(park_alt, lunar_orbit_alt)
        b.add("trans-lunar injection", est.dv_tli, note=f"Hohmann to lunar distance, TOF {est.tof/3600:.1f} h")
        b.add("lunar orbit insertion", est.loi.dv, note=f"v_inf {est.vinf_moon:.3f} km/s, circular {lunar_orbit_alt:.0f} km")
        detail = est
    else:
        # find v0 giving the requested TOF (scan), then lambda1 for the perilune
        from scipy.optimize import brentq
        r0 = EARTH.radius + park_alt
        target_tof = tof_hours * 3600.0

        def tof_err(v0):
            sols = design_lunar_transfer(park_alt, lunar_orbit_alt, v0=v0, direction="retrograde")
            if not sols:
                return math.nan
            return sols[0].tof_total - target_tof
        v_lo = lunar_hohmann_estimate(park_alt, lunar_orbit_alt).v_tli + 0.003
        v_hi = math.sqrt(2 * EARTH.mu / r0) * 0.995
        vs = [v_lo + (v_hi - v_lo) * k / 60 for k in range(61)]
        errs = [tof_err(v) for v in vs]
        root = None
        for k in range(60):
            if not (math.isnan(errs[k]) or math.isnan(errs[k + 1])) and errs[k] * errs[k + 1] < 0:
                root = brentq(tof_err, vs[k], vs[k + 1], xtol=1e-7)
                break
        if root is None:
            raise ValueError("no patched-conic transfer found for that time of flight")
        pc = design_lunar_transfer(park_alt, lunar_orbit_alt, v0=root, direction="retrograde")[0]
        b.add("trans-lunar injection", pc.dv_tli,
              note=f"patched conic, TLI->perilune {pc.tof_total/3600:.1f} h, lambda1={math.degrees(pc.lambda1):.1f} deg")
        b.add("lunar orbit insertion", pc.loi_dv(), note=f"v_inf {pc.vinf:.3f} km/s, circular {lunar_orbit_alt:.0f} km")
        detail = pc
    if land:
        b.add("powered descent + landing", APOLLO_LM_DESCENT_DV, kind="propulsive", note="Apollo LM reference value")
    if ascend:
        b.add("ascent to lunar orbit", APOLLO_LM_ASCENT_DV, note="Apollo LM reference value")
    if return_to_earth:
        est = lunar_hohmann_estimate(park_alt, lunar_orbit_alt)
        b.add("trans-Earth injection", est.loi.dv, note="mirror of LOI (Hohmann estimate)")
        b.add("Earth entry (aero)", est.dv_tli, kind="aero", note="direct entry, no propulsive capture")
    if margin:
        b.add_margin(margin)
    return b, detail
