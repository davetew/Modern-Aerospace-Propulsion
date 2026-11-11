"""Rocket equation, propellant fractions and multistage sizing.

Nomenclature (Sutton; chosen to avoid the symbol collisions flagged in the
module review -- epsilon is reserved for nozzle area ratio):

  MR    = m_f / m_0                 mass ratio (final/initial)      [Sutton]
  zeta  = m_p / m_0                 propellant mass fraction        [Sutton]
  sigma = m_s / (m_s + m_p)         stage structural fraction (inert fraction)
  lambda= m_L / (m_s + m_p)         payload ratio of a stage
  r_k   = m_0,k / m_f,k = exp(dv_k / (g0 Isp_k))   stage mass ratio

For N stages with structural fractions sigma_k and effective exhaust speeds
c_k = g0 Isp_k, the initial-to-payload mass ratio is

  m_0 / m_L = prod_k  r_k (1 - sigma_k) / (1 - r_k sigma_k),

and the dv split minimizing m_0 for a given total dv satisfies
  r_k = (lam c_k - 1) / (lam c_k sigma_k)  with  sum_k c_k ln r_k = dv_total,
solved for the Lagrange multiplier lam (closed form; equal c and sigma give the
familiar equal-dv-per-stage result).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from scipy.optimize import brentq

from .bodies import G0

G0_MS = G0 * 1e3   # m/s^2


# ---------------------------------------------------------------------------
# Single-stage relations
# ---------------------------------------------------------------------------
def exhaust_velocity(isp: float) -> float:
    """c = g0 Isp [km/s] for Isp in seconds."""
    return G0 * isp


def tsiolkovsky_dv(isp: float, m0: float, mf: float) -> float:
    """dv = g0 Isp ln(m0/mf) [km/s]."""
    return exhaust_velocity(isp) * math.log(m0 / mf)


def mass_ratio_for_dv(isp: float, dv: float) -> float:
    """m0/mf = exp(dv / (g0 Isp))  (dv in km/s)."""
    return math.exp(dv / exhaust_velocity(isp))


def propellant_fraction_for_dv(isp: float, dv: float) -> float:
    """zeta = m_p/m_0 = 1 - exp(-dv/c)."""
    return 1.0 - math.exp(-dv / exhaust_velocity(isp))


def required_isp(dv: float, propellant_fraction: float) -> float:
    """Isp [s] needed to deliver dv [km/s] with propellant fraction zeta
    (design-roadmap step 3: mission dv + achievable zeta -> required Isp)."""
    if not 0 < propellant_fraction < 1:
        raise ValueError("propellant fraction must be in (0, 1)")
    return dv / (G0 * math.log(1.0 / (1.0 - propellant_fraction)))


def propellant_mass_for_dv(isp: float, dv: float, m_final: float) -> float:
    """Propellant needed to give dv to a vehicle whose mass *after* the burn is m_final."""
    return m_final * (mass_ratio_for_dv(isp, dv) - 1.0)


# ---------------------------------------------------------------------------
# Multistage vehicles
# ---------------------------------------------------------------------------
@dataclass
class Stage:
    name: str
    isp: float                 # s
    structural_fraction: float # sigma = m_s/(m_s+m_p)
    dv: float = 0.0            # km/s assigned to this stage
    # filled by sizing
    m_prop: float = 0.0
    m_struct: float = 0.0
    m_payload_above: float = 0.0

    @property
    def c(self) -> float:
        return exhaust_velocity(self.isp)

    @property
    def mass_ratio(self) -> float:
        return mass_ratio_for_dv(self.isp, self.dv)

    @property
    def m_stage(self) -> float:
        return self.m_prop + self.m_struct

    @property
    def m0(self) -> float:
        return self.m_stage + self.m_payload_above

    @property
    def mf(self) -> float:
        return self.m_struct + self.m_payload_above

    @property
    def payload_ratio(self) -> float:
        return self.m_payload_above / self.m_stage if self.m_stage else math.nan

    @property
    def max_dv(self) -> float:
        """dv if this stage carried zero payload: c ln(1/sigma)."""
        return self.c * math.log(1.0 / self.structural_fraction)


@dataclass
class Vehicle:
    stages: list[Stage]          # stage 1 = first to burn
    payload: float               # kg (or any mass unit)

    @property
    def m0(self) -> float:
        return self.stages[0].m0 if self.stages else self.payload

    @property
    def dv_total(self) -> float:
        return sum(s.dv for s in self.stages)

    @property
    def payload_fraction(self) -> float:
        return self.payload / self.m0

    def table(self) -> str:
        lines = [f"{'stage':10s} {'Isp[s]':>7s} {'sigma':>6s} {'dv[km/s]':>9s} {'r=m0/mf':>8s} "
                 f"{'m_prop':>12s} {'m_struct':>12s} {'m_0':>12s} {'lambda':>8s}"]
        for s in self.stages:
            lines.append(f"{s.name:10s} {s.isp:7.1f} {s.structural_fraction:6.3f} {s.dv:9.4f} {s.mass_ratio:8.4f} "
                         f"{s.m_prop:12.1f} {s.m_struct:12.1f} {s.m0:12.1f} {s.payload_ratio:8.4f}")
        lines.append(f"payload {self.payload:.1f}  |  gross m0 = {self.m0:.1f}  |  payload fraction = "
                     f"{self.payload_fraction:.4%}  |  total dv = {self.dv_total:.4f} km/s")
        return "\n".join(lines)


def size_vehicle(stages: Sequence[Stage], payload: float) -> Vehicle:
    """Size stage masses from the bottom of the stack up (last stage first),
    given each stage's dv, Isp and structural fraction."""
    stages = [Stage(s.name, s.isp, s.structural_fraction, s.dv) for s in stages]
    above = payload
    for s in reversed(stages):
        r = s.mass_ratio
        if r * s.structural_fraction >= 1.0:
            raise ValueError(f"stage '{s.name}' cannot deliver dv={s.dv:.3f} km/s: "
                             f"r*sigma={r*s.structural_fraction:.3f} >= 1 (max dv = {s.max_dv:.3f} km/s)")
        m_stage = above * (r - 1.0) / (1.0 - r * s.structural_fraction)
        s.m_struct = s.structural_fraction * m_stage
        s.m_prop = m_stage - s.m_struct
        s.m_payload_above = above
        above = s.m0
    return Vehicle(stages, payload)


def optimal_dv_split(dv_total: float, isps: Sequence[float], sigmas: Sequence[float]) -> list[float]:
    """dv per stage minimizing gross mass for fixed total dv (Lagrange solution)."""
    cs = [exhaust_velocity(i) for i in isps]
    n = len(cs)
    if n != len(sigmas):
        raise ValueError("isps and sigmas must have the same length")
    if n == 1:
        return [dv_total]

    def r_k(lam, c, s):
        return (lam * c - 1.0) / (lam * c * s)

    def g(lam):
        return sum(c * math.log(r_k(lam, c, s)) for c, s in zip(cs, sigmas)) - dv_total

    lam_lo = max(1.0 / (c * (1.0 - s)) for c, s in zip(cs, sigmas)) * (1.0 + 1e-9)   # each r_k > 1
    lam_hi = lam_lo
    while g(lam_hi) < 0.0 and lam_hi < 1e12:
        lam_hi *= 2.0
    if g(lam_hi) < 0.0:
        raise ValueError("dv_total not achievable with these stages (sum of max stage dv too small)")
    lam = brentq(g, lam_lo, lam_hi, xtol=1e-14, rtol=1e-12)
    return [c * math.log(r_k(lam, c, s)) for c, s in zip(cs, sigmas)]


def design_vehicle(dv_total: float, payload: float, isps: Sequence[float], sigmas: Sequence[float],
                   names: Sequence[str] | None = None, split: Sequence[float] | None = None) -> Vehicle:
    """Build and size an N-stage vehicle for ``dv_total`` [km/s].  ``split`` =
    dv per stage (defaults to the optimal split)."""
    dvs = list(split) if split is not None else optimal_dv_split(dv_total, isps, sigmas)
    if abs(sum(dvs) - dv_total) > 1e-9:
        raise ValueError("split must sum to dv_total")
    names = names or [f"stage {k+1}" for k in range(len(isps))]
    stages = [Stage(nm, i, s, d) for nm, i, s, d in zip(names, isps, sigmas, dvs)]
    return size_vehicle(stages, payload)


def single_stage_equivalent(dv_total: float, payload: float, isp: float, sigma: float) -> Vehicle | None:
    """The SSTO comparison case; None if sigma is too large for the dv."""
    try:
        return size_vehicle([Stage("SSTO", isp, sigma, dv_total)], payload)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Mission-leg propellant sizing (one stage, several burns, payload drops)
# ---------------------------------------------------------------------------
@dataclass
class Burn:
    name: str
    dv: float               # km/s
    isp: float              # s
    mass_dropped_after: float = 0.0   # kg jettisoned after this burn (e.g., a lander, a heat shield)


def size_burn_sequence(burns: Sequence[Burn], final_mass: float) -> list[dict]:
    """Work backward from the mass remaining after the last burn to find the
    propellant for each burn.  Returns a list of dicts in burn order with
    m_before, m_after, m_prop."""
    out = []
    m_after = final_mass
    for b in reversed(burns):
        m_after_burn = m_after + b.mass_dropped_after
        m_before = m_after_burn * mass_ratio_for_dv(b.isp, b.dv)
        out.append(dict(name=b.name, dv=b.dv, isp=b.isp, m_before=m_before, m_after=m_after_burn,
                        m_prop=m_before - m_after_burn, dropped_after=b.mass_dropped_after))
        m_after = m_before
    out.reverse()
    return out
