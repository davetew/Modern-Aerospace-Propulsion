"""Vehicle input deck: engines, stages, derived propulsion quantities.

The engine model is the physically consistent one from the plan:
    T(h)   = tau * (T_vac - p_a(h) * A_e)
    mdot   = tau * mdot_max,      mdot_max = T_vac / (Isp_vac * g0)
    Isp(h) = T(h) / (mdot * g0)
so an engine is fully specified by (T_vac, Isp_vac, A_e).  If the deck gives sea-level
thrust / Isp instead, A_e is *derived* from the SL-vac difference (A_e = (T_vac - T_sl)/p_sl)
and the two numbers must be consistent with one exit area -- the deck loader checks that.

Design-variable hooks (co-design mode) are the same quantities; the OpenMDAO propulsion
group that produces them from (p_c, eps, O/F) is a later milestone.
"""
from dataclasses import dataclass, field
from typing import List, Optional
import math
import yaml

from .constants import G0, P_SL


@dataclass
class Engine:
    name: str
    T_vac: float                 # N, single engine, 100 % throttle
    Isp_vac: float               # s
    A_e: Optional[float] = None  # m^2 exit area; derived from T_sl if None
    T_sl: Optional[float] = None  # N (optional, used to derive A_e)
    d_exit: Optional[float] = None  # m (optional, alternative to A_e)
    throttle_min: float = 0.4

    def __post_init__(self):
        if self.A_e is None:
            if self.d_exit is not None:
                self.A_e = math.pi * (0.5 * self.d_exit) ** 2
            elif self.T_sl is not None:
                self.A_e = (self.T_vac - self.T_sl) / P_SL
            else:
                raise ValueError(f"Engine {self.name}: give A_e, d_exit or T_sl")
        if self.T_sl is None:
            self.T_sl = self.T_vac - P_SL * self.A_e

    @property
    def mdot_max(self):
        return self.T_vac / (self.Isp_vac * G0)

    @property
    def Isp_sl(self):
        return self.T_sl / (self.mdot_max * G0)

    def thrust(self, p_a, tau=1.0):
        return tau * (self.T_vac - p_a * self.A_e)


@dataclass
class EngineSet:
    engine: Engine
    n: int

    @property
    def T_vac(self):
        return self.n * self.engine.T_vac

    @property
    def A_e(self):
        return self.n * self.engine.A_e

    @property
    def mdot_max(self):
        return self.n * self.engine.mdot_max


@dataclass
class Stage:
    name: str
    dry_mass: float               # kg, structure + engines (+ reserves are separate)
    propellant: float             # kg, loaded
    engines: List[EngineSet]
    reserve: float = 0.0          # kg propellant held back (boostback/landing) -> not burned on ascent
    throttle_min: float = 0.4

    @property
    def T_vac(self):
        return sum(e.T_vac for e in self.engines)

    @property
    def A_e(self):
        return sum(e.A_e for e in self.engines)

    @property
    def mdot_max(self):
        return sum(e.mdot_max for e in self.engines)

    @property
    def Isp_vac_eff(self):
        """Thrust-weighted vacuum Isp of the engine set."""
        return self.T_vac / (self.mdot_max * G0)

    @property
    def usable_propellant(self):
        return self.propellant - self.reserve


@dataclass
class Vehicle:
    name: str
    stages: List[Stage]          # ordered bottom (first to burn) -> top
    diameter: float              # m
    cd: float = 0.5
    payload: float = 0.0         # kg (initial guess / fixed value)
    notes: str = ""

    @property
    def A_ref(self):
        return math.pi * (0.5 * self.diameter) ** 2

    def mass_above(self, k):
        """Total mass (dry + propellant + reserve) of stages above stage k, excluding payload."""
        return sum(s.dry_mass + s.propellant for s in self.stages[k + 1:])

    def glow(self):
        return sum(s.dry_mass + s.propellant for s in self.stages) + self.payload

    def summary(self):
        lines = [f"Vehicle: {self.name}  GLOW = {self.glow()/1e3:,.1f} t  payload = {self.payload/1e3:,.1f} t  D = {self.diameter} m  Cd = {self.cd}"]
        for k, s in enumerate(self.stages):
            eng = " + ".join(f"{e.n}x{e.engine.name}" for e in s.engines)
            lines.append(f"  [{k}] {s.name}: dry {s.dry_mass/1e3:,.1f} t, prop {s.propellant/1e3:,.1f} t (reserve {s.reserve/1e3:,.1f} t), "
                         f"{eng}; T_vac {s.T_vac/1e6:,.2f} MN, T_sl {(s.T_vac-P_SL*s.A_e)/1e6:,.2f} MN, Isp_vac {s.Isp_vac_eff:.1f} s, "
                         f"mdot {s.mdot_max:,.0f} kg/s, A_e {s.A_e:.2f} m^2")
            m0 = s.dry_mass + s.propellant + self.mass_above(k) + self.payload
            lines.append(f"       liftoff/ignition T/W (vac) = {s.T_vac/(m0*G0):.2f}, burn time at 100% = {s.usable_propellant/s.mdot_max:.0f} s")
        return "\n".join(lines)


def load_vehicle(path) -> Vehicle:
    with open(path) as f:
        d = yaml.safe_load(f)
    engines = {}
    for name, e in d["engines"].items():
        engines[name] = Engine(name=name, **e)
    stages = []
    for s in d["stages"]:
        esets = [EngineSet(engines[en], int(n)) for en, n in s["engines"].items()]
        stages.append(Stage(name=s["name"], dry_mass=float(s["dry_mass"]), propellant=float(s["propellant"]),
                            engines=esets, reserve=float(s.get("reserve", 0.0)),
                            throttle_min=float(s.get("throttle_min", min(e.engine.throttle_min for e in esets)))))
    return Vehicle(name=d["name"], stages=stages, diameter=float(d["diameter"]), cd=float(d.get("cd", 0.5)),
                   payload=float(d.get("payload", 0.0)), notes=d.get("notes", ""))
