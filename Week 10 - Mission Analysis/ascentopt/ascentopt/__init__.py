"""ascentopt — launch-trajectory / propulsion co-design tool (Dymos + OpenMDAO).

MECE 4305 (Columbia, Fall 2026) rocket module.  Successor to the Week 9 slide-33
``LaunchTrajectoryAnalysis.ipynb`` forward simulation: same 3-DOF point-mass physics,
but the steering and throttle are solved for by direct collocation, staging is a
phase linkage, and vehicle/engine parameters can be design variables.
"""
from .constants import MU_E, R_E, OMEGA_E, G0, P_SL
from .vehicle import Engine, Stage, Vehicle, load_vehicle
from .problem import build_problem, run, Mission
from .guess import gravity_turn_guess

__all__ = ["MU_E", "R_E", "OMEGA_E", "G0", "P_SL", "Engine", "Stage", "Vehicle", "load_vehicle",
           "build_problem", "run", "Mission", "gravity_turn_guess"]
__version__ = "0.1.0"
