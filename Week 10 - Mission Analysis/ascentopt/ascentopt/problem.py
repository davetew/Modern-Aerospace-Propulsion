"""Build and run the multi-phase Dymos ascent problem."""
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import openmdao.api as om
import dymos as dm

from .constants import MU_E, R_E, OMEGA_E, G0
from .ode import AscentODE
from .guess import gravity_turn_guess, tune_kick, launch_state

LOSS_STATES = ("dv_vac", "L_grav", "L_drag", "L_steer", "L_bp")


@dataclass
class Mission:
    h_target: float = 275e3          # m, circular orbit altitude
    lat_deg: float = 25.997          # launch latitude (Starbase)
    lon_deg: float = 0.0
    azimuth_deg: float = 90.0        # initial-guess launch azimuth
    inc_deg: Optional[float] = None  # target inclination; None = free (minimum = latitude)
    q_max: Optional[float] = None    # Pa, dynamic-pressure path constraint
    a_max: Optional[float] = None    # m/s^2, thrust-acceleration path constraint
    t_rise: float = 10.0             # s, vertical-rise phase duration (fixed)
    free_throttle: bool = True
    objective: str = "max_payload"   # 'max_payload' | 'min_propellant' | 'min_time'
    name: str = "LEO"

    @property
    def r_target(self):
        return R_E + self.h_target

    @property
    def v_circ(self):
        return np.sqrt(MU_E / self.r_target)


@dataclass
class Options:
    num_segments: int = 20
    order: int = 3
    optimizer: str = "IPOPT"        # 'IPOPT' (pyoptsparse) or 'SLSQP' (scipy)
    max_iter: int = 500
    tol: float = 1e-6
    refine_iteration_limit: int = 0
    simulate: bool = True
    kick_deg: Optional[float] = None  # None -> tuned
    verbose: bool = True
    restart: Optional[str] = None     # path to a previous dymos_solution.db to warm-start from
    acceptable_tol: float = 1e-4


def _phase(vehicle, k, mission, opts, vertical=False, num_segments=None):
    tx = dm.Radau(num_segments=num_segments or opts.num_segments, order=opts.order, compressed=True)
    ph = dm.Phase(ode_class=AscentODE, transcription=tx,
                  ode_init_kwargs=dict(cd=vehicle.cd, A_ref=vehicle.A_ref, vertical=vertical))
    ph.add_state("r", shape=(3,), rate_source="rdot", units="m", ref=1e6, defect_ref=1e6)
    ph.add_state("v", shape=(3,), rate_source="vdot", units="m/s", ref=1e3, defect_ref=1e3)
    ph.add_state("m_prop", rate_source="mdot", units="kg", ref=1e5, defect_ref=1e5, lower=0.0)
    for s in LOSS_STATES:
        ph.add_state(s, rate_source=f"{s}_rate", units="m/s", ref=1e3, defect_ref=1e3)
    st = vehicle.stages[k]
    if vertical:
        ph.add_parameter("u", shape=(3,), units=None, opt=False, static_target=False, val=np.array([1.0, 0.0, 0.0]))
        ph.add_parameter("tau", units=None, opt=False, static_target=False, val=1.0)
    else:
        ph.add_control("u", shape=(3,), units=None, lower=-1.5, upper=1.5, ref=1.0, continuity=True, rate_continuity=False)
        if mission.free_throttle:
            ph.add_control("tau", units=None, lower=st.throttle_min, upper=1.0, ref=1.0, continuity=True, rate_continuity=False)
        else:
            ph.add_parameter("tau", units=None, opt=False, static_target=False, val=1.0)
        ph.add_path_constraint("u_norm2", equals=1.0, ref=1.0)
        ph.add_path_constraint("h", lower=0.0, ref=1e4)
        if mission.q_max is not None:
            ph.add_path_constraint("q", upper=mission.q_max, ref=1e4)
        if mission.a_max is not None:
            ph.add_path_constraint("accel", upper=mission.a_max, ref=10.0)
    for name, un in (("T_vac", "N"), ("A_e", "m**2"), ("mdot_max", "kg/s"), ("m_struct", "kg")):
        ph.add_parameter(name, units=un, opt=False, static_target=True)
    ph.add_timeseries_output(["m", "h", "vmag", "q", "mach", "thrust", "isp", "accel", "alpha", "gamma", "inc", "lat",
                              "lon_inertial", "rdotv", "energy"])
    return ph


def build_problem(vehicle, mission: Mission, opts: Options = Options()):
    p = om.Problem(model=om.Group(), name=f"ascentopt_{vehicle.name.replace(' ', '_')}")
    if opts.optimizer.upper() == "IPOPT":
        p.driver = om.pyOptSparseDriver(optimizer="IPOPT", print_results=False)
        p.driver.opt_settings.update({"print_level": 5 if opts.verbose else 0, "tol": opts.tol,
                                      "max_iter": opts.max_iter, "mu_strategy": "adaptive",
                                      "nlp_scaling_method": "gradient-based", "linear_solver": "mumps",
                                      "acceptable_tol": opts.acceptable_tol, "acceptable_iter": 5,
                                      "limited_memory_max_history": 25, "bound_push": 1e-4,
                                      "output_file": "IPOPT.out"})
    else:
        p.driver = om.ScipyOptimizeDriver(optimizer="SLSQP", maxiter=opts.max_iter, tol=opts.tol, disp=opts.verbose)
    p.driver.declare_coloring(tol=1e-12)

    traj = dm.Trajectory()
    p.model.add_subsystem("traj", traj)
    ns = len(vehicle.stages)
    names = ["rise"] + [f"stage{k}" for k in range(ns)]
    phases = {}
    phases["rise"] = traj.add_phase("rise", _phase(vehicle, 0, mission, opts, vertical=True, num_segments=3))
    for k in range(ns):
        phases[f"stage{k}"] = traj.add_phase(f"stage{k}", _phase(vehicle, k, mission, opts))

    # payload as a trajectory-level parameter feeding every phase's ODE
    traj.add_parameter("m_pay", units="kg", opt=(mission.objective == "max_payload"), val=vehicle.payload,
                       lower=0.0, upper=1e6, ref=1e5, static_target=True,
                       targets={n: ["m_pay"] for n in names})

    # time options
    rise = phases["rise"]
    rise.set_time_options(fix_initial=True, fix_duration=True, units="s", initial_val=0.0, duration_val=mission.t_rise)
    for k in range(ns):
        ph = phases[f"stage{k}"]
        st = vehicle.stages[k]
        tb = st.usable_propellant / st.mdot_max
        ph.set_time_options(fix_initial=False, duration_bounds=(0.2 * tb, 3.0 * tb), duration_ref=100.0, units="s")

    # initial state of the rise phase
    r0, v0 = launch_state(mission.lat_deg, mission.lon_deg)
    for s in ("r", "v") + LOSS_STATES:
        rise.set_state_options(s, fix_initial=True)
    rise.set_state_options("m_prop", fix_initial=True)
    # stage phases: propellant is its own state per stage, fixed at the load when the stage ignites
    for k in range(ns):
        ph = phases[f"stage{k}"]
        if k > 0:
            ph.set_state_options("m_prop", fix_initial=True)
        ph.add_boundary_constraint("m_prop", loc="final", lower=vehicle.stages[k].reserve, ref=1e5)

    # linkages: continuity of time, r, v and loss accumulators; m_prop continuity only rise->stage0
    for a, b in zip(names[:-1], names[1:]):
        traj.link_phases([a, b], vars=["time"], ref=100.0)
        traj.link_phases([a, b], vars=["r"], ref=1e6)
        traj.link_phases([a, b], vars=["v"], ref=1e3)
        traj.link_phases([a, b], vars=list(LOSS_STATES), ref=1e3)
    traj.link_phases(["rise", "stage0"], vars=["m_prop"], ref=1e5)

    # terminal constraints
    last = phases[names[-1]]
    last.add_boundary_constraint("h", loc="final", equals=mission.h_target, ref=1e5)
    last.add_boundary_constraint("gamma", loc="final", equals=0.0, ref=0.1)
    last.add_boundary_constraint("vmag", loc="final", equals=mission.v_circ, ref=1e3)
    if mission.inc_deg is not None:
        last.add_boundary_constraint("inc", loc="final", equals=np.radians(mission.inc_deg), ref=1.0)

    # objective
    if mission.objective == "max_payload":
        p.model.add_objective("traj.parameter_vals:m_pay", ref=-1e5)
    elif mission.objective == "min_propellant":
        last.add_objective("m_prop", loc="final", ref=-1e5)
    elif mission.objective == "min_time":
        last.add_objective("time", loc="final", ref=100.0)
    else:
        raise ValueError(mission.objective)

    p.model.linear_solver = om.DirectSolver()
    p.setup(check=False, force_alloc_complex=False)

    # ---- parameters ----
    def apply_deck_parameters():
        for k in range(ns):
            st = vehicle.stages[k]
            m_struct = st.dry_mass + vehicle.mass_above(k)
            for n in (["rise"] if k == 0 else []) + [f"stage{k}"]:
                ph = phases[n]
                ph.set_parameter_val("T_vac", st.T_vac, units="N")
                ph.set_parameter_val("A_e", st.A_e, units="m**2")
                ph.set_parameter_val("mdot_max", st.mdot_max, units="kg/s")
                ph.set_parameter_val("m_struct", m_struct, units="kg")
    apply_deck_parameters()
    traj.set_parameter_val("m_pay", vehicle.payload, units="kg")
    p._ascentopt_apply_deck = apply_deck_parameters
    p._ascentopt_phases = phases

    # ---- initial guess from the gravity turn ----
    gkw = dict(t_rise=mission.t_rise, azimuth_deg=mission.azimuth_deg, lon_deg=mission.lon_deg, a_max=mission.a_max)
    kick = opts.kick_deg if opts.kick_deg is not None else tune_kick(vehicle, mission.lat_deg, **gkw)
    g = gravity_turn_guess(vehicle, mission.lat_deg, kick_deg=kick, **gkw)
    p._ascentopt_guess = dict(kick_deg=kick, phases=g)
    for n in names:
        if n not in g:
            continue
        gp, ph = g[n], phases[n]
        t = gp["t"]
        ph.set_time_val(initial=t[0], duration=max(t[-1] - t[0], 1.0))
        ph.set_state_val("r", gp["r"], time_vals=t, units="m")
        ph.set_state_val("v", gp["v"], time_vals=t, units="m/s")
        ph.set_state_val("m_prop", gp["m_prop"], time_vals=t, units="kg")
        for s in LOSS_STATES:
            ph.set_state_val(s, [0.0, 10.0 * (t[-1] - t[0])], units="m/s")
        if n != "rise":
            ph.set_control_val("u", gp["u"], time_vals=t)
            if mission.free_throttle:
                ph.set_control_val("tau", gp["tau"], time_vals=t)
    return p


def run(p, opts: Options = Options()):
    if opts.restart:
        # Load the previous solution as the initial guess, then re-apply the *deck* parameters:
        # Dymos' restart would otherwise restore the recorded m_struct/T_vac/... and silently undo
        # any vehicle change (dry-mass sweeps, engine changes).  The payload parameter is kept
        # from the record as a starting point when it is a design variable.
        case = om.CaseReader(opts.restart).get_case("final")
        p.load_case(case)
        p._ascentopt_apply_deck()
    dm.run_problem(p, run_driver=True, simulate=opts.simulate, make_plots=False, restart=None,
                   refine_iteration_limit=opts.refine_iteration_limit, refine_method="hp")
    return p
