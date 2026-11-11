"""3-DOF point-mass ascent dynamics in Earth-centered inertial Cartesian coordinates (jax).

States (per phase):  r (3) [m], v (3) [m/s], m_prop [kg], and loss accumulators
                     dv_vac, L_grav, L_drag, L_steer, L_bp [m/s]
Controls:            u (3) thrust direction (unit vector enforced by path constraint), tau throttle
Phase parameters:    T_vac [N], A_e [m^2], mdot_max [kg/s], m_struct [kg] (everything on board
                     that is not this phase's propellant: this stage's dry mass + upper stages
                     + payload), m_pay [kg] (added to m_struct so it can be a design variable)

The rotation credit comes out of the initial condition v0 = omega x r0 (as in the slide-33
notebook); drag uses the air-relative velocity v - omega x r (the notebook used inertial v).
The loss accumulators integrate the terms of  d|v|/dt = T/m cos(alpha_T) - D/m - g sin(gamma)
so that   dv_vac = (|v_f| - |v_0|) + L_grav + L_drag + L_steer + L_bp   closes identically
(slide 28:  dv_ideal = v_orbit + losses - v_rotation).
"""
import numpy as np
import jax
import jax.numpy as jnp
import openmdao.api as om

from .constants import MU_E, R_E, OMEGA_E, G0
from .atmosphere import ATM

jax.config.update("jax_enable_x64", True)


class AscentODE(om.JaxExplicitComponent):

    def initialize(self):
        self.options.declare("num_nodes", types=int)
        self.options.declare("cd", types=float, default=0.5)
        self.options.declare("A_ref", types=float, default=63.617)
        self.options.declare("vertical", types=bool, default=False,
                             desc="if True the thrust direction is forced radial (vertical-rise phase)")
        self.options.declare("mu", types=float, default=MU_E)
        self.options.declare("R_E", types=float, default=R_E)
        self.options.declare("omega", types=float, default=OMEGA_E)

    def get_self_statics(self):
        o = self.options
        return (o["cd"], o["A_ref"], o["vertical"], o["mu"], o["R_E"], o["omega"])

    def setup(self):
        nn = self.options["num_nodes"]
        # states
        self.add_input("r", shape=(nn, 3), units="m")
        self.add_input("v", shape=(nn, 3), units="m/s")
        self.add_input("m_prop", shape=(nn,), units="kg")
        # controls
        self.add_input("u", shape=(nn, 3), units=None, val=np.tile([1.0, 0.0, 0.0], (nn, 1)))
        self.add_input("tau", shape=(nn,), units=None, val=1.0)
        # static phase parameters
        self.add_input("T_vac", shape=(1,), units="N", val=1.0e6)
        self.add_input("A_e", shape=(1,), units="m**2", val=1.0)
        self.add_input("mdot_max", shape=(1,), units="kg/s", val=300.0)
        self.add_input("m_struct", shape=(1,), units="kg", val=1.0e4)
        self.add_input("m_pay", shape=(1,), units="kg", val=0.0)

        # state rates
        self.add_output("rdot", shape=(nn, 3), units="m/s")
        self.add_output("vdot", shape=(nn, 3), units="m/s**2")
        self.add_output("mdot", shape=(nn,), units="kg/s")
        self.add_output("dv_vac_rate", shape=(nn,), units="m/s**2")
        self.add_output("L_grav_rate", shape=(nn,), units="m/s**2")
        self.add_output("L_drag_rate", shape=(nn,), units="m/s**2")
        self.add_output("L_steer_rate", shape=(nn,), units="m/s**2")
        self.add_output("L_bp_rate", shape=(nn,), units="m/s**2")
        # outputs for constraints / plots
        self.add_output("m", shape=(nn,), units="kg")
        self.add_output("h", shape=(nn,), units="m")
        self.add_output("vmag", shape=(nn,), units="m/s")
        self.add_output("rdotv", shape=(nn,), units="m**2/s", desc="r . v  (zero at circular insertion)")
        self.add_output("energy", shape=(nn,), units="m**2/s**2", desc="v^2/2 - mu/r")
        self.add_output("q", shape=(nn,), units="Pa")
        self.add_output("mach", shape=(nn,), units=None)
        self.add_output("thrust", shape=(nn,), units="N")
        self.add_output("isp", shape=(nn,), units="s")
        self.add_output("accel", shape=(nn,), units="m/s**2", desc="thrust acceleration T/m")
        self.add_output("u_norm2", shape=(nn,), units=None, desc="|u|^2 (path-constrained to 1)")
        self.add_output("alpha", shape=(nn,), units="rad", desc="angle between thrust and air-relative velocity")
        self.add_output("gamma", shape=(nn,), units="rad", desc="inertial flight-path angle")
        self.add_output("inc", shape=(nn,), units="rad", desc="inclination of the osculating orbit")
        self.add_output("lat", shape=(nn,), units="rad")
        self.add_output("lon_inertial", shape=(nn,), units="rad")

        # Node-diagonal sparsity: output at node i depends only on inputs at node i (and on the
        # static parameters).  Declaring it is what lets Dymos/OpenMDAO color the total Jacobian.
        ar = np.arange(nn)
        vec_out = ("rdot", "vdot")
        sca_out = ("mdot", "dv_vac_rate", "L_grav_rate", "L_drag_rate", "L_steer_rate", "L_bp_rate", "m", "h", "vmag",
                   "rdotv", "energy", "q", "mach", "thrust", "isp", "accel", "u_norm2", "alpha", "gamma", "inc", "lat",
                   "lon_inertial")
        vec_in = ("r", "v", "u")
        sca_in = ("m_prop", "tau")
        static_in = ("T_vac", "A_e", "mdot_max", "m_struct", "m_pay")
        # (nn,3) outputs
        r3 = np.repeat(3 * ar, 3) + np.tile(np.arange(3), nn)          # 3i+a for a in 0..2
        for of in vec_out:
            for wrt in vec_in:   # 3x3 block per node
                rows = np.repeat(r3, 3)
                cols = np.repeat(3 * ar, 9) + np.tile(np.arange(3), 3 * nn)
                self.declare_partials(of, wrt, rows=rows, cols=cols)
            for wrt in sca_in:
                self.declare_partials(of, wrt, rows=r3, cols=np.repeat(ar, 3))
            for wrt in static_in:
                self.declare_partials(of, wrt, rows=r3, cols=np.zeros(3 * nn, dtype=int))
        # (nn,) outputs
        for of in sca_out:
            for wrt in vec_in:
                self.declare_partials(of, wrt, rows=np.repeat(ar, 3), cols=np.repeat(3 * ar, 3) + np.tile(np.arange(3), nn))
            for wrt in sca_in:
                self.declare_partials(of, wrt, rows=ar, cols=ar)
            for wrt in static_in:
                self.declare_partials(of, wrt, rows=ar, cols=np.zeros(nn, dtype=int))

    def compute_primal(self, r, v, m_prop, u, tau, T_vac, A_e, mdot_max, m_struct, m_pay):
        cd, A_ref, vertical, mu, Re, omega = self.get_self_statics()

        rmag = jnp.linalg.norm(r, axis=1)
        h = rmag - Re
        rhat = r / rmag[:, None]
        m = m_prop + m_struct[0] + m_pay[0]

        # atmosphere & air-relative velocity (atmosphere co-rotates with Earth)
        w = jnp.array([0.0, 0.0, omega])
        v_rel = v - jnp.cross(jnp.broadcast_to(w, r.shape), r)
        vrel_mag = jnp.sqrt(jnp.sum(v_rel ** 2, axis=1) + 1e-6)
        rho = ATM.rho_j(h)
        pa = ATM.p_j(h)
        a_snd = ATM.a_j(h)
        q = 0.5 * rho * vrel_mag ** 2
        D = q * cd * A_ref
        mach = vrel_mag / a_snd

        # thrust direction
        if vertical:
            uhat = rhat
            u_norm2 = jnp.ones_like(h)
        else:
            u_norm2 = jnp.sum(u ** 2, axis=1)
            uhat = u / jnp.sqrt(u_norm2 + 1e-12)[:, None]

        thrust = tau * (T_vac[0] - pa * A_e[0])
        mdot_burn = tau * mdot_max[0]
        mdot = -mdot_burn
        isp = thrust / (mdot_burn * G0 + 1e-9)
        T = thrust

        g_vec = -mu * r / rmag[:, None] ** 3
        vdot = g_vec + (T / m)[:, None] * uhat - (D / m)[:, None] * v_rel / vrel_mag[:, None]
        rdot = v

        # losses (inertial-velocity definitions, see module docstring)
        vmag = jnp.sqrt(jnp.sum(v ** 2, axis=1) + 1e-6)
        vhat = v / vmag[:, None]
        sin_gamma = jnp.sum(rhat * vhat, axis=1)
        # smooth angle forms (atan2) -- arcsin/arccos have infinite slope at +-1, which the
        # gravity-turn initial guess hits exactly (thrust along velocity)
        v_perp = v - sin_gamma[:, None] * vmag[:, None] * rhat
        gamma = jnp.arctan2(sin_gamma * vmag, jnp.sqrt(jnp.sum(v_perp ** 2, axis=1) + 1e-6))
        cos_alphaT = jnp.sum(uhat * vhat, axis=1)
        dv_vac_rate = tau * T_vac[0] / m
        L_bp_rate = tau * pa * A_e[0] / m
        L_grav_rate = (mu / rmag ** 2) * sin_gamma
        L_drag_rate = D / m * jnp.sum(vhat * v_rel / vrel_mag[:, None], axis=1)
        L_steer_rate = (T / m) * (1.0 - cos_alphaT)

        rdotv = jnp.sum(r * v, axis=1)
        energy = 0.5 * vmag ** 2 - mu / rmag
        hvec = jnp.cross(r, v)
        inc = jnp.arctan2(jnp.sqrt(hvec[:, 0] ** 2 + hvec[:, 1] ** 2 + 1e-6), hvec[:, 2])
        vrel_hat = v_rel / vrel_mag[:, None]
        cos_alpha = jnp.sum(uhat * vrel_hat, axis=1)
        sin_alpha = jnp.sqrt(jnp.sum(jnp.cross(uhat, vrel_hat) ** 2, axis=1) + 1e-12)
        alpha = jnp.arctan2(sin_alpha, cos_alpha)
        lat = jnp.arctan2(r[:, 2], jnp.sqrt(r[:, 0] ** 2 + r[:, 1] ** 2 + 1e-6))
        lon_inertial = jnp.arctan2(r[:, 1], r[:, 0])
        accel = T / m

        return (rdot, vdot, mdot, dv_vac_rate, L_grav_rate, L_drag_rate, L_steer_rate, L_bp_rate,
                m, h, vmag, rdotv, energy, q, mach, thrust, isp, accel, u_norm2, alpha, gamma, inc, lat, lon_inertial)
