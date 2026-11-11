# sfmission — patched-conic mission analysis for lunar & interplanetary design

A small, tested Python package (plus an MCP server and CLI) for preliminary
mission design within the solar system: launch Δv, Hohmann and Lambert
transfers, porkchop launch-window maps, planetocentric departure/capture burns,
gravity assists, Earth–Moon patched conics, aerobraking/aerocapture credits,
the rocket equation and multistage sizing, and Δv budgets that feed propellant
mass. Built for MECE 4305 (Mechanics & Thermodynamics of Propulsion, Columbia
University) Week 10 *Mission Analysis* and HW 1, and for preliminary design
work.

Fidelity: **patched conics with analytic ephemerides.** Planet positions come
from the JPL mean-element tables (1800–2050 default; 3000 BC–3000 AD fallback),
the Moon from a truncated Meeus series. No perturbations, no n-body
propagation, no low-thrust. Expect Δv estimates good to a few percent of flown
missions (see *Validation*), not navigation-grade trajectories.


## Install

From this folder (`Week 10 - Mission Analysis/sfmission/`):

```bash
pip install -e .            # package + CLI (numpy, scipy, matplotlib)
pip install -e ".[mcp]"     # + MCP server
pip install -e ".[test]"    # + pytest, astropy (independent ephemeris check)
pytest                       # 85 tests, ~1 s
```

## Quick start (Python)

```python
from sfmission import *

# textbook estimate, circular coplanar planet orbits
print(hohmann_interplanetary("Earth", "Mars", park_alt=300, capture_peri_alt=300).summary())

# dated transfer from the ephemeris (Lambert arc), with departure & capture burns
leg = heliocentric_leg("Earth", "Mars", "2026-10-30", "2027-09-06")
print(leg.summary(), departure_dv("Earth", leg.vinf_dep_mag, 300).summary(), sep="\n")

# porkchop plot of the 2026/27 window
pc = porkchop("Earth", "Mars", "2026-09-01", "2027-02-15", 120, 450, park_alt=300, capture_peri_alt=400)
print(pc.summary());  from sfmission.plotting import plot_porkchop;  plot_porkchop(pc).savefig("pork.png")

# gravity-assist sequence (Europa Clipper dates)
m = interplanetary_mission(["Earth","Mars","Earth","Jupiter"], ["2024-10-14","2025-03-01","2026-12-03","2030-04-11"],
                           park_alt=300, capture_peri_alt=None)
print(m.summary())

# Earth-Moon: Hohmann estimate and BMW patched conic to a 100 km perilune
print(lunar_hohmann_estimate(185, 100).summary())
for s in design_lunar_transfer(185, 100, v0=10.95): print(s.summary())

# launch dv with losses and site credit, staging, budgets
print(launch_to_orbit(300, inclination_deg=51.6, latitude_deg=28.5).summary())
print(design_vehicle(9.3, 10000, isps=[300, 450], sigmas=[0.08, 0.10]).table())
b, mission = interplanetary_budget("Earth", "Mars", "2026-10-30", "2027-09-06", aerocapture=True)
print(b.table()); print(b.size_propellant(final_mass=1000, default_isp=320))
```

## CLI

```
sfmission hohmann Earth Mars
sfmission leg Earth Mars 2026-10-30 2027-09-06 --plot transfer.png
sfmission porkchop Earth Mars 2026-09-01 2027-02-15 --tof 120 450 --plot pork.png
sfmission windows Earth Mars 2026-01-01 --years 10
sfmission mission Earth Mars Earth Jupiter --epochs 2024-10-14 2025-03-01 2026-12-03 2030-04-11
sfmission lunar --park 185 --perilune 100 --tof 72 --plot lunar.png
sfmission launch --alt 300 --inc 51.6 --lat 28.5
sfmission stage 9.3 10000 --isp 300 450 --sigma 0.08 0.10
sfmission budget Earth Mars 2026-10-30 2027-09-06 --aerocapture --isp 320 --final-mass 1000 --plot budget.png
sfmission mcp                      # MCP server on stdio
```

## MCP server

`sfmission-mcp` (stdio) or `python -m sfmission.mcp_server --http 8765`. Works
with the 1.x (`FastMCP`) and 2.x (`MCPServer`) Python SDKs. Figures are
written to `$SFMISSION_FIG_DIR` (default `./sfmission_figures`) and their paths
returned. Claude Desktop / Claude Code config:

```json
{"mcpServers": {"sfmission": {"command": "sfmission-mcp"}}}
```

Tools: `list_bodies`, `planet_state`, `phase_angle`, `hohmann_transfer`,
`bielliptic_transfer`, `interplanetary_hohmann`, `lambert_leg`, `porkchop`,
`launch_windows`, `departure_dv`, `capture_dv`, `gravity_assist_mission`,
`flyby_capability`, `lunar_transfer`, `launch_dv`, `aerocapture`,
`aerobraking`, `stage_vehicle`, `rocket_equation`, `burn_sequence_propellant`,
`mission_budget`, `lunar_mission_budget`, `version`.

## Package layout

| module | contents |
|---|---|
| `bodies` | μ (DE440), radii, SOI, rotation, exponential atmospheres |
| `kepler` | vis-viva, elements↔state (Curtis Alg. 4.2/4.5), Kepler's equation, universal-variable propagation (Alg. 3.4), hyperbola relations |
| `ephemeris` | JPL mean-element planets (Tables 1, 2a/2b), truncated Meeus Moon, time utilities |
| `lambert` | universal-variable Lambert solver (BMW §5.3 / Curtis Alg. 5.2), bracketed root, prograde/retrograde |
| `transfers` | Hohmann, bi-elliptic, plane change, combined optimum split, phasing, round trips |
| `patched_conics` | departure/capture burns, unpowered & powered flybys, heliocentric legs, multi-leg missions |
| `lunar` | Hohmann-to-Moon estimate; BMW §7.4 patched conic (vector form) with design solver |
| `launch` | Δv to orbit: ideal vector difference incl. site rotation & azimuth, loss bands |
| `aerobraking` | aerocapture credit, per-pass Δv (gravity-aware & straight-line forms), pass-by-pass campaign, numerical pass integrator |
| `staging` | rocket equation, required Isp, N-stage sizing with closed-form optimal Δv split, burn sequences |
| `budget` | `DeltaVBudget` (legs, bands, margins, propellant sizing) and mission composers |
| `porkchop` | departure × TOF grids of C3, v∞, Δv; optima; windows; multi-year scans |
| `plotting` | porkchop, transfer geometry, budget bars, staging trade, lunar transfer |
| `mcp_server`, `cli` | agent and command-line front ends |

## Units and nomenclature

km, s, kg; radians in low-level functions, degrees for arguments ending in
`_deg`; epochs as `datetime`, ISO-8601 string or Julian Date (UTC; TT−UTC =
69.184 s applied). Symbols follow Sutton to avoid the collisions noted in the
module review:

| symbol | meaning |
|---|---|
| μ | gravitational parameter GM [km³/s²] |
| v∞, C3 = v∞² | hyperbolic excess speed, launch energy |
| δ | flyby turning angle = 2 asin(1/e) |
| ζ = m_p/m₀ | propellant mass fraction |
| MR = m_f/m₀ | mass ratio; r_k = m₀,k/m_f,k = exp(Δv_k/(g₀ Isp_k)) stage mass ratio |
| σ = m_s/(m_s+m_p) | stage structural (inert) fraction |
| λ = m_L/(m_s+m_p) | payload ratio |
| λ₁ | lunar SOI arrival angle (BMW) |
| β = m/(C_D A) | ballistic coefficient [kg/m²] |
| H | atmospheric scale height |

ε is deliberately **not** used (reserved for nozzle area ratio).

## Theory notes (what the code implements)

**Launch.** Δv = |v_orbit − v_site| + losses, with v_site = ωR cos φ eastward
and launch azimuth from cos i = cos φ sin Az. KSC due east: 7.726 − 0.409 +
(1.15…2.05) = 8.47–9.37 km/s. (Corrects the sign error on Week-9 slide 32.)

**Patched conics.** Heliocentric leg from the ephemeris positions by Lambert;
v∞ = v_sc − v_planet at each end. Departure: Δv = √(v∞² + 2μ/r_p) − v_park.
Capture: Δv = √(v∞² + 2μ/r_p) − v_peri(orbit). Flyby: turn δ = 2 asin(1/(1 +
r_p v∞²/μ)); heliocentric |Δv| = 2 v∞ sin(δ/2). Powered flyby (MGA model): a
shared periapsis r_p satisfies δ_in/2 + δ_out/2 = ∠(v∞,in, v∞,out), with the
speed mismatch made up at periapsis.

**Earth–Moon (BMW §7.4).** Geocentric ellipse from (r₀, v₀, φ₀) to the SOI
point set by λ₁; selenocentric state v₂ = v₁ − v_Moon; hyperbola (e, r_p,
v_p); time of flight to the SOI plus the hyperbolic transit to perilune; Moon
lead angle at TLI. Implemented with vectors — the textbook trig relations are
checked in the tests.

**Aerobraking.** Per pass Δv = ρ_p V_p² /(2β) √(2πH/r̈_p) with r̈_p = V_p²/r_p
− μ/r_p² at periapsis. Dropping the gravity term (straight-line flythrough)
gives the compact form ρ_p V_p √(2πH r_p)/(2β) used on the Week-11 slide; it
is **low by √(V_p²/(V_p²−μ/r_p)) → √2 for a near-parabolic pass**. The
numerical integrator in `aerobraking_pass_dv_numeric` confirms the
gravity-aware form to 0.1 %.

**Staging.** m₀/m_L = Π r_k(1−σ_k)/(1−r_kσ_k); the optimal split for fixed
total Δv is r_k = (λc_k − 1)/(λc_kσ_k) with Σ c_k ln r_k = Δv (one scalar
root for the multiplier λ).

## Validation (all in `tests/`)

| check | result |
|---|---|
| Lambert, Curtis Example 5.2 | v₁, v₂ match to 1e-4 km/s; propagation closes to 1e-8 km |
| Elements, Curtis Example 4.3 | h, e, i, Ω, ω, ν match |
| Planet ephemeris vs astropy builtin (ERFA/plan94) | inner planets < 35″, Earth velocity < 1 m/s; outer planets within JPL's quoted table accuracy |
| Moon vs astropy builtin | < 40″, < 40 km |
| Mars oppositions 2022-12-08, 2025-01-16, 2027-02-19 | Earth/Mars longitudes agree < 1.5° |
| Earth→Mars Hohmann (Curtis Ex. 8.4–8.6) | v∞ 2.943/2.648 km/s, Δv 3.590 km/s from 300 km, phase 44.3°, synodic 779.9 d |
| LEO→GEO Hohmann; 28.5° combined plane change | 3.893 km/s; 4.23 km/s |
| **Mars 2020** (launch 2020-07-30 11:50, entry 2021-02-18 20:37) | path length 470.9 Mkm vs press-kit ≈471 Mkm; C3 14.5 km²/s²; entry speed consistent with 19,500 km/h |
| Mars 2020 porkchop | optimum in July 2020, C3 < 15 |
| **Voyager 2** E-J-S dates | Jupiter flyby turn 97°, closest approach 620,000 km (actual ≈570,000 km above cloud tops), powered correction 0.05 km/s; C3 105 (Titan IIIE-Centaur ≈102–106) |
| **Apollo 11** TLI state → LOI | 68–73 h vs actual 73.0 h; LOI 0.89–0.92 km/s vs actual 0.891 km/s |
| Lunar Hohmann estimate | TLI 3.135 km/s, LOI 0.82 km/s |
| Aerobraking pass, analytic vs integrated | 1.952 vs 1.951 m/s |
| Staging identities, optimal split, SSTO limit | exact |

## Examples (`../examples/`)

1. `01_basics_hohmann_launch.py` — Δv to LEO from several sites, Hohmann estimates to the planets, rocket equation & staging.
2. `02_mars_porkchop_2026.py` — the Earth→Mars window opening Oct 2026: porkchop plots, optimum transfer geometry, upcoming opportunities to 2035.
3. `03_lunar_mission.py` — Hohmann vs patched-conic lunar transfers, Apollo 11 check, full landing-and-return budget.
4. `04_gravity_assists.py` — Europa Clipper MEGA and Voyager 2 sequences; flyby capability by planet.
5. `05_aerobraking_mars.py` — aerocapture credit, aerobraking campaigns vs ballistic coefficient, analytic vs numeric pass.
6. `06_hw1_reference_solution.py` — HW 1 worked solution (Mars orbiter, 2026 window): launch band → dated legs → budgets with/without aerocapture → required Isp → spacecraft propellant → two-stage launcher.

Figures from these runs are in `../figures/`; the technical report (`../docs/report/`), the Beamer summary (`../docs/slides/`) and the tutorial *Orbital Mechanics for Propulsion Engineers* (`../docs/tutorial/`) describe the modeling approach, validation and limitations in full.

## Known limitations / next steps

* Single-revolution Lambert only (no multi-rev porkchop lobes).
* "Earth" in the ephemeris is the Earth–Moon barycenter unless `earth_state(..., correct_for_moon=True)`.
* Atmosphere models are single-scale-height fits for the aerobraking band; supply ρ_p directly when it matters.
* Lunar module assumes a circular Moon orbit at 384,400 km; use `moon_state_geocentric` for the real distance on a date and pass `r_moon` to the Hohmann estimate.
* Candidates for v2: low-thrust (Edelbaum, power-limited rocket equation), DE440 via jplephem, n-body verification, multi-rev Lambert, B-plane targeting, launch-vehicle performance curves (C3 vs payload).

## References

Sutton & Biblarz, *Rocket Propulsion Elements* (Ch. 4); Curtis, *Orbital
Mechanics for Engineering Students* (Ch. 3–8); Bate, Mueller & White,
*Fundamentals of Astrodynamics* (Ch. 5, 7, 8); Standish, *Keplerian Elements
for Approximate Positions of the Major Planets* (JPL SSD); Meeus,
*Astronomical Algorithms* 2nd ed. (Ch. 47); Park et al. 2021, DE440; Izzo
2010 (MGA powered-flyby model).
