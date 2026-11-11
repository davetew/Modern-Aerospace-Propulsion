# ascentopt — launch-trajectory / propulsion co-design tool (Dymos + OpenMDAO)

MECE 4305 (Columbia, Fall 2026), rocket module.  Successor to the Week 9 slide-33
`LaunchTrajectoryAnalysis.ipynb`: the same 3-DOF point-mass physics, but the steering and
throttle are **solved for** by direct collocation (Legendre–Gauss–Radau), staging is a phase
linkage, and the vehicle deck (stage masses, reserves, engines) is an input that can be swept —
and, in the next milestone, optimized in the same NLP.

Status (9 Nov 2026): **prototype v0.1** — fixed-vehicle mode, multi-stage, max-payload /
min-propellant / min-time objectives, path constraints (max-q, acceleration, altitude ≥ 0),
circular-orbit terminal set, loss accumulators, gravity-turn initial guess, explicit
re-integration check.  Co-design mode (engine model from p_c, ε, O/F; mass model) is the
next milestone (see the plan document in the course project).

## Install

```bash
conda env create -f environment.yml     # conda-forge: openmdao, pyoptsparse, cyipopt (IPOPT), jax; pip: dymos 1.16
conda activate ascentopt
```
`cyipopt` must be present or the IPOPT driver fails with `ModuleNotFoundError: cyipopt`
(pyoptsparse ≥ 2.17 calls IPOPT through it).  SciPy's SLSQP (`--optimizer SLSQP`) runs without
pyoptsparse but does not converge these problems — it hit the iteration limit even on Dymos'
own SSTO example — so treat it as a smoke test only.

## Run

```bash
# Starship Block 3 baseline → 275 km circular, free inclination (= 26° from Starbase), q ≤ 35 kPa, a ≤ 4 g
python scripts/run_case.py --vehicle vehicles/starship_v3_baseline.yaml --out runs/starship \
    --h-target 275e3 --lat 25.997 --q-max 35e3 --a-max 39.2266 --segments 16 --max-iter 400 --simulate

# change the deck without editing it, warm-started from the baseline solution
python scripts/run_case.py ... --out runs/heavy --set stage0.dry_mass=250e3 --set stage1.dry_mass=120e3 \
    --warm-from runs/starship/<...>_out/dymos_solution.db

python scripts/make_figures.py runs/starship --title "..." --out results/figs
python scripts/make_figures.py --sweep runs/* --out results/figs
bash reproduce_demo.sh          # baseline + 6-case sweep + figures (≈ 15–25 min)
```
`--restart` resumes a case from its own `dymos_solution.db` (the IPOPT barrier state is not
kept, but the primal solution is; 2–3 restarts of 100 iterations behave like one long run).
Results land in `<out>/result.json`; the Dymos record is `<out>/<problem>_out/dymos_solution.db`.

## What the model is

* **Frame / states.** ECI Cartesian r, v (3 each), per-stage propellant m_prop, and five
  loss accumulators (Δv_vac, gravity, drag, steering, back-pressure).  Rotation credit comes
  from v₀ = ω × r₀; drag uses air-relative velocity v − ω × r (the notebook used inertial v).
* **Controls.** Thrust-direction unit vector u (|u|² = 1 path constraint) and throttle τ ∈ [0.4, 1].
* **Propulsion.** T(h) = τ (T_vac − p_a(h) A_e), ṁ = τ ṁ_max, ṁ_max = T_vac/(Isp_vac g₀).  An engine
  is (T_vac, Isp_vac, A_e); if the deck gives T_sl, A_e is derived from the SL–vac pair.
* **Environment.** US Standard Atmosphere 1976 (analytic layers to 86 km, tabulated above) as
  a smooth log-interpolant; spherical Earth, point-mass gravity; C_D constant (0.5), A_ref = π(D/2)².
* **Phases.** `rise` (vertical, fixed t_rise, 33 engines) → `stage0` → `stage1` … ; time, r, v
  and the loss states are linked; propellant is a separate state per stage so stage dry masses
  can be swept (or made design variables) without touching linkages; booster/ship *reserves*
  are lower bounds on final propellant.
* **Terminal set.** h = h_target, γ = 0, |v| = √(μ/r); optional inclination.
* **Numerics.** `om.JaxExplicitComponent` ODE with explicit node-diagonal partial sparsity
  (43 colors on an 843-variable problem; without the declaration it was 504), Radau order 3,
  12–16 segments per phase, IPOPT (L-BFGS, mumps), scaling via Dymos `ref`/`defect_ref`.

## Demo results (12 Nov 2026) — Starship Block 3 to 275 km × 26°

Inputs: booster 3,650 t propellant / 33 Raptor 3 (250 tf SL, 275 tf vac, Isp_vac 350 s); ship
1,600 t / 3 Raptor 3 + 3 RVac (258 tf, Isp 380 s); 9 m; C_D 0.5; q ≤ 35 kPa; a ≤ 4 g; Starbase 26.0° N.
Dry masses are **not published** — slide-42 mid-points 200 t / 100 t are the baseline and are swept.

| Case | Max payload |
|---|---|
| dry 160/85 t (slide-42 low), no reserves | 314.9 t |
| **baseline: dry 200/100 t, no reserves** | **293.3 t** |
| dry 250/120 t, no reserves | 265.3 t |
| ship dry 150 t (booster 200 t), no reserves | 243.3 t  (−50.0 t for +50 t ship dry: exactly 1:1) |
| RTLS reserves: booster 292 t (8 %), ship 30 t | 204.5 t |
| dry 250/120 t + reserves 292/30 t | 178.4 t |
| RTLS reserves: booster 548 t (15 %), ship 30 t | 162.7 t |

Baseline trajectory: throttle bucket to 0.74 holds q = 35 kPa at 50–75 s; booster reaches the
4 g limit at MECO (t = 145 s, h = 60 km, v = 2.61 km/s inertial); ship burns 463 s, throttling
to 0.4 for the last ~170 s; insertion at t = 608 s.  Δv budget: Δv_vac 9,155 = 7,322 (|v_f| − |v_0|,
v_0 = 418 m/s) + 1,049 gravity + 690 steering + 76 back-pressure + 18 drag m/s — the slide-28
identity closes to 0.001 m/s.  Losses total 1.83 km/s, inside the 1.5–2.0 km/s rule of thumb.

The slide-33 gravity-turn law applied to the same vehicle (kick 3.3°) lofts to 218 km at
10.1 km/s — an eccentric orbit, not the 275 km circle — which is the picture for "why optimize".

**Verification.** (V1) Explicit re-integration of the optimized controls (Dymos `simulate`):
terminal mismatch +211 m, +4.5 m/s, 0.0002° on the 16-segment grid (+322 m, +8.8 m/s at 12
segments).  (V2) Dymos SSTO benchmark (Longuski Titan II) reproduced: t_f = 143.77 s.  (V3)
Loss-identity residual < 0.004 m/s in every case.  (V4) Staging altitude/time (60 km / 145 s
expendable; 53 km / 133 s with 8 % reserve) are in the range of the IFT-3 public timeline
(72 km / 170 s), which flew a return-to-site profile.  Grid independence: 293.20 t (12 seg)
vs 293.31 t (16 seg).

**Not achieved / caveats.** (1) The notebook's own "Falcon-9-like" single stage (T/W 3.75,
Isp 280/310, 95 % propellant fraction) did not converge as a max-payload problem; a
min-propellant run got within 0.1 constraint violation of a 250 km circular orbit with
≈ 0 payload and 79 g at burnout — the vehicle is unphysical and is not used in the demo.
(2) C_D is constant; no C_D(M) table yet.  (3) No J2, no winds, no 6-DOF.  (4) The final
0.4-throttle segment of the ship burn is what IPOPT returned as optimal on this grid; it should
be re-checked with grid refinement and a different initial guess before drawing conclusions from it.
(5) Reserves are inputs: boostback/landing trajectories are not simulated.

## Layout

```
ascentopt/   atmosphere.py  US76 table (numpy + jax)      guess.py   gravity-turn integrator (initial guess / verification)
             ode.py         jax ODE (JaxExplicitComponent) problem.py Mission/Options, build_problem(), run()
             vehicle.py     YAML deck → Engine/Stage/Vehicle post.py  figures
scripts/     run_case.py (CLI, restart, --set overrides)   make_figures.py
vehicles/    starship_v3_baseline.yaml, slide33_falcon9like.yaml
results/     result_*.json (7 cases), figs/*.png, and the Dymos records the documents are built from
             (starship16_dymos_solution.db, starship16_dymos_simulation.db, sw_reserves_dymos_solution.db, *_guess.npz)
docs/        report.pdf (technical report, 13 pp.), slides.pdf (23 frames, course lecture format: Beamer Madrid +
             Columbia colour theme, beamercolorthemeColumbia.sty alongside), their .tex sources, figs/, and
             make_doc_figs.py, which regenerates every figure from results/ (then: latexmk -pdf report.tex / slides.tex)
```

The launch leg this tool computes hands off to the in-space analysis in the sibling `sfmission/` package
(parking orbit → departure, capture, landing Δv budgets).
