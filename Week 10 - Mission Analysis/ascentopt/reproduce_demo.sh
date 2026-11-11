#!/bin/bash
# Reproduce the 12 Nov 2026 demo: Starship Block 3 baseline + dry-mass / reserve sweep + figures.
# Each case takes 1-4 min on a laptop (IPOPT, 16 Radau segments per phase).
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python}
COMMON="--vehicle vehicles/starship_v3_baseline.yaml --h-target 275e3 --lat 25.997 --q-max 35e3 --a-max 39.2266 --segments 16"

# 1. baseline (cold start from the gravity-turn guess), with the explicit re-integration check
$PY scripts/run_case.py $COMMON --out runs/starship --max-iter 400 --simulate --tag "baseline (dry 200/100 t, no reserves)"
BASE=$(find runs/starship -path "*_out/dymos_solution.db" | head -1)

# 2. sweep, warm-started from the baseline solution
$PY scripts/run_case.py $COMMON --out runs/sw_light     --max-iter 200 --warm-from "$BASE" --tag "dry 160/85 t (low estimate), no reserves" --set stage0.dry_mass=160e3 --set stage1.dry_mass=85e3
$PY scripts/run_case.py $COMMON --out runs/sw_heavy     --max-iter 200 --warm-from "$BASE" --tag "dry 250/120 t, no reserves" --set stage0.dry_mass=250e3 --set stage1.dry_mass=120e3
$PY scripts/run_case.py $COMMON --out runs/sw_shipdry   --max-iter 200 --warm-from "$BASE" --tag "ship dry 150 t (booster 200 t), no reserves" --set stage1.dry_mass=150e3
$PY scripts/run_case.py $COMMON --out runs/sw_reserves  --max-iter 200 --warm-from "$BASE" --tag "RTLS reserves: booster 292 t (8 %), ship 30 t" --set stage0.reserve=292e3 --set stage1.reserve=30e3
$PY scripts/run_case.py $COMMON --out runs/sw_res15     --max-iter 200 --warm-from "$BASE" --tag "RTLS reserves: booster 548 t (15 %), ship 30 t" --set stage0.reserve=548e3 --set stage1.reserve=30e3
$PY scripts/run_case.py $COMMON --out runs/sw_heavy_res --max-iter 200 --warm-from "$BASE" --tag "dry 250/120 t + RTLS reserves 292/30 t" --set stage0.dry_mass=250e3 --set stage1.dry_mass=120e3 --set stage0.reserve=292e3 --set stage1.reserve=30e3

# 3. figures
$PY scripts/make_figures.py runs/starship --title "Starship Block 3 → 275 km × 26°, Starbase (expendable trajectory, dry 200/100 t)" --out results/figs
$PY scripts/make_figures.py --sweep runs/starship runs/sw_light runs/sw_heavy runs/sw_shipdry runs/sw_reserves runs/sw_res15 runs/sw_heavy_res --out results/figs
