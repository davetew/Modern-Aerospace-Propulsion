# FFSC Engine Design Optimization

## Overview

The `optimize_ffsc_design()` function provides automated design optimization for Full-Flow Staged Combustion (FFSC) rocket engines. It finds the optimal combination of engine parameters that maximizes specific impulse (Isp) while satisfying all feasibility constraints.

## Key Features

### Optimization Variables (5 DOF)
1. **Chamber Pressure (p0)**: 10-35 MPa typical range
2. **O/F Ratio**: Propellant-dependent (CH4/O2: 2.0-4.5, H2/O2: 4.0-8.0)
3. **Throat Radius (r_t)**: Scaled to meet thrust requirement
4. **Expansion Ratio (ε)**: 10-40 for vacuum, 10-20 for sea level
5. **Nozzle Length (L_noz)**: Affects cooling and structural mass

### Constraints
1. **Turbopump Power Balance**: Both fuel and oxidizer turbines must provide sufficient power
2. **Wall Temperature Limit**: Maximum wall temperature ≤ 900 K (copper limit)
3. **Coolant Fraction**: Coolant flow ≤ 60% of fuel flow (practical limit)
4. **Cycle Feasibility**: Complete power balance and cooling analysis

### Optimization Algorithm
- Uses **scipy.optimize.minimize** with SLSQP (Sequential Least Squares Programming)
- Handles nonlinear constraints via penalty functions
- Typical convergence: 30-80 function evaluations (30-150 seconds)
- Alternative methods: "trust-constr", "COBYLA"

## Usage

### Basic Example
```python
from modern_aerospace_propulsion.ffsc_nozzle.cycle import optimize_ffsc_design

# Optimize 500 kN CH4/O2 vacuum engine
result = optimize_ffsc_design(
    F_vac=500e3,              # Target thrust (N)
    fuel_name="CH4",          # Cantera species
    ox_name="O2",
    fuel_coolprop="Methane",  # CoolProp fluid
    ox_coolprop="Oxygen",
    T_fuel_tank=110.0,        # Fuel tank temp (K)
    T_ox_tank=90.0,           # Oxidizer tank temp (K)
    p_amb=0.0,                # Ambient pressure (0 = vacuum)
    T_wall_max_limit=900.0,   # Max wall temp (K)
    verbose=True,             # Print progress
)

# Extract optimal parameters
if result['success']:
    print(f"Optimal Isp: {result['Isp_opt']:.2f} s")
    print(f"Chamber pressure: {result['p0_opt']/1e6:.2f} MPa")
    print(f"O/F ratio: {result['OF_opt']:.3f}")
    print(f"Expansion ratio: {result['eps_opt']:.1f}")
```

### Sea-Level Operation
```python
# Optimize for sea-level (adjust bounds for lower expansion)
result_sl = optimize_ffsc_design(
    F_vac=1000e3,
    fuel_name="CH4",
    ox_name="O2",
    fuel_coolprop="Methane",
    ox_coolprop="Oxygen",
    T_fuel_tank=110.0,
    T_ox_tank=90.0,
    p_amb=101325.0,           # Sea-level pressure
    eps_bounds=(10.0, 20.0),  # Lower expansion ratio
    verbose=True,
)
```

### Custom Bounds and Initial Guess
```python
result = optimize_ffsc_design(
    F_vac=500e3,
    # ... propellant parameters ...
    p0_bounds=(15e6, 30e6),        # Tighter chamber pressure range
    OF_bounds=(3.0, 3.5),          # Focus on specific O/F range
    r_t_bounds=(0.08, 0.15),       # Specific throat size range
    eps_bounds=(20.0, 35.0),       # High expansion for vacuum
    L_noz_bounds=(1.0, 2.5),       # Longer nozzles
    initial_guess={                 # Starting point
        'p0': 22e6,
        'OF': 3.2,
        'r_t': 0.10,
        'eps': 28.0,
        'L_noz': 1.8
    },
    max_iterations=150,             # Allow more iterations
    optimizer_method="trust-constr", # Alternative optimizer
)
```

## Return Values

```python
{
    # Optimal parameters
    'p0_opt': float,        # Chamber pressure (Pa)
    'OF_opt': float,        # O/F ratio
    'r_t_opt': float,       # Throat radius (m)
    'eps_opt': float,       # Expansion ratio
    'L_noz_opt': float,     # Nozzle length (m)
    
    # Performance
    'Isp_opt': float,       # Maximum Isp (s)
    'F_vac': float,         # Vacuum thrust (N)
    'm_dot_total': float,   # Total propellant flow (kg/s)
    
    # Optimization status
    'success': bool,        # True if converged
    'message': str,         # Optimizer message
    'n_iterations': int,    # Iterations performed
    'n_evaluations': int,   # Function evaluations
    
    # Complete cycle analysis at optimum
    'cycle_results': dict,  # Full ffsc_full_flow_cycle output
}
```

## Typical Results

### CH4/O2 (Methalox)
- **Vacuum Isp**: 330-350 s
- **Sea-Level Isp**: 300-320 s
- **Optimal O/F**: 3.0-3.5 (slightly fuel-rich)
- **Optimal p0**: 20-30 MPa
- **Vacuum Expansion**: 25-40
- **Sea-Level Expansion**: 12-18

### H2/O2 (Hydrolox)
- **Vacuum Isp**: 450-470 s
- **Sea-Level Isp**: 380-420 s
- **Optimal O/F**: 5.5-6.5
- **Optimal p0**: 20-35 MPa
- **Vacuum Expansion**: 30-60
- **Sea-Level Expansion**: 15-25

## Dependencies

Required packages (add to `requirements.txt`):
```
numpy>=1.20.0
scipy>=1.7.0
cantera>=2.5.0
CoolProp>=6.4.0
matplotlib>=3.3.0
```

Install via:
```bash
pip install numpy scipy cantera CoolProp matplotlib
```

## Performance Tips

1. **Start with reasonable bounds** based on propellant and application
2. **Use tighter bounds** if you know the approximate operating regime
3. **Relax constraints** if no feasible solution exists
4. **Try different initial guesses** to verify global optimum
5. **Use "trust-constr"** for stricter constraint handling
6. **Increase max_iterations** for difficult problems

## Limitations

1. **Computational cost**: Each evaluation takes 0.5-2 seconds
2. **Local optima**: SLSQP may find local optimum, try multiple starting points
3. **Constraint violations**: If no feasible solution exists, try relaxing limits
4. **Propellant compatibility**: Requires Cantera mechanism with specified species
5. **CoolProp support**: Coolant must be available in CoolProp database

## Examples

See `FFSC_Optimization_Example.ipynb` for detailed examples:
1. Vacuum engine optimization (CH4/O2)
2. Sea-level engine optimization (CH4/O2)
3. Vacuum vs. sea-level comparison
4. Sensitivity to initial guess
5. Custom bounds and constraints

## References

- Hill, P. G., & Peterson, C. R. (1992). *Mechanics and Thermodynamics of Propulsion*
- Huzel, D. K., & Huang, D. H. (1992). *Modern Engineering for Design of Liquid-Propellant Rocket Engines*
- Sutton, G. P., & Biblarz, O. (2016). *Rocket Propulsion Elements*
