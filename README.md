# Modern-Aerospace-Propulsion
Analyses and Data for the Modern Aerospace Propulsion course @ Columbia

## Installation

This repository is structured as a Python package. You can install it in several ways:

### Install from Source
```bash
# Clone the repository
git clone https://github.com/davetew/Modern-Aerospace-Propulsion.git
cd Modern-Aerospace-Propulsion

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode (editable)
pip install -e .
```

### Install Development Dependencies
For development work (testing, linting, etc.):
```bash
pip install -r requirements-dev.txt
# Or install with optional dev dependencies
pip install -e ".[dev]"
```

### Using the Package
Once installed, you can import the compressible flow utilities in your Python code:
```python
from modern_aerospace_propulsion import compressible_flow

# Use temperature conversions
T_celsius = compressible_flow.K2C(300)  # Convert 300 K to Celsius

# Use compressible flow relations
theta = compressible_flow.θ(2.0)  # Temperature ratio at Mach 2.0
```

## Repository Structure

This repository contains Jupyter notebooks and analysis tools organized by topic, following the course curriculum structure:

### Week 01 - Introduction
Introduction to aerospace propulsion concepts and foundational models.
- **Altitude_vs_Mach_Number.ipynb** - Visualization of altitude vs Mach number for selected aircraft
- **Cylinder_Blowdown.ipynb** - Model for analyzing cylinder blowdown dynamics
- **Shock_Tube_Model.ipynb** - One-dimensional shock tube simulation

### Week 02 - Fundamental Principles
Core thermodynamic principles applied to propulsion systems.
- **Thermodynamics_of_Gases.ipynb** - Fundamental gas dynamics and thermodynamic relationships

### Week 03 - Cycle Analysis
Analysis of thermodynamic cycles for various propulsion systems.
- **Cycle_Analysis.ipynb** - Thermodynamic cycle analysis for jet engines

### Week 04 - Inlets & Nozzles
Study of inlet and nozzle flows and their performance characteristics.
- **Inlet_Model.ipynb** - Computational model for inlet flow analysis

### Week 05 - Combustors & Augmentors
Combustion modeling and augmentor (afterburner) analysis.
- **Gibbs_Energy_Minimization.ipynb** - Tutorial on Gibbs free energy minimization for equilibrium combustion modeling
- **Homework_2_Combustors_&_Augementors.ipynb** - Homework assignment on combustor and augmentor analysis

### Week 06 - Turbomachinery
Turbomachinery components and performance analysis.
- **Gas_Turbine_T4_Performance_Sensitivity.ipynb** - Analysis of gas turbine performance sensitivity to T4 (turbine inlet temperature)
- **Polytropic_&_Isentropic_Efficiency.ipynb** - Study of polytropic and isentropic efficiencies in turbomachinery

### Week 07 - Advanced Cycles
Advanced propulsion cycles including turbofans and other configurations.
- **SimpleTurboFan.ipynb** - Simplified turbofan engine cycle analysis

### Week 09 - Introduction to Rockets
Rocket propulsion fundamentals and trajectory analysis.
- **RocketPerformance.ipynb** - Rocket performance calculations and analysis
- **LaunchTrajectoryAnalysis.ipynb** - Launch trajectory simulation and optimization

### Week 11 - Nozzle Design & Cooling, Braking, Advanced Concepts
Advanced rocket nozzle design and thermal management.
- **LiquidRocketDesign.ipynb** - Liquid rocket engine design and analysis

### Week 13 - Solid Chemical, Electric & Nuclear Propulsion Systems
Solid rocket motors, electric propulsion, and advanced concepts.
- **Models/SRM_ODE_Model.ipynb** - Solid rocket motor ODE simulation model
- **Models/SRM_ODE_Model_with_Combustion.ipynb** - Enhanced SRM model with combustion chemistry
- **Models/run_FFSC_Model.ipynb** - Full-flow staged combustion cycle analysis with regenerative nozzle cooling

### Aircraft Project Tools
Tools and resources for the aircraft propulsion system design project.
- **Aircraft_Propulsion_System_Design_Project.ipynb** - Comprehensive project notebook with design tools and calculations

### Homework Solutions
Solutions to course homework assignments.
- **Homework_1_Fundamental_Principles.ipynb** - Homework #1 solutions covering fundamental principles
- **Homework_2_Turbojet_Optimization.ipynb** - Homework #2 solutions on turbojet optimization

### Python Package: `modern_aerospace_propulsion`
Reusable Python utilities for aerospace propulsion calculations.

#### Core Modules
- **compressible_flow.py** - Compressible flow relations and temperature conversions

#### `ffsc_nozzle` Sub-package
Full-flow staged combustion (FFSC) cycle and regenerative nozzle cooling analysis tools:
- **thermo.py** - Thermodynamic property calculations with Cantera integration for equilibrium combustion
- **regen.py** - 1D regenerative nozzle cooling model with Bartz correlation and conjugate heat transfer
- **cycle.py** - FFSC cycle modeling including pumps, turbines, and power balance
- **sweep.py** - Parameter sweep utilities for feasibility mapping and optimization

## Using the Notebooks

All notebooks in this repository are designed to be compatible with:
- Google Colab (links included at the top of each notebook)
- Local Jupyter Notebook/JupyterLab installations
- Any standard Python notebook environment

The notebooks include interactive visualizations, computational models, and analysis tools for learning aerospace propulsion concepts.
