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

### Week 1 - Introduction
Introduction to aerospace propulsion concepts and foundational models.
- **Altitude_vs_Mach_Number.ipynb** - Visualization of altitude vs Mach number for selected aircraft
- **Cylinder_Blowdown.ipynb** - Model for analyzing cylinder blowdown dynamics
- **Shock_Tube_Model.ipynb** - One-dimensional shock tube simulation

### Week 2 - Fundamental Principles
Core thermodynamic principles applied to propulsion systems.
- **Thermodynamics_of_Gases.ipynb** - Fundamental gas dynamics and thermodynamic relationships

### Week 3 - Cycle Analysis
Analysis of thermodynamic cycles for various propulsion systems.
- **Cycle_Analysis.ipynb** - Thermodynamic cycle analysis for jet engines

### Week 4 - Inlets & Nozzles
Study of inlet and nozzle flows and their performance characteristics.
- **Inlet_Model.ipynb** - Computational model for inlet flow analysis

### Week 5 - Combustors & Augmentors
Combustion modeling and augmentor (afterburner) analysis.
- **Gibbs_Energy_Minimization.ipynb** - Tutorial on Gibbs free energy minimization for equilibrium combustion modeling

### Aircraft Project Tools
Tools and resources for the aircraft propulsion system design project.
- **Aircraft_Propulsion_System_Design_Project.ipynb** - Comprehensive project notebook with design tools and calculations

### Homework Solutions
Solutions to course homework assignments.
- **Homework_1_Fundamental_Principles.ipynb** - Homework #1 solutions covering fundamental principles
- **Homework_2_Turbojet_Optimization.ipynb** - Homework #2 solutions on turbojet optimization

### Python Package: `modern_aerospace_propulsion`
Reusable Python utilities for aerospace propulsion calculations.
- **compressible_flow.py** - Compressible flow relations and temperature conversions

## Using the Notebooks

All notebooks in this repository are designed to be compatible with:
- Google Colab (links included at the top of each notebook)
- Local Jupyter Notebook/JupyterLab installations
- Any standard Python notebook environment

The notebooks include interactive visualizations, computational models, and analysis tools for learning aerospace propulsion concepts.
