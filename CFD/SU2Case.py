import os
import subprocess
import pandas as pd
import pyvista as pv
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
import yaml


class SU2Case:
    """
    A helper class to run SU2_CFD cases, extract results, 
    and visualize flow fields or surface data.
    """

    def __init__(self, case_dir, su2_exec="SU2_CFD"):
        self.case_dir = Path(case_dir)
        self.su2_exec = su2_exec
        self.cfg_path = None
        self.last_run_successful = False

        if not self.case_dir.exists():
            raise ValueError(f"Case directory {case_dir} does not exist.")

    # -------------------------------------------------
    # Configuration file utilities
    # -------------------------------------------------

    def load_config(self, cfg_filename):
        """Load SU2 .cfg file into memory."""
        cfg_file = self.case_dir / cfg_filename
        if not cfg_file.exists():
            raise FileNotFoundError(f"Config file {cfg_file} not found.")
        self.cfg_path = cfg_file
        print(f"Loaded config: {cfg_file}")

    def edit_config(self, updates_dict, new_filename=None):
        """
        Update config values. updates_dict = {"MACH_NUMBER": 0.8}
        If new_filename is provided, write to a new file.
        """
        if self.cfg_path is None:
            raise RuntimeError("Load a config first via load_config()")

        cfg_new = []    
        with open(self.cfg_path, "r") as f:
            for line in f:
                key = line.split("=")[0].strip()
                if key in updates_dict:
                    val = updates_dict[key]
                    cfg_new.append(f"{key}= {val}\n")
                else:
                    cfg_new.append(line)

        if new_filename:
            out_path = self.case_dir / new_filename
        else:
            out_path = self.cfg_path

        with open(out_path, "w") as f:
            f.writelines(cfg_new)

        self.cfg_path = out_path
        print(f"Updated config written to {out_path}")

    # -------------------------------------------------
    # Running SU2
    # -------------------------------------------------

    def run(self):
        """Run SU2_CFD on the loaded .cfg file."""
        if self.cfg_path is None:
            raise RuntimeError("No config loaded. Use load_config().")

        print(f"Running SU2: {self.cfg_path}")
        result = subprocess.run(
            [self.su2_exec, str(self.cfg_path)],
            cwd=self.case_dir,
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            self.last_run_successful = True
            print("SU2 run completed successfully.")
        else:
            print("SU2 run failed:")
            print(result.stderr)
            self.last_run_successful = False

        return result

    # -------------------------------------------------
    # Data Extraction
    # -------------------------------------------------

    def load_history(self, filename="history.csv"):
        """Load iteration history into a pandas DataFrame."""
        hist_path = self.case_dir / filename
        if not hist_path.exists():
            raise FileNotFoundError(f"No history file found at {hist_path}")
        return pd.read_csv(hist_path)

    def load_surface(self, filename="surface_flow.csv"):
        """Load surface flow data."""
        surf_path = self.case_dir / filename
        if not surf_path.exists():
            raise FileNotFoundError(f"No surface file found at {surf_path}")
        return pd.read_csv(surf_path)

    # -------------------------------------------------
    # Visualization Utilities
    # -------------------------------------------------

    # ----- 1. Plot history -----

    def plot_history(self, coeffs=("CL", "CD")):
        """Plot CL, CD, or any coefficients from history.csv."""
        df = self.load_history()
        plt.figure(figsize=(10,6))
        for c in coeffs:
            if c in df.columns:
                plt.plot(df["Iteration"], df[c], label=c)

        plt.xlabel("Iteration")
        plt.ylabel("Coefficient")
        plt.title("Coefficient History")
        plt.grid(True)
        plt.legend()
        plt.show()

    # ----- 2. Plot Cp distribution -----

    def plot_surface_cp(self):
        """Plot Cp over x for the surface data."""
        df = self.load_surface()
        plt.figure(figsize=(10,5))
        plt.scatter(df["x"], df["Cp"], s=15)
        plt.gca().invert_yaxis()
        plt.xlabel("x")
        plt.ylabel("Cp")
        plt.title("Surface Cp Distribution")
        plt.grid(True)
        plt.show()

    # ----- 3. Visualize flow field with PyVista -----

    def visualize_flow(self, filename="solution_flow.vtu", field="Pressure"):
        """Visualize VTK flow field using PyVista."""
        vtu_path = self.case_dir / filename
        if not vtu_path.exists():
            raise FileNotFoundError(f"{vtu_path} not found (did you set OUTPUT_FORMAT=PARAVIEW?).")

        mesh = pv.read(vtu_path)
        p = pv.Plotter()
        p.add_mesh(mesh, scalars=field, cmap="viridis")
        p.show()

    def slice_plot(self, filename="solution_flow.vtu", field="Mach", normal=(1,0,0), origin=None):
        """Create a slice through the flow field."""
        vtu_path = self.case_dir / filename
        if not vtu_path.exists():
            raise FileNotFoundError(f"{vtu_path} not found.")

        mesh = pv.read(vtu_path)
        if origin is None:
            origin = mesh.center

        slice_plane = mesh.slice(normal=normal, origin=origin)
        p = pv.Plotter()
        p.add_mesh(slice_plane, scalars=field, cmap="plasma")
        p.show()

    def streamlines(self, filename="solution_flow.vtu", seed_radius=0.1, field="Velocity"):
        """Generate streamlines using the `Velocity` vector field."""
        vtu_path = self.case_dir / filename
        if not vtu_path.exists():
            raise FileNotFoundError(f"{vtu_path} not found.")

        mesh = pv.read(vtu_path)
        stream = mesh.streamlines('Velocity', source_center=mesh.center, radius=seed_radius)

        p = pv.Plotter()
        p.add_mesh(stream, line_width=2)
        p.show()

