# Handy compressible flow relations
γ1 = lambda γ: γ / (γ-1) 

# Ratio of total to static temperature
def Tt_T(Mach: float, γ: float = 1.4) -> float:
  """Calculate and return the total to static temperature ratio given
    **Mach** (*float*): the Mach number, and
    **γ** (*float*): the ratio of specific heats (default=1.4)"""
  return θ(Mach, γ) 

def θ(Mach: float, γ: float = 1.4) -> float:
  """Calculate and return the total to static temperature ratio given
    **Mach** (*float*): the Mach number, and
    **γ** (*float*): the ratio of specific heats (default=1.4)"""
  return 1 + (γ-1)/2*Mach**2

# Ratio of total to static pressure
def Pt_P(Mach: float, γ: float = 1.4) -> float:
  """Calculate and return the total to static pressure ratio given
    **Mach** (*float*): the Mach number, and
    **γ** (*float*): the ratio of specific heats (default=1.4)"""
  return δ(Mach, γ)

def δ(Mach: float, γ: float = 1.4) -> float:
  """Calculate and return the total to static pressure ratio given
  **Mach** (*float*): the Mach number, and
  **γ** (*float*): the ratio of specific heats (default=1.4)"""
  return θ(Mach, γ)**(γ1(γ))

# Nozzle area ratio
def A_Astar(Mach: float, γ: float=1.4) -> float:
  """Calculate and return the nozzle area ratio given
    **Mach** (*float*): the Mach number, and
    **γ** (*float*): the ratio of specific heats (default=1.4)"""
  return 1 / Mach * (2/(γ+1)*θ(Mach, γ))**((γ+1)/2/(γ-1))

# Handy temperature conversions 

# Kelvin to Celsius
K2C = lambda T_K: T_K - 273.15

# Celsius to Kelvin
C2K = lambda T_C: T_C + 273.15
