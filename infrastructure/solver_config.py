'''
Solver Configuration Code
- Defines domain and initial conditions as EulerConfig class
    - User must instantiate an instance and feed it to solver
    - This is specificially to feed into the numerical solver
        - For physical problem setup, go to physics_config.py
'''
from dataclasses import dataclass, field
import numpy as np

@dataclass()
class EulerConfig:
    '''
    Configuration for the 1D Euler Equation Solver
    - User must instantiate an instance and feed it to solver
    '''
    #define domain
    domain_size: float #m
    N_cells: int #number of cells

    #initial conditions
    IC: np.ndarray #(3, N_cells) Initial Condition of Conservative State Matrix

    #boundary conditions
    BC: str = field(default = 'Zero-Gradient') #Zero-Gradient, Reflective, Periodical
    N_ghost: int = field(default = 3) #number of ghost cells


    # other parameters
    gamma: float = field(default = 1.4) #ratio of specific heats
    t_max: float = field(default = 10) #s
    dt: float = field(default = 0.01) #s



    def __post_init__(self):
        self.dx = self.domain_size / self.N_cells  # m
        self.IC = np.array(self.IC, dtype=float)  #own copy, caller's array is not modified
        self.IC[0, :] = np.maximum(self.IC[0, :], 1e-9) #density floor
        self.IC[2, :] = np.maximum(self.IC[2, :], 1e-9) #total energy floor



#useful conversion
def cons_to_prim(Q, gamma = 1.4):
    '''
    Convert conservative Q vector to primitive R vector
    '''
    rho = Q[0, :]
    u = Q[1, :]/rho
    P = (gamma - 1) * (Q[2, :] - 0.5 * rho * u**2) #Equation of State
    return np.array([rho, u, P])
def prim_to_cons(R, gamma = 1.4):
    '''
    Convert primitive R vector to conservative Q vector
    '''
    rho = R[0]
    mom = rho * R[1]
    E = R[2]/(gamma - 1) + 0.5 * rho * R[1]**2
    return np.array([rho, mom, E])