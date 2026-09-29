'''
Solver Configuration Code
- Creates physical 1D Euler Problem [IC, BC, gamma]
    - "Randomly" instantiates a problem
        - Using set seed though to help with reproducability
    - This is specificially to define and generate various physical problem definition
        - For numerical problem setup, go to numerical_config.py
'''
from dataclasses import dataclass, field
import numpy as np


split_generator = np.random.default_rng(seed = 12345)
vel_generator = np.random.default_rng(seed = 42)
dens_generator = np.random.default_rng(seed = 0)
pres_generator = np.random.default_rng(seed = 67)


@dataclass()
class PhysicsConfig:
    '''
    Configuration for a 1D Euler Problem
    - Generatees a 1D Euler Problem with random initial conditions
        - Uniform sampling with fixed seeds for generator
        - ranges arbitrarily chosen
            - values clamped to positivity
            - Bounds of generation ranges arbitraily chosen
        - Does need N_cells from user
    - Does contain nescessary input parameters to exact_Reimann_solver function
    
    ***ATTRIBUTES***
    BC
    N_cells
    gamma
    Q_L, Q_R
    x_split
    IC
    domain_split_percent
    '''

    #boundary conditions
    BC: str = field(default = 'Zero-Gradient') #Zero-Gradient, Reflective, Periodical


    # other parameters
    N_cells: int = field(default = 500) #N cells in domain
    gamma: float = field(default = 1.4) #ratio of specific heats

    use_Sod: bool = field(default = False) #use the Sod shock tube setup



    def __post_init__(self):
        def get_cons(rho, u, P, gamma = self.gamma):
            mom = rho * u
            E = (P / (gamma - 1.0)) + 0.5 * rho * u**2
            return np.array([rho, mom, E])

        rho_L = dens_generator.uniform(0.1, 2.5)
        rho_R = dens_generator.uniform(0.1, 2.5)

        #normal chosen to center around u = 0, like in the sod shock tube setup
        u_L = vel_generator.normal(0., 5.) 
        u_R = vel_generator.normal(0., 5.) 

        P_L = pres_generator.uniform(10000., 125000)
        P_R = pres_generator.uniform(10000., 125000.)


        # Hardcode use Sod Shock Tube
        if self.use_Sod:
            rho_L, u_L, P_L = 1.0, 0, 100000.0
            rho_R, u_R, P_R = 0.125, 0, 10000.0

                    
        # conservative 
        self.Q_L = get_cons(rho_L, u_L, P_L, self.gamma)
        self.Q_R = get_cons(rho_R, u_R, P_R, self.gamma)


        #Init Condition of Conservative State Matrix
        self.IC = np.zeros((3, self.N_cells)) 
        self.x_split = split_generator.integers(int(self.N_cells*0.2), int(self.N_cells*0.8), endpoint = True)
        if self.use_Sod:
            self.x_split = int(self.N_cells*0.5)
        self.IC[:, self.x_split] = self.Q_L
        for i in range(self.N_cells):
            self.IC[:, i] = self.Q_L if i < self.x_split else self.Q_R

        # domain split location in terms of %domain from the left
        self.domain_split_percent = self.x_split/self.N_cells

        
