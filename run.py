'''
Run randomly generated Shock Tube Problem solved via Euler Solver
- Essentially a copy and paste of the test_Sod_Shock function, but why not
- Also useful for the data_generation code
'''

import numpy as np
from infrastructure.solver_config import EulerConfig
from infrastructure.physics_config import PhysicsConfig
from euler1d.solver import EulerSolver
from infrastructure.results import ExactRiemannSolver, animate_comparison




def random_Run():
    """
    Runs Sod Shock Tube problem test case to compare against exact solution.
    """

    #NOTE: THESE KEY SOLVER PARAMETERS ARE FIXED ACROSS ALL SIMS
    N_cells = 1000
    domain_size = 1.0
    t_max = 0.00075
    dt = 1e-6
    N_ghost = 3
    gamma = 1.4

    #NOTE: SCHEME SELECTION HERE (NO-DUH)
    '''
    SCHEME SELECTION
    '''
    flux_scheme = 'HLLC'
    reconstruction_scheme = 'WENO5Z'

    # flux_scheme = 'AUSM+'
    # reconstruction_scheme = 'WENO5'

    physics_config = PhysicsConfig(
        BC='Zero-Gradient',
        N_cells=N_cells,
        gamma=gamma
    )

    solver_config = EulerConfig(
        domain_size=domain_size,
        N_cells=N_cells,
        IC=physics_config.IC,
        BC=physics_config.BC,
        t_max=t_max,
        dt=dt,
        gamma=gamma,
        N_ghost=N_ghost
    )

    print(f"Running EulerSolver with {flux_scheme} & {reconstruction_scheme}...")
    results = EulerSolver(
        config=solver_config,
        flux=flux_scheme,
        reconstruction=reconstruction_scheme
    )

    return results



if __name__ == '__main__':
    random_Run()