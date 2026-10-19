'''
Run randomly generated Shock Tube Problem solved via Euler Solver
- Essentially a copy and paste of the test_Sod_Shock function, but why not
- Also useful for the data_generation code
'''

import numpy as np
from infrastructure.solver_config import EulerConfig
from infrastructure.physics_config import PhysicsConfig
from euler1d.solver import EulerSolver
from infrastructure.results import ExactRiemannSolver, animate_comparison, animate_results




# def random_Run():
#     """
#     Runs Sod Shock Tube problem test case to compare against exact solution.
#     """

#     #NOTE: THESE KEY SOLVER PARAMETERS ARE FIXED ACROSS ALL SIMS
#     N_cells = 1000
#     domain_size = 1.0
#     t_max = 0.0002
#     dt = 1e-6
#     N_ghost = 3
#     gamma = 1.4

#     #NOTE: SCHEME SELECTION HERE (NO-DUH)
#     '''
#     SCHEME SELECTION
#     '''
#     flux_scheme = 'HLLC'
#     reconstruction_scheme = 'WENO5Z'

#     # flux_scheme = 'AUSM+'
#     # reconstruction_scheme = 'WENO5'

#     physics_config = PhysicsConfig(
#         BC='Zero-Gradient',
#         N_cells=N_cells,
#         gamma=gamma
#     )

#     solver_config = EulerConfig(
#         domain_size=domain_size,
#         N_cells=N_cells,
#         IC=physics_config.IC,
#         BC=physics_config.BC,
#         QL=physics_config.Q_L,
#         QR=physics_config.Q_R,
#         x_split_percent=physics_config.domain_split_percent,
#         t_max=t_max,
#         dt=dt,
#         gamma=gamma,
#         N_ghost=N_ghost
#     )

#     print(f"Running EulerSolver with {flux_scheme} & {reconstruction_scheme}...")
#     results = EulerSolver(
#         config=solver_config,
#         flux=flux_scheme,
#         reconstruction=reconstruction_scheme
#     )

#     return results


def random_Run(unit_sample=None, use_Sod=False):    # CHANGED: two optional args
    """
    Runs Sod Shock Tube problem test case to compare against exact solution.
    unit_sample : (u_rho, u_P, u_x) row of a Latin-hypercube design, or None to use the random generators
    use_Sod     : run the canonical Sod problem (scaled to the reference pressure)
    """
 
    #NOTE: THESE KEY SOLVER PARAMETERS ARE FIXED ACROSS ALL SIMS
    N_cells = 1000
    domain_size = 1.0
    t_max = 0.0002
    dt = 1e-7
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
        gamma=gamma,
        domain_size=domain_size,    # NEW: so the CFL / containment checks use the values actually run
        t_max=t_max,                # NEW
        dt=dt,                      # NEW
        unit_sample=unit_sample,    # NEW
        use_Sod=use_Sod             # NEW
    )
 
    solver_config = EulerConfig(
        domain_size=domain_size,
        N_cells=N_cells,
        IC=physics_config.IC,
        BC=physics_config.BC,
        QL=physics_config.Q_L,
        QR=physics_config.Q_R,
        x_split_percent=physics_config.domain_split_percent,
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
    r=random_Run()
    animate_results(r, interval = 5)