import numpy as np
from infrastructure.solver_config import EulerConfig
from infrastructure.physics_config import PhysicsConfig
from euler1d.solver import EulerSolver
from infrastructure.results import ExactRiemannSolver, animate_comparison


'''
Gemini-Produced Sod Shock Tube Problem
    - Modifications made by pranet
'''


def test_Sod_Shock():
    """
    Runs Sod Shock Tube problem test case to compare against exact solution.
    """
    N_cells = 1000
    domain_size = 1.0
    t_max = 0.001
    dt = 1e-6
    N_ghost = 3
    gamma = 1.4

    '''
    SCHEME SELECTION
    '''

    flux_scheme = 'HLLC'
    reconstruction_scheme = 'WENO5Z'

    # flux_scheme = 'AUSM+'
    # reconstruction_scheme = 'WENO5'

    # Standard Sod Shock Tube problem (or custom initial states)
    rho_L, u_L, P_L = 1.0, 0, 100000.0
    rho_R, u_R, P_R = 0.125, 0, 10000.0

    # Primitive states for Exact Solver
    state_L = (rho_L, u_L, P_L)
    state_R = (rho_R, u_R, P_R)


    physics_config = PhysicsConfig(
        BC='Zero-Gradient',
        N_cells=N_cells,
        gamma=gamma,
        use_Sod=True
    )

    config = EulerConfig(
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
        config=config,
        flux=flux_scheme,
        reconstruction=reconstruction_scheme
    )

    # Instantiate Exact Solver
    exact_solver = ExactRiemannSolver(state_L, state_R, x_interface=domain_size / 2.0, gamma=gamma)

    # Animate
    animate_comparison(results, exact_solver)
    return results


if __name__ == '__main__':
    test_Sod_Shock()