'''
1D Euler Equation Solver

- Flux:            HLLC, AUSM+, Rusanov
- Reconstruction:  WENO5Z, WENO5, Const (first order)
- Flux 'NFV':      learned flux (euler1d/neural), maps the padded state to fluxes directly
- Time stepping:   rk4, or euler (forward Euler; default for NFV)
'''

from datetime import datetime
from euler1d.boundary import apply_BC
from infrastructure.solver_config import EulerConfig
from infrastructure.results import EulerResults

from euler1d.schemes.flux.hllc import HLLC
from euler1d.schemes.flux.ausm_plus import AUSMp
from euler1d.schemes.flux.rusanov import Rusanov

from euler1d.schemes.reconstruction.weno5Z import WENO5Z
from euler1d.schemes.reconstruction.weno5 import WENO5
from euler1d.schemes.reconstruction.constant import Const

FLUX = {'HLLC': HLLC, 'AUSM+': AUSMp, 'AUSMp': AUSMp, 'Rusanov': Rusanov}
REC = {'WENO5Z': WENO5Z, 'WENO5': WENO5, 'Const': Const}


def EulerSolver(config: EulerConfig,
                flux = 'HLLC',
                reconstruction = 'WENO5Z',
                integrator = None,
                nfv = None,
                verbose = True
                ):
    '''
    :param EulerConfig config: Configuration for the problem
    :param str flux: Flux Scheme (HLLC, AUSM+, Rusanov, NFV)
    :param str reconstruction: Reconstruction Method (WENO5Z, WENO5, Const); ignored for NFV
    :param str integrator: rk4 or euler (default: euler for NFV, rk4 otherwise)
    :param nfv: checkpoint path or NFVFlux, required when flux == 'NFV'
    :param bool verbose: print progress

    :return EulerResults: Q_hist, F_hist, t_hist (t_hist[k] is the time of Q_hist[k])
    '''
    dx, t_max, dt, gamma = config.dx, config.t_max, config.dt, config.gamma
    BC, N_ghost = config.BC, config.N_ghost

    integrator = integrator or ('euler' if flux == 'NFV' else 'rk4')
    if integrator not in ('rk4', 'euler'):
        raise ValueError("Invalid Integrator")

    if flux == 'NFV':
        from euler1d.neural.adapter import NFVFlux
        if nfv is None:
            raise ValueError("flux='NFV' requires nfv (checkpoint path or NFVFlux)")
        nfv = nfv if isinstance(nfv, NFVFlux) else NFVFlux(nfv)
        g = nfv.g
        if N_ghost < g:
            raise ValueError(f'NFV needs N_ghost >= {g}')
        if abs(nfv.gamma - gamma) > 1e-12:
            raise ValueError('gamma differs from the one the model was trained with')

        def calc_Flux(Q):
            Q_num = apply_BC(Q=Q, BC=BC, ghost_N=N_ghost)
            return nfv(Q_num[:, N_ghost - g:Q_num.shape[1] - N_ghost + g], dx=dx, gamma=gamma)
    else:
        if flux not in FLUX:
            raise ValueError("Invalid Flux Scheme")
        if reconstruction not in REC:
            raise ValueError("Invalid Reconstruction Scheme")
        F_scheme, R_scheme = FLUX[flux], REC[reconstruction]

        def calc_Flux(Q):
            Q_num = apply_BC(Q=Q, BC=BC, ghost_N=N_ghost)
            Q_L, Q_R = R_scheme(Q_num, dx=dx)
            return F_scheme(Q_L, Q_R, dx=dx, gamma=gamma)

    def rhs(Q):
        F = calc_Flux(Q)
        return -(F[:, 1:] - F[:, :-1]) / dx

    def advance(Q, h):
        if integrator == 'euler':
            return Q + h * rhs(Q)
        k1 = rhs(Q)
        k2 = rhs(Q + h * k1 / 2)
        k3 = rhs(Q + h * k2 / 2)
        k4 = rhs(Q + h * k3)
        return Q + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    Q0 = config.IC.copy()
    Q_hist, F_hist, t_hist = [Q0], [calc_Flux(Q0)], [0.0]

    t, n = 0.0, 0
    t0 = datetime.now()
    while t_max - t > 1e-9 * dt:
        h = min(dt, t_max - t)
        Q = advance(Q_hist[-1], h)
        t += h
        n += 1
        Q_hist.append(Q)
        F_hist.append(calc_Flux(Q))
        t_hist.append(t)
        if verbose and n % 10 == 0:
            el = (datetime.now() - t0).total_seconds()
            print(f"Progress: {n} | iterations ({el:.3f}s) | Sim Time ({t:.6f}s): ")

    if verbose:
        print("Simulation Complete")
    return EulerResults(Q_hist, F_hist, t_hist, config)