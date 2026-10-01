'''
Multi-discontinuity piecewise-constant problems with a fine-grid numerical reference
(HLLC + WENO5Z + RK4 on ref x finer cells, zero-gradient boundaries, cell-averaged back).
The reference is numerical, not exact. Discontinuities sit in the middle 40% of the window
so waves stay inside for the default horizon.
'''
import numpy as np

from infrastructure.nfv_dataset import RHO, U_STD, P
from euler1d.neural import rollouts as ro


def piecewise_ic(pc, pos, nx, dx):
    B, k, _ = pc.shape
    e = np.arange(nx + 1) * dx
    cuts = np.concatenate([np.zeros((B, 1)), pos, np.full((B, 1), nx * dx)], axis=1)
    lo = np.maximum(e[None, :-1, None], cuts[:, None, :-1])
    hi = np.minimum(e[None, 1:, None], cuts[:, None, 1:])
    w = np.clip(hi - lo, 0.0, None) / dx
    return np.einsum('bnk,bkc->bcn', w, pc)


def multi_set(n, nx, nt, dx, k=4, seed=0, cfl=0.3, ref=8, G=3, gamma=1.4):
    rng = np.random.default_rng([seed, 3])
    rho = rng.uniform(*RHO, (n, k))
    u = rng.normal(0.0, U_STD, (n, k))
    p = rng.uniform(*P, (n, k))
    pos = np.sort(rng.uniform(0.3, 0.7, (n, k - 1)), axis=1) * nx * dx
    pc = np.stack([rho, rho * u, p / (gamma - 1) + 0.5 * rho * u**2], axis=-1)
    smax = (np.abs(u) + np.sqrt(gamma * p / rho)).max(axis=1)
    dt = cfl * dx / smax
    step = ro.classical_step('HLLC', 'WENO5Z', 'rk4', dx / ref, gamma, ro.make_pad('zero'))
    Qf = ro.run(piecewise_ic(pc, pos, nx * ref, dx / ref), dt / ref, dx / ref, nt * ref, step, stride=ref)
    if not np.isfinite(Qf).all():
        raise RuntimeError('reference run is not finite')
    Q = Qf.reshape(n, nt + 1, 3, nx, ref).mean(axis=-1)
    Q = np.concatenate([np.repeat(Q[..., :1], G, -1), Q, np.repeat(Q[..., -1:], G, -1)], axis=-1)
    return dict(Q=Q, dt=dt, smax=smax, dx=dx, G=G, cfl=cfl, nx=nx, nt=nt, gamma=gamma, bc='zero',
                pc=pc, pos=pos)