'''
Exact-reference dataset for supervised NFV training.

- Random Riemann problems, same ranges as PhysicsConfig (rho, u, P),
  interface location uniform in [0.2, 0.8] of the window (continuous)
- Exact conservative cell averages at t = k*dt, k = 0..nt
- Each cell array includes G ghost cells per side taken from the exact solution
- dt per sample = cfl * dx / smax, so the CFL is the same for every sample
- Separate random streams for train / eval / test

Q: (B, nt+1, 3, nx + 2G)
'''
import argparse
import os
import time
import numpy as np
from infrastructure.exact import ExactRiemannSolver

RHO = (0.1, 2.5)
U_STD = 5.0
P = (1e4, 1.25e5)
XF = (0.2, 0.8)
SPLITS = {'train': 0, 'eval': 1, 'test': 2}
STAGES = [(10, 10), (50, 50), (100, 100), (200, 200)]
S_GLOBAL = 1500.0  # >= max wave speed over the sampling ranges (measured max 1486 m/s)


def sample_specs(n, rng):
    def side():
        return np.stack([rng.uniform(*RHO, n), rng.normal(0.0, U_STD, n), rng.uniform(*P, n)], axis=1)
    pl, pr = side(), side()
    xf = rng.uniform(*XF, n)
    return pl, pr, xf


def from_states(pl, pr, xf, nx, nt, dx, cfl=0.5, G=3, sub=10, gamma=1.4, dtype=np.float32, dt=None):
    pl, pr, xf = np.atleast_2d(pl), np.atleast_2d(pr), np.atleast_1d(xf)
    n = len(xf)
    edges = (np.arange(nx + 2 * G + 1) - G) * dx
    Q = np.empty((n, nt + 1, 3, nx + 2 * G), dtype=dtype)
    dts, sm = np.empty(n), np.empty(n)
    for i in range(n):
        ex = ExactRiemannSolver(pl[i], pr[i], xf[i] * nx * dx, gamma)
        sm[i] = ex.smax()
        dts[i] = cfl * dx / sm[i] if dt is None else dt
        Q[i] = ex.cell_avg_t(edges, np.arange(nt + 1) * dts[i], sub)
    return dict(Q=Q, dt=dts, smax=sm, pl=pl, pr=pr, xf=xf,
                dx=dx, G=G, cfl=cfl, nx=nx, nt=nt, gamma=gamma)


def make_set(n, nx, nt, dx, split='train', seed=0, cfl=0.5, G=3, sub=10, gamma=1.4, dtype=np.float32,
             dt_mode='per_sample'):
    pl, pr, xf = sample_specs(n, np.random.default_rng([seed, SPLITS[split]]))
    dt = cfl * dx / S_GLOBAL if dt_mode == 'global' else None
    d = from_states(pl, pr, xf, nx, nt, dx, cfl, G, sub, gamma, dtype, dt)
    return dict(d, split=split, seed=seed, dt_mode=dt_mode)


def fname(split, nx, nt, dt_mode='per_sample'):
    return f'{split}_nx{nx}_nt{nt}' + ('_gdt' if dt_mode == 'global' else '') + '.npz'


def split_ghost(d):
    Q, G = d['Q'], d['G']
    return Q[..., G:-G], Q[..., :G], Q[..., -G:]


def save_set(path, d):
    np.savez(path, **d)


def load_set(path):
    with np.load(path) as f:
        return {k: (f[k].item() if f[k].ndim == 0 else f[k]) for k in f.files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='generated_nfv')
    ap.add_argument('--n_train', type=int, default=2048)
    ap.add_argument('--n_eval', type=int, default=100)
    ap.add_argument('--n_test', type=int, default=100)
    ap.add_argument('--eval_nx', type=int, default=200)
    ap.add_argument('--eval_nt', type=int, default=400)
    ap.add_argument('--dx', type=float, default=1e-3)
    ap.add_argument('--cfl', type=float, default=0.5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--dt_mode', default='per_sample', choices=['per_sample', 'global'])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    jobs = [('train', a.n_train, nx, nt) for nx, nt in STAGES]
    jobs += [('eval', a.n_eval, a.eval_nx, a.eval_nt), ('test', a.n_test, a.eval_nx, a.eval_nt)]
    for split, n, nx, nt in jobs:
        t0 = time.time()
        d = make_set(n, nx, nt, a.dx, split, a.seed, a.cfl, dt_mode=a.dt_mode)
        path = os.path.join(a.out, fname(split, nx, nt, a.dt_mode))
        save_set(path, d)
        print(f'{path}  n={n}  {d["Q"].nbytes / 1e6:.0f} MB  {time.time() - t0:.1f}s')


if __name__ == '__main__':
    main()