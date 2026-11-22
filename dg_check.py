'''
Classical DG checks for P1 and P2: smooth convergence, and Sod / Lax / 123 against the exact solution and the FV schemes.

python dg_check.py [--N 200] [--M 0] [--p 1,2]
'''
import argparse
import math
import numpy as np
import torch

from euler1d.dg import solver as dg
from infrastructure.exact import ExactRiemannSolver
from infrastructure.nfv_dataset import from_states
from euler1d.neural import rollouts as ro

# (left, right) primitive states, domain length, final time (None: 100 FV steps at CFL 0.5)
PROBLEMS = {'sod': (((1.0, 0.0, 1e5), (0.125, 0.0, 1e4)), 0.4, None),
            'lax': (((0.445, 0.698, 3.528), (0.5, 0.0, 0.571)), 1.0, 0.14),
            '123': (((1.0, -2.0, 0.4), (1.0, 2.0, 0.4)), 1.0, 0.15)}


def density_wave(x, t=0.0):
    rho = 1 + 0.2 * np.sin(2 * np.pi * (x - t))
    return dg.cons(rho, np.ones_like(x), np.ones_like(x))


def convergence(p, flux='hllc', M=None, Ns=(10, 20, 40, 80), T=0.5):
    '''advected density wave, periodic [0, 1]; M None = no limiter. L1 density errors and rates'''
    errs = []
    xq, wq = dg.gauss(8)
    for N in Ns:
        e = np.linspace(0, 1, N + 1)
        U = dg.project(density_wave, e, p + 1)[None]
        U, _ = dg.solve(U, 1 / N, T, flux=flux, bc=dg.periodic, M=M or 0.0, limit=M is not None)
        ref = dg.project(lambda x: density_wave(x, T), e, p + 1)[None]
        errs.append(((dg.at(U, xq) - dg.at(ref, xq))[0, 0].abs() * wq).sum().item() / N)
    return errs, [math.log2(a / b) for a, b in zip(errs, errs[1:])]


def riemann(name, N, p, flux='hllc', M=0.0):
    (pl, pr), Lx, T = PROBLEMS[name]
    ex = ExactRiemannSolver(pl, pr, 0.5 * Lx, dg.GAMMA)
    dx = Lx / N
    T = T or 100 * 0.5 * dx / ex.smax()
    e = np.linspace(0, Lx, N + 1)
    U, n = dg.solve(dg.project_exact(ex, e, 0.0, p + 1)[None], dx, T, flux=flux, M=M)
    return dg.l1_density(U, ex, e, T), T, n


def fv(name, N):
    (pl, pr), Lx, T = PROBLEMS[name]
    ex = ExactRiemannSolver(pl, pr, 0.5 * Lx, dg.GAMMA)
    dx = Lx / N
    T = T or 100 * 0.5 * dx / ex.smax()
    nt = max(1, round(T / (0.5 * dx / ex.smax())))
    d = from_states(pl, pr, 0.5, N, nt, dx, 0.5, G=3, dtype=np.float64, dt=T / nt)
    d['bc'] = 'zero'
    out = ro.predict_all(d, ['HLLC-1', 'HLLC-WENO5Z'])
    ref = dg.project_exact(ex, np.linspace(0, Lx, N + 1), T, 1)[0, :, 0].numpy()
    return {k: float(dx * np.abs(P[0, -1, 0] - ref).sum()) for k, P in out.items()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--N', type=int, default=200)
    ap.add_argument('--M', type=float, default=0.0)
    ap.add_argument('--p', default='1,2')
    a = ap.parse_args(argv)
    ps = [int(x) for x in a.p.split(',')]
    print('Smooth density wave, periodic, T = 0.5, HLLC: L1 density error, rates')
    for p in ps:
        for M, lab in ((None, 'no limiter'), (100.0, 'TVB M=100')):
            errs, rates = convergence(p, M=M)
            print(f'  P{p} {lab:11s} ' + ' '.join(f'{e:.2e}' for e in errs) + '   rates ' + ' '.join(f'{r:.2f}' for r in rates))
    print(f'\nRiemann problems, N = {a.N} cells (P1 has 2 and P2 3 unknowns per cell), L1 density of the means, M = {a.M}')
    for name in PROBLEMS:
        row = fv(name, a.N)
        for p in ps:
            for flux in ('rusanov', 'hllc'):
                row[f'DG-P{p} {flux}'] = riemann(name, a.N, p, flux, a.M)[0]
        print(f'  {name}: ' + '  '.join(f'{k} {v:.3e}' for k, v in row.items()))


if __name__ == '__main__':
    main()
