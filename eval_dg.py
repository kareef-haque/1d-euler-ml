'''
Evaluate a learned DG flux: test split, named problems (Sod / Lax / 123), smooth convergence order.

python eval_dg.py --model runs/dg_p1_scalar/model_final.pt [--out results.json] [--plot sod.png]
Lax and 123 are in nondimensional units, far outside the training ranges (SI, p ~ 1e4-1e5): extrapolation tests.
'''
import argparse
import json
import os
import numpy as np
import torch

import dg_check as dc
from euler1d.dg import data as dd, solver as dg
from euler1d.dg.flux import DissFlux
from infrastructure.exact import ExactRiemannSolver


def test_split(m, a):
    d = dd.get_set(a.data_dir, 'test', a.n_test, a.test_nx, a.test_nt, a.dx, m.p, a.data_seed)
    td = dd.tensors(d)
    with torch.no_grad():
        e = {'learned': dd.mean_err(dd.rollout(td, m), td), 'DG-HLLC': dd.mean_err(dd.rollout(td, 'hllc'), td),
             'DG-Rusanov': dd.mean_err(dd.rollout(td, 'rusanov'), td),
             'FV HLLC-1': dd.mean_err(dd.fv_first_order(d)[..., None], td)}
    out = {}
    for k, v in e.items():
        ok = torch.isfinite(v)
        out[k] = dict(err=v[ok].mean().item(), median=v[ok].median().item(), bad=int((~ok).sum()),
                      ratio_fv=(e['FV HLLC-1'][ok].mean() / v[ok].mean()).item(),
                      win_vs_hllc=(v < e['DG-HLLC']).double().mean().item())
    return out


def named(m, N):
    out = {}
    for name in dc.PROBLEMS:
        (pl, pr), Lx, T = dc.PROBLEMS[name]
        ex = ExactRiemannSolver(pl, pr, 0.5 * Lx, dg.GAMMA)
        dx = Lx / N
        T = T or 100 * 0.5 * dx / ex.smax()
        e = np.linspace(0, Lx, N + 1)
        row = {}
        for k, f in (('learned', m), ('DG-HLLC', 'hllc'), ('DG-Rusanov', 'rusanov')):
            try:
                with torch.no_grad():
                    U, _ = dg.solve(dg.project_exact(ex, e, 0.0, m.p + 1)[None], dx, T, flux=f)
                row[k] = dg.l1_density(U, ex, e, T)
            except (RuntimeError, torch.linalg.LinAlgError):      # blow-up: non-physical means, singular eigenvectors
                row[k] = float('inf')
            if not np.isfinite(row[k]):
                row[k] = float('inf')
        out[name] = row
    return out


def convergence(m):
    with torch.no_grad():
        res = {}
        for k, f in (('learned', m), ('DG-HLLC', 'hllc')):
            errs = []
            for N in (10, 20, 40):
                e = np.linspace(0, 1, N + 1)
                U = dg.project(dc.density_wave, e, m.p + 1)[None]
                U, _ = dg.solve(U, 1 / N, 0.25, flux=f, bc=dg.periodic, limit=False)
                ref = dg.project(lambda x: dc.density_wave(x, 0.25), e, m.p + 1)[None]
                xq, wq = dg.gauss(8)
                errs.append(((dg.at(U, xq) - dg.at(ref, xq))[0, 0].abs() * wq).sum().item() / N)
            res[k] = dict(errs=errs, rate=float(np.log2(errs[-2] / errs[-1])))
    return res


def plot_sod(m, N, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    (pl, pr), Lx, _ = dc.PROBLEMS['sod']
    ex = ExactRiemannSolver(pl, pr, 0.5 * Lx, dg.GAMMA)
    dx = Lx / N
    T = 100 * 0.5 * dx / ex.smax()
    e = np.linspace(0, Lx, N + 1)
    xc = 0.5 * (e[1:] + e[:-1])
    fig, ax = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    xs = np.linspace(0, Lx, 3000)
    rho, u, p = ex.sample(xs, T)
    for j, (lab, v) in enumerate((('density', rho), ('velocity', u), ('pressure', p))):
        ax[j].plot(xs, v, 'k-', lw=1.2, label='exact')
    for k, f, st in (('DG-HLLC', 'hllc', '--'), ('learned', m, '-')):
        with torch.no_grad():
            U, _ = dg.solve(dg.project_exact(ex, e, 0.0, m.p + 1)[None], dx, T, flux=f)
        Q = U[0, ..., 0].numpy()
        r_, u_ = Q[0], Q[1] / Q[0]
        for j, v in enumerate((r_, u_, 0.4 * (Q[2] - 0.5 * Q[1] * u_))):
            ax[j].plot(xc, v, st, lw=1.2, label=k)
    for j, lab in enumerate(('density', 'velocity', 'pressure')):
        ax[j].set_ylabel(lab)
    ax[0].legend()
    ax[0].set_title(f'Sod, DG-P{m.p} cell means, N={N}, t={T:.2e} s')
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--n_test', type=int, default=100)
    ap.add_argument('--test_nx', type=int, default=200)
    ap.add_argument('--test_nt', type=int, default=0)
    ap.add_argument('--named_N', type=int, default=200)
    ap.add_argument('--dx', type=float, default=1e-3)
    ap.add_argument('--data_seed', type=int, default=0)
    ap.add_argument('--data_dir', default='generated_dg')
    ap.add_argument('--out', default=None)
    ap.add_argument('--plot', default=None)
    a = ap.parse_args(argv)
    m = DissFlux.load(a.model)
    a.test_nt = a.test_nt or round(60 * 0.3 / dd.default_cfl(m.p))
    res = dict(model=a.model, p=m.p, mode=m.mode, test=test_split(m, a), named=named(m, a.named_N), convergence=convergence(m))
    print(f'P{m.p} {m.mode}: test split ({a.n_test} x {a.test_nt} steps, nx {a.test_nx}), scaled MSE of the means')
    for k, v in res['test'].items():
        print(f"  {k:10s} err {v['err']:.3e} (median {v['median']:.2e}) x{v['ratio_fv']:.2f} vs FV, "
              f"wins vs DG-HLLC {100 * v['win_vs_hllc']:.0f}%, bad {v['bad']}")
    print(f'named problems, N = {a.named_N}, L1 density error (Lax, 123: far outside the training ranges)')
    for k, row in res['named'].items():
        print(f'  {k}: ' + '  '.join(f'{n} {v:.3e}' for n, v in row.items()))
    print('smooth density wave, no limiter: rate ' + '  '.join(f"{k} {v['rate']:.2f}" for k, v in res['convergence'].items()))
    if a.plot:
        plot_sod(m, a.named_N, a.plot)
    out = a.out or os.path.join(os.path.dirname(a.model), 'eval.json')
    json.dump(res, open(out, 'w'), indent=1)
    print('written', out)
    return res


if __name__ == '__main__':
    main()
