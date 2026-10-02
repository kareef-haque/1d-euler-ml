'''
Sod shock tube: plot and error table for several NFV checkpoints (e.g. ablation variants).

python plot_sod_ablation.py "ablation/*/model_final.pt" --out sod_ablation.png
Label = run folder name (<variant>_s<seed>). Patterns are expanded here, so quote them.
'''
import argparse
import glob
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from infrastructure.nfv_dataset import from_states
from infrastructure.exact import ExactRiemannSolver
from euler1d.neural.model import NFVModel
from euler1d.neural import rollouts as ro
import sod_plateaus as sp

CLASSICAL = ['HLLC-1', 'HLLC-WENO5Z']


def prim(P):
    rho, u = P[0], P[1] / P[0]
    return rho, u, 0.4 * (P[2] - 0.5 * P[0] * u**2)


def labels_for(paths):
    lab = {}
    for p in paths:
        parent, stem = os.path.basename(os.path.dirname(os.path.abspath(p))), os.path.splitext(os.path.basename(p))[0]
        lab[p] = parent if stem in ('model_final', 'model_best') else f'{parent}/{stem}'
    if len(set(lab.values())) < len(lab):
        lab = {p: f'{os.path.basename(os.path.dirname(os.path.abspath(p)))}/{os.path.splitext(os.path.basename(p))[0]}'
               for p in paths}
    return lab


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('models', nargs='+')
    ap.add_argument('--out', default='sod_ablation.png')
    ap.add_argument('--nx', type=int, default=400)
    ap.add_argument('--nt', type=int, default=100)
    ap.add_argument('--dx', type=float, default=1e-3)
    ap.add_argument('--xlim', type=float, nargs=2, default=None, help='zoom, e.g. --xlim 0.15 0.26')
    ap.add_argument('--no_classical', action='store_true')
    a = ap.parse_args(argv)

    paths = sorted(dict.fromkeys(p for pat in a.models for p in (glob.glob(pat) or [pat])))
    lab = labels_for(paths)
    models = {lab[p]: NFVModel.load(p) for p in paths}
    G = max([3] + [m.g for m in models.values()])
    d = from_states(sp.SOD[0], sp.SOD[1], 0.5, a.nx, a.nt, a.dx, 0.5, G=G, dtype=np.float64)
    ex = ExactRiemannSolver(sp.SOD[0], sp.SOD[1], 0.5 * a.nx * a.dx, 1.4)
    t = a.nt * d['dt'][0]

    sols = {}
    if not a.no_classical:
        sols.update({n: P[0, -1] for n, P in ro.predict_all(d, CLASSICAL).items()})
    for name, m in models.items():
        sols[name] = ro.predict_all(d, ['NFV'], m)['NFV'][0, -1]

    w = max(len(n) for n in sols) + 1
    print(f"{'run':{w}s} {'params':>7s} {'rho L1':>9s} | {'rho*L':>7s} {'rho*R':>7s} {'u*L':>7s} {'u*R':>7s} {'p*L':>7s} {'p*R':>7s}"
          f" | RH shock mom  RH contact energy   (plateau errors vs exact; RH residual / flux scale)")
    for name, P in sols.items():
        r = sp.measure(P, ex, t, a.dx)
        l1 = np.abs(P[0] - ex.cell_avg(np.arange(a.nx + 1) * a.dx, t)[0]).mean() / sp.SOD[0][0]
        pc = ' '.join(f'{100 * r[s][k]:+6.1f}%' for k in ('rho', 'u', 'p') for s in 'LR')
        if name in models:
            rh = sp.rh_residual(sp.nfv_diag_flux(models[name]))
            tail = f"{rh['shock']['over_FS'][1]:+13.4f} {rh['contact']['over_FS'][2]:+17.4f}"
            npar = models[name].num_params()
        else:
            tail, npar = f"{'-':>13s} {'-':>17s}", '-'
        print(f'{name:{w}s} {npar!s:>7s} {l1:9.2e} | {pc} | {tail}')

    e = np.arange(a.nx + 1) * a.dx
    xc, xs = (e[:-1] + e[1:]) / 2, np.linspace(0, a.nx * a.dx, 3000)
    ex_pr = ex.sample(xs, t)
    fig, ax = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    cm = plt.get_cmap('tab10')
    for j, ylab in enumerate(['density', 'velocity', 'pressure']):
        ax[j].plot(xs, ex_pr[j], 'k-', lw=1.2, label='exact')
        k = 0
        for name, P in sols.items():
            cl = name in CLASSICAL
            ax[j].plot(xc, prim(P)[j], '--' if cl else '-', lw=1 if cl else 1.3, color='gray' if cl else cm(k % 10),
                       alpha=0.8 if cl else 1, label=name)
            k += 0 if cl else 1
        ax[j].set_ylabel(ylab)
    if a.xlim:
        ax[0].set_xlim(*a.xlim)
    ax[0].legend(fontsize=8)
    ax[0].set_title(f'Sod shock tube (N={a.nx}, t={t:.2e} s)')
    fig.tight_layout()
    fig.savefig(a.out, dpi=120)
    plt.close(fig)
    print(f'\nplot written to {a.out}')


if __name__ == '__main__':
    main()