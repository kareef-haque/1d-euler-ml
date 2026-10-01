'''
Evaluate a trained NFV model against classical schemes.

suites:
  riemann  held-out random Riemann problems, exact references, exact boundary values
  multi    piecewise-constant states with several discontinuities, fine-grid numerical reference
  named    Sod, Lax, 123 problem (exact references; Lax and 123 are outside the training ranges)
Also reports positivity, mass/energy drift, flux consistency (offset-aware), mirror symmetry
and rarefaction-fan glitches (expansion shocks).
'''
import argparse
import json
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from infrastructure.nfv_dataset import make_set, from_states, load_set, save_set
from infrastructure.multi_ref import multi_set
from infrastructure.exact import ExactRiemannSolver
from euler1d.neural.model import NFVModel
from euler1d.neural import rollouts as ro, diagnostics as dg, evaluate as ev

NAMES = list(ro.SCHEMES) + ['NFV']
S = 316.2277660168379
NAMED = {'Sod': ((1.0, 0.0, 1e5), (0.125, 0.0, 1e4)),
         'Lax': ((0.445, 0.698 * S, 3.528e5), (0.5, 0.0, 0.571e5)),
         '123': ((1.0, -2 * S, 0.4e5), (1.0, 2 * S, 0.4e5))}


def score(d, P):
    Q = torch.as_tensor(np.asarray(d['Q']), dtype=torch.float64)
    G, N = d['G'], Q.shape[-1] - 2 * d['G']
    s = ev.summarize(torch.as_tensor(P), Q[..., G:G + N], 1, Q.shape[1] - 1)
    return {k: v.numpy() for k, v in s.items() if k in ('l1', 'l2')}


def table(res, base='HLLC-1'):
    rows = {}
    for n, s in res.items():
        rows[n] = dict(l1=float(s['l1'].mean()), l2=float(s['l2'].mean()), l2_median=float(np.median(s['l2'])),
                       n_div=int(np.isinf(s['l2']).sum()), win_vs_base=float((s['l2'] < res[base]['l2']).mean()),
                       ratio_vs_base=float(res[base]['l2'].mean() / s['l2'].mean()))
    print(f"{'scheme':14s} {'L1':>10s} {'L2':>10s} {'L2 median':>10s} {'win vs ' + base:>14s} {'ratio':>7s} {'div':>4s}")
    for n, r in rows.items():
        print(f"{n:14s} {r['l1']:10.3e} {r['l2']:10.3e} {r['l2_median']:10.3e} {100 * r['win_vs_base']:13.0f}% "
              f"{r['ratio_vs_base']:7.2f} {r['n_div']:4d}")
    wm = {a: {b: float((res[a]['l2'] < res[b]['l2']).mean()) for b in res if b != a} for a in res}
    return dict(rows=rows, win_matrix=wm)


def box(res, path, title):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot([np.clip(res[n]['l2'], 1e-12, 1e6) for n in res], showfliers=False)
    ax.set_xticks(range(1, len(res) + 1), list(res), rotation=45, ha='right')
    ax.set_yscale('log')
    ax.set_title(title)
    ax.set_ylabel('scaled L2 error')
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def pos_report(P):
    return dg.positivity(P)


def run_suite(name, d, model, out):
    print(f'\n== {name} ==')
    P = ro.predict_all(d, NAMES, model)
    res = {n: score(d, P[n]) for n in NAMES}
    r = dict(table(res), positivity={n: pos_report(P[n]) for n in NAMES})
    print('positivity (NFV):', r['positivity']['NFV'])
    box(res, os.path.join(out, f'{name}_box.png'), name)
    return r, P


def probes(model, d, out):
    m = NFVModel.load(model) if isinstance(model, str) else model
    c = dg.consistency_error(m, states=dg.states_from_set(d))
    r = dict(consistency={k: v.tolist() for k, v in c.items()})
    if m.b == 1:
        mg = dg.mirror_gap(dg.nfv_center_flux(NFVModel.load(model) if isinstance(model, str) else model),
                           dg.windows_from_set(d, m.a))
        r['mirror'] = {k: v.tolist() for k, v in mg.items()}
    print('consistency (solution states): offset-removed err/FS', np.round(c['err_shift'], 4),
          ' flux spread/FS', np.round(c['ref_std'], 4))
    if 'mirror' in r:
        print('mirror symmetry err/FS', np.round(r['mirror']['err'], 4), ' flux spread/FS', np.round(r['mirror']['spread'], 4))
    return r


def named_suite(model, out, nx, nt, dx):
    r = {}
    for tn, (pl, pr) in NAMED.items():
        d = from_states(pl, pr, 0.5, nx, nt, dx, 0.5, dtype=np.float64)
        P = ro.predict_all(d, NAMES, model)
        ex = ExactRiemannSolver(pl, pr, 0.5 * nx * dx, 1.4)
        e = np.arange(nx + 1) * dx
        Qe = ex.cell_avg(e, nt * d['dt'][0])
        rows = {}
        cons = lambda q: np.array([[q[0], q[0] * q[1], q[2] / 0.4 + 0.5 * q[0] * q[1]**2]])
        fin, fout = dg.flux_of(cons(pl)), dg.flux_of(cons(pr))
        for n in NAMES:
            Pf = P[n][0, -1]
            rows[n] = dict(rho_l1=float(np.abs(Pf[0] - Qe[0]).mean() / pl[0]), l2=float(score(d, P[n])['l2'][0]),
                           positivity=dg.positivity(P[n]), drift=dg.drift(P[n], d['dt'], dx, fin, fout),
                           fan=dg.fan_glitch(ex, nt * d['dt'][0], dx, nx, Pf[0]))
        r[tn] = rows
        print(f"\n== {tn} (ex. p*={ex.P_star:.4g}, u*={ex.u_star:.4g}) ==")
        for n, v in rows.items():
            fan = ' '.join(f"fan x{f['num'] / max(f['exact'], 1e-12):.1f}" for f in v['fan'])
            print(f"{n:14s} rho L1/rho_L={v['rho_l1']:.3e}  L2={v['l2']:.3e}  min rho={v['positivity']['min_rho']:.3g} "
                  f"min p={v['positivity']['min_p']:.3g}  boundary-flux mismatch (mass)={v['drift']['mass']:.1e}  {fan}")
        fig, ax = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
        xc = (e[:-1] + e[1:]) / 2
        xs = np.linspace(0, nx * dx, 2000)
        pr_ex = ex.sample(xs, nt * d['dt'][0])
        for j, lab in enumerate(['density', 'velocity', 'pressure']):
            ax[j].plot(xs, pr_ex[j], 'k-', lw=1, label='exact')
            for n, sty in (('NFV', 'r-'), ('HLLC-1', 'b-'), ('HLLC-WENO5Z', 'g-')):
                Pf = P[n][0, -1]
                v = [Pf[0], Pf[1] / Pf[0], 0.4 * (Pf[2] - 0.5 * Pf[1]**2 / Pf[0])][j]
                ax[j].plot(xc, v, sty, ms=3, lw=1, label=n)
            ax[j].set_ylabel(lab)
        ax[0].legend()
        ax[0].set_title(tn)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f'named_{tn}.png'), dpi=110)
        plt.close(fig)
    return r


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--suites', default='riemann,multi,named')
    p.add_argument('--data_dir', default='generated_nfv')
    p.add_argument('--out', default='eval_out')
    p.add_argument('--riemann_nx', type=int, default=200)
    p.add_argument('--riemann_nt', type=int, default=400)
    p.add_argument('--n_riemann', type=int, default=100)
    p.add_argument('--n_multi', type=int, default=30)
    p.add_argument('--multi_nx', type=int, default=200)
    p.add_argument('--multi_nt', type=int, default=100)
    p.add_argument('--multi_ref', type=int, default=8)
    p.add_argument('--named_nx', type=int, default=400)
    p.add_argument('--named_nt', type=int, default=100)
    p.add_argument('--dx', type=float, default=1e-3)
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    model = NFVModel.load(a.model)
    print(f'model a={model.a} b={model.b} act={model.act} params={model.num_params()} step={model.extra.get("step")}')
    suites, results = a.suites.split(','), {}

    if 'riemann' in suites:
        path = os.path.join(a.data_dir, f'test_nx{a.riemann_nx}_nt{a.riemann_nt}.npz')
        if os.path.exists(path):
            d = load_set(path)
        else:
            os.makedirs(a.data_dir, exist_ok=True)
            d = make_set(a.n_riemann, a.riemann_nx, a.riemann_nt, a.dx, 'test', 0, G=max(3, model.g))
            save_set(path, d)
        d = {k: (v[:a.n_riemann] if k in ('Q', 'dt', 'smax', 'pl', 'pr', 'xf') else v) for k, v in d.items()}
        r, _ = run_suite('riemann', d, model, a.out)
        r.update(probes(a.model, d, a.out))
        results['riemann'] = r
    if 'multi' in suites:
        d = multi_set(a.n_multi, a.multi_nx, a.multi_nt, a.dx, ref=a.multi_ref)
        r, P = run_suite('multi', d, model, a.out)
        r['drift'] = {n: dg.drift(P[n], d['dt'], a.dx, dg.flux_of(d['pc'][:, 0]), dg.flux_of(d['pc'][:, -1])) for n in NAMES}
        results['multi'] = r
    if 'named' in suites:
        results['named'] = named_suite(model, a.out, a.named_nx, a.named_nt, a.dx)

    with open(os.path.join(a.out, 'results.json'), 'w') as f:
        json.dump(results, f, indent=1, default=float)
    print(f'\nwritten to {a.out}')
    return results


if __name__ == '__main__':
    main()