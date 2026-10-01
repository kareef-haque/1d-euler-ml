'''
Refinement studies at fixed CFL.

riemann: random Riemann problems on a fixed physical domain, error at the time the fastest wave
         has travelled a quarter of the domain; reports observed order between refinements
smooth:  density wave advected by a constant velocity on a periodic domain (the paper never tests this)
Errors are on Q / QS, averaged over cells (and over problems).
'''
import argparse
import json
import numpy as np

from infrastructure.nfv_dataset import sample_specs, SPLITS
from infrastructure.exact import ExactRiemannSolver
from euler1d.neural.model import NFVModel
from euler1d.neural import rollouts as ro, scaling

NAMES = list(ro.SCHEMES) + ['NFV']
QS = scaling.QS.reshape(1, 3, 1)


def make_step(name, model, dx, pad, gamma=1.4):
    return ro.nfv_step(model, dx, pad) if name == 'NFV' else ro.classical_step(*ro.SCHEMES[name], dx, gamma, pad)


def orders(Ns, errs):
    return [float(np.log(errs[i - 1] / errs[i]) / np.log(Ns[i] / Ns[i - 1])) for i in range(1, len(Ns))]


def riemann(model, Ns, n=20, cfl=0.5, seed=0, L=0.2, names=NAMES):
    pl, pr, xf = sample_specs(n, np.random.default_rng([seed, SPLITS['test']]))
    out = {k: [] for k in names}
    for N in Ns:
        dx, nt = L / N, int(round(0.25 * N / cfl))
        exs = [ExactRiemannSolver(pl[i], pr[i], xf[i] * L, 1.4) for i in range(n)]
        dt = cfl * dx / np.array([e.smax() for e in exs])
        e = np.arange(N + 1) * dx
        Q0 = np.stack([x.cell_avg(e, 0.0) for x in exs])
        Qf = np.stack([x.cell_avg(e, nt * dt[i]) for i, x in enumerate(exs)])
        Gl, Gr = ro.exact_ghosts(pl, pr, xf, N, dx, dt, 2 * nt + 1, max(3, model.g))
        pad = ro.make_pad('exact', Gl, Gr)
        for k in names:
            P = ro.run(Q0, dt, dx, nt, make_step(k, model, dx, pad), stride=nt)[:, -1]
            out[k].append(float((np.abs(P - Qf) / QS).mean()))
    return {k: dict(err=v, order=orders(Ns, v)) for k, v in out.items()}


def smooth(model, Ns, cfl=0.5, u0=20.0, p0=1e5, amp=0.2, shift=0.1, names=NAMES):
    out = {k: [] for k in names}
    cons = lambda rho: np.stack([rho, rho * u0, p0 / 0.4 + 0.5 * rho * u0**2])[None]
    avg = lambda x0, x1: 1 + amp * (np.cos(2 * np.pi * x0) - np.cos(2 * np.pi * x1)) / (2 * np.pi * (x1 - x0))
    for N in Ns:
        dx = 1.0 / N
        T = shift / u0
        nt = int(np.ceil(T / (cfl * dx / (u0 + np.sqrt(1.4 * p0 / (1 - amp))))))
        dt = np.array([T / nt])
        e = np.arange(N + 1) * dx
        Q0, Qf = cons(avg(e[:-1], e[1:])), cons(avg(e[:-1] - u0 * T, e[1:] - u0 * T))
        pad = ro.make_pad('periodic')
        for k in names:
            P = ro.run(Q0, dt, dx, nt, make_step(k, model, dx, pad), stride=nt)[:, -1]
            out[k].append(float((np.abs(P - Qf) / QS).mean()))
    return {k: dict(err=v, order=orders(Ns, v)) for k, v in out.items()}


def show(title, Ns, res):
    print(f'\n{title}\n{"scheme":14s}' + ''.join(f'{"N=" + str(N):>11s}' for N in Ns) + '   observed order')
    for k, v in res.items():
        print(f'{k:14s}' + ''.join(f'{e:11.2e}' for e in v['err']) + '   ' + ' '.join(f'{o:5.2f}' for o in v['order']))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--suites', default='riemann,smooth')
    p.add_argument('--Ns', default='100,200,400,800')
    p.add_argument('--n', type=int, default=20)
    p.add_argument('--out', default='convergence.json')
    a = p.parse_args(argv)
    model, Ns, res = NFVModel.load(a.model), [int(x) for x in a.Ns.split(',')], {}
    if 'riemann' in a.suites:
        res['riemann'] = riemann(model, Ns, a.n)
        show('Riemann problems (error at fixed physical time)', Ns, res['riemann'])
    if 'smooth' in a.suites:
        res['smooth'] = smooth(model, Ns)
        show('Smooth density wave, periodic', Ns, res['smooth'])
    json.dump(res, open(a.out, 'w'), indent=1)
    return res


if __name__ == '__main__':
    main()