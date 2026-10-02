'''
Star-state plateau errors of NFV on the Sod problem, per checkpoint.
If the errors shrink steadily with training, more training helps; if they stall, the error is structural.

python sod_plateaus.py runs/<run>/model_{30001,40001,50001,57001}.pt
'''
import argparse
import numpy as np
import torch

from infrastructure.nfv_dataset import from_states
from infrastructure.exact import ExactRiemannSolver
from euler1d.neural.model import NFVModel
from euler1d.neural import rollouts as ro, stepper, scaling
from euler1d.neural.diagnostics import flux_of

SOD = ((1.0, 0.0, 1e5), (0.125, 0.0, 1e4))


def measure(Pf, ex, t, dx):
    """Pf: (3, N) conservative cell averages at time t. Returns plateau values and relative errors."""
    g, nx = ex.gamma, Pf.shape[1]
    rho, u = Pf[0], Pf[1] / Pf[0]
    p = (g - 1) * (Pf[2] - 0.5 * rho * u**2)
    x = (np.arange(nx) + 0.5) * dx
    tail = ex.x_int + t * (ex.u_star - ex.a_L * (ex.P_star / ex.P_L) ** ((g - 1) / (2 * g)))
    cont = ex.x_int + t * ex.u_star
    shock = ex.x_int + t * (ex.u_R + ex.a_R * np.sqrt((g + 1) / (2 * g) * ex.P_star / ex.P_R + (g - 1) / (2 * g)))
    out = {}
    for name, (lo, hi), rho_ex in (('L', (tail, cont), ex.rho_star_L), ('R', (cont, shock), ex.rho_star_R)):
        w = hi - lo
        m = (x > lo + 0.3 * w) & (x < hi - 0.3 * w)
        out[name] = dict(rho=rho[m].mean() / rho_ex - 1, u=u[m].mean() / ex.u_star - 1, p=p[m].mean() / ex.P_star - 1)
    return out


def plateaus(model, nx=400, nt=100, dx=1e-3):
    d = from_states(SOD[0], SOD[1], 0.5, nx, nt, dx, 0.5, G=max(3, model.g), dtype=np.float64)
    P = ro.predict_all(d, ['NFV'], model)['NFV'][0, -1]
    ex = ExactRiemannSolver(SOD[0], SOD[1], 0.5 * nx * dx, 1.4)
    return measure(P, ex, nt * d['dt'][0], dx)


def cons(rho, u, p, g=1.4):
    return np.array([rho, rho * u, p / (g - 1) + 0.5 * rho * u**2])


def rh_residual(Fdiag, pl=SOD[0], pr=SOD[1]):
    """Rankine-Hugoniot residual S*[Q] - [F] of a diagonal flux F(Q) across the exact shock and contact.
    A steady numerical wave between constant states satisfies this exactly with F(Q) = F_num(Q, Q),
    so a non-zero residual means the numerical plateaus must leave the exact wave curves.
    Returns per wave: residual / FS and residual / |[F]| for (mass, momentum, energy)."""
    g = 1.4
    ex = ExactRiemannSolver(pl, pr, 0.0, g)
    QL, QR = cons(*pl), cons(*pr)
    QsL, QsR = cons(ex.rho_star_L, ex.u_star, ex.P_star), cons(ex.rho_star_R, ex.u_star, ex.P_star)
    S = ex.u_R + ex.a_R * np.sqrt((g + 1) / (2 * g) * ex.P_star / ex.P_R + (g - 1) / (2 * g))
    out = {}
    for name, lo, hi, sp in (('contact', QsL, QsR, ex.u_star), ('shock', QsR, QR, S)):
        dF = Fdiag(np.stack([hi, lo]))
        dF = dF[0] - dF[1]
        r = sp * (hi - lo) - dF
        out[name] = dict(over_FS=(r / scaling.FS).tolist(), over_jump=(r / np.abs(dF)).tolist())
    return out


def nfv_diag_flux(model):
    m = model.double()

    def fn(Q):
        x = torch.as_tensor(Q, dtype=torch.float64)[:, :, None].expand(len(Q), 3, m.a)
        with torch.no_grad():
            return stepper.flux(m, [x] * m.b)[:, :, 0].numpy()
    return fn


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('models', nargs='+')
    p.add_argument('--rh', action='store_true', help='also print Rankine-Hugoniot residuals of the learned flux')
    a = p.parse_args(argv)
    print(f"{'step':>7s} | {'rho* L':>8s} {'rho* R':>8s} | {'u* L':>8s} {'u* R':>8s} | {'p* L':>8s} {'p* R':>8s}   (relative error vs exact)")
    for path in a.models:
        m = NFVModel.load(path)
        r = plateaus(m)
        print(f"{m.extra.get('step', '?'):>7} | " + ' | '.join(
            ' '.join(f'{100 * r[s][k]:+7.1f}%' for s in 'LR') for k in ('rho', 'u', 'p')))
        if a.rh:
            for w, v in rh_residual(nfv_diag_flux(m)).items():
                print(f"        RH {w:8s} residual/FS  mass {v['over_FS'][0]:+.4f}  mom {v['over_FS'][1]:+.4f}  energy {v['over_FS'][2]:+.4f}"
                      f"   | as fraction of the flux jump: {v['over_jump'][0]:+.3f} {v['over_jump'][1]:+.3f} {v['over_jump'][2]:+.3f}")


if __name__ == '__main__':
    main()