'''
Batched NumPy rollouts on (B, 3, N) interior states, for classical schemes and NFV alike.

pad(x, m, g): states with g ghost cells per side at half-step index m (time m * dt / 2),
so RK stages see exact boundary values. bc: 'exact' (exact solution), 'zero', 'periodic'.
'''
import copy
import numpy as np
import torch

from euler1d.solver import FLUX, REC
from euler1d.neural import stepper
from infrastructure.exact import ExactRiemannSolver

SCHEMES = {
    'HLLC-1': ('HLLC', 'Const', 'euler'),
    'Rusanov-1': ('Rusanov', 'Const', 'euler'),
    'AUSM+-1': ('AUSM+', 'Const', 'euler'),
    'HLLC-WENO5Z': ('HLLC', 'WENO5Z', 'rk4'),
    'AUSM+-WENO5': ('AUSM+', 'WENO5', 'rk4'),
}


def exact_ghosts(pl, pr, xf, nx, dx, dt, M, g=3, gamma=1.4):
    B = len(xf)
    Gl, Gr = np.empty((B, M, 3, g)), np.empty((B, M, 3, g))
    el, er = (np.arange(g + 1) - g) * dx, (nx + np.arange(g + 1)) * dx
    for i in range(B):
        ex = ExactRiemannSolver(pl[i], pr[i], xf[i] * nx * dx, gamma)
        ts = np.arange(M) * dt[i] / 2
        Gl[i], Gr[i] = ex.cell_avg_t(el, ts), ex.cell_avg_t(er, ts)
    return Gl, Gr


def make_pad(mode, Gl=None, Gr=None):
    def pad(x, m, g):
        if mode == 'exact':
            return np.concatenate([Gl[:, m, :, -g:], x, Gr[:, m, :, :g]], axis=-1)
        if mode == 'zero':
            return np.concatenate([np.repeat(x[..., :1], g, -1), x, np.repeat(x[..., -1:], g, -1)], axis=-1)
        return np.concatenate([x[..., -g:], x, x[..., :g]], axis=-1)
    return pad


def classical_step(flux, rec, integ, dx, gamma, pad):
    F_s, R_s = FLUX[flux], REC[rec]

    def rhs(x, m):
        P = pad(x, m, 3)
        B, _, Wp = P.shape
        QL, QR = R_s(P.transpose(1, 0, 2).reshape(3, -1), dx=dx)
        idx = np.arange(B)[:, None] * Wp + np.arange(Wp - 5)[None]
        F = F_s(QL[:, idx].reshape(3, -1), QR[:, idx].reshape(3, -1), dx=dx, gamma=gamma)
        F = F.reshape(3, B, Wp - 5).transpose(1, 0, 2)
        return -(F[..., 1:] - F[..., :-1]) / dx

    def step(x, k, h):
        if integ == 'euler':
            return x + h * rhs(x, 2 * k)
        k1 = rhs(x, 2 * k)
        k2 = rhs(x + h * k1 / 2, 2 * k + 1)
        k3 = rhs(x + h * k2 / 2, 2 * k + 1)
        k4 = rhs(x + h * k3, 2 * k + 2)
        return x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return step


def nfv_step(model, dx, pad):
    m = copy.deepcopy(model).double().eval()
    hist = []

    def step(x, k, h):
        xp = pad(x, 2 * k, m.g)
        prev = hist[len(hist) - (m.b - 1):] if m.b > 1 else []
        prev = [xp] * (m.b - 1 - len(prev)) + prev
        hist.append(xp)
        del hist[:-m.b]
        with torch.no_grad():
            return stepper.advance(m, [torch.from_numpy(np.ascontiguousarray(q)) for q in prev + [xp]],
                                   torch.from_numpy(h.reshape(-1)), dx).numpy()

    def prime(init):
        hist.extend(pad(init[j], 2 * j, m.g) for j in range(len(init) - 1))
    step.prime = prime
    return step


def run(Q0, dt, dx, nt, step, stride=1, init=None):
    """init: optional list of the first b exact states (times 0..b-1), as in training."""
    h = np.asarray(dt, dtype=float).reshape(-1, 1, 1)
    if init is None:
        init = [Q0]
    init = [np.array(q, dtype=float) for q in init]
    if hasattr(step, 'prime'):
        step.prime(init)
    out, x = list(init), init[-1]
    for k in range(len(init) - 1, nt):
        x = step(x, k, h)
        if (k + 1) % stride == 0:
            out.append(x)
    return np.stack(out, axis=1)


def predict_all(d, names, model=None):
    G, N = d['G'], d['Q'].shape[-1] - 2 * d['G']
    dx, gamma, nt = d['dx'], d['gamma'], d['Q'].shape[1] - 1
    Q0 = np.asarray(d['Q'][:, 0, :, G:G + N], dtype=float)
    if d.get('bc', 'exact') == 'exact':
        Gl, Gr = exact_ghosts(d['pl'], d['pr'], d['xf'], N, dx, d['dt'], 2 * nt + 1,
                              max(3, model.g if model else 3), gamma)
        pad = make_pad('exact', Gl, Gr)
    else:
        pad = make_pad('zero')
    out = {}
    for n in names:
        if n == 'NFV':
            init = [np.asarray(d['Q'][:, j, :, G:G + N], dtype=float) for j in range(model.b)]
            out[n] = run(Q0, d['dt'], dx, nt, nfv_step(model, dx, pad), init=init)
        else:
            out[n] = run(Q0, d['dt'], dx, nt, classical_step(*SCHEMES[n], dx, gamma, pad))
    return out