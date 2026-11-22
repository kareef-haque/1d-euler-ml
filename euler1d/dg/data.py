'''
Exact-reference datasets for DG training, and rollouts through the DG solver with exact ghost cells.

Same random Riemann problems as the FV data (nfv_dataset.sample_specs, same seeds per split).
Per sample dt = cfl * dx / smax (cfl default 0.9 / (2p + 1)).
    Q:  (B, nt+1, 3, N, P)    exact projection of the interior cells at t = k dt
    Gh: (B, 2nt+1, 3, 2, P)   the left and right ghost cell at t = k dt / 2 (SSP-RK3 needs t, t + dt, t + dt/2)
'''
import os
import numpy as np
import torch

from euler1d.dg import solver as dg
from euler1d.neural import scaling
from euler1d.neural.baseline import first_order
from infrastructure.exact import ExactRiemannSolver
from infrastructure.nfv_dataset import sample_specs, SPLITS, save_set, load_set

D = torch.float64


def default_cfl(p):
    return 0.9 / (2 * p + 1)


def from_states(pl, pr, xf, nx, nt, dx, p, cfl=None, gamma=1.4, dtype=np.float32):
    cfl = cfl or default_cfl(p)
    P = p + 1
    n = len(xf)
    e = (np.arange(nx + 3) - 1) * dx                     # ghost, interior, ghost
    Q = np.empty((n, nt + 1, 3, nx, P), dtype=dtype)
    Gh = np.empty((n, 2 * nt + 1, 3, 2, P), dtype=dtype)
    dts, sm = np.empty(n), np.empty(n)
    gl, gr = e[:2], e[-2:]
    for i in range(n):
        ex = ExactRiemannSolver(pl[i], pr[i], xf[i] * nx * dx, gamma)
        sm[i] = ex.smax()
        dts[i] = cfl * dx / sm[i]
        for k in range(2 * nt + 1):
            t = 0.5 * k * dts[i]
            if k % 2 == 0:
                full = dg.project_exact(ex, e, t, P).numpy()
                Q[i, k // 2] = full[:, 1:-1]
                Gh[i, k] = full[:, [0, -1]]
            else:
                Gh[i, k] = np.concatenate([dg.project_exact(ex, gl, t, P).numpy(),
                                           dg.project_exact(ex, gr, t, P).numpy()], 1)
    return dict(Q=Q, Gh=Gh, dt=dts, smax=sm, pl=pl, pr=pr, xf=xf, dx=dx, p=p, cfl=cfl, nx=nx, nt=nt, gamma=gamma)


def make_set(n, nx, nt, dx, p, split='train', seed=0, cfl=None):
    pl, pr, xf = sample_specs(n, np.random.default_rng([seed, SPLITS[split]]))
    return dict(from_states(pl, pr, xf, nx, nt, dx, p, cfl), split=split, seed=seed)


def fname(split, nx, nt, p, cfl):
    return f'dg_p{p}_{split}_nx{nx}_nt{nt}_cfl{cfl:.4g}.npz'


def get_set(data_dir, split, n, nx, nt, dx, p, seed=0, cfl=None):
    '''load from data_dir if it matches, else build and save'''
    cfl = cfl or default_cfl(p)
    path = os.path.join(data_dir, fname(split, nx, nt, p, cfl))
    if os.path.exists(path):
        d = load_set(path)
        if d['Q'].shape[0] < n or abs(d['dx'] - dx) > 1e-15 or d['seed'] != seed:
            raise ValueError(f'{path} does not match (n, dx, seed); delete it or use another data_dir')
        return {k: (v[:n] if k in ('Q', 'Gh', 'dt', 'smax', 'pl', 'pr', 'xf') else v) for k, v in d.items()}
    os.makedirs(data_dir, exist_ok=True)
    d = make_set(n, nx, nt, dx, p, split, seed, cfl)
    save_set(path, d)
    return d


def tensors(d, device='cpu'):
    t = lambda x: torch.as_tensor(np.asarray(x), dtype=D, device=device)
    return dict(Q=t(d['Q']), Gh=t(d['Gh']), dt=t(d['dt']), dx=float(d['dx']))


def rollout(td, flux='hllc', n=None, M=0.0, limit=True, pos=True):
    '''SSP-RK3 DG rollout from the exact initial state with exact ghost cells; flux: name or callable.
    Returns (B, n+1, 3, N, P)'''
    Q, Gh, dt, h = td['Q'], td['Gh'], td['dt'], td['dx']
    n = n or Q.shape[1] - 1
    k4 = dt.view(-1, 1, 1, 1)
    if pos:                                          # projected jumps can overshoot to rho, p < 0 at the cell ends
        B, T2, _, _, P = Gh.shape
        Gh = dg.positivity(Gh.permute(0, 1, 3, 2, 4).reshape(B * T2, 2, 3, P).transpose(1, 2))
        Gh = Gh.transpose(1, 2).reshape(B, T2, 2, 3, P).permute(0, 1, 3, 2, 4)
    pad = lambda U, j: torch.cat([Gh[:, j, :, :1], U, Gh[:, j, :, 1:]], 2)
    L = lambda U, j: dg.rhs(pad(U, j), h, flux)

    def lim(U, j):
        if limit:
            U = dg.tvb(pad(U, j), h, M)
        if pos:
            U = dg.positivity(U)
        return U

    U = lim(Q[:, 0], 0)
    out = [U]
    for k in range(n):
        a, mid, b = 2 * k, 2 * k + 1, 2 * k + 2
        U1 = lim(U + k4 * L(U, a), b)
        U2 = lim(0.75 * U + 0.25 * (U1 + k4 * L(U1, b)), mid)
        U = lim(U / 3 + 2 / 3 * (U2 + k4 * L(U2, mid)), b)
        out.append(U)
    return torch.stack(out, 1)


def mean_err(P, td, H=None):
    '''per-sample MSE of the cell means scaled by QS, over steps 1..H (inf if not finite)'''
    ref = td['Q'][:, :P.shape[1], ..., 0]
    H = H or P.shape[1] - 1
    qs = torch.as_tensor(scaling.QS, dtype=D).view(1, 1, 3, 1)
    e = ((P[:, 1:H + 1, ..., 0] - ref[:, 1:H + 1]) / qs).pow(2).mean((1, 2, 3))
    ok = torch.isfinite(P[:, :H + 1]).all(-1).all(-1).all(-1).all(-1)
    return torch.where(ok, e, torch.full_like(e, float('inf')))


def fv_first_order(d):
    '''first-order HLLC (FV, forward Euler) on the cell means with the exact ghost means: (B, nt+1, 3, N)'''
    Q = np.asarray(d['Q'], dtype=float)[..., 0]
    G = np.asarray(d['Gh'], dtype=float)[:, ::2, :, :, 0]
    pad = np.concatenate([G[..., :1], Q, G[..., 1:]], -1)
    return torch.as_tensor(first_order(pad, 1, d['dt'], d['dx']), dtype=D)


def compare(d, H=None, fluxes=('rusanov', 'hllc')):
    '''DG baselines and first-order FV on a dataset: mean error, ratio to FV, win rate, non-finite count'''
    td = tensors(d)
    base = mean_err(fv_first_order(d)[..., None], td, H)
    res = {'FV HLLC-1': dict(err=base.mean().item(), ratio=1.0, win=0.0, bad=0)}
    for f in fluxes:
        with torch.no_grad():
            e = mean_err(rollout(td, f, n=H), td, H)
        ok = torch.isfinite(e)
        res[f'DG-P{d["p"]} {f}'] = dict(err=e[ok].mean().item(), ratio=(base[ok].mean() / e[ok].mean()).item(),
                                        win=(e < base).double().mean().item(), bad=int((~ok).sum()))
    return res
