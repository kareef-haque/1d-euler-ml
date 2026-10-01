'''
Consistency probe: a numerical flux should satisfy F(Q, Q) = f(Q).
Nothing in the vanilla NFV architecture enforces it, and the update only sees flux
differences, so F is only determined up to a constant per component. The probe therefore
reports the raw error and the error after removing the median offset (err_shift).

All values are divided by FS and averaged over sampled states.
  err, offset, err_shift: model vs f     ref: |f|     ref_std: std of f
'''
import numpy as np
import torch

from euler1d.neural import scaling, stepper
from infrastructure.nfv_dataset import sample_specs


def gap(F, f):
    d = (F - f) / scaling.FS
    off = np.median(d, axis=0)
    return dict(err=np.abs(d).mean(0), offset=off, err_shift=np.abs(d - off).mean(0),
                ref=(np.abs(f) / scaling.FS).mean(0), ref_std=(f / scaling.FS).std(0))


def flux_of(Q, gamma=1.4):
    rho, mom, E = Q.T
    u = mom / rho
    p = (gamma - 1.0) * (E - 0.5 * rho * u**2)
    return np.stack([mom, mom * u + p, u * (E + p)], axis=1)


def consistency_error(model, n=2000, seed=0, gamma=1.4, states=None):
    '''states: (n, 3) conservative states; default: initial left states of random Riemann problems'''
    if states is None:
        pl, _, _ = sample_specs(n, np.random.default_rng(seed))
        rho, u, p = pl.T
        states = np.stack([rho, rho * u, p / (gamma - 1.0) + 0.5 * rho * u**2], axis=1)
    Q, n = np.asarray(states, dtype=float), len(states)
    f = flux_of(Q, gamma)
    dt = next(model.parameters()).dtype
    x = torch.as_tensor(Q, dtype=dt)[:, :, None].expand(n, 3, model.a)
    with torch.no_grad():
        F = stepper.flux(model, [x] * model.b)[:, :, 0].double().numpy()
    return gap(F, f)


def states_from_set(d, n=5000, seed=0):
    G, W = d['G'], d['Q'].shape[-1]
    r = np.random.default_rng(seed)
    i, k, c = r.integers(0, d['Q'].shape[0], n), r.integers(0, d['Q'].shape[1], n), r.integers(G, W - G, n)
    return np.asarray(d['Q'][i, k, :, c], dtype=float)


def windows_from_set(d, a, n=5000, seed=0):
    W = d['Q'].shape[-1]
    r = np.random.default_rng(seed)
    i, k, c = r.integers(0, d['Q'].shape[0], n), r.integers(0, d['Q'].shape[1], n), r.integers(0, W - a + 1, n)
    return np.stack([np.asarray(d['Q'][i, k, :, c + j], dtype=float) for j in range(a)], axis=-1)


def mirror_gap(fn, W):
    """fn: (n, 3, a) windows -> (n, 3) flux at the centre interface. Reflection maps F to S * F."""
    S = np.array([-1.0, 1.0, -1.0])
    R = W[..., ::-1].copy()
    R[:, 1] *= -1
    d = (fn(R) - S * fn(W)) / scaling.FS
    return dict(err=np.abs(d - np.median(d, axis=0)).mean(0), spread=(S * fn(W) / scaling.FS).std(0))


def nfv_center_flux(model):
    m = model.double()

    def fn(W):
        with torch.no_grad():
            return stepper.flux(m, [torch.from_numpy(W)])[:, :, 0].numpy()
    return fn


def positivity(P, gamma=1.4):
    """P: (B, T, 3, N)"""
    rho = P[:, :, 0]
    u = P[:, :, 1] / rho
    p = (gamma - 1.0) * (P[:, :, 2] - 0.5 * rho * u**2)
    bad = ~np.isfinite(P).all(axis=(1, 2, 3)) | (rho <= 0).any(axis=(1, 2)) | (p <= 0).any(axis=(1, 2))
    return dict(min_rho=float(np.nanmin(rho)), min_p=float(np.nanmin(p)), n_bad=int(bad.sum()))


def drift(P, dt=None, dx=None, f_in=None, f_out=None):
    """Relative change of total mass and energy between first and last step (valid while waves stay inside).
    With f_in, f_out (B, 3) the expected boundary-flux contribution is removed first."""
    s = P.sum(axis=-1)
    ch = s[:, -1] - s[:, 0]
    if f_in is not None:
        ch = ch - (np.asarray(dt) * (P.shape[1] - 1) / dx)[:, None] * (np.asarray(f_in) - np.asarray(f_out))
    return {k: float((np.abs(ch[:, c]) / s[:, 0, c]).max()) for k, c in (('mass', 0), ('energy', 2))}


def fan_glitch(ex, t, dx, nx, rho_num):
    """Largest relative density jump between neighbouring cells inside each exact rarefaction fan."""
    g = ex.gamma
    regs = []
    if ex.P_star <= ex.P_L:
        regs.append((ex.u_L - ex.a_L, ex.u_star - ex.a_L * (ex.P_star / ex.P_L) ** ((g - 1) / (2 * g))))
    if ex.P_star <= ex.P_R:
        regs.append((ex.u_star + ex.a_R * (ex.P_star / ex.P_R) ** ((g - 1) / (2 * g)), ex.u_R + ex.a_R))
    e = np.arange(nx + 1) * dx
    xc, rho_ex = (e[:-1] + e[1:]) / 2, ex.cell_avg(e, t)[0]
    out = []
    for lo, hi in regs:
        m = (xc >= ex.x_int + lo * t) & (xc <= ex.x_int + hi * t)
        ins = m[:-1] & m[1:]
        if ins.sum() >= 3:
            jump = lambda r: float((np.abs(np.diff(r)) / r[:-1])[ins].max())
            out.append(dict(ncells=int(m.sum()), num=jump(rho_num), exact=jump(rho_ex)))
    return out