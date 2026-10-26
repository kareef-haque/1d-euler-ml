'''
Differentiable torch stepper for NFV (learned flux). Integrator from model.integrator:
'euler' (default, as in the paper) or 'rk3' (SSP-RK3; stages reuse the ghost cells of the latest slice).
Models with safe_flux (weighted stencil) may replace fluxes after a trial update, per stage.

hist: list of b padded states, oldest first, each (B, 3, N + 2g) with g = a // 2
flux: (B, 3, N + 1) in physical units
'''
import torch

from euler1d.neural import scaling


def _sc(x, v):
    return torch.as_tensor(v, dtype=x.dtype, device=x.device).view(1, 3, 1)


def flux(model, hist):
    x = torch.cat([h / _sc(h, scaling.QS) for h in hist], dim=1)
    F = model(x)
    return F * _sc(F, scaling.FS)


def advance(model, hist, dt, dx):
    g = model.g
    Q = hist[-1][..., g:-g]
    r = torch.as_tensor(dt, dtype=Q.dtype, device=Q.device).reshape(-1, 1, 1) / dx

    def step(U):
        h = torch.cat([hist[-1][..., :g], U, hist[-1][..., -g:]], dim=-1)
        F = flux(model, hist[:-1] + [h])
        new = U - r * (F[..., 1:] - F[..., :-1])
        if hasattr(model, 'safe_flux'):
            F = model.safe_flux(h, F, new)
            new = U - r * (F[..., 1:] - F[..., :-1])
        return new

    if getattr(model, 'integrator', 'euler') == 'rk3':
        U1 = step(Q)
        U2 = 0.75 * Q + 0.25 * step(U1)
        return Q / 3 + 2 / 3 * step(U2)
    return step(Q)


def pad_zero(Q, g):
    l = Q[..., :1].expand(*Q.shape[:-1], g)
    r = Q[..., -1:].expand(*Q.shape[:-1], g)
    return torch.cat([l, Q, r], dim=-1)


def rollout(model, Q, G, dt, dx, bc='exact'):
    '''
    Q: (B, T, 3, N + 2G) exact data. The first b slices are given, the rest is predicted.
    bc='exact': ghost cells at every step come from Q
    bc='zero':  zero-gradient ghost cells from the prediction itself (as in the solver)
    returns (B, T, 3, N)
    '''
    g, b = model.g, model.b
    B, T, _, W = Q.shape
    N = W - 2 * G
    if bc == 'exact' and g > G:
        raise ValueError(f'model needs {g} ghost cells, data has {G}')
    inner = Q[..., G:G + N]

    def pad(k, x):
        if bc == 'zero':
            return pad_zero(x, g)
        return torch.cat([Q[:, k, :, G - g:G], x, Q[:, k, :, G + N:G + N + g]], dim=-1)

    hist = [pad(k, inner[:, k]) for k in range(b)]
    out = [inner[:, k] for k in range(b)]
    for k in range(b, T):
        new = advance(model, hist, dt, dx)
        out.append(new)
        hist = hist[1:] + [pad(k, new)]
    return torch.stack(out, dim=1)