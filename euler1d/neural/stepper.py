'''
Differentiable torch stepper for NFV (forward Euler, learned flux).

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
    F = flux(model, hist)
    Q = hist[-1][..., g:-g]
    r = torch.as_tensor(dt, dtype=Q.dtype, device=Q.device).reshape(-1, 1, 1) / dx
    return Q - r * (F[..., 1:] - F[..., :-1])


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