import torch

from euler1d.neural import scaling, stepper
from infrastructure.nfv_dataset import RHO, P


def phys_flux(Q, gamma=1.4):
    '''Q: (n, 3) conservative states -> (n, 3)'''
    rho, mom, E = Q.unbind(-1)
    u = mom / rho
    p = (gamma - 1.0) * (E - 0.5 * rho * u**2)
    return torch.stack([mom, mom * u + p, u * (E + p)], dim=-1)


def pick(Q, n, gen=None):
    '''n random states from Q: (B, T, 3, W) -> (n, 3)'''
    B, T, _, W = Q.shape
    i, k, c = (torch.randint(0, m, (n,), device=Q.device, generator=gen) for m in (B, T, W))
    return Q[i, k, :, c]


def box(n, like, umax=450.0, gamma=1.4, gen=None):
    '''n states with rho, P as in the dataset and u uniform in [-umax, umax]; like: tensor giving device and dtype'''
    r = lambda lo, hi: lo + (hi - lo) * torch.rand(n, device=like.device, dtype=like.dtype, generator=gen)
    rho, u, p = r(*RHO), r(-umax, umax), r(*P)
    return torch.stack([rho, rho * u, p / (gamma - 1.0) + 0.5 * rho * u**2], dim=-1)


def states(Q, n, mode='data', umax=450.0, gamma=1.4, gen=None):
    if mode == 'data':
        return pick(Q, n, gen)
    if mode == 'box':
        return box(n, Q, umax, gamma, gen)
    if mode == 'mix':
        return torch.cat([pick(Q, n // 2, gen), box(n - n // 2, Q, umax, gamma, gen)])
    raise ValueError(f'unknown mode {mode}')


def flux_at(model, S):
    '''learned flux of constant windows: S (n, 3) -> (n, 3), physical units'''
    x = S[:, :, None].expand(len(S), 3, model.a)
    return stepper.flux(model, [x] * model.b)[:, :, 0]


def penalty(model, Q, n=512, gamma=1.4, gen=None, mode='data', umax=450.0):
    S = states(Q, n, mode, umax, gamma, gen)
    fs = torch.as_tensor(scaling.FS, dtype=S.dtype, device=S.device).view(1, 3)
    return ((flux_at(model, S) - phys_flux(S, gamma)) / fs).pow(2).mean()