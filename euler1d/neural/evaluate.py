'''
Evaluation against exact references, on the same protocol for the model and the baseline.

Errors are on Q / QS (so all three variables count equally), averaged over cells and steps b..T-1.
The _h versions only use steps up to the horizon H (e.g. the length the model was trained on).
A sample whose rollout is not finite gets inf.
'''
import copy
import numpy as np
import torch

from euler1d.neural import scaling, stepper
from euler1d.neural.baseline import first_order


def prep(d, device='cpu', dtype=torch.float64):
    t = lambda x: torch.as_tensor(x, dtype=dtype, device=device)
    return dict(Q=t(d['Q']), dt=t(d['dt']), dx=float(d['dx']), G=int(d['G']))


def base_pred(d, device='cpu'):
    P = first_order(np.asarray(d['Q'], dtype=float), int(d['G']), d['dt'], float(d['dx']), float(d['gamma']))
    return torch.as_tensor(P, dtype=torch.float64, device=device)


def summarize(P, ref, b, H):
    qs = torch.as_tensor(scaling.QS, dtype=P.dtype, device=P.device).view(1, 1, 3, 1)
    e = (P - ref) / qs
    l1, l2 = e.abs().mean((2, 3)), e.pow(2).mean((2, 3))
    inf = torch.tensor(float('inf'), dtype=P.dtype, device=P.device)
    ok = torch.isfinite(P).all(-1).all(-1).all(-1)
    okh = torch.isfinite(P[:, :H + 1]).all(-1).all(-1).all(-1)
    return dict(l1=torch.where(ok, l1[:, b:].mean(1), inf), l2=torch.where(ok, l2[:, b:].mean(1), inf),
                l1h=torch.where(okh, l1[:, b:H + 1].mean(1), inf), l2h=torch.where(okh, l2[:, b:H + 1].mean(1), inf))


def report(s, sb):
    r = {}
    for k in ('', 'h'):
        m, mb = s['l2' + k], sb['l2' + k]
        r['l1' + k] = s['l1' + k].mean().item()
        r['l2' + k] = m.mean().item()
        r['base_l2' + k] = mb.mean().item()
        r['n_div' + k] = int(torch.isinf(m).sum())
        r['win' + k] = (m < mb).double().mean().item()
        r['ratio' + k] = mb.mean().item() / m.mean().item() if m.mean().item() > 0 else float('inf')
    return r


def evaluate(model, ed, Pbase, H, return_pred=False):
    m = copy.deepcopy(model).double().eval()
    G, N = ed['G'], ed['Q'].shape[-1] - 2 * ed['G']
    with torch.no_grad():
        P = stepper.rollout(m, ed['Q'], G, ed['dt'], ed['dx'], 'exact')
    ref = ed['Q'][..., G:G + N]
    H = min(H, ed['Q'].shape[1] - 1)
    s, sb = summarize(P, ref, m.b, H), summarize(Pbase, ref, m.b, H)
    r = report(s, sb)
    return (r, P, s, sb) if return_pred else r