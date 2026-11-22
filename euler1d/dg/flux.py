'''
Learned DG edge flux: central flux minus learned dissipation (consistent and conservative by construction).

    F = 1/2 (f(u-) + f(u+)) - 1/2 D (u+ - u-)
    mode 'scalar': D = alpha I,                 alpha   = lam_max * s,     s in [s_min, S_MAX]  (s = 1: Rusanov)
    mode 'char':   D = R diag(alpha_k) R^-1,    alpha_k = lam_max * s_k,   R at the arithmetic mean of the traces
lam_max = max(|u| + c) of the two traces. The network sees the two cells next to the edge (all DG modes, scaled by QS);
init s = 1 (Rusanov). s in [s_min, S_MAX]: s_min = 1 keeps the Lax-Friedrichs positivity property that the
Zhang-Shu limiter relies on; s_min = 0 lets the flux go to central (more accurate, can fail near vacuum, e.g. 123).
Called as flux(QL, QR, Up) by euler1d.dg.solver.rhs.
'''
import math
import torch
from torch import nn
from torch.nn import init

from euler1d.dg import solver as dg
from euler1d.neural import scaling

D = torch.float64
S_MAX = 2.0


class DissFlux(nn.Module):
    def __init__(self, p=1, mode='scalar', hidden=16, depth=4, s_min=0.0):
        super().__init__()
        self.p, self.mode, self.hidden, self.depth, self.s_min = p, mode, hidden, depth, s_min
        P = p + 1
        out = 1 if mode == 'scalar' else 3
        layers = [nn.Conv1d(3 * P, hidden, 2, dtype=D), nn.ELU()]
        for _ in range(depth - 1):
            layers += [nn.Conv1d(hidden, hidden, 1, dtype=D), nn.ELU()]
        layers.append(nn.Conv1d(hidden, out, 1, dtype=D))
        self.net = nn.Sequential(*layers)
        for m in self.net.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                init.zeros_(m.bias)
        init.zeros_(self.net[-1].weight)
        with torch.no_grad():
            q = (1 - s_min) / (S_MAX - s_min)                       # s = 1 at init
            self.net[-1].bias.fill_(math.log(q / (1 - q)) if 0 < q < 1 else -30.0)

    def scale(self, Up):
        '''s (B, out, N+1) in [s_min, S_MAX] from the cells on both sides of each edge'''
        B, _, W, P = Up.shape
        qs = torch.as_tensor(scaling.QS, dtype=D, device=Up.device).view(1, 3, 1, 1)
        x = (Up / qs).permute(0, 1, 3, 2).reshape(B, 3 * P, W)
        return self.s_min + (S_MAX - self.s_min) * torch.sigmoid(self.net(x))

    def forward(self, QL, QR, Up):
        rL, uL, pL = dg._prim(QL)
        rR, uR, pR = dg._prim(QR)
        lam = torch.maximum(uL.abs() + torch.sqrt(dg.GAMMA * pL / rL), uR.abs() + torch.sqrt(dg.GAMMA * pR / rR))
        s = self.scale(Up)
        central = 0.5 * (dg.euler_flux(QL) + dg.euler_flux(QR))
        dq = QR - QL
        if self.mode == 'scalar':
            return central - 0.5 * (lam[:, None] * s) * dq
        R = dg.eigvecs(0.5 * (QL + QR))                      # (B, n, 3, 3)
        w = torch.linalg.solve(R, dq.permute(0, 2, 1).unsqueeze(-1)).squeeze(-1)     # (B, n, 3) wave strengths
        w = w * (lam[..., None] * s.permute(0, 2, 1))
        return central - 0.5 * torch.einsum('bnij,bnj->bin', R, w)

    def num_params(self):
        return sum(q.numel() for q in self.parameters())

    def save(self, path, **extra):
        cfg = dict(p=self.p, mode=self.mode, hidden=self.hidden, depth=self.depth, s_min=self.s_min)
        torch.save(dict(cfg=cfg, state=self.state_dict(), extra=extra), path)

    @classmethod
    def load(cls, path, device='cpu'):
        ck = torch.load(path, map_location=device, weights_only=True)
        m = cls(**ck['cfg'])
        m.load_state_dict(ck['state'])
        m.extra = ck.get('extra', {})
        return m.to(device)
