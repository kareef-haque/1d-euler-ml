'''
Pairwise-exchange NFV for large CFL: cell i exchanges with cells i+d, |d| <= k, in one step.

Each source cell i sees cells i-2k..i+2k (conv kernel 4k+1).
mode 'free':      antisymmetric exchange rates P[i, i+d], d = 1..k, all 3 variables (FS units).
mode 'transport': fractions T[i -> i+d], d = -k..k (softmax incl. stay), move the whole state Q_i;
                  plus antisymmetric rates for momentum and energy only (pressure part).
                  rho_new_j = sum_i T[i -> j] rho_i >= 0. Needs the global dt (r0 = cfl / S_GLOBAL).

forward returns the derived interface flux F_{m+1/2} = sum of exchanges crossing m+1/2 (/ FS), so
the update Q - r (F_{i+1/2} - F_{i-1/2}) equals the pairwise update exactly. a = 6k, g = 3k.
'''
import math
import torch
from torch import nn
from torch.nn import init

from euler1d.neural import scaling
from euler1d.neural.model import ACT
from infrastructure.nfv_dataset import S_GLOBAL

P_STAY = 0.9


def _sc(x, v):
    return torch.as_tensor(v, dtype=x.dtype, device=x.device).view(1, 3, 1)


class PairModel(nn.Module):
    kind = 'pair'

    def __init__(self, k=3, b=1, hidden=15, depth=6, act='ELU', mode='free', cfl=0.5, dtype=torch.float32):
        super().__init__()
        self.k, self.b, self.hidden, self.depth, self.act = k, b, hidden, depth, act
        self.mode, self.cfl = mode, cfl
        self.a = 6 * k
        self.extra = {}
        out = 3 * k if mode == 'free' else 2 * k + 1 + 2 * k
        A = ACT[act]
        layers = [nn.Conv1d(3 * b, hidden, 4 * k + 1, dtype=dtype), A()]
        for _ in range(depth - 1):
            layers += [nn.Conv1d(hidden, hidden, 1, dtype=dtype), A()]
        layers.append(nn.Conv1d(hidden, out, 1, dtype=dtype))
        self.net = nn.Sequential(*layers)
        self._init()

    def _init(self):
        for m in self.net.modules():
            if isinstance(m, nn.Conv1d):
                if self.act == 'ReLU':
                    init.kaiming_normal_(m.weight, nonlinearity='relu')
                elif self.act == 'ELU':
                    init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                else:
                    init.xavier_normal_(m.weight)
                init.zeros_(m.bias)
        init.zeros_(self.net[-1].weight)
        if self.mode == 'transport':
            with torch.no_grad():
                self.net[-1].bias[self.k] = math.log(P_STAY * 2 * self.k / (1 - P_STAY))

    @property
    def g(self):
        return 3 * self.k

    @property
    def r0(self):
        return self.cfl / S_GLOBAL

    def exchanges(self, x):
        '''x: (B, 3b, W) scaled -> P: (B, k, 3, L) rates for d = 1..k (scaled),
        T: (B, 2k+1, L) fractions for d = -k..k or None; output p belongs to cell 2k + p'''
        h = self.net(x)
        B, L, k = h.shape[0], h.shape[-1], self.k
        if self.mode == 'free':
            return h.view(B, k, 3, L), None
        P = h[:, 2 * k + 1:].view(B, k, 2, L)
        P = torch.cat([torch.zeros_like(P[:, :, :1]), P], dim=2)
        return P, h[:, :2 * k + 1].softmax(dim=1)

    def forward(self, x):
        k = self.k
        n = x.shape[-1] - 6 * k + 1
        P, T = self.exchanges(x)
        F = 0
        for d in range(1, k + 1):
            for s in range(d):
                F = F + P[:, d - 1, :, k - 1 - s:k - 1 - s + n]
        if T is None:
            return F
        Q = x[:, -3:] * _sc(x, scaling.QS)
        A = 0
        for s in range(k):
            out_r = T[:, k + s + 1:].sum(1, keepdim=True)[..., k - 1 - s:k - 1 - s + n]
            out_l = T[:, :k - s].sum(1, keepdim=True)[..., k + s:k + s + n]
            A = A + out_r * Q[..., 3 * k - 1 - s:3 * k - 1 - s + n] - out_l * Q[..., 3 * k + s:3 * k + s + n]
        return F + A / self.r0 / _sc(x, scaling.FS)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def save(self, path, **extra):
        meta = dict(kind=self.kind, k=self.k, a=self.a, b=self.b, hidden=self.hidden, depth=self.depth, act=self.act,
                    mode=self.mode, rho_ref=scaling.RHO_REF, a_ref=scaling.A_REF, **extra)
        meta.setdefault('cfl', self.cfl)
        torch.save(dict(meta=meta, state=self.state_dict()), path)

    @classmethod
    def from_ckpt(cls, ck, dtype=torch.float32):
        m = ck['meta']
        model = cls(m['k'], m['b'], m['hidden'], m['depth'], m['act'], m['mode'], m['cfl'], dtype)
        model.load_state_dict({k: v.to(dtype) for k, v in ck['state'].items()})
        model.extra = {k: v for k, v in m.items() if k not in
                       ('kind', 'k', 'a', 'b', 'hidden', 'depth', 'act', 'mode', 'rho_ref', 'a_ref')}
        return model
