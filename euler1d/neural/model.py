'''
NFV model, as in the paper: a CNN with kernel a over space and b time slices as channels,
then 1x1 convs. Euler has 3 variables, so channels in = 3b and out = 3.
Layers: conv(a) + (depth-1) x 1x1 conv (width hidden) + 1x1 conv out. No output clipping.
Params: 45*a*b + 1263 for depth 6, hidden 15.
'''
import torch
from torch import nn
from torch.nn import init

from euler1d.neural import scaling

ACT = {'ELU': nn.ELU, 'ReLU': nn.ReLU, 'Tanh': nn.Tanh}


class NFVModel(nn.Module):
    def __init__(self, a=2, b=1, hidden=15, depth=6, act='ELU', dtype=torch.float32):
        super().__init__()
        if a % 2:
            raise ValueError('a must be even')
        self.a, self.b, self.hidden, self.depth, self.act = a, b, hidden, depth, act
        self.extra = {}
        A = ACT[act]
        layers = [nn.Conv1d(3 * b, hidden, a, dtype=dtype), A()]
        for _ in range(depth - 1):
            layers += [nn.Conv1d(hidden, hidden, 1, dtype=dtype), A()]
        layers.append(nn.Conv1d(hidden, 3, 1, dtype=dtype))
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

    @property
    def g(self):
        return self.a // 2

    def forward(self, x):
        return self.net(x)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def save(self, path, **extra):
        meta = dict(a=self.a, b=self.b, hidden=self.hidden, depth=self.depth, act=self.act,
                    rho_ref=scaling.RHO_REF, a_ref=scaling.A_REF, **extra)
        torch.save(dict(meta=meta, state=self.state_dict()), path)

    @classmethod
    def load(cls, path, dtype=torch.float32):
        ck = torch.load(path, map_location='cpu', weights_only=True)
        m = ck['meta']
        if m['rho_ref'] != scaling.RHO_REF or m['a_ref'] != scaling.A_REF:
            raise ValueError('checkpoint was trained with different scaling constants')
        model = cls(m['a'], m['b'], m['hidden'], m['depth'], m['act'], dtype)
        model.load_state_dict({k: v.to(dtype) for k, v in ck['state'].items()})
        model.extra = {k: v for k, v in m.items() if k not in
                       ('a', 'b', 'hidden', 'depth', 'act', 'rho_ref', 'a_ref')}
        return model