'''
Weighted-stencil NFV: the network only chooses convex weights, the flux stays a Riemann solver.

Interface i+1/2, window c0..c5 = cells i-2..i+3 (a = 6, g = 3).
Left state: candidates p0, p1, p2 (WENO5 sub-stencil parabolas centred on c2) and p3 = c2.
Right state: the mirror image (centred on c3).
Weights: softmax over the 4 candidates, per conserved variable. Left logits = net(window),
right logits = net(mirrored window), so the scheme is reflection symmetric.
Positivity fallback: a reconstructed state with rho or p below POS times the cell value is
replaced by the cell value (first order there).
Flux: HLLC or Rusanov of (QL, QR) in physical units.
A posteriori fallback (safe_flux, MOOD style): where an update leaves rho or p below POS times the
smallest old value of the cell and its neighbours, both faces of that cell use first-order Riemann fluxes.
Integrator: SSP-RK3 by default (forward Euler with high-order stencils is unstable), see stepper.advance.

Every candidate reproduces constants and the weights sum to 1, so F(Q,..,Q) = f(Q) exactly.
Input / output scaling as NFVModel: forward takes Q / QS and returns F / FS.
'''
import math
import torch
from torch import nn
from torch.nn import init

from euler1d.neural import scaling
from euler1d.neural.model import ACT

RHO_MIN, P_MIN, EPS = 1e-6, 1e-2, 1e-9
NC = 4
P_CONST = 0.95
POS = 0.1


def _sc(x, v):
    return torch.as_tensor(v, dtype=x.dtype, device=x.device).view(1, 3, 1)


def rho_p(Q, gamma):
    return Q[:, 0], (gamma - 1.0) * (Q[:, 2] - 0.5 * Q[:, 1]**2 / Q[:, 0])


def positive(Qr, Qc, gamma):
    '''Qr: reconstructed, Qc: cell value'''
    r, p = rho_p(Qr, gamma)
    rc, pc = rho_p(Qc, gamma)
    ok = (r > POS * rc) & (p > POS * pc)
    return torch.where(ok[:, None], Qr, Qc)


def prim(Q, gamma):
    rho = Q[:, 0].clamp(min=RHO_MIN)
    u = Q[:, 1] / rho
    p = ((gamma - 1.0) * (Q[:, 2] - 0.5 * rho * u**2)).clamp(min=P_MIN)
    return rho, u, p


def phys(Q, u, p):
    return torch.stack([Q[:, 1], Q[:, 1] * u + p, u * (Q[:, 2] + p)], dim=1)


def hllc(QL, QR, gamma=1.4):
    '''QL, QR: (B, 3, n) -> (B, 3, n), same wave cases as schemes/flux/hllc.py'''
    rL, uL, pL = prim(QL, gamma)
    rR, uR, pR = prim(QR, gamma)
    aL, aR = torch.sqrt(gamma * pL / rL), torch.sqrt(gamma * pR / rR)
    SL = torch.minimum(uL - aL, uR - aR)
    SR = torch.maximum(uL + aL, uR + aR)
    Ss = (pR - pL + rL * uL * (SL - uL) - rR * uR * (SR - uR)) / (rL * (SL - uL) - rR * (SR - uR))

    def star(Q, r, u, p, S):
        d = S - Ss
        d = torch.where(d >= 0, d.clamp(min=EPS), d.clamp(max=-EPS))
        c = r * (S - u) / d
        return c[:, None] * torch.stack([torch.ones_like(c), Ss, Q[:, 2] / r + (Ss - u) * (Ss + p / (r * (S - u)))], dim=1)

    FL, FR = phys(QL, uL, pL), phys(QR, uR, pR)
    FLs = FL + SL[:, None] * (star(QL, rL, uL, pL, SL) - QL)
    FRs = FR + SR[:, None] * (star(QR, rR, uR, pR, SR) - QR)
    w = lambda m: m[:, None]
    return torch.where(w(SL >= 0), FL, torch.where(w(SR <= 0), FR, torch.where(w(Ss >= 0), FLs, FRs)))


def rusanov(QL, QR, gamma=1.4):
    rL, uL, pL = prim(QL, gamma)
    rR, uR, pR = prim(QR, gamma)
    s = torch.maximum(uL.abs() + torch.sqrt(gamma * pL / rL), uR.abs() + torch.sqrt(gamma * pR / rR))
    return 0.5 * (phys(QL, uL, pL) + phys(QR, uR, pR)) - 0.5 * s[:, None] * (QR - QL)


RIEMANN = {'hllc': hllc, 'rusanov': rusanov}


def candidates(Q):
    '''Q: (B, 3, n + 5) -> left, right: (B, 3, NC, n)'''
    n = Q.shape[-1] - 5
    c = [Q[..., k:k + n] for k in range(6)]
    left = [c[2] / 3 + 5 * c[3] / 6 - c[4] / 6,
            -c[1] / 6 + 5 * c[2] / 6 + c[3] / 3,
            c[0] / 3 - 7 * c[1] / 6 + 11 * c[2] / 6,
            c[2]]
    right = [c[3] / 3 + 5 * c[2] / 6 - c[1] / 6,
             -c[4] / 6 + 5 * c[3] / 6 + c[2] / 3,
             c[5] / 3 - 7 * c[4] / 6 + 11 * c[3] / 6,
             c[3]]
    return torch.stack(left, dim=2), torch.stack(right, dim=2)


def mirror(x):
    '''reverse space and flip the sign of every momentum channel'''
    s = torch.ones(x.shape[1], dtype=x.dtype, device=x.device)
    s[1::3] = -1
    return x.flip(-1) * s.view(1, -1, 1)


class WSModel(nn.Module):
    kind = 'ws'

    def __init__(self, b=1, hidden=15, depth=6, act='ELU', riemann='hllc', gamma=1.4, dtype=torch.float32, integrator='rk3'):
        super().__init__()
        self.a, self.b, self.hidden, self.depth, self.act = 6, b, hidden, depth, act
        self.riemann, self.gamma, self.integrator = riemann, gamma, integrator
        self.extra = {}
        A = ACT[act]
        layers = [nn.Conv1d(3 * b, hidden, 6, dtype=dtype), A()]
        for _ in range(depth - 1):
            layers += [nn.Conv1d(hidden, hidden, 1, dtype=dtype), A()]
        layers.append(nn.Conv1d(hidden, 3 * NC, 1, dtype=dtype))
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
        last = self.net[-1]
        init.zeros_(last.weight)
        with torch.no_grad():
            last.bias.view(3, NC)[:, -1] = math.log(P_CONST * (NC - 1) / (1 - P_CONST))

    @property
    def g(self):
        return 3

    def weights(self, x):
        '''x: (B, 3b, n + 5) scaled -> wL, wR: (B, 3, NC, n)'''
        B, n = x.shape[0], x.shape[-1] - 5
        wL = self.net(x).view(B, 3, NC, n).softmax(dim=2)
        wR = self.net(mirror(x)).flip(-1).view(B, 3, NC, n).softmax(dim=2)
        return wL, wR

    def states(self, x):
        Q = x[:, -3:] * _sc(x, scaling.QS)
        cL, cR = candidates(Q)
        wL, wR = self.weights(x)
        QL = positive((wL * cL).sum(2), cL[:, :, -1], self.gamma)
        QR = positive((wR * cR).sum(2), cR[:, :, -1], self.gamma)
        return QL, QR

    def forward(self, x):
        QL, QR = self.states(x)
        F = RIEMANN[self.riemann](QL, QR, self.gamma)
        return F / _sc(F, scaling.FS)

    def safe_flux(self, h, F, U):
        '''h: padded old state, F: flux, U: updated cells (all physical) -> F with fallback faces'''
        n = h.shape[-1] - 6
        r, p = rho_p(h, self.gamma)
        lo = lambda v: torch.stack([v[:, 2:n + 2], v[:, 3:n + 3], v[:, 4:n + 4]]).amin(0)
        rn, pn = rho_p(U, self.gamma)
        ok = (rn > POS * lo(r)) & (pn > POS * lo(p))
        bad = ~(ok & torch.isfinite(U).all(1))
        if not bad.any():
            return F
        f = nn.functional.pad(bad, (1, 0)) | nn.functional.pad(bad, (0, 1))
        low = RIEMANN[self.riemann](h[..., 2:n + 3], h[..., 3:n + 4], self.gamma)
        return torch.where(f[:, None], low, F)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def save(self, path, **extra):
        meta = dict(kind=self.kind, a=self.a, b=self.b, hidden=self.hidden, depth=self.depth, act=self.act,
                    riemann=self.riemann, integrator=self.integrator, rho_ref=scaling.RHO_REF, a_ref=scaling.A_REF, **extra)
        meta.setdefault('gamma', self.gamma)
        torch.save(dict(meta=meta, state=self.state_dict()), path)

    @classmethod
    def from_ckpt(cls, ck, dtype=torch.float32):
        m = ck['meta']
        model = cls(m['b'], m['hidden'], m['depth'], m['act'], m['riemann'], m.get('gamma', 1.4), dtype, m.get('integrator', 'rk3'))
        model.load_state_dict({k: v.to(dtype) for k, v in ck['state'].items()})
        model.extra = {k: v for k, v in m.items() if k not in
                       ('kind', 'a', 'b', 'hidden', 'depth', 'act', 'riemann', 'integrator', 'rho_ref', 'a_ref')}
        return model
