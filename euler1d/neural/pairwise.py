'''
Pairwise-exchange NFV for large CFL: cell i exchanges with cells i+d, |d| <= k, in one step.

mode 'free':      CNN (each source sees cells i-2k..i+2k); antisymmetric rates P[i, i+d], d = 1..k,
                  all 3 variables (FS units).
mode 'transport': same CNN; fractions T[i -> i+d], d = -k..k (softmax incl. stay), move the whole state Q_i;
                  plus antisymmetric rates for momentum and energy only (pressure part).
mode 'attn':      as 'transport', but T comes from attention. Encoder (cells j-k..j+k) -> embedding h_j;
                  query q_i = Wq h_i (source), key k_j = Wk h_j (destination),
                  score s_ij = q_i.k_j / sqrt(dk) + b_{j-i} [- beta (j - i - u_i dt/dx)^2 if prior];
                  T[i -> j] = softmax over destinations j; value = Q_i, scattered (T transposed).
                  P[i, i+d] = g(h_i, h_{i+d}, e_d) - g(h_{i+d}, h_i, e_{-d}).
transport / attn: rho_new_j = sum_i T[i -> j] rho_i >= 0. Need the global dt (r0 = cfl / S_GLOBAL).

forward returns the derived interface flux F_{m+1/2} = sum of exchanges crossing m+1/2 (/ FS), so
the update Q - r (F_{i+1/2} - F_{i-1/2}) equals the pairwise update exactly. a = 6k, g = 3k.

mode 'split':     f = u (rho, rho u, E) + (0, p, pu), each part consistent by construction.
                  advection: u_bar_i = sum_j a_ij u_j (attention, convex); shift s = u_bar_i dt/dx;
                  T = (1 - c) hat(s) + c/2 (hat(s - 1) + hat(s + 1)), hat = linear-interpolation split,
                  c in [0, C_MAX] learned spreading; mean hop = s exactly. |s| <= k - 2, so k >= 3.
                  pressure: interface flux (0, sum_j w_j p_j, sum_j w_j p_j u_j), w convex (interface query).
                  dissipation: wide-stencil, per width w = 1..k adds nu_w (Q_{i+w} - 2 Q_i + Q_{i-w}) as
                  interface fluxes; nu_w = nu_tot pi_w, nu_tot <= NU_MAX, pi = softmax (monotone if sum <= 1/2).
                  init = wide Lax-Friedrichs-like scheme (uniform pressure weights, uniform widths, nu_tot 0.25),
                  the only init found stable without the limiter at CFL ~2.
                  uniform flow gives f(Q) exactly. Optional limiter (heuristic): scales pressure + dissipation
                  per face so cells stay at rho, rho e > EPS_LIM x the advection-only state. a = 6k + 2.
'''
import math
import torch
from torch import nn
from torch.nn import init
from torch.nn.functional import softplus

from euler1d.neural import scaling
from euler1d.neural.model import ACT
from infrastructure.nfv_dataset import S_GLOBAL

P_STAY = 0.9
BETA0 = 0.5
EMB_D = 4
GAMMA = 1.4
NU_MAX, NU0 = 0.5, 0.25
C_MAX, C0 = 0.5, 0.05
EPS_LIM = 0.01


def _logit(p):
    return math.log(p / (1 - p))


def _rhoe(S):
    return S[:, 2] - 0.5 * S[:, 1] ** 2 / S[:, 0].clamp(min=1e-12)


def _ok(S, R):
    '''S, R: (B, 3, n) states; S valid relative to the reference R'''
    return (S[:, 0] > EPS_LIM * R[:, 0]) & (S[:, 0] > 0) & (_rhoe(S) > EPS_LIM * _rhoe(R))


def _cross(T, Q, k, n):
    '''amount crossing interfaces 3k-1+q, q < n, from fractions T (B, 2k+1, L) of source cells 2k + p'''
    A = 0
    for s in range(k):
        out_r = T[:, k + s + 1:].sum(1, keepdim=True)[..., k - 1 - s:k - 1 - s + n]
        out_l = T[:, :k - s].sum(1, keepdim=True)[..., k + s:k + s + n]
        A = A + out_r * Q[..., 3 * k - 1 - s:3 * k - 1 - s + n] - out_l * Q[..., 3 * k + s:3 * k + s + n]
    return A


def _sc(x, v):
    return torch.as_tensor(v, dtype=x.dtype, device=x.device).view(1, 3, 1)


class PairModel(nn.Module):
    kind = 'pair'

    def __init__(self, k=3, b=1, hidden=15, depth=6, act='ELU', mode='free', cfl=0.5, dtype=torch.float32,
                 emb=16, dk=8, prior=0, limit=1):
        super().__init__()
        self.k, self.b, self.hidden, self.depth, self.act = k, b, hidden, depth, act
        self.mode, self.cfl, self.emb, self.dk, self.prior = mode, cfl, emb, dk, int(prior)
        self.limit = int(limit)
        self.a = 6 * k + 2 if mode == 'split' else 6 * k
        self.extra = {}
        A = ACT[act]
        if mode in ('attn', 'split'):
            D = emb
            self.enc = nn.Sequential(nn.Conv1d(3 * b, D, 2 * k + 1, dtype=dtype), A(),
                                     nn.Conv1d(D, D, 1, dtype=dtype), A(),
                                     nn.Conv1d(D, D, 1, dtype=dtype), A())
            self.wq = nn.Conv1d(D, dk, 1, bias=False, dtype=dtype)
            self.wk = nn.Conv1d(D, dk, 1, bias=False, dtype=dtype)
            self.rel = nn.Parameter(torch.zeros(2 * k + 1, dtype=dtype))
        if mode == 'split':
            if k < 3:
                raise ValueError('split mode needs k >= 3')
            self.wqi = nn.Conv1d(2 * D, dk, 1, bias=False, dtype=dtype)
            self.cb = nn.Parameter(torch.zeros(2 * k, dtype=dtype))
            self.whead = nn.Conv1d(D, 1, 1, dtype=dtype)
            self.wnu = nn.Conv1d(2 * D, 1 + k, 1, dtype=dtype)
        elif mode == 'attn':
            self.off = nn.Parameter(torch.zeros(2 * k + 1, EMB_D, dtype=dtype))
            self.pg = nn.Sequential(nn.Conv1d(2 * D + EMB_D, D, 1, dtype=dtype), A(), nn.Conv1d(D, 2, 1, dtype=dtype))
            self.beta = nn.Parameter(torch.tensor(math.log(math.expm1(BETA0)), dtype=dtype))
        else:
            out = 3 * k if mode == 'free' else 2 * k + 1 + 2 * k
            layers = [nn.Conv1d(3 * b, hidden, 4 * k + 1, dtype=dtype), A()]
            for _ in range(depth - 1):
                layers += [nn.Conv1d(hidden, hidden, 1, dtype=dtype), A()]
            layers.append(nn.Conv1d(hidden, out, 1, dtype=dtype))
            self.net = nn.Sequential(*layers)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                if self.act == 'ReLU':
                    init.kaiming_normal_(m.weight, nonlinearity='relu')
                elif self.act == 'ELU':
                    init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                else:
                    init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)
        stay = math.log(P_STAY * 2 * self.k / (1 - P_STAY))
        with torch.no_grad():
            if self.mode == 'split':
                k = self.k
                init.zeros_(self.wq.weight)
                init.zeros_(self.wqi.weight)
                self.rel[k] = stay
                init.zeros_(self.whead.weight)
                self.whead.bias.fill_(_logit(C0 / C_MAX))
                init.zeros_(self.wnu.weight)
                self.wnu.bias.zero_()
                self.wnu.bias[0] = _logit(NU0 / NU_MAX)
                return
            if self.mode == 'attn':
                init.zeros_(self.wq.weight)
                init.zeros_(self.pg[-1].weight)
                init.normal_(self.off, std=0.1)
                self.rel[self.k] = stay
                return
            init.zeros_(self.net[-1].weight)
            if self.mode == 'transport':
                self.net[-1].bias[self.k] = stay

    @property
    def g(self):
        return self.a // 2

    @property
    def r0(self):
        return self.cfl / S_GLOBAL

    def exchanges(self, x):
        '''x: (B, 3b, W) scaled -> P: (B, k, 3, L) rates for d = 1..k (scaled),
        T: (B, 2k+1, L) fractions for d = -k..k or None; output p belongs to cell 2k + p'''
        if self.mode == 'attn':
            return self.attend(x)
        h = self.net(x)
        B, L, k = h.shape[0], h.shape[-1], self.k
        if self.mode == 'free':
            return h.view(B, k, 3, L), None
        P = h[:, 2 * k + 1:].view(B, k, 2, L)
        P = torch.cat([torch.zeros_like(P[:, :, :1]), P], dim=2)
        return P, h[:, :2 * k + 1].softmax(dim=1)

    def scores(self, x):
        '''attention scores s[:, k + d, p] for source cell 2k + p and destination 2k + p + d; embeddings H'''
        k, L = self.k, x.shape[-1] - 4 * self.k
        H = self.enc(x)
        q, K = self.wq(H)[..., k:k + L], self.wk(H)
        s = torch.stack([(q * K[..., k + d:k + d + L]).sum(1) for d in range(-k, k + 1)], dim=1)
        s = s / math.sqrt(self.dk) + self.rel.view(1, -1, 1)
        if self.prior:
            Q = x[:, -3:] * _sc(x, scaling.QS)
            u = (Q[:, 1] / Q[:, 0])[..., 2 * k:2 * k + L]
            disp = (u * self.r0).clamp(-k - 1, k + 1)
            d = torch.arange(-k, k + 1, dtype=x.dtype, device=x.device).view(1, -1, 1)
            s = s - softplus(self.beta) * (d - disp[:, None]) ** 2
        return s, H

    def attend(self, x):
        k, L = self.k, x.shape[-1] - 4 * self.k
        s, H = self.scores(x)
        T = s.softmax(dim=1)
        B = H.shape[0]
        e = lambda d: self.off[d + k].view(1, -1, 1).expand(B, -1, L)
        hi = H[..., k:k + L]
        P = []
        for d in range(1, k + 1):
            hj = H[..., k + d:k + d + L]
            P.append(self.pg(torch.cat([hi, hj, e(d)], 1)) - self.pg(torch.cat([hj, hi, e(-d)], 1)))
        P = torch.stack(P, dim=1)
        return torch.cat([torch.zeros_like(P[:, :, :1]), P], dim=2), T

    def split_parts(self, x):
        '''Q (B, 3, W) physical; T (B, 2k+1, L) fractions and shift (B, L) of source cells 2k + p;
        A amounts and Fc pressure + dissipation flux (physical) at interfaces 3k-1+q, q < W - 6k + 1'''
        k, W, sq = self.k, x.shape[-1], math.sqrt(self.dk)
        L, n2 = W - 4 * k, W - 6 * k + 1
        Q = x[:, -3:] * _sc(x, scaling.QS)
        u = Q[:, 1] / Q[:, 0].clamp(min=1e-6)
        p = (GAMMA - 1) * (Q[:, 2] - 0.5 * Q[:, 1] * u)
        H = self.enc(x)
        K = self.wk(H)
        q = self.wq(H)[..., k:k + L]
        s = torch.stack([(q * K[..., k + d:k + d + L]).sum(1) for d in range(-k, k + 1)], 1) / sq
        a = (s + self.rel.view(1, -1, 1)).softmax(1)
        ub = sum(a[:, d + k] * u[..., 2 * k + d:2 * k + d + L] for d in range(-k, k + 1))
        c = C_MAX * torch.sigmoid(self.whead(H[..., k:k + L]))
        sh = (ub * self.r0).clamp(-(k - 2), k - 2)
        d = torch.arange(-k, k + 1, dtype=x.dtype, device=x.device).view(1, -1, 1)
        hat = lambda s: (1 - (d - s[:, None]).abs()).clamp(min=0)
        T = (1 - c) * hat(sh) + c / 2 * (hat(sh - 1) + hat(sh + 1))
        A = _cross(T, Q, k, n2)
        hi = torch.cat([H[..., 2 * k - 1:2 * k - 1 + n2], H[..., 2 * k:2 * k + n2]], 1)
        qi = self.wqi(hi)
        offs = range(-k + 1, k + 1)
        s = torch.stack([(qi * K[..., 2 * k - 1 + o:2 * k - 1 + o + n2]).sum(1) for o in offs], 1) / sq
        w = (s + self.cb.view(1, -1, 1)).softmax(1)
        cell = lambda v, o: v[..., 3 * k - 1 + o:3 * k - 1 + o + n2]
        pb = sum(w[:, o + k - 1] * cell(p, o) for o in offs)
        pub = sum(w[:, o + k - 1] * cell(p * u, o) for o in offs)
        z = self.wnu(hi)
        nu = NU_MAX * torch.sigmoid(z[:, :1]) * z[:, 1:].softmax(1)
        G = 0
        for w in range(1, k + 1):
            Dw = Q[..., w:] - Q[..., :-w]
            G = G + nu[:, w - 1:w] * sum(Dw[..., 3 * k - 1 - j:3 * k - 1 - j + n2] for j in range(w))
        Fc = torch.stack([torch.zeros_like(pb), pb, pub], 1) - G / self.r0
        return Q, T, sh, A, Fc

    def theta(self, Q, A, Fc):
        '''per-face scale of Fc (B, 1, n) keeping cells valid relative to the advection-only state'''
        k, r, n = self.k, self.r0, A.shape[-1] - 2
        with torch.no_grad():
            Ql = Q[..., 3 * k:3 * k + n + 1] - (A[..., 1:] - A[..., :-1])
            C = r * (Fc[..., :-1] - Fc[..., 1:])
            lo, hi = torch.zeros_like(Ql[:, :1]), torch.ones_like(Ql[:, :1])
            for _ in range(16):
                mid = (lo + hi) / 2
                f = _ok(Ql + mid * C, Ql)[:, None]
                lo, hi = torch.where(f, mid, lo), torch.where(f, hi, mid)
            tc = torch.where(_ok(Ql + C, Ql)[:, None], torch.ones_like(lo), lo)
            tf = torch.minimum(tc[..., :-1], tc[..., 1:])
            Fn = Fc[..., 1:n + 1]
            for _ in range(2):
                U = Ql[..., 1:n] + r * (tf[..., :-1] * Fn[..., :-1] - tf[..., 1:] * Fn[..., 1:])
                bad = ~_ok(U, Ql[..., 1:n])
                z = nn.functional.pad(bad, (0, 1)) | nn.functional.pad(bad, (1, 0))
                tf = torch.where(z[:, None], torch.zeros_like(tf), tf)
        return tf

    def forward(self, x):
        k = self.k
        if self.mode == 'split':
            Q, T, sh, A, Fc = self.split_parts(x)
            n = A.shape[-1] - 2
            th = self.theta(Q, A, Fc) if self.limit else 1
            F = A[..., 1:n + 1] / self.r0 + th * Fc[..., 1:n + 1]
            return F / _sc(x, scaling.FS)
        n = x.shape[-1] - 6 * k + 1
        P, T = self.exchanges(x)
        F = 0
        for d in range(1, k + 1):
            for s in range(d):
                F = F + P[:, d - 1, :, k - 1 - s:k - 1 - s + n]
        if T is None:
            return F
        Q = x[:, -3:] * _sc(x, scaling.QS)
        return F + _cross(T, Q, k, n) / self.r0 / _sc(x, scaling.FS)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def save(self, path, **extra):
        meta = dict(kind=self.kind, k=self.k, a=self.a, b=self.b, hidden=self.hidden, depth=self.depth, act=self.act,
                    mode=self.mode, emb=self.emb, dk=self.dk, prior=self.prior, limit=self.limit,
                    rho_ref=scaling.RHO_REF, a_ref=scaling.A_REF, **extra)
        meta.setdefault('cfl', self.cfl)
        torch.save(dict(meta=meta, state=self.state_dict()), path)

    @classmethod
    def from_ckpt(cls, ck, dtype=torch.float32):
        m = ck['meta']
        model = cls(m['k'], m['b'], m['hidden'], m['depth'], m['act'], m['mode'], m['cfl'], dtype,
                    m.get('emb', 16), m.get('dk', 8), m.get('prior', 0), m.get('limit', 1))
        model.load_state_dict({k: v.to(dtype) for k, v in ck['state'].items()})
        model.extra = {k: v for k, v in m.items() if k not in
                       ('kind', 'k', 'a', 'b', 'hidden', 'depth', 'act', 'mode', 'emb', 'dk', 'prior', 'limit',
                        'rho_ref', 'a_ref')}
        return model
