'''
Classical DG for 1D Euler (torch, float64, differentiable).

U: (B, 3, N, P) modal coefficients, P = p + 1, scaled Legendre basis on xi in [-1/2, 1/2]:
   phi_0 = 1, phi_1 = 2 sqrt(3) xi, phi_2 = sqrt(5) (6 xi^2 - 1/2); (1/h) int phi_i phi_j = delta_ij.
Semi-discrete: dU_i/dt = (1/h) [ int f(u_h) dphi_i/dxi dxi - (phi_i(1/2) F_R - phi_i(-1/2) F_L) ].
Edge flux F(u-, u+): Rusanov or HLLC (euler1d.neural.weighted). SSP-RK3; after every stage the TVB-minmod
limiter (Cockburn-Shu, characteristic variables, P1 slopes) and the Zhang-Shu positivity limiter.
Ghost cells: padded U (B, 3, N + 2, P) from a bc function.
'''
import math
import numpy as np
import torch

D = torch.float64
GAMMA = 1.4
EPS = 1e-14       # safety floor only; the weighted.py fluxes floor p at 1e-2 (SI units), which breaks the 123 problem


def gauss(n):
    x, w = np.polynomial.legendre.leggauss(n)
    return torch.tensor(x / 2, dtype=D), torch.tensor(w / 2, dtype=D)


def _legendre(P, s):
    '''Legendre P_k(s) and P_k'(s), k < P, by recurrence'''
    v, d = [torch.ones_like(s), s], [torch.zeros_like(s), torch.ones_like(s)]
    for k in range(1, P - 1):
        v.append(((2 * k + 1) * s * v[k] - k * v[k - 1]) / (k + 1))
        d.append(d[k - 1] + (2 * k + 1) * v[k])
    return v[:P], d[:P]


def phi(P, xi):
    '''phi_k = sqrt(2k+1) P_k(2 xi): (P, *xi.shape)'''
    xi = torch.as_tensor(xi, dtype=D)
    v, _ = _legendre(P, 2 * xi)
    return torch.stack([math.sqrt(2 * k + 1) * v[k] for k in range(P)])


def dphi(P, xi):
    '''d phi_k / d xi'''
    xi = torch.as_tensor(xi, dtype=D)
    _, d = _legendre(P, 2 * xi)
    return torch.stack([2 * math.sqrt(2 * k + 1) * d[k] for k in range(P)])


def at(U, xi):
    '''values at reference points xi (q,): (B, 3, N, q)'''
    return torch.einsum('bcnp,pq->bcnq', U, phi(U.shape[-1], xi).to(U))


def euler_flux(Q, gamma=GAMMA):
    rho, m, E = Q[:, 0], Q[:, 1], Q[:, 2]
    u = m / rho
    p = (gamma - 1) * (E - 0.5 * m * u)
    return torch.stack([m, m * u + p, u * (E + p)], 1)


def pressure(Q, gamma=GAMMA):
    return (gamma - 1) * (Q[:, 2] - 0.5 * Q[:, 1]**2 / Q[:, 0])


def _prim(Q, gamma=GAMMA):
    rho = Q[:, 0].clamp(min=EPS)
    u = Q[:, 1] / rho
    p = ((gamma - 1) * (Q[:, 2] - 0.5 * rho * u**2)).clamp(min=EPS)
    return rho, u, p


def rusanov(QL, QR, gamma=GAMMA):
    rL, uL, pL = _prim(QL, gamma)
    rR, uR, pR = _prim(QR, gamma)
    s = torch.maximum(uL.abs() + torch.sqrt(gamma * pL / rL), uR.abs() + torch.sqrt(gamma * pR / rR))
    return 0.5 * (euler_flux(QL, gamma) + euler_flux(QR, gamma)) - 0.5 * s[:, None] * (QR - QL)


def hllc(QL, QR, gamma=GAMMA):
    '''HLLC with Davis wave-speed estimates; QL, QR (B, 3, n)'''
    rL, uL, pL = _prim(QL, gamma)
    rR, uR, pR = _prim(QR, gamma)
    aL, aR = torch.sqrt(gamma * pL / rL), torch.sqrt(gamma * pR / rR)
    SL, SR = torch.minimum(uL - aL, uR - aR), torch.maximum(uL + aL, uR + aR)
    Ss = (pR - pL + rL * uL * (SL - uL) - rR * uR * (SR - uR)) / (rL * (SL - uL) - rR * (SR - uR))

    def star(Q, r, u, p, S):
        d = S - Ss
        d = torch.where(d >= 0, d.clamp(min=1e-300), d.clamp(max=-1e-300))
        k = r * (S - u) / d
        return k[:, None] * torch.stack([torch.ones_like(k), Ss, Q[:, 2] / r + (Ss - u) * (Ss + p / (r * (S - u)))], 1)

    FL, FR = euler_flux(QL, gamma), euler_flux(QR, gamma)
    FLs = FL + SL[:, None] * (star(QL, rL, uL, pL, SL) - QL)
    FRs = FR + SR[:, None] * (star(QR, rR, uR, pR, SR) - QR)
    w = lambda m: m[:, None]
    return torch.where(w(SL >= 0), FL, torch.where(w(SR <= 0), FR, torch.where(w(Ss >= 0), FLs, FRs)))


FLUX = {'rusanov': rusanov, 'hllc': hllc}


# ---------------------------------------------------------------- projection of exact solutions

def wave_points(ex, t):
    '''positions where the exact Riemann solution is non-smooth at time t'''
    g = ex.gamma
    sh = lambda Pk, ak: math.sqrt((g + 1) / (2 * g) * ex.P_star / Pk + (g - 1) / (2 * g)) * ak
    s = [ex.u_star]
    if ex.P_star > ex.P_L:
        s.append(ex.u_L - sh(ex.P_L, ex.a_L))
    else:
        s += [ex.u_L - ex.a_L, ex.u_star - ex.a_L * (ex.P_star / ex.P_L) ** ((g - 1) / (2 * g))]
    if ex.P_star > ex.P_R:
        s.append(ex.u_R + sh(ex.P_R, ex.a_R))
    else:
        s += [ex.u_R + ex.a_R, ex.u_star + ex.a_R * (ex.P_star / ex.P_R) ** ((g - 1) / (2 * g))]
    return sorted(ex.x_int + t * v for v in s)


def cons(rho, u, p, gamma=GAMMA):
    return np.stack([rho, rho * u, p / (gamma - 1) + 0.5 * rho * u**2])


def project_exact(ex, edges, t, P, nq=6):
    '''L2 projection of the exact Riemann solution on cells [edges[k], edges[k+1]]: (3, N, P)'''
    edges = np.asarray(edges, dtype=float)
    a, b = edges[:-1], edges[1:]
    h = b - a
    cuts = [a] + [np.clip(x, a, b) for x in (wave_points(ex, t) if t > 0 else [ex.x_int])] + [b]
    xq, wq = (v.numpy() for v in gauss(nq))
    out = np.zeros((3, len(a), P))
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        L = hi - lo
        x = lo[:, None] + L[:, None] * (xq + 0.5)
        Q = cons(*ex.sample(x, t))
        xi = (x - a[:, None]) / h[:, None] - 0.5
        ph = phi(P, xi).numpy()
        out += np.einsum('cnq,pnq,q->cnp', Q, ph, wq) * (L / h)[None, :, None]
    return torch.tensor(out, dtype=D)


def project(f, edges, P, nq=8):
    '''L2 projection of a smooth function f(x) -> conservative (3, ...)'''
    edges = np.asarray(edges, dtype=float)
    a, h = edges[:-1], np.diff(edges)
    xq, wq = (v.numpy() for v in gauss(nq))
    x = a[:, None] + h[:, None] * (xq + 0.5)
    return torch.tensor(np.einsum('cnq,pq,q->cnp', f(x), phi(P, xq).numpy(), wq), dtype=D)


# ---------------------------------------------------------------- spatial operator

def rhs(Up, h, flux='hllc', nq=None):
    '''Up (B, 3, N+2, P) with one ghost cell per side -> dU/dt (B, 3, N, P).
    flux: a name in FLUX, or a callable flux(QL, QR, Up) -> (B, 3, N+1) (e.g. a learned flux).
    h: scalar or (B,) tensor.'''
    P = Up.shape[-1]
    nq = nq or P + 1
    half = torch.tensor([0.5, -0.5], dtype=D)
    tr = at(Up, half)                                   # (B, 3, N+2, 2): right and left traces
    QL, QR = tr[:, :, :-1, 0], tr[:, :, 1:, 1]          # edges between cells k, k+1: (B, 3, N+1)
    F = flux(QL, QR, Up) if callable(flux) else FLUX[flux](QL, QR)
    xq, wq = gauss(nq)
    U = Up[:, :, 1:-1]
    fq = euler_flux(at(U, xq))                           # (B, 3, N, q)
    vol = torch.einsum('bcnq,pq,q->bcnp', fq, dphi(P, xq).to(Up), wq.to(Up))
    ph = phi(P, half).to(Up)                             # (P, 2)
    edge = F[:, :, 1:, None] * ph[:, 0] - F[:, :, :-1, None] * ph[:, 1]
    return (vol - edge) / h


# ---------------------------------------------------------------- limiters

def eigvecs(Q, gamma=GAMMA):
    '''right eigenvectors of the Euler Jacobian at states Q (B, 3, N) -> (B, N, 3, 3)'''
    rho = Q[:, 0].clamp(min=EPS)
    u = Q[:, 1] / rho
    p = pressure(torch.stack([rho, Q[:, 1], Q[:, 2]], 1)).clamp(min=EPS)
    c = torch.sqrt(gamma * p / rho)
    H = (Q[:, 2] + p) / rho
    one = torch.ones_like(u)
    R = torch.stack([torch.stack([one, one, one], -1),
                     torch.stack([u - c, u, u + c], -1),
                     torch.stack([H - u * c, 0.5 * u**2, H + u * c], -1)], -2)
    return R


def minmod(a, b, c):
    s = torch.sign(a)
    same = (torch.sign(b) == s) & (torch.sign(c) == s)
    return torch.where(same, s * torch.minimum(a.abs(), torch.minimum(b.abs(), c.abs())), torch.zeros_like(a))


def tvb(Up, h, M=0.0, char=True):
    '''TVB-minmod limiter (Cockburn-Shu) for any degree; Up padded (B, 3, N+2, P) -> interior (B, 3, N, P).
    The edge deviations u(1/2) - mean and mean - u(-1/2) are compared with the neighbouring mean differences
    (modified minmod, TVB constant M). Where either changes, the polynomial is replaced by the linear one whose
    slope is minmod of the P1 slope and the two mean differences; other cells are left untouched.'''
    U = Up[:, :, 1:-1].clone()
    P = U.shape[-1]
    if P < 2:
        return U
    mean = Up[..., 0]
    ends = phi(P, torch.tensor([0.5, -0.5], dtype=D)).to(U)    # (P, 2)
    up = (U[..., 1:] * ends[1:, 0]).sum(-1)              # u(1/2) - mean
    dn = -(U[..., 1:] * ends[1:, 1]).sum(-1)             # mean - u(-1/2)
    s1 = math.sqrt(3) * U[..., 1]                        # P1 part of the edge deviation
    a2 = mean[:, :, 2:] - mean[:, :, 1:-1]
    a3 = mean[:, :, 1:-1] - mean[:, :, :-2]
    if char:
        R = eigvecs(mean[:, :, 1:-1])
        L = torch.linalg.inv(R)
        tc = lambda v: torch.einsum('bnij,bjn->bin', L, v)
        up, dn, s1, a2, a3 = tc(up), tc(dn), tc(s1), tc(a2), tc(a3)
    mm = lambda a: torch.where(a.abs() <= M * h**2, a, minmod(a, a2, a3))
    changed = ((mm(up) != up) | (mm(dn) != dn)).any(1, keepdim=True)
    m = mm(s1)
    if char:
        m = torch.einsum('bnij,bjn->bin', R, m)
    U[..., 1] = torch.where(changed, m / math.sqrt(3), U[..., 1])
    if P > 2:
        U[..., 2:] = torch.where(changed[..., None], torch.zeros_like(U[..., 2:]), U[..., 2:])
    return U


def positivity(U, eps=1e-12, gamma=GAMMA, nq=None):
    '''Zhang-Shu scaling towards the cell mean so rho, p > eps at the cell ends and quadrature points'''
    P = U.shape[-1]
    if P < 2:
        return U
    xq, _ = gauss(nq or P + 1)
    pts = torch.cat([torch.tensor([-0.5, 0.5], dtype=D), xq])
    mean = U[..., :1]
    V = at(U, pts)                                       # (B, 3, N, q)
    rho_min = V[:, 0].amin(-1)
    if (rho_min > eps).all() and (pressure(V) > eps).all():
        return U
    m0 = mean[:, 0, :, 0]
    th1 = torch.where(rho_min < eps, ((m0 - eps) / (m0 - rho_min).clamp(min=1e-300)).clamp(0, 1), torch.ones_like(m0))
    U = torch.cat([U[..., :1], U[..., 1:] * th1[:, None, :, None]], -1)
    V = at(U, pts)
    Mq = mean.expand_as(V)
    lo, hi = torch.zeros_like(V[:, 0]), torch.ones_like(V[:, 0])
    ok_full = pressure(V) > eps
    if ok_full.all():
        return U
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        ok = pressure(Mq + mid[:, None] * (V - Mq)) > eps
        lo, hi = torch.where(ok, mid, lo), torch.where(ok, hi, mid)
    th2 = torch.where(ok_full, torch.ones_like(lo), lo).amin(-1)
    return torch.cat([U[..., :1], U[..., 1:] * th2[:, None, :, None]], -1)


# ---------------------------------------------------------------- boundary conditions and time stepping

def transmissive(U, t=None):
    return torch.cat([U[:, :, :1], U, U[:, :, -1:]], 2)


def periodic(U, t=None):
    return torch.cat([U[:, :, -1:], U, U[:, :, :1]], 2)


def make_limiter(h, bc, M=0.0, limit=True, pos=True, char=True):
    def lim(U, t):
        if limit:
            U = tvb(bc(U, t), h, M, char)
        if pos:
            U = positivity(U)
        return U
    return lim


def ssp_rk3(U, t, dt, h, bc, flux, lim):
    L = lambda V, s: rhs(bc(V, s), h, flux)
    U1 = lim(U + dt * L(U, t), t + dt)
    U2 = lim(0.75 * U + 0.25 * (U1 + dt * L(U1, t + dt)), t + 0.5 * dt)
    return lim(U / 3 + 2 / 3 * (U2 + dt * L(U2, t + 0.5 * dt)), t + dt)


def max_speed(U, gamma=GAMMA):
    m = U[..., 0]
    rho = m[:, 0].clamp(min=EPS)
    u = m[:, 1] / rho
    p = pressure(m).clamp(min=EPS)
    return (u.abs() + torch.sqrt(gamma * p / rho)).max().item()


def solve(U, h, T, cfl=None, flux='hllc', bc=transmissive, M=0.0, limit=True, pos=True, char=True, dt=None):
    '''advance to time T; dt fixed if given, else from the CFL condition each step (default 0.9 / (2p + 1)).
    Returns (U, steps)'''
    cfl = cfl or 0.9 / (2 * U.shape[-1] - 1)
    lim = make_limiter(h, bc, M, limit, pos, char)
    U, t, n = lim(U, 0.0), 0.0, 0
    while t < T - 1e-14:
        k = dt or cfl * h / max_speed(U)
        k = min(k, T - t)
        U = ssp_rk3(U, t, k, h, bc, flux, lim)
        t += k
        n += 1
    return U, n


def l1_density(U, ex, edges, t):
    '''L1 error of the density cell means against the exact solution'''
    ref = project_exact(ex, edges, t, 1)
    h = np.diff(np.asarray(edges, dtype=float))
    return float((torch.as_tensor(h) * (U[0, 0, :, 0] - ref[0, :, 0]).abs()).sum())
