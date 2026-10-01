'''
Exact Riemann solver for the 1D Euler equations (Toro, ch. 4).
- sample():   primitive (rho, u, P) at points x and time t
- cell_avg(): conservative cell averages (3, N) over given cell edges
'''
import numpy as np
from scipy.optimize import brentq


class ExactRiemannSolver:
    def __init__(self, state_L, state_R, x_interface=0.5, gamma=1.4):
        self.rho_L, self.u_L, self.P_L = state_L
        self.rho_R, self.u_R, self.P_R = state_R
        self.x_int = x_interface
        self.gamma = gamma

        self.a_L = np.sqrt(gamma * self.P_L / self.rho_L)
        self.a_R = np.sqrt(gamma * self.P_R / self.rho_R)

        self.P_star = self._solve_p_star()
        self.u_star = 0.5 * (self.u_L + self.u_R) \
            + 0.5 * (self._f_k(self.P_star, 'R') - self._f_k(self.P_star, 'L'))
        self.rho_star_L = self._rho_star(self.P_star, 'L')
        self.rho_star_R = self._rho_star(self.P_star, 'R')

    def _f_k(self, P, side):
        Pk, rk, ak = ((self.P_L, self.rho_L, self.a_L) if side == 'L'
                      else (self.P_R, self.rho_R, self.a_R))
        g = self.gamma
        if P > Pk:
            A = 2.0 / ((g + 1.0) * rk)
            B = (g - 1.0) / (g + 1.0) * Pk
            return (P - Pk) * np.sqrt(A / (P + B))
        return 2.0 * ak / (g - 1.0) * ((P / Pk) ** ((g - 1.0) / (2.0 * g)) - 1.0)

    def _solve_p_star(self):
        g = self.gamma
        if 2.0 * (self.a_L + self.a_R) / (g - 1.0) <= self.u_R - self.u_L:
            raise ValueError('Vacuum is generated')
        func = lambda P: self._f_k(P, 'L') + self._f_k(P, 'R') + (self.u_R - self.u_L)
        lo, hi = 1e-10, 10.0 * max(self.P_L, self.P_R)
        while func(hi) < 0.0:
            hi *= 10.0
        return brentq(func, lo, hi, xtol=1e-12, rtol=1e-14)

    def _rho_star(self, P_star, side):
        Pk, rk = (self.P_L, self.rho_L) if side == 'L' else (self.P_R, self.rho_R)
        g = self.gamma
        if P_star > Pk:
            r = P_star / Pk
            return rk * (r + (g - 1.0) / (g + 1.0)) / ((g - 1.0) / (g + 1.0) * r + 1.0)
        return rk * (P_star / Pk) ** (1.0 / g)

    def _side(self, s, rk, uk, Pk, ak, u_st, r_st):
        # left-side wave pattern; the right side is handled by mirroring
        g, ps = self.gamma, self.P_star
        rho = np.full_like(s, rk)
        u = np.full_like(s, uk)
        p = np.full_like(s, Pk)
        if ps > Pk:
            sh = uk - ak * np.sqrt((g + 1.0) / (2.0 * g) * ps / Pk + (g - 1.0) / (2.0 * g))
            st = s >= sh
        else:
            head = uk - ak
            tail = u_st - ak * (ps / Pk) ** ((g - 1.0) / (2.0 * g))
            st = s > tail
            fan = (s >= head) & ~st
            sf = s[fan]
            af = 2.0 / (g + 1.0) * (ak + (g - 1.0) / 2.0 * (uk - sf))
            u[fan] = 2.0 / (g + 1.0) * (ak + (g - 1.0) / 2.0 * uk + sf)
            rho[fan] = rk * (af / ak) ** (2.0 / (g - 1.0))
            p[fan] = Pk * (af / ak) ** (2.0 * g / (g - 1.0))
        rho[st], u[st], p[st] = r_st, u_st, ps
        return rho, u, p

    def _from_s(self, s):
        left = s < self.u_star
        rho, u, p = (np.empty_like(s) for _ in range(3))
        rho[left], u[left], p[left] = self._side(
            s[left], self.rho_L, self.u_L, self.P_L, self.a_L, self.u_star, self.rho_star_L)
        r = ~left
        rr, ur, pr = self._side(
            -s[r], self.rho_R, -self.u_R, self.P_R, self.a_R, -self.u_star, self.rho_star_R)
        rho[r], u[r], p[r] = rr, -ur, pr
        return rho, u, p

    def sample(self, x, t):
        x = np.asarray(x, dtype=float)
        shape = x.shape
        x = x.ravel()
        if t <= 1e-12:
            left = x < self.x_int
            return (np.where(left, self.rho_L, self.rho_R).reshape(shape),
                    np.where(left, self.u_L, self.u_R).reshape(shape),
                    np.where(left, self.P_L, self.P_R).reshape(shape))
        rho, u, p = self._from_s((x - self.x_int) / t)
        return rho.reshape(shape), u.reshape(shape), p.reshape(shape)

    def smax(self):
        g = self.gamma
        aL = np.sqrt(g * self.P_star / self.rho_star_L)
        aR = np.sqrt(g * self.P_star / self.rho_star_R)
        return max(abs(self.u_L) + self.a_L, abs(self.u_R) + self.a_R,
                   abs(self.u_star) + aL, abs(self.u_star) + aR)

    def cell_avg(self, edges, t, n=10):
        edges = np.asarray(edges, dtype=float)
        dx = np.diff(edges)
        x = edges[:-1, None] + dx[:, None] * (np.arange(n) + 0.5) / n
        rho, u, p = self.sample(x, t)
        E = p / (self.gamma - 1.0) + 0.5 * rho * u**2
        return np.array([rho.mean(axis=1), (rho * u).mean(axis=1), E.mean(axis=1)])

    def cell_avg_t(self, edges, ts, n=10):
        edges = np.asarray(edges, dtype=float)
        ts = np.asarray(ts, dtype=float)
        dx = np.diff(edges)
        x = (edges[:-1, None] + dx[:, None] * (np.arange(n) + 0.5) / n).ravel()
        rho, u, p = (np.empty((len(ts), len(x))) for _ in range(3))
        pos = ts > 1e-12
        if pos.any():
            s = (x[None, :] - self.x_int) / ts[pos][:, None]
            r, v, q = self._from_s(s.ravel())
            rho[pos], u[pos], p[pos] = (a.reshape(-1, len(x)) for a in (r, v, q))
        if (~pos).any():
            r, v, q = self.sample(x, 0.0)
            rho[~pos], u[~pos], p[~pos] = r, v, q
        E = p / (self.gamma - 1.0) + 0.5 * rho * u**2
        Q = np.stack([rho, rho * u, E], axis=1)
        return Q.reshape(len(ts), 3, len(dx), n).mean(axis=-1)