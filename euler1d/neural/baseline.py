'''
First-order HLLC + forward Euler on the same protocol as stepper.rollout(bc='exact'):
exact ghost cells at every step, dt per sample. NumPy, float64.
Q: (B, T, 3, N + 2G) -> (B, T, 3, N)
'''
import numpy as np

from euler1d.schemes.flux.hllc import HLLC


def first_order(Q, G, dt, dx, gamma=1.4):
    B, T, _, W = Q.shape
    N = W - 2 * G
    r = (np.asarray(dt, dtype=float) / dx).reshape(-1, 1, 1)
    cur = Q[:, 0, :, G:G + N]
    out = [cur]
    for k in range(1, T):
        pad = np.concatenate([Q[:, k - 1, :, G - 1:G], cur, Q[:, k - 1, :, G + N:G + N + 1]], axis=-1)
        QL = pad[..., :-1].transpose(1, 0, 2).reshape(3, -1)
        QR = pad[..., 1:].transpose(1, 0, 2).reshape(3, -1)
        F = HLLC(QL, QR, gamma=gamma).reshape(3, B, N + 1).transpose(1, 0, 2)
        cur = cur - r * (F[..., 1:] - F[..., :-1])
        out.append(cur)
    return np.stack(out, axis=1)