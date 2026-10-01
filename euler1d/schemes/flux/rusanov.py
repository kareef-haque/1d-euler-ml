'''
Rusanov (local Lax-Friedrichs) flux
'''
import numpy as np


def Rusanov(QL, QR, dx=1.0, gamma=1.4):
    def prim(Q):
        u = Q[1] / Q[0]
        P = (gamma - 1.0) * (Q[2] - 0.5 * Q[0] * u**2)
        return u, P

    def flux(Q, u, P):
        return np.array([Q[1], Q[1] * u + P, u * (Q[2] + P)])

    uL, PL = prim(QL)
    uR, PR = prim(QR)
    smax = np.maximum(np.abs(uL) + np.sqrt(gamma * PL / QL[0]),
                      np.abs(uR) + np.sqrt(gamma * PR / QR[0]))
    return 0.5 * (flux(QL, uL, PL) + flux(QR, uR, PR)) - 0.5 * smax * (QR - QL)