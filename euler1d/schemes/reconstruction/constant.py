'''
Piecewise-constant reconstruction (first order).
Q: (3, N + 6) -> (QL, QR), each (3, N + 1)
'''


def Const(Q, dx=1.0):
    n = Q.shape[1] - 6
    return Q[:, 2:n+3], Q[:, 3:n+4]