import numpy as np
import pytest

import sod_plateaus as sp
from infrastructure.exact import ExactRiemannSolver
from euler1d.neural import rollouts as ro
from infrastructure.nfv_dataset import from_states


def setup(scheme):
    nx, nt, dx = 400, 100, 1e-3
    d = from_states(*sp.SOD[:1], sp.SOD[1], 0.5, nx, nt, dx, 0.5, dtype=np.float64)
    ex = ExactRiemannSolver(sp.SOD[0], sp.SOD[1], 0.5 * nx * dx, 1.4)
    t = nt * d['dt'][0]
    return d, ex, t, dx


def test_measure_is_zero_for_exact_solution():
    d, ex, t, dx = setup(None)
    Pf = ex.cell_avg(np.arange(401) * dx, t)
    r = sp.measure(Pf, ex, t, dx)
    assert all(abs(v) < 2e-3 for s in r.values() for v in s.values())


def test_measure_sees_a_known_bias():
    d, ex, t, dx = setup(None)
    Pf = ex.cell_avg(np.arange(401) * dx, t)
    Pf[0] *= 1.2
    assert sp.measure(Pf, ex, t, dx)['L']['rho'] == pytest.approx(0.2, abs=5e-3)


def test_classical_schemes_have_small_plateau_errors():
    d, ex, t, dx = setup(None)
    P = ro.predict_all(d, ['HLLC-1', 'HLLC-WENO5Z'])
    for n in P:
        r = sp.measure(P[n][0, -1], ex, t, dx)
        assert all(abs(v) < 0.03 for s in r.values() for v in s.values()), n