import numpy as np
import pytest

from infrastructure.exact import ExactRiemannSolver
from infrastructure.solver_config import EulerConfig
from infrastructure.nfv_dataset import (make_set, sample_specs, split_ghost, save_set, load_set,
                                        RHO, P, XF, U_STD)
from euler1d.solver import EulerSolver

G = 1.4
DX = 1e-3


def small(split='train', seed=0, dtype=np.float64, **kw):
    a = dict(n=6, nx=40, nt=30, dx=DX, split=split, seed=seed, dtype=dtype)
    a.update(kw)
    return make_set(**a)


def test_shapes():
    d = small()
    assert d['Q'].shape == (6, 31, 3, 46)
    i, gl, gr = split_ghost(d)
    assert i.shape == (6, 31, 3, 40) and gl.shape == gr.shape == (6, 31, 3, 3)
    assert d['dt'].shape == d['smax'].shape == (6,)


def test_cfl_is_constant():
    d = small(cfl=0.5)
    assert np.allclose(d['dt'] * d['smax'] / DX, 0.5)


def test_ranges():
    pl, pr, xf = sample_specs(5000, np.random.default_rng(1))
    for s in (pl, pr):
        assert RHO[0] <= s[:, 0].min() and s[:, 0].max() <= RHO[1]
        assert P[0] <= s[:, 2].min() and s[:, 2].max() <= P[1]
        assert abs(s[:, 1].std() - U_STD) < 0.2
    assert XF[0] <= xf.min() and xf.max() <= XF[1]


def test_reproducible_and_splits_differ():
    a, b = small(seed=3), small(seed=3)
    assert np.array_equal(a['Q'], b['Q'])
    assert not np.allclose(a['pl'], small('eval', seed=3)['pl'])
    assert not np.allclose(small('eval', seed=3)['pl'], small('test', seed=3)['pl'])
    assert not np.allclose(a['pl'], small(seed=4)['pl'])


def test_matches_independent_exact_call():
    d = small()
    i, k = 2, 17
    ex = ExactRiemannSolver(d['pl'][i], d['pr'][i], d['xf'][i] * d['nx'] * DX, G)
    edges = (np.arange(d['nx'] + 2 * d['G'] + 1) - d['G']) * DX
    want = ex.cell_avg(edges, k * d['dt'][i])
    assert np.allclose(d['Q'][i, k], want, rtol=1e-12)


def test_initial_slice_is_the_jump():
    d = small()
    for i in range(3):
        rho, u, p = d['pl'][i]
        QL = np.array([rho, rho * u, p / (G - 1) + 0.5 * rho * u**2])
        rho, u, p = d['pr'][i]
        QR = np.array([rho, rho * u, p / (G - 1) + 0.5 * rho * u**2])
        Q0 = d['Q'][i, 0]
        j = int(d['xf'][i] * d['nx']) + d['G']
        assert np.allclose(Q0[:, :j], QL[:, None]) and np.allclose(Q0[:, j + 1:], QR[:, None])


def test_roundtrip(tmp_path):
    d = small(dtype=np.float32)
    save_set(tmp_path / 'a.npz', d)
    e = load_set(tmp_path / 'a.npz')
    assert np.array_equal(d['Q'], e['Q']) and e['nx'] == 40 and e['dx'] == DX and e['split'] == 'train'


def test_agrees_with_numerical_solver():
    # waves stay inside the window for nt <= 60 at CFL 0.5, so zero-gradient BCs are exact
    d = make_set(n=8, nx=200, nt=60, dx=DX, split='eval', seed=0, dtype=np.float64)
    e1, e5 = [], []
    for i in range(8):
        Qi = d['Q'][i, :, :, d['G']:-d['G']]
        for rec, integ, out in [('Const', 'euler', e1), ('WENO5Z', 'rk4', e5)]:
            cfg = EulerConfig(domain_size=200 * DX, N_cells=200, IC=Qi[0], t_max=60 * d['dt'][i], dt=d['dt'][i])
            r = EulerSolver(cfg, flux='HLLC', reconstruction=rec, integrator=integ, verbose=False)
            rng = np.ptp(Qi[0, 0])
            out.append(np.mean(np.abs(r.Q_hist[-1][0] - Qi[-1, 0])) / max(rng, 1e-9))
    print(f"\nrel. L1(rho) vs dataset  first order: {np.mean(e1):.3e}  WENO5Z: {np.mean(e5):.3e}")
    assert np.mean(e5) < np.mean(e1) and np.mean(e5) < 0.05