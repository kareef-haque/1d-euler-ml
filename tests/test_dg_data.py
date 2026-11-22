import numpy as np
import pytest
import torch

from euler1d.dg import data as dd, solver as dg
from infrastructure.exact import ExactRiemannSolver
from infrastructure.nfv_dataset import make_set as nfv_make_set

DX = 1e-3


@pytest.fixture(scope='module')
def small():
    return dd.make_set(6, 30, 6, DX, 1, 'test', 0)


def test_shapes_dt_and_same_problems_as_fv(small):
    d = small
    assert d['Q'].shape == (6, 7, 3, 30, 2) and d['Gh'].shape == (6, 13, 3, 2, 2)
    assert np.allclose(d['dt'], dd.default_cfl(1) * DX / d['smax'])
    f = nfv_make_set(6, 30, 2, DX, 'test', 0)
    assert np.array_equal(f['pl'], d['pl']) and np.array_equal(f['xf'], d['xf'])


def test_ghosts_and_interior_are_exact_projections(small):
    d, i, k = small, 2, 3
    ex = ExactRiemannSolver(d['pl'][i], d['pr'][i], d['xf'][i] * 30 * DX, 1.4)
    e = (np.arange(33) - 1) * DX
    t = k * d['dt'][i]
    full = dg.project_exact(ex, e, t, 2).numpy()
    assert np.allclose(d['Q'][i, k], full[:, 1:-1], rtol=1e-5)
    assert np.allclose(d['Gh'][i, 2 * k], full[:, [0, -1]], rtol=1e-5)
    half = dg.project_exact(ex, e, t + 0.5 * d['dt'][i], 2).numpy()
    assert np.allclose(d['Gh'][i, 2 * k + 1], half[:, [0, -1]], rtol=1e-5)


def test_get_set_roundtrip_and_mismatch(tmp_path):
    d = dd.get_set(tmp_path, 'eval', 4, 20, 3, DX, 1)
    e = dd.get_set(tmp_path, 'eval', 3, 20, 3, DX, 1)
    assert np.array_equal(e['Q'], d['Q'][:3])
    with pytest.raises(ValueError):
        dd.get_set(tmp_path, 'eval', 4, 20, 3, 2e-3, 1)
    assert dd.fname('train', 50, 50, 2, 0.18) != dd.fname('train', 50, 50, 1, 0.18)


@pytest.mark.parametrize('flux', ['hllc', 'rusanov'])
def test_uniform_problem_stays_uniform(flux):
    s = np.array([[1.2, 50.0, 6e4]])
    d = dd.from_states(s, s, np.array([0.5]), 20, 5, DX, 2)
    td = dd.tensors(d)
    P = dd.rollout(td, flux)
    assert torch.allclose(P, td['Q'][:, :1].expand_as(P), rtol=1e-10)


def test_callable_flux_matches_named_and_is_differentiable(small):
    td = dd.tensors(small)
    a = dd.rollout(td, 'rusanov')
    b = dd.rollout(td, lambda QL, QR, Up: dg.rusanov(QL, QR))
    assert torch.allclose(a, b)
    w = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    f = lambda QL, QR, Up: w * dg.rusanov(QL, QR) + (1 - w) * dg.hllc(QL, QR)
    loss = dd.mean_err(dd.rollout(td, f, n=3), td).mean()
    loss.backward()
    assert torch.isfinite(w.grad)


def test_dg_beats_first_order_fv(small):
    r = dd.compare(small)
    assert r['DG-P1 hllc']['ratio'] > 1.5 and r['DG-P1 hllc']['bad'] == 0
