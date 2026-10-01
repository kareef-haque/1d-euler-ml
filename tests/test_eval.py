import json
import numpy as np
import pytest
import torch

import convergence_nfv as cv
import eval_nfv
from infrastructure.exact import ExactRiemannSolver
from infrastructure.multi_ref import multi_set, piecewise_ic
from infrastructure.nfv_dataset import make_set, from_states
from infrastructure.solver_config import EulerConfig
from euler1d.solver import EulerSolver
from euler1d.neural import rollouts as ro, diagnostics as dg, stepper
from euler1d.neural.model import NFVModel
from euler1d.schemes.flux.rusanov import Rusanov

DX = 1e-3
D64 = torch.float64


def small(n=2, nx=120, nt=20, **kw):
    return make_set(n, nx, nt, DX, 'eval', 0, dtype=np.float64, **kw)


@pytest.mark.parametrize('name', ['HLLC-1', 'Rusanov-1', 'AUSM+-1', 'HLLC-WENO5Z', 'AUSM+-WENO5'])
@pytest.mark.parametrize('bc', ['exact', 'zero'])
def test_classical_rollout_matches_solver(name, bc):
    d = small()
    d['bc'] = bc
    P = ro.predict_all(d, [name])[name]
    fl, rec, integ = ro.SCHEMES[name]
    for i in range(2):
        cfg = EulerConfig(domain_size=0.12, N_cells=120, IC=d['Q'][i, 0, :, 3:-3], t_max=20 * d['dt'][i], dt=d['dt'][i])
        r = EulerSolver(cfg, flux=fl, reconstruction=rec, integrator=integ, verbose=False)
        assert np.allclose(np.stack(r.Q_hist), P[i], rtol=1e-8, atol=1e-6)


def test_nfv_rollout_matches_stepper():
    torch.manual_seed(0)
    m = NFVModel(6, 1, dtype=D64)
    d = small(nt=10)
    P = ro.predict_all(d, ['NFV'], m)['NFV']
    Q, dt = torch.from_numpy(d['Q']), torch.from_numpy(d['dt'])
    want = stepper.rollout(m, Q, 3, dt, DX, 'exact').detach().numpy()
    assert np.allclose(P, want, rtol=1e-9, atol=1e-6)


def test_exact_ghosts_have_expected_shape_and_values():
    d = small(nt=4)
    Gl, Gr = ro.exact_ghosts(d['pl'], d['pr'], d['xf'], 120, DX, d['dt'], 9, g=5)
    assert Gl.shape == Gr.shape == (2, 9, 3, 5)
    assert np.allclose(Gl[:, ::2, :, -3:], d['Q'][:, :5, :, :3], rtol=1e-12)


def test_piecewise_ic_and_multi_set():
    pc = np.array([[[1.0, 0.0, 2.0], [3.0, 0.0, 4.0]]])
    Q = piecewise_ic(pc, np.array([[0.25]]), 4, 0.1)
    assert np.allclose(Q[0, 0, :4], [1.0, 1.0, 2.0, 3.0])
    d = multi_set(3, 60, 20, DX, k=3, ref=4)
    assert d['Q'].shape == (3, 21, 3, 66) and np.isfinite(d['Q']).all()
    tot = d['Q'][..., 3:-3].sum(-1)
    T = 20 * d['dt']
    flow = T / DX * (d['pc'][:, 0, 1] - d['pc'][:, -1, 1])
    assert np.allclose(tot[:, -1, 0] - tot[:, 0, 0], flow, rtol=1e-6, atol=1e-6)


def test_diagnostics():
    P = np.ones((2, 3, 3, 5))
    P[..., 2, :] = 2.5e5
    assert dg.positivity(P)['n_bad'] == 0
    P[1, 2, 0, 3] = -1.0
    P[0, 1, 1, 2] = np.nan
    r = dg.positivity(P)
    assert r['n_bad'] == 2 and r['min_rho'] == -1.0
    A = np.ones((1, 4, 3, 6)) * np.array([1.0, 0.0, 2.5e5])[None, None, :, None]
    assert dg.drift(A) == {'mass': 0.0, 'energy': 0.0}
    A[0, -1, 0, 0] += 0.5
    assert dg.drift(A)['mass'] == pytest.approx(0.5 / 6)


def test_drift_with_boundary_flux():
    d = small(nt=6)
    P = d['Q'][..., 3:-3]
    cons = lambda q: np.stack([q[:, 0], q[:, 0] * q[:, 1], q[:, 2] / 0.4 + 0.5 * q[:, 0] * q[:, 1]**2], axis=1)
    fin, fout = dg.flux_of(cons(d['pl'])), dg.flux_of(cons(d['pr']))
    assert max(dg.drift(P, d['dt'], DX, fin, fout).values()) < 1e-3
    assert dg.drift(P)['mass'] > 1e-3


def test_mirror_gap_symmetric_flux_vs_broken():
    d = small(nt=6)
    W = dg.windows_from_set(d, 2, n=400)
    ok = lambda W: Rusanov(W[:, :, 0].T, W[:, :, 1].T, gamma=1.4).T
    assert np.all(dg.mirror_gap(ok, W)['err'] < 1e-9)
    bad = lambda W: ok(W) + 50.0 * W[:, :, 0] * np.array([1.0, 0.0, 0.0]) * np.sign(W[:, 1:2, 0])
    assert dg.mirror_gap(bad, W)['err'][0] > 1e-3


def test_consistency_with_given_states():
    d = small(nt=6)
    st = dg.states_from_set(d, 300)
    assert st.shape == (300, 3)
    c = dg.consistency_error(NFVModel(2, 1, dtype=D64), states=st)
    assert np.all(np.isfinite(c['err']))
    assert dg.windows_from_set(d, 4, 50).shape == (50, 3, 4)


def test_fan_glitch():
    ex = ExactRiemannSolver((1.0, 0.0, 1e5), (0.125, 0.0, 1e4), 0.5, 1.4)
    t, nx = 6e-4, 1000
    rho = ex.cell_avg(np.arange(nx + 1) * 1e-3, t)[0]
    f = dg.fan_glitch(ex, t, 1e-3, nx, rho)
    assert len(f) == 1 and f[0]['num'] == pytest.approx(f[0]['exact']) and f[0]['ncells'] > 20
    rho2 = rho.copy()
    rho2[int(0.5 * nx) - 30:int(0.5 * nx) - 20] *= 1.1
    assert dg.fan_glitch(ex, t, 1e-3, nx, rho2)[0]['num'] > 3 * f[0]['exact']


def test_convergence_orders():
    m = NFVModel(2, 1, dtype=D64)
    names = ['HLLC-1', 'HLLC-WENO5Z']
    r = cv.smooth(m, [40, 80], names=names)
    assert 0.8 < r['HLLC-1']['order'][0] < 1.3
    assert r['HLLC-WENO5Z']['order'][0] > 3.0
    r = cv.riemann(m, [100, 200], n=4, names=['HLLC-1'])
    assert 0.4 < r['HLLC-1']['order'][0] < 1.4


def test_scripts_smoke(tmp_path):
    m = NFVModel(2, 1)
    m.save(tmp_path / 'm.pt', gamma=1.4, step=0)
    out = tmp_path / 'e'
    eval_nfv.main(['--model', str(tmp_path / 'm.pt'), '--out', str(out), '--data_dir', str(tmp_path / 'd'),
                   '--riemann_nx', '40', '--riemann_nt', '20', '--n_riemann', '4', '--n_multi', '2',
                   '--multi_nx', '60', '--multi_nt', '20', '--multi_ref', '4', '--named_nx', '120', '--named_nt', '20'])
    r = json.load(open(out / 'results.json'))
    assert set(r) == {'riemann', 'multi', 'named'} and set(r['named']) == {'Sod', 'Lax', '123'}
    assert 'consistency' in r['riemann'] and 'mirror' in r['riemann']
    assert (out / 'riemann_box.png').exists() and (out / 'named_Sod.png').exists()
    cv.main(['--model', str(tmp_path / 'm.pt'), '--Ns', '20,40', '--n', '3', '--out', str(tmp_path / 'c.json')])
    assert set(json.load(open(tmp_path / 'c.json'))) == {'riemann', 'smooth'}