import json
import numpy as np
import pytest
import torch

import train_nfv
from infrastructure.solver_config import EulerConfig
from infrastructure.nfv_dataset import make_set
from euler1d.solver import EulerSolver
from euler1d.schemes.flux.hllc import HLLC
from euler1d.schemes.flux.rusanov import Rusanov
from euler1d.neural import stepper, consistency, weighted as ws
from euler1d.neural.model import NFVModel, build
from euler1d.neural.weighted import WSModel
from euler1d.neural.diagnostics import mirror_gap, nfv_center_flux, windows_from_set

DX = 1e-3
D64 = torch.float64


def rand_states(n, seed=0):
    r = np.random.default_rng(seed)
    rho, u, p = r.uniform(0.1, 2.5, n), r.normal(0, 200, n), r.uniform(1e4, 1.25e5, n)
    return np.stack([rho, rho * u, p / 0.4 + 0.5 * rho * u**2])


def scrambled(riemann='hllc', b=1, seed=0):
    torch.manual_seed(seed)
    m = WSModel(b, riemann=riemann, dtype=D64)
    torch.nn.init.normal_(m.net[-1].weight, std=1.0)
    return m


@pytest.mark.parametrize('name,ref', [('hllc', HLLC), ('rusanov', Rusanov)])
def test_riemann_matches_numpy(name, ref):
    QL, QR = rand_states(500, 0), rand_states(500, 1)
    F = ws.RIEMANN[name](torch.from_numpy(QL)[None], torch.from_numpy(QR)[None])[0].numpy()
    assert np.allclose(F, ref(QL, QR), rtol=1e-10, atol=1e-6)


def test_params():
    assert WSModel().num_params() == 45 * 6 + 15 + 5 * 240 + 15 * 12 + 12


@pytest.mark.parametrize('riemann,b', [('hllc', 1), ('rusanov', 1), ('hllc', 3)])
def test_exactly_consistent(riemann, b):
    m = scrambled(riemann, b)
    S = torch.from_numpy(rand_states(1000).T.copy())
    pen = consistency.penalty(m, S, mode='box', n=1000, gen=torch.Generator().manual_seed(0))
    assert torch.allclose(consistency.flux_at(m, S), consistency.phys_flux(S), rtol=1e-12, atol=1e-6)
    assert pen < 1e-24


def test_reflection_symmetric():
    d = make_set(4, 40, 10, DX, 'test', 0, dtype=np.float64)
    W = windows_from_set(d, 6, n=500)
    g = mirror_gap(nfv_center_flux(scrambled()), W)
    assert np.all(g['err'] < 1e-10 * np.maximum(g['spread'], 1))
    g = mirror_gap(nfv_center_flux(NFVModel(6, 1, dtype=D64)), W)
    assert np.any(g['err'] > 1e-3)


def test_weights_convex():
    m = scrambled()
    x = torch.randn(2, 3, 40, dtype=D64)
    for w in m.weights(x):
        assert w.shape == (2, 3, 4, 35) and (w >= 0).all() and torch.allclose(w.sum(2), torch.ones_like(w.sum(2)))


def test_init_near_first_order():
    m = WSModel(dtype=D64)
    wL, _ = m.weights(torch.randn(1, 3, 20, dtype=D64))
    assert torch.allclose(wL[:, :, -1], torch.full((), ws.P_CONST, dtype=D64))


def test_const_limit_is_hllc1():
    m = WSModel(dtype=D64)
    m.net[-1].bias.data.view(3, 4)[:, -1] = 60.0
    Q = rand_states(30)
    with torch.no_grad():
        F = stepper.flux(m, [torch.from_numpy(Q)[None]])[0].numpy()
    assert np.allclose(F, HLLC(Q[:, 2:-3], Q[:, 3:-2]), rtol=1e-10, atol=1e-6)


def test_weno_weights_fifth_order():
    m = WSModel(dtype=D64)
    m.net[-1].bias.data.view(3, 4)[:] = torch.log(torch.tensor([0.3, 0.6, 0.1, 1e-30], dtype=D64))
    err = []
    for N in (40, 80):
        x = (np.arange(-3, N + 3) + 0.5) / N
        h = 1 / N
        avg = lambda f: (f(x + h / 2) - f(x - h / 2)) / h
        Q = np.stack([avg(lambda s: s + np.sin(2 * np.pi * s) / (2 * np.pi) * 0.1),
                      avg(lambda s: s**2 / 2), avg(lambda s: 3 * s - np.cos(2 * np.pi * s) / (2 * np.pi))])
        with torch.no_grad():
            QL, _ = m.states(torch.from_numpy(Q / np.array([1, 300, 9e4])[:, None])[None])
        xf = np.arange(N + 1) / N
        exact = np.stack([1 + 0.1 * np.cos(2 * np.pi * xf), xf, 3 + np.sin(2 * np.pi * xf)])
        err.append(np.abs(QL[0].numpy() - exact).max())
    assert np.log2(err[0] / err[1]) > 4.5


def test_save_load_and_solver(tmp_path):
    m = scrambled('rusanov', 1).float()
    m.save(tmp_path / 'm.pt', gamma=1.4, step=3)
    k = NFVModel.load(tmp_path / 'm.pt')
    assert isinstance(k, WSModel) and (k.a, k.g, k.riemann, k.integrator, k.extra['step']) == (6, 3, 'rusanov', 'rk3', 3)
    x = torch.randn(2, 3, 30)
    with torch.no_grad():
        assert torch.equal(k(x), m(x))
    cfg = EulerConfig(domain_size=0.04, N_cells=40, IC=make_set(1, 40, 1, DX, 'test', 0)['Q'][0, 0, :, 3:-3],
                      t_max=5e-6, dt=1e-6)
    r = EulerSolver(cfg, flux='NFV', nfv=str(tmp_path / 'm.pt'), verbose=False)
    assert len(r.Q_hist) == 6 and np.isfinite(r.Q_hist[-1]).all()


def test_build():
    assert isinstance(build('ws', b=2), WSModel) and isinstance(build('nfv', 4, 1), NFVModel)


def test_training_smoke(tmp_path):
    a = dict(arch='ws', schedule='20:1e-3:10:10', n_train=32, batch_size=16, n_eval=6, eval_nx=40, eval_nt=30,
             eval_every=10, data_dir=str(tmp_path / 'd'), out=str(tmp_path / 'o'), device='cpu')
    out = train_nfv.main([f'--{k}={v}' for k, v in a.items()])
    L = [json.loads(l) for l in open(out + '/log.jsonl')]
    assert L[-1]['loss'] < L[0]['loss'] and all(np.isfinite(r['l2h']) for r in L)
    assert all(r['pen'] < 1e-10 for r in L)
    assert json.load(open(out + '/config.json'))['a'] == 6
    assert isinstance(NFVModel.load(out + '/model_final.pt'), WSModel)


def test_positivity_fallback():
    m = WSModel(dtype=D64)
    m.net[-1].bias.data.view(3, 4)[:] = torch.tensor([0.0, 0.0, 10.0, 0.0])
    Q = np.ones((3, 12)) * np.array([[1.0], [0.0], [2.5e5]])
    Q[:, 6:] = np.array([[0.05], [0.0], [2.5e3]])
    with torch.no_grad():
        QL, QR = m.states(torch.from_numpy(Q / np.array([1, 300, 9e4])[:, None])[None])
    for S in (QL, QR):
        r, p = ws.rho_p(S, 1.4)
        assert (r > 0).all() and (p > 0).all()


def test_rk3_steps():
    d = make_set(3, 30, 12, DX, 'train', 0, G=3, dtype=np.float64)
    Q, dt = torch.from_numpy(d['Q']), torch.from_numpy(d['dt'])
    m = WSModel(dtype=D64)
    m.net[-1].bias.data.view(3, 4)[:, -1] = 60.0
    with torch.no_grad():
        P = stepper.rollout(m, Q, 3, dt, DX)
        m.integrator = 'euler'
        Pe = stepper.rollout(m, Q, 3, dt, DX)
    assert torch.isfinite(P).all() and not torch.allclose(P[:, 2:], Pe[:, 2:])
    with torch.no_grad():
        U = Q[:, 0, :, 3:-3]
        L = lambda U: -(dt.view(-1, 1, 1) / DX) * torch.diff(stepper.flux(m, [torch.cat([Q[:, 0, :, :3], U, Q[:, 0, :, -3:]], -1)]), dim=-1)
        U1 = U + L(U)
        U2 = 0.75 * U + 0.25 * (U1 + L(U1))
        ref = U / 3 + 2 / 3 * (U2 + L(U2))
    assert torch.allclose(P[:, 1], ref)


def test_safe_flux():
    m = WSModel(dtype=D64)
    h = torch.from_numpy(rand_states(16))[None]
    F = torch.zeros(1, 3, 11, dtype=D64)
    U = h[..., 3:-3].clone()
    assert m.safe_flux(h, F, U) is F
    U[0, 0, 4] = -1.0
    G = m.safe_flux(h, F, U)
    low = ws.hllc(h[..., 2:13], h[..., 3:14])
    assert torch.equal(G[..., 4:6], low[..., 4:6]) and (G[..., :4] == 0).all() and (G[..., 6:] == 0).all()


def test_rollout_survives_strong_contact():
    from infrastructure.nfv_dataset import from_states
    d = from_states(np.array([[2.411, -3.571, 2.026e4]]), np.array([[0.1216, -1.559, 7.871e4]]), 0.5, 40, 30, DX, 0.5,
                    G=3, dtype=np.float64)
    torch.manual_seed(1)
    m = WSModel(dtype=D64)
    m.net[-1].bias.data.view(3, 4)[:] = torch.tensor([0.0, 0.0, 3.0, 0.0])
    with torch.no_grad():
        P = stepper.rollout(m, torch.from_numpy(d['Q']), 3, torch.from_numpy(d['dt']), DX)
    r, p = ws.rho_p(P.flatten(0, 1), 1.4)
    assert torch.isfinite(P).all() and (r > 0).all() and (p > 0).all()
