import numpy as np
import pytest
import torch

from infrastructure.solver_config import EulerConfig
from infrastructure.nfv_dataset import make_set
from euler1d.solver import EulerSolver
from euler1d.neural import scaling, stepper
from euler1d.neural.model import NFVModel
from euler1d.neural.adapter import NFVFlux

DX = 1e-3
D64 = torch.float64


def data(n=3, nx=30, nt=12, G=3):
    d = make_set(n, nx, nt, DX, 'train', 0, G=G, dtype=np.float64)
    return d, torch.from_numpy(d['Q']), torch.from_numpy(d['dt'])


@pytest.mark.parametrize('a,b', [(2, 1), (4, 5), (6, 1), (10, 11)])
def test_param_count_and_shapes(a, b):
    m = NFVModel(a, b)
    assert m.num_params() == 45 * a * b + 1263
    y = m(torch.zeros(4, 3 * b, 50))
    assert y.shape == (4, 3, 50 - a + 1)


def test_nfv_2_1_size():
    assert NFVModel(2, 1).num_params() == 1353


def test_odd_stencil_rejected():
    with pytest.raises(ValueError):
        NFVModel(3, 1)


@pytest.mark.parametrize('act', ['ELU', 'ReLU', 'Tanh'])
def test_bias_zero_init(act):
    m = NFVModel(2, 1, act=act)
    for p in m.net.modules():
        if isinstance(p, torch.nn.Conv1d):
            assert torch.all(p.bias == 0)


def test_save_load(tmp_path):
    m = NFVModel(4, 3, act='ReLU')
    m.save(tmp_path / 'm.pt', gamma=1.4)
    k = NFVModel.load(tmp_path / 'm.pt')
    x = torch.randn(2, 9, 20)
    assert (k.a, k.b, k.act, k.extra['gamma']) == (4, 3, 'ReLU', 1.4)
    assert torch.equal(m(x), k(x))


def test_scales_make_data_order_one():
    d, Q, dt = data(n=20)
    x = Q / torch.as_tensor(scaling.QS).view(1, 1, 3, 1)
    assert 0.05 < x.abs().max() < 20
    rho, mom, E = d['Q'][:, 0, 0], d['Q'][:, 0, 1], d['Q'][:, 0, 2]
    u = mom / rho
    p = 0.4 * (E - 0.5 * rho * u**2)
    F = np.stack([mom, mom * u + p, u * (E + p)], axis=1)
    f = F / scaling.FS.reshape(1, 3, 1)
    assert 0.01 < np.abs(f).max() < 50


def test_step_is_conservative():
    torch.manual_seed(0)
    m = NFVModel(2, 1, dtype=D64)
    _, Q, dt = data()
    h = [Q[:, 0, :, 2:-2]]
    F = stepper.flux(m, h)
    new = stepper.advance(m, h, dt, DX)
    r = dt.view(-1, 1) / DX
    dsum = new.sum(-1) - h[0][..., 1:-1].sum(-1)
    assert torch.allclose(dsum, -r * (F[..., -1] - F[..., 0]), rtol=1e-10, atol=1e-6)


def test_rollout_shapes_and_given_slices():
    torch.manual_seed(0)
    _, Q, dt = data(nt=8)
    for a, b in [(2, 1), (6, 1), (4, 3)]:
        m = NFVModel(a, b, dtype=D64)
        P = stepper.rollout(m, Q, 3, dt, DX)
        assert P.shape == (3, 9, 3, 30)
        assert torch.equal(P[:, :b], Q[:, :b, :, 3:-3])


def test_history_order_b2():
    torch.manual_seed(0)
    _, Q, dt = data(nt=4)
    m = NFVModel(2, 2, dtype=D64)
    P = stepper.rollout(m, Q, 3, dt, DX)
    sl = lambda k: Q[:, k, :, 2:-2]
    want = stepper.advance(m, [sl(0), sl(1)], dt, DX)
    assert torch.allclose(P[:, 2], want)


def test_exact_ghost_needs_enough_cells():
    _, Q, dt = data(nt=3, G=3)
    with pytest.raises(ValueError):
        stepper.rollout(NFVModel(10, 1, dtype=D64), Q, 3, dt, DX)


def test_gradients_flow():
    torch.manual_seed(0)
    _, Q, dt = data(nt=10)
    m = NFVModel(2, 1, dtype=D64)
    P = stepper.rollout(m, Q, 3, dt, DX)
    loss = ((P[:, 1:] - Q[:, 1:, :, 3:-3]) / torch.as_tensor(scaling.QS).view(1, 1, 3, 1)).pow(2).mean()
    loss.backward()
    for p in m.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
    assert sum(p.grad.abs().sum() for p in m.parameters()) > 0


@pytest.mark.parametrize('a', [2, 6])
def test_stepper_matches_solver(a):
    torch.manual_seed(1)
    m = NFVModel(a, 1, dtype=D64)
    d, Q, dt = data(n=2, nt=10)
    P = stepper.rollout(m, Q, 3, dt, DX, bc='zero').detach().numpy()
    for i in range(2):
        cfg = EulerConfig(domain_size=30 * DX, N_cells=30, IC=d['Q'][i, 0, :, 3:-3],
                          t_max=10 * d['dt'][i], dt=d['dt'][i])
        r = EulerSolver(cfg, flux='NFV', nfv=NFVFlux(m), verbose=False)
        S = np.stack(r.Q_hist)
        assert S.shape == P[i].shape
        assert np.allclose(S, P[i], rtol=1e-10, atol=1e-8)


def test_solver_defaults_and_errors(tmp_path):
    m = NFVModel(2, 1, dtype=D64)
    cfg = EulerConfig(domain_size=0.03, N_cells=30, IC=np.tile([[1.0], [0.0], [2.5e5]], (1, 30)),
                      t_max=3e-6, dt=1e-6)
    with pytest.raises(ValueError):
        EulerSolver(cfg, flux='NFV', verbose=False)
    with pytest.raises(NotImplementedError):
        NFVFlux(NFVModel(2, 2, dtype=D64))
    m.extra['gamma'] = 1.3
    with pytest.raises(ValueError):
        EulerSolver(cfg, flux='NFV', nfv=NFVFlux(m), verbose=False)
    m.extra['gamma'] = 1.4
    m.save(tmp_path / 'm.pt', gamma=1.4)
    r = EulerSolver(cfg, flux='NFV', nfv=str(tmp_path / 'm.pt'), verbose=False)
    assert len(r.Q_hist) == 4
    cfg.N_ghost = 0
    with pytest.raises(ValueError):
        EulerSolver(cfg, flux='NFV', nfv=NFVFlux(m), verbose=False)