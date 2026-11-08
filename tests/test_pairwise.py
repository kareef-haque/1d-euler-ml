import json
import numpy as np
import pytest
import torch

import train_nfv
import sod_plateaus as sp
from euler1d.neural import stepper, consistency, scaling
from euler1d.neural.model import NFVModel, build
from euler1d.neural.pairwise import PairModel

DX = 1e-3
D64 = torch.float64
QS = torch.tensor(scaling.QS).view(1, 3, 1)
FS = torch.tensor(scaling.FS).view(1, 3)


def states(W, B=2, seed=0):
    r = np.random.default_rng(seed)
    rho, u, p = r.uniform(0.1, 2.5, (B, W)), r.normal(0, 200, (B, W)), r.uniform(1e4, 1.25e5, (B, W))
    return torch.from_numpy(np.stack([rho, rho * u, p / 0.4 + 0.5 * rho * u**2], axis=1))


def scrambled(mode, k=2, cfl=2.0, std=0.3):
    torch.manual_seed(0)
    m = PairModel(k, mode=mode, cfl=cfl, dtype=D64)
    torch.nn.init.normal_(m.net[-1].weight, std=std)
    return m


def explicit(m, h, r):
    '''pairwise update cell by cell; h: (B, 3, N + 6k) physical'''
    k, N = m.k, h.shape[-1] - 6 * m.k
    with torch.no_grad():
        P, T = m.exchanges(h / QS)
    out = h[..., 3 * k:3 * k + N].clone()
    for c in range(3 * k, 3 * k + N):
        dq = 0
        for d in range(1, k + 1):
            dq = dq - P[:, d - 1, :, c - 2 * k] * FS + P[:, d - 1, :, c - d - 2 * k] * FS
        dq = r * dq
        if T is not None:
            dq = dq - h[..., c] + sum(T[:, d + k, c - d - 2 * k, None] * h[..., c - d] for d in range(-k, k + 1))
        out[..., c - 3 * k] += dq
    return out


def test_shapes():
    m = PairModel(3, mode='free')
    assert (m.a, m.g) == (18, 9) and m(torch.zeros(2, 3, 40)).shape == (2, 3, 40 - 18 + 1)
    t = PairModel(3, b=2, mode='transport')
    assert t(torch.ones(2, 6, 40)).shape == (2, 3, 23)
    assert t.num_params() == 3 * 2 * 13 * 15 + 15 + 5 * 240 + 15 * 13 + 13


@pytest.mark.parametrize('mode', ['free', 'transport'])
def test_matches_explicit_pairwise(mode):
    m = scrambled(mode)
    h = states(12 + 6 * m.k)
    r = m.r0
    with torch.no_grad():
        U = stepper.advance(m, [h], torch.full((2,), r * DX, dtype=D64), DX)
    assert torch.allclose(U, explicit(m, h, r), rtol=1e-10, atol=1e-8)


def test_transport_keeps_density_positive():
    m = scrambled('transport', std=5.0)
    h = states(30 + 6 * m.k)
    h[:, 0, 20:25] = 1e-3
    with torch.no_grad():
        U = stepper.advance(m, [h], torch.full((2,), m.r0 * DX, dtype=D64), DX)
        _, T = m.exchanges(h / QS)
    assert (U[:, 0] > 0).all() and T.max() > 0.5
    assert torch.allclose(T.sum(1), torch.ones_like(T.sum(1)))


@pytest.mark.parametrize('mode', ['free', 'transport'])
def test_uniform_state_unchanged(mode):
    m = scrambled(mode)
    h = states(1, B=1).expand(1, 3, 10 + 6 * m.k).contiguous()
    with torch.no_grad():
        U = stepper.advance(m, [h], torch.full((1,), m.r0 * DX, dtype=D64), DX)
    assert torch.allclose(U, h[..., 3 * m.k:-3 * m.k], rtol=1e-12)


def test_init():
    h = states(20)
    m = PairModel(2, mode='free', cfl=2.0, dtype=D64)
    with torch.no_grad():
        U = stepper.advance(m, [h], torch.full((2,), m.r0 * DX, dtype=D64), DX)
        assert torch.equal(U, h[..., 6:-6])
        _, T = PairModel(2, mode='transport', cfl=2.0, dtype=D64).exchanges(h / QS)
    assert torch.allclose(T[:, 2], torch.full_like(T[:, 2], 0.9))


def test_consistency_probe_and_load(tmp_path):
    m = scrambled('transport')
    S = states(1, B=50)[..., 0]
    assert consistency.flux_at(m, S).shape == (50, 3)
    m.save(tmp_path / 'm.pt', step=7, dt_mode='global')
    k = NFVModel.load(tmp_path / 'm.pt', D64)
    assert isinstance(k, PairModel) and (k.k, k.mode, k.cfl, k.extra['step']) == (2, 'transport', 2.0, 7)
    x = torch.rand(1, 3, 30, dtype=D64) + 0.5
    with torch.no_grad():
        assert torch.equal(k(x), m(x))
    r = sp.plateaus(k, nx=200)
    assert all(np.isfinite(list(r[s].values())).all() for s in 'LR')


def test_build():
    m = build('pair', k=4, mode='transport', cfl=3.0)
    assert isinstance(m, PairModel) and m.k == 4 and m.cfl == 3.0


def argv(tmp, **kw):
    a = dict(arch='pair', cfl=2.0, schedule='6:1e-3:20:4', n_train=16, batch_size=8, n_eval=4, eval_nx=30,
             eval_nt=6, eval_every=3, data_dir=str(tmp / 'd'), out=str(tmp / 'o'), device='cpu')
    a.update(kw)
    return [f'--{k}={v}' for k, v in a.items()]


def test_training_smoke(tmp_path):
    out = train_nfv.main(argv(tmp_path, pair_mode='transport', dt_mode='global'))
    L = [json.loads(l) for l in open(out + '/log.jsonl')]
    assert np.isfinite(L[-1]['loss'])
    cfg = json.load(open(out + '/config.json'))
    assert (cfg['k'], cfg['a']) == (3, 18)
    m = NFVModel.load(out + '/model_final.pt')
    assert isinstance(m, PairModel) and m.extra['dt_mode'] == 'global'
    assert (tmp_path / 'd' / 'eval_nx30_nt6_gdt_cfl2.npz').exists()


def test_transport_needs_global_dt(tmp_path):
    with pytest.raises(ValueError):
        train_nfv.parse(argv(tmp_path, pair_mode='transport'))
