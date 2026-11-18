import json
import numpy as np
import pytest
import torch

import train_nfv
import sod_plateaus as sp
from euler1d.neural import stepper, consistency, scaling
from euler1d.neural.model import NFVModel, build
from euler1d.neural.pairwise import PairModel, P_STAY

DX = 1e-3
D64 = torch.float64
QS = torch.tensor(scaling.QS).view(1, 3, 1)
FS = torch.tensor(scaling.FS).view(1, 3)


def states(W, B=2, seed=0):
    r = np.random.default_rng(seed)
    rho, u, p = r.uniform(0.1, 2.5, (B, W)), r.normal(0, 200, (B, W)), r.uniform(1e4, 1.25e5, (B, W))
    return torch.from_numpy(np.stack([rho, rho * u, p / 0.4 + 0.5 * rho * u**2], axis=1))


def scrambled(k=2, prior=0, std=0.3, seed=0):
    torch.manual_seed(seed)
    m = PairModel(k, mode='attn', cfl=4.0, dtype=D64, prior=prior)
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(std=std)
    return m


def advance(m, h):
    with torch.no_grad():
        return stepper.advance(m, [h], torch.full((len(h),), m.r0 * DX, dtype=D64), DX)


def explicit(m, h):
    '''pairwise update cell by cell; h: (B, 3, N + 6k) physical'''
    k, N, r = m.k, h.shape[-1] - 6 * m.k, m.r0
    with torch.no_grad():
        P, T = m.exchanges(h / QS)
    out = h[..., 3 * k:3 * k + N].clone()
    for c in range(3 * k, 3 * k + N):
        dq = 0
        for d in range(1, k + 1):
            dq = dq - P[:, d - 1, :, c - 2 * k] * FS + P[:, d - 1, :, c - d - 2 * k] * FS
        dq = r * dq - h[..., c] + sum(T[:, d + k, c - d - 2 * k, None] * h[..., c - d] for d in range(-k, k + 1))
        out[..., c - 3 * k] += dq
    return out


@pytest.mark.parametrize('prior', [0, 1])
def test_matches_explicit_pairwise(prior):
    m = scrambled(prior=prior)
    h = states(12 + 6 * m.k)
    assert torch.allclose(advance(m, h), explicit(m, h), rtol=1e-10, atol=1e-8)


def test_fractions_and_positive_density():
    m = scrambled(std=2.0)
    h = states(30 + 6 * m.k)
    h[:, 0, 20:25] = 1e-3
    with torch.no_grad():
        P, T = m.exchanges(h / QS)
    assert T.shape == (2, 2 * m.k + 1, h.shape[-1] - 4 * m.k) and (T >= 0).all()
    assert torch.allclose(T.sum(1), torch.ones_like(T.sum(1)))
    assert (P[:, :, 0] == 0).all() and P.abs().max() > 0
    assert (advance(m, h)[:, 0] >= -1e-12).all()


@pytest.mark.parametrize('prior', [0, 1])
def test_uniform_state_unchanged(prior):
    m = scrambled(prior=prior)
    h = states(1, B=1).expand(1, 3, 10 + 6 * m.k).contiguous()
    assert torch.allclose(advance(m, h), h[..., 3 * m.k:-3 * m.k], rtol=1e-12)


def test_init_stays():
    m = PairModel(2, mode='attn', cfl=4.0, dtype=D64)
    with torch.no_grad():
        P, T = m.exchanges(states(20) / QS)
    assert torch.allclose(T[:, 2], torch.full_like(T[:, 2], P_STAY)) and (P == 0).all()


def test_prior_peaks_at_flow_displacement():
    m = PairModel(3, mode='attn', cfl=4.0, dtype=D64, prior=1)
    with torch.no_grad():
        m.rel.zero_()
        m.beta.fill_(5.0)
    u = 2.0 / m.r0
    rho = torch.ones(1, 1, 30, dtype=D64)
    h = torch.cat([rho, rho * u, 3e5 * rho], 1)
    with torch.no_grad():
        _, T = m.exchanges(h / QS)
    assert (T.argmax(1) == m.k + 2).all()


def test_save_load_and_sod(tmp_path):
    m = scrambled(k=3, prior=1, std=0.05)
    m.save(tmp_path / 'm.pt', step=3, dt_mode='global')
    k = NFVModel.load(tmp_path / 'm.pt', D64)
    assert isinstance(k, PairModel) and (k.mode, k.emb, k.dk, k.prior, k.extra['step']) == ('attn', 16, 8, 1, 3)
    x = torch.rand(1, 3, 40, dtype=D64) + 0.5
    with torch.no_grad():
        assert torch.equal(k(x), m(x))
    S = states(1, B=20)[..., 0]
    assert consistency.flux_at(k, S).shape == (20, 3)
    r = sp.plateaus(k, nx=200)
    assert all(np.isfinite(list(r[s].values())).all() for s in 'LR')


def test_build_sizes():
    m = build('pair', k=5, mode='attn', cfl=4.0, emb=8, dk=4)
    assert (m.emb, m.dk, m.a, m.g) == (8, 4, 30, 15) and m.wq.weight.shape == (4, 8, 1)


def argv(tmp, **kw):
    a = dict(arch='pair', pair_mode='attn', cfl=4.0, dt_mode='global', schedule='6:1e-3:30:3', n_train=16,
             batch_size=8, n_eval=4, eval_nx=40, eval_nt=4, eval_every=3, data_dir=str(tmp / 'd'),
             out=str(tmp / 'o'), device='cpu')
    a.update(kw)
    return [f'--{k}={v}' for k, v in a.items()]


def test_training_smoke(tmp_path):
    out = train_nfv.main(argv(tmp_path, attn_prior=1))
    L = [json.loads(l) for l in open(out + '/log.jsonl')]
    assert np.isfinite(L[-1]['loss'])
    m = NFVModel.load(out + '/model_final.pt')
    assert isinstance(m, PairModel) and m.mode == 'attn' and m.prior == 1 and m.k == 5


def test_attn_needs_global_dt(tmp_path):
    with pytest.raises(ValueError):
        train_nfv.parse(argv(tmp_path, dt_mode='per_sample'))
