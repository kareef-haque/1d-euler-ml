import json
import numpy as np
import pytest
import torch

import train_nfv
import sod_plateaus as sp
from infrastructure.nfv_dataset import from_states
from euler1d.neural import stepper, consistency, scaling
from euler1d.neural.model import NFVModel, build
from euler1d.neural.pairwise import PairModel, NU0, C0

DX = 1e-3
D64 = torch.float64
QS = torch.tensor(scaling.QS).view(1, 3, 1)


def states(W, B=2, seed=0):
    r = np.random.default_rng(seed)
    rho, u, p = r.uniform(0.1, 2.5, (B, W)), r.normal(0, 200, (B, W)), r.uniform(1e4, 1.25e5, (B, W))
    return torch.from_numpy(np.stack([rho, rho * u, p / 0.4 + 0.5 * rho * u**2], axis=1))


def scrambled(k=3, limit=1, std=0.3, seed=0):
    torch.manual_seed(seed)
    m = PairModel(k, mode='split', cfl=4.0, dtype=D64, limit=limit)
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(std=std)
    return m


def advance(m, h):
    with torch.no_grad():
        return stepper.advance(m, [h], torch.full((len(h),), m.r0 * DX, dtype=D64), DX)


def prim(U):
    rho = U[:, 0]
    return rho, (0.4 * (U[:, 2] - 0.5 * U[:, 1] ** 2 / rho))


def test_sizes():
    m = PairModel(5, mode='split', cfl=4.0)
    assert (m.a, m.g) == (32, 16) and m(torch.ones(2, 3, 40)).shape == (2, 3, 40 - 32 + 1)
    with pytest.raises(ValueError):
        PairModel(2, mode='split')


@pytest.mark.parametrize('limit', [0, 1])
def test_uniform_flow_is_exact(limit):
    m = scrambled(k=5, limit=limit)
    S = states(1, B=200)[..., 0]
    assert (S[:, 1] / S[:, 0]).abs().max() * m.r0 < m.k - 2
    with torch.no_grad():
        F = consistency.flux_at(m, S)
    assert torch.allclose(F, consistency.phys_flux(S), rtol=1e-10, atol=1e-6)
    rh = sp.rh_residual(sp.nfv_diag_flux(m))
    assert np.allclose(rh['shock']['over_FS'], 0, atol=1e-9) and np.allclose(rh['contact']['over_FS'], 0, atol=1e-9)


def test_fractions_have_the_shift_as_mean():
    m = scrambled(std=1.0)
    h = states(20 + m.a)
    with torch.no_grad():
        Q, T, sh, A, Fc = m.split_parts(h / QS)
    d = torch.arange(-m.k, m.k + 1, dtype=D64).view(1, -1, 1)
    assert (T >= 0).all() and torch.allclose(T.sum(1), torch.ones_like(T.sum(1)))
    assert torch.allclose((T * d).sum(1), sh, atol=1e-12)
    assert sh.abs().max() > 0.05


def test_init():
    m = PairModel(3, mode='split', cfl=4.0, dtype=D64)
    h = states(30)
    with torch.no_grad():
        H = m.enc(h / QS)
        z = m.wnu(torch.cat([H[:, :, :-1], H[:, :, 1:]], 1))
        nu, pi = 0.5 * torch.sigmoid(z[:, :1]), z[:, 1:].softmax(1)
        c = 0.5 * torch.sigmoid(m.whead(H))
    assert torch.allclose(nu, torch.full_like(nu, NU0)) and torch.allclose(pi, torch.full_like(pi, 1 / 3))
    assert torch.allclose(c, torch.full_like(c, C0)) and (m.cb == 0).all()


def test_init_is_stable_on_sod():
    m = PairModel(5, mode='split', cfl=4.0, dtype=D64)
    m.extra = dict(cfl=4.0, dt_mode='global')
    r = sp.plateaus(m)
    v = [r[s][q] for s in 'LR' for q in ('rho', 'u', 'p')]
    assert np.isfinite(v).all() and np.mean(np.abs(v)) < 0.15


def test_limiter_keeps_cells_valid():
    d = from_states(np.array([[2.4, 0.0, 1.2e5]]), np.array([[0.1, 0.0, 1.0e4]]), 0.5, 60, 1, DX, 0.5,
                    G=PairModel(3, mode='split').g, dtype=np.float64)
    h = torch.from_numpy(d['Q'][:, 0])
    bad = {}
    for limit in (0, 1):
        m = scrambled(limit=limit, std=1.5, seed=3)
        rho, p = prim(advance(m, h))
        bad[limit] = int(((rho <= 0) | (p <= 0)).sum())
    assert bad[1] == 0


def test_conservative():
    m = scrambled()
    h = states(40 + m.a, B=1)
    with torch.no_grad():
        F = m(h / QS) * torch.tensor(scaling.FS).view(1, 3, 1)
    U = advance(m, h)
    g = m.g
    change = (U - h[..., g:-g]).sum(-1)
    assert torch.allclose(change, -m.r0 * (F[..., -1] - F[..., 0]), rtol=1e-9, atol=1e-6)


def test_save_load_build_sod(tmp_path):
    m = PairModel(5, mode='split', cfl=4.0, dtype=D64, limit=1)
    m.save(tmp_path / 'm.pt', step=2, dt_mode='global')
    k = NFVModel.load(tmp_path / 'm.pt', D64)
    assert isinstance(k, PairModel) and (k.mode, k.limit, k.a) == ('split', 1, 32)
    x = torch.rand(1, 3, 50, dtype=D64) + 0.5
    with torch.no_grad():
        assert torch.equal(k(x), m(x))
    r = sp.plateaus(k, nx=200)
    assert all(np.isfinite(list(r[s].values())).all() for s in 'LR')
    assert build('pair', k=5, mode='split', cfl=4.0, limit=0).limit == 0


def argv(tmp, **kw):
    a = dict(arch='pair', pair_mode='split', cfl=4.0, dt_mode='global', schedule='6:1e-3:30:3', n_train=16,
             batch_size=8, n_eval=4, eval_nx=40, eval_nt=4, eval_every=3, data_dir=str(tmp / 'd'),
             out=str(tmp / 'o'), device='cpu')
    a.update(kw)
    return [f'--{k}={v}' for k, v in a.items()]


def test_training_smoke(tmp_path):
    out = train_nfv.main(argv(tmp_path))
    L = [json.loads(l) for l in open(out + '/log.jsonl')]
    assert np.isfinite(L[-1]['loss']) and L[-1]['pen'] < 1e-12
    cfg = json.load(open(out + '/config.json'))
    assert (cfg['k'], cfg['a']) == (5, 32)
    m = NFVModel.load(out + '/model_final.pt')
    assert m.mode == 'split' and m.limit == 1


def test_split_needs_global_dt(tmp_path):
    with pytest.raises(ValueError):
        train_nfv.parse(argv(tmp_path, dt_mode='per_sample'))


def test_dataset_with_fewer_ghosts_is_kept(tmp_path):
    a = argv(tmp_path, pair_mode='transport', out=str(tmp_path / 't'))
    train_nfv.main(a)
    before = sorted(p.name for p in (tmp_path / 'd').iterdir())
    train_nfv.main(argv(tmp_path, out=str(tmp_path / 's')))
    after = sorted(p.name for p in (tmp_path / 'd').iterdir())
    assert set(before) < set(after) and any(n.endswith('_g16.npz') for n in after)
