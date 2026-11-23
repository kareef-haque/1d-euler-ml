import json
import math
import numpy as np
import pytest
import torch

import eval_dg
import train_dg
from euler1d.dg import data as dd, solver as dg
from euler1d.dg.flux import DissFlux, S_MAX

D = torch.float64


def scrambled(mode, p=1, std=0.5, seed=0, s_min=0.0):
    torch.manual_seed(seed)
    m = DissFlux(p, mode, s_min=s_min)
    with torch.no_grad():
        for q in m.parameters():
            q.add_(std * torch.randn_like(q))
    return m


@pytest.fixture(scope='module')
def td():
    return dd.tensors(dd.make_set(6, 30, 6, 1e-3, 1, 'test', 0))


@pytest.mark.parametrize('mode', ['scalar', 'char'])
def test_consistent_and_starts_as_rusanov(mode, td):
    Up = torch.cat([td['Gh'][:, 0, :, :1], td['Q'][:, 0], td['Gh'][:, 0, :, 1:]], 2)
    tr = dg.at(Up, torch.tensor([0.5, -0.5], dtype=D))
    QL, QR = tr[:, :, :-1, 0], tr[:, :, 1:, 1]
    m = scrambled(mode)
    with torch.no_grad():
        assert torch.allclose(m(QL, QL, Up), dg.euler_flux(QL), rtol=1e-12)
        assert torch.allclose(DissFlux(1, mode)(QL, QR, Up), dg.rusanov(QL, QR), rtol=1e-9)


@pytest.mark.parametrize('s_min', [0.0, 0.5])
def test_scale_bounds(s_min, td):
    m = scrambled('char', std=3.0, s_min=s_min)
    Up = torch.cat([td['Gh'][:, 0, :, :1], td['Q'][:, 0], td['Gh'][:, 0, :, 1:]], 2)
    with torch.no_grad():
        s = m.scale(Up)
    assert s.shape == (6, 3, 31) and (s >= s_min).all() and (s <= S_MAX).all()
    assert torch.allclose(DissFlux(1, 'scalar', s_min=s_min).scale(Up), torch.ones(6, 1, 31, dtype=D))


@pytest.mark.parametrize('mode', ['scalar', 'char'])
def test_uniform_state_and_save_load(mode, tmp_path):
    s = np.array([[1.2, 50.0, 6e4]])
    d = dd.tensors(dd.from_states(s, s, np.array([0.5]), 20, 4, 1e-3, 1))
    m = scrambled(mode)
    with torch.no_grad():
        P = dd.rollout(d, m)
    assert torch.allclose(P, d['Q'][:, :1].expand_as(P), rtol=1e-10)
    m.save(tmp_path / 'm.pt', step=1)
    k = DissFlux.load(tmp_path / 'm.pt')
    with torch.no_grad():
        assert torch.equal(dd.rollout(d, k), P)


def test_train_and_eval_smoke(tmp_path):
    args = ['--p', '1', '--schedule', '4:1e-3:16:4,2:5e-4:20:6', '--n_train', '8', '--batch', '4', '--n_eval', '4',
            '--eval_nx', '30', '--eval_nt', '6', '--eval_every', '2', '--data_dir', str(tmp_path / 'd'),
            '--out', str(tmp_path / 'o'), '--device', 'cpu', '--s_min', '0.3']
    out = train_dg.main(args)
    L = [json.loads(l) for l in open(tmp_path / 'o' / 'log.jsonl')]
    assert len(L) >= 3 and math.isfinite(L[-1]['loss'])
    assert DissFlux.load(tmp_path / 'o' / 'model_final.pt').s_min == 0.3
    res = eval_dg.main(['--model', str(tmp_path / 'o' / 'model_final.pt'), '--n_test', '4', '--test_nx', '30',
                        '--test_nt', '6', '--named_N', '40', '--data_dir', str(tmp_path / 'd')])
    assert set(res['test']) == {'learned', 'DG-HLLC', 'DG-Rusanov', 'FV HLLC-1'}
    assert set(res['named']) == {'sod', 'lax', '123'} and abs(res['convergence']['DG-HLLC']['rate'] - 2) < 0.3
    # a second seed must reuse the same cached datasets (the run seed is not the data seed)
    args[args.index('--out') + 1] = str(tmp_path / 'o1')
    train_dg.main(args + ['--seed', '1'])
    assert (tmp_path / 'o1' / 'model_final.pt').exists()
