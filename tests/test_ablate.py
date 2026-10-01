import json
import numpy as np
import pytest
import torch

import ablate_nfv
import train_nfv
from infrastructure.nfv_dataset import make_set, from_states, S_GLOBAL, fname
from euler1d.neural import rollouts as ro, stepper
from euler1d.neural.model import NFVModel

DX = 1e-3


def small_args(tmp, **kw):
    a = ['--variants=base,b3,gdt', '--schedule=4:1e-3:10:10', f'--out={tmp / "ab"}', f'--data_dir={tmp / "d"}',
         '--n_train=16', '--batch_size=8', '--n_eval=4', '--eval_nx=40', '--eval_nt=20', '--eval_every=4',
         '--device=cpu']
    return a + [f'--{k}={v}' for k, v in kw.items()]


def test_global_dt_dataset():
    g = make_set(6, 40, 10, DX, 'train', 0, dt_mode='global')
    p = make_set(6, 40, 10, DX, 'train', 0)
    assert np.allclose(g['dt'], 0.5 * DX / S_GLOBAL) and g['dt_mode'] == 'global'
    assert np.array_equal(g['pl'], p['pl']) and np.array_equal(g['xf'], p['xf'])
    assert np.all(g['dt'] * g['smax'] / DX <= 0.5 + 1e-12) and np.allclose(p['dt'] * p['smax'] / DX, 0.5)
    assert fname('train', 10, 10) != fname('train', 10, 10, 'global')
    assert S_GLOBAL >= 1486


def test_nfv_history_matches_stepper():
    torch.manual_seed(0)
    for b in (2, 3):
        m = NFVModel(2, b, dtype=torch.float64)
        d = make_set(2, 60, 12, DX, 'eval', 0, dtype=np.float64)
        P = ro.predict_all(d, ['NFV'], m)['NFV']
        want = stepper.rollout(m, torch.from_numpy(d['Q']), 3, torch.from_numpy(d['dt']), DX, 'exact').detach().numpy()
        assert np.allclose(P, want, rtol=1e-9, atol=1e-6)


def test_train_rejects_b_not_below_nt(tmp_path):
    with pytest.raises(ValueError):
        train_nfv.main(['--b=10', '--schedule=2:1e-3:10:10', f'--out={tmp_path / "o"}', f'--data_dir={tmp_path / "d"}'])


def test_global_dt_training_uses_own_files(tmp_path):
    base = ['--schedule=2:1e-3:10:10', '--n_train=8', '--batch_size=8', '--n_eval=4', '--eval_nx=40', '--eval_nt=20',
            '--eval_every=2', '--device=cpu', f'--data_dir={tmp_path / "d"}']
    train_nfv.main(base + [f'--out={tmp_path / "o1"}', '--dt_mode=global'])
    train_nfv.main(base + [f'--out={tmp_path / "o2"}'])
    names = sorted(p.name for p in (tmp_path / 'd').iterdir())
    assert 'train_nx10_nt10_gdt.npz' in names and 'train_nx10_nt10.npz' in names and 'eval_nx40_nt20.npz' in names


def test_ablation_driver_end_to_end_and_resume(tmp_path, monkeypatch):
    S = ablate_nfv.main(small_args(tmp_path, seeds='0,1'))
    assert set(S) == {'base', 'b3', 'gdt'} and all(s['n_ok'] == 2 and len(s['ratio_h']) == 2 for s in S.values())
    assert S['b3']['params'] == 45 * 2 * 3 + 1263 and S['base']['params'] == 1353
    assert (tmp_path / 'ab' / 'summary.json').exists() and (tmp_path / 'ab' / 'gdt_s1' / 'result.json').exists()
    r = json.load(open(tmp_path / 'ab' / 'base_s0' / 'result.json'))
    assert r['H'] == 10 and 'cons' in r and np.isfinite(r['test']['l2h'])
    calls = {'n': 0}
    real = train_nfv.main
    monkeypatch.setattr(train_nfv, 'main', lambda *a, **k: calls.__setitem__('n', calls['n'] + 1) or real(*a, **k))
    ablate_nfv.main(small_args(tmp_path, seeds='0,1'))
    assert calls['n'] == 0


def test_ablation_records_failures(tmp_path, monkeypatch):
    monkeypatch.setattr(train_nfv, 'bad', lambda gn, loss: True)
    S = ablate_nfv.main(small_args(tmp_path, variants='base,relu'))
    assert all(s['n_ok'] == 0 and s['failed_at'] == [0] for s in S.values())