import json
import numpy as np
import torch

import ablate_nfv
import sod_plateaus as sp
from infrastructure.nfv_dataset import make_set, fname
from euler1d.neural import evaluate as ev
from euler1d.neural.baseline import first_order
from euler1d.neural.model import NFVModel

DX = 1e-3


def test_fname_keeps_old_names():
    assert fname('train', 10, 10) == 'train_nx10_nt10.npz'
    assert fname('train', 10, 10, cfl=0.5) == 'train_nx10_nt10.npz'
    assert fname('train', 10, 10, cfl=2.0) == 'train_nx10_nt10_cfl2.npz'
    assert fname('eval', 200, 50, 'global', 1.5) == 'eval_nx200_nt50_gdt_cfl1.5.npz'


def test_n_sub():
    assert [ev.n_sub(c) for c in (0.3, 0.5, 0.6, 1.0, 2.0, 4.0)] == [1, 1, 2, 2, 4, 8]


def err(P, d):
    G = d['G']
    return np.abs(P[:, -1] - d['Q'][:, -1, :, G:-G]).mean() / np.abs(d['Q'][:, -1, :, G:-G]).mean()


def test_substepped_baseline():
    d = make_set(4, 80, 6, DX, 'eval', 0, cfl=2.0, G=5, dtype=np.float64)
    Q, G = d['Q'], d['G']
    bad = first_order(Q, G, d['dt'], DX)
    good = first_order(Q, G, d['dt'], DX, sub=4)
    assert np.isfinite(good).all() and err(good, d) < 0.05
    assert not np.isfinite(bad).all() or err(bad, d) > 2 * err(good, d)
    P = ev.base_pred(d)
    assert torch.allclose(P, torch.from_numpy(good))


def test_sub1_unchanged():
    d = make_set(3, 40, 8, DX, 'eval', 0, dtype=np.float64)
    assert np.array_equal(first_order(d['Q'], 3, d['dt'], DX), first_order(d['Q'], 3, d['dt'], DX, sub=1))


def test_scale_cfl():
    kw = dict(schedule='10:1e-4:10:10,20:1e-5:50:50', eval_nt=400, eval_nx=200, cfl=2.0)
    s = ablate_nfv.scale_cfl(dict(kw))
    assert s['schedule'] == '10:0.0001:10:2,20:1e-05:50:12' and s['eval_nt'] == 100
    s = ablate_nfv.scale_cfl(dict(kw), 'nx')
    assert s['schedule'] == '10:0.0001:40:10,20:1e-05:200:50' and s['eval_nx'] == 800
    s = ablate_nfv.scale_cfl(dict(kw, cfl=4.0, b=3))
    assert s['schedule'].split(',')[0].endswith(':4')
    assert ablate_nfv.scale_cfl(dict(kw, cfl=0.5)) == dict(kw, cfl=0.5)


def test_sod_uses_checkpoint_cfl(tmp_path):
    m = NFVModel(6, 1, dtype=torch.float64)
    m.extra = dict(cfl=2.0)
    r = sp.plateaus(m)
    assert set(r) == {'L', 'R'} and all(np.isfinite(list(r[s].values())).all() for s in 'LR')


def test_ablate_cfl_variant(tmp_path):
    a = ['--variants=cfl2', '--schedule=4:1e-3:10:10', f'--out={tmp_path / "ab"}', f'--data_dir={tmp_path / "d"}',
         '--n_train=16', '--batch_size=8', '--n_eval=4', '--eval_nx=40', '--eval_nt=20', '--eval_every=4',
         '--device=cpu']
    S = ablate_nfv.main(a)
    r = json.load(open(tmp_path / 'ab' / 'cfl2_s0' / 'result.json'))
    cfg = json.load(open(tmp_path / 'ab' / 'cfl2_s0' / 'config.json'))
    assert (cfg['cfl'], cfg['a'], cfg['schedule'], cfg['eval_nt']) == (2.0, 6, '4:0.001:10:2', 5)
    assert (r['cfl'], r['base_sub'], r['H']) == (2.0, 4, 2) and S['cfl2']['cfl'] == 2.0
    assert (tmp_path / 'd' / 'train_nx10_nt2_cfl2.npz').exists()
    assert NFVModel.load(tmp_path / 'ab' / 'cfl2_s0' / 'model_final.pt').extra['cfl'] == 2.0
