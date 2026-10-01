import json
import numpy as np
import pytest
import torch

import train_nfv
from infrastructure.solver_config import EulerConfig
from infrastructure.nfv_dataset import make_set
from euler1d.solver import EulerSolver
from euler1d.neural import evaluate as ev
from euler1d.neural.baseline import first_order
from euler1d.neural.diagnostics import consistency_error, gap
from euler1d.neural.model import NFVModel
from euler1d.neural import scaling

DX = 1e-3


def argv(tmp, **kw):
    a = dict(schedule='40:1e-3:10:10,20:1e-3:20:20', n_train=32, batch_size=16, n_eval=6, eval_nx=40,
             eval_nt=30, eval_every=20, data_dir=str(tmp / 'd'), out=str(tmp / 'o'), device='cpu')
    a.update(kw)
    return [f'--{k}={v}' for k, v in a.items()]


def logs(out):
    return [json.loads(l) for l in open(out + '/log.jsonl')]


def test_parse_schedule():
    s = train_nfv.parse_schedule(train_nfv.SCHEDULE)
    assert s[0] == (10000, 1e-4, 10, 10) and s[-1] == (20000, 1e-6, 200, 200) and len(s) == 4


def test_baseline_matches_solver():
    d = make_set(3, 200, 40, DX, 'eval', 0, dtype=np.float64)
    P = first_order(d['Q'], d['G'], d['dt'], DX)
    for i in range(3):
        cfg = EulerConfig(domain_size=0.2, N_cells=200, IC=d['Q'][i, 0, :, 3:-3], t_max=40 * d['dt'][i], dt=d['dt'][i])
        r = EulerSolver(cfg, flux='HLLC', reconstruction='Const', integrator='euler', verbose=False)
        assert np.allclose(np.stack(r.Q_hist), P[i], rtol=1e-10, atol=1e-8)


def test_summarize_perfect_and_diverged():
    d = make_set(4, 40, 20, DX, 'eval', 0, dtype=np.float64)
    ed = ev.prep(d)
    G = d['G']
    ref = ed['Q'][..., G:-G]
    Pb = ev.base_pred(d)
    P = ref.clone()
    P[1, 5, 0, 3] = float('nan')
    s, sb = ev.summarize(P, ref, 1, 10), ev.summarize(Pb, ref, 1, 10)
    assert s['l2'][0] == 0 and s['l2'][1] == float('inf') and s['l2h'][1] == float('inf')
    r = ev.report(s, sb)
    assert r['n_div'] == 1 and r['win'] == 0.75
    P2 = ref.clone()
    P2[1, 15, 0, 3] = float('nan')
    s2 = ev.summarize(P2, ref, 1, 10)
    assert s2['l2'][1] == float('inf') and np.isfinite(s2['l2h'][1])


def test_consistency_probe():
    m = NFVModel(2, 1, dtype=torch.float64)
    c = consistency_error(m)
    assert c['err'].shape == c['ref'].shape == (3,) and np.all(np.isfinite(c['err']))
    m.net[-1].weight.data.zero_()
    m.net[-1].bias.data.zero_()
    c = consistency_error(m)
    assert np.allclose(c['err'], c['ref'])


def test_gap_ignores_constant_offset():
    rng = np.random.default_rng(0)
    f = rng.normal(size=(500, 3)) * scaling.FS
    g = gap(f + np.array([0.3, -0.7, 0.1]) * scaling.FS, f)
    assert np.allclose(g['offset'], [0.3, -0.7, 0.1]) and np.allclose(g['err_shift'], 0, atol=1e-12)
    assert np.all(g['err'] > 0.05)


def test_training_smoke(tmp_path):
    out = train_nfv.main(argv(tmp_path))
    L = logs(out)
    assert [r['step'] for r in L] == [1, 21, 40, 41, 60]
    assert L[2]['loss'] < L[0]['loss']
    assert all(np.isfinite(r['l2h']) for r in L) and L[0]['attempt'] == 0
    for f in ('config.json', 'model_best.pt', 'model_final.pt', 'model_60.pt'):
        assert (tmp_path / 'o' / f).exists()
    m = NFVModel.load(out + '/model_final.pt')
    assert (m.a, m.b, m.extra['gamma'], m.extra['step']) == (2, 1, 1.4, 60)
    cfg = EulerConfig(domain_size=0.04, N_cells=40, IC=make_set(1, 40, 1, DX, 'test', 0)['Q'][0, 0, :, 3:-3],
                      t_max=5e-6, dt=1e-6)
    r = EulerSolver(cfg, flux='NFV', nfv=out + '/model_final.pt', verbose=False)
    assert len(r.Q_hist) == 6 and np.isfinite(r.Q_hist[-1]).all()


def test_stencil_and_history_variants(tmp_path):
    for i, (a, b) in enumerate([(10, 1), (4, 3)]):
        out = train_nfv.main(argv(tmp_path, a=a, b=b, schedule='3:1e-3:10:10', out=str(tmp_path / f'o{i}'),
                                  eval_every=3))
        assert np.isfinite(logs(out)[-1]['loss'])


def test_restart_then_success(tmp_path, monkeypatch):
    calls = {'n': 0}
    real = train_nfv.bad

    def flaky(gn, loss):
        calls['n'] += 1
        return calls['n'] == 3 or real(gn, loss)
    monkeypatch.setattr(train_nfv, 'bad', flaky)
    out = train_nfv.main(argv(tmp_path, schedule='10:1e-3:10:10', eval_every=5))
    assert sorted({r['attempt'] for r in logs(out)}) == [0, 1]


def test_restart_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(train_nfv, 'bad', lambda gn, loss: True)
    with pytest.raises(RuntimeError):
        train_nfv.main(argv(tmp_path, schedule='5:1e-3:10:10', max_restarts=2))


def test_stale_data_file_is_rejected(tmp_path):
    train_nfv.main(argv(tmp_path, schedule='2:1e-3:10:10', eval_every=2))
    with pytest.raises(ValueError):
        train_nfv.main(argv(tmp_path, schedule='2:1e-3:10:10', dx=2e-3, out=str(tmp_path / 'o2')))


def test_loss_is_scaled_mse():
    d = make_set(2, 10, 5, DX, 'train', 0, dtype=np.float64)
    Q = torch.from_numpy(d['Q'])
    ref = Q[..., 3:-3]
    P = ref * 1.01
    e = ((P - ref) / torch.as_tensor(scaling.QS).view(1, 1, 3, 1)).pow(2).mean()
    want = ((0.01 * ref) / torch.as_tensor(scaling.QS).view(1, 1, 3, 1)).pow(2).mean()
    assert torch.allclose(e, want)