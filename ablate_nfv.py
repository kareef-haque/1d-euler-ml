'''
Ablation driver for vanilla NFV.

Trains each (variant, seed) with the same schedule, restarts disabled (a failed run is recorded,
not hidden), then evaluates the final model on the held-out TEST split against first-order HLLC.
Resumable: finished runs are skipped. Results: <out>/<variant>_s<seed>/result.json and <out>/summary.json

variants: base (a=2, b=1, ELU), a4, a6, b3, a4b5 (stencil / time history), relu (activation),
          gdt (global dt instead of per-sample dt for the training data)
'''
import argparse
import json
import os
import time
import numpy as np

import train_nfv
from euler1d.neural import evaluate as ev, diagnostics as dg
from euler1d.neural.model import NFVModel

VARIANTS = {
    'base': {},
    'a4': {'a': 4},
    'a6': {'a': 6},
    'b3': {'b': 3},
    'a4b5': {'a': 4, 'b': 5},
    'relu': {'act': 'ReLU'},
    'gdt': {'dt_mode': 'global'},
}
SCHEDULE = '10000:1e-4:10:10,20000:1e-5:50:50,20000:5e-6:100:100'


def last_log(path):
    f = os.path.join(path, 'log.jsonl')
    L = [json.loads(l) for l in open(f)] if os.path.exists(f) else []
    return L[-1] if L else {}


def train_eval(variant, seed, a):
    out = os.path.join(a.out, f'{variant}_s{seed}')
    res_path = os.path.join(out, 'result.json')
    if os.path.exists(res_path):
        return json.load(open(res_path))
    os.makedirs(out, exist_ok=True)
    kw = dict(schedule=a.schedule, n_train=a.n_train, batch_size=a.batch_size, n_eval=a.n_eval,
              eval_nx=a.eval_nx, eval_nt=a.eval_nt, eval_every=a.eval_every, seed=seed,
              data_dir=a.data_dir, out=out, device=a.device, max_restarts=0)
    kw.update(VARIANTS[variant])
    argv = [f'--{k}={v}' for k, v in kw.items()]
    t0 = time.time()
    r = dict(variant=variant, seed=seed, failed=False)
    try:
        train_nfv.main(argv)
    except RuntimeError:
        r['failed'] = True
    r['sec'] = round(time.time() - t0, 1)
    r['last'] = last_log(out)
    r['steps_done'] = r['last'].get('step', 0)
    if not r['failed']:
        args = train_nfv.parse(argv)
        d = train_nfv.get_set(args, 'test', a.n_eval, a.eval_nx, a.eval_nt)
        d = {k: (v[:a.n_eval] if k in ('Q', 'dt', 'smax', 'pl', 'pr', 'xf') else v) for k, v in d.items()}
        model = NFVModel.load(os.path.join(out, 'model_final.pt'))
        H = int(r['last']['nt'])
        r['params'] = model.num_params()
        r['test'] = ev.evaluate(model, ev.prep(d), ev.base_pred(d), H)
        r['H'] = H
        c = dg.consistency_error(model, states=dg.states_from_set(d))
        r['cons'] = {k: v.tolist() for k, v in c.items()}
    json.dump(r, open(res_path, 'w'), indent=1, default=float)
    return r


def med(x):
    return float(np.median(x)) if len(x) else float('nan')


def summarize(runs):
    out = {}
    for v in dict.fromkeys(r['variant'] for r in runs):
        R = [r for r in runs if r['variant'] == v]
        ok = [r for r in R if not r['failed']]
        t = [r['test'] for r in ok]
        out[v] = dict(
            n=len(R), n_ok=len(ok), params=ok[0]['params'] if ok else None,
            ratio_h=[x['ratioh'] for x in t], win_h=[x['winh'] for x in t],
            ratio_full=[x['ratio'] for x in t], n_div_full=[x['n_div'] for x in t],
            cons_shift=[float(np.mean(np.array(r['cons']['err_shift']) / np.array(r['cons']['ref_std']))) for r in ok],
            sec=[r['sec'] for r in R],
            failed_at=[r['steps_done'] for r in R if r['failed']])
    return out


def show(S, H):
    print(f"\nTEST split. ratio = first-order HLLC error / model error (higher is better); h = {H} steps\n")
    print(f"{'variant':8s} {'ok':>5s} {'params':>7s} {'ratio_h med [min,max]':>24s} {'win_h':>6s} {'ratio@full':>11s} "
          f"{'diverged':>9s} {'flux err/spread':>16s} {'h/run':>6s}  failed at step")
    for v, s in S.items():
        rh = s['ratio_h']
        rng = f"{med(rh):.2f} [{min(rh):.2f},{max(rh):.2f}]" if rh else '-'
        print(f"{v:8s} {s['n_ok']}/{s['n']:<3d} {str(s['params']):>7s} {rng:>24s} {100 * med(s['win_h']):5.0f}% "
              f"{med(s['ratio_full']):11.2f} {med(s['n_div_full']):9.0f} {med(s['cons_shift']):16.2f} "
              f"{med(s['sec']) / 3600:6.1f}  {s['failed_at'] or ''}")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--variants', default='base,a6,b3,relu,gdt')
    p.add_argument('--seeds', default='0')
    p.add_argument('--schedule', default=SCHEDULE)
    p.add_argument('--out', default='ablation')
    p.add_argument('--data_dir', default='generated_nfv')
    p.add_argument('--n_train', type=int, default=2048)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--n_eval', type=int, default=100)
    p.add_argument('--eval_nx', type=int, default=200)
    p.add_argument('--eval_nt', type=int, default=400)
    p.add_argument('--eval_every', type=int, default=5000)
    p.add_argument('--device', default='auto')
    a = p.parse_args(argv)
    runs = [train_eval(v, int(s), a) for v in a.variants.split(',') for s in a.seeds.split(',')]
    S = summarize(runs)
    show(S, next((r['H'] for r in runs if 'H' in r), '?'))
    json.dump(S, open(os.path.join(a.out, 'summary.json'), 'w'), indent=1, default=float)
    return S


if __name__ == '__main__':
    main()