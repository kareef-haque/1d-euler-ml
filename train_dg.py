'''
Train the learned DG flux (euler1d/dg/flux.py) on exact Riemann data through the differentiable DG solver.

python train_dg.py --p 1 --mode scalar --out runs/dg_p1_scalar
schedule: steps:lr:nx:nt per stage (rollout length grows). Loss: MSE of all DG modes scaled by QS over the rollout.
Eval: scaled MSE of the cell means on the eval split; baselines DG-HLLC, DG-Rusanov (same p), FV HLLC-1.
'''
import argparse
import json
import math
import os
import time
import numpy as np
import torch

from euler1d.dg import data as dd
from euler1d.dg.flux import DissFlux
from euler1d.neural import scaling

SCHEDULE = '3000:1e-3:20:10,3000:5e-4:50:25,3000:2e-4:100:50'


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--p', type=int, default=1)
    ap.add_argument('--mode', default='scalar', choices=['scalar', 'char'])
    ap.add_argument('--hidden', type=int, default=16)
    ap.add_argument('--depth', type=int, default=4)
    ap.add_argument('--s_min', type=float, default=0.0, help='lower bound of the dissipation scale (1 = Rusanov)')
    ap.add_argument('--schedule', default=SCHEDULE)
    ap.add_argument('--n_train', type=int, default=512)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--n_eval', type=int, default=64)
    ap.add_argument('--eval_nx', type=int, default=200)
    ap.add_argument('--eval_nt', type=int, default=0, help='0: 60 steps x 0.3 / cfl (same physical time for every p)')
    ap.add_argument('--eval_every', type=int, default=500)
    ap.add_argument('--dx', type=float, default=1e-3)
    ap.add_argument('--data_dir', default='generated_dg')
    ap.add_argument('--seed', type=int, default=0, help='model init and batch order')
    ap.add_argument('--data_seed', type=int, default=0, help='datasets (shared by all runs)')
    ap.add_argument('--max_restarts', type=int, default=3)
    ap.add_argument('--out', default=None)
    ap.add_argument('--device', default='auto')
    a = ap.parse_args(argv)
    a.eval_nt = a.eval_nt or round(60 * 0.3 / dd.default_cfl(a.p))
    return a


def stages(s):
    return [(int(a), float(b), int(c), int(d)) for a, b, c, d in (x.split(':') for x in s.split(','))]


def to(td, dev):
    return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in td.items()}


class Restart(Exception):
    pass


def attempt(a, dev, out, k, etd, base):
    torch.manual_seed(a.seed + 1000 * k)
    m = DissFlux(a.p, a.mode, a.hidden, a.depth, a.s_min).to(dev)
    qs = torch.as_tensor(scaling.QS, dtype=torch.float64, device=dev).view(1, 1, 3, 1, 1)
    log = open(os.path.join(out, 'log.jsonl'), 'a')
    step, best, t0 = 0, float('inf'), time.time()
    for si, (steps, lr, nx, nt) in enumerate(stages(a.schedule)):
        td = to(dd.tensors(dd.get_set(a.data_dir, 'train', a.n_train, nx, nt, a.dx, a.p, a.data_seed)), dev)
        opt = torch.optim.Adam(m.parameters(), lr=lr)
        for it in range(steps):
            idx = torch.randperm(a.n_train, device=dev)[:a.batch]
            b = {key: (v[idx] if torch.is_tensor(v) and v.dim() > 0 else v) for key, v in td.items()}
            P = dd.rollout(b, m)
            loss = ((P[:, 1:] - b['Q'][:, 1:]) / qs).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0).item()
            if not (math.isfinite(loss.item()) and math.isfinite(gn)):
                raise Restart(f'attempt {k} stage {si} it {it}: loss {loss.item()} gradnorm {gn}')
            opt.step()
            step += 1
            if it % a.eval_every == 0 or it == steps - 1:
                with torch.no_grad():
                    e = dd.mean_err(dd.rollout(etd, m), etd)
                ok = torch.isfinite(e)
                err = e[ok].mean().item() if ok.any() else float('inf')
                with torch.no_grad():
                    s = m.scale(torch.cat([etd['Gh'][:, 0, :, :1], etd['Q'][:, 0], etd['Gh'][:, 0, :, 1:]], 2))
                rec = dict(attempt=k, step=step, stage=si, nx=nx, nt=nt, loss=loss.item(), gn=gn, err=err,
                           ratio_hllc=base['hllc'] / err, ratio_fv=base['fv'] / err,
                           win_hllc=(e < base['e_hllc']).double().mean().item(), bad=int((~ok).sum()),
                           s_mean=s.mean().item(), sec=round(time.time() - t0, 1))
                log.write(json.dumps(rec) + '\n')
                log.flush()
                print(f"[{si}] step {step:5d} nt {nt:3d} loss {loss.item():.3e} | eval err {err:.3e} vs DG-HLLC x{rec['ratio_hllc']:.2f} "
                      f"(wins {100 * rec['win_hllc']:.0f}%) vs FV x{rec['ratio_fv']:.2f} bad {rec['bad']} s_mean {rec['s_mean']:.2f} ({rec['sec']}s)")
                m.cpu().save(os.path.join(out, f'model_{step}.pt'), step=step)
                if err < best:
                    best = err
                    m.save(os.path.join(out, 'model_best.pt'), step=step)
                m.to(dev)
    m.cpu().save(os.path.join(out, 'model_final.pt'), step=step)
    return m


def main(argv=None):
    a = parse(argv)
    dev = torch.device('cuda' if a.device == 'auto' and torch.cuda.is_available() else
                       ('cpu' if a.device == 'auto' else a.device))
    out = a.out or os.path.join('runs', time.strftime('%Y%m%d_%H%M%S') + f'_dg_p{a.p}_{a.mode}')
    os.makedirs(out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(out, 'config.json'), 'w'), indent=1)
    ed = dd.get_set(a.data_dir, 'eval', a.n_eval, a.eval_nx, a.eval_nt, a.dx, a.p, a.data_seed)
    etd = to(dd.tensors(ed), dev)
    with torch.no_grad():
        e_h = dd.mean_err(dd.rollout(etd, 'hllc'), etd)
        e_fv = dd.mean_err(dd.fv_first_order(ed)[..., None].to(dev), etd)
    base = dict(hllc=e_h.mean().item(), fv=e_fv.mean().item(), e_hllc=e_h)
    print(f'device {dev}, out {out}; eval {a.n_eval} problems x {a.eval_nt} steps at nx {a.eval_nx}: '
          f'DG-P{a.p} HLLC err {base["hllc"]:.3e}, FV HLLC-1 err {base["fv"]:.3e}')
    for k in range(a.max_restarts + 1):
        try:
            attempt(a, dev, out, k, etd, base)
            return out
        except Restart as r:
            print('RESTART:', r)
    raise RuntimeError(f'training failed after {a.max_restarts + 1} attempts')


if __name__ == '__main__':
    main()
