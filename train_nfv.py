'''
Supervised NFV training, as in the paper:
- autoregressive rollout on exact Riemann solutions, curriculum in (nx, nt)
- MSE on Q / QS, Adam, grad-norm clip, restart on non-finite loss or dead gradients
- one optimizer step per "step" of the schedule

--schedule steps:lr:nx:nt,...   (default: the paper's four stages)
'''
import argparse
import copy
import json
import os
import time
import numpy as np
import torch

from infrastructure.nfv_dataset import make_set, save_set, load_set, fname
from euler1d.neural import scaling, stepper
from euler1d.neural import evaluate as ev
from euler1d.neural.model import NFVModel
from euler1d.neural.diagnostics import consistency_error

SCHEDULE = '10000:1e-4:10:10,20000:1e-5:50:50,20000:5e-6:100:100,20000:1e-6:200:200'


class Restart(Exception):
    pass


def bad(gn, loss):
    return (not np.isfinite(gn)) or gn < 1e-9 or (not np.isfinite(loss))


def parse_schedule(s):
    return [(int(a), float(b), int(c), int(d)) for a, b, c, d in (x.split(':') for x in s.split(','))]


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--a', type=int, default=2)
    p.add_argument('--b', type=int, default=1)
    p.add_argument('--act', default='ELU', choices=['ELU', 'ReLU', 'Tanh'])
    p.add_argument('--depth', type=int, default=6)
    p.add_argument('--hidden', type=int, default=15)
    p.add_argument('--schedule', default=SCHEDULE)
    p.add_argument('--n_train', type=int, default=2048)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--loss_coef', type=float, default=1.0)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--dx', type=float, default=1e-3)
    p.add_argument('--cfl', type=float, default=0.5)
    p.add_argument('--dt_mode', default='per_sample', choices=['per_sample', 'global'], help='training data only')
    p.add_argument('--n_eval', type=int, default=100)
    p.add_argument('--eval_nx', type=int, default=200)
    p.add_argument('--eval_nt', type=int, default=400)
    p.add_argument('--eval_every', type=int, default=1000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--data_seed', type=int, default=0)
    p.add_argument('--max_restarts', type=int, default=5)
    p.add_argument('--data_dir', default='generated_nfv')
    p.add_argument('--out', default=None)
    p.add_argument('--device', default='auto')
    return p.parse_args(argv)


def get_set(args, split, n, nx, nt, dt_mode='per_sample'):
    G = max(3, args.a // 2)
    path = os.path.join(args.data_dir, fname(split, nx, nt, dt_mode))
    if os.path.exists(path):
        d = load_set(path)
        if (d['Q'].shape[0] < n or d['G'] < args.a // 2 or abs(d['dx'] - args.dx) > 1e-15
                or abs(d['cfl'] - args.cfl) > 1e-12 or d['seed'] != args.data_seed
                or d.get('dt_mode', 'per_sample') != dt_mode):
            raise ValueError(f'{path} does not match the requested settings (n, a, dx, cfl, data_seed, dt_mode); '
                             f'delete it or use another --data_dir')
        return d
    os.makedirs(args.data_dir, exist_ok=True)
    d = make_set(n, nx, nt, args.dx, split, args.data_seed, args.cfl, G=G, dt_mode=dt_mode)
    save_set(path, d)
    return d


def attempt(args, dev, out, k, ed, Pbase):
    torch.manual_seed(args.seed + k)
    model = NFVModel(args.a, args.b, args.hidden, args.depth, args.act).to(dev)
    qs = torch.as_tensor(scaling.QS, dtype=torch.float32, device=dev).view(1, 1, 3, 1)
    sched = parse_schedule(args.schedule)
    step, best, t0 = 0, float('inf'), time.time()
    logf = open(os.path.join(out, 'log.jsonl'), 'a')

    def save(name, **kw):
        model.cpu().save(os.path.join(out, name), gamma=1.4, cfl=args.cfl, dx=args.dx, step=step, **kw)
        model.to(dev)

    for si, (steps, lr, nx, nt) in enumerate(sched):
        d = get_set(args, 'train', args.n_train, nx, nt, args.dt_mode)
        n = min(args.n_train, d['Q'].shape[0])
        Q = torch.from_numpy(d['Q'][:n]).to(dev, torch.float32)
        dt = torch.from_numpy(d['dt'][:n]).to(dev, torch.float32)
        G, dx = int(d['G']), float(d['dx'])
        tgt = Q[..., G:Q.shape[-1] - G]
        bs = min(args.batch_size or n, n)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for it in range(steps):
            idx = torch.randperm(n, device=dev)[:bs]
            P = stepper.rollout(model, Q[idx], G, dt[idx], dx, 'exact')
            loss = args.loss_coef * ((P[:, args.b:] - tgt[idx][:, args.b:]) / qs).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf')).item()
            if bad(gn, loss.item()):
                raise Restart(f'attempt {k}: stage {si} step {it}: loss={loss.item():.3e} gradnorm={gn:.3e}')
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            step += 1
            if it % args.eval_every == 0 or it == steps - 1:
                r = ev.evaluate(model, ed, Pbase, nt)
                c = consistency_error(copy.deepcopy(model).cpu())
                rec = dict(attempt=k, step=step, stage=si, nx=nx, nt=nt, lr=lr, loss=loss.item(), gradnorm=gn,
                           cons={k: v.tolist() for k, v in c.items()},
                           sec=round(time.time() - t0, 1), **r)
                logf.write(json.dumps(rec) + '\n')
                logf.flush()
                print(f"[{si}] step {step:6d} loss={rec['loss']:.3e} gn={gn:.2e} | L2={r['l2']:.3e} "
                      f"(base {r['base_l2']:.3e}) win={100 * r['win']:.0f}% div={r['n_div']} | "
                      f"h={nt}: L2={r['l2h']:.3e} (base {r['base_l2h']:.3e}) win={100 * r['winh']:.0f}%")
                save(f'model_{step}.pt', stage=si)
                if r['l2'] < best:
                    best = r['l2']
                    save('model_best.pt', stage=si)
    save('model_final.pt', stage=len(sched) - 1)
    logf.close()
    return model


def main(argv=None):
    args = parse(argv)
    if any(nt <= args.b for _, _, _, nt in parse_schedule(args.schedule)):
        raise ValueError(f'every stage needs nt > b (b = {args.b})')
    dev = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available()
                       else ('cpu' if args.device == 'auto' else args.device))
    out = args.out or os.path.join('runs', time.strftime('%Y%m%d_%H%M%S') + f'_a{args.a}b{args.b}_{args.act}')
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'config.json'), 'w') as f:
        json.dump(dict(vars(args), rho_ref=scaling.RHO_REF, a_ref=scaling.A_REF), f, indent=2)
    print(f'device {dev}, output {out}')

    d = get_set(args, 'eval', args.n_eval, args.eval_nx, args.eval_nt)
    d = {k: (v[:args.n_eval] if k in ('Q', 'dt', 'smax', 'pl', 'pr', 'xf') else v) for k, v in d.items()}
    ed, Pbase = ev.prep(d, dev), ev.base_pred(d, dev)

    for k in range(args.max_restarts + 1):
        try:
            attempt(args, dev, out, k, ed, Pbase)
            return out
        except Restart as e:
            print('RESTART:', e)
    raise RuntimeError(f'training failed after {args.max_restarts + 1} attempts')


if __name__ == '__main__':
    main()