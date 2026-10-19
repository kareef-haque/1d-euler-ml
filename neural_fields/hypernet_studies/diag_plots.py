"""
diag_plots.py -- the figures of one experiment (EXP2 = "Init. Cond.", EXP3 = "Time"), drawn from the results of diagnostics.run.
============================================================================================================================

    plot_all(results, out_dir)    results = {sample: diagnostics.run(...)} of ONE experiment  ->  the figures below in out_dir
    save_results / load_results   pickle the results so the figures can be redrawn without loading any model

    A1_error_vs_condition.png          error of baseline / FiLM / LoRA along the condition, and the numerical solver's own error
    B2_modulation_dimensionality.png   how many principal components the modulation path needs (cumulative variance, PR, rank)
    C1_lora_subspace_alignment.png     overlap of the LoRA update with the base weights' dominant subspace, per layer, along the condition
    C2_lora_update_rank.png            how many directions the LoRA update really uses, and how large it is
    D1_film_gating.png                 mean |gamma - 1| per layer along the condition
    D4_film_representation_drift.png   how far each layer's activations have moved from the reference condition (log scale)
    F3_coupling_metrics.png            correlation / log-log slope of distances along condition -> modulation -> weights -> function
    snapshots/<sample>.png             exact vs predicted solutions at three conditions

Curves are aggregated over the samples (mean +- std over the samples; the two seeds of a kind are averaged within a sample first, so n =
number of samples). The two experiments are never combined: call plot_all once per experiment.
"""

import os
import pickle
import traceback
import warnings
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d

for _m in ("Mean of empty slice", "Degrees of freedom <= 0 for slice", "All-NaN slice encountered"):
    warnings.filterwarnings("ignore", message=_m)

KINDS = ("film", "lora")
COLOR = {"film": "tab:blue", "lora": "tab:orange", "baseline": "black"}
LABEL = {"film": "FiLM", "lora": "LoRA", "baseline": "baseline"}
EXP_NAME = {"exp2": "Init. Cond.", "exp3": "Time"}                                   # how the experiments are named on the plots
FIELD_LABELS = ("Density (kg/m^3)", "Velocity (m/s)", "Pressure (Pa)")


# ==============================================================================================
# HELPERS
# ==============================================================================================

def mean_std_ci(A):
    """Mean and sample std (ddof = 1) over axis 0 (the samples) -> (mean, std, n); std is 0 for a single sample."""
    A = np.asarray(A, float)
    n = np.sum(np.isfinite(A), axis=0)
    sd = np.nanstd(A, axis=0, ddof=1) if A.shape[0] > 1 else np.zeros(A.shape[1:])

    return np.nanmean(A, axis=0), sd, n


def save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    has_title = bool(fig.get_suptitle()) if hasattr(fig, "get_suptitle") else getattr(fig, "_suptitle", None) is not None
    fig.tight_layout(rect=(0, 0, 1, 0.96) if has_title else None)                  # leave room for the figure title
    path = os.path.join(out_dir, name + ".png")
    fig.savefig(path, dpi=120)
    plt.close(fig)

    return path


def dial_from_meta(meta):
    """Rebuild a Dial-like object from run()'s results['meta']['dial'] (so figures need no Family)."""
    d = meta["dial"]

    return SimpleNamespace(fn=np.exp if d["fn"] == "exp" else (lambda v: np.asarray(v, float)), name=d["name"], log=d["log"], lo=d["lo"], hi=d["hi"],
                           lim=d["lim"], train_s=np.asarray(d["train_s"]), marks=d["marks"])


def exp_name(e):
    return {"exp2": "Init. Cond.", "exp3": "Time"}.get(e, e)                 # how the experiments are named on the plots


def x_label(dial):
    return dial.name + (" (log scale)" if dial.log else "")


def dial_axis(ax, dial, ticks=False, note=False):
    """Dial on the x axis; the range outside training is shaded; training conditions as black ticks; dial.marks as dotted lines."""
    lo, hi = dial.fn(np.array(dial.lim))
    a, b = dial.fn(np.array([dial.lo, dial.hi]))
    ax.axvspan(lo, a, color="0.92", zorder=0)
    ax.axvspan(b, hi, color="0.92", zorder=0)
    for v, text in dial.marks:
        ax.axvline(v, color="0.35", ls=":", lw=1, zorder=1)
    if dial.log:
        ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    if ticks:
        ax.plot(dial.fn(dial.train_s), np.zeros(len(dial.train_s)), "|", color="k", ms=8, transform=ax.get_xaxis_transform())
    if note:
        for p, q in ((lo, a), (b, hi)):
            if q - p > 1e-9:
                ax.text(np.sqrt(p * q) if dial.log else 0.5 * (p + q), 0.97, "outside\ntraining range", transform=ax.get_xaxis_transform(),
                        ha="center", va="top", fontsize=7, color="0.4")


def band(ax, x, A, color, label=None, ls="-", log=False, alpha=0.2, lw=2):
    """Mean +- std over axis 0 (the samples) of A (S, K) as a line with a band. log=True: mean and std of log10 (geometric mean, band
    multiplicative); use for errors."""
    A = np.asarray(A, float)
    if log:
        A = np.log10(np.maximum(A, 1e-12))
    m, sd, _ = mean_std_ci(A)
    f = (lambda v: 10 ** v) if log else (lambda v: v)
    ax.plot(x, f(m), color=color, ls=ls, lw=lw, label=label)
    if A.shape[0] > 1:
        ax.fill_between(x, f(m - sd), f(m + sd), color=color, alpha=alpha, lw=0)


def grouped_bars(ax, labels, series, colors, hline=None, ylabel=None):
    """Mean +- std over samples as bars, one group per label, one bar per series; black dots = the individual samples.
    series: {name: (S, n_labels)}."""
    n, xs = len(series), np.arange(len(labels))
    w = 0.8 / max(n, 1)
    for i, (name, A) in enumerate(series.items()):
        A = np.asarray(A, float)
        m, sd, _ = mean_std_ci(A)
        pos = xs + (i - (n - 1) / 2) * w
        ax.bar(pos, m, w, yerr=np.nan_to_num(sd), color=colors[name], alpha=0.75, capsize=3, label=name)
        for row in A:
            ax.plot(pos, row, "o", ms=3, color="k", alpha=0.55)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    if hline is not None:
        ax.axhline(hline, color="k", lw=0.8)
    if ylabel:
        ax.set_ylabel(ylabel)


def layer_colors(L):
    return [plt.get_cmap("viridis")(l / max(L - 1, 1)) for l in range(L)]


def layer_legend(ax, L, **kw):
    ax.legend(handles=[plt.Line2D([], [], color=c, lw=2, label=f"layer {l}") for l, c in enumerate(layer_colors(L))], fontsize=7, **kw)


def ntitle(text, agg):
    return f"{text}\n(mean ± std over {agg.n} samples)"


class Aggregate:
    """results: {sample: dict returned by diagnostics.run} for ONE experiment (all samples share the experiment and the dense dial grid)."""

    def __init__(self, results):
        self.res, self.names = results, list(results)
        self.n = len(self.names)
        meta = results[self.names[0]]["meta"]
        for sn in self.names[1:]:
            m = results[sn]["meta"]
            if m["experiment"] != meta["experiment"] or not np.allclose(m["s"], meta["s"]):
                raise ValueError(f"{sn} is not on the same experiment / dial grid as {self.names[0]}: samples of different experiments must not be aggregated")
        self.meta, self.exp = meta, meta["experiment"]
        self.dial = dial_from_meta(meta)
        self.s, self.x = np.asarray(meta["s"]), self.dial.fn(meta["s"])
        self.inside = np.asarray(meta["split"]) == "in"
        self.act_s = None
        if any(r["kind"] == "film" for r in results[self.names[0]]["R"].values()):
            self.act_s = np.asarray(self.any_model("film")["deep"]["film"]["act"]["s"])

    def models(self, sample, kind):
        return [(n, r) for n, r in self.res[sample]["R"].items() if r["kind"] == kind]

    def any_model(self, kind):
        return self.models(self.names[0], kind)[0][1]

    def curve(self, kind, fn):
        """(n_samples, ...): fn(name, Rm, res) evaluated for each model of `kind` in the sample and averaged over its seeds."""
        out = []
        for sn in self.names:
            ms = self.models(sn, kind)
            if not ms:
                raise ValueError(f"{sn} has no {kind} model")
            out.append(np.nanmean(np.stack([np.asarray(fn(n, r, self.res[sn]), float) for n, r in ms]), axis=0))

        return np.stack(out)

    def base(self, fn):
        return np.stack([np.asarray(fn(self.res[sn]), float) for sn in self.names])


# ==============================================================================================
# FIGURES
# ==============================================================================================

def _smooth(A, frac):
    """Moving average along the condition axis (last axis) in log10 space = geometric smoothing; window = frac of the grid points."""
    A = np.asarray(A, float)
    w = int(round(frac * A.shape[-1]))
    if w < 2:
        return A

    return 10 ** uniform_filter1d(np.log10(np.maximum(A, 1e-12)), size=w, axis=-1, mode="nearest")


def plot_A1(agg, out_dir, smooth=0.02):
    """A1: error vs condition for baseline / FiLM / LoRA (geometric mean over samples, band = x/÷ one std of log10 error) and the
    numerical solver's own error. Each sample's curve is smoothed along the condition (window = `smooth` x the number of grid
    points, geometric mean) to remove grid-level jitter; smooth=0 draws the raw curves. Errors below a tenth of the solver's error are clipped."""
    floor = float(np.mean(agg.base(lambda res: res["floor"])))
    prep = lambda A: _smooth(np.maximum(A, floor / 10), smooth)
    fig, ax = plt.subplots(figsize=(10, 6))
    for kind in KINDS:
        band(ax, agg.x, prep(agg.curve(kind, lambda n, Rm, res: Rm["err_d"])), COLOR[kind], LABEL[kind], log=True)
    band(ax, agg.x, prep(agg.base(lambda res: res["base_err"])), COLOR["baseline"], LABEL["baseline"], log=True)
    ax.axhline(floor, color="0.5", ls="--", lw=1, label="numerical solver vs exact")
    ax.set(yscale="log", xlabel=x_label(agg.dial), ylabel="error vs exact (L1)", title=ntitle(f"Error vs condition ({exp_name(agg.exp)})", agg))
    dial_axis(ax, agg.dial, ticks=False, note=True)
    ax.set_ylim(bottom=floor / 3)
    ax.legend(fontsize=8)

    return save(fig, out_dir, "A1_error_vs_condition")


def plot_B2(agg, out_dir):
    """B2 / headline 4: how many dimensions does the modulation path use? Cumulative variance explained by the first k principal components
    (mean +- std over samples), and global participation ratio / effective rank / components needed for 90 and 99 % of the variance."""
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.2), gridspec_kw={"width_ratios": [1.4, 1, 1]})
    for kind in KINDS:
        cum = agg.curve(kind, lambda n, Rm, res: Rm["deep"]["mod"]["pca"]["cum"])
        band(ax[0], np.arange(1, cum.shape[1] + 1), cum, COLOR[kind], LABEL[kind])
    ax[0].axhline(0.99, color="0.6", ls=":", lw=1)
    ax[0].set(xlabel="number of principal components", ylabel="cumulative variance explained", ylim=(None, 1.01), xscale="log",
              title=ntitle(f"Variance explained by the modulation path ({exp_name(agg.exp)})", agg))
    ax[0].legend(fontsize=8)
    cols = {LABEL[k]: COLOR[k] for k in KINDS}
    g = lambda keys: {LABEL[k]: agg.curve(k, lambda n, Rm, res: np.array([Rm["deep"]["mod"]["pca"][q] for q in keys], float)) for k in KINDS}
    grouped_bars(ax[1], ["participation\nratio", "effective\nrank"], g(("pr", "erank")), cols, ylabel="dimensions")
    ax[1].set_title("global dimensionality of the path", fontsize=9); ax[1].legend(fontsize=8)
    grouped_bars(ax[2], ["90 % of\nvariance", "99 % of\nvariance"], g(("n90", "n99")), cols, ylabel="principal components needed")
    ax[2].set_title("components needed", fontsize=9)

    return save(fig, out_dir, "B2_modulation_dimensionality")


def _lora_layers(Rm):
    return Rm["deep"]["lora"]["layers"]


def plot_C1(agg, out_dir, sides=("left",)):
    """C1: principal-angle overlap between the top-k singular subspaces of the base weight W and of the LoRA update dW(s), per layer, along
    the conditioning dial. 1 = dW lies in directions W already uses (reuse), dotted line = what unrelated random subspaces would give.
    sides: ("left",) = output side only (default); ("left", "right") adds the input side as dashed lines. Layers whose top-k subspace is
    the whole space (overlap is 1 by construction) are left out."""
    ref = _lora_layers(agg.any_model("lora"))
    L = len(ref)
    colors = layer_colors(L)
    style = {"left": "-", "right": "--"}
    name = {"left": "output side", "right": "input side"}
    full = lambda side, l: ref[l][f"chance_{side}"] >= 0.99
    fig, ax = plt.subplots(figsize=(8, 5.5))
    handles = []
    for l in range(L):
        for side in sides:
            if full(side, l):
                continue
            A = agg.curve("lora", lambda n, Rm, res, l=l, side=side: _lora_layers(Rm)[l][f"overlap_{side}"])
            band(ax, agg.x, A, colors[l], ls=style[side], alpha=0.12)
            ax.axhline(ref[l][f"chance_{side}"], color=colors[l], ls=":", lw=1, alpha=0.8)
        if any(not full(s, l) for s in sides):
            handles.append(plt.Line2D([], [], color=colors[l], lw=2, label=f"layer {l} (k = {ref[l]['k']})"))
    if len(sides) > 1:
        handles += [plt.Line2D([], [], color="k", ls=style[s], lw=1.5, label=name[s]) for s in sides]
    handles.append(plt.Line2D([], [], color="k", ls=":", lw=1, label="chance (random subspaces)"))
    ax.set(xlabel=x_label(agg.dial), ylabel="subspace overlap with W\n(1 = reuse of W's directions)", ylim=(0, 1.05))
    dial_axis(ax, agg.dial, ticks=False, note=True)
    ax.legend(handles=handles, fontsize=8, loc="upper right", bbox_to_anchor=(1.0, 0.86))
    ax.set_title(ntitle(f"LoRA update vs base-weight subspace along the condition ({exp_name(agg.exp)})", agg), fontsize=11)

    return save(fig, out_dir, "C1_lora_subspace_alignment")


def plot_C2(agg, out_dir):
    """C2: how many of its `rank` directions does the update dW(s) really use? Per layer, averaged over the training range:
    effective rank / participation ratio of the singular values of dW (at most `rank`), the share of energy in the top-1 / top-2 modes,
    and the size of the update ||dW||_F / ||W||_F along the dial."""
    ref = _lora_layers(agg.any_model("lora"))
    L = len(ref)
    ins = agg.inside
    labels = [f"layer {l}" for l in range(L)]
    fig, ax = plt.subplots(1, 3, figsize=(20, 5.3), gridspec_kw={"width_ratios": [1, 1, 1.3]})
    f = lambda key: agg.curve("lora", lambda n, Rm, res: np.array([np.nanmean(l[key][ins]) for l in _lora_layers(Rm)]))
    grouped_bars(ax[0], labels, {"effective rank": f("erank"), "participation ratio": f("pr")}, {"effective rank": "tab:orange", "participation ratio": "tab:brown"}, ylabel="directions in use")
    for l in range(L):
        ax[0].plot([l - 0.45, l + 0.45], [ref[l]["k"]] * 2, "k--", lw=1, label="LoRA rank (max)" if l == 0 else None)
    ax[0].legend(fontsize=7); ax[0].set_title("update rank per layer", fontsize=9)
    top = lambda k: agg.curve("lora", lambda n, Rm, res: np.array([np.nanmean(l["topk"][k][ins]) for l in _lora_layers(Rm)]))
    ks = sorted(ref[1]["topk"]) if L > 1 else sorted(ref[0]["topk"])
    grouped_bars(ax[1], labels, {f"top-{k} mode(s)": top(k) for k in ks[:2]}, {f"top-{k} mode(s)": c for k, c in zip(ks[:2], ("tab:orange", "tab:red"))}, ylabel="share of ||dW||²")
    ax[1].legend(fontsize=7); ax[1].set_title("energy concentration", fontsize=9)
    colors = layer_colors(L)
    for l in range(L):
        band(ax[2], agg.x, agg.curve("lora", lambda n, Rm, res, l=l: _lora_layers(Rm)[l]["rel_norm"]), colors[l], log=True)
    ax[2].set(yscale="log", xlabel=x_label(agg.dial), ylabel="||dW||_F / ||W||_F")
    dial_axis(ax[2], agg.dial, ticks=True, note=True)
    layer_legend(ax[2], L, loc="lower left"); ax[2].set_title("size of the update relative to the base layer", fontsize=9)
    fig.suptitle(ntitle(f"LoRA update effective rank ({exp_name(agg.exp)})", agg), fontsize=11)

    return save(fig, out_dir, "C2_lora_update_rank")


def _film_gating(Rm, key):
    return np.stack([g[key] for g in Rm["deep"]["film"]["gating"]], 1)                  # (K, L)


def plot_D1(agg, out_dir):
    """D1: FiLM gating strength: mean |gamma - 1| per layer along the dial (0 = the channel scaling is untouched)."""
    key, lab = "mean_abs_gamma_dev", "mean |γ − 1|"
    L = _film_gating(agg.any_model("film"), key).shape[1]
    colors = layer_colors(L)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for l in range(L):
        band(ax, agg.x, agg.curve("film", lambda n, Rm, res, l=l: _film_gating(Rm, key)[:, l]), colors[l])
    ax.set(ylabel=lab, xlabel=x_label(agg.dial))
    dial_axis(ax, agg.dial, ticks=True, note=True)
    layer_legend(ax, L, loc="upper center")
    ax.set_title(ntitle(f"FiLM gating strength per layer ({exp_name(agg.exp)})", agg), fontsize=11)

    return save(fig, out_dir, "D1_film_gating")


def plot_D4(agg, out_dir, floor=1e-3):
    """D4: representation drift of the FiLM network: how far the hidden activations of each layer (on fixed probe inputs) have moved from
    their values at the reference condition, ||sigma(x) - sigma(x_ref)|| / ||sigma(x_ref)|| (sigma(x) = the layer's activations). 0 = identical
    to the reference; 1 = a change as large as the activations themselves. Log y axis; values below `floor` (exactly at the reference) are clipped."""
    L = agg.any_model("film")["deep"]["film"]["act"]["drift"].shape[1]
    colors, x = layer_colors(L), agg.dial.fn(agg.act_s)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for l in range(L):
        A = agg.curve("film", lambda n, Rm, res, l=l: Rm["deep"]["film"]["act"]["drift"][:, l])
        band(ax, x, np.maximum(A, floor), colors[l], log=True)
    ax.set_yscale("log")
    ax.set(xlabel=x_label(agg.dial), ylabel="representation drift\n" + r"$\|\sigma(x) - \sigma(x_{ref})\|\,/\,\|\sigma(x_{ref})\|$")
    dial_axis(ax, agg.dial, ticks=False, note=True)
    ax.set_ylim(floor, None)
    layer_legend(ax, L, loc="lower right")
    ax.set_title(ntitle(f"FiLM representation drift from the reference condition ({exp_name(agg.exp)})", agg), fontsize=11)

    return save(fig, out_dir, "D4_film_representation_drift")


REL_LABEL = {"cond->mod": "condition → modulation", "cond->func": "condition → function", "mod->weight": "modulation → weights",
             "weight->func (F1)": "weights → function", "mod->func (F2)": "modulation → function",
             "speed_mod->speed_func": "modulation speed → function speed", "speed_weight->speed_func": "weight speed → function speed"}


def _metric(deep, rel, region, key):
    for r in deep["coupling"]["metrics"]:
        if r["relation"] == rel and r["region"] == region:
            return r[key]
    return np.nan


def plot_F3(agg, out_dir):
    """F3: coupling metrics for every link of the chain condition -> modulation -> weights -> function, over pairs of in-range conditions
    (and along the path for the speeds): Spearman correlation and log-log slope of distance (1 = proportional, < 1 = saturating),
    mean ± std over samples (dots = samples)."""
    rels = list(REL_LABEL)
    fig, ax = plt.subplots(1, 2, figsize=(18, 5.3))
    for a, key, lab in zip(ax, ("spearman", "slope"), ("Spearman correlation", "log-log slope")):
        series = {LABEL[k]: agg.curve(k, lambda n, Rm, res: np.array([_metric(Rm["deep"], r, "in", key) for r in rels])) for k in KINDS}
        grouped_bars(a, [REL_LABEL[r].replace(" → ", "\n→ ") for r in rels], series, {LABEL[k]: COLOR[k] for k in KINDS}, hline=0 if key == "spearman" else 1, ylabel=lab)
        a.legend(fontsize=8); a.set_title(ntitle(f"{lab} ({exp_name(agg.exp)}, in-range pairs)", agg), fontsize=10)
        a.tick_params(axis="x", labelsize=7)

    return save(fig, out_dir, "F3_coupling_metrics")


def plot_snapshots(res, sample, out_dir):
    """Exact (black dashed) vs predicted solutions for 3 conditions (columns) and the three fields (rows): the baseline and the first FiLM and
    LoRA model of this sample. Axis limits follow the exact solution, so a wildly wrong prediction runs off the plot."""
    sn = res["snap"]
    fig, ax = plt.subplots(3, 3, figsize=(16, 10), sharex=True)
    for j in range(3):
        for f in range(3):
            a = ax[f, j]
            a.plot(sn["x"], sn["exact"][j, :, f], "k--", lw=2, label="exact")
            a.plot(sn["x"], sn["baseline"][j, :, f], "k", lw=1, label="baseline")
            for n, p in sn["preds"].items():
                a.plot(sn["x"], p[j, :, f], color=COLOR[res["R"][n]["kind"]], lw=1.2, label=n)
            lo, hi = sn["exact"][j, :, f].min(), sn["exact"][j, :, f].max()
            pad = 0.15 * (hi - lo + 1e-9)
            a.set_ylim(lo - pad, hi + pad)
            if j == 0:
                a.set_ylabel(FIELD_LABELS[f])
        ax[0, j].set_title(sn["titles"][j], fontsize=10)
        ax[2, j].set_xlabel("x (m)")
    ax[0, 0].legend(fontsize=7)
    fig.suptitle(f"{sample}: exact vs predicted solutions ({exp_name(res['meta']['experiment'])})", fontsize=11)

    return save(fig, out_dir, sample)


FIGURES = (plot_A1, plot_B2, plot_C1, plot_C2, plot_D1, plot_D4, plot_F3)


def plot_all(results, out_dir, strict=False):
    """Draw every figure for ONE experiment (results = {sample: diagnostics.run(...)}) into out_dir; snapshots go to out_dir/snapshots/.
    A figure that fails is reported and skipped (strict=True re-raises), so one problem does not lose the others."""
    agg = Aggregate(results)
    done = []
    for fn in FIGURES:
        try:
            done.append(fn(agg, out_dir))
        except Exception as e:
            if strict:
                raise
            print(f"[figure skipped] {fn.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
    for sn, res in results.items():
        if res.get("snap") is not None:
            done.append(plot_snapshots(res, sn, os.path.join(out_dir, "snapshots")))
    print(f"[plots] {exp_name(agg.exp)}: {len(done)} figures in {out_dir}")

    return done


def save_results(results, path):
    """Pickle the results ({sample: run() output}) so plot_all can be re-run without the models."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(results, fh, protocol=4)


def load_results(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)