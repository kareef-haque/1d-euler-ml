"""
diagnostics.py -- how does a hypernetwork-modulated INR respond to a dial, compared with a plain INR (the baseline)?
=================================================================================================================

EXP2 ("Init. Cond."): dial = pressure ratio p_R / p_L (trained over 0.1-10; internally s = ln p). Base INR (x, t) frozen and shared;
      the hypernetwork takes the raw (rho_ratio, p_ratio, x_split). Baseline = plain INR on (x, t, rho_ratio, p_ratio, x_split).
EXP3 ("Time"): dial = time t / t_max of the single Sod run. Hypernetwork takes t, base takes x, trained jointly. Baseline = plain INR on (x, t).
"Modulation" (m_eff) is what the base INR actually receives, per layer:
    FiLM: [gamma - 1 | beta]   a per-neuron rescale and shift of a hidden layer
    LoRA: dW = lora_scale * B @ A   a low-rank change added to a layer's weight matrix

THIS FILE computes; diag_plots.py draws. Blocks:
    1. settings, Dial, error measure
    2. models:       Model (base INR + hypernetwork), Baseline                          (torch)
    3. experiments:  exact solver, Grid / Family, exp2_family, exp3_family              (what each experiment feeds the models)
    4. analysis:     numpy-only measurements behind the figures (one model at a time)
    5. run:          evaluate every model of one sample -> a plain dict (picklable; diag_plots.save_results)

WHAT run() MEASURES (and the figure that uses it)
    error vs condition            normalised L1 per dense condition for every model, the baseline, the numerical-solver floor      A1
    modulation path PCA           cumulative variance explained, participation ratio, effective rank, components for 90 / 99 %     B2
    LoRA                          per layer: principal-angle overlap of dW with W (vs chance), update effective rank, energy, size  C1, C2
    FiLM                          per layer: mean |gamma - 1| along the dial; activation drift from a reference condition            D1, D4
    coupling                      distances in condition / modulation / weight / function space and how they correlate             F3
    snapshots                     exact vs predicted solutions at three conditions                                                  snapshots

Typical use (see run_samples.py):
    fam = exp2_family("generated_data", {"train": [-1, *range(21)], "validate": range(5), "test": range(5)})
    res = run(fam, models, baseline)          # models {name: Model}, kept on the CPU; one sample
    diag_plots.plot_all({"sample_1": res, ...}, "diagnostics/exp2")
"""

import os

# Windows: PyTorch and NumPy/Matplotlib can each load their own OpenMP runtime ("OMP: Error #15").
# Must be set BEFORE torch/numpy are imported (so also first in your run script if it imports torch first).
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
import yaml
from scipy import stats
from scipy.special import erf

from infrastructure.results import ExactRiemannSolver

for _m in ("Mean of empty slice", "Degrees of freedom <= 0 for slice", "All-NaN slice encountered"):          # NaN gaps are intentional
    warnings.filterwarnings("ignore", message=_m)


# ==============================================================================================
# 1. SETTINGS, DIAL, ERROR
# ==============================================================================================

RHO_L, U_L, P_L = 1.0, 0.0, 1.0e5                        # fixed left state of every Riemann problem
SOD_RHO_RATIO, SOD_P_RATIO, SOD_X_SPLIT = 0.125, 0.1, 0.5
DOMAIN, GAMMA = 1.0, 1.4
N_CELL, T_MAX = 1000, 2.0e-4                             # solver cells; physical time of the last snapshot (normalised t = 1)

N_DENSE, T_DENSE = 300, 10                               # dense dial grid (exact solver); EXP2 snapshots per dense condition
EXTEND, EXTRAP_MAX = 0.25, 1.5                           # EXP2: training range +-25 % in ln p; EXP3: dense grid runs to 1.5 t_max
PTS_PER_CHUNK = 200_000                                  # render batch size in (s, t, x) points, keeps memory flat
SHOCK_MIN_JUMP = 0.1                                     # a shock exists only if its pressure jump is >= 10 % (none near p_ratio = 1)

PHYSICS_YAML = r".\neural_fields\hypernet_studies\configs\physics.yaml"


_REF = None                                              # (rho, u, P) normalisation constants, read lazily


def to_physical(out):
    """Raw INR output (value / reference) -> physical (rho, u, P) = value * reference."""
    global _REF
    if _REF is None:
        with open(PHYSICS_YAML) as fh:
            norm = yaml.safe_load(fh)["normalize"]
        _REF = torch.tensor([norm["rho"], norm["u"], norm["P"]], dtype=torch.float32)

    return out * _REF.to(out.device)


@dataclass
class Dial:
    """How the dial s is shown on every x axis. fn: s -> shown number (EXP2: exp, i.e. the pressure ratio, on a log axis);
    lo / hi: the training range in s; lim: (min, max) s of the plotted grid; train_s: the training values;
    marks: [(shown value, text)] drawn as dotted vertical lines."""
    fn: Callable
    name: str
    log: bool
    lo: float
    hi: float
    lim: tuple
    train_s: np.ndarray
    marks: list = field(default_factory=list)


def _norm(ref, pooled):
    """Per-condition, per-field normaliser (K,1,1,3): std of the EXACT solution of that condition over (t, x), floored at 5 % of the
    pooled std. Each condition is judged relative to its OWN size, so strong-shock conditions do not dominate."""
    return np.maximum(ref.std((1, 2)), 0.05 * pooled)[:, None, None, :]


def rel_error(pred, truth, ref, pooled):
    """Normalised L1 error per condition: mean over (t, x, fields) of |pred - truth| / _norm(ref). Arrays are (K, T, X, 3) -> (K,)."""
    return (np.abs(pred - truth) / _norm(ref, pooled)).mean((1, 2, 3))


def ext_fraction(s, dial, ext_width):
    """Distance beyond the training range as a fraction of the extension (0 inside the range, 1 at the far end)."""
    return np.where(s < dial.lo, dial.lo - s, np.where(s > dial.hi, s - dial.hi, 0.0)) / ext_width


# ==============================================================================================
# 2. MODELS (torch)
# ==============================================================================================

class _OnDevice:
    """Networks stay on the CPU until analysed: run() calls load(device) / unload(). Subclasses list their networks in `nets`."""

    @property
    def device(self):
        return next(self.nets[0].parameters()).device

    def load(self, device):
        for n in self.nets:
            n.to(device)
        return self

    def unload(self):
        for n in self.nets:
            n.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@dataclass
class Model(_OnDevice):
    """One base INR + hypernetwork. kind = 'film' | 'lora'. base_trained: False for EXP2 (frozen base: only the hypernetwork counts as trained),
    True for EXP3 (trained jointly)."""
    name: str
    inr: torch.nn.Module
    hyper: torch.nn.Module
    kind: str
    to_physical: Callable = lambda out: out
    base_trained: bool = False

    def __post_init__(self):
        assert self.kind in ("film", "lora")
        for net in self.nets:
            net.eval()
            for p in net.parameters():
                p.requires_grad_(False)

    @property
    def nets(self):
        return (self.inr, self.hyper)

    @property
    def n_trained(self):
        n = sum(p.numel() for p in self.hyper.parameters())
        return n + sum(p.numel() for p in self.inr.parameters()) if self.base_trained else n

    def ref(self):
        """Per-layer reference size for reading a modulation: ||W_l||_F for LoRA (dW becomes a RELATIVE update), 1 for FiLM."""
        return [1.0] * (len(self.inr.layers) - 1) if self.kind == "film" else [float(l.linear.weight.norm()) for l in self.inr.layers]

    @torch.no_grad()
    def m_eff(self, cond, chunk=256):
        """Hypernetwork -> effective modulation: list over layers of numpy arrays (K, ...).
        FiLM: (K, 2F) = [gamma-1 | beta].   LoRA: (K, out, in) = lora_scale * B @ A."""
        outs = []
        for i in range(0, len(cond), chunk):
            mod = self.hyper(cond[i:i + chunk].to(self.device))
            if self.kind == "film":
                outs.append([torch.cat([g - 1.0, b], -1).cpu().numpy() for g, b in mod])
            else:
                outs.append([(self.inr.lora_scale * torch.matmul(B, A)).cpu().numpy() for A, B in mod])

        return [np.concatenate(layer) for layer in zip(*outs)]

    @torch.no_grad()
    def raw_mod(self, cond, chunk=256):
        """LoRA only: what the hypernetwork actually emits, the factors A (r, in) and B (out, r) of every layer, flattened and concatenated
        -> (K, D) numpy. (m_eff keeps only the product B @ A; the factor space is needed for the modulation-vs-function coupling.)"""
        assert self.kind == "lora", "raw_mod is for LoRA; FiLM's raw output is m_eff"
        outs = []
        for i in range(0, len(cond), chunk):
            mod = self.hyper(cond[i:i + chunk].to(self.device))
            outs.append(torch.cat([torch.cat([A.flatten(1), B.flatten(1)], 1) for A, B in mod], 1).cpu().numpy())

        return np.concatenate(outs)

    @torch.no_grad()
    def render(self, m, grid):
        """Prediction (K,T,X,3), physical units, from an effective modulation m (as returned by m_eff) on grid.coords. Runs the INR layers by
        hand: z = W h + b [+ h dW^T for LoRA] [gamma * z + beta for FiLM, not on the output layer], then the activation (the order of INRLayer,
        but dW is used directly: no A, B, no lora_scale)."""
        K, N = len(m[0]), grid.coords.shape[0]
        coords = grid.coords.to(self.device)
        step = max(1, PTS_PER_CHUNK // N)
        outs = []

        for i in range(0, K, step):
            h = coords.unsqueeze(0).expand(min(step, K - i), -1, -1)
            for l, layer in enumerate(self.inr.layers):
                z = layer.linear(h)
                if l < len(m):                                               # FiLM has no modulation on the output layer
                    ml = torch.as_tensor(m[l][i:i + step]).to(self.device)
                    if self.kind == "lora":
                        z = z + torch.matmul(h, ml.transpose(-1, -2))
                    else:
                        z = (1.0 + ml[:, None, :layer.N_out]) * z + ml[:, None, layer.N_out:]
                h = layer.act(z) if layer.act is not None else z
            outs.append(self.to_physical(h).cpu())

        return torch.cat(outs).numpy().reshape(K, grid.T, -1, 3)


@dataclass
class Baseline(_OnDevice):
    """A plain INR with no modulation; its input is built by fam.baseline_in."""
    name: str
    inr: torch.nn.Module
    to_physical: Callable = lambda out: out

    def __post_init__(self):
        self.inr.eval()
        for p in self.inr.parameters():
            p.requires_grad_(False)

    @property
    def nets(self):
        return (self.inr,)

    @property
    def n_trained(self):
        return sum(p.numel() for p in self.inr.parameters())

    @torch.no_grad()
    def predict(self, inputs, grid):
        """inputs (K, N, N_in) -> (K, T, X, 3) physical."""
        K, N, _ = inputs.shape
        step = max(1, PTS_PER_CHUNK // N)
        outs = [self.to_physical(self.inr(inputs[i:i + step].to(self.device))[0]).cpu() for i in range(0, K, step)]

        return torch.cat(outs).numpy().reshape(K, grid.T, -1, 3)


# ==============================================================================================
# 3. EXPERIMENTS: exact solver, grids, families
# ==============================================================================================

def cons_to_prim(Q):
    """Conservative (..., 3) = (rho, rho*u, E) -> primitive (rho, u, P)."""
    rho, mom, E = Q[..., 0], Q[..., 1], Q[..., 2]
    u = mom / rho

    return np.stack([rho, u, (GAMMA - 1) * (E - 0.5 * rho * u ** 2)], -1)


def exact_solution(ics, times, x):
    """Exact Riemann solution for K initial conditions, each at its own T times. ics (K,3): (rho_ratio, p_ratio, x_split); times (K,T);
    x (X,) cell centres. Returns u (K,T,X,3) and the shock position (K,T), NaN where there is no shock (pressure jump < SHOCK_MIN_JUMP).
    The solver gives P*, ... but not wave positions, so the shock is rebuilt: x0 + S t with S = -/+ a_k sqrt((g+1)/(2g) P*/P_k + (g-1)/(2g))
    (left / right; u = 0 on both sides)."""
    K, T = times.shape
    g = GAMMA
    u = np.zeros((K, T, len(x), 3), np.float32)
    shock_x = np.full((K, T), np.nan)

    for k, (rho_r, p_r, x0) in enumerate(ics):
        sol = ExactRiemannSolver((RHO_L, U_L, P_L), (RHO_L * rho_r, 0.0, P_L * p_r), x_interface=x0 * DOMAIN, gamma=g)
        for j, t in enumerate(times[k]):
            u[k, j] = np.stack(sol.sample(x, t), -1)
        for p_k, a_k, sign in ((sol.P_L, sol.a_L, -1.0), (sol.P_R, sol.a_R, +1.0)):
            if sol.P_star > p_k * (1 + SHOCK_MIN_JUMP):                    # P* well above this side's pressure -> a shock on this side
                shock_x[k] = x0 * DOMAIN + sign * a_k * np.sqrt((g + 1) / (2 * g) * sol.P_star / p_k + (g - 1) / (2 * g)) * times[k]

    return u, shock_x


@dataclass
class Grid:
    """K dial values, each with a solution of T snapshots on X cells (physical units).
    s (K,)  split (K,): 'train' | 'val' | 'test' (stored) or 'in' | 'extrap' (dense) or 'pair'   u (K,T,X,3): the TRUTH of this grid
    u_exact: the exact solution (= u except on the stored grid, where u is numerical)   shock_x (K,T): exact shock position
    coords (N, N_in): base-INR input for ONE s (N = T*X, flat index = it*X + ix)   T: snapshots per s"""
    s: np.ndarray
    split: np.ndarray
    u: np.ndarray
    u_exact: np.ndarray
    coords: torch.Tensor
    T: int
    shock_x: np.ndarray

    def err(self, pred, pooled, rows=slice(None)):
        return rel_error(pred, self.u[rows], self.u_exact[rows], pooled)


def exact_grid(s, ics, times, x, coords):
    """Grid whose truth is the exact solution (split 'pair')."""
    u, sx = exact_solution(ics, times, x)

    return Grid(np.asarray(s, float), np.array(["pair"] * len(s)), u, u, coords, times.shape[1], sx)


@dataclass
class Family:
    """Everything the models and plots need from one experiment (built by exp2_family / exp3_family).
    st: stored numerical runs (train / val / test)   dn: dense grid over the training range + extension, truth = exact solver
    pooled (3,): per-field std of the dense exact family (a floor for the per-condition normaliser)   x: cell centres
    dial: how the dial is shown   train_u: training solutions in the dense snapshot layout (for the nearest-solution control)
    ext_width: length of the extension on one side, in s   snap_s: 3 conditions for the static comparison (one held-out, two extrapolated)
    anim_s: frames of the animation   label(s): kind of condition ('training condition', 'extrapolation', ...)
    cond(s) -> hypernetwork input (K, cond_dim)   baseline_in(s, grid) -> (K, N, N_in)   exact(s) -> Grid on the dense layout"""
    name: str
    x: np.ndarray
    pooled: np.ndarray
    st: Grid
    dn: Grid
    dial: Dial
    train_u: np.ndarray
    ext_width: float
    snap_s: np.ndarray
    anim_s: np.ndarray
    label: Callable
    cond: Callable
    baseline_in: Callable
    exact: Callable


def make_coords(x_in, t_in):
    """(T*X, 2) base-INR inputs (x, t); flat index = it * X + ix."""
    XX, TT = np.meshgrid(x_in, t_in)

    return torch.tensor(np.stack([XX.ravel(), TT.ravel()], -1), dtype=torch.float32)


def exp2_family(data_path, indices, t_stride=40):
    """EXP2: s = ln(p_ratio), shown as the pressure ratio. The hypernetwork receives the RAW triple (rho_ratio, p_ratio, x_split); the base takes
    (x, t) with x = linspace(-1, 1, N_cell), t = t / T_MAX. indices: {"train": [-1, *range(21)], "validate": range(5), "test": range(5)}
    (train index -1 = canonical Sod). Files: {split}_{i}.npy (N_time, 3, N_cell) conservative; {split}_init_{i}.npy rows (Q_L, Q_R, x_split)."""
    ics, split, U = [], [], []
    for sp, ids in indices.items():
        for i in ids:
            Q = np.load(os.path.join(data_path, sp, f"{sp}_{i}.npy"))
            init = np.load(os.path.join(data_path, sp, f"{sp}_init_{i}.npy"))
            QL, QR = cons_to_prim(init[0]), cons_to_prim(init[1])
            ics.append([QR[0] / QL[0], QR[2] / QL[2], init[2][0]])
            U.append(cons_to_prim(np.moveaxis(Q, 1, -1)[::t_stride]).astype(np.float32))
            split.append("val" if sp == "validate" else sp)

    ics, split, U = np.array(ics), np.array(split), np.stack(U)
    assert np.allclose(ics[:, 0], SOD_RHO_RATIO) and np.allclose(ics[:, 2], SOD_X_SPLIT), "data is not the 1-D family (rho_ratio = 0.125, x_split = 0.5 fixed)"
    nt, nx = U.shape[1:3]
    assert nx == N_CELL, f"file has {nx} cells but N_CELL = {N_CELL}"
    t = np.linspace(0, T_MAX, Q.shape[0])[::t_stride]                       # ASSUMPTION: first snapshot t = 0, last = T_MAX
    x = (np.arange(nx) + 0.5) * DOMAIN / nx                                 # physical cell centres
    x_in, t_in = np.linspace(-1, 1, nx), t / T_MAX                          # what the base INR was trained on
    s, train = np.log(ics[:, 1]), split == "train"
    lo, hi = s[train].min(), s[train].max()

    lo_e, hi_e = lo - EXTEND * (hi - lo), hi + EXTEND * (hi - lo)
    if hi <= 0: hi_e = min(hi_e, -0.05)                                     # only if the whole training range is below p_ratio = 1
    if lo >= 0: lo_e = max(lo_e, 0.05)
    s_d = np.linspace(lo_e, hi_e, N_DENSE)
    tsel = np.linspace(1, nt - 1, T_DENSE).round().astype(int)              # dense snapshots = a subset of the stored ones

    ic = lambda s_: np.stack([np.full(len(s_), SOD_RHO_RATIO), np.exp(s_), np.full(len(s_), SOD_X_SPLIT)], -1)
    cond = lambda s_: torch.tensor(ic(np.asarray(s_, float)), dtype=torch.float32)
    coords_st, coords_d = make_coords(x_in, t_in), make_coords(x_in, t_in[tsel])
    exact = lambda s_: exact_grid(s_, ic(np.asarray(s_, float)), np.tile(t[tsel], (len(s_), 1)), x, coords_d)

    def baseline_in(s_, grid):                                              # (x, t, rho_ratio, p_ratio, x_split)
        c = cond(s_)[:, None, :].expand(-1, grid.coords.shape[0], -1)
        return torch.cat([grid.coords[None].expand(len(c), -1, -1), c], -1)

    def label(s_):
        if s_ < lo - 1e-9 or s_ > hi + 1e-9:
            return "extrapolation"
        k = np.abs(s - s_).argmin()
        if abs(s[k] - s_) < 1e-6:
            return {"train": "training condition", "val": "validation condition", "test": "test condition"}[split[k]]
        return "inside the training range (not in the data)"

    ex = exact_grid(s, ics, np.tile(t, (len(s), 1)), x, coords_st)
    st = Grid(s, split, U, ex.u, coords_st, nt, ex.shock_x)
    dn = exact(s_d)
    dn.split = np.where((s_d >= lo) & (s_d <= hi), "in", "extrap")

    val_s = s[split == "val"]
    strong = val_s[np.abs(val_s) >= 0.7]                                    # a held-out condition with a clear shock
    pool = np.sort(strong if len(strong) else val_s)
    snap_s = np.array([pool[len(pool) // 2], 0.5 * (lo + lo_e), 0.5 * (hi + hi_e)])
    anim_s = np.unique(np.concatenate([s, np.linspace(lo_e, lo, 9)[:-1], np.linspace(hi, hi_e, 9)[1:]]))
    dial = Dial(np.exp, "pressure ratio p_R / p_L", True, lo, hi, (lo_e, hi_e), s[train], [(1.0, "p = 1: no shock,\nshock changes side")])

    return Family("exp2", x, dn.u.reshape(-1, 3).std(0).astype(np.float32), st, dn, dial, U[train][:, tsel], EXTEND * (hi - lo),
                  snap_s, anim_s, label, cond, baseline_in, exact)


def exp3_family(sod_path, t_stride=10):
    """EXP3: s = tau = t / t_max of the single Sod run. The hypernetwork receives tau, the base receives x = linspace(-1, 1, N_cell). Splits follow
    your 8 % blocks: val at tau 0.2 and 0.6, test at 0.4 and 0.8, everything else train. The dense grid runs from 0 to 1.5 (beyond 1 =
    extrapolation, scored against the exact solver). T = 1: every s is one snapshot."""
    Q = np.load(sod_path)
    nt_full, _, nx = Q.shape
    assert nx == N_CELL, f"file has {nx} cells but N_CELL = {N_CELL}"
    u_num = cons_to_prim(np.moveaxis(Q, 1, -1)[::t_stride])[:, None].astype(np.float32)        # (K, 1, X, 3)
    tau = np.linspace(0, 1, nt_full)[::t_stride]                            # your preprocessor: t = linspace(0, 1, N_time)
    block = lambda tt, c: (tt >= c - 0.04) & (tt < c + 0.04)
    kind = lambda tt: np.where(block(tt, 0.2) | block(tt, 0.6), "val", np.where(block(tt, 0.4) | block(tt, 0.8), "test", "train"))
    split, train = kind(tau), kind(tau) == "train"

    x = (np.arange(nx) + 0.5) * DOMAIN / nx
    coords = torch.tensor(np.linspace(-1, 1, nx)[:, None], dtype=torch.float32)               # (X, 1)
    ic = np.array([SOD_RHO_RATIO, SOD_P_RATIO, SOD_X_SPLIT])
    exact = lambda s_: exact_grid(s_, np.tile(ic, (len(s_), 1)), (np.asarray(s_, float) * T_MAX)[:, None], x, coords)

    def baseline_in(s_, grid):                                              # (x, t)
        t_col = torch.tensor(np.asarray(s_, np.float32))[:, None, None].expand(-1, len(grid.coords), 1)
        return torch.cat([grid.coords[None].expand(len(t_col), -1, -1), t_col], -1)

    def label(s_):
        return "extrapolation" if s_ > 1 + 1e-9 else {"train": "training time", "val": "validation time", "test": "test time"}[str(kind(np.array([s_]))[0])]

    ex = exact(tau)
    st = Grid(tau, split, u_num, ex.u, coords, 1, ex.shock_x)
    s_d = np.linspace(0, EXTRAP_MAX, N_DENSE)
    dn = exact(s_d)
    dn.split = np.where(s_d <= 1 + 1e-9, "in", "extrap")
    dial = Dial(lambda s_: np.asarray(s_, float), "time t / t_max", False, 0.0, 1.0, (0.0, EXTRAP_MAX), tau[train])

    return Family("exp3", x, dn.u.reshape(-1, 3).std(0).astype(np.float32), st, dn, dial, u_num[train], EXTRAP_MAX - 1.0,
                  np.array([0.6, 1.25, 1.5]), np.linspace(0, EXTRAP_MAX, 151), label,
                  lambda s_: torch.tensor(np.asarray(s_, np.float32))[:, None], baseline_in, exact)


# ==============================================================================================
# 4. ANALYSIS (numpy only): the measurements behind the figures
# ==============================================================================================

SETTINGS = dict(
    n_pair=100,          # conditions used for the pairwise distances of the coupling analysis (evenly spaced out of the dense grid)
    n_probe=512,         # probe coordinates for the FiLM activation drift
    n_act_cond=60,       # conditions at which activations are evaluated
    n_comp=20,           # principal components kept for the variance-explained curve
)
REF_EXP2_P_RATIO = SOD_P_RATIO   # the reference condition of EXP2 = the canonical Sod problem (a training condition at the lower edge of the range)


def effective_rank(sv):
    """Roy-Vetterli effective rank exp(H(p)), p = s / sum(s), of singular values sv. 1 for a pure line; 0 for an all-zero spectrum."""
    s = np.asarray(sv, float)
    s = s[s > 0]
    if len(s) == 0:
        return 0.0
    p = s / s.sum()

    return float(np.exp(-(p * np.log(p)).sum()))


def participation_ratio(sv):
    """(sum s^2)^2 / sum s^4: the number of directions that effectively carry the variance. 0 for an all-zero spectrum."""
    lam = np.asarray(sv, float) ** 2
    den = (lam ** 2).sum()

    return float(lam.sum() ** 2 / den) if den > 0 else 0.0


def _cut(a, n):
    """First n entries of a 1-D array, padded with NaN."""
    a = np.asarray(a, float)

    return np.concatenate([a[:n], np.full(max(0, n - len(a)), np.nan)])


def pca_summary(X, fit_mask, n_comp=20):
    """PCA of a path X (K, D) (one row per condition), fitted on the rows in fit_mask (the training range), via the small Gram matrix.
    The mean is removed; no per-column scaling. Returns cum (cumulative variance explained, n_comp entries), participation ratio and
    effective rank of the spectrum, and the number of components needed for 90 / 99 % of the variance."""
    X = np.asarray(X, float)
    Xf = X[np.asarray(fit_mask, bool)]
    Xf = Xf - Xf.mean(0)
    lam = np.clip(np.linalg.eigvalsh(Xf @ Xf.T)[::-1], 0, None)
    sv, tot = np.sqrt(lam), lam.sum()
    keep = sv > sv[0] * 1e-6 if sv[0] > 0 else np.zeros(len(sv), bool)          # Gram eigenvalue noise is ~1e-8 of the top singular value
    cum = np.cumsum(lam) / tot if tot > 0 else np.zeros_like(lam)
    first = lambda q: int(np.searchsorted(cum, q - 1e-12) + 1) if tot > 0 else 0

    return {"cum": _cut(cum, n_comp), "pr": participation_ratio(sv[keep]), "erank": effective_rank(sv[keep]), "n90": first(0.90), "n99": first(0.99)}


def derivative(X, s):
    """dX / ds along axis 0 by finite differences (s strictly increasing, not necessarily uniform)."""
    return np.gradient(np.asarray(X, float), np.asarray(s, float), axis=0, edge_order=2)


def speed(X, s):
    """|| dX / ds || per row (X: (K, ...)) -> (K,)."""
    d = derivative(X, s)

    return np.linalg.norm(d.reshape(len(d), -1), axis=1)


def subsample_idx(K, n):
    """About n evenly spaced indices in range(K), always including both ends."""
    return np.unique(np.linspace(0, K - 1, min(n, K)).round().astype(int))


def pdist(X, idx):
    """Euclidean distance matrix between the rows idx of X (K, ...), via the Gram matrix. Exact zeros on the diagonal."""
    X = np.asarray(X, float)[idx].reshape(len(idx), -1)
    g = X @ X.T
    d = np.diag(g)
    D = np.sqrt(np.clip(d[:, None] + d[None] - 2 * g, 0, None))
    np.fill_diagonal(D, 0.0)

    return D


def pair_values(M):
    """Upper-triangle (i < j) entries of a square matrix -> 1-D."""
    return M[np.triu_indices(len(M), 1)]


def coupling_metrics(x, y):
    """How well does distance x in one space predict distance y in another? x, y: matching 1-D arrays. Returns n, pearson, spearman and the
    log-log slope of y ~ x^slope (1 = proportional, < 1 = saturating, > 1 = amplifying)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    out = {"n": int(len(x)), "pearson": np.nan, "spearman": np.nan, "slope": np.nan, "r2_loglog": np.nan}
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return out
    out["pearson"], out["spearman"] = float(np.corrcoef(x, y)[0, 1]), float(stats.spearmanr(x, y)[0])
    pos = (x > 0) & (y > 0)
    if pos.sum() >= 3 and np.std(np.log(x[pos])) > 0:
        lr = stats.linregress(np.log(x[pos]), np.log(y[pos]))
        out["slope"], out["r2_loglog"] = float(lr.slope), float(lr.rvalue ** 2)

    return out


def reference_index(fam):
    """Index on the dense grid of the reference condition for 'drift from a reference': EXP2 the canonical Sod problem (p_ratio = 0.1);
    EXP3 the middle of the training time range."""
    s_ref = np.log(REF_EXP2_P_RATIO) if fam.name == "exp2" else 0.5 * (fam.dial.lo + fam.dial.hi)

    return int(np.abs(fam.dn.s - s_ref).argmin())


# ---- LoRA: does dW point along directions W already uses? ------------------------------------

def lora_layer(W, dW, rank, tol=1e-9):
    """W (out, in) base weights, dW (K, out, in) weight change per condition. For the left (output side) and right (input side) singular
    subspaces separately, with k = min(rank, dims):
        overlap_*  principal-angle overlap (mean cos^2) of the top-k singular subspaces of W and of dW(s) -> (K,); NaN where dW ~ 0
        chance_*   k / dim: what unrelated random subspaces give (= 1 if k fills the whole space: then overlap is 1 by construction)
    and the update spectrum: erank / pr of the singular values of dW(s), topk {1, 2, rank}: share of ||dW||^2 in the top modes,
    rel_norm = ||dW||_F / ||W||_F."""
    W, dW = np.asarray(W, float), np.asarray(dW, float)
    K, o, i = dW.shape
    k = min(rank, o, i)
    Uw, _, Vwt = np.linalg.svd(W, full_matrices=False)
    Ud, Sd, Vdt = np.linalg.svd(dW, full_matrices=False)                       # batched over the conditions
    size = np.linalg.norm(dW.reshape(K, -1), axis=1)
    ok = size > tol * np.linalg.norm(W)
    out = {"k": k}
    for side, Uref, Uc, dim in (("left", Uw, Ud, o), ("right", Vwt.T, np.swapaxes(Vdt, -1, -2), i)):
        cos = [np.clip(np.linalg.svd(Uref[:, :k].T @ Uc[c][:, :k], compute_uv=False), 0, 1) if ok[c] else None for c in range(K)]
        out[f"overlap_{side}"] = np.array([np.mean(c ** 2) if c is not None else np.nan for c in cos])
        out[f"chance_{side}"] = k / dim
    rk = min(rank, Sd.shape[1])
    en = (Sd ** 2).sum(1)
    out.update({"erank": np.array([effective_rank(r) for r in Sd]), "pr": np.array([participation_ratio(r) for r in Sd]),
                "topk": {q: (Sd[:, :q] ** 2).sum(1) / np.maximum(en, 1e-300) for q in sorted({1, 2, rk})}, "rel_norm": size / np.linalg.norm(W)})

    return out


# ---- FiLM: gating strength and representation drift -------------------------------------------

ACTIVATIONS = {
    "ReLU": lambda z: np.maximum(z, 0.0),
    "LeakyReLU": lambda z: np.where(z > 0, z, 0.01 * z),
    "Tanh": np.tanh,
    "Sigmoid": lambda z: 1.0 / (1.0 + np.exp(-z)),
    "SiLU": lambda z: z / (1.0 + np.exp(-z)),
    "GELU": lambda z: 0.5 * z * (1.0 + erf(z / np.sqrt(2.0))),
    "Identity": lambda z: z,
}


def _np(t):
    return np.asarray(t.detach().cpu().numpy() if hasattr(t, "detach") else t, dtype=np.float64)


def layer_params(inr):
    """Per-layer numpy weights of a base INR (anything with `.layers`, each with `.linear.weight`, `.linear.bias`, `.act`)
    -> [{"W": (out, in), "b": (out,), "act": class name of the activation or None for the linear output layer}]."""
    return [{"W": _np(l.linear.weight), "b": _np(l.linear.bias), "act": None if l.act is None else type(l.act).__name__} for l in inr.layers]


def forward_hidden(params, kind, m, coords):
    """Numpy forward pass of the modulated INR (mirrors Model.render) that returns the post-activation values of every hidden layer:
    list of (K, N, F). m: effective modulation (Model.m_eff), coords (N, N_in)."""
    coords = np.asarray(coords, np.float64)
    h = np.broadcast_to(coords, (len(m[0]),) + coords.shape)
    hidden = []
    for l, p in enumerate(params):
        z = h @ p["W"].T + p["b"]
        if l < len(m):
            ml = np.asarray(m[l], np.float64)
            if kind == "lora":
                z = z + h @ np.swapaxes(ml, -1, -2)
            else:
                F = p["W"].shape[0]
                z = (1.0 + ml[:, None, :F]) * z + ml[:, None, F:]
        if p["act"] not in ACTIVATIONS and p["act"] is not None:
            raise ValueError(f"activation {p['act']!r} not supported (add it to ACTIVATIONS)")
        h = ACTIVATIONS[p["act"]](z) if p["act"] is not None else z
        if p["act"] is not None:
            hidden.append(h)

    return hidden


def film_analysis(params, m_d, coords, s, ref_idx, cfg):
    """gating[l]: mean |gamma - 1| per condition (K,) for hidden layer l.  act: representation drift ||H_c - H_ref|| / ||H_ref|| of every
    hidden layer's activations H (probe points x channels) at conditions act['s'] relative to the reference condition -> drift (C, L)."""
    K, L = len(s), len(m_d)
    gating = [{"mean_abs_gamma_dev": np.abs(np.asarray(x)[:, :np.asarray(x).shape[1] // 2]).mean(1)} for x in m_d]
    n = min(cfg["n_probe"], len(coords))
    probe = np.sort(np.random.default_rng(0).choice(len(coords), n, replace=False))
    cond_idx = np.unique(np.append(subsample_idx(K, cfg["n_act_cond"]), ref_idx))
    sel = lambda c: [x[c:c + 1] for x in m_d]
    h_ref = [h[0] for h in forward_hidden(params, "film", sel(ref_idx), coords[probe])]
    den = [np.linalg.norm(h) for h in h_ref]
    drift = np.zeros((len(cond_idx), L))
    for ci, c in enumerate(cond_idx):
        for l, h in enumerate(forward_hidden(params, "film", sel(c), coords[probe])):
            drift[ci, l] = np.linalg.norm(h[0] - h_ref[l]) / den[l] if den[l] > 0 else np.nan

    return {"gating": gating, "act": {"s": s[cond_idx], "cond_idx": cond_idx, "drift": drift}}


# ---- coupling between spaces ------------------------------------------------------------------

def weight_vectors(kind, params, m, idx):
    """Effective weights of the base network at conditions idx, flattened over layers -> (len(idx), D). LoRA: dW of every layer.
    FiLM: diag(1 + g) W and (1 + g) b + beta of every hidden layer."""
    n = len(idx)
    if kind == "lora":
        return np.concatenate([np.asarray(x)[idx].reshape(n, -1) for x in m], 1)
    parts = []
    for p, ml in zip(params, m):
        F = p["W"].shape[0]
        g, be = 1.0 + np.asarray(ml)[idx, :F], np.asarray(ml)[idx, F:]
        parts += [(g[:, :, None] * p["W"][None]).reshape(n, -1), g * p["b"][None] + be]

    return np.concatenate(parts, 1)


def coupling_analysis(kind, params, m_d, raw_mod, F, s, inside, idx):
    """Distances for all pairs of the conditions idx in four spaces and how they correlate. condition: |s_i - s_j|; modulation: FiLM
    Euclid distance of [gamma - 1 | beta], LoRA of the hypernetwork's raw A, B factors; weights: the effective weights (LoRA dW, FiLM
    diag(gamma) W and gamma*b + beta); function: RMS distance of the predicted fields (each divided by its pooled std).
    metrics: one row per (relation, region in {in, all}): n, pearson, spearman, slope. 'in' = both conditions inside the training range
    (speeds: the condition inside). Relations: cond->mod, cond->func, mod->weight, weight->func, mod->func, and along the dial
    speed_mod->speed_func, speed_weight->speed_func."""
    K = len(s)
    mvec = raw_mod if raw_mod is not None else np.concatenate([np.asarray(x, float).reshape(K, -1) for x in m_d], 1)
    wall = weight_vectors(kind, params, m_d, np.arange(K))
    root_d = np.sqrt(F.shape[1])
    pairs = {"sdist": pair_values(np.abs(s[idx][:, None] - s[idx][None])), "mdist": pair_values(pdist(mvec, idx)),
             "wdist": pair_values(pdist(wall, idx)), "fdist": pair_values(pdist(F, idx) / root_d)}
    iu = np.triu_indices(len(idx), 1)
    both_in = inside[idx][iu[0]] & inside[idx][iu[1]]
    sp_f = speed(F, s) / root_d
    rows = []
    for rel, x, y in (("cond->mod", "sdist", "mdist"), ("cond->func", "sdist", "fdist"), ("mod->weight", "mdist", "wdist"),
                      ("weight->func (F1)", "wdist", "fdist"), ("mod->func (F2)", "mdist", "fdist")):
        for region, mk in (("in", both_in), ("all", np.ones_like(both_in))):
            rows.append({"relation": rel, "region": region, **coupling_metrics(pairs[x][mk], pairs[y][mk])})
    for rel, x in (("speed_mod->speed_func", speed(mvec, s)), ("speed_weight->speed_func", speed(wall, s))):
        for region, mk in (("in", inside), ("all", np.ones_like(inside))):
            rows.append({"relation": rel, "region": region, **coupling_metrics(x[mk], sp_f[mk])})

    return {"metrics": rows}


def analyze_model(kind, fam, r, params, rank=None, raw_mod=None, cfg=None):
    """All measurements of one model. kind 'film' | 'lora'; r: the model's evaluate() output (needs m_d, M_d, pred_d); params: layer_params of
    its base INR; rank: LoRA rank; raw_mod: (K, D) the hypernetwork's raw LoRA factors on the dense grid (LoRA only)."""
    cfg = {**SETTINGS, **(cfg or {})}
    dn = fam.dn
    s, inside = dn.s, dn.split == "in"
    K = len(s)
    idx = subsample_idx(K, cfg["n_pair"])
    ref = reference_index(fam)
    F = (np.asarray(r["pred_d"], np.float32) / np.asarray(fam.pooled, np.float32)).reshape(K, -1)
    V = np.concatenate([np.asarray(x, float).reshape(K, -1) for x in r["M_d"]], 1)       # one modulation vector per condition, all layers
    out = {"kind": kind, "s": s, "inside": inside, "ref_idx": ref, "idx": idx, "mod": {"pca": pca_summary(V, inside, cfg["n_comp"])}}
    if kind == "lora":
        out["lora"] = {"layers": [lora_layer(p["W"], dw, rank) for p, dw in zip(params, r["m_d"])]}
    else:
        out["film"] = film_analysis(params, r["m_d"], _np(dn.coords), s, ref, cfg)
    out["coupling"] = coupling_analysis(kind, params, r["m_d"], raw_mod, F, s, inside, idx)

    return out


# ==============================================================================================
# 5. RUN
# ==============================================================================================

def evaluate(m, fam, snap=None):
    """Modulation, predictions and error of one model on the dense grid; with snap (a Grid) also its last-snapshot solutions there."""
    dn = fam.dn
    md = m.m_eff(fam.cond(dn.s))
    pred_d = m.render(md, dn)                                              # kept only until analyze_model has used it
    S = {"err_d": dn.err(pred_d, fam.pooled), "m_d": md, "M_d": [x / r for x, r in zip(md, m.ref())], "pred_d": pred_d}
    if snap is not None:
        S["snap"] = m.render(m.m_eff(fam.cond(snap.s)), snap)[:, -1]       # (K, X, 3)

    return S


def run(fam, models, baseline, device=None):
    """Measure every model of ONE sample. models {name: Model} and baseline stay on the CPU (each is moved to `device` only while analysed).
    Returns a plain, picklable dict:
        base_err (K,)   normalised L1 of the baseline on the dense grid       floor   numerical solver vs exact (val median)
        meta            experiment, dial grid s, split, dial description (enough to redraw without `fam`)
        R[name]         {'kind', 'err_d' (K,), 'deep': analyze_model(...)}
        snap            {'x', 'titles', 'exact', 'baseline', 'preds'} solutions at three conditions (first model of each kind)"""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    st, dn, dial = fam.st, fam.dn, fam.dial
    kinds = {n: m.kind for n, m in models.items()}
    show = [next(n for n in models if kinds[n] == k) for k in dict.fromkeys(kinds.values())]       # first model of each kind
    snap = fam.exact(fam.snap_s)
    floor = float(np.median(st.err(st.u_exact, fam.pooled)[st.split == "val"]))                    # numerical solver vs exact
    print(f"[{fam.name}] numerical-vs-exact error floor (val median): {floor:.5f}")

    baseline.load(device)
    try:
        base_err = dn.err(baseline.predict(fam.baseline_in(dn.s, dn), dn), fam.pooled)
        base_snap = baseline.predict(fam.baseline_in(snap.s, snap), snap)[:, -1]
    finally:
        baseline.unload()

    R = {}
    for name, m in models.items():
        print(f"=== {name} ({m.kind}) on {device} ===")
        m.load(device)
        try:
            r = evaluate(m, fam, snap if name in show else None)
            deep = analyze_model(m.kind, fam, r, layer_params(m.inr), rank=getattr(m.inr, "lora_rank", None),
                                 raw_mod=m.raw_mod(fam.cond(dn.s)) if m.kind == "lora" else None)
        finally:
            m.unload()                                                      # always free the GPU, even on an error
        R[name] = {"kind": m.kind, "err_d": r["err_d"], "deep": deep, "snap": r.get("snap")}

    title = lambda s: f"{fam.label(s)}: {dial.name} = {dial.fn(s):.3g}"
    meta = {"experiment": fam.name, "s": dn.s, "split": dn.split, "ext_frac": ext_fraction(dn.s, dial, fam.ext_width),
            "dial": {"name": dial.name, "log": dial.log, "lo": dial.lo, "hi": dial.hi, "lim": tuple(dial.lim), "train_s": dial.train_s,
                     "marks": list(dial.marks), "fn": "exp" if fam.name == "exp2" else "identity"}}                    # enough to redraw without `fam`
    res = {"base_err": base_err, "floor": floor, "meta": meta, "R": {n: {k: v for k, v in r.items() if k != "snap"} for n, r in R.items()},
           "snap": {"x": fam.x, "titles": [title(s) for s in snap.s], "exact": snap.u[:, -1], "baseline": base_snap,
                    "preds": {n: R[n]["snap"] for n in show}}}

    return res