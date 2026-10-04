"""
Base INR + hypernetwork modulation (FiLM and LoRA)
==================================================

Plan:
  1. Train a plain INR (coordinates -> field values) by itself.
  2. Freeze it.
  3. A hypernetwork maps a conditioning input `cond` to modulation tensors that are fed into
     the frozen INR's forward pass. Only the trunk and heads are trained.

      cond -> HyperTrunk -> features -+-> FiLMHead -> (gamma, beta) per layer -+
                                      |                                        +-> INR(coords, ...) -> out
                                      +-> LoRAHead -> (A, B) per layer --------+

Per layer (weights W, bias b), in this order:
  LoRA:  z = W x + b + scale * x A^T B^T     (low-rank weight update, W_eff = W + scale * B @ A)
  FiLM:  z = gamma * z + beta                (per-feature scale and shift)
  then the activation.

Shapes: bs = batch (one modulation per conditioning input), N = coordinate points.
  coords (bs, N, N_in); modulation (bs, F) for FiLM, (bs, r, in) and (bs, out, r) for LoRA.
  Unbatched coords (N, N_in) work with unbatched modulation, or with LoRA alone. To share
  one coordinate grid across a batch of modulations, expand it first (a view, no copy):
      coords = coords.unsqueeze(0).expand(bs, -1, -1)
"""

import torch
import torch.nn as nn


# ======================================================================================
# INR
# ======================================================================================

class INRLayer(nn.Module):
    """Linear -> optional LoRA -> optional FiLM -> optional activation.
    Owns no modulation parameters: they are passed to forward() by the hypernetwork."""

    def __init__(self, N_in, N_out, act=nn.ReLU):
        super().__init__()
        self.N_in, self.N_out = N_in, N_out
        self.linear = nn.Linear(N_in, N_out)
        self.act = act() if act else None     # act=None -> linear output layer

    def forward(self, x, film=None, lora=None, lora_scale=1.0):
        z = self.linear(x)                    # (..., N, N_out)

        if lora is not None:
            A, B = lora                       # A: (..., r, N_in), B: (..., N_out, r)
            # x A^T B^T equals x (B A)^T but never builds the full (N_out, N_in) matrix.
            # matmul batches over leading dims, so each sample gets its own update.
            z = z + lora_scale * torch.matmul(torch.matmul(x, A.transpose(-1, -2)), B.transpose(-1, -2))

        if film is not None:
            gamma, beta = film                # (..., N_out), gamma already centered at 1
            # Broadcasting aligns from the right, so (bs, F) would meet (bs, N, F) wrongly.
            # Insert axes until it is (bs, 1, F): same modulation for every point in a sample.
            while gamma.dim() < z.dim():
                gamma, beta = gamma.unsqueeze(-2), beta.unsqueeze(-2)
            z = gamma * z + beta

        return self.act(z) if self.act else z


class INR(nn.Module):
    """
    INRLayers stacked: N_in -> N_hidden (x N_layers) -> N_out, with a linear output layer.
    N_layers counts hidden-width layers, so there are N_layers + 1 INRLayers in total.
    """

    def __init__(self, N_in, N_hidden, N_layers, N_out, lora_rank=4, lora_alpha=4.0, act=nn.ReLU):
        super().__init__()
        self.lora_rank = lora_rank                 # fixes the shapes of the LoRA tensors
        self.lora_scale = lora_alpha / lora_rank

        dims = [N_in] + [N_hidden] * N_layers + [N_out]
        self.layers = nn.ModuleList(
            INRLayer(dims[i], dims[i + 1], act=act if i < len(dims) - 2 else None)
            for i in range(len(dims) - 1)
        )

    def forward(self, x, film=None, lora=None):
        """
        film: list of (gamma, beta), one per layer except the output layer, or None
        lora: list of (A, B), one per layer including the output layer, or None
        Returns (output, x), with x connected to output in the autograd graph (d out / d x).
        """
        x = x.to(torch.float32)              # differentiable cast; keeps any upstream graph
        if not x.requires_grad:
            # No graph to preserve, so make a fresh leaf. Not x.requires_grad_(): that would
            # flip the flag on the caller's own tensor in place.
            x = x.detach().requires_grad_(True)

        # per-sample modulation with flat (B, N_in) coords: add a points axis of 1
        squeeze = False
        if x.dim() == 2 and lora is not None and lora[0][0].dim() == 3:
            h, squeeze = x.unsqueeze(1), True          # (B, 1, N_in)
        else:
            h = x

        for i, layer in enumerate(self.layers):
            f = film[i] if film is not None and i < len(self.layers) - 1 else None
            l = lora[i] if lora is not None else None
            h = layer(h, film=f, lora=l, lora_scale=self.lora_scale)

        if squeeze:
            h = h.squeeze(1)                            # back to (B, N_out)
        return h, x

    # Build the heads from these so their sizes can't drift out of sync with the INR.
    @property
    def layer_dims_film(self):
        return [l.N_out for l in self.layers[:-1]]

    @property
    def layer_shapes_lora(self):
        return [(l.N_out, l.N_in) for l in self.layers]

    def freeze(self):
        """Call after base training."""
        for p in self.parameters():
            p.requires_grad = False
        return self.eval()


# ======================================================================================
# HYPERNETWORK
# ======================================================================================

class HyperTrunk(nn.Module):
    """MLP: conditioning input -> feature vector shared by the heads."""

    def __init__(self, N_in, N_hidden, N_layers, N_out, act=nn.GELU):
        super().__init__()
        dims = [N_in] + [N_hidden] * N_layers + [N_out]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:                  # no activation on the last Linear
                layers.append(act())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.to(torch.float32))


class FiLMHead(nn.Module):
    """
    Trunk features -> list of (gamma, beta), one pair per FiLM'd layer.
    One Linear emits a flat vector [all gammas | all betas] that forward() cuts up.
    Zero init gives gamma = 1, beta = 0 (identity). The "+1" lives here, and only here.
    """

    def __init__(self, N_trunk_out, base_layer_dims):
        super().__init__()
        self.dims = base_layer_dims
        self.head = nn.Linear(N_trunk_out, 2 * sum(base_layer_dims))
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, feats):
        gammas, betas = self.head(feats).split(sum(self.dims), dim=-1)
        return list(zip((1.0 + gammas).split(self.dims, dim=-1), betas.split(self.dims, dim=-1)))


class LoRAHead(nn.Module):
    """
    Trunk features -> list of (A, B), one pair per layer (output layer included).
    One Linear emits a flat vector [A_0 | B_0 | A_1 | B_1 | ...] with A: (r, in), B: (out, r).

    Init: dW = B @ A must start at 0, but all-zero A and B never train (each one's gradient
    is proportional to the other). So A is random and B is exactly 0: dW = 0, yet B gets
    gradient. The head WEIGHT is zero and the BIAS carries the init (random on A slices).
    """

    def __init__(self, N_trunk_out, base_layer_shapes, lora_rank):
        super().__init__()
        self.shapes, self.r = base_layer_shapes, lora_rank          # shapes: [(out, in), ...]
        self.sizes = [s for o, i in base_layer_shapes for s in (lora_rank * i, o * lora_rank)]
        self.head = nn.Linear(N_trunk_out, sum(self.sizes))

        nn.init.zeros_(self.head.weight)
        with torch.no_grad():
            self.head.bias.zero_()
            for k, chunk in enumerate(self.head.bias.split(self.sizes)):   # chunks are views
                if k % 2 == 0:                                      # even chunks are A slices
                    bound = self.shapes[k // 2][1] ** -0.5
                    chunk.uniform_(-bound, bound)

    def forward(self, feats):
        chunks = self.head(feats).split(self.sizes, dim=-1)
        lead = feats.shape[:-1]
        return [
            (chunks[2 * k].reshape(*lead, self.r, i), chunks[2 * k + 1].reshape(*lead, o, self.r))
            for k, (o, i) in enumerate(self.shapes)
        ]


# ======================================================================================
# FiLM Hypernetwork
# ======================================================================================

class ModFiLM(nn.Module):
    """
    Conditioning input -> HyperTrunk -> FiLM modulation parameters.
    """

    def __init__(
        self,
        cond_dim,
        trunk_hidden,
        trunk_layers,
        trunk_out,
        inr,
    ):
        super().__init__()

        self.trunk = HyperTrunk(
            N_in=cond_dim,
            N_hidden=trunk_hidden,
            N_layers=trunk_layers,
            N_out=trunk_out,
        )

        self.head = FiLMHead(
            N_trunk_out=trunk_out,
            base_layer_dims=inr.layer_dims_film,
        )

    def forward(self, cond):
        features = self.trunk(cond)
        film = self.head(features)
        return film




# ======================================================================================
# LoRA Hypernetwork
# ======================================================================================

class ModLoRA(nn.Module):
    """
    Conditioning input -> HyperTrunk -> LoRA modulation parameters.
    """

    def __init__(
        self,
        cond_dim,
        trunk_hidden,
        trunk_layers,
        trunk_out,
        inr,
    ):
        super().__init__()

        self.trunk = HyperTrunk(
            N_in=cond_dim,
            N_hidden=trunk_hidden,
            N_layers=trunk_layers,
            N_out=trunk_out,
        )

        self.head = LoRAHead(
            N_trunk_out=trunk_out,
            base_layer_shapes=inr.layer_shapes_lora,
            lora_rank=inr.lora_rank,
        )

    def forward(self, cond):
        features = self.trunk(cond)
        lora = self.head(features)
        return lora



