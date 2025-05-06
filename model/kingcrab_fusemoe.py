"""
CrabNet‑FuseMoE ================
A CrabNet‑style composition‑only property predictor that uses **three descriptor
embeddings per element** – MAGPIE, mat2vec and Oliynyk – and fuses them with a
**Laplace‑gated Mixture‑of‑Experts** layer.  Falls back to the classic single‑
embedding formulation if `use_moe=False`.
"""

import numpy as np
import pandas as pd
import torch
from torch import nn

RNG_SEED = 42
torch.manual_seed(RNG_SEED)
np.random.seed(RNG_SEED)
DTYPE = torch.float32

# ──────────────────────────────────────────────────────────────────────────────
# MoE building blocks
# ──────────────────────────────────────────────────────────────────────────────
class MoEElementFusion(nn.Module):
    """Laplace‑gated sparse Mixture‑of‑Experts for element‑level fusion.

    Args
    ----
    d_model : int
        Hidden size expected by the Transformer.
    n_experts : int
        Number of feed‑forward experts shared by all views.
    top_k : int
        How many experts a router sends each token to.
    n_views : int
        Number of alternative descriptor embeddings per element.
    """

    def __init__(self, d_model: int, n_experts: int = 16,
                 top_k: int = 4, n_views: int = 3):
        super().__init__()
        self.d_model, self.n_experts, self.top_k, self.n_views = (
            d_model, n_experts, top_k, n_views)

        # Shared expert pool (position‑wise FFNs)
        self.experts = nn.ModuleList([nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model))
            for _ in range(n_experts)])

        # One simple linear router per view  →  n_experts logits
        self.routers = nn.ModuleList([nn.Linear(d_model, n_experts, bias=False)
                                      for _ in range(n_views)])

        # Expert keys for Laplace gating (initialised orthogonal)
        self.register_parameter('expert_keys', nn.Parameter(
            torch.empty(n_experts, d_model)))
        nn.init.orthogonal_(self.expert_keys)

    # ---------------------------------------------------------------------
    def _laplace_gate(self, h: torch.Tensor, router: nn.Module) -> torch.Tensor:
        """Compute Laplace gate:  −‖h−Wj‖² for each expert j, top‑k softmax."""
        # (batch, d) −> (batch, n_experts)
        logits = -torch.cdist(h, self.expert_keys, p=2) ** 2
        logits = logits + router(h)  # router adds view‑specific bias
        topk_val, topk_idx = logits.topk(self.top_k, dim=-1)
        gate = (topk_val - topk_val.logsumexp(-1, keepdim=True)).exp()
        return gate, topk_idx

    # ---------------------------------------------------------------------
    def forward(self, views: list[torch.Tensor]):
        if len(views) != self.n_views:
            raise ValueError(f"Expected {self.n_views} views, got {len(views)}")
        fused = 0
        for v, router in zip(views, self.routers):  # iterate over views
            gate, idx = self._laplace_gate(v, router)  # (B, k)
            # Aggregate expert outputs
            out = torch.zeros_like(v)
            for j in range(self.top_k):
                exp_j = self.experts[idx[:, j]]  # ModuleList is indexable with tensor
                out = out + gate[:, j:j+1] * exp_j(v)
            fused = fused + out
        return fused  # (batch, d_model)

# ──────────────────────────────────────────────────────────────────────────────
# Descriptor embedder
# ──────────────────────────────────────────────────────────────────────────────
class MultiDescriptorEmbedder(nn.Module):
    """Returns three projected descriptor embeddings per element: mat2vec,
    MAGPIE and Oliynyk.  Outputs a *list* of tensors length 3.
    """

    def __init__(self, d_model: int, elem_dir: str,
                 compute_device: torch.device | None = None):
        super().__init__()
        self.d_model = d_model
        self.device = compute_device

        def _load(fname: str):
            arr = pd.read_csv(f"{elem_dir}/{fname}", index_col=0).values
            return np.concatenate([np.zeros((1, arr.shape[1])), arr])  # pad row‑0

        tbl_mat2vec = _load('mat2vec.csv')      # 200‑d
        tbl_magpie  = _load('magpie.csv')       # 145‑d
        tbl_oliy    = _load('oliynyk.csv')      # 112‑d

        self.emb_mat2vec = nn.Embedding.from_pretrained(
            torch.tensor(tbl_mat2vec, dtype=DTYPE), freeze=True)
        self.emb_magpie  = nn.Embedding.from_pretrained(
            torch.tensor(tbl_magpie, dtype=DTYPE),  freeze=True)
        self.emb_oliy    = nn.Embedding.from_pretrained(
            torch.tensor(tbl_oliy, dtype=DTYPE),    freeze=True)

        self.proj_mat2vec = nn.Linear(tbl_mat2vec.shape[1], d_model)
        self.proj_magpie  = nn.Linear(tbl_magpie.shape[1],  d_model)
        self.proj_oliy    = nn.Linear(tbl_oliy.shape[1],    d_model)

    # ------------------------------------------------------------------
    def forward(self, Z: torch.Tensor):
        # Z: (batch, n_elem) atomic‑number indices (0=pad)
        v1 = self.proj_mat2vec(self.emb_mat2vec(Z))
        v2 = self.proj_magpie(self.emb_magpie(Z))
        v3 = self.proj_oliy(self.emb_oliy(Z))
        return [v1, v2, v3]  # list length 3, each (batch, n_elem, d_model)

# ──────────────────────────────────────────────────────────────────────────────
# Fractional encoders (unchanged)
# ──────────────────────────────────────────────────────────────────────────────
class FractionalEncoder(nn.Module):
    """Sine/cosine encoding for the atomic fraction."""
    def __init__(self, d_model: int, resolution: int = 5000, log10: bool = False):
        super().__init__()
        half = d_model // 2
        x = torch.linspace(0, resolution - 1, resolution).view(resolution, 1)
        frac = torch.linspace(0, half - 1, half).view(1, half).repeat(resolution, 1)
        pe = torch.zeros(resolution, half)
        pe[:, 0::2] = torch.sin(x / torch.pow(50, 2 * frac[:, 0::2] / half))
        pe[:, 1::2] = torch.cos(x / torch.pow(50, 2 * frac[:, 1::2] / half))
        self.register_buffer('pe', pe)
        self.log10 = log10
        self.resolution = resolution

    def forward(self, r: torch.Tensor):
        if self.log10:
            r = 0.0025 * (torch.log2(r)) ** 2
            r = torch.clamp(r, max=1)
        r = torch.clamp(r, min=1 / self.resolution)
        idx = torch.round(r * self.resolution).long() - 1
        return self.pe[idx]

# ──────────────────────────────────────────────────────────────────────────────
# Encoder
# ──────────────────────────────────────────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self, d_model: int, N: int, heads: int, use_moe: bool = True,
                 compute_device: torch.device | None = None):
        super().__init__()
        self.d_model, self.use_moe = d_model, use_moe
        elem_dir = 'madani/data/element_properties'
        self.embedder = MultiDescriptorEmbedder(d_model, elem_dir, compute_device)
        if use_moe:
            self.moe = MoEElementFusion(d_model, n_experts=16, top_k=4, n_views=3)
        self.pe_lin = FractionalEncoder(d_model, log10=False)
        self.pe_log = FractionalEncoder(d_model, log10=True)

        enc_layer = nn.TransformerEncoderLayer(d_model, heads, 4 * d_model, 0.1)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=N)

        self.emb_scale = nn.Parameter(torch.tensor(0.0))
        self.pe_scale  = nn.Parameter(torch.tensor(0.0))
        self.ple_scale = nn.Parameter(torch.tensor(0.0))

    # ------------------------------------------------------------------
    def forward(self, Z: torch.Tensor, frac: torch.Tensor):
        """Z (batch, n_elem), frac (batch, n_elem)."""
        views = self.embedder(Z)
        if self.use_moe:
            x = self.moe([v * 2 ** self.emb_scale for v in views])
        else:
            x = sum(views) / len(views)

        # Fractional positional encodings
        pe  = torch.zeros_like(x)
        ple = torch.zeros_like(x)
        pe[:, :, :self.d_model // 2] = self.pe_lin(frac) * 2 ** self.pe_scale
        ple[:, :, self.d_model // 2:] = self.pe_log(frac) * 2 ** self.ple_scale

        x = x + pe + ple
        x = self.transformer(x.transpose(0, 1)).transpose(0, 1)
        return x * frac.unsqueeze(-1)  # optional fractional weighting

# ──────────────────────────────────────────────────────────────────────────────
# Residual head (unchanged)
# ──────────────────────────────────────────────────────────────────────────────
class ResidualNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden: list[int]):
        super().__init__()
        dims = [input_dim] + hidden
        self.fcs  = nn.ModuleList([nn.Linear(dims[i], dims[i + 1])
                                   for i in range(len(hidden))])
        self.res  = nn.ModuleList([nn.Linear(dims[i], dims[i + 1], bias=False)
                                   if dims[i] != dims[i + 1] else nn.Identity()
                                   for i in range(len(hidden))])
        self.acts = nn.ModuleList([nn.LeakyReLU() for _ in hidden])
        self.fc_out = nn.Linear(dims[-1], output_dim)

    def forward(self, h):
        for fc, r, a in zip(self.fcs, self.res, self.acts):
            h = a(fc(h)) + r(h)
        return self.fc_out(h)

# ──────────────────────────────────────────────────────────────────────────────
# CrabNet‑FuseMoE model
# ──────────────────────────────────────────────────────────────────────────────
class CrabNet(nn.Module):
    def __init__(self, out_dims: int = 3, d_model: int = 512, N: int = 3,
                 heads: int = 4, use_moe: bool = True):
        super().__init__()
        self.encoder = Encoder(d_model, N, heads, use_moe)
        self.output_nn = ResidualNetwork(d_model, out_dims * 2, [1024, 512, 256, 128])
        self.out_dims = out_dims

    def forward(self, Z, frac):
        h = self.encoder(Z, frac)            # (batch, n_elem, d_model)
        mask = (Z == 0).unsqueeze(-1).repeat(1, 1, self.out_dims * 2)
        out = self.output_nn(h).masked_fill(mask, 0)
        out = out.sum(1) / (~mask).sum(1)    # element‑wise average
        y, logit = out.chunk(2, dim=-1)
        p = torch.sigmoid(logit)
        return y * p

# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Dummy forward pass
    Z    = torch.tensor([[1, 8, 0, 0], [11, 17, 8, 0]])  # H2O, NaClO
    frac = torch.tensor([[0.666, 0.333, 0, 0], [0.5, 0.25, 0.25, 0]])
    model = CrabNet(out_dims=1, use_moe=True)
    y_hat = model(Z, frac)
    print("Output shape:", y_hat.shape)
