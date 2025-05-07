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
data_type_torch = torch.float32

def get_default_device():
    """Return CUDA if available, else CPU."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ──────────────────────────────────────────────────────────────────────────────
# MoE building blocks
# ──────────────────────────────────────────────────────────────────────────────
class MoEElementFusion(nn.Module):
    def __init__(self, d_model: int, n_experts: int = 16, top_k: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = top_k

        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(d_model, 4*d_model),
                                                    nn.GELU(),
                                                    nn.Linear(4*d_model, d_model))
                                       for _ in range(n_experts)])
        # View‑specific routers will be supplied from outside
        self.routers = nn.ModuleList()
        # Expert keys used by Laplace gate ‑‑ give them a batch‑broadcast axis
        key = torch.empty(1, n_experts, d_model)
        nn.init.uniform_(key, -0.02, 0.02)
        self.register_buffer("expert_keys", key)

    def add_router(self):
        """Append a new router (1×lin) for an additional view."""
        self.routers.append(nn.Linear(self.d_model, self.n_experts))
        return len(self.routers) - 1   # index

    # ---------------------------------------------------------------------
    # Laplace gate (patched): works on 3‑D tokens (B, L, d)
    # ---------------------------------------------------------------------
    def _laplace_gate(self, h, router):
        """Compute soft Laplace gating weights & top‑k indices."""
        B, L, D = h.shape
        # (B, L, n) where n = n_experts
        dist = torch.cdist(h, self.expert_keys.expand(B, -1, -1), p=2)
        logits = -(dist ** 2) + router(h)              # add view bias
        top_val, top_idx = logits.topk(self.top_k, dim=-1)
        weights = torch.softmax(top_val, dim=-1)       # (B, L, k)
        return weights, top_idx

    def forward(self, views):
        # views: list of (B, L, d_model)
        if not self.routers:            # lazily create routers to match views
            for _ in views:
                self.add_router()

        fused = 0.0
        for v, router in zip(views, self.routers):
            w, idx = self._laplace_gate(v, router)
            # Gather expert outputs — loop since k is small (≤4)
            out = 0.0
            for k_slot in range(self.top_k):
                sel = idx[..., k_slot]                 # (B, L)
                # (B, L, d) gather
                exp_out = torch.stack([self.experts[j](v[i]) for i, j in
                                       torch.cartesian_prod(torch.arange(v.shape[0]),
                                                            torch.arange(v.shape[1]))])
                exp_out = exp_out.view(v.shape)
                out += w[..., k_slot:k_slot+1] * exp_out
            fused += out
        return fused

# ---------------------------------------------------------------------------
# Multi‑descriptor embedder (MAGPIE, mat2vec, Oliynyk)
# ---------------------------------------------------------------------------
class MultiDescriptorEmbedder(nn.Module):
    def __init__(self, d_model: int, elem_dir: str, device: torch.device):
        super().__init__()
        tables = {
            "mat2vec":  "mat2vec.csv",
            "magpie":   "magpie.csv",
            "oliynyk":  "oliynyk.csv",
        }
        self.embs, self.projs = nn.ModuleDict(), nn.ModuleDict()
        for k, fn in tables.items():
            arr = pd.read_csv(f"{elem_dir}/{fn}", index_col=0).values
            feat = arr.shape[-1]
            pad = np.zeros((1, feat))
            weight = torch.as_tensor(np.vstack([pad, arr]), dtype=data_type_torch)
            self.embs[k] = nn.Embedding.from_pretrained(weight, freeze=False).to(device)
            self.projs[k] = nn.Linear(feat, d_model).to(device)

    def forward(self, Z):
        views = []
        for k in self.embs.keys():
            emb = self.embs[k](Z)
            views.append(self.projs[k](emb))     # (B, L, d)
        return views
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
    def __init__(self, d_model: int, N: int, heads: int, use_moe: bool, n_experts: int, top_k: int, device):
        super().__init__()
        self.device = device
        self.embedder = MultiDescriptorEmbedder(d_model, "madani/data/element_properties", device)
        self.use_moe = use_moe
        if use_moe:
            self.fuser = MoEElementFusion(d_model=d_model, n_experts=n_experts, top_k=top_k)
        self.pe  = FractionalEncoder(d_model, resolution=5000, log10=False)
        self.ple = FractionalEncoder(d_model, resolution=5000, log10=True)
        enc_layer = nn.TransformerEncoderLayer(d_model, heads, dim_feedforward=2048, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, N)

    def forward(self, Z, frac):  # Z: (B, L) element indices
        views = self.embedder(Z)
        if self.use_moe:
            x = self.fuser(views)                    # (B, L, d_model)
        else:
            x = sum(views)                           # simple additive fusion
        # positional encodings
        x += self.pe(frac) + self.ple(frac)
        x = self.transformer(x)
        return x * frac.unsqueeze(-1)                # optional fractional scaling

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
    def __init__(self, out_dims=3, d_model=512, N=3, heads=4, use_moe=True, n_experts=16, top_k=4, residual_nn='roost', device=None):
        super().__init__()
        self.device = get_default_device() if device is None else device
        self.encoder = Encoder(d_model, N, heads, use_moe, n_experts, top_k, self.device).to(self.device)
        self.out_dims = out_dims
        self.d_model = d_model
        self.N = N
        self.heads = heads
        hidden = [1024, 512, 256, 128] if residual_nn == 'roost' else [256, 128]
        self.output_nn = ResidualNetwork(d_model, out_dims, hidden).to(self.device)

    @property
    def compute_device(self):
        return self.device

    def forward(self, Z, frac):
        Z, frac = Z.to(self.device), frac.to(self.device)
        h = self.encoder(Z, frac)                    # (B, L, d_model)
        out = self.output_nn(h)
        mask = (Z == 0).unsqueeze(-1).expand_as(out)
        out = out.masked_fill(mask, 0)
        out = out.sum(1) / (~mask).sum(1)            # average element contributions
        return out

# =============================================================
if __name__ == "__main__":
    B, L = 4, 8               # batch size, max elements per formula
    Z = torch.randint(1, 20, (B, L))
    frac = torch.rand(B, L)
    model = CrabNet().train()
    pred = model(Z, frac)
    print(pred.shape)
