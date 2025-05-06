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
    """Return three projected descriptor views per element
    (mat2vec, MAGPIE, Oliynyk)."""
    def __init__(self, d_model: int, elem_dir: str, device):
        super().__init__()
        self.device = device
        tables = {}
        for name in ("mat2vec", "magpie", "oliynyk"):
            path = f"{elem_dir}/{name}.csv"
            tables[name] = pd.read_csv(path, index_col=0).values

        self.embs, self.projs = nn.ModuleDict(), nn.ModuleDict()
        for k, arr in tables.items():
            feat = arr.shape[-1]
            pad = np.zeros((1, feat))              # idx 0 = padding
            weight = torch.as_tensor(np.vstack([pad, arr]), dtype=data_type_torch)
            self.embs[k] = nn.Embedding.from_pretrained(weight, freeze=False)
            self.projs[k] = nn.Linear(feat, d_model)

    def forward(self, z):
        views = []
        for k in ("mat2vec", "magpie", "oliynyk"):
            emb = self.embs[k](z).to(self.device)
            views.append(self.projs[k](emb))
        return views  # list len = 3, each (B, L, d_model)
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
            self.fuser = MoEElementFusion(d_model=d_model, n_experts=n_experts, k=top_k)
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
