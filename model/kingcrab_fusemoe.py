"""
CrabNet‑FuseMoE (stable 2025‑05‑07)
==================================
* Fixes the CUDA `device‑side assert triggered` by ensuring **all tensors live on
  the same device** inside the Laplace gate.
* Re‑implements expert aggregation with vectorised `torch.gather` ‑‑ no huge
  Python loops, vastly faster and memory‑safe.
* Adds plenty of inline shapes so you can follow the tensor algebra.
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
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Laplace‑gated Mixture‑of‑Experts fusion layer (vectorised, device‑safe)
# ---------------------------------------------------------------------------
class MoEElementFusion(nn.Module):
    def __init__(self, d_model: int, n_experts: int = 16, top_k: int = 4):
        super().__init__()
        self.d_model, self.n_experts, self.top_k = d_model, n_experts, top_k
        # ‑‑ FFN experts ----------------------------------------------------
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, 4*d_model), nn.GELU(),
                          nn.Linear(4*d_model, d_model))
            for _ in range(n_experts)])
        # ‑‑ View‑specific routers (1×Linear each) will be **added lazily**
        self.routers = nn.ModuleList()
        # Expert keys for Laplace gate (n_experts, d_model)
        key = torch.empty(n_experts, d_model)
        nn.init.uniform_(key, -0.02, 0.02)
        self.register_buffer("expert_keys", key)   # device‑agnostic; move in fwd

    # ------------------ helper to sync router count with view count ----------
    def _ensure_router(self, n_views: int):
        while len(self.routers) < n_views:
            self.routers.append(nn.Linear(self.d_model, self.n_experts))

    # ------------------ Laplace gate  (B,L,D) ‑> (weights, idx) --------------
    def _laplace_gate(self, h: torch.Tensor, router: nn.Linear):
        """h: (B,L,D)  →  weights (B,L,k), idx (B,L,k)"""
        B, L, D = h.shape
        # Broadcast expert keys across batch: (B, n_experts, D)
        keys = self.expert_keys.to(h.device).unsqueeze(0).expand(B, -1, -1)
        # Pairwise Euclidean distance → (B, L, n)
        dist = torch.cdist(h, keys, p=2)
        logits = -(dist ** 2) + router(h)          # Laplace score + bias
        top_val, top_idx = logits.topk(self.top_k, dim=-1)  # (...,k)
        weights = torch.softmax(top_val, dim=-1)             # (...,k)
        return weights, top_idx

    # ------------------ forward over *all* views ----------------------------
    def forward(self, views):
        """views: list of tensors, each (B,L,D)"""
        self._ensure_router(len(views))
        fused = 0.0
        for v, router in zip(views, self.routers):
            w, idx = self._laplace_gate(v, router)      # (B,L,k) each
            # Compute every expert output once: (B,L,n,D)
            exp_out = torch.stack([e(v) for e in self.experts], dim=2)
            # Gather the k selected experts: (B,L,k,D)
            idx_exp = idx.unsqueeze(-1).expand(-1, -1, -1, self.d_model)
            sel = torch.gather(exp_out, 2, idx_exp)
            # Weight & sum across k: (B,L,D)
            fused += (w.unsqueeze(-1) * sel).sum(2)
        return fused  # (B,L,D)

# ---------------------------------------------------------------------------
# Multi‑descriptor embedder (MAGPIE, mat2vec, Oliynyk)
# ---------------------------------------------------------------------------
class MultiDescriptorEmbedder(nn.Module):
    def __init__(self, d_model: int, elem_dir: str, device: torch.device):
        super().__init__()
        tables = {"mat2vec": "mat2vec.csv", "magpie": "magpie.csv", "oliynyk": "oliynyk.csv"}
        self.embs, self.projs = nn.ModuleDict(), nn.ModuleDict()
        for name, file in tables.items():
            arr = pd.read_csv(f"{elem_dir}/{file}", index_col=0).values
            feat = arr.shape[-1]
            pad  = np.zeros((1, feat))                    # idx 0 = padding
            weight = torch.as_tensor(np.vstack([pad, arr]), dtype=data_type_torch)
            self.embs[name]  = nn.Embedding.from_pretrained(weight, freeze=False).to(device)
            self.projs[name] = nn.Linear(feat, d_model).to(device)

    def forward(self, Z):
        return [self.projs[k](self.embs[k](Z)) for k in self.embs]  # list[(B,L,D)]

# ---------------------------------------------------------------------------
# Fractional encoders (unchanged)
# ---------------------------------------------------------------------------
class FractionalEncoder(nn.Module):
    def __init__(self, d_model: int, resolution: int = 100, log10: bool = False):
        super().__init__()
        half = d_model // 2
        x = torch.linspace(0, resolution-1, resolution).view(resolution, 1)
        freq = torch.linspace(0, half-1, half).view(1, half).repeat(resolution, 1)
        pe = torch.zeros(resolution, half)
        pe[:, 0::2] = torch.sin(x / torch.pow(50, 2*freq[:, 0::2] / half))
        pe[:, 1::2] = torch.cos(x / torch.pow(50, 2*freq[:, 1::2] / half))
        self.register_buffer("pe", pe)
        self.resolution, self.log10 = resolution, log10

    def forward(self, frac):
        x = frac.clone()
        if self.log10:
            x = 0.0025 * (torch.log2(x.clamp_min(1e-8)))**2
            x = torch.clamp(x, max=1)
        x = torch.clamp(x, min=1/self.resolution)
        idx = torch.round(x * self.resolution).long() - 1
        return self.pe[idx]

# ---------------------------------------------------------------------------
# Encoder + CrabNet
# ---------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, d_model: int, N: int, heads: int, use_moe: bool,
                 n_experts: int, top_k: int, device):
        super().__init__()
        self.device, self.use_moe = device, use_moe
        self.embed = MultiDescriptorEmbedder(d_model, "madani/data/element_properties", device)
        if use_moe:
            self.fuser = MoEElementFusion(d_model, n_experts, top_k).to(device)
        self.pe  = FractionalEncoder(d_model, 5000, log10=False)
        self.ple = FractionalEncoder(d_model, 5000, log10=True)
        enc_layer = nn.TransformerEncoderLayer(d_model, heads, 2048, 0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, N)

    def forward(self, Z, frac):
        views = self.embed(Z.to(self.device))
        x = self.fuser(views) if self.use_moe else sum(views)
        # add fractional encodings
        half = x.size(-1) // 2
        x = x + torch.cat([self.pe(frac)[..., :half], self.ple(frac)[..., :half]], dim=-1)
        x = self.encoder(x, src_key_padding_mask=(Z == 0))
        return x

class ResidualNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden):
        super().__init__()
        dims = [input_dim] + hidden
        self.blocks = nn.ModuleList()
        for i in range(len(dims)-1):
            self.blocks.append(nn.Sequential(nn.Linear(dims[i], dims[i+1]), nn.GELU()))
        self.fc_out = nn.Linear(dims[-1], output_dim)
    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return self.fc_out(x)

class CrabNet(nn.Module):
    def __init__(self, out_dims=3, d_model=512, N=3, heads=4, use_moe=True,
                 n_experts=16, top_k=4, residual_nn='roost', device=None):
        super().__init__()
        self.device = get_default_device() if device is None else device
        self.encoder = Encoder(d_model, N, heads, use_moe, n_experts, top_k, self.device).to(self.device)
        hidden = [1024,512,256,128] if residual_nn=='roost' else [256,128]
        self.output_nn = ResidualNetwork(d_model, out_dims*2, hidden).to(self.device)

    def forward(self, Z, frac):
        h = self.encoder(Z.to(self.device), frac.to(self.device))
        out = self.output_nn(h)
        mask = (Z == 0).unsqueeze(-1).expand_as(out)
        out = out.masked_fill(mask, 0).sum(1) / (~mask).sum(1)
        pred, logit = out.chunk(2, -1)
        prob = torch.ones_like(pred)
        prob[:, :logit.size(-1)] = torch.sigmoid(logit)
        return pred * prob

# ---------------------------------------------------------------------------
# Quick smoke test -----------------------------------------------------------
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    B, L = 2, 5
    Z = torch.tensor([[1,8,13,52,0],[17,17,30,0,0]])  # element indices
    frac = torch.rand(B, L)
    model = CrabNet(use_moe=True)
    y = model(Z, frac)
    print("Output:", y.shape)
