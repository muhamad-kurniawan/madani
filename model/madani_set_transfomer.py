import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import pandas as pd
from pathlib import Path

# -----------------------------------------------------------------------------
# Helper Blocks for Set Transformer
# -----------------------------------------------------------------------------
class MultiheadAttentionBlock(nn.Module):
    """Multi‑head Attention Block (MAB) from Set Transformer."""
    def __init__(self, dim_Q, dim_K, dim_V, num_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=dim_V, num_heads=num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(dim_V)
        self.ff = nn.Sequential(
            nn.Linear(dim_V, 4 * dim_V),
            nn.SiLU(),
            nn.Linear(4 * dim_V, dim_V)
        )
        self.ln2 = nn.LayerNorm(dim_V)

    def forward(self, Q, K):
        H, _ = self.mha(Q, K, K, need_weights=False)
        H = self.ln1(H + Q)
        out = self.ff(H)
        return self.ln2(out + H)


class InducedSetAttentionBlock(nn.Module):
    """Induced Set Attention Block (ISAB)."""
    def __init__(self, dim_in, num_heads, m_inducing):
        super().__init__()
        self.I = nn.Parameter(torch.randn(1, m_inducing, dim_in))
        self.mab1 = MultiheadAttentionBlock(dim_in, dim_in, dim_in, num_heads)
        self.mab2 = MultiheadAttentionBlock(dim_in, dim_in, dim_in, num_heads)

    def forward(self, X):
        H = self.mab1(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab2(X, H)


class PoolingByMultiheadAttention(nn.Module):
    """Pooling by Multi‑head Attention (PMA) for set aggregation."""
    def __init__(self, dim, num_heads, num_seeds=1):
        super().__init__()
        self.S = nn.Parameter(torch.randn(1, num_seeds, dim))
        self.mab = MultiheadAttentionBlock(dim, dim, dim, num_heads)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)

# -----------------------------------------------------------------------------
# Embedding Block
# -----------------------------------------------------------------------------
class ElementEmbedder(nn.Module):
    """Embeds element Mat2Vec vectors + fractional amounts."""
    def __init__(self, d_model, mat2vec_path):
        super().__init__()
        df = pd.read_csv(Path(mat2vec_path), index_col=0)
        features = df.values.astype(np.float32)  # (118, feat_dim)
        zeros = np.zeros((1, features.shape[1]), dtype=np.float32)
        weight = torch.tensor(np.concatenate([zeros, features]))
        self.cbfv = nn.Embedding.from_pretrained(weight, freeze=True)
        self.proj = nn.Linear(features.shape[1], d_model)

        self.frac_mlp = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, d_model)
        )

    def forward(self, elem_idx, frac):
        e_emb = self.proj(self.cbfv(elem_idx))
        f_emb = self.frac_mlp(frac.unsqueeze(-1))
        return e_emb + f_emb

# -----------------------------------------------------------------------------
# Set Transformer Encoder
# -----------------------------------------------------------------------------
class SetEncoder(nn.Module):
    def __init__(self, d_model=256, num_heads=4, num_layers=3, m_inducing=16):
        super().__init__()
        self.layers = nn.ModuleList([
            InducedSetAttentionBlock(d_model, num_heads, m_inducing) for _ in range(num_layers)
        ])
        self.pma = PoolingByMultiheadAttention(d_model, num_heads)

    def forward(self, X):
        for layer in self.layers:
            X = layer(X)
        return self.pma(X).squeeze(1)  # (B, d)

# -----------------------------------------------------------------------------
# Output MLP with SwiGLU‑like non‑linearity
# -----------------------------------------------------------------------------
class ResidualMLP(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 4 * d_in),
            nn.SiLU(),
            nn.Linear(4 * d_in, d_out)
        )

    def forward(self, x):
        return self.net(x)

# -----------------------------------------------------------------------------
# Full Model
# -----------------------------------------------------------------------------
class Madani(nn.Module):
    """Composition‑only property predictor with Set Transformer backbone."""
    def __init__(self, out_dims=3, d_model=256, num_heads=4, num_layers=3,
                 m_inducing=16, mat2vec_path="madani/data/element_properties/mat2vec.csv",
                 compute_device=None):
        super().__init__()
        self.out_dims = out_dims
        self.d_model = d_model
        self.N = num_layers
        self.heads = num_heads
        self.compute_device = compute_device or "cpu"

        self.embed = ElementEmbedder(d_model, mat2vec_path)
        self.encoder = SetEncoder(d_model, num_heads, num_layers, m_inducing)
        self.proj = ResidualMLP(d_model, out_dims)

    def forward(self, elem_idx, frac):
        X = self.embed(elem_idx, frac)
        mask = (frac == 0)
        X = X.masked_fill(mask.unsqueeze(-1), 0)
        z = self.encoder(X)
        return self.proj(z)

# -----------------------------------------------------------------------------
# Example usage
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    B, T = 2, 6
    elem = torch.randint(0, 119, (B, T))
    frac = F.softmax(torch.randn(B, T), dim=-1)
    model = SetMadani()
    print(model.out_dims, model.d_model, model.N, model.heads)
    y = model(elem, frac)
    print("Output shape:", y.shape)
