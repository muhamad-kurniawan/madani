import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Helper Blocks for Set Transformer
# -----------------------------------------------------------------------------
class MultiheadAttentionBlock(nn.Module):
    """ Multihead Attention Block (MAB) from Set Transformer. """
    def __init__(self, dim_Q, dim_K, dim_V, num_heads):
        super().__init__()
        self.dim_V = dim_V
        self.mha = nn.MultiheadAttention(embed_dim=dim_V, num_heads=num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(dim_V)
        self.ff = nn.Sequential(
            nn.Linear(dim_V, 4 * dim_V),
            nn.SiLU(),  # SiLU ~ Swish
            nn.Linear(4 * dim_V, dim_V)
        )
        self.ln2 = nn.LayerNorm(dim_V)

    def forward(self, Q, K):
        # Standard multi‑head attention followed by SwiGLU‑like FFN
        H, _ = self.mha(Q, K, K, need_weights=False)
        H = self.ln1(H + Q)
        out = self.ff(H)
        out = self.ln2(out + H)
        return out


class InducedSetAttentionBlock(nn.Module):
    """ Induced Set Attention Block (ISAB) with m inducing points. """
    def __init__(self, dim_in, num_heads, m_inducing):
        super().__init__()
        self.I = nn.Parameter(torch.randn(1, m_inducing, dim_in))
        self.mab1 = MultiheadAttentionBlock(dim_in, dim_in, dim_in, num_heads)
        self.mab2 = MultiheadAttentionBlock(dim_in, dim_in, dim_in, num_heads)

    def forward(self, X):
        H = self.mab1(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab2(X, H)


class PoolingByMultiheadAttention(nn.Module):
    """ PMA block from Set Transformer for order‑invariant pooling. """
    def __init__(self, dim, num_heads, num_seeds=1):
        super().__init__()
        self.S = nn.Parameter(torch.randn(1, num_seeds, dim))
        self.mab = MultiheadAttentionBlock(dim, dim, dim, num_heads)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)

# -----------------------------------------------------------------------------
# Embedding blocks (reuse from original implementation with minor tweaks)
# -----------------------------------------------------------------------------
class ElementEmbedder(nn.Module):
    """ Embeds element Mat2Vec + fraction via small MLP. """
    def __init__(self, d_model, mat2vec_path):
        super().__init__()
        cbfv = np.loadtxt(mat2vec_path, delimiter=",", skiprows=1)  # expects no header after first col
        zeros = np.zeros((1, cbfv.shape[1]))
        cat = np.concatenate([zeros, cbfv])
        self.cbfv = nn.Embedding.from_pretrained(torch.tensor(cat, dtype=torch.float32), freeze=True)
        self.proj = nn.Linear(cbfv.shape[1], d_model)

        # small MLP for fractional amount
        self.frac_mlp = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, d_model)
        )

    def forward(self, elem_idx, frac):
        e_emb = self.proj(self.cbfv(elem_idx))                # [B, T, d]
        f_emb = self.frac_mlp(frac.unsqueeze(-1))             # [B, T, d]
        return e_emb + f_emb

# -----------------------------------------------------------------------------
# Main Set‑Transformer based encoder
# -----------------------------------------------------------------------------
class SetEncoder(nn.Module):
    def __init__(self, d_model=256, num_heads=4, num_layers=3, m_inducing=16):
        super().__init__()
        self.layers = nn.ModuleList([
            InducedSetAttentionBlock(d_model, num_heads, m_inducing) for _ in range(num_layers)
        ])
        self.pma = PoolingByMultiheadAttention(d_model, num_heads, num_seeds=1)

    def forward(self, X):
        for layer in self.layers:
            X = layer(X)
        Z = self.pma(X)         # [B, 1, d]
        return Z.squeeze(1)     # [B, d]

# -----------------------------------------------------------------------------
# Residual projector with SwiGLU (Shazeer 2020)
# -----------------------------------------------------------------------------
class ResidualMLP(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 4 * d_in),
            nn.SiLU(),  # approximates SwiGLU when paired with gate below
            nn.Linear(4 * d_in, d_out)
        )

    def forward(self, x):
        return self.net(x)

# -----------------------------------------------------------------------------
# Full Model
# -----------------------------------------------------------------------------
class Madani(nn.Module):
    """ Composition‑only predictor using Set Transformer backbone. """
    def __init__(self, out_dims=3, d_model=256, num_heads=4, num_layers=3,
                 m_inducing=16, mat2vec_path="madani/data/element_properties/mat2vec.csv"):
        super().__init__()
        self.embed = ElementEmbedder(d_model, mat2vec_path)
        self.encoder = SetEncoder(d_model, num_heads, num_layers, m_inducing)
        self.proj = ResidualMLP(d_model, out_dims)

    def forward(self, elem_idx, frac):
        X = self.embed(elem_idx, frac)        # [B, T, d]
        mask = (frac == 0)
        X = X.masked_fill(mask.unsqueeze(-1), 0)
        Z = self.encoder(X)                   # [B, d]
        out = self.proj(Z)
        return out

# -----------------------------------------------------------------------------
# Example usage
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    B, T = 4, 8
    dummy_elem = torch.randint(0, 119, (B, T))
    dummy_frac = F.softmax(torch.randn(B, T), dim=-1)
    model = SetMadani()
    y = model(dummy_elem, dummy_frac)
    print(y.shape)  # Expected [B, out_dims]
