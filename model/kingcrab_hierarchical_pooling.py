import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

# ----------------------------
# Set random seeds for reproducibility
# ----------------------------
RNG_SEED = 42
torch.manual_seed(RNG_SEED)
np.random.seed(RNG_SEED)
data_type_torch = torch.float32

# ----------------------------
# Residual Network Module
# ----------------------------
class ResidualNetwork(nn.Module):
    """
    Feed-forward Residual Neural Network as seen in Roost.
    https://doi.org/10.1038/s41467-020-19964-7
    """
    def __init__(self, input_dim, output_dim, hidden_layer_dims):
        """
        input_dim: int, output_dim: int, hidden_layer_dims: list(int)
        """
        super(ResidualNetwork, self).__init__()
        dims = [input_dim] + hidden_layer_dims
        self.fcs = nn.ModuleList([nn.Linear(dims[i], dims[i+1])
                                  for i in range(len(dims)-1)])
        self.res_fcs = nn.ModuleList([
            nn.Linear(dims[i], dims[i+1], bias=False)
            if dims[i] != dims[i+1] else nn.Identity()
            for i in range(len(dims)-1)
        ])
        self.acts = nn.ModuleList([nn.LeakyReLU() for _ in range(len(dims)-1)])
        self.fc_out = nn.Linear(dims[-1], output_dim)

    def forward(self, fea):
        for fc, res_fc, act in zip(self.fcs, self.res_fcs, self.acts):
            fea = act(fc(fea)) + res_fc(fea)
        return self.fc_out(fea)

# ----------------------------
# Embedder Module
# ----------------------------
class Embedder(nn.Module):
    def __init__(self, d_model, compute_device=None):
        super().__init__()
        self.d_model = d_model
        self.compute_device = compute_device

        elem_dir = 'madani/data/element_properties'
        # Choose what element information the model receives:
        mat2vec = f'{elem_dir}/mat2vec.csv'  # element embedding
        cbfv = pd.read_csv(mat2vec, index_col=0).values
        feat_size = cbfv.shape[-1]
        self.fc_mat2vec = nn.Linear(feat_size, d_model).to(self.compute_device)
        zeros = np.zeros((1, feat_size))
        cat_array = np.concatenate([zeros, cbfv])
        cat_array = torch.as_tensor(cat_array, dtype=data_type_torch)
        self.cbfv = nn.Embedding.from_pretrained(cat_array)\
            .to(self.compute_device, dtype=data_type_torch)

    def forward(self, src):
        # src: (batch, set_size) element indices; index 0 is assumed padding.
        mat2vec_emb = self.cbfv(src)
        x_emb = self.fc_mat2vec(mat2vec_emb)
        return x_emb

# ----------------------------
# Fractional Encoder Module
# ----------------------------
class FractionalEncoder(nn.Module):
    """
    Encoding element fractional amount using a "fractional encoding" inspired
    by the positional encoder from Vaswani et al.
    https://arxiv.org/abs/1706.03762
    """
    def __init__(self, d_model, resolution=100, log10=False, compute_device=None):
        super().__init__()
        self.d_model = d_model // 2
        self.resolution = resolution
        self.log10 = log10
        self.compute_device = compute_device

        x = torch.linspace(0, self.resolution - 1, self.resolution, requires_grad=False).view(self.resolution, 1)
        fraction = torch.linspace(0, self.d_model - 1, self.d_model, requires_grad=False).view(1, self.d_model).repeat(self.resolution, 1)
        pe = torch.zeros(self.resolution, self.d_model)
        pe[:, 0::2] = torch.sin(x / torch.pow(50, 2 * fraction[:, 0::2] / self.d_model))
        pe[:, 1::2] = torch.cos(x / torch.pow(50, 2 * fraction[:, 1::2] / self.d_model))
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, set_size, 1) fractional amounts.
        x = x.clone()
        if self.log10:
            x = 0.0025 * (torch.log2(x))**2
            x = torch.clamp(x, max=1)
        x = torch.clamp(x, min=1/self.resolution)
        frac_idx = torch.round(x * self.resolution).to(dtype=torch.long) - 1
        out = self.pe[frac_idx]
        return out

# ----------------------------
# PMA Block (Pooling by Multihead Attention)
# ----------------------------
class PMA(nn.Module):
    """
    Pooling by Multihead Attention block.
    Aggregates a set into a fixed-size output.
    If num_seeds=1, the output is a single vector per example.
    """
    def __init__(self, d_model, n_heads, num_seeds):
        super(PMA, self).__init__()
        self.num_seeds = num_seeds
        # Learnable seed vectors: (num_seeds, d_model)
        self.seed = nn.Parameter(torch.randn(num_seeds, d_model))
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads)
        self.ln
