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
        # Choose what element information the model receives
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
        # src: (batch, set_size) indices for elements
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
        self.ln = nn.LayerNorm(d_model)

    def forward(self, X):
        # X: (batch, set_size, d_model)
        batch_size = X.size(0)
        # Expand seeds to (num_seeds, batch, d_model)
        seeds = self.seed.unsqueeze(1).repeat(1, batch_size, 1)
        # Rearrange X to (set_size, batch, d_model)
        X_t = X.transpose(0, 1)
        attn_output, _ = self.mha(seeds, X_t, X_t)
        attn_output = self.ln(attn_output)
        # Transpose back to (batch, num_seeds, d_model)
        return attn_output.transpose(0, 1)

# ----------------------------
# Encoder Module with PMA Block
# ----------------------------
class Encoder(nn.Module):
    def __init__(self, d_model, N, heads, frac=False, attn=True, compute_device=None, use_pma=True):
        """
        d_model: model dimension,
        N: number of transformer layers,
        heads: number of attention heads,
        frac: whether to apply fractional scaling,
        attn: whether to use attention layers,
        use_pma: if True, use PMA to pool set elements.
        """
        super().__init__()
        self.d_model = d_model
        self.N = N
        self.heads = heads
        self.fractional = frac
        self.attention = attn
        self.compute_device = compute_device
        self.use_pma = use_pma

        self.embed = Embedder(d_model=self.d_model, compute_device=self.compute_device)
        self.pe = FractionalEncoder(self.d_model, resolution=5000, log10=False, compute_device=self.compute_device)
        self.ple = FractionalEncoder(self.d_model, resolution=5000, log10=True, compute_device=self.compute_device)

        self.emb_scaler = nn.Parameter(torch.tensor([1.]))
        self.pos_scaler = nn.Parameter(torch.tensor([1.]))
        self.pos_scaler_log = nn.Parameter(torch.tensor([1.]))

        if self.attention:
            encoder_layer = nn.TransformerEncoderLayer(self.d_model,
                                                       nhead=self.heads,
                                                       dim_feedforward=2048,
                                                       dropout=0.1)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.N)

        # Instantiate the PMA block for pooling across set elements.
        # Here we use one seed vector so the output is (batch, 1, d_model)
        if self.use_pma:
            self.pma = PMA(self.d_model, self.heads, num_seeds=1)

    def forward(self, src, frac):
        """
        src: (batch, set_size) element indices
        frac: (batch, set_size, 1) fractional amounts
        """
        x = self.embed(src) * 2 ** self.emb_scaler
        mask = frac.unsqueeze(dim=-1)
        mask = torch.matmul(mask, mask.transpose(-2, -1))
        mask[mask != 0] = 1
        src_mask = mask[:, 0] != 1

        pe = torch.zeros_like(x)
        ple = torch.zeros_like(x)
        pe_scaler = 2 ** (1 - self.pos_scaler) ** 2
        ple_scaler = 2 ** (1 - self.pos_scaler_log) ** 2
        pe[:, :, :self.d_model // 2] = self.pe(frac) * pe_scaler
        ple[:, :, self.d_model // 2:] = self.ple(frac) * ple_scaler

        if self.attention:
            x_src = x + pe + ple
            x_src = x_src.transpose(0, 1)
            x = self.transformer_encoder(x_src, src_key_padding_mask=src_mask)
            x = x.transpose(0, 1)

        if self.fractional:
            x = x * frac.unsqueeze(2).repeat(1, 1, self.d_model)

        hmask = mask[:, :, 0:1].repeat(1, 1, self.d_model)
        if mask is not None:
            x = x.masked_fill(hmask == 0, 0)

        # Use the PMA block to pool set elements into a single aggregated representation.
        if self.use_pma:
            x = self.pma(x)  # (batch, 1, d_model)
            x = x.squeeze(1)  # now (batch, d_model)
        return x

# ----------------------------
# CrabNet Model
# ----------------------------
class CrabNet(nn.Module):
    def __init__(self,
                 out_dims=3,
                 d_model=512,
                 N=3,
                 heads=4,
                 compute_device=None,
                 residual_nn='roost'):
        super().__init__()
        self.avg = False  # With PMA, we already aggregate the set.
        self.out_dims = out_dims
        self.d_model = d_model
        self.N = N
        self.heads = heads
        self.compute_device = compute_device
        self.encoder = Encoder(d_model=self.d_model,
                               N=self.N,
                               heads=self.heads,
                               compute_device=self.compute_device,
                               use_pma=True)
        if residual_nn == 'roost':
            self.out_hidden = [1024, 512, 256, 128]
            self.output_nn = ResidualNetwork(self.d_model,
                                             self.out_dims,
                                             self.out_hidden)
        else:
            self.out_hidden = [256, 128]
            self.output_nn = ResidualNetwork(self.d_model,
                                             self.out_dims,
                                             self.out_hidden)

    def forward(self, src, frac):
        # src: (batch, set_size)
        # frac: (batch, set_size, 1)
        x = self.encoder(src, frac)  # (batch, d_model) after PMA aggregation
        output = self.output_nn(x)   # (batch, out_dims)
        # For a Roost-like model, output may be split into prediction and uncertainty.
        # Here we assume output is 2 * target_dim. Adjust as needed.
        output, logits = output.chunk(2, dim=-1)
        probability = torch.sigmoid(logits)
        output = output * probability
        return output

# ----------------------------
# Example Usage
# ----------------------------
if __name__ == '__main__':
    # Example parameters
    batch_size = 4
    set_size = 5  # Each material sample has 1 to 5 elements.
    vocab_size = 100  # Assume 100 possible element types (0 is reserved)
    d_model = 128  # For demonstration purposes
    out_dims = 1024  # Must be even; e.g., for splitting into two halves of 512

    # Create dummy element indices and fractional amounts
    src = torch.randint(1, vocab_size, (batch_size, set_size))
    frac = torch.rand(batch_size, set_size, 1)

    # Instantiate the model
    model = CrabNet(out_dims=out_dims, d_model=d_model, N=3, heads=4, compute_device='cpu', residual_nn='roost')
    output = model(src, frac)
    print("Model output shape:", output.shape)  # Expected: (batch_size, out_dims/2) e.g. (4, 512)
