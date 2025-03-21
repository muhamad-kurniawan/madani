import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

# Set random seeds for reproducibility
RNG_SEED = 42
torch.manual_seed(RNG_SEED)
np.random.seed(RNG_SEED)
data_type_torch = torch.float32

# ----------------------------
# Residual Network Module
# ----------------------------
class ResidualNetwork(nn.Module):
    """
    Feed-forward Residual Neural Network (inspired by Roost).
    """
    def __init__(self, input_dim, output_dim, hidden_layer_dims):
        super(ResidualNetwork, self).__init__()
        dims = [input_dim] + hidden_layer_dims
        self.fcs = nn.ModuleList([nn.Linear(dims[i], dims[i+1])
                                  for i in range(len(dims)-1)])
        self.res_fcs = nn.ModuleList([
            nn.Linear(dims[i], dims[i+1], bias=False) if dims[i] != dims[i+1] else nn.Identity()
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
    """
    Embeds element information using pre-computed element properties.
    """
    def __init__(self, d_model, compute_device=None):
        super().__init__()
        self.d_model = d_model
        self.compute_device = compute_device

        # Load element property data; change the path or file if needed.
        elem_dir = 'madani/data/element_properties'
        # For example, use mat2vec embeddings (CSV file with element properties)
        mat2vec = f'{elem_dir}/mat2vec.csv'
        cbfv = pd.read_csv(mat2vec, index_col=0).values  # shape: (num_elements, feat_size)
        feat_size = cbfv.shape[-1]
        self.fc_mat2vec = nn.Linear(feat_size, d_model)
        # Create an embedding lookup table (with a zero-vector prepended)
        zeros = np.zeros((1, feat_size))
        cat_array = np.concatenate([zeros, cbfv])
        cat_array = torch.as_tensor(cat_array, dtype=data_type_torch)
        self.cbfv = nn.Embedding.from_pretrained(cat_array)

    def forward(self, src):
        # src: (batch, set_size) indices for elements (e.g., atomic numbers)
        mat2vec_emb = self.cbfv(src)  # (batch, set_size, feat_size)
        x_emb = self.fc_mat2vec(mat2vec_emb)  # (batch, set_size, d_model)
        return x_emb

# ----------------------------
# Fractional Encoder Module
# ----------------------------
class FractionalEncoder(nn.Module):
    """
    Fractional encoding inspired by the positional encoding in Transformers.
    Maps fractional amounts to a sinusoidal encoding.
    """
    def __init__(self, d_model, resolution=100, log10=False, compute_device=None):
        super().__init__()
        self.d_model = d_model // 2  # using half the dimension
        self.resolution = resolution
        self.log10 = log10
        self.compute_device = compute_device

        # Create a fixed encoding matrix (pe) and register it as a buffer
        x = torch.linspace(0, self.resolution - 1, self.resolution).view(self.resolution, 1)
        fraction = torch.linspace(0, self.d_model - 1, self.d_model).view(1, self.d_model).repeat(self.resolution, 1)
        pe = torch.zeros(self.resolution, self.d_model)
        pe[:, 0::2] = torch.sin(x / torch.pow(50, 2 * fraction[:, 0::2] / self.d_model))
        pe[:, 1::2] = torch.cos(x / torch.pow(50, 2 * fraction[:, 1::2] / self.d_model))
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x is expected to be a tensor with fractional amounts in [0,1]
        if self.log10:
            x = 0.0025 * (torch.log2(x))**2
            x = torch.clamp(x, max=1)
        x = torch.clamp(x, min=1/self.resolution)
        # Convert fractional amounts to indices
        frac_idx = torch.round(x * self.resolution).to(dtype=torch.long) - 1
        out = self.pe[frac_idx]
        return out

# ----------------------------
# Set Transformer Components
# ----------------------------
class SAB(nn.Module):
    """
    Self-Attention Block from the Set Transformer.
    Expects input of shape (batch, set_size, d_model).
    """
    def __init__(self, d_model, n_heads, d_ff):
        super(SAB, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)
    
    def forward(self, X):
        # X: (batch, set_size, d_model) -> transpose to (set_size, batch, d_model)
        X_t = X.transpose(0, 1)
        attn_output, _ = self.mha(X_t, X_t, X_t)
        X_t = self.ln1(X_t + attn_output)
        X = X_t.transpose(0, 1)  # back to (batch, set_size, d_model)
        ff_out = self.ff(X)
        X = self.ln2(X + ff_out)
        return X

class PMA(nn.Module):
    """
    Pooling by Multihead Attention: aggregates a set into a fixed-size representation.
    """
    def __init__(self, d_model, n_heads, num_seeds):
        super(PMA, self).__init__()
        self.num_seeds = num_seeds
        # Learnable seed vectors
        self.seed = nn.Parameter(torch.randn(num_seeds, d_model))
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads)
        self.ln = nn.LayerNorm(d_model)
    
    def forward(self, X):
        # X: (batch, set_size, d_model)
        batch_size = X.size(0)
        # Expand seeds: (num_seeds, batch, d_model)
        seeds = self.seed.unsqueeze(1).repeat(1, batch_size, 1)
        X_t = X.transpose(0, 1)  # (set_size, batch, d_model)
        attn_output, _ = self.mha(seeds, X_t, X_t)
        attn_output = self.ln(attn_output)
        # Transpose back to (batch, num_seeds, d_model)
        return attn_output.transpose(0, 1)

class SetTransformerEncoder(nn.Module):
    """
    A simple Set Transformer Encoder: multiple SAB layers followed by PMA pooling.
    """
    def __init__(self, d_model, n_heads, d_ff, num_SAB=2, num_seeds=1):
        super(SetTransformerEncoder, self).__init__()
        self.sab_layers = nn.ModuleList([SAB(d_model, n_heads, d_ff) for _ in range(num_SAB)])
        self.pma = PMA(d_model, n_heads, num_seeds)
    
    def forward(self, X):
        for sab in self.sab_layers:
            X = sab(X)
        pooled = self.pma(X)  # (batch, num_seeds, d_model)
        # If using one seed, squeeze the dimension to get (batch, d_model)
        if pooled.size(1) == 1:
            pooled = pooled.squeeze(1)
        return pooled

# ----------------------------
# Set Transformer Encoder for CrabNet
# ----------------------------
class SetTransformerCrabEncoder(nn.Module):
    """
    Encoder module for CrabNet that uses a Set Transformer to produce an
    aggregated, permutation-invariant representation.
    """
    def __init__(self, d_model, n_heads, d_ff, num_SAB=2, num_seeds=1, compute_device=None):
        super(SetTransformerCrabEncoder, self).__init__()
        self.d_model = d_model
        self.compute_device = compute_device
        # Embedding module for element features
        self.embed = Embedder(d_model=d_model, compute_device=compute_device)
        # Fractional encoding (for element fractions)
        self.pe = FractionalEncoder(d_model, resolution=5000, log10=False, compute_device=compute_device)
        self.ple = FractionalEncoder(d_model, resolution=5000, log10=True, compute_device=compute_device)
        self.emb_scaler = nn.Parameter(torch.tensor([1.]))
        self.pos_scaler = nn.Parameter(torch.tensor([1.]))
        self.pos_scaler_log = nn.Parameter(torch.tensor([1.]))
        # Set Transformer encoder replaces the standard Transformer
        self.set_transformer = SetTransformerEncoder(d_model, n_heads, d_ff, num_SAB=num_SAB, num_seeds=num_seeds)

    def forward(self, src, frac):
        # src: (batch, set_size) element indices
        # frac: (batch, set_size, 1) fractional amounts for each element
        x = self.embed(src) * 2 ** self.emb_scaler  # (batch, set_size, d_model)
        
        # Compute fractional encodings
        pe = torch.zeros_like(x)
        ple = torch.zeros_like(x)
        pe_scaler = 2 ** (1 - self.pos_scaler) ** 2
        ple_scaler = 2 ** (1 - self.pos_scaler_log) ** 2
        pe[:, :, :self.d_model // 2] = self.pe(frac) * pe_scaler
        ple[:, :, self.d_model // 2:] = self.ple(frac) * ple_scaler
        
        # Combine element embeddings with fractional encodings
        x = x + pe + ple  # (batch, set_size, d_model)
        # Pass through the Set Transformer encoder (permutation invariant)
        x = self.set_transformer(x)  # (batch, d_model)
        return x

# ----------------------------
# CrabNet Model with Set Transformer
# ----------------------------
class CrabNetSetTransformer(nn.Module):
    def __init__(self,
                 out_dims=3,
                 d_model=512,
                 n_heads=4,
                 d_ff=2048,
                 num_SAB=2,
                 num_seeds=1,
                 compute_device=None,
                 residual_nn='roost'):
        super(CrabNetSetTransformer, self).__init__()
        self.out_dims = out_dims
        self.d_model = d_model
        self.compute_device = compute_device
        self.n_heads=n_heads
        # Use the new Set Transformer–based encoder
        self.encoder = SetTransformerCrabEncoder(d_model=d_model,
                                                 n_heads=n_heads,
                                                 d_ff=d_ff,
                                                 num_SAB=num_SAB,
                                                 num_seeds=num_seeds,
                                                 compute_device=compute_device)
        # Choose between a Roost-style or simpler residual network for final prediction
        if residual_nn == 'roost':
            self.out_hidden = [1024, 512, 256, 128]
            self.output_nn = ResidualNetwork(d_model, out_dims, self.out_hidden)
        else:
            self.out_hidden = [256, 128]
            self.output_nn = ResidualNetwork(d_model, out_dims, self.out_hidden)

    def forward(self, src, frac):
        # src: (batch, set_size) element indices
        # frac: (batch, set_size, 1) fractional amounts
        x = self.encoder(src, frac)  # (batch, d_model)
        output = self.output_nn(x)   # (batch, out_dims)
        return output

# ----------------------------
# Example Usage
# ----------------------------
if __name__ == '__main__':
    # Example dimensions and dummy data
    batch_size = 4
    set_size = 5  # e.g., each material sample has 1 to 5 elements
    vocab_size = 100  # assume 100 possible element types (index 0 reserved)
    d_model = 128  # lower d_model for demonstration
    out_dims = 3

    # Create dummy element indices (simulate material composition)
    src = torch.randint(1, vocab_size, (batch_size, set_size))
    # Create dummy fractional amounts (e.g., fraction of each element), shape: (batch, set_size, 1)
    frac = torch.rand(batch_size, set_size, 1)

    # Instantiate the model
    model = CrabNetSetTransformer(out_dims=out_dims,
                                  d_model=d_model,
                                  n_heads=4,
                                  d_ff=256,
                                  num_SAB=2,
                                  num_seeds=1,
                                  compute_device='cpu',
                                  residual_nn='roost')
    # Forward pass
    output = model(src, frac)
    print("Model output shape:", output.shape)  # Expected: (batch_size, out_dims)
