import numpy as np
import pandas as pd

import torch
from torch import nn


# %%
RNG_SEED = 42
torch.manual_seed(RNG_SEED)
np.random.seed(RNG_SEED)
data_type_torch = torch.float32


# %%
class ResidualNetwork(nn.Module):
    """
    Feed forward Residual Neural Network as seen in Roost.
    https://doi.org/10.1038/s41467-020-19964-7
    """

    def __init__(self, input_dim, output_dim, hidden_layer_dims):
        """
        Inputs
        ----------
        input_dim: int
        output_dim: int
        hidden_layer_dims: list(int)
        """
        super(ResidualNetwork, self).__init__()
        dims = [input_dim]+hidden_layer_dims
        self.fcs = nn.ModuleList([nn.Linear(dims[i], dims[i+1])
                                  for i in range(len(dims)-1)])
        self.res_fcs = nn.ModuleList([nn.Linear(dims[i], dims[i+1], bias=False)
                                      if (dims[i] != dims[i+1])
                                      else nn.Identity()
                                      for i in range(len(dims)-1)])
        self.acts = nn.ModuleList([nn.LeakyReLU() for _ in range(len(dims)-1)])
        self.fc_out = nn.Linear(dims[-1], output_dim)

    def forward(self, fea):
        for fc, res_fc, act in zip(self.fcs, self.res_fcs, self.acts):
            fea = act(fc(fea))+res_fc(fea)
        return self.fc_out(fea)

    def __repr__(self):
        return f'{self.__class__.__name__}'


# %%
class Embedder(nn.Module):
    def __init__(self,
                 d_model,
                 compute_device=None):
        super().__init__()
        self.d_model = d_model
        self.compute_device = compute_device

        elem_dir = 'madani/data/element_properties'
        # # Choose what element information the model receives
        mat2vec = f'{elem_dir}/mat2vec.csv'  # element embedding
        # mat2vec = f'{elem_dir}/onehot.csv'  # onehot encoding (atomic number)
        # mat2vec = f'{elem_dir}/random_200.csv'  # random vec for elements
        # mat2vec = f'{elem_dir}/oliynyk.csv'  # random vec for elements
        # mat2vec = f'{elem_dir}/magpie.csv'  # random vec for elements

        cbfv = pd.read_csv(mat2vec, index_col=0).values
        feat_size = cbfv.shape[-1]
        self.fc_mat2vec = nn.Linear(feat_size, d_model).to(self.compute_device)
        zeros = np.zeros((1, feat_size))
        cat_array = np.concatenate([zeros, cbfv])
        cat_array = torch.as_tensor(cat_array, dtype=data_type_torch)
        self.cbfv = nn.Embedding.from_pretrained(cat_array) \
            .to(self.compute_device, dtype=data_type_torch)

    def forward(self, src):
        mat2vec_emb = self.cbfv(src)
        x_emb = self.fc_mat2vec(mat2vec_emb)
        return x_emb

import torch
import torch.nn as nn

def bspline_basis_1d(x, knots, degree):
    """
    Compute B-spline basis functions for input x (1D), using De Boor's recursion.
    x: shape [N]  (1D tensor of fractions in [0,1])
    knots: shape [n_knots]  (assumed sorted, possibly with repeats at ends)
    degree: int

    Returns: shape [N, n_basis]
        where n_basis = n_knots - (degree + 1).
    """
    n_knots = knots.shape[0]
    n_basis = n_knots - (degree + 1)  # final number of B-spline basis functions

    device = x.device
    N = x.shape[0]

    # B_prev has shape [n_knots - 1, N]
    B_prev = torch.zeros((n_knots - 1, N), dtype=x.dtype, device=device)

    # Initialize B_{i,0}(x)
    for i in range(n_knots - 1):
        left  = knots[i]
        right = knots[i + 1]
        # condition: left <= x < right
        mask = (x >= left) & (x < right)
        B_prev[i, mask] = 1.0

    # Build up recursively for d=1..degree
    for d in range(1, degree + 1):
        B_curr = torch.zeros((n_knots - d - 1, N), dtype=x.dtype, device=device)
        for i in range(n_knots - d - 1):
            denom1 = knots[i + d] - knots[i]
            denom2 = knots[i + d + 1] - knots[i + 1]

            term1, term2 = 0.0, 0.0
            if denom1 != 0:
                term1 = (x - knots[i]) / denom1 * B_prev[i]
            if denom2 != 0:
                term2 = (knots[i + d + 1] - x) / denom2 * B_prev[i + 1]
            B_curr[i] = term1 + term2

        B_prev = B_curr

    # B_prev: shape [n_basis, N] => transpose => [N, n_basis]
    B_final = B_prev.transpose(0, 1).contiguous()
    return B_final


class BSplineEncoder(nn.Module):
    """
    B-spline encoder for fractions in [0,1].
    Produces a learned embedding of size out_dim, using n_basis B-spline basis.
    """
    def __init__(self,
                 out_dim,
                 n_basis=10,   # number of final basis functions
                 degree=3,
                 log10=False,
                 compute_device=None):
        super().__init__()
        self.out_dim = out_dim
        self.n_basis = n_basis
        self.degree = degree
        self.log10 = log10
        self.compute_device = compute_device

        # n_knots = n_basis + degree + 1
        self.n_knots = n_basis + degree + 1

        # Create uniform knots in [0, 1] and clamp ends
        #   e.g. for n_basis=10, degree=3 => n_knots=14
        #   we can place interior knots uniformly, then pad with repeats at ends
        #   so we have repeated endpoints for a "clamped" B-spline
        knots_interior = torch.linspace(0, 1, self.n_knots - 2*self.degree)
        knots = torch.cat([
            knots_interior[:1].repeat(self.degree),
            knots_interior,
            knots_interior[-1:].repeat(self.degree)
        ], dim=0)
        self.register_buffer('knots', knots)

        # Projection from n_basis to out_dim
        self.out_proj = nn.Linear(self.n_basis, out_dim)

    def forward(self, x):
        """
        x: shape [B, T], entries in [0,1]
        Returns: shape [B, T, out_dim]
        """
        frac = x.clone()
        frac = torch.clamp(frac, min=1e-9, max=1.0)

        # optional log transform
        # you can adapt this if you want a different transform
        if self.log10:
            frac = 0.0025 * (torch.log2(frac))**2
            frac = torch.clamp(frac, 0.0, 1.0)

        B, T = frac.shape
        frac_1d = frac.reshape(-1)

        # compute B-spline basis => [B*T, n_basis]
        basis_1d = bspline_basis_1d(frac_1d, self.knots, self.degree)

        # project to out_dim
        out_1d = self.out_proj(basis_1d)   # [B*T, out_dim]

        # reshape to [B, T, out_dim]
        out = out_1d.view(B, T, self.out_dim)
        return out

# %%
class FractionalEncoder(nn.Module):
    """
    Encoding element fractional amount using a "fractional encoding" inspired
    by the positional encoder discussed by Vaswani.
    https://arxiv.org/abs/1706.03762
    """
    def __init__(self,
                 d_model,
                 resolution=100,
                 log10=False,
                 compute_device=None):
        super().__init__()
        self.d_model = d_model//2
        self.resolution = resolution
        self.log10 = log10
        self.compute_device = compute_device

        x = torch.linspace(0, self.resolution - 1,
                           self.resolution,
                           requires_grad=False) \
            .view(self.resolution, 1)
        fraction = torch.linspace(0, self.d_model - 1,
                                  self.d_model,
                                  requires_grad=False) \
            .view(1, self.d_model).repeat(self.resolution, 1)

        pe = torch.zeros(self.resolution, self.d_model)
        pe[:, 0::2] = torch.sin(x /torch.pow(
            50,2 * fraction[:, 0::2] / self.d_model))
        pe[:, 1::2] = torch.cos(x / torch.pow(
            50, 2 * fraction[:, 1::2] / self.d_model))
        pe = self.register_buffer('pe', pe)

    def forward(self, x):
        x = x.clone()
        if self.log10:
            x = 0.0025 * (torch.log2(x))**2
            # clamp x[x > 1] = 1
            x = torch.clamp(x, max=1)
            # x = 1 - x  # for sinusoidal encoding at x=0
        # clamp x[x < 1/self.resolution] = 1/self.resolution
        x = torch.clamp(x, min=1/self.resolution)
        frac_idx = torch.round(x * (self.resolution)).to(dtype=torch.long) - 1
        out = self.pe[frac_idx]

        return out


# %%
class Encoder(nn.Module):
    def __init__(self,
                 d_model,
                 N,
                 heads,
                 frac=False,
                 attn=True,
                 compute_device=None):
        super().__init__()
        self.d_model = d_model
        self.N = N
        self.heads = heads
        self.fractional = frac
        self.attention = attn
        self.compute_device = compute_device

        # Keep your original element embedder
        self.embed = Embedder(d_model=self.d_model,
                              compute_device=self.compute_device)

        # --- Replace or supplement the old FractionalEncoder with B-spline: ---
        # B-spline versions for "linear frac" and "log frac"
        # self.pe  = BSplineEncoder(d_model//2,
        #                                     n_basis=10,
        #                                     degree=3,
        #                                     log10=False,
        #                                     compute_device=self.compute_device)
        # self.ple = BSplineEncoder(d_model//2,
        #                                     n_basis=10,
        #                                     degree=3,
        #                                     log10=True,
        #                                     compute_device=self.compute_device)

        self.pe = FractionalEncoder(d_model//2,
                                            resolution=5000,
                                            log10=False,
                                            compute_device=self.compute_device)
        self.ple = FractionalEncoder(d_model//2,
                                            resolution=5000,
                                            log10=True,
                                            compute_device=self.compute_device)
                     
        self.emb_scaler = nn.parameter.Parameter(torch.tensor([1.]))
        self.pos_scaler = nn.parameter.Parameter(torch.tensor([1.]))
        self.pos_scaler_log = nn.parameter.Parameter(torch.tensor([1.]))

        if self.attention:
            encoder_layer = nn.TransformerEncoderLayer(self.d_model,
                                                       nhead=self.heads,
                                                       dim_feedforward=2048,
                                                       dropout=0.1)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer,
                                                             num_layers=self.N)

    def forward(self, src, frac):
        """
        src:  [B, T] containing integer-coded elements
        frac: [B, T] containing fractional amounts in [0,1]
        """
        # element embedding
        x = self.embed(src) * 2**self.emb_scaler

        # build source padding mask from frac=0 => elements that don't exist
        mask = frac.unsqueeze(dim=-1)
        mask = torch.matmul(mask, mask.transpose(-2, -1))
        mask[mask != 0] = 1
        src_mask = (mask[:, 0] != 1)

        # B-spline fractional embeddings, split half for 'pe' vs. 'ple'
        pe_scaler = 2**(1 - self.pos_scaler)**2
        ple_scaler = 2**(1 - self.pos_scaler_log)**2

        # shape [B, T, d_model//2]
        pe_embedding = self.pe(frac) * pe_scaler
        ple_embedding = self.ple(frac) * ple_scaler

        # combine them into [B, T, d_model]
        # x_src = torch.zeros_like(x)
        # x_src[:, :, :self.d_model//2] = pe_embedding
        # x_src[:, :, self.d_model//2:] = ple_embedding

        pe = torch.zero_like(x)
        ple = torch.zero_like(x)
        pe[:, :, :self.d_model//2] = pe_embedding
        ple[:, :, self.d_model//2:] = ple_embedding

        # pass through transformer (if enabled)
        if self.attention:
            x_src = x + pe + ple
            # x_src = x_src.transpose(0, 1)   # shape => [T, B, d_model]
            x = self.transformer_encoder(x_src, src_key_padding_mask=src_mask)
            x = x.transpose(0, 1)          # shape => [B, T, d_model]
        else:
            x = x_src

        # optionally multiply by frac to scale element embedding by fraction
        if self.fractional:
            x = x * frac.unsqueeze(2)

        # zero out any "dummy" element positions
        hmask = mask[:, :, 0:1].repeat(1, 1, self.d_model)
        x = x.masked_fill(hmask == 0, 0)
        return x



# %%
class CrabNet(nn.Module):
    def __init__(self,
                 out_dims=3,
                 d_model=512,
                 N=3,
                 heads=4,
                 compute_device=None,
                 residual_nn='roost'):
        super().__init__()
        self.avg = True
        self.out_dims = out_dims
        self.d_model = d_model
        self.N = N
        self.heads = heads
        self.compute_device = compute_device
        self.encoder = Encoder(d_model=self.d_model,
                               N=self.N,
                               heads=self.heads,
                               compute_device=self.compute_device)
        if residual_nn == 'roost':
            # use the Roost residual network
            self.out_hidden = [1024, 512, 256, 128]
            self.output_nn = ResidualNetwork(self.d_model,
                                             self.out_dims,
                                             self.out_hidden)
        else:
            # use a simpler residual network
            self.out_hidden = [256, 128]
            self.output_nn = ResidualNetwork(self.d_model,
                                             self.out_dims,
                                             self.out_hidden)

    def forward(self, src, frac):
        output = self.encoder(src, frac)

        # average the "element contribution" at the end
        # mask so you only average "elements"
        mask = (src == 0).unsqueeze(-1).repeat(1, 1, self.out_dims)
        output = self.output_nn(output)  # simple linear
        if self.avg:
            output = output.masked_fill(mask, 0)
            output = output.sum(dim=1)/(~mask).sum(dim=1)
            output, logits = output.chunk(2, dim=-1)
            probability = torch.ones_like(output)
            probability[:, :logits.shape[-1]] = torch.sigmoid(logits)
            output = output * probability

        return output


# %%
if __name__ == '__main__':
    model = CrabNet()
