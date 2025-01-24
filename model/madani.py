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


# %%
class FractionalEncoder(nn.Module):
    """
    Fractional Encoder with soft transitions using interpolation.
    Inspired by positional encoders, but designed for smooth fractional encoding.
    """
    def __init__(self, d_model, resolution=100, log10=False, compute_device=None):
        super().__init__()
        self.d_model = d_model // 2
        self.resolution = resolution
        self.log10 = log10
        self.compute_device = compute_device

        # Create a grid for the encoding
        x = torch.linspace(0, self.resolution - 1, self.resolution, requires_grad=False).view(self.resolution, 1)
        fraction = torch.linspace(0, self.d_model - 1, self.d_model, requires_grad=False).view(1, self.d_model).repeat(self.resolution, 1)

        # Compute the sinusoidal and cosine encoding
        pe = torch.zeros(self.resolution, self.d_model)
        pe[:, 0::2] = torch.sin(x / torch.pow(50, 2 * fraction[:, 0::2] / self.d_model))
        pe[:, 1::2] = torch.cos(x / torch.pow(50, 2 * fraction[:, 1::2] / self.d_model))

        # Register buffer for encoding table
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Clone the input tensor to avoid modifications
        x = x.clone()

        # Apply log transformation if enabled
        if self.log10:
            x = 0.0025 * (torch.log2(x)) ** 2
            x = torch.clamp(x, max=1)  # Clamp values to [0, 1]
        x = torch.clamp(x, min=1 / self.resolution)  # Ensure values >= 1/resolution

        # Scale input to the range [0, resolution)
        x_scaled = x * (self.resolution - 1)

        # Get lower and upper indices for interpolation
        idx_lower = torch.floor(x_scaled).to(dtype=torch.long)
        idx_upper = torch.ceil(x_scaled).to(dtype=torch.long)
        idx_upper = torch.clamp(idx_upper, max=self.resolution - 1)

        # Compute interpolation weights
        weight_upper = x_scaled - idx_lower
        weight_lower = 1 - weight_upper

        # Interpolate between lower and upper encoding values
        encoding_lower = self.pe[idx_lower]
        encoding_upper = self.pe[idx_upper]
        out = weight_lower.unsqueeze(-1) * encoding_lower + weight_upper.unsqueeze(-1) * encoding_upper

        return out

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

def bspline_basis_1d(x, knots, degree):
    """
    Compute B-spline basis functions for input x (1D), using De Boor's recursion.
    x: shape [N]  (1D tensor of fractions in [0,1])
    knots: shape [n_knots]  (assumed sorted, possibly with repeats at ends)
    degree: int

    Returns: shape [N, n_basis]
        where n_basis = n_knots - (degree + 1).
    """
    # We assume:
    #   n_knots = n_basis + degree + 1
    # Initialize 'B' for the 0th-degree piecewise constant basis:
    n_knots = knots.shape[0]
    n_basis = n_knots - (degree + 1)  # final number of B-spline basis functions

    # B[d, i, n] will store B_{i,d}(x_n)
    # We'll build up from d=0..degree
    # For efficiency, store only 2 layers at a time
    device = x.device
    N = x.shape[0]

    # shape for B: [n_knots-1, N] at d=0
    # (since B_{i,0} is nonzero only in [t_i, t_{i+1}))
    B_prev = torch.zeros((n_knots-1, N), dtype=x.dtype, device=device)

    # Initialize B_{i,0}(x)
    # For each i, B_{i,0}(x)=1 if knots[i] <= x < knots[i+1], else 0
    # We'll do a loop or broadcast logic
    for i in range(n_knots - 1):
        left  = knots[i]
        right = knots[i+1]
        # condition: left <= x < right
        mask = (x >= left) & (x < right)
        # For the last knot interval, we can include equality at the end
        # or handle boundary carefully if you want continuous coverage at 1.0
        B_prev[i, mask] = 1.0

    # Now recursively build up to higher degrees
    for d in range(1, degree+1):
        # new array B_curr for B_{i,d}
        B_curr = torch.zeros((n_knots - d - 1, N), dtype=x.dtype, device=device)

        for i in range(n_knots - d - 1):
            denom1 = knots[i + d] - knots[i]
            denom2 = knots[i + d + 1] - knots[i + 1]

            term1 = 0.0
            term2 = 0.0

            if denom1 != 0:
                term1 = (x - knots[i]) / denom1 * B_prev[i]
            if denom2 != 0:
                term2 = (knots[i + d + 1] - x) / denom2 * B_prev[i + 1]

            B_curr[i] = term1 + term2

        B_prev = B_curr  # move up one degree

    # After d=degree, B_prev has shape [n_basis, N]
    # Transpose to [N, n_basis] for convenience
    B_final = B_prev.transpose(0, 1).contiguous()  # [N, n_basis]
    return B_final

class BSplineEncoder(nn.Module):
    """
    B-spline encoder for fractions in [0,1].
    1) We'll define uniform knots with optional repeated endpoints.
    2) Compute B-spline basis via De Boor recursion for each x in [B,T].
    3) Project to a final dimension 'out_dim' with a linear layer.
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

        # We define n_knots = n_basis + degree + 1
        self.n_knots = self.n_basis + self.degree + 1

        # Create uniform knots in [0, 1].
        # For B-splines, often we "clamp" the endpoints by repeating them degree times.
        # For simplicity, let's do a naive approach:
        #   knots = [0,0,0,...(degree times)..., 1,1,1...]
        # or uniform. We'll do uniform plus repeats at the ends:
        knots = torch.linspace(0, 1, self.n_knots - 2*self.degree)
        # expand by repeating the first knot `degree` times and last knot `degree` times
        knots = torch.cat([
            knots[:1].repeat(self.degree), 
            knots,
            knots[-1:].repeat(self.degree)
        ], dim=0)
        # Now we have n_knots total
        # Register as a buffer (non-trainable) or as a param if you want:
        self.register_buffer('knots', knots)

        # Linear projection from n_basis -> out_dim
        self.out_proj = nn.Linear(self.n_basis, out_dim)

    def forward(self, x):
        """
        x: shape [B, T], fractions in [0,1].
        returns: shape [B, T, out_dim]
        """
        frac = x.clone()
        frac = torch.clamp(frac, min=1e-9, max=1.0)

        # Optional log10 transform
        if self.log10:
            frac = 0.0025 * (torch.log2(frac))**2
            frac = torch.clamp(frac, 0.0, 1.0)

        B, T = frac.shape
        # Flatten to 1D for basis computation: shape [B*T]
        frac_1d = frac.view(-1)

        # Evaluate the B-spline basis -> shape [B*T, n_basis]
        basis_1d = bspline_basis_1d(frac_1d, self.knots, self.degree)

        # Project to out_dim
        out_1d = self.out_proj(basis_1d)  # [B*T, out_dim]

        # Reshape back to [B, T, out_dim]
        out = out_1d.view(B, T, self.out_dim)
        return out



import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomMultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        # Linear layers for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim**-0.5

        # A trainable scalar for stoichiometric ALiBi-like bias
        self.stoich_alpha = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        query,                     # [B, T_q, d_model]
        key,                       # [B, T_k, d_model]
        value,                     # [B, T_k, d_model]
        key_padding_mask=None,     # [B, T_k]
        attn_mask=None,            # [T_q, T_k] or [B, T_q, T_k]
        stoich_frac=None           # [B, T_k] or [B, T_q] – stoichiometric fraction
    ):
        B, T_q, _ = query.shape
        Bk, T_k, _ = key.shape
        assert B == Bk, "Batch dimension mismatch between query and key"

        # 1) Project Q, K, V
        Q = self.W_q(query)  # [B, T_q, d_model]
        K = self.W_k(key)    # [B, T_k, d_model]
        V = self.W_v(value)  # [B, T_k, d_model]

        # 2) Reshape to [B, nhead, T, head_dim]
        Q = Q.view(B, T_q, self.nhead, self.head_dim).transpose(1, 2)  # [B, nhead, T_q, head_dim]
        K = K.view(B, T_k, self.nhead, self.head_dim).transpose(1, 2)  # [B, nhead, T_k, head_dim]
        V = V.view(B, T_k, self.nhead, self.head_dim).transpose(1, 2)  # [B, nhead, T_k, head_dim]

        # 3) Compute scaled dot-product attention
        # attn_weights shape = [B, nhead, T_q, T_k]
        attn_weights = torch.matmul(Q, K.transpose(-1, -2)) * self.scale

        # ---- STOICHIOMETRIC ALiBi-LIKE BIAS ----
        if stoich_frac is not None:
            # stoich_frac has shape [B, T_k] if key/value side, or [B, T_q] if query side 
            # If T_q == T_k, and you're tying them together, you can unify them. 
            # Otherwise, adapt for separate stoich_frac for the query vs. key.
            #
            # Suppose we have a single stoich_frac for each position [B, T]
            # We'll do difference across T_q, T_k:
            # shape: [B, T_q, T_k]
            # NOTE: If you want query vs. key to share the same stoich_frac, be sure T_q == T_k
            # or else handle them carefully.
            if T_q == T_k:
                # shape [B, T]
                frac_q = stoich_frac
                frac_k = stoich_frac
            else:
                # Example if you have separate fractions for query vs. key
                frac_q = stoich_frac[0]  # [B, T_q]
                frac_k = stoich_frac[1]  # [B, T_k]
            
            # shape [B, T_q, 1] minus [B, 1, T_k] -> [B, T_q, T_k]
            # stoich_diff = frac_q.unsqueeze(-1) - frac_k.unsqueeze(1)
            stoich_diff = frac_q.unsqueeze(-1) - frac_k.unsqueeze(1)
            # stoich_diff = stoich_diff/torch.mean(stoich_diff, 0)
            # For example: ALiBi-like transform
            # log1p(|x|) * sign(x)
            stoich_bias = torch.log1p(stoich_diff.abs()) * stoich_diff.sign()
            # Multiply by stoich_alpha
            stoich_bias = self.stoich_alpha * stoich_bias
            # Expand for nhead: [B, 1, T_q, T_k]
            stoich_bias = stoich_bias.unsqueeze(1)
            # Add to attention logits
            attn_weights = attn_weights + stoich_bias
        # ----------------------------------------

        # If attn_mask is provided (e.g., subsequent mask), apply it
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask

        # key_padding_mask shape = [B, T_k], True means "ignore"
        if key_padding_mask is not None:
            expanded_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T_k]
            attn_weights = attn_weights.masked_fill(expanded_mask, float('-inf'))

        # Convert logits to probabilities
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # 4) Weighted sum over V
        out = torch.matmul(attn_probs, V)  # [B, nhead, T_q, head_dim]

        # 5) Reshape back to [B, T_q, d_model]
        out = out.transpose(1, 2).contiguous().view(B, T_q, self.d_model)

        # 6) Final linear projection
        out = self.out_proj(out)

        return out


class MoEFeedForwardTop2(nn.Module):
    """
    MoE feed-forward module with top-2 gating and optional gating noise.
    Each token's output is a weighted combination of two experts.
    """
    def __init__(self, 
                 d_model, 
                 dim_feedforward=2048, 
                 dropout=0.1, 
                 num_experts=4, 
                 gating_noise=0.0):
        """
        Args:
            d_model (int): Model embedding dimension.
            dim_feedforward (int): Hidden size in each expert's FFN.
            dropout (float): Dropout probability.
            num_experts (int): Number of parallel experts.
            gating_noise (float): Standard deviation for gating noise; 0.0 = no noise.
        """
        super().__init__()
        self.num_experts = num_experts
        self.gating_noise = gating_noise

        # Gating network: from d_model -> num_experts
        self.gate = nn.Linear(d_model, num_experts)

        # Experts: each is a small feed-forward block (d_model -> dim_feedforward -> d_model)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, d_model),
                nn.Dropout(dropout),
            ) 
            for _ in range(num_experts)
        ])

    def forward(self, x):
        """
        x: [B, T, d_model]
        Returns:
            out: [B, T, d_model]
        """
        B, T, d = x.shape

        # 1) Compute gating logits
        gating_logits = self.gate(x)  # [B, T, num_experts]

        # 2) Inject gating noise (only during training, often)
        if self.training and self.gating_noise > 0.0:
            noise = torch.randn_like(gating_logits) * self.gating_noise
            gating_logits = gating_logits + noise

        # 3) Softmax over experts
        gating_scores = F.softmax(gating_logits, dim=-1)  # [B, T, num_experts]

        # 4) Top-2 gating
        top2_scores, top2_experts = gating_scores.topk(2, dim=-1)  
        # top2_scores:   [B, T, 2] (the gating probabilities for the top 2 experts)
        # top2_experts:  [B, T, 2] (the expert indices)

        # 5) Initialize output buffer
        out = torch.zeros_like(x)

        # 6) For each of the two selected experts, gather tokens, forward them, and add weighted output
        for rank in range(2):
            # expert indices for this rank
            expert_idx = top2_experts[..., rank]    # [B, T]
            # gating weights for this rank
            gating_weight = top2_scores[..., rank]  # [B, T]

            for e_idx in range(self.num_experts):
                mask = (expert_idx == e_idx)  # [B, T] bool
                if mask.any():
                    selected_x = x[mask]  # shape [N, d_model]
                    expert_out = self.experts[e_idx](selected_x)  # [N, d_model]

                    # Multiply by gating weight
                    weight = gating_weight[mask].unsqueeze(-1)  # [N, 1]
                    weighted_out = weight * expert_out

                    # Scatter back to the output buffer
                    out[mask] += weighted_out

        return out

class CustomTransformerEncoderMoELayer(nn.Module):
    def __init__(self, 
                 d_model, 
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 layer_norm_eps=1e-5,
                 num_experts=4,
                 gating_noise=0.0):
        super().__init__()
        self.self_attn = CustomMultiHeadAttention(d_model, nhead, dropout=dropout)

        self.feed_forward = MoEFeedForwardTop2(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_experts=num_experts,
            gating_noise=gating_noise
        )

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, stoich_frac=None):
        # 1) Self-attention with stoichiometric bias
        attn_out = self.self_attn(
            query=src,
            key=src,
            value=src,
            key_padding_mask=src_key_padding_mask,
            attn_mask=src_mask,
            stoich_frac=stoich_frac  # <--- pass in stoich_frac here
        )
        src = src + self.dropout1(attn_out)
        src = self.norm1(src)

        # 2) MoE feed-forward
        ff_out = self.feed_forward(src)
        src = src + self.dropout2(ff_out)
        src = self.norm2(src)

        return src




class CustomTransformerEncoderMoE(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])

    def forward(self, src, mask=None, src_key_padding_mask=None, stoich_frac=None):
        output = src
        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                stoich_frac=stoich_frac  # pass down
            )
        return output


import pandas as pd
import numpy as np
import torch
import torch.nn as nn

# Same data_type_torch as in your code
data_type_torch = torch.float32

import torch
import torch.nn as nn

# Reuse the same fraction encoders, embedder, etc. from your original code
# Reuse the same CustomMultiHeadAttention from your previous custom attention code

class EncoderMoE(nn.Module):
    def __init__(self,
                 d_model,
                 N,
                 heads,
                 frac=False,
                 attn=True,
                 compute_device=None,
                 num_experts=4,
                 gating_noise=0.0):
        super().__init__()
        self.d_model = d_model
        self.N = N
        self.heads = heads
        self.fractional = frac
        self.attention = attn
        self.compute_device = compute_device

        # same embedder
        self.embed = Embedder(d_model=self.d_model, compute_device=self.compute_device)

        # Replace RBF with B-spline:
        self.pe = BSplineEncoder(
            out_dim=self.d_model, 
            n_basis=10,   #  how many B-spline basis
            degree=4,
            log10=False,
            compute_device=self.compute_device
        )
        # (Optionally) separate B-spline for log-scale
        self.ple = BSplineEncoder(
            out_dim=self.d_model // 2,
            n_basis=20,
            degree=3,
            log10=True,
            compute_device=self.compute_device
        )

        self.emb_scaler = nn.parameter.Parameter(torch.tensor([1.]))
        self.pos_scaler = nn.parameter.Parameter(torch.tensor([1.]))
        self.pos_scaler_log = nn.parameter.Parameter(torch.tensor([1.]))

        if self.attention:
            encoder_layer = CustomTransformerEncoderMoELayer(
                d_model=self.d_model,
                nhead=self.heads,
                dim_feedforward=1024,
                dropout=0.1,
                layer_norm_eps=1e-5,
                num_experts=num_experts,
                gating_noise=gating_noise
            )
            self.transformer_encoder = CustomTransformerEncoderMoE(
                encoder_layer=encoder_layer,
                num_layers=self.N
            )

    def forward(self, src, frac):
        x = self.embed(src) * 2.0**self.emb_scaler

        # B-spline outputs
        pe_scaler = 2.0**((1 - self.pos_scaler)**2)
        ple_scaler = 2.0**((1 - self.pos_scaler_log)**2)
        pe_out = self.pe(frac) * pe_scaler       # [B, T, d_model]
        ple_out = self.ple(frac) * ple_scaler    # [B, T, d_model//2]

        # Maybe combine them or just use one:
        pos_enc = pe_out  # or torch.cat([pe_out, ple_out], dim=-1) if dimensions match

        if self.attention:
            x_src = x + pos_enc
            x = self.transformer_encoder(
                x_src,
                mask=None,
                src_key_padding_mask=(frac==0),
                stoich_frac=frac
            )

        if self.fractional:
            x = x * frac.unsqueeze(-1)

        x = x.masked_fill((frac==0).unsqueeze(-1), 0.0)
        return x




class Madani(nn.Module):
    def __init__(self,
                 out_dims=3,
                 d_model=512,
                 N=3,
                 heads=4,
                 compute_device=None,
                 residual_nn='roost',
                 num_experts=4,
                 gating_noise=0.1):
        super().__init__()
        self.avg = True
        self.out_dims = out_dims
        self.d_model = d_model
        self.N = N
        self.heads = heads
        self.compute_device = compute_device

        self.encoder = EncoderMoE(
            d_model=self.d_model,
            N=self.N,
            heads=self.heads,
            compute_device=self.compute_device,
            num_experts=num_experts,
            gating_noise=gating_noise
        )

        if residual_nn == 'roost':
            self.out_hidden = [1024, 512, 256, 128]
            self.output_nn = ResidualNetwork(self.d_model, self.out_dims, self.out_hidden)
        else:
            self.out_hidden = [256, 128]
            self.output_nn = ResidualNetwork(self.d_model, self.out_dims, self.out_hidden)

    def forward(self, src, frac):
        output = self.encoder(src, frac)
        mask = (src == 0).unsqueeze(-1).repeat(1, 1, self.out_dims)
        output = self.output_nn(output)

        if self.avg:
            output = output.masked_fill(mask, 0)
            output = output.sum(dim=1) / (~mask).sum(dim=1)
            output, logits = output.chunk(2, dim=-1)
            probability = torch.ones_like(output)
            probability[:, :logits.shape[-1]] = torch.sigmoid(logits)
            output = output * probability

        return output




# %%
if __name__ == '__main__':
    model = Madani()

