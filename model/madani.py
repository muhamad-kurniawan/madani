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

class StoichiometricBias(nn.Module):
    def __init__(self, d_model, resolution=100, compute_device=None):
        super().__init__()
        self.d_model = d_model
        self.resolution = resolution
        self.compute_device = compute_device

    def forward(self, frac1, frac2):
        """
        Compute the relative stoichiometric bias between two fractional amounts.
        frac1, frac2: [B, T] tensor containing fractional stoichiometric values
        Returns: [B, T, d_model] soft bias term for stoichiometry
        """
        # Compute relative difference (difference normalized by total sum for stability)
        frac_diff = torch.abs(frac1 - frac2)  # [B, T] difference
        bias = frac_diff.unsqueeze(-1) * torch.ones_like(frac_diff).unsqueeze(-1)  # Bias of shape [B, T, 1]

        # Optional: scale by magnitude
        # You can apply some non-linearity or function to scale it, e.g., softmax or a power-law scale.
        bias = bias * torch.log(1 + frac_diff)  # Soft scaling by log of the difference

        return bias


import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomMultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, stoich_bias=None, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.stoich_bias = stoich_bias

        # Linear layers for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        # Scaled factor for dot-product attention
        self.scale = self.head_dim**-0.5

    def forward(self, query, key, value, stoichiometry, key_padding_mask=None, attn_mask=None):
        B, T, _ = query.shape

        # 1) Project Q, K, V
        Q = self.W_q(query)  # [B, T, d_model]
        K = self.W_k(key)    # [B, T, d_model]
        V = self.W_v(value)  # [B, T, d_model]

        # 2) Reshape to [B, nhead, T, head_dim]
        Q = Q.view(B, T, self.nhead, self.head_dim).transpose(1, 2)  # [B, nhead, T, head_dim]
        K = K.view(B, T, self.nhead, self.head_dim).transpose(1, 2)  # [B, nhead, T, head_dim]
        V = V.view(B, T, self.nhead, self.head_dim).transpose(1, 2)  # [B, nhead, T, head_dim]

        # 3) Compute scaled dot-product attention
        attn_weights = torch.matmul(Q, K.transpose(-1, -2)) * self.scale  # [B, nhead, T_q, T_k]

        # 4) Add stoichiometric bias if provided
        if self.stoich_bias is not None:
            stoich = self.stoich_bias(stoichiometry, stoichiometry)  # Bias for the stoichiometric adjustment
            attn_weights += stoich

        # If attn_mask is provided (e.g., subsequent mask), apply it
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask

        # Apply key_padding_mask if necessary
        if key_padding_mask is not None:
            expanded_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T_k]
            attn_weights = attn_weights.masked_fill(expanded_mask, float('-inf'))

        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # 5) Get the attention output
        out = torch.matmul(attn_probs, V)  # [B, nhead, T_q, head_dim]

        # 6) Reshape back to [B, T_q, d_model]
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)

        # 7) Final linear projection
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
        # Self-attention block
        self.self_attn = CustomMultiHeadAttention(d_model, nhead, dropout=dropout)

        # MoE feed-forward with top-2 gating + gating noise
        self.feed_forward = MoEFeedForwardTop2(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_experts=num_experts,
            gating_noise=gating_noise
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # 1) Self-attention
        attn_out = self.self_attn(
            src, src, src,
            key_padding_mask=src_key_padding_mask,
            attn_mask=src_mask
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
        # We omit final LayerNorm for simplicity

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        for layer in self.layers:
            output = layer(output,
                           src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask)
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
                 stoich_bias=None,  # Include stoichiometric bias
                 compute_device=None, 
                 num_experts=4, 
                 gating_noise=0.0):
        super().__init__()
        self.d_model = d_model
        self.N = N
        self.heads = heads
        self.compute_device = compute_device

        # Embedder and encoders as before
        self.embed = Embedder(d_model=self.d_model, compute_device=self.compute_device)
        self.pe = FractionalEncoder(self.d_model, resolution=5000, log10=False)
        self.ple = FractionalEncoder(self.d_model, resolution=5000, log10=True)

        # Initialize stoichiometric bias
        self.stoich_bias = StoichiometricBias(self.d_model, compute_device=self.compute_device)

        if self.attention:
            encoder_layer = CustomTransformerEncoderMoELayer(
                d_model=self.d_model,
                nhead=self.heads,
                dim_feedforward=2048,
                dropout=0.1,
                num_experts=num_experts,
                gating_noise=gating_noise,
                stoich_bias=self.stoich_bias  # Pass stoichiometric bias to the layer
            )
            self.transformer_encoder = CustomTransformerEncoderMoE(
                encoder_layer=encoder_layer,
                num_layers=self.N
            )

    def forward(self, src, frac):
        x = self.embed(src)
        stoichiometry = frac  # Fractional stoichiometries

        # Apply the stoichiometric bias during the forward pass
        x = self.transformer_encoder(x, stoichiometry=stoichiometry)
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

