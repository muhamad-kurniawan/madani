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

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FourierPositionalBias(nn.Module):
    """
    Learns random Fourier features from fractional amounts and maps them to
    a scalar bias for each pair of positions (i.e., a [B, T, T] matrix).
    """
    def __init__(self, num_features=16, scale=10.0):
        super().__init__()
        self.num_features = num_features
        self.scale = scale

        # Example random Fourier parameters:
        #   W, b, plus a final linear projection to 1D
        self.W = nn.Parameter(torch.randn(1, num_features) * 0.1)
        self.b = nn.Parameter(torch.rand(num_features) * 2 * np.pi)
        self.a = nn.Parameter(torch.randn(num_features))

    def forward(self, fractions):
        """
        fractions: [B, T], fractional amounts
        Returns:
            bias: [B, T, T], the pairwise Fourier-based bias
        """
        B, T = fractions.shape
        # Pairwise differences: shape [B, T, T]
        diffs = fractions.unsqueeze(2) - fractions.unsqueeze(1)  # (B, T, T)

        # Scale them
        diffs_scaled = diffs * self.scale  # [B, T, T]
        # Expand to [B, T, T, 1] so we can matmul with W
        diffs_scaled = diffs_scaled.unsqueeze(-1)

        # random Fourier features => cos(diffs_scaled * W + b)
        # shape => [B, T, T, num_features]
        # We'll broadcast W [1, num_features] across B, T, T
        rff = torch.cos(torch.matmul(diffs_scaled, self.W) + self.b)

        # Then map down to a single scalar per pair via a
        # shape => [B, T, T]
        bias = torch.matmul(rff, self.a)
        return bias


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

class FourierLearnerMoELayer(nn.Module):
    """
    One 'transformer block' that does:
      1) Fourier-based "attention" (using pairwise bias).
      2) MoE feed-forward (top-2 gating).
    """
    def __init__(self, 
                 d_model, 
                 num_heads=4,
                 dim_feedforward=2048,
                 dropout=0.1,
                 num_experts=4,
                 gating_noise=0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.d_model = d_model

        # "Key" and "Value" projection for Fourier-based attention
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)

        self.dropout_attn = nn.Dropout(dropout)
        self.norm_attn = nn.LayerNorm(d_model)

        # MoE feed-forward (top-2 gating)
        self.moe_ff = MoEFeedForwardTop2(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_experts=num_experts,
            gating_noise=gating_noise
        )
        self.dropout_ff = nn.Dropout(dropout)
        self.norm_ff = nn.LayerNorm(d_model)

    def forward(self, x, fourier_bias):
        """
        x: [B, T, d_model]
        fourier_bias: [B, T, T]
        Returns:
            x: [B, T, d_model]
        """
        B, T, D = x.shape

        # --- 1) Fourier-based "attention" sub-layer ---
        # Project K and V
        K = self.key_linear(x)   # => [B, T, d_model]
        V = self.value_linear(x) # => [B, T, d_model]

        # Reshape for multi-head => [B, heads, T, head_dim]
        K = K.view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Expand fourier_bias => [B, 1, T, T] => [B, heads, T, T]
        fb = fourier_bias.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

        # We'll define "scores" = (K + fb) and multiply by V
        # shape of K.unsqueeze(2) => [B, heads, 1, T, head_dim]
        # but we want each query position to incorporate a bias for each key position
        # So "scores" => [B, heads, T, T, head_dim]
        scores = K.unsqueeze(2) + fb.unsqueeze(-1)
        # Weighted sum across the T (key) dimension
        # => [B, heads, T, head_dim]
        weighted_values = torch.sum(scores * V.unsqueeze(2), dim=3)

        # Combine heads
        weighted_values = weighted_values.permute(0, 2, 1, 3).contiguous()
        weighted_values = weighted_values.view(B, T, D)

        # Output projection
        out_attn = self.output_linear(weighted_values)

        # Residual + norm
        x = x + self.dropout_attn(out_attn)
        x = self.norm_attn(x)

        # --- 2) MoE feed-forward sub-layer ---
        ff_out = self.moe_ff(x)     # => [B, T, d_model]
        x = x + self.dropout_ff(ff_out)
        x = self.norm_ff(x)

        return x

class EncoderFourierMoE(nn.Module):
    def __init__(self,
                 d_model,
                 N=3,
                 heads=4,
                 dim_feedforward=2048,
                 dropout=0.1,
                 num_experts=4,
                 gating_noise=0.0,
                 compute_device=None,
                 fractional=True):
        """
        d_model: dimension of embeddings
        N: number of layers
        heads: # heads for the Fourier-based 'attention'
        dim_feedforward: hidden size for MoE feed-forward
        fractional: whether to multiply final x by frac or not (like your original code)
        """
        super().__init__()
        self.d_model = d_model
        self.N = N
        self.heads = heads
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.num_experts = num_experts
        self.gating_noise = gating_noise
        self.compute_device = compute_device
        self.fractional = fractional

        # Embedding: same as your code
        self.embed = Embedder(d_model=self.d_model, compute_device=self.compute_device)

        # Fractional encoders from your code
        self.pe = FractionalEncoder(self.d_model, resolution=5000, log10=False)
        self.ple = FractionalEncoder(self.d_model, resolution=5000, log10=True)

        self.emb_scaler = nn.Parameter(torch.tensor([1.0]))
        self.pos_scaler = nn.Parameter(torch.tensor([1.0]))
        self.pos_scaler_log = nn.Parameter(torch.tensor([1.0]))

        # Fourier bias module (like random Fourier features)
        self.fourier_bias = FourierPositionalBias(num_features=16, scale=10.0)

        # Stack of N Fourier+MoE layers
        self.layers = nn.ModuleList([
            FourierLearnerMoELayer(
                d_model=self.d_model,
                num_heads=self.heads,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout,
                num_experts=self.num_experts,
                gating_noise=self.gating_noise
            ) for _ in range(N)
        ])

    def forward(self, src, frac):
        """
        src: [B, T], element indices
        frac: [B, T], fractional amounts in [0,1]
        Returns: [B, T, d_model]
        """
        # 1) Embedding
        x = self.embed(src) * 2.0**self.emb_scaler  # e.g. scale up embeddings

        # 2) Build "positional" encodings from your FractionalEncoder
        pe_scaler = 2.0**((1 - self.pos_scaler)**2)
        ple_scaler = 2.0**((1 - self.pos_scaler_log)**2)

        # shape => [B, T, d_model//2], or 0 if frac=0
        pe = self.pe(frac) * pe_scaler
        ple = self.ple(frac) * ple_scaler

        # Combine them into a [B, T, d_model] structure
        # E.g. first half from pe, second half from ple
        fourier_pe = torch.zeros_like(x)
        fourier_pe[:, :, : self.d_model // 2] = pe
        fourier_pe[:, :, self.d_model // 2 :] = ple

        x = x + fourier_pe

        # 3) Compute the **Fourier-based positional bias** for the entire sequence
        fb = self.fourier_bias(frac)  # => [B, T, T]

        # If you want to mask padding (where frac=0), you can set those row/column
        # entries in fb to something, e.g., large negative or zero. We'll do a simple approach:
        # mask out rows & columns for frac=0
        with torch.no_grad():
            pad_mask = (frac == 0)  # [B, T] bool
            fb[pad_mask.unsqueeze(2).expand(-1, -1, fb.size(2))] = 0
            fb[pad_mask.unsqueeze(1).expand(-1, fb.size(1), -1)] = 0

        # 4) Pass through the N Fourier+MoE layers
        for layer in self.layers:
            x = layer(x, fb)

        # 5) If fractional scaling is desired (like your original code)
        if self.fractional:
            x = x * frac.unsqueeze(-1)

        # 6) Zero out any positions that do not exist
        hmask = (frac == 0).unsqueeze(-1)
        x = x.masked_fill(hmask, 0)

        return x

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
    

class Madani(nn.Module):
    def __init__(self,
                 out_dims=3,
                 d_model=512,
                 N=3,
                 heads=4,
                 compute_device=None,
                 residual_nn='roost',
                 num_experts=4,
                 gating_noise=0.1,
                 dim_feedforward=2048,
                 fractional=True):
        """
        out_dims: # of output dimensions
        d_model: dimension for embeddings and internal layers
        N: # of Fourier+MoE layers
        heads: # of heads in the Fourier-based attention
        num_experts: # of experts in MoE feed-forward
        gating_noise: std dev of gating noise
        dim_feedforward: hidden size in MoE feed-forward
        fractional: whether to multiply the final x by frac (like your old code)
        """
        super().__init__()
        self.out_dims = out_dims
        self.d_model = d_model
        self.N = N
        self.heads = heads
        self.compute_device = compute_device
        self.num_experts = num_experts
        self.gating_noise = gating_noise
        self.dim_feedforward = dim_feedforward
        self.fractional = fractional

        # The Fourier+MoE encoder
        self.encoder = EncoderFourierMoE(
            d_model=self.d_model,
            N=self.N,
            heads=self.heads,
            dim_feedforward=self.dim_feedforward,
            dropout=0.1,
            num_experts=self.num_experts,
            gating_noise=self.gating_noise,
            compute_device=self.compute_device,
            fractional=self.fractional
        )

        # Output MLP
        if residual_nn == 'roost':
            self.out_hidden = [1024, 512, 256, 128]
            self.output_nn = ResidualNetwork(self.d_model, self.out_dims, self.out_hidden)
        else:
            self.out_hidden = [256, 128]
            self.output_nn = ResidualNetwork(self.d_model, self.out_dims, self.out_hidden)

        # If you want to average the last dimension across elements
        # and chunk to get [property, logits], do so
        self.avg = True

    def forward(self, src, frac):
        """
        src: [B, T], element indices (0 => padding)
        frac: [B, T], fractional amounts
        Returns: [B, out_dims]
        """
        # 1) Fourier+MoE encoder output
        x = self.encoder(src, frac)  # => [B, T, d_model]

        # 2) Output MLP => [B, T, out_dims]
        x = self.output_nn(x)

        # 3) If self.avg, do the "average + gating" approach
        mask = (src == 0).unsqueeze(-1).repeat(1, 1, self.out_dims)
        if self.avg:
            x = x.masked_fill(mask, 0)
            x = x.sum(dim=1) / (~mask).sum(dim=1)
            # chunk => [props, logits]
            x, logits = x.chunk(2, dim=-1)
            probability = torch.ones_like(x)
            probability[:, :logits.shape[-1]] = torch.sigmoid(logits)
            x = x * probability

        return x


# %%
if __name__ == '__main__':
    model = Madani()

