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

import torch
import torch.nn as nn
import torch.nn.functional as F

class MoEFeedForwardTop2(nn.Module):
    """
    MoE feed-forward with top-2 gating.
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
            d_model (int): Model dimensionality.
            dim_feedforward (int): Hidden size of each expert's MLP.
            dropout (float): Dropout probability in each expert.
            num_experts (int): Number of parallel experts.
            gating_noise (float): Std dev for optional gating noise. 0 => no noise.
        """
        super().__init__()
        self.num_experts = num_experts
        self.gating_noise = gating_noise

        # Gating network: projects [B, T, d_model] -> [B, T, num_experts]
        self.gate = nn.Linear(d_model, num_experts)

        # Experts: each is a 2-layer MLP
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, d_model),
                nn.Dropout(dropout)
            )
            for _ in range(num_experts)
        ])

    def forward(self, x):
        """
        x: [B, T, d_model]
        Returns: [B, T, d_model]
        """
        B, T, _ = x.shape

        # 1) Gating logits
        gating_logits = self.gate(x)  # => [B, T, num_experts]

        # 2) Optional gating noise (for exploration)
        if self.training and self.gating_noise > 0.0:
            noise = torch.randn_like(gating_logits) * self.gating_noise
            gating_logits = gating_logits + noise

        # 3) Softmax => gating scores
        gating_scores = F.softmax(gating_logits, dim=-1)  # => [B, T, num_experts]

        # 4) Top-2 gating
        top2_scores, top2_experts = gating_scores.topk(2, dim=-1)
        # => top2_scores:   [B, T, 2]
        # => top2_experts:  [B, T, 2]

        out = torch.zeros_like(x)

        # 5) For each rank, gather tokens and forward them through the selected expert
        for rank in range(2):
            expert_idx = top2_experts[..., rank]   # => [B, T]
            expert_weight = top2_scores[..., rank] # => [B, T]

            for e_idx in range(self.num_experts):
                mask = (expert_idx == e_idx)  # bool mask => [B, T]
                if mask.any():
                    selected_x = x[mask]  # => [N, d_model]
                    expert_out = self.experts[e_idx](selected_x)  # => [N, d_model]
                    w = expert_weight[mask].unsqueeze(-1)         # => [N, 1]
                    out[mask] += w * expert_out

        return out

class FourierPositionalBias(nn.Module):
    def __init__(self, num_features, scale=10):
        super(FourierPositionalBias, self).__init__()
        self.scale = scale
        self.W = nn.Parameter(torch.randn(1, num_features) * 0.1)
        self.b = nn.Parameter(torch.rand(num_features) * 2 * np.pi)
        self.a = nn.Parameter(torch.randn(num_features))

    def forward(self, fractions):
        diffs = fractions.unsqueeze(1) - fractions.unsqueeze(2)  # (batch_size, seq_len, seq_len)
        scaled_diffs = diffs * self.scale
        scaled_diffs = scaled_diffs.unsqueeze(-1)  # (batch_size, seq_len, seq_len, 1)
        rff = torch.cos(torch.matmul(scaled_diffs, self.W) + self.b)  # (batch_size, seq_len, seq_len, num_features)
        positional_bias = torch.matmul(rff, self.a)  # (batch_size, seq_len, seq_len)
        return positional_bias
    
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
    
class FourierMoETransformerLayer(nn.Module):
    """
    A transformer-like layer with:
      - Fourier-based attention
      - MoE feed-forward (top-2 gating)
    """
    def __init__(self, model_dim, num_heads,
                 dim_feedforward=2048,
                 dropout=0.1,
                 num_experts=4,
                 gating_noise=0.0):
        super().__init__()
        assert model_dim % num_heads == 0, "model_dim must be divisible by num_heads"
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

        # Fourier-based "attention"
        self.key_linear = nn.Linear(model_dim, model_dim)
        self.value_linear = nn.Linear(model_dim, model_dim)
        self.output_linear = nn.Linear(model_dim, model_dim)
        self.norm_attn = nn.LayerNorm(model_dim)
        self.dropout_attn = nn.Dropout(dropout)

        # MoE feed-forward
        self.moe_ff = MoEFeedForwardTop2(
            d_model=model_dim,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_experts=num_experts,
            gating_noise=gating_noise
        )
        self.norm_ff = nn.LayerNorm(model_dim)
        self.dropout_ff = nn.Dropout(dropout)

    def forward(self, x, positional_bias):
        """
        x: [B, T, model_dim]
        positional_bias: [B, T, T]
        """
        B, T, D = x.shape

        # --- 1) Fourier-based attention sub-layer ---
        K = self.key_linear(x).view(B, T, self.num_heads, self.head_dim)
        V = self.value_linear(x).view(B, T, self.num_heads, self.head_dim)

        K = K.permute(0, 2, 1, 3)  # => [B, heads, T, head_dim]
        V = V.permute(0, 2, 1, 3)

        # broadcast positional_bias => [B, 1, T, T] => [B, heads, T, T]
        pb = positional_bias.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

        # "scores" = K + pb
        # shape => [B, heads, T, T, head_dim]
        scores = K.unsqueeze(2) + pb.unsqueeze(-1)

        # Weighted sum with V => sum over T dimension
        # => [B, heads, T, head_dim]
        weighted_values = torch.sum(scores * V.unsqueeze(2), dim=3)

        # combine heads => [B, T, model_dim]
        weighted_values = weighted_values.permute(0, 2, 1, 3).contiguous()
        weighted_values = weighted_values.view(B, T, D)

        # output projection
        out_attn = self.output_linear(weighted_values)

        # residual + norm
        x = x + self.dropout_attn(out_attn)
        x = self.norm_attn(x)

        # --- 2) MoE feed-forward sub-layer ---
        ff_out = self.moe_ff(x)  # => [B, T, D]
        x = x + self.dropout_ff(ff_out)
        x = self.norm_ff(x)

        return x

class Encoder(nn.Module):
    def __init__(self,
                 d_model,
                 N,
                 heads,
                 compute_device=None,
                 dim_feedforward=2048,
                 num_experts=4,
                 gating_noise=0.0):
        """
        d_model: dimension of embeddings
        N: number of layers
        heads: number of attention heads
        dim_feedforward: hidden size for the MoE FFN
        num_experts: how many experts in MoE
        gating_noise: optional gating noise
        """
        super().__init__()
        self.d_model = d_model
        self.N = N
        self.heads = heads
        self.compute_device = compute_device

        # Use your same embedder
        self.embed = Embedder(d_model=self.d_model,
                              compute_device=self.compute_device)

        # The Fourier Positional Bias module
        self.positional_bias = FourierPositionalBias(num_features=16, scale=10.0)

        # Stack N layers of Fourier + MoE
        self.transformer_layers = nn.ModuleList([
            FourierMoETransformerLayer(
                model_dim=d_model,
                num_heads=heads,
                dim_feedforward=dim_feedforward,
                dropout=0.1,
                num_experts=num_experts,
                gating_noise=gating_noise
            ) for _ in range(N)
        ])

    def forward(self, src, frac):
        """
        src: [B, T], element indices
        frac: [B, T], fractional amounts
        """
        # 1) Embed input
        x = self.embed(src)  # => [B, T, d_model]

        # 2) Compute positional bias => [B, T, T]
        pb = self.positional_bias(frac)

        # 3) Pass through N Fourier+MoE layers
        for layer in self.transformer_layers:
            x = layer(x, pb)

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
                 heads=8,
                 compute_device=None,
                 residual_nn='',
                 dim_feedforward=2048,
                 num_experts=4,
                 gating_noise=0.0):
        super().__init__()
        self.avg = True
        self.out_dims = out_dims
        self.d_model = d_model
        self.N = N
        self.heads = heads
        self.compute_device = compute_device

        self.encoder = Encoder(
            d_model=self.d_model,
            N=self.N,
            heads=self.heads,
            compute_device=self.compute_device,
            dim_feedforward=dim_feedforward,
            num_experts=num_experts,
            gating_noise=gating_noise
        )

        # Output MLP
        if residual_nn == 'roost':
            # Roost-like large residual network
            self.out_hidden = [1024, 512, 256, 128]
            self.output_nn = ResidualNetwork(self.d_model,
                                             self.out_dims,
                                             self.out_hidden)
        else:
            # Simpler residual network
            self.out_hidden = [1024, 256]
            self.output_nn = ResidualNetwork(self.d_model,
                                             self.out_dims,
                                             self.out_hidden)

    def forward(self, src, frac):
        """
        src: [B, T] integer element IDs
        frac: [B, T] fractional amounts
        """
        # 1) Fourier + MoE encoder
        output = self.encoder(src, frac)  # => [B, T, d_model]

        # 2) Output MLP => [B, T, out_dims]
        output = self.output_nn(output)

        # 3) Average the "element contribution"
        mask = (src == 0).unsqueeze(-1).repeat(1, 1, self.out_dims)
        if self.avg:
            output = output.masked_fill(mask, 0)
            output = output.sum(dim=1)/(~mask).sum(dim=1)
            # chunk => [property, logits]
            output, logits = output.chunk(2, dim=-1)
            probability = torch.ones_like(output)
            probability[:, :logits.shape[-1]] = torch.sigmoid(logits)
            output = output * probability

        return output


# %%
if __name__ == '__main__':
    model = Madani()

