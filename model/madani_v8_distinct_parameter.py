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
        dims = [input_dim] + hidden_layer_dims
        self.fcs = nn.ModuleList([
            nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        ])
        self.res_fcs = nn.ModuleList([
            nn.Linear(dims[i], dims[i + 1], bias=False)
            if (dims[i] != dims[i + 1])
            else nn.Identity()
            for i in range(len(dims) - 1)
        ])
        self.acts = nn.ModuleList([
            nn.LeakyReLU() for _ in range(len(dims) - 1)
        ])
        self.fc_out = nn.Linear(dims[-1], output_dim)

    def forward(self, fea):
        for fc, res_fc, act in zip(self.fcs, self.res_fcs, self.acts):
            fea = act(fc(fea)) + res_fc(fea)
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
        # Choose what element information the model receives
        mat2vec = f'{elem_dir}/mat2vec.csv'  # element embedding

        cbfv = pd.read_csv(mat2vec, index_col=0).values
        feat_size = cbfv.shape[-1]
        self.fc_mat2vec = nn.Linear(feat_size, d_model).to(self.compute_device)
        zeros = np.zeros((1, feat_size))
        cat_array = np.concatenate([zeros, cbfv])
        cat_array = torch.as_tensor(cat_array, dtype=data_type_torch)
        self.cbfv = nn.Embedding.from_pretrained(cat_array) \
            .to(self.compute_device, dtype=data_type_torch)

    def forward(self, src):
        mat2vec_emb = self.cbfv(src)   # [B, T, feat_size]
        x_emb = self.fc_mat2vec(mat2vec_emb)  # [B, T, d_model]
        return x_emb


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
        self.d_model = d_model // 2
        self.resolution = resolution
        self.log10 = log10
        self.compute_device = compute_device

        x = torch.linspace(0, self.resolution - 1,
                           self.resolution,
                           requires_grad=False).view(self.resolution, 1)
        fraction = torch.linspace(0, self.d_model - 1,
                                  self.d_model,
                                  requires_grad=False) \
            .view(1, self.d_model).repeat(self.resolution, 1)

        pe = torch.zeros(self.resolution, self.d_model)
        pe[:, 0::2] = torch.sin(
            x / torch.pow(50, 2 * fraction[:, 0::2] / self.d_model))
        pe[:, 1::2] = torch.cos(
            x / torch.pow(50, 2 * fraction[:, 1::2] / self.d_model))
        pe = self.register_buffer('pe', pe)

    def forward(self, x):
        # x: fractional amounts [B, T]
        x = x.clone()
        if self.log10:
            x = 0.0025 * (torch.log2(x))**2
            x = torch.clamp(x, max=1)

        x = torch.clamp(x, min=1/self.resolution)
        frac_idx = torch.round(x * (self.resolution)).to(dtype=torch.long) - 1
        out = self.pe[frac_idx]  # [B, T, d_model//2]
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
        self.scale = self.head_dim**-0.5  # scaling for dot-product attention

    def forward(
        self,
        query,              # [B, T, d_model]
        key,                # [B, T, d_model]
        value,              # [B, T, d_model]
        key_padding_mask=None,  # [B, T] mask where True indicates padding
        attn_mask=None          # optional attn mask [T, T] or [B, T, T]
    ):
        B, T, _ = query.shape

        # 1) Project Q, K, V
        Q = self.W_q(query)  # [B, T, d_model]
        K = self.W_k(key)    # [B, T, d_model]
        V = self.W_v(value)  # [B, T, d_model]

        # 2) Reshape to [B, nhead, T, head_dim]
        Q = Q.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.nhead, self.head_dim).transpose(1, 2)

        # 3) Scaled dot-product
        attn_weights = torch.matmul(Q, K.transpose(-1, -2)) * self.scale

        # 4) Apply attn_mask or key_padding_mask if given
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask
        if key_padding_mask is not None:
            expanded_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(expanded_mask, float('-inf'))

        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # 5) Weighted sum over values
        out = torch.matmul(attn_probs, V)  # [B, nhead, T, head_dim]

        # 6) Reshape back to [B, T, d_model]
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
        super().__init__()
        self.num_experts = num_experts
        self.gating_noise = gating_noise

        # Gating network: from d_model -> num_experts
        self.gate = nn.Linear(d_model, num_experts)

        # Experts: each is a small feed-forward block
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

        # 2) Optional gating noise
        if self.training and self.gating_noise > 0.0:
            noise = torch.randn_like(gating_logits) * self.gating_noise
            gating_logits = gating_logits + noise

        # 3) Softmax over experts
        gating_scores = F.softmax(gating_logits, dim=-1)  # [B, T, num_experts]

        # 4) Top-2 gating
        top2_scores, top2_experts = gating_scores.topk(2, dim=-1)
        # top2_scores:   [B, T, 2]
        # top2_experts:  [B, T, 2]

        # 5) Prepare output
        out = torch.zeros_like(x)

        # 6) Route each token to its top-2 experts
        for rank in range(2):
            # which expert index
            expert_idx = top2_experts[..., rank]    # [B, T]
            # gating weight for this expert
            gating_weight = top2_scores[..., rank]  # [B, T]

            for e_idx in range(self.num_experts):
                # gather tokens assigned to this expert
                mask = (expert_idx == e_idx)
                if mask.any():
                    selected_x = x[mask]  # shape [N, d_model]
                    expert_out = self.experts[e_idx](selected_x)  # [N, d_model]

                    # multiply by gating weight
                    weight = gating_weight[mask].unsqueeze(-1)  # [N, 1]
                    weighted_out = weight * expert_out

                    # scatter back to out
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

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # 1) Self-attn
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
    """
    Creates N distinct CustomTransformerEncoderMoELayers.
    Each layer has its own gating network and set of experts.
    """
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 layer_norm_eps=1e-5,
                 num_experts=4,
                 gating_noise=0.0,
                 num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList([
            CustomTransformerEncoderMoELayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                layer_norm_eps=layer_norm_eps,
                num_experts=num_experts,
                gating_noise=gating_noise
            )
            for _ in range(num_layers)
        ])
        # You can add a final LayerNorm if desired

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        for layer in self.layers:
            output = layer(output,
                           src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask)
        return output


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

        # Same embedder, fraction encoders
        self.embed = Embedder(d_model=self.d_model,
                              compute_device=self.compute_device)
        self.pe = FractionalEncoder(self.d_model, resolution=5000, log10=False)
        self.ple = FractionalEncoder(self.d_model, resolution=5000, log10=True)

        self.emb_scaler = nn.parameter.Parameter(torch.tensor([1.]))
        self.pos_scaler = nn.parameter.Parameter(torch.tensor([1.]))
        self.pos_scaler_log = nn.parameter.Parameter(torch.tensor([1.]))

        # Create a stack of MoE transformer layers, each distinct
        if self.attention:
            self.transformer_encoder = CustomTransformerEncoderMoE(
                d_model=self.d_model,
                nhead=self.heads,
                dim_feedforward=2048,
                dropout=0.1,
                layer_norm_eps=1e-5,
                num_experts=num_experts,
                gating_noise=gating_noise,
                num_layers=self.N
            )

    def forward(self, src, frac):
        # Embed
        x = self.embed(src) * 2**self.emb_scaler

        src_key_padding_mask = (frac == 0)

        # Create fractional encodings
        pe = torch.zeros_like(x)
        ple = torch.zeros_like(x)
        pe_scaler = 2**((1 - self.pos_scaler)**2)
        ple_scaler = 2**((1 - self.pos_scaler_log)**2)

        pe[:, :, :self.d_model//2] = self.pe(frac) * pe_scaler
        ple[:, :, self.d_model//2:] = self.ple(frac) * ple_scaler

        if self.attention:
            x_src = x + pe + ple
            x = self.transformer_encoder(
                x_src,
                mask=None,
                src_key_padding_mask=src_key_padding_mask
            )

        # Optionally apply fractional weighting to outputs
        if self.fractional:
            x = x * frac.unsqueeze(2).repeat(1, 1, self.d_model)

        # Zero out any positions that were padding
        hmask = (frac == 0).unsqueeze(-1).repeat(1, 1, self.d_model)
        x = x.masked_fill(hmask, 0)
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
        # 1) Encoder
        output = self.encoder(src, frac)

        # 2) Output MLP
        mask = (src == 0).unsqueeze(-1).repeat(1, 1, self.out_dims)
        output = self.output_nn(output)

        # 3) (Optional) Average pool across tokens
        if self.avg:
            output = output.masked_fill(mask, 0)
            output = output.sum(dim=1) / (~mask).sum(dim=1)

            # Suppose we want half the channels to be raw output
            # and half to be logits for a gating. Just an example
            # from your code: splitting into output, logits
            output, logits = output.chunk(2, dim=-1)
            probability = torch.ones_like(output)
            probability[:, :logits.shape[-1]] = torch.sigmoid(logits)
            output = output * probability

        return output


# %%
if __name__ == '__main__':
    model = Madani()
    print(model)
