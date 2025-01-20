import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

###############################################################################
# Setup & Reproducibility
###############################################################################
RNG_SEED = 42
torch.manual_seed(RNG_SEED)
np.random.seed(RNG_SEED)
data_type_torch = torch.float32

###############################################################################
# ResidualNetwork (unchanged)
###############################################################################
class ResidualNetwork(nn.Module):
    """
    Feed forward Residual Neural Network as seen in Roost.
    https://doi.org/10.1038/s41467-020-19964-7
    """
    def __init__(self, input_dim, output_dim, hidden_layer_dims):
        super().__init__()
        dims = [input_dim] + hidden_layer_dims
        self.fcs = nn.ModuleList(
            [nn.Linear(dims[i], dims[i+1]) for i in range(len(dims)-1)]
        )
        self.res_fcs = nn.ModuleList([
            nn.Linear(dims[i], dims[i+1], bias=False)
            if (dims[i] != dims[i+1]) else nn.Identity()
            for i in range(len(dims)-1)
        ])
        self.acts = nn.ModuleList([nn.LeakyReLU() for _ in range(len(dims)-1)])
        self.fc_out = nn.Linear(dims[-1], output_dim)

    def forward(self, fea):
        for fc, res_fc, act in zip(self.fcs, self.res_fcs, self.acts):
            fea = act(fc(fea)) + res_fc(fea)
        return self.fc_out(fea)

###############################################################################
# Simple Embedder (unchanged or adapt paths)
###############################################################################
class Embedder(nn.Module):
    def __init__(self, d_model, compute_device=None):
        super().__init__()
        self.d_model = d_model
        self.compute_device = compute_device

        elem_dir = 'madani/data/element_properties'
        mat2vec = f'{elem_dir}/mat2vec.csv'  # e.g. element embedding
        cbfv = pd.read_csv(mat2vec, index_col=0).values
        feat_size = cbfv.shape[-1]

        self.fc_mat2vec = nn.Linear(feat_size, d_model).to(self.compute_device)

        zeros = np.zeros((1, feat_size))
        cat_array = np.concatenate([zeros, cbfv])  # first row => 'pad'
        cat_array = torch.as_tensor(cat_array, dtype=data_type_torch)
        self.cbfv = nn.Embedding.from_pretrained(cat_array) \
            .to(self.compute_device, dtype=data_type_torch)

    def forward(self, src):
        """
        src: [B, T] (long), 0 => pad index
        """
        mat2vec_emb = self.cbfv(src)            # [B, T, feat_size]
        x_emb = self.fc_mat2vec(mat2vec_emb)    # [B, T, d_model]
        return x_emb

###############################################################################
# Custom Multi-Head Attention with (Q*K) + bias * fraction_factor, sum => scalar
###############################################################################
class CustomMultiHeadAttention(nn.Module):
    """
    For each pair (i,j):
      1) elementwise multiply Q_i*K_j
      2) add a learnable bias vector * fraction_factor(i,j)
      3) sum across the dimension => scalar logit
    """
    def __init__(self, d_model, nhead, dropout=0.1, eps=1e-8):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.eps = eps

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Learnable bias vector for each head-dim
        self.attn_bias = nn.Parameter(torch.zeros(self.head_dim))

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, query, key, value, frac,
                key_padding_mask=None, attn_mask=None):
        """
        query,key,value: [B, T, d_model]
        frac: [B, T], fractional amounts in [0,1]
        """
        B, T, _ = query.shape

        # 1) Project Q, K, V
        Q = self.W_q(query)  # [B, T, d_model]
        K = self.W_k(key)    # [B, T, d_model]
        V = self.W_v(value)  # [B, T, d_model]

        # 2) Reshape => [B, nhead, T, head_dim]
        Q = Q.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.nhead, self.head_dim).transpose(1, 2)

        # 3) fraction factor_{i,j} = (frac_j - frac_i)/(frac_i*frac_j + eps)
        frac_i = frac.unsqueeze(2)  # [B, T, 1]
        frac_j = frac.unsqueeze(1)  # [B, 1, T]
        factor_ij = (frac_j - frac_i) / (frac_i * frac_j + self.eps)  # [B, T, T]

        factor_ij = factor_ij.unsqueeze(1).unsqueeze(-1)  # => [B,1,T,T,1]

        # 4) Elementwise Q*K => [B, nhead, T, T, head_dim]
        QK_eltwise = Q.unsqueeze(3) * K.unsqueeze(2)

        # 5) Add bias * factor => same shape [B, nhead, T, T, head_dim]
        bias_term = self.attn_bias.view(1,1,1,1,self.head_dim) * factor_ij
        combined = QK_eltwise + bias_term

        # 6) Sum over last dim => attn_logits [B, nhead, T, T]
        attn_logits = combined.sum(dim=-1)

        # 7) Scale
        attn_logits = attn_logits * self.scale

        # 8) Apply optional masks
        if attn_mask is not None:
            attn_logits = attn_logits + attn_mask

        if key_padding_mask is not None:
            # shape [B, T], True => ignore
            expanded_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_logits = attn_logits.masked_fill(expanded_mask, float('-inf'))

        # 9) Softmax
        attn_probs = F.softmax(attn_logits, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # 10) Weighted sum over V => [B, nhead, T, head_dim]
        out = torch.matmul(attn_probs, V)

        # 11) Reshape => [B, T, d_model]
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)

        # 12) Final linear
        out = self.out_proj(out)
        return out

###############################################################################
# MoE FeedForward
###############################################################################
class MoEFeedForwardTop2(nn.Module):
    """
    MoE feed-forward module with top-2 gating and optional gating noise.
    """
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1,
                 num_experts=4, gating_noise=0.0):
        super().__init__()
        self.num_experts = num_experts
        self.gating_noise = gating_noise
        self.gate = nn.Linear(d_model, num_experts)

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
        B, T, d = x.shape
        gating_logits = self.gate(x)  # [B, T, num_experts]

        if self.training and self.gating_noise > 0.0:
            noise = torch.randn_like(gating_logits) * self.gating_noise
            gating_logits = gating_logits + noise

        gating_scores = F.softmax(gating_logits, dim=-1)
        top2_scores, top2_experts = gating_scores.topk(2, dim=-1)
        out = torch.zeros_like(x)

        for rank in range(2):
            expert_idx = top2_experts[..., rank]    # [B, T]
            gating_weight = top2_scores[..., rank]  # [B, T]
            for e_idx in range(self.num_experts):
                mask = (expert_idx == e_idx)
                if mask.any():
                    selected_x = x[mask]
                    expert_out = self.experts[e_idx](selected_x)
                    weight = gating_weight[mask].unsqueeze(-1)
                    out[mask] += weight * expert_out

        return out

###############################################################################
# Transformer Encoder Layer (uses CustomMultiHeadAttention)
###############################################################################
class CustomTransformerEncoderMoELayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048,
                 dropout=0.1, layer_norm_eps=1e-5,
                 num_experts=4, gating_noise=0.0):
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

    def forward(self, src, frac, src_mask=None, src_key_padding_mask=None):
        # 1) Self-attn
        attn_out = self.self_attn(
            query=src,
            key=src,
            value=src,
            frac=frac,
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

###############################################################################
# Transformer Encoder Stack
###############################################################################
class CustomTransformerEncoderMoE(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        # no final layer norm for simplicity

    def forward(self, src, frac, mask=None, src_key_padding_mask=None):
        x = src
        for layer in self.layers:
            x = layer(
                x,
                frac=frac,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask
            )
        return x

###############################################################################
# EncoderMoE: NO FractionalEncoder usage
###############################################################################
class EncoderMoE(nn.Module):
    def __init__(
        self,
        d_model,
        N,
        heads,
        frac=False,
        attn=True,
        compute_device=None,
        num_experts=4,
        gating_noise=0.0
    ):
        super().__init__()
        self.d_model = d_model
        self.N = N
        self.heads = heads
        self.fractional = frac  # if True, multiply final reps by frac
        self.attention = attn
        self.compute_device = compute_device

        # We ONLY embed elements; no fractional encoding
        self.embed = Embedder(d_model=self.d_model, compute_device=self.compute_device)

        if self.attention:
            encoder_layer = CustomTransformerEncoderMoELayer(
                d_model=self.d_model,
                nhead=self.heads,
                dim_feedforward=2048,
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
        """
        src: [B, T] integer-coded elements (0 => pad)
        frac: [B, T] fractional amounts in [0,1], 0 => pad
        """
        # 1) Embed
        x = self.embed(src)  # [B, T, d_model]

        # 2) Key padding mask from frac==0
        src_key_padding_mask = (frac == 0)

        # 3) If using attention, pass fraction => factor in MHA
        if self.attention:
            x = self.transformer_encoder(
                src=x,
                frac=frac,
                mask=None,
                src_key_padding_mask=src_key_padding_mask
            )

        # 4) If "fractional" => multiply final by frac
        if self.fractional:
            x = x * frac.unsqueeze(-1)

        # 5) zero out padded positions
        x = x.masked_fill(src_key_padding_mask.unsqueeze(-1), 0.0)

        return x

###############################################################################
# Final Madani model
###############################################################################
class Madani(nn.Module):
    def __init__(
        self,
        out_dims=3,
        d_model=512,
        N=3,
        heads=4,
        compute_device=None,
        residual_nn='roost',
        num_experts=4,
        gating_noise=0.1
    ):
        super().__init__()
        self.avg = True
        self.out_dims = out_dims
        self.d_model = d_model
        self.N = N
        self.heads = heads
        self.compute_device = compute_device

        # Build the encoder, which uses custom MHA + MoE
        self.encoder = EncoderMoE(
            d_model=self.d_model,
            N=self.N,
            heads=self.heads,
            frac=False,               # or True if you want x *= frac
            attn=True,
            compute_device=self.compute_device,
            num_experts=num_experts,
            gating_noise=gating_noise
        )

        # Output network
        # produce 2*out_dims => chunk into (prediction, logits) if you want uncertainties
        if residual_nn == 'roost':
            self.out_hidden = [1024, 512, 256, 128]
            self.output_nn = ResidualNetwork(self.d_model, 2*self.out_dims, self.out_hidden)
        else:
            self.out_hidden = [256, 128]
            self.output_nn = ResidualNetwork(self.d_model, 2*self.out_dims, self.out_hidden)

    def forward(self, src, frac):
        """
        src: [B, T],   0 => pad
        frac: [B, T],  0 => pad
        """
        # 1) Encode
        x = self.encoder(src, frac)  # => [B, T, d_model]

        # 2) Output feed-forward => [B, T, 2*out_dims]
        out = self.output_nn(x)

        # 3) If we're averaging across T:
        if self.avg:
            # mask => [B, T, 2*out_dims]
            mask = (src == 0).unsqueeze(-1).expand(-1, -1, 2*self.out_dims)
            out = out.masked_fill(mask, 0.0)

            # sum + divide by number of real tokens
            valid_counts = (~mask).sum(dim=1).clamp(min=1e-9)  # shape [B, 2*out_dims]
            out = out.sum(dim=1) / valid_counts

            # chunk => first half is "values", second half is "logits"
            values, logits = out.chunk(2, dim=-1)

            # If you do a gating approach with logits => probabilities
            probability = torch.ones_like(values)
            probability[:, :logits.shape[-1]] = torch.sigmoid(logits)
            values = values * probability
            return values
        else:
            # Return full sequence
            return out

# %%
if __name__ == '__main__':
    model = Madani()
