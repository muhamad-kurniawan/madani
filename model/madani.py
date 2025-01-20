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
###############################################################################
# ResidualNetwork (unchanged)
###############################################################################
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

    def __repr__(self):
        return f'{self.__class__.__name__}'


###############################################################################
# Embedder (unchanged or adapt to your liking)
###############################################################################
class Embedder(nn.Module):
    def __init__(self, d_model, compute_device=None):
        super().__init__()
        self.d_model = d_model
        self.compute_device = compute_device

        elem_dir = 'madani/data/element_properties'
        mat2vec = f'{elem_dir}/mat2vec.csv'  # example path
        cbfv = pd.read_csv(mat2vec, index_col=0).values
        feat_size = cbfv.shape[-1]

        self.fc_mat2vec = nn.Linear(feat_size, d_model).to(self.compute_device)
        zeros = np.zeros((1, feat_size))
        cat_array = np.concatenate([zeros, cbfv])  # row-0 => pad
        cat_array = torch.as_tensor(cat_array, dtype=data_type_torch)

        # Index 0 => PAD
        self.cbfv = nn.Embedding.from_pretrained(cat_array).to(
            self.compute_device, dtype=data_type_torch
        )

    def forward(self, src):
        """
        src: [B, T], integer-coded elements. 0 => "pad" element.
        """
        mat2vec_emb = self.cbfv(src)      # [B, T, feat_size]
        x_emb = self.fc_mat2vec(mat2vec_emb)  # [B, T, d_model]
        return x_emb


###############################################################################
# CustomMultiHeadAttention with vector-bias + fraction factor
###############################################################################
class CustomMultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, eps=1e-8):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.dropout = nn.Dropout(dropout)
        self.eps = eps

        # Q, K, V projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Vector bias for each head-dim
        self.attn_bias = nn.Parameter(torch.zeros(self.head_dim))

        # Final projection
        self.out_proj = nn.Linear(d_model, d_model)

        # Scale factor for "scaled dot-product"
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        query,          # [B, T, d_model]
        key,            # [B, T, d_model]
        value,          # [B, T, d_model]
        frac,           # [B, T], fractional amounts
        key_padding_mask=None,  # [B, T], True => pad
        attn_mask=None          # [T, T] or [B, T, T], optional
    ):
        B, T, _ = query.shape

        # 1) Linear projections
        Q = self.W_q(query)  # [B, T, d_model]
        K = self.W_k(key)    # [B, T, d_model]
        V = self.W_v(value)  # [B, T, d_model]

        # 2) Reshape => [B, nhead, T, head_dim]
        Q = Q.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.nhead, self.head_dim).transpose(1, 2)

        # 3) fraction factor_{i,j} = (frac_j - frac_i)/(frac_i*frac_j + eps)
        #    shape => [B, T, T], then broadcast
        frac_i = frac.unsqueeze(2)  # [B, T, 1]
        frac_j = frac.unsqueeze(1)  # [B, 1, T]
        factor_ij = (frac_j - frac_i) / (frac_i * frac_j + self.eps)  # [B, T, T]

        # Expand to [B, 1, T, T, 1] to match QK's broadcast shape
        factor_ij = factor_ij.unsqueeze(1).unsqueeze(-1)

        # 4) Elementwise Q*K => shape [B, nhead, T, T, head_dim]
        #    - Q.unsqueeze(3) => [B, nhead, T, 1, head_dim]
        #    - K.unsqueeze(2) => [B, nhead, 1, T, head_dim]
        QK_eltwise = Q.unsqueeze(3) * K.unsqueeze(2)

        # 5) Add bias * factor_ij => same shape [B, nhead, T, T, head_dim]
        #    bias => [head_dim] => broadcast to [1, 1, 1, 1, head_dim]
        bias_term = self.attn_bias.view(1, 1, 1, 1, self.head_dim) * factor_ij
        combined = QK_eltwise + bias_term

        # 6) Sum over the last dimension => attention logits [B, nhead, T, T]
        attn_logits = combined.sum(dim=-1)

        # 7) Scale
        attn_logits = attn_logits * self.scale

        # 8) If attn_mask => apply
        if attn_mask is not None:
            attn_logits = attn_logits + attn_mask

        # 9) If key_padding_mask => shape [B, T], expand + mask out
        if key_padding_mask is not None:
            expanded_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [B,1,1,T]
            attn_logits = attn_logits.masked_fill(expanded_mask, float('-inf'))

        # 10) Softmax
        attn_probs = F.softmax(attn_logits, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # 11) Weighted sum over V => out shape [B, nhead, T, head_dim]
        out = torch.matmul(attn_probs, V)

        # 12) Reshape => [B, T, d_model]
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)

        # 13) Final linear projection
        out = self.out_proj(out)
        return out


###############################################################################
# Mixture-of-Experts feed-forward block (unchanged)
###############################################################################
class MoEFeedForwardTop2(nn.Module):
    """
    MoE feed-forward module with top-2 gating and optional gating noise.
    Each token's output is a weighted combination of two experts.
    """
    def __init__(
        self,
        d_model,
        dim_feedforward=2048,
        dropout=0.1,
        num_experts=4,
        gating_noise=0.0
    ):
        super().__init__()
        self.num_experts = num_experts
        self.gating_noise = gating_noise

        # Gating network: d_model -> num_experts
        self.gate = nn.Linear(d_model, num_experts)

        # Experts: each is feed-forward
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
        # 1) gating logits
        gating_logits = self.gate(x)  # [B, T, num_experts]

        # 2) gating noise
        if self.training and self.gating_noise > 0.0:
            noise = torch.randn_like(gating_logits) * self.gating_noise
            gating_logits = gating_logits + noise

        # 3) softmax
        gating_scores = F.softmax(gating_logits, dim=-1)  # [B, T, num_experts]

        # 4) top-2 gating
        top2_scores, top2_experts = gating_scores.topk(2, dim=-1)
        # shapes: [B, T, 2]

        # 5) prepare output buffer
        out = torch.zeros_like(x)

        # 6) route each token to top-2 experts
        for rank in range(2):
            expert_idx = top2_experts[..., rank]    # [B, T]
            gating_weight = top2_scores[..., rank]  # [B, T]

            for e_idx in range(self.num_experts):
                mask = (expert_idx == e_idx)  # bool
                if mask.any():
                    selected_x = x[mask]  # [N, d]
                    expert_out = self.experts[e_idx](selected_x)  # [N, d]
                    weight = gating_weight[mask].unsqueeze(-1)    # [N, 1]
                    weighted_out = weight * expert_out
                    out[mask] += weighted_out

        return out


###############################################################################
# CustomTransformerEncoderMoELayer that calls our custom MHA
###############################################################################
class CustomTransformerEncoderMoELayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        layer_norm_eps=1e-5,
        num_experts=4,
        gating_noise=0.0
    ):
        super().__init__()
        self.self_attn = CustomMultiHeadAttention(d_model, nhead, dropout=dropout)

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

    def forward(self, src, frac, src_mask=None, src_key_padding_mask=None):
        # 1) Self-attn (pass frac for factor_{i,j}):
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
# CustomTransformerEncoderMoE stack
###############################################################################
class CustomTransformerEncoderMoE(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        # no final layer norm in this example

    def forward(self, src, frac, mask=None, src_key_padding_mask=None):
        out = src
        for layer in self.layers:
            out = layer(
                out,
                frac=frac,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask
            )
        return out


###############################################################################
# EncoderMoE: now WITHOUT fractional encoding
###############################################################################
class EncoderMoE(nn.Module):
    def __init__(
        self,
        d_model,
        N,
        heads,
        frac=False,      # if True, we'll multiply final features by fraction
        attn=True,
        compute_device=None,
        num_experts=4,
        gating_noise=0.0
    ):
        super().__init__()
        self.d_model = d_model
        self.N = N
        self.heads = heads
        self.fractional = frac   # if True => multiply final reps by frac
        self.attention = attn
        self.compute_device = compute_device

        # Simple embedder only (no FractionalEncoder)
        self.embed = Embedder(d_model=self.d_model, compute_device=self.compute_device)

        # If we want attention:
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
        src: [B, T], element indices (0 => pad)
        frac: [B, T], fractional amounts
        """
        x = self.embed(src)  # [B, T, d_model]

        # Build key_padding_mask from fraction=0 => True
        src_key_padding_mask = (frac == 0)

        # If using attention, pass fraction => factor_{i,j}
        if self.attention:
            x = self.transformer_encoder(
                src=x,
                frac=frac,
                mask=None,
                src_key_padding_mask=src_key_padding_mask
            )

        # If "fractional" => multiply final reps by frac
        if self.fractional:
            x = x * frac.unsqueeze(2)  # broadcast over d_model

        # Zero out padded positions
        hmask = src_key_padding_mask.unsqueeze(-1)
        x = x.masked_fill(hmask, 0)
        return x


###############################################################################
# Madani: the main model
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

        self.encoder = EncoderMoE(
            d_model=self.d_model,
            N=self.N,
            heads=self.heads,
            compute_device=self.compute_device,
            num_experts=num_experts,
            gating_noise=gating_noise
        )

        # Output feed-forward
        if residual_nn == 'roost':
            self.out_hidden = [1024, 512, 256, 128]
            # We'll produce 2*out_dims => final half are "logits"
            self.output_nn = ResidualNetwork(self.d_model, 2*self.out_dims, self.out_hidden)
        else:
            self.out_hidden = [256, 128]
            self.output_nn = ResidualNetwork(self.d_model, 2*self.out_dims, self.out_hidden)

    def forward(self, src, frac):
        """
        src: [B, T] (integer-coded elements)
        frac: [B, T] (fractional amounts in [0,1], 0 => pad)
        """
        # 1) Encode
        x = self.encoder(src, frac)  # [B, T, d_model]

        # 2) Final feed-forward => [B, T, 2*out_dims]
        out = self.output_nn(x)

        # 3) If we're averaging predictions across T
        if self.avg:
            mask = (src == 0).unsqueeze(-1).expand(-1, -1, 2*self.out_dims)
            out = out.masked_fill(mask, 0.0)
            # sum + divide by count
            denom = (~mask).sum(dim=1).clamp(min=1e-9)  # [B, 2*out_dims]
            out = out.sum(dim=1) / denom

            # Suppose we interpret it as (values, logits)
            values, logits = out.chunk(2, dim=-1)  # each => [B, out_dims]

            # gating or bounding approach
            probability = torch.ones_like(values)
            probability[:, :logits.shape[-1]] = torch.sigmoid(logits)
            values = values * probability
            return values

        else:
            # Return full [B, T, 2*out_dims]
            return out



# %%
if __name__ == '__main__':
    model = Madani()

