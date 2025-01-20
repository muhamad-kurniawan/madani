import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

RNG_SEED = 42
torch.manual_seed(RNG_SEED)
np.random.seed(RNG_SEED)
data_type_torch = torch.float32

# --------------------------------------------------------------------------
# 1) ResidualNetwork (unchanged)
# --------------------------------------------------------------------------
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
            nn.Linear(dims[i], dims[i+1]) for i in range(len(dims) - 1)
        ])
        self.res_fcs = nn.ModuleList([
            nn.Linear(dims[i], dims[i+1], bias=False) if (dims[i] != dims[i+1]) else nn.Identity()
            for i in range(len(dims) - 1)
        ])
        self.acts = nn.ModuleList([nn.LeakyReLU() for _ in range(len(dims)-1)])
        self.fc_out = nn.Linear(dims[-1], output_dim)

    def forward(self, fea):
        for fc, res_fc, act in zip(self.fcs, self.res_fcs, self.acts):
            fea = act(fc(fea)) + res_fc(fea)
        return self.fc_out(fea)

    def __repr__(self):
        return f'{self.__class__.__name__}'


# --------------------------------------------------------------------------
# 2) Simple embedding layer (unchanged)
# --------------------------------------------------------------------------
class Embedder(nn.Module):
    def __init__(self,
                 d_model,
                 compute_device=None):
        super().__init__()
        self.d_model = d_model
        self.compute_device = compute_device

        elem_dir = 'madani/data/element_properties'
        # # Choose what element information the model receives
        mat2vec = f'{elem_dir}/mat2vec.csv'  # element embedding by default
        # Options: onehot.csv, random_200.csv, oliynyk.csv, magpie.csv, etc.

        cbfv = pd.read_csv(mat2vec, index_col=0).values  # shape [118, feat_size]
        feat_size = cbfv.shape[-1]

        self.fc_mat2vec = nn.Linear(feat_size, d_model).to(self.compute_device)

        zeros = np.zeros((1, feat_size))  # for "padding" index=0
        cat_array = np.concatenate([zeros, cbfv], axis=0)
        cat_array = torch.as_tensor(cat_array, dtype=data_type_torch)

        self.cbfv = nn.Embedding.from_pretrained(cat_array) \
            .to(self.compute_device, dtype=data_type_torch)

    def forward(self, src):
        # src is [B, T] of element indices (0 => padding)
        mat2vec_emb = self.cbfv(src)        # [B, T, feat_size]
        x_emb = self.fc_mat2vec(mat2vec_emb)  # [B, T, d_model]
        return x_emb


# --------------------------------------------------------------------------
# 3) (Optional) Multi-head attention & MoE FF from your code for the "second" layer
# --------------------------------------------------------------------------
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
        self.scale = self.head_dim ** -0.5  # for scaled dot-product

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        B, T, _ = query.shape

        # 1) Project Q, K, V
        Q = self.W_q(query)  # [B, T, d_model]
        K = self.W_k(key)    # [B, T, d_model]
        V = self.W_v(value)  # [B, T, d_model]

        # 2) Reshape to [B, nhead, T, head_dim]
        Q = Q.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.nhead, self.head_dim).transpose(1, 2)

        # 3) Compute scaled dot-product attention => [B, nhead, T, T]
        attn_weights = torch.matmul(Q, K.transpose(-1, -2)) * self.scale

        # apply optional masks
        if attn_mask is not None:
            attn_weights += attn_mask  # shape must be broadcastable

        if key_padding_mask is not None:
            expanded_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [B,1,1,T]
            attn_weights = attn_weights.masked_fill(expanded_mask, float('-inf'))

        # 4) softmax & dropout
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # 5) Weighted sum => out
        out = torch.matmul(attn_probs, V)  # [B, nhead, T, head_dim]

        # 6) Reshape => [B, T, d_model], final linear
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.out_proj(out)
        return out


class MoEFeedForwardTop2(nn.Module):
    """
    MoE feed-forward module with top-2 gating. See your original code.
    """
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1,
                 num_experts=4, gating_noise=0.0):
        super().__init__()
        self.num_experts = num_experts
        self.gating_noise = gating_noise

        # gating network
        self.gate = nn.Linear(d_model, num_experts)

        # experts
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

        # gating
        gating_logits = self.gate(x)  # [B, T, num_experts]

        # gating noise (optionally)
        if self.training and self.gating_noise > 0.0:
            noise = torch.randn_like(gating_logits) * self.gating_noise
            gating_logits += noise

        gating_scores = F.softmax(gating_logits, dim=-1)  # [B, T, num_experts]

        # top-2 gating
        top2_scores, top2_experts = gating_scores.topk(2, dim=-1)
        # shape => [B, T, 2]

        # gather outputs
        out = torch.zeros_like(x)

        for rank in range(2):
            expert_idx = top2_experts[..., rank]    # [B, T]
            gating_weight = top2_scores[..., rank]  # [B, T]

            for e_idx in range(self.num_experts):
                mask = (expert_idx == e_idx)  # [B, T]
                if mask.any():
                    selected_x = x[mask]  # shape [N, d]
                    expert_out = self.experts[e_idx](selected_x)  # [N, d]
                    weight = gating_weight[mask].unsqueeze(-1)    # [N, 1]
                    out[mask] += weight * expert_out

        return out


class CustomTransformerEncoderMoELayer(nn.Module):
    def __init__(self, d_model, nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 layer_norm_eps=1e-5,
                 num_experts=4,
                 gating_noise=0.0):
        super().__init__()
        self.self_attn = CustomMultiHeadAttention(d_model, nhead, dropout=dropout)

        # MoE feed-forward
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
        # Self-attention
        attn_out = self.self_attn(src, src, src,
                                  key_padding_mask=src_key_padding_mask,
                                  attn_mask=src_mask)
        src = src + self.dropout1(attn_out)
        src = self.norm1(src)

        # MoE feed-forward
        ff_out = self.feed_forward(src)
        src = src + self.dropout2(ff_out)
        src = self.norm2(src)

        return src


# --------------------------------------------------------------------------
# 4) Your custom 2-phase convolution layer for the *first* layer
# --------------------------------------------------------------------------
class TwoPhaseConvAttention(nn.Module):
    """
    Custom first-layer attention using:
      1) Two 1D convolutions over (x_frac, x_no_frac)
      2) Pairwise 2D conv to produce scalar attention
      3) Weighted sum => output
    """
    def __init__(self, d_model, hidden_ff=2048, kernel_size_1d=3):
        super().__init__()
        self.d_model = d_model

        # 1D convolutions
        self.conv_frac = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size_1d,
            padding=kernel_size_1d // 2
        )
        self.conv_nofrac = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size_1d,
            padding=kernel_size_1d // 2
        )

        # feedforward => "V"
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, hidden_ff),
            nn.ReLU(),
            nn.Linear(hidden_ff, d_model)
        )

        # 2D conv for pairwise => scalar
        # We treat each pair (d,2) as "1 input channel" in 2D,
        # and kernel_size=(d,2) to produce scalar
        self.conv2d = nn.Conv2d(
            in_channels=1, 
            out_channels=1, 
            kernel_size=(d_model, 2), 
            bias=True
        )

        self.norm_h = nn.LayerNorm(d_model)
        self.norm_v = nn.LayerNorm(d_model)

    def forward(self, x, frac):
        """
        x:    [B, T, d_model]
        frac: [B, T]  (0 => padding)
        Returns:
            out: [B, T, d_model]
        """
        B, T, d = x.shape
        assert d == self.d_model, "d_model mismatch"

        # branch 1: multiply by frac
        x_frac = x * frac.unsqueeze(-1)  # [B, T, d]
        # branch 2: no frac => x
        x_nofrac = x

        # 1D conv is along time dimension => (B, d, T)
        x_frac_conv = self.conv_frac(x_frac.transpose(1, 2)).transpose(1, 2)   # [B, T, d]
        x_nofrac_conv = self.conv_nofrac(x_nofrac.transpose(1, 2)).transpose(1, 2)

        # sum => h
        h = x_frac_conv + x_nofrac_conv
        h = self.norm_h(h)

        # feedforward => V
        V = self.feedforward(h)
        V = self.norm_v(V)

        # pairwise 2D conv => scalar
        # shape => (B, T, T, d, 2)
        h_i = h.unsqueeze(2).expand(-1, -1, T, -1)  # [B, T, T, d]
        h_j = h.unsqueeze(1).expand(-1, T, -1, -1)  # [B, T, T, d]
        pair_ij = torch.stack([h_i, h_j], dim=-1)   # [B, T, T, d, 2]
        pair_ij = pair_ij.view(B * T * T, 1, d, 2)  # => [B*T*T, 1, d, 2]

        scalar_ij = self.conv2d(pair_ij)           # => [B*T*T, 1, 1, 1]
        scalar_ij = scalar_ij.view(B, T, T)        # => [B, T, T]

        # softmax over "key" dimension => [B, T, T]
        attn = F.softmax(scalar_ij, dim=-1)

        # weighted sum => out
        # out[i] = sum_j attn[i,j] * V_j
        out = torch.bmm(attn, V)  # => [B, T, d]

        return out


# --------------------------------------------------------------------------
# 5) The "EncoderConvThenTransformer" that:
#    - Embeds
#    - First layer: TwoPhaseConvAttention
#    - Second layer: normal CustomTransformerEncoderMoELayer
# --------------------------------------------------------------------------
class EncoderConvThenTransformer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 layer_norm_eps=1e-5,
                 num_experts=4,
                 gating_noise=0.0,
                 N=4
                 compute_device=None):
        super().__init__()
        self.compute_device = compute_device

        # The input embedder
        self.embed = Embedder(d_model=d_model, compute_device=compute_device)

        # The special first layer (2-phase conv)
        self.first_conv_attn = TwoPhaseConvAttention(
            d_model=d_model,
            hidden_ff=dim_feedforward
        )

        # The "second layer" is just one CustomTransformerEncoderMoELayer
        # If you want more layers, you can wrap them in a loop or use CustomTransformerEncoderMoE
        self.second_layer = CustomTransformerEncoderMoELayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps,
            num_experts=num_experts,
            gating_noise=gating_noise
        )

    def forward(self, src, frac):
        """
        src:  [B, T]  integer indices of elements
        frac: [B, T]  fractional amounts, 0 => padding
        """
        # embed
        x = self.embed(src)  # => [B, T, d_model]

        # first layer conv attention
        out1 = self.first_conv_attn(x, frac)  # => [B, T, d_model]

        # second layer normal transformer
        src_key_padding_mask = (frac == 0)    # True => ignore
        out2 = self.second_layer(out1, src_mask=None, 
                                 src_key_padding_mask=src_key_padding_mask)
        return out2


# --------------------------------------------------------------------------
# 6) "Madani" model that uses the new encoder (2-phase conv -> normal)
# --------------------------------------------------------------------------
class Madani(nn.Module):
    def __init__(self,
                 out_dims=3,
                 d_model=512,
                 heads=4,
                 compute_device=None,
                 residual_nn='roost',
                 num_experts=4,
                 gating_noise=0.1,):
        super().__init__()
        self.avg = True
        self.out_dims = out_dims
        self.d_model = d_model
        self.heads = heads
        self.N = 4
        self.compute_device = compute_device

        # Use our special 2-layer encoder
        self.encoder = EncoderConvThenTransformer(
            d_model=self.d_model,
            nhead=self.heads,
            dim_feedforward=2048,
            dropout=0.1,
            N=self.N,
            layer_norm_eps=1e-5,
            num_experts=num_experts,
            gating_noise=gating_noise,
            compute_device=self.compute_device
        )

        # Output MLP
        if residual_nn == 'roost':
            self.out_hidden = [1024, 512, 256, 128]
        else:
            self.out_hidden = [256, 128]

        self.output_nn = ResidualNetwork(self.d_model, self.out_dims, self.out_hidden)

    def forward(self, src, frac):
        """
        src: [B, T] element indices
        frac: [B, T] fractional amounts
        """
        # 1) run the 2-phase conv + normal Transformer
        enc_output = self.encoder(src, frac)  # => [B, T, d_model]

        # 2) pass through final MLP
        #    we mask out positions where src==0
        mask = (src == 0).unsqueeze(-1).repeat(1, 1, self.out_dims)  # [B, T, out_dims]
        output = self.output_nn(enc_output)  # => [B, T, out_dims]

        if self.avg:
            # Mean-pool valid positions
            output = output.masked_fill(mask, 0)
            valid_counts = (~mask).sum(dim=1)  # [B, out_dims]
            output = output.sum(dim=1) / valid_counts

            # If you are chunking into (regression, logits), do that here
            output, logits = output.chunk(2, dim=-1)
            probability = torch.sigmoid(logits)
            output = output * probability
            # (Adjust this logic as needed for your tasks)

        return output
# %%
if __name__ == '__main__':
    model = Madani()
