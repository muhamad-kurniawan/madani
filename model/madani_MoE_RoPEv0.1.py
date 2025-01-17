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
        # print(f'x0 {x}')
        if self.log10:
            x = 0.0025 * (torch.log2(x))**2
            # clamp x[x > 1] = 1
            x = torch.clamp(x, max=1)
            # x = 1 - x  # for sinusoidal encoding at x=0
        # clamp x[x < 1/self.resolution] = 1/self.resolution
        x = torch.clamp(x, min=1/self.resolution)
        # print(f'x {x}')
        frac_idx = torch.round(x * (self.resolution)).to(dtype=torch.long) - 1
        # print(f'frac_idx {frac_idx}')
        out = self.pe[frac_idx]

        return out
    
import torch
import torch.nn as nn

def rope_encoding(x, freqs):
    """
    Applies rotary position embedding to the last dimension of x.
    x:     (..., 2*d_model_half)
    freqs: (..., d_model_half) of angles (theta)
    
    We treat x as chunked into pairs: x[..., 0::2], x[..., 1::2] (cosine/sine pairs).
    Then we rotate them by angles in freqs.
    """
    d_model = x.shape[-1]
    assert d_model % 2 == 0, "Last dimension must be even for RoPE"

    d_half = d_model // 2
    x_left, x_right = x[..., :d_half], x[..., d_half:]  # chunk into two halves

    # The standard RoPE mix:
    #   x_rotated_left  = x_left * cos(freqs) - x_right * sin(freqs)
    #   x_rotated_right = x_left * sin(freqs) + x_right * cos(freqs)
    #
    # You can also do a single pass if you prefer:
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)

    x_rotated_left  = x_left  * cos - x_right * sin
    x_rotated_right = x_left  * sin + x_right * cos

    return torch.cat([x_rotated_left, x_rotated_right], dim=-1)


def build_rope_frequencies(num_positions, d_half, base=10000.0):
    """
    Builds a frequency buffer for RoPE, shaped [num_positions, d_half].
    Typically for token positions, you do something like:
      theta = position / (base^(2*i/d_model)) .
    For fractions, we might adapt the 'position' to be the fraction * some_scale.
    """
    # positions: shape [num_positions]
    positions = torch.arange(num_positions).float()  # 0..(num_positions-1)

    # frequencies: shape [d_half]
    freq_seq = torch.arange(d_half).float() / d_half
    # denominator: base^(2*i/d_model)
    freq_div = base ** freq_seq  # shape [d_half]

    # shape => [num_positions, d_half]
    # each row is position / freq_div
    freqs = positions.unsqueeze(-1) / freq_div

    return freqs

class FractionalEncoderRoPE(nn.Module):
    """
    Rotary embedding for fraction scalars. 
    Instead of a big table of sin/cos (like sinusoidal encoding),
    we dynamically build the rotation angles based on fraction.

    NOTE: This example treats each fraction value as if it were a "position",
    scaled by 'base'. The returned embedding has shape [B, T, d_model].
    """
    def __init__(self,
                 d_model,
                 base=10000.0,
                 log10=False,
                 resolution=5000,
                 compute_device=None):
        super().__init__()
        # We'll do a half-split approach for RoPE: half dims get rotated
        assert d_model % 2 == 0, "d_model must be even to use RoPE easily"
        self.d_model = d_model
        self.d_model_half = d_model // 2

        self.base = base
        self.log10 = log10
        self.resolution = resolution
        self.compute_device = compute_device

        # Optionally you could register a small parameter or buffer to scale fraction
        # but let's keep it simple here
        self.register_buffer("dummy", torch.empty(0))  # just for device handling

    def forward(self, fractions):
        """
        fractions: [B, T], each entry is a float in [0,1] (or similar range).
        We produce an embedding [B, T, d_model].
        """
        # 1) Possibly transform fractions
        #    (Your code used 'log10' or resolution clamping, etc.)
        x = fractions.clone()
        if self.log10:
            # Example of your old logic
            x = 0.0025 * (torch.log2(x))**2
            x = torch.clamp(x, max=1.0)
        x = torch.clamp(x, min=1.0/self.resolution)

        # 2) Turn x into "angle" => shape [B, T, d_model_half]
        #    For standard RoPE, angle = position / base^(2*i/d_model).
        #    We'll do a rough approximation by letting "position" = (x * scale).
        #    E.g., angle[b,t,i] = ( x[b,t] * some_scale ) / base^( i / d_model_half )
        #
        #    One simplistic approach is: angle[b,t] = x[b,t] * (some large integer).
        #    Then we vary i in [0, d_model_half).
        #
        #    We'll build freq_div, shape [d_model_half]
        i_idx = torch.arange(self.d_model_half, device=self.dummy.device, dtype=torch.float32)
        freq_div = (self.base ** (i_idx / self.d_model_half))  # shape [d_model_half]
        
        # shape => [B, T, d_model_half]
        # "position" = x * resolution (completely up to you)
        angle = (x * self.resolution).unsqueeze(-1) / freq_div

        # 3) Build a "blank" embedding prior to rotation
        #    We'll do something like a linear ramp for the "unrotated" part,
        #    or just do an identity basis for the two halves.
        #    Typically, you might have an input vector from another embed
        #    that you rotate in Q or K. Here, we must artificially create a 2*d_model_half dimension to rotate.
        emb = torch.zeros((*angle.shape[:-1], self.d_model), device=angle.device)
        # For example, let half be "x" and the other half be "0" before rotation:
        # shape => [B, T, d_model_half]
        left_half  = torch.ones_like(angle)
        right_half = torch.zeros_like(angle)
        unrotated  = torch.cat([left_half, right_half], dim=-1)  # shape [B, T, d_model]

        # 4) Apply the RoPE transformation
        #    We rotate 'unrotated' by 'angle'.
        emb_rotated = rope_encoding(unrotated, angle)

        return emb_rotated



import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomMultiHeadAttentionStoich(nn.Module):
    """
    Multi-Head Attention with a per-head bias based on stoichiometric differences.
    Each head has two learned parameters: alpha_pos[h], alpha_neg[h].
    The attention logits get a bias:
        alpha_pos[h] * max(frac_j - frac_i, 0) + alpha_neg[h] * min(frac_j - frac_i, 0).
    """
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

        # Scale factor for dot-product
        self.scale = self.head_dim ** -0.5

        # Two learnable bias parameters per head (for positive & negative differences)
        self.alpha_pos = nn.Parameter(torch.zeros(nhead))  # [nhead]
        self.alpha_neg = nn.Parameter(torch.zeros(nhead))  # [nhead]

    def forward(
        self,
        query,              # [B, T, d_model]
        key,                # [B, T, d_model]
        value,              # [B, T, d_model]
        frac,               # [B, T] stoichiometric fractions
        key_padding_mask=None,  # [B, T] True means "ignore"
        attn_mask=None          # optional attn mask [T, T] or [B, T, T]
    ):
        """
        Args:
            query: [B, T, d_model]
            key:   [B, T, d_model]
            value: [B, T, d_model]
            frac:  [B, T] stoichiometric fractions for each element
            key_padding_mask: [B, T] (True = masked out)
            attn_mask: optional [T, T] or [B, T, T]
        Returns:
            out: [B, T, d_model]
        """
        B, T, _ = query.shape

        # 1) Project Q, K, V
        Q = self.W_q(query)  # [B, T, d_model]
        K = self.W_k(key)    # [B, T, d_model]
        V = self.W_v(value)  # [B, T, d_model]

        # 2) Reshape to [B, nhead, T, head_dim]
        Q = Q.view(B, T, self.nhead, self.head_dim).transpose(1, 2)  # -> [B, nhead, T, head_dim]
        K = K.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.nhead, self.head_dim).transpose(1, 2)

        # 3) Compute scaled dot-product attention logits
        # shape = [B, nhead, T_q, T_k]
        attn_weights = torch.matmul(Q, K.transpose(-1, -2)) * self.scale

        # 4) Compute stoichiometric difference: D[b, i, j] = frac[b, j] - frac[b, i]
        # shape => [B, T, T]
        frac_i = frac.unsqueeze(2)    # [B, T, 1]
        frac_j = frac.unsqueeze(1)    # [B, 1, T]
        D = frac_j - frac_i           # [B, T, T]

        # positive part & negative part
        #  D_pos = relu(D), D_neg = "negative part" of D
        D_pos = torch.relu(D)             # [B, T, T] (zeros out negative)
        D_neg = -torch.relu(-D)           # [B, T, T] (zeros out positive)

        # 5) Expand for nheads
        # alpha_pos, alpha_neg => shape [nhead]
        # We'll expand to [B, nhead, T, T]
        alpha_pos = self.alpha_pos.view(1, self.nhead, 1, 1)  # [1, nhead, 1, 1]
        alpha_neg = self.alpha_neg.view(1, self.nhead, 1, 1)  # [1, nhead, 1, 1]

        D_pos_exp = D_pos.unsqueeze(1)  # [B, 1, T, T]
        D_neg_exp = D_neg.unsqueeze(1)  # [B, 1, T, T]

        stoich_bias = alpha_pos * D_pos_exp + alpha_neg * D_neg_exp
        # shape => [B, nhead, T, T]

        # 6) Add stoichiometry bias to attention logits
        attn_weights = attn_weights + stoich_bias

        # 7) Apply optional masks
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask

        if key_padding_mask is not None:
            expanded_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
            attn_weights = attn_weights.masked_fill(expanded_mask, float('-inf'))

        # 8) Softmax and dropout
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # 9) Weighted sum of values
        out = torch.matmul(attn_probs, V)  # [B, nhead, T, head_dim]

        # 10) Reshape back to [B, T, d_model]
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)

        # 11) Final linear projection
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

class CustomTransformerEncoderMoELayerStoich(nn.Module):
    def __init__(self, 
                 d_model, 
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 layer_norm_eps=1e-5,
                 num_experts=4,
                 gating_noise=0.0):
        super().__init__()
        # Replace standard attention with Stoich-based attention
        self.self_attn = CustomMultiHeadAttentionStoich(d_model, nhead, dropout=dropout)

        # MoE feed-forward (top-2 gating)
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
        """
        Args:
            src: [B, T, d_model]
            frac: [B, T]
            src_mask: optional [T, T] or [B, T, T]
            src_key_padding_mask: [B, T]
        """
        # 1) Stoich-based self-attention
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



class CustomTransformerEncoderMoEStoich(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        # Omit final LayerNorm for simplicity

    def forward(self, src, frac, mask=None, src_key_padding_mask=None):
        """
        Args:
            src: [B, T, d_model]
            frac: [B, T]
            mask: optional attn mask
            src_key_padding_mask: [B, T]
        """
        output = src
        for layer in self.layers:
            output = layer(output, frac=frac,
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

class EncoderMoEStoich(nn.Module):
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
        self.embed = Embedder(d_model=self.d_model,
                              compute_device=self.compute_device)

        # --- OLD: sinusoidal fractional encoders ---
        # self.pe = FractionalEncoder(self.d_model, resolution=5000, log10=False)
        # self.ple = FractionalEncoder(self.d_model, resolution=5000, log10=True)

        # --- NEW: Replace with RoPE-based fraction encoders (or just one) ---
        self.pe = FractionalEncoderRoPE(self.d_model,
                                        base=10000.0,
                                        log10=False,
                                        resolution=5000,
                                        compute_device=self.compute_device)
        self.ple = FractionalEncoderRoPE(self.d_model,
                                         base=10000.0,
                                         log10=True,
                                         resolution=5000,
                                         compute_device=self.compute_device)

        self.emb_scaler = nn.parameter.Parameter(torch.tensor([1.]))
        self.pos_scaler = nn.parameter.Parameter(torch.tensor([1.]))
        self.pos_scaler_log = nn.parameter.Parameter(torch.tensor([1.]))

        if self.attention:
            # Use the Stoich-based transformer
            encoder_layer = CustomTransformerEncoderMoELayerStoich(
                d_model=self.d_model,
                nhead=self.heads,
                dim_feedforward=2048,
                dropout=0.1,
                layer_norm_eps=1e-5,
                num_experts=num_experts,
                gating_noise=gating_noise
            )
            self.transformer_encoder = CustomTransformerEncoderMoEStoich(
                encoder_layer=encoder_layer,
                num_layers=self.N
            )

    def forward(self, src, frac):
        """
        src: [B, T] with element indices
        frac: [B, T] stoichiometric fractions
        """
        x = self.embed(src) * 2**self.emb_scaler

        src_key_padding_mask = (frac == 0)
        
        # Build RoPE embeddings
        pe_raw  = self.pe(frac)   # [B, T, d_model]
        ple_raw = self.ple(frac)  # [B, T, d_model]

        # Scale them
        pe_scaler  = 2**((1 - self.pos_scaler)     **2)
        ple_scaler = 2**((1 - self.pos_scaler_log) **2)
        pe  = pe_raw  * pe_scaler
        ple = ple_raw * ple_scaler

        # Add them
        if self.attention:
            x_src = x + pe + ple
            x = self.transformer_encoder(
                x_src,
                frac=frac,  # stoichiometry
                mask=None,
                src_key_padding_mask=src_key_padding_mask
            )

        if self.fractional:
            x = x * frac.unsqueeze(2).repeat(1, 1, self.d_model)

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

        # Use the Stoich version of EncoderMoE
        self.encoder = EncoderMoEStoich(
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
        """
        src: [B, T], each element is an index (0=pad)
        frac: [B, T], stoichiometric fractions
        """
        output = self.encoder(src, frac)
        mask = (src == 0).unsqueeze(-1).repeat(1, 1, self.out_dims)
        output = self.output_nn(output)

        if self.avg:
            output = output.masked_fill(mask, 0)
            output = output.sum(dim=1) / (~mask).sum(dim=1)
            # separate final outputs from logits
            output, logits = output.chunk(2, dim=-1)
            probability = torch.ones_like(output)
            probability[:, :logits.shape[-1]] = torch.sigmoid(logits)
            output = output * probability

        return output




# %%
if __name__ == '__main__':
    model = Madani()

