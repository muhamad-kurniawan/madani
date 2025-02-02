import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

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
        super(ResidualNetwork, self).__init__()
        dims = [input_dim] + hidden_layer_dims
        self.fcs = nn.ModuleList([nn.Linear(dims[i], dims[i+1])
                                  for i in range(len(dims)-1)])
        self.res_fcs = nn.ModuleList([nn.Linear(dims[i], dims[i+1], bias=False)
                                      if dims[i] != dims[i+1] else nn.Identity()
                                      for i in range(len(dims)-1)])
        self.acts = nn.ModuleList([nn.LeakyReLU() for _ in range(len(dims)-1)])
        self.fc_out = nn.Linear(dims[-1], output_dim)

    def forward(self, fea):
        for fc, res_fc, act in zip(self.fcs, self.res_fcs, self.acts):
            fea = act(fc(fea)) + res_fc(fea)
        return self.fc_out(fea)

    def __repr__(self):
        return f'{self.__class__.__name__}'


class Embedder(nn.Module):
    def __init__(self, d_model, compute_device=None):
        super().__init__()
        self.d_model = d_model
        self.compute_device = compute_device

        elem_dir = 'madani/data/element_properties'
        mat2vec = f'{elem_dir}/mat2vec.csv'  # element embedding

        cbfv = pd.read_csv(mat2vec, index_col=0).values
        feat_size = cbfv.shape[-1]
        self.fc_mat2vec = nn.Linear(feat_size, d_model).to(self.compute_device)

        zeros = np.zeros((1, feat_size))
        cat_array = np.concatenate([zeros, cbfv])
        cat_array = torch.as_tensor(cat_array, dtype=data_type_torch)
        self.cbfv = nn.Embedding.from_pretrained(cat_array)\
            .to(self.compute_device, dtype=data_type_torch)

    def forward(self, src):
        mat2vec_emb = self.cbfv(src)          # [B, T, feat_size]
        x_emb = self.fc_mat2vec(mat2vec_emb)    # [B, T, d_model]
        return x_emb


class FractionalEncoder(nn.Module):
    """
    Encoding element fractional amount using a "fractional encoding" inspired by the positional encoder.
    https://arxiv.org/abs/1706.03762
    """
    def __init__(self, d_model, resolution=100, log10=False, compute_device=None):
        super().__init__()
        self.d_model = d_model // 2
        self.resolution = resolution
        self.log10 = log10
        self.compute_device = compute_device

        x = torch.linspace(0, self.resolution - 1, self.resolution, requires_grad=False).view(self.resolution, 1)
        fraction = torch.linspace(0, self.d_model - 1, self.d_model, requires_grad=False).view(1, self.d_model)
        fraction = fraction.repeat(self.resolution, 1)

        pe = torch.zeros(self.resolution, self.d_model)
        pe[:, 0::2] = torch.sin(x / torch.pow(50, 2 * fraction[:, 0::2] / self.d_model))
        pe[:, 1::2] = torch.cos(x / torch.pow(50, 2 * fraction[:, 1::2] / self.d_model))
        pe = self.register_buffer('pe', pe)

    def forward(self, x):
        if self.log10:
            x = 0.0025 * (torch.log2(x))**2
            x = torch.clamp(x, max=1)
        x = torch.clamp(x, min=1/self.resolution)
        frac_idx = torch.round(x * self.resolution).to(dtype=torch.long) - 1
        out = self.pe[frac_idx]  # [B, T, d_model//2]
        return out


class CustomMultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        B, T, _ = query.shape
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        Q = Q.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        attn_weights = torch.matmul(Q, K.transpose(-1, -2)) * self.scale
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask
        if key_padding_mask is not None:
            expanded_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(expanded_mask, float('-inf'))
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)
        out = torch.matmul(attn_probs, V)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.out_proj(out)
        return out


###############################################################################
# New: MoE Feed-Forward module with Adaptive/Dynamic Capacity Routing
###############################################################################
class MoEFeedForwardAdaptive(nn.Module):
    """
    MoE feed-forward module that dynamically selects a variable number of experts per token.
    
    For each token:
      - It computes gating scores and sorts them in descending order.
      - It selects experts until the cumulative sum exceeds a threshold (p_threshold),
        with an optional maximum number (k_max).
      - It unsorts the selection mask back to the original expert order and normalizes
        the effective gating weights.
      
    Returns both the output and an auxiliary load-balancing loss.
    """
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1,
                 num_experts=4, gating_noise=0.0, p_threshold=0.9, k_max=None):
        super().__init__()
        self.num_experts = num_experts
        self.gating_noise = gating_noise
        self.p_threshold = p_threshold
        self.k_max = k_max if k_max is not None else num_experts
        self.gate = nn.Linear(d_model, num_experts)
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
        # x: [B, T, d]
        B, T, d = x.shape
        gating_logits = self.gate(x)  # [B, T, num_experts]
        if self.training and self.gating_noise > 0.0:
            gating_logits = gating_logits + torch.randn_like(gating_logits) * self.gating_noise
        gating_scores = F.softmax(gating_logits, dim=-1)  # [B, T, num_experts]

        # Compute load-balancing loss as before.
        importance = gating_scores.sum(dim=(0, 1))  # [num_experts]
        load_loss = self.num_experts * (importance**2).sum() / (importance.sum()**2 + 1e-10)

        # Adaptive selection: for each token, select experts until cumulative score >= p_threshold.
        # Sort gating scores in descending order.
        sorted_scores, sorted_indices = torch.sort(gating_scores, dim=-1, descending=True)  # [B, T, E]
        cumsum_scores = sorted_scores.cumsum(dim=-1)  # [B, T, E]

        # Create a mask selecting experts until the cumulative sum exceeds p_threshold.
        # Ensure at least one expert is selected by forcing the first index to True.
        mask_sorted = cumsum_scores <= self.p_threshold
        mask_sorted[..., 0] = True

        # Limit the maximum number of experts per token.
        expert_range = torch.arange(self.num_experts, device=x.device).view(1, 1, self.num_experts)
        mask_range = expert_range < self.k_max
        mask_sorted = mask_sorted & mask_range

        # Unsort: create a mask in the original expert order.
        selection_mask = torch.zeros_like(mask_sorted, dtype=torch.float32)
        selection_mask.scatter_(dim=-1, index=sorted_indices, src=mask_sorted.float())
        # Compute effective gating: zero out unselected experts.
        effective_gating = gating_scores * selection_mask
        # Normalize effective gating weights so that they sum to 1 per token.
        effective_sum = effective_gating.sum(dim=-1, keepdim=True) + 1e-10
        effective_gating = effective_gating / effective_sum

        # Route tokens through each expert weighted by the effective gating.
        out = torch.zeros_like(x)
        for e_idx in range(self.num_experts):
            weight_e = effective_gating[..., e_idx].unsqueeze(-1)  # [B, T, 1]
            expert_out = self.experts[e_idx](x)  # [B, T, d]
            out += weight_e * expert_out

        return out, load_loss


###############################################################################
# Standard Transformer Layer (for non-MoE layers)
###############################################################################
class CustomTransformerEncoderFFLayer(nn.Module):
    """
    A standard Transformer Encoder layer that uses a regular feed-forward block.
    Returns (output, auxiliary_loss) where aux_loss is zero.
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, layer_norm_eps=1e-5):
        super().__init__()
        self.self_attn = CustomMultiHeadAttention(d_model, nhead, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        attn_out = self.self_attn(src, src, src,
                                  key_padding_mask=src_key_padding_mask,
                                  attn_mask=src_mask)
        src = src + self.dropout1(attn_out)
        src = self.norm1(src)
        ff_out = self.feed_forward(src)
        src = src + self.dropout2(ff_out)
        src = self.norm2(src)
        return src, 0.0


###############################################################################
# MoE Transformer Layer (for the FIRST layer)
###############################################################################
class CustomTransformerEncoderMoELayer(nn.Module):
    """
    A Transformer Encoder layer that uses an MoE feed-forward block with adaptive routing in its first layer.
    Returns (output, auxiliary_loss).
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 layer_norm_eps=1e-5, num_experts=4, gating_noise=0.0, p_threshold=0.9, k_max=None):
        super().__init__()
        self.self_attn = CustomMultiHeadAttention(d_model, nhead, dropout=dropout)
        self.feed_forward = MoEFeedForwardAdaptive(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_experts=num_experts,
            gating_noise=gating_noise,
            p_threshold=p_threshold,
            k_max=k_max
        )
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        attn_out = self.self_attn(src, src, src,
                                  key_padding_mask=src_key_padding_mask,
                                  attn_mask=src_mask)
        src = src + self.dropout1(attn_out)
        src = self.norm1(src)
        ff_out, aux_loss = self.feed_forward(src)
        src = src + self.dropout2(ff_out)
        src = self.norm2(src)
        return src, aux_loss


###############################################################################
# Mixed Encoder: First layer is MoE (with adaptive routing), remaining layers are standard.
# It accumulates auxiliary load balancing losses.
###############################################################################
class CustomTransformerEncoderMixedFirstMoE(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 layer_norm_eps=1e-5, num_experts=4, gating_noise=0.0, num_layers=3,
                 p_threshold=0.9, k_max=None):
        super().__init__()
        self.layers = nn.ModuleList()
        # First layer uses adaptive MoE
        if num_layers > 0:
            self.layers.append(
                CustomTransformerEncoderMoELayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    layer_norm_eps=layer_norm_eps,
                    num_experts=num_experts,
                    gating_noise=gating_noise,
                    p_threshold=p_threshold,
                    k_max=k_max
                )
            )
        # The remaining layers are standard.
        for _ in range(num_layers - 1):
            self.layers.append(
                CustomTransformerEncoderFFLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    layer_norm_eps=layer_norm_eps
                )
            )

    def forward(self, src, mask=None, src_key_padding_mask=None):
        total_aux_loss = 0.0
        out = src
        for layer in self.layers:
            out, aux_loss = layer(out, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            total_aux_loss += aux_loss
        return out, total_aux_loss


###############################################################################
# Encoder MoE - uses MoE only in the FIRST layer and returns auxiliary loss.
###############################################################################
class EncoderMoE(nn.Module):
    def __init__(self, d_model, N, heads, frac=False, attn=True,
                 compute_device=None, num_experts=4, gating_noise=0.0, p_threshold=0.9, k_max=None):
        super().__init__()
        self.d_model = d_model
        self.N = N
        self.heads = heads
        self.fractional = frac
        self.attention = attn
        self.compute_device = compute_device

        self.embed = Embedder(d_model=self.d_model, compute_device=self.compute_device)
        self.pe = FractionalEncoder(self.d_model, resolution=5000, log10=False)
        self.ple = FractionalEncoder(self.d_model, resolution=5000, log10=True)

        self.emb_scaler = nn.parameter.Parameter(torch.tensor([1.]))
        self.pos_scaler = nn.parameter.Parameter(torch.tensor([1.]))
        self.pos_scaler_log = nn.parameter.Parameter(torch.tensor([1.]))

        if self.attention:
            self.transformer_encoder = CustomTransformerEncoderMixedFirstMoE(
                d_model=self.d_model,
                nhead=self.heads,
                dim_feedforward=1024,
                dropout=0.1,
                layer_norm_eps=1e-5,
                num_experts=num_experts,
                gating_noise=gating_noise,
                num_layers=self.N,
                p_threshold=p_threshold,
                k_max=k_max
            )

    def forward(self, src, frac):
        x = self.embed(src) * 2**self.emb_scaler
        src_key_padding_mask = (frac == 0)
        pe = torch.zeros_like(x)
        ple = torch.zeros_like(x)
        pe_scaler = 2**((1 - self.pos_scaler)**2)
        ple_scaler = 2**((1 - self.pos_scaler_log)**2)
        pe[:, :, :self.d_model//2] = self.pe(frac) * pe_scaler
        ple[:, :, self.d_model//2:] = self.ple(frac) * ple_scaler

        if self.attention:
            x_src = x + pe + ple
            x, aux_loss = self.transformer_encoder(
                x_src,
                mask=None,
                src_key_padding_mask=src_key_padding_mask
            )
        else:
            aux_loss = 0.0

        if self.fractional:
            x = x * frac.unsqueeze(2).repeat(1, 1, self.d_model)
        hmask = (frac == 0).unsqueeze(-1).repeat(1, 1, self.d_model)
        x = x.masked_fill(hmask, 0)
        return x, aux_loss


###############################################################################
# Madani Model
###############################################################################
class Madani(nn.Module):
    def __init__(self, out_dims=3, d_model=512, N=3, heads=4, compute_device=None,
                 residual_nn='roost', num_experts=4, gating_noise=0.1, p_threshold=0.9, k_max=None):
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
            gating_noise=gating_noise,
            p_threshold=p_threshold,
            k_max=k_max
        )

        if residual_nn == 'roost':
            self.out_hidden = [1024, 512, 256, 128]
            self.output_nn = ResidualNetwork(self.d_model, self.out_dims, self.out_hidden)
        else:
            self.out_hidden = [256, 128]
            self.output_nn = ResidualNetwork(self.d_model, self.out_dims, self.out_hidden)

    def forward(self, src, frac):
        output, aux_loss = self.encoder(src, frac)
        mask = (src == 0).unsqueeze(-1).repeat(1, 1, self.out_dims)
        output = self.output_nn(output)
        if self.avg:
            output = output.masked_fill(mask, 0)
            output = output.sum(dim=1) / (~mask).sum(dim=1)
            output, logits = output.chunk(2, dim=-1)
            probability = torch.ones_like(output)
            probability[:, :logits.shape[-1]] = torch.sigmoid(logits)
            output = output * probability
        return output, aux_loss


# %%
if __name__ == '__main__':
    model = Madani()
    print(model)
