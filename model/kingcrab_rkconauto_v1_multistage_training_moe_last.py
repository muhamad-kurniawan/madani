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
                                      if (dims[i] != dims[i+1])
                                      else nn.Identity()
                                      for i in range(len(dims)-1)])
        self.acts = nn.ModuleList([nn.LeakyReLU() for _ in range(len(dims)-1)])
        self.fc_out = nn.Linear(dims[-1], output_dim)

    def forward(self, fea):
        for fc, res_fc, act in zip(self.fcs, self.res_fcs, self.acts):
            fea = act(fc(fea)) + res_fc(fea)
        return self.fc_out(fea)

    def __repr__(self):
        return f'{self.__class__.__name__}'

# --- Embedder and Global Feature Selector ---
class GlobalRankedFeatureSelector(nn.Module):
    """
    Global feature selector that learns a mask over the feature dimension.
    Uses a binary concrete (Gumbel–Sigmoid) relaxation.
    """
    def __init__(self, input_dim, k, init_temp=5.0, final_temp=0.01):
        super().__init__()
        assert k <= input_dim, f"Target k ({k}) must be <= input_dim ({input_dim})."
        self.input_dim = input_dim
        self.k = k
        self.logits = nn.Parameter(torch.randn(input_dim) * 0.01)
        self.init_temp = init_temp
        self.final_temp = final_temp
        self.current_temp = init_temp
        self.last_soft_probs = None

    def forward(self, x):
        noise = -torch.log(-torch.log(torch.rand_like(self.logits).clamp(min=1e-6)) + 1e-6)
        soft_probs = torch.sigmoid((self.logits + noise) / self.current_temp)
        self.last_soft_probs = soft_probs
        k_effective = min(self.k, soft_probs.numel())
        kth_value = torch.topk(soft_probs, k_effective)[0][-1].detach()
        hard_mask = (soft_probs >= kth_value).float()
        mask = hard_mask.detach() - soft_probs.detach() + soft_probs
        mask = mask.view(1, 1, self.input_dim)
        return x * mask

    def update_temperature(self, epoch, total_epochs=1000, feature_selection_phase=0.5):
        if epoch < feature_selection_phase * total_epochs:
            self.current_temp = self.init_temp * ((self.final_temp / self.init_temp) ** (epoch / total_epochs))
        else:
            self.current_temp = 1e-12

    def hard_mask(self):
        k_effective = min(self.k, self.logits.numel())
        kth_value = torch.topk(self.logits, k_effective)[0][-1].detach()
        return (self.logits >= kth_value).float()

class Embedder(nn.Module):
    def __init__(self, d_model, compute_device=None, selector_k=None):
        super().__init__()
        self.d_model = d_model
        self.compute_device = compute_device

        elem_dir = 'madani/data/element_properties'
        mat2vec = f'{elem_dir}/mat2vec.csv'
        df = pd.read_csv(mat2vec, index_col=0, encoding='unicode_escape')
        if df.isnull().values.any():
            print("NaN values detected. Filling with random numbers.")
            random_fill = pd.DataFrame(np.random.randn(*df.shape), index=df.index, columns=df.columns)
            df = df.fillna(random_fill)
        cbfv = df.values
        self.feat_size = cbfv.shape[-1]

        required_rows = 98
        current_rows = cbfv.shape[0]
        if current_rows < required_rows:
            missing = required_rows - current_rows
            print(f"mat2vec has only {current_rows} rows. Filling {missing} missing rows with random numbers.")
            random_fill = np.random.randn(missing, self.feat_size)
            cbfv = np.concatenate([cbfv, random_fill], axis=0)

        if selector_k is not None:
            self.feature_selector = GlobalRankedFeatureSelector(input_dim=self.feat_size, k=selector_k)
        else:
            self.feature_selector = None

        self.selector_scale = 1.0
        self.fc_mat2vec = nn.Linear(self.feat_size, d_model).to(self.compute_device)

        zeros = np.zeros((1, self.feat_size))
        cat_array = np.concatenate([zeros, cbfv])
        cat_array = torch.as_tensor(cat_array, dtype=torch.float32)
        self.cbfv = nn.Embedding.from_pretrained(cat_array) \
            .to(self.compute_device, dtype=torch.float32)

    def forward(self, src):
        x = self.cbfv(src)
        if self.feature_selector is not None:
            selected = self.feature_selector(x)
            x = (1 - self.selector_scale) * x + self.selector_scale * selected
        x_emb = self.fc_mat2vec(x)
        return x_emb

# --- Fractional Encoder ---
class FractionalEncoder(nn.Module):
    """
    Encoding element fractional amount using a "fractional encoding" inspired
    by the positional encoder discussed by Vaswani.
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
        self.register_buffer('pe', pe)

    def forward(self, x):
        if self.log10:
            x = 0.0025 * (torch.log2(x))**2
            x = torch.clamp(x, max=1)
        x = torch.clamp(x, min=1/self.resolution)
        frac_idx = torch.round(x * self.resolution).to(dtype=torch.long) - 1
        out = self.pe[frac_idx]
        return out

# --- Custom Transformer Modules with MoE in the Last Layer ---

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

class MoEFeedForwardTop2(nn.Module):
    """
    MoE feed-forward module with top-2 gating.
    """
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1, num_experts=4, gating_noise=0.0):
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
                nn.Dropout(dropout)
            ) for _ in range(num_experts)
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
            gating_weight = top2_scores[..., rank]    # [B, T]
            for e_idx in range(self.num_experts):
                mask = (expert_idx == e_idx)
                if mask.any():
                    selected_x = x[mask]
                    expert_out = self.experts[e_idx](selected_x)
                    weight = gating_weight[mask].unsqueeze(-1)
                    out[mask] += weight * expert_out
        return out

class CustomTransformerEncoderFFLayer(nn.Module):
    """
    Standard transformer encoder layer with a regular feed-forward block.
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
        return src

class CustomTransformerEncoderMoELayer(nn.Module):
    """
    Transformer encoder layer that uses an MoE feed-forward block.
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 layer_norm_eps=1e-5, num_experts=4, gating_noise=0.0):
        super().__init__()
        self.self_attn = CustomMultiHeadAttention(d_model, nhead, dropout=dropout)
        self.feed_forward = MoEFeedForwardTop2(d_model=d_model,
                                               dim_feedforward=dim_feedforward,
                                               dropout=dropout,
                                               num_experts=num_experts,
                                               gating_noise=gating_noise)
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
        return src

class CustomTransformerEncoderMixedLastMoE(nn.Module):
    """
    Transformer encoder that stacks N-1 standard layers and one final MoE layer.
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 layer_norm_eps=1e-5, num_experts=4, gating_noise=0.0, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.layers.append(
                CustomTransformerEncoderFFLayer(d_model=d_model,
                                                  nhead=nhead,
                                                  dim_feedforward=dim_feedforward,
                                                  dropout=dropout,
                                                  layer_norm_eps=layer_norm_eps)
            )
        if num_layers > 0:
            self.layers.append(
                CustomTransformerEncoderMoELayer(d_model=d_model,
                                                 nhead=nhead,
                                                 dim_feedforward=dim_feedforward,
                                                 dropout=dropout,
                                                 layer_norm_eps=layer_norm_eps,
                                                 num_experts=num_experts,
                                                 gating_noise=gating_noise)
            )

    def forward(self, src, mask=None, src_key_padding_mask=None):
        out = src
        for layer in self.layers:
            out = layer(out, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        return out

# --- Encoder using the Mixed MoE Transformer ---
class Encoder(nn.Module):
    def __init__(self, d_model, N, heads, frac=False, attn=True,
                 compute_device=None, selector_k=None, num_experts=4, gating_noise=0.1):
        super().__init__()
        self.d_model = d_model
        self.N = N
        self.heads = heads
        self.fractional = frac
        self.attention = attn
        self.selector_k = selector_k
        self.compute_device = compute_device

        self.embed = Embedder(d_model=self.d_model,
                              compute_device=self.compute_device,
                              selector_k=self.selector_k)
        self.pe = FractionalEncoder(self.d_model, resolution=5000, log10=False)
        self.ple = FractionalEncoder(self.d_model, resolution=5000, log10=True)

        self.emb_scaler = nn.parameter.Parameter(torch.tensor([1.]))
        self.pos_scaler = nn.parameter.Parameter(torch.tensor([1.]))
        self.pos_scaler_log = nn.parameter.Parameter(torch.tensor([1.]))

        if self.attention:
            self.transformer_encoder = CustomTransformerEncoderMixedLastMoE(
                d_model=self.d_model,
                nhead=self.heads,
                dim_feedforward=1024,
                dropout=0.1,
                layer_norm_eps=1e-5,
                num_experts=num_experts,
                gating_noise=gating_noise,
                num_layers=self.N
            )

    def forward(self, src, frac):
        x = self.embed(src) * 2**self.emb_scaler
        # Create a key padding mask from frac: assume positions with frac==0 are pads.
        # Here we expect src of shape [B, T] and frac of shape [B, T]
        src_key_padding_mask = (frac == 0)

        pe = torch.zeros_like(x)
        ple = torch.zeros_like(x)
        pe_scaler = 2**((1 - self.pos_scaler)**2)
        ple_scaler = 2**((1 - self.pos_scaler_log)**2)
        pe[:, :, :self.d_model//2] = self.pe(frac) * pe_scaler
        ple[:, :, self.d_model//2:] = self.ple(frac) * ple_scaler

        if self.attention:
            x_src = x + pe + ple
            # Our custom encoder expects [B, T, d_model]
            x = self.transformer_encoder(x_src, mask=None, src_key_padding_mask=src_key_padding_mask)

        if self.fractional:
            x = x * frac.unsqueeze(2).repeat(1, 1, self.d_model)

        # Create a mask to zero out padded positions
        hmask = (frac == 0).unsqueeze(-1).repeat(1, 1, self.d_model)
        x = x.masked_fill(hmask, 0)
        return x

# --- CrabNet Model ---
class CrabNet(nn.Module):
    def __init__(self, out_dims=3, d_model=512, N=3, heads=4,
                 compute_device=None, residual_nn='roost', feature_selector_k=None,
                 num_experts=4, gating_noise=0.1):
        super().__init__()
        self.avg = True
        self.out_dims = out_dims
        self.d_model = d_model
        self.N = N
        self.heads = heads
        self.selector_k = feature_selector_k
        self.compute_device = compute_device
        self.encoder = Encoder(d_model=self.d_model,
                               N=self.N,
                               heads=self.heads,
                               compute_device=self.compute_device,
                               selector_k=self.selector_k,
                               num_experts=num_experts,
                               gating_noise=gating_noise)
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
    # Example: using d_model=512, N=3 layers, and selecting 200 features.
    model = CrabNet(out_dims=3, d_model=512, N=3, heads=4, feature_selector_k=200)
    print(model)
