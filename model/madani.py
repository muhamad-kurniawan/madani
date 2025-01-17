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
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

# class CustomMultiHeadAttention(nn.Module):
class CustomMultiHeadAttentionStoich(nn.Module):
    """
    Multi-Head Attention that does:
        1) compute dotprod_logits = QK^T
        2) compute stoich_bias based on frac_j - frac_i
        3) z-score normalize each separately
        4) combine them as gamma * dotprod_logits + delta * stoich_bias
    """
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

        # Two per-head stoich parameters for positive/negative difference
        self.alpha_pos = nn.Parameter(torch.zeros(nhead))
        self.alpha_neg = nn.Parameter(torch.zeros(nhead))

        # Weighted mix parameters for final combination
        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.delta = nn.Parameter(torch.tensor(0.2))

    def forward(
        self,
        query,               # [B, T, d_model]
        key,                 # [B, T, d_model]
        value,               # [B, T, d_model]
        frac,                # [B, T]
        key_padding_mask=None,  
        attn_mask=None,
        add_frac_bias=None
    ):
        """
        Args:
            query: [B, T, d_model]
            key:   [B, T, d_model]
            value: [B, T, d_model]
            frac:  [B, T], stoichiometric fractions
        """
        B, T, _ = query.shape

        # 1) Q, K, V
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # 2) Reshape [B, T, d_model] => [B, nhead, T, head_dim]
        Q = Q.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.nhead, self.head_dim).transpose(1, 2)

        # 3) Standard dot-product => shape [B, nhead, T, T]
        dotprod_logits = torch.matmul(Q, K.transpose(-1, -2)) * self.scale

        # 4) Stoichiometric difference => D = frac_j - frac_i => [B, T, T]
        frac_i = frac.unsqueeze(-1)  # => [B, T, 1]
        frac_j = frac.unsqueeze(1)   # => [B, 1, T]
        D = (frac_j - frac_i)(frac_i*frac_j)          # => [B, T, T]

        # Positive & negative split
        D_pos = F.relu(D)      # zero out negatives
        D_neg = -F.relu(-D)    # zero out positives

        # expand => [B, nhead, T, T]
        alpha_pos = self.alpha_pos.view(1, self.nhead, 1, 1)
        alpha_neg = self.alpha_neg.view(1, self.nhead, 1, 1)
        stoich_bias = alpha_pos * D_pos.unsqueeze(1) + alpha_neg * D_neg.unsqueeze(1)

        # 5) (Optional) z-score normalize each of dotprod_logits, stoich_bias
        #    We'll do a simple approach: mean & std across the last dimension (-1)
        #    i.e., each [B, nhead, T, T] is normalized T_k-dimension wise.
        dotprod_logits = self._zscore_last_dim(dotprod_logits)
        stoich_bias = self._zscore_last_dim(stoich_bias)

        # 6) Weighted mix
        if add_frac_bias:
            attn_weights = self.gamma * dotprod_logits + self.delta * stoich_bias
        else:
            attn_weights = self.gamma * dotprod_logits
        # 7) Masks
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask

        if key_padding_mask is not None:
            expanded_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(expanded_mask, float('-inf'))

        # 8) Softmax
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # 9) Weighted sum of V => [B, nhead, T, head_dim]
        out = torch.matmul(attn_probs, V)

        # 10) Reshape => [B, T, d_model]
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)

        # 11) Final projection
        out = self.out_proj(out)
        return out

    @staticmethod
    def _zscore_last_dim(tensor, eps=1e-5):
        """
        z-score normalize across the last dimension of shape [B, nhead, T, T].
        i.e. for each (B, nhead, T_q) slice, we standardize across T_k:
          t' = (t - mean_tk) / (std_tk + eps)
        """
        # shape of tensor => [B, nhead, T_q, T_k]
        # we want mean & std over the last dimension => T_k
        mean = tensor.mean(dim=-1, keepdim=True)
        std = tensor.std(dim=-1, keepdim=True)
        return (tensor - mean) / (std + eps)
        
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

        # Experts
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
        """
        B, T, d = x.shape
        # 1) Gating logits
        gating_logits = self.gate(x)  # => [B, T, num_experts]

        # 2) Noise
        if self.training and self.gating_noise > 0.0:
            noise = torch.randn_like(gating_logits) * self.gating_noise
            gating_logits = gating_logits + noise

        # 3) Softmax
        gating_scores = F.softmax(gating_logits, dim=-1)

        # 4) Top-2 gating
        top2_scores, top2_experts = gating_scores.topk(2, dim=-1)  
        out = torch.zeros_like(x)

        for rank in range(2):
            expert_idx = top2_experts[..., rank]   
            gating_weight = top2_scores[..., rank]  

            for e_idx in range(self.num_experts):
                mask = (expert_idx == e_idx)  
                if mask.any():
                    selected_x = x[mask]          # shape [N, d]
                    expert_out = self.experts[e_idx](selected_x) 
                    weight = gating_weight[mask].unsqueeze(-1)
                    weighted_out = weight * expert_out
                    out[mask] += weighted_out
        return out


# class CustomTransformerEncoderMoELayerStoich(nn.Module):
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
        
        # Stoich-based multi-head attention
        self.self_attn = CustomMultiHeadAttentionStoich(d_model, nhead, dropout=dropout)

        # MoE feed-forward
        self.feed_forward = MoEFeedForwardTop2(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_experts=num_experts,
            gating_noise=gating_noise
        )

        # LayerNorms and Dropouts
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # -----------------------------------------------------------------------
        # (A) Fractional MLP or "encoder" to generate a bias vector [d_model]
        # -----------------------------------------------------------------------
        self.frac_mlp = nn.Sequential(
            nn.Linear(1, d_model),   # from scalar fraction to d_model
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, src, frac, src_mask=None, src_key_padding_mask=None,add_frac_bias=False):
        """
        Args:
            src: [B, T, d_model] hidden representation
            frac: [B, T] stoichiometric fractions
        """
        # 1) Stoich-based self-attention
        attn_out = self.self_attn(
            query=src,
            key=src,
            value=src,
            frac=frac,
            key_padding_mask=src_key_padding_mask,
            attn_mask=src_mask,
            add_frac_bias=add_frac_bias
        )
        attn_out = self.dropout1(attn_out)

        # -----------------------------------------------------------------------
        # (B) Compute an extra fraction-based bias to add into the residual
        # -----------------------------------------------------------------------
        if add_frac_bias:
            stoich_bias = self.compute_stoich_bias(frac)  # [B, T, d_model]
        
            

        # 2) Residual connection = src + attn_out + stoich_bias
        
            src = src + attn_out + stoich_bias
        else:
            src = src + attn_out
        src = self.norm1(src)

        # 3) MoE feed-forward
        ff_out = self.feed_forward(src)
        ff_out = self.dropout2(ff_out)
        src = src + ff_out
        src = self.norm2(src)

        return src

    def compute_stoich_bias(self, frac):
        """
        frac: [B, T], each is a scalar fraction
        returns: [B, T, d_model] to be added in the residual
        """
        B, T = frac.shape
        # Flatten: [B*T, 1]
        frac_flat = frac.view(B*T, 1)  # shape [B*T, 1]

        # MLP -> shape [B*T, d_model]
        bias_flat = self.frac_mlp(frac_flat)

        # Reshape back -> [B, T, d_model]
        return bias_flat.view(B, T, -1)


class CustomTransformerEncoderMoEStoich(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])

    def forward(self, src, frac, mask=None, src_key_padding_mask=None):
        out = src
        for i, layer in enumerate(self.layers):
            add_bias = (i==0)
            out = layer(
                out, frac=frac,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                add_frac_bias=add_bias
            )
        return out



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

class EncoderMoEStoichNoFracEnc(nn.Module):
    """
    Stoich-based Encoder with MoE, but WITHOUT the old fractional encoder logic.
    """
    def __init__(self,
                 d_model,
                 N,
                 heads,
                 compute_device=None,
                 num_experts=4,
                 gating_noise=0.0):
        super().__init__()
        self.d_model = d_model
        self.N = N
        self.heads = heads
        self.compute_device = compute_device

        # We still embed the element indices => [B, T, d_model]
        self.embed = Embedder(d_model=self.d_model, compute_device=self.compute_device)

        # Build the stoich-based Transformer
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
        src: [B, T], element indices
        frac: [B, T], stoichiometric fractions
        """
        # 1) Basic element embedding
        x = self.embed(src)  # => [B, T, d_model]

        # 2) Key padding mask => True for padded positions
        src_key_padding_mask = (src == 0)

        # 3) Pass to stoich-based transformer
        #    The stoich attention will do frac_j - frac_i
        out = self.transformer_encoder(
            x, frac=frac,
            mask=None,
            src_key_padding_mask=src_key_padding_mask
        )

        return out



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

        # This uses no fractional encoding
        self.encoder = EncoderMoEStoichNoFracEnc(
            d_model=self.d_model,
            N=self.N,
            heads=self.heads,
            compute_device=self.compute_device,
            num_experts=num_experts,
            gating_noise=gating_noise
        )

        # The existing residual MLP
        if residual_nn == 'roost':
            self.out_hidden = [1024, 512, 256, 128]
            self.output_nn = ResidualNetwork(self.d_model, self.out_dims, self.out_hidden)
        else:
            self.out_hidden = [256, 128]
            self.output_nn = ResidualNetwork(self.d_model, self.out_dims, self.out_hidden)

    def forward(self, src, frac):
        """
        src: [B, T] element indices
        frac: [B, T] stoichiometric fractions
        """
        # Pass to the stoich-based encoder
        output = self.encoder(src, frac)   # => [B, T, d_model]

        # Build a mask for final output if needed
        mask = (src == 0).unsqueeze(-1).repeat(1, 1, self.out_dims)
        output = self.output_nn(output)

        # If you do averaging or chunking:
        if self.avg:
            output = output.masked_fill(mask, 0)
            # average over T dimension
            output = output.sum(dim=1) / (~mask).sum(dim=1)
            # split into final predictions + logits if you like
            output, logits = output.chunk(2, dim=-1)
            probability = torch.ones_like(output)
            probability[:, :logits.shape[-1]] = torch.sigmoid(logits)
            output = output * probability

        return output
# %%
if __name__ == '__main__':
    model = Madani()

