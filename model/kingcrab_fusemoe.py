import numpy as np
import pandas as pd

import torch
from torch import nn

# =============================================================
# FUSE-MOE MODULES
# =============================================================
class MoEElementFusion(nn.Module):
    """Laplace‑gated Mixture‑of‑Experts fusion layer
    that adaptively merges *n_views* alternative embeddings
    of the *same element token*.
    Each view has its own router (per‑modality routing) while
    all views share the expert pool, following FuseMoE (NeurIPS 2024).
    """
    def __init__(self, n_views: int, d_model: int,
                 n_experts: int = 16, k: int = 4):
        super().__init__()
        self.n_views   = n_views
        self.d_model   = d_model
        self.n_experts = n_experts
        self.k         = k  # top‑k experts per token

        # One linear projection per view ⇒ shared dim d_model
        self.proj = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(n_views)
        ])

        # Per‑modality routers  (simple linear → expert logits)
        self.routers = nn.ModuleList([
            nn.Linear(d_model, n_experts, bias=False)
            for _ in range(n_views)
        ])

        # Expert FFNs (shared across views)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Linear(4 * d_model, d_model)
            ) for _ in range(n_experts)
        ])

        # Expert keys for Laplace gating (initialised orthogonally)
        self.register_parameter("expert_keys", nn.Parameter(
            torch.empty(n_experts, d_model)))
        nn.init.orthogonal_(self.expert_keys)

    # ‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑
    def laplace_gate(self, h: torch.Tensor, router: nn.Linear):
        """Computes top‑k routing weights using the negative
        L2 distance between token *h* and learned expert keys."""
        # h: (B, T, d_model)
        logits = -torch.cdist(
            router(h).unsqueeze(2),  # shape (B,T,1,d_model)
            self.expert_keys.unsqueeze(0).unsqueeze(0)  # (1,1,E,d_model)
        ).squeeze(3)  # → (B, T, E)
        topk_val, topk_idx = logits.topk(self.k, dim=-1)  # (B,T,k)
        gate = (topk_val - topk_val.logsumexp(-1, keepdim=True)).exp()  # softmax
        return gate, topk_idx

    # ‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑
    def forward(self, view_list):
        # view_list: list[n_views] each shaped (B,T,d_model)
        assert len(view_list) == self.n_views, "#views mismatch"
        B, T, _ = view_list[0].shape
        fused = torch.zeros(B, T, self.d_model, device=view_list[0].device)

        for i, v in enumerate(view_list):
            h = self.proj[i](v)
            gate, idx = self.laplace_gate(h, self.routers[i])  # (B,T,k)
            # Gather expert outputs
            expert_outs = []
            for j in range(self.k):
                sel_exp = self.experts[idx[..., j]]  # ModuleList supports indexing tensor?

            # PyTorch cannot index ModuleList with tensor, use loop
            out = torch.zeros_like(h)
            for token_pos in range(T):
                g_vec = gate[:, token_pos, :]  # (B,k)
                idx_vec = idx[:, token_pos, :]  # (B,k)
                # Process batch element‑wise for clarity (could be vectorised)
                batch_out = []
                for b in range(B):
                    token_sum = 0
                    for j in range(self.k):
                        expert_j = self.experts[idx_vec[b, j]]
                        token_sum += g_vec[b, j:j+1] * expert_j(h[b:b+1, token_pos])
                    batch_out.append(token_sum)
                out[:, token_pos] = torch.cat(batch_out, dim=0)
            fused += out
        return fused


# =============================================================
# THE ORIGINAL CRABNET COMPONENTS (unchanged unless noted)
# =============================================================
RNG_SEED = 42
torch.manual_seed(RNG_SEED)
np.random.seed(RNG_SEED)
data_type_torch = torch.float32

class ResidualNetwork(nn.Module):
    """Feed‑forward Residual Neural Network as seen in Roost."""
    def __init__(self, input_dim, output_dim, hidden_layer_dims):
        super().__init__()
        dims = [input_dim] + hidden_layer_dims
        self.fcs = nn.ModuleList([
            nn.Linear(dims[i], dims[i+1]) for i in range(len(dims)-1)
        ])
        self.res_fcs = nn.ModuleList([
            nn.Linear(dims[i], dims[i+1], bias=False) if dims[i] != dims[i+1] else nn.Identity()
            for i in range(len(dims)-1)
        ])
        self.acts = nn.ModuleList([nn.LeakyReLU() for _ in range(len(dims)-1)])
        self.fc_out = nn.Linear(dims[-1], output_dim)
    def forward(self, fea):
        for fc, res_fc, act in zip(self.fcs, self.res_fcs, self.acts):
            fea = act(fc(fea)) + res_fc(fea)
        return self.fc_out(fea)

class Embedder(nn.Module):
    def __init__(self, d_model, compute_device=None):
        super().__init__()
        self.d_model = d_model
        self.compute_device = compute_device
        elem_dir = 'madani/data/element_properties'
        mat2vec = f'{elem_dir}/mat2vec.csv'
        cbfv = pd.read_csv(mat2vec, index_col=0).values
        feat_size = cbfv.shape[-1]
        self.fc_mat2vec = nn.Linear(feat_size, d_model).to(self.compute_device)
        zeros = np.zeros((1, feat_size))
        cat_array = np.concatenate([zeros, cbfv])
        cat_array = torch.as_tensor(cat_array, dtype=data_type_torch)
        self.cbfv = nn.Embedding.from_pretrained(cat_array).to(self.compute_device, dtype=data_type_torch)
    def forward(self, src):
        mat2vec_emb = self.cbfv(src)
        return self.fc_mat2vec(mat2vec_emb)

class FractionalEncoder(nn.Module):
    def __init__(self, d_model, resolution=100, log10=False, compute_device=None):
        super().__init__()
        self.d_model = d_model // 2
        self.resolution = resolution
        self.log10 = log10
        self.compute_device = compute_device
        x = torch.linspace(0, self.resolution - 1, self.resolution, requires_grad=False).view(self.resolution, 1)
        fraction = torch.linspace(0, self.d_model - 1, self.d_model, requires_grad=False).view(1, self.d_model).repeat(self.resolution, 1)
        pe = torch.zeros(self.resolution, self.d_model)
        pe[:, 0::2] = torch.sin(x / torch.pow(50, 2 * fraction[:, 0::2] / self.d_model))
        pe[:, 1::2] = torch.cos(x / torch.pow(50, 2 * fraction[:, 1::2] / self.d_model))
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x.clone()
        if self.log10:
            x = 0.0025 * (torch.log2(x)) ** 2
        x = torch.clamp(x, min=1 / self.resolution)
        frac_idx = torch.round(x * self.resolution).to(dtype=torch.long) - 1
        return self.pe[frac_idx]

class Encoder(nn.Module):
    def __init__(self, d_model, N, heads, frac=False, attn=True, compute_device=None,
                 use_moe=True, n_moe_views=3, n_experts=16, k=4):
        super().__init__()
        self.d_model = d_model
        self.N = N
        self.heads = heads
        self.fractional = frac
        self.attention = attn
        self.compute_device = compute_device
        self.embed = Embedder(d_model=self.d_model, compute_device=self.compute_device)
        self.pe  = FractionalEncoder(self.d_model, resolution=5000, log10=False)
        self.ple = FractionalEncoder(self.d_model, resolution=5000, log10=True)
        self.emb_scaler = nn.parameter.Parameter(torch.tensor([1.]))
        self.pos_scaler = nn.parameter.Parameter(torch.tensor([1.]))
        self.pos_scaler_log = nn.parameter.Parameter(torch.tensor([1.]))
        # --- MoE Fusion layer ---------------------------------
        self.use_moe = use_moe
        if self.use_moe:
            self.moe_fusion = MoEElementFusion(n_views=n_moe_views,
                                               d_model=d_model,
                                               n_experts=n_experts,
                                               k=k)
        # -------------------------------------------------------
        if self.attention:
            encoder_layer = nn.TransformerEncoderLayer(d_model, nhead=self.heads, dim_feedforward=2048, dropout=0.1)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.N)
    # -----------------------------------------------------------
    def forward(self, src, frac):
        # View‑1: learnable element ID embedding
        x_id = self.embed(src) * 2 ** self.emb_scaler  # (B,T,d_model)
        # Positional / fractional sinusoidal
        pe_scaler  = 2 ** (1 - self.pos_scaler) ** 2
        ple_scaler = 2 ** (1 - self.pos_scaler_log) ** 2
        view_pe  = torch.zeros_like(x_id)
        view_ple = torch.zeros_like(x_id)
        view_pe[:, :, :self.d_model // 2]  = self.pe(frac)  * pe_scaler
        view_ple[:, :, self.d_model // 2:] = self.ple(frac) * ple_scaler
        # Fuse the three views
        if self.use_moe:
            x = self.moe_fusion([x_id, view_pe, view_ple])
        else:
            x = x_id + view_pe + view_ple
        # Mask & Transformer
        mask = frac.unsqueeze(dim=-1)
        mask = torch.matmul(mask, mask.transpose(-2, -1))
        mask[mask != 0] = 1
        src_mask = mask[:, 0] != 1
        if self.attention:
            x_src = x.transpose(0, 1)
            x = self.transformer_encoder(x_src, src_key_padding_mask=src_mask)
            x = x.transpose(0, 1)
        if self.fractional:
            x = x * frac.unsqueeze(2).repeat(1, 1, self.d_model)
        hmask = mask[:, :, 0:1].repeat(1, 1, self.d_model)
        x = x.masked_fill(hmask == 0, 0)
        return x

class CrabNet(nn.Module):
    def __init__(self, out_dims=3, d_model=512, N=3, heads=4,
                 compute_device=None, residual_nn='roost',
                 use_moe=True):
        super().__init__()
        self.avg = True
        self.out_dims = out_dims
        self.d_model = d_model
        self.N = N
        self.heads = heads
        self.compute_device = compute_device
        self.encoder = Encoder(d_model=self.d_model, N=self.N, heads=self.heads,
                               compute_device=self.compute_device, use_moe=use_moe)
        if residual_nn == 'roost':
            self.out_hidden = [1024, 512, 256, 128]
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

# =============================================================
if __name__ == '__main__':
    B, T = 4, 10
    src  = torch.randint(1, 100, (B, T))  # fake atomic numbers
    frac = torch.rand(B, T)
    frac = frac / frac.sum(dim=1, keepdim=True)  # normalise

    model = CrabNet(use_moe=True)
    out = model(src, frac)
    print('Output shape:', out.shape)
