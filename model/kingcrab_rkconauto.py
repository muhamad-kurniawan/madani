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
            nn.Linear(dims[i], dims[i + 1])
            for i in range(len(dims) - 1)
        ])
        self.res_fcs = nn.ModuleList([
            nn.Linear(dims[i], dims[i + 1], bias=False)
            if (dims[i] != dims[i + 1])
            else nn.Identity()
            for i in range(len(dims) - 1)
        ])
        self.acts = nn.ModuleList([
            nn.LeakyReLU()
            for _ in range(len(dims) - 1)
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
    def __init__(self, d_model, compute_device=None):
        super().__init__()
        self.d_model = d_model
        self.compute_device = compute_device

        elem_dir = 'madani/data/element_properties'
        mat2vec = f'{elem_dir}/mat2vec.csv'  # element embedding
        mat2vec = f'{elem_dir}/onehot.csv'
        mat2vec = f'{elem_dir}/oliynyk.csv'
        mat2vec = f'{elem_dir}/magpie.csv'
        # mat2vec = f'{elem_dir}/jarvis.csv'
        
        # cbfv = pd.read_csv(mat2vec, index_col=0).values
        cbfv = pd.read_csv(mat2vec, index_col=0)
        # cbfv = ((cbfv-cbfv.mean())/cbfv.std()).values
        cbfv = ((cbfv-cbfv.min())/(cbfv.max()-cbfv.min())).values
        feat_size = cbfv.shape[-1]
        self.fc_mat2vec = nn.Linear(feat_size, d_model).to(self.compute_device)
        zeros = np.zeros((1, feat_size))
        cat_array = np.concatenate([zeros, cbfv])
        cat_array = torch.as_tensor(cat_array, dtype=data_type_torch)
        self.cbfv = nn.Embedding.from_pretrained(cat_array).to(self.compute_device, dtype=data_type_torch)

    def forward(self, src):
        mat2vec_emb = self.cbfv(src)  # [B, T, feat_size]
        x_emb = self.fc_mat2vec(mat2vec_emb)  # [B, T, d_model]
        return x_emb


# %%
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
        # x is assumed to be a fractional amount tensor of shape [B, T]
        x = torch.clamp(x, min=1/self.resolution)
        frac_idx = torch.round(x * self.resolution).to(dtype=torch.long) - 1
        out = self.pe[frac_idx]  # [B, T, d_model//2]
        return out


# %%
class GlobalRankedFeatureSelector(nn.Module):
    """
    Global feature selector that learns a mask over the feature dimension.
    It produces a binary mask (using a binary concrete, Gumbel–Sigmoid relaxation)
    that is applied uniformly to all tokens, so that the same feature indices are kept/dropped.
    """
    def __init__(self, input_dim, k, init_temp=5.0, final_temp=0.1):
        """
        Args:
            input_dim (int): The dimension of each token (d_model).
            k (int): The number of features to keep.
            init_temp (float): The initial temperature for the concrete relaxation.
            final_temp (float): The final temperature.
        """
        super().__init__()
        self.input_dim = input_dim
        self.k = k
        self.logits = nn.Parameter(torch.zeros(input_dim))  # Global scoring for each feature.
        self.init_temp = init_temp
        self.final_temp = final_temp
        self.current_temp = init_temp

    def forward(self, x):
        """
        x: Tensor of shape [B, T, input_dim]
        Returns:
            x_masked: x with features not selected (according to the mask) zeroed out.
        """
        # Sample Gumbel noise for each feature (shape: [input_dim])
        noise = -torch.log(-torch.log(torch.rand_like(self.logits) + 1e-20) + 1e-20)
        # Compute soft probabilities using Gumbel–Sigmoid relaxation.
        soft_probs = torch.sigmoid((self.logits + noise) / self.current_temp)
        # Apply a ranking rule: keep only the top-k features.
        kth_value = torch.topk(soft_probs, self.k)[0][-1]
        # Create binary mask: features with probability >= kth_value are kept.
        mask = (soft_probs >= kth_value).float()  # [input_dim]
        mask = mask.view(1, 1, self.input_dim)      # Expand for broadcasting.
        return x * mask

    def update_temperature(self, epoch, total_epochs):
        # Exponential annealing schedule.
        self.current_temp = self.init_temp * ((self.final_temp / self.init_temp) ** (epoch / total_epochs))
    
    def hard_mask(self):
        kth_value = torch.topk(self.logits, self.k)[0][-1]
        return (self.logits >= kth_value).float()


# %%
class Encoder(nn.Module):
    def __init__(self, d_model, N, heads, frac=False, attn=True, compute_device=None, feature_selector_k=None):
        super().__init__()
        self.d_model = d_model
        self.N = N
        self.heads = heads
        self.fractional = frac
        self.attention = attn
        self.compute_device = compute_device

        self.embed = Embedder(d_model=self.d_model, compute_device=self.compute_device)
        # Incorporate global feature selection over the embedding output.
        # If feature_selector_k is defined, we add a feature selector.
        if feature_selector_k is not None:
            self.feature_selector = GlobalRankedFeatureSelector(input_dim=self.d_model,
                                                                 k=feature_selector_k)
        else:
            self.feature_selector = None

        self.pe = FractionalEncoder(self.d_model, resolution=5000, log10=False)
        self.ple = FractionalEncoder(self.d_model, resolution=5000, log10=True)

        self.emb_scaler = nn.Parameter(torch.tensor([1.]))
        self.pos_scaler = nn.Parameter(torch.tensor([1.]))
        self.pos_scaler_log = nn.Parameter(torch.tensor([1.]))

        if self.attention:
            encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model,
                                                       nhead=self.heads,
                                                       dim_feedforward=2048,
                                                       dropout=0.1)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.N)

    def forward(self, src, frac):
        # Get embedding; x shape: [B, T, d_model]
        x = self.embed(src) * 2 ** self.emb_scaler
        
        # Apply the global feature selector if present.
        if self.feature_selector is not None:
            x = self.feature_selector(x)

        mask = frac.unsqueeze(dim=-1)
        mask = torch.matmul(mask, mask.transpose(-2, -1))
        mask[mask != 0] = 1
        src_mask = mask[:, 0] != 1

        pe = torch.zeros_like(x)
        ple = torch.zeros_like(x)
        pe_scaler = 2 ** (1 - self.pos_scaler) ** 2
        ple_scaler = 2 ** (1 - self.pos_scaler_log) ** 2
        pe[:, :, :self.d_model // 2] = self.pe(frac) * pe_scaler
        ple[:, :, self.d_model // 2:] = self.ple(frac) * ple_scaler

        if self.attention:
            x_src = x + pe + ple
            x_src = x_src.transpose(0, 1)
            x = self.transformer_encoder(x_src, src_key_padding_mask=src_mask)
            x = x.transpose(0, 1)

        if self.fractional:
            x = x * frac.unsqueeze(2).repeat(1, 1, self.d_model)

        hmask = mask[:, :, 0:1].repeat(1, 1, self.d_model)
        if mask is not None:
            x = x.masked_fill(hmask == 0, 0)
        return x


# %%
class CrabNet(nn.Module):
    def __init__(self, out_dims=3, d_model=512, N=3, heads=4, compute_device=None, residual_nn='roost', feature_selector_k=None):
        super().__init__()
        self.avg = True
        self.out_dims = out_dims
        self.d_model = d_model
        self.N = N
        self.heads = heads
        self.compute_device = compute_device
        # Pass feature_selector_k to Encoder; if provided, the global ranked feature selector is applied.
        self.encoder = Encoder(d_model=self.d_model,
                               N=self.N,
                               heads=self.heads,
                               compute_device=self.compute_device,
                               feature_selector_k=feature_selector_k)
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
    # Example: Use CrabNet with global feature selection on the embedding.
    # Let's assume the input dimension (after embedding) is d_model=512.
    # We want to keep, for example, 300 features (indices) globally.
    model = CrabNet(out_dims=3, d_model=512, N=3, heads=4, feature_selector_k=30)
    print(model)
