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

class GlobalRankedFeatureSelector(nn.Module):
    """
    Global feature selector that learns a mask over the feature dimension.
    Instead of applying a hard threshold with a straight-through estimator,
    this version uses the soft probabilities directly as a continuous mask.
    """
    def __init__(self, input_dim, k, init_temp=5.0, final_temp=0.01):
        """
        Args:
            input_dim (int): Number of features per token.
            k (int): Target number of features to keep (used only for reference or later evaluation).
            init_temp (float): Initial temperature.
            final_temp (float): Final temperature.
        """
        super().__init__()
        assert k <= input_dim, f"Target k ({k}) must be less than or equal to input_dim ({input_dim})."
        self.input_dim = input_dim
        self.k = k
        # Initialize logits with small random values to break symmetry.
        self.logits = nn.Parameter(torch.randn(input_dim) * 0.01)
        self.init_temp = init_temp
        self.final_temp = final_temp
        self.current_temp = init_temp

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [B, T, input_dim]
        Returns:
            Tensor of shape [B, T, input_dim] after applying the soft feature mask.
        """
        # Sample Gumbel noise for each feature.
        noise = -torch.log(-torch.log(torch.rand_like(self.logits).clamp(min=1e-6)) + 1e-6)
        # Compute soft probabilities via Gumbel–Sigmoid relaxation.
        soft_probs = torch.sigmoid((self.logits + noise) / self.current_temp)
        # Use soft_probs directly as a continuous mask (without thresholding)
        mask = soft_probs.view(1, 1, self.input_dim)  # Expand for broadcasting.
        return x * mask

    def update_temperature(self, epoch, total_epochs, feature_selection_phase):
        # Exponential annealing schedule.
        if feature_selection_phase * total_epochs >= epoch:
            self.current_temp = self.init_temp * ((self.final_temp / self.init_temp) ** (epoch / total_epochs))
        else:
            self.current_temp = 1e-12

    def hard_mask(self):
        # This method remains available if you later wish to obtain a discrete mask.
        k_effective = min(self.k, self.logits.numel())
        kth_value = torch.topk(self.logits, k_effective)[0][-1].detach()
        return (self.logits >= kth_value).float()



# %%
class Embedder(nn.Module):
    def __init__(self, d_model, compute_device=None, selector_k=None):
        super().__init__()
        self.d_model = d_model
        self.compute_device = compute_device

        elem_dir = 'madani/data/element_properties'
        mat2vec = f'{elem_dir}/mat2vec.csv'  # element embedding file
        mat2vec = f'{elem_dir}/oliynyk.csv'
        mat2vec = f'{elem_dir}/magpie.csv'
        # mat2vec = f'{elem_dir}/jarvis.csv'

        # Load the mat2vec CSV file into a pandas DataFrame.
        df = pd.read_csv(mat2vec, index_col=0, encoding='unicode_escape')
        
        # Check for NaN values and replace them with random numbers.
        if df.isnull().values.any():
            print("NaN values detected in CSV. Filling them with random numbers.")
            # Create a DataFrame of random numbers with the same shape.
            random_fill = pd.DataFrame(np.random.randn(*df.shape), 
                                       index=df.index, 
                                       columns=df.columns)
            df = df.fillna(random_fill)
        
        cbfv = df.values
        self.feat_size = cbfv.shape[-1]

        # If the number of rows is less than 98, fill the missing rows with random numbers.
        required_rows = 118
        current_rows = cbfv.shape[0]
        if current_rows < required_rows:
            missing = required_rows - current_rows
            print(f"mat2vec has only {current_rows} rows. Filling {missing} missing rows with random numbers.")
            random_fill = abs(np.random.randn(missing, self.feat_size))
            cbfv = np.concatenate([cbfv, random_fill], axis=0)

        print(f'feat_sizes : {self.feat_size}')
        
        # Create a global feature selector to be applied on the raw element features.
        if selector_k is not None:
            self.feature_selector = GlobalRankedFeatureSelector(input_dim=self.feat_size, k=selector_k)
        else:
            self.feature_selector = None

        # fc_mat2vec projects the (possibly masked) raw vector to d_model.
        self.fc_mat2vec = nn.Linear(self.feat_size, d_model).to(self.compute_device)

        zeros = np.zeros((1, self.feat_size))
        cat_array = np.concatenate([zeros, cbfv])
        cat_array = torch.as_tensor(cat_array, dtype=data_type_torch)
        self.cbfv = nn.Embedding.from_pretrained(cat_array) \
            .to(self.compute_device, dtype=data_type_torch)

        
    def forward(self, src):
        # Obtain raw element features [B, T, feat_size]
        x = self.cbfv(src)
        # Apply the global feature selector before projection, if provided.
        if self.feature_selector is not None:
            x = self.feature_selector(x)
        # Project the (masked) features to the d_model space.
        x_emb = self.fc_mat2vec(x)
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
        if self.log10:
            x = 0.0025 * (torch.log2(x))**2
            # clamp x[x > 1] = 1
            x = torch.clamp(x, max=1)
            # x = 1 - x  # for sinusoidal encoding at x=0
        # clamp x[x < 1/self.resolution] = 1/self.resolution
        x = torch.clamp(x, min=1/self.resolution)
        frac_idx = torch.round(x * (self.resolution)).to(dtype=torch.long) - 1
        out = self.pe[frac_idx]

        return out


# %%
class Encoder(nn.Module):
    def __init__(self,
                 d_model,
                 N,
                 heads,
                 frac=False,
                 attn=True,
                 compute_device=None, selector_k=None):
        super().__init__()
        self.d_model = d_model
        self.N = N
        self.heads = heads
        self.fractional = frac
        self.attention = attn
        self.selector_k = selector_k
        self.compute_device = compute_device
        self.embed = Embedder(d_model=self.d_model,
                              compute_device=self.compute_device, selector_k=self.selector_k)
        self.pe = FractionalEncoder(self.d_model, resolution=5000, log10=False)
        self.ple = FractionalEncoder(self.d_model, resolution=5000, log10=True)

        self.emb_scaler = nn.parameter.Parameter(torch.tensor([1.]))
        self.pos_scaler = nn.parameter.Parameter(torch.tensor([1.]))
        self.pos_scaler_log = nn.parameter.Parameter(torch.tensor([1.]))

        if self.attention:
            encoder_layer = nn.TransformerEncoderLayer(self.d_model,
                                            nhead=self.heads,
                                            dim_feedforward=2048,
                                            dropout=0.1)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer,
                                                             num_layers=self.N)

    def forward(self, src, frac):
        x = self.embed(src) * 2**self.emb_scaler
        mask = frac.unsqueeze(dim=-1)
        mask = torch.matmul(mask, mask.transpose(-2, -1))
        mask[mask != 0] = 1
        src_mask = mask[:, 0] != 1

        pe = torch.zeros_like(x)
        ple = torch.zeros_like(x)
        pe_scaler = 2**(1-self.pos_scaler)**2
        ple_scaler = 2**(1-self.pos_scaler_log)**2
        pe[:, :, :self.d_model//2] = self.pe(frac) * pe_scaler
        ple[:, :, self.d_model//2:] = self.ple(frac) * ple_scaler

        if self.attention:
            x_src = x + pe + ple
            x_src = x_src.transpose(0, 1)
            x = self.transformer_encoder(x_src,
                                         src_key_padding_mask=src_mask)
            x = x.transpose(0, 1)

        if self.fractional:
            x = x * frac.unsqueeze(2).repeat(1, 1, self.d_model)

        hmask = mask[:, :, 0:1].repeat(1, 1, self.d_model)
        if mask is not None:
            x = x.masked_fill(hmask == 0, 0)
        # print(f'emb_scaler{self.emb_scaler}')
        return x


# %%
class CrabNet(nn.Module):
    def __init__(self,
                 out_dims=3,
                 d_model=512,
                 N=3,
                 heads=4,
                 compute_device=None,
                 residual_nn='roost', feature_selector_k=None):
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
                               compute_device=self.compute_device, selector_k=self.selector_k)
        if residual_nn == 'roost':
            # use the Roost residual network
            self.out_hidden = [1024, 512, 256, 128]
            self.output_nn = ResidualNetwork(self.d_model,
                                             self.out_dims,
                                             self.out_hidden)
        else:
            # use a simpler residual network
            self.out_hidden = [256, 128]
            self.output_nn = ResidualNetwork(self.d_model,
                                             self.out_dims,
                                             self.out_hidden)

    def forward(self, src, frac):
        output = self.encoder(src, frac)

        # average the "element contribution" at the end
        # mask so you only average "elements"
        mask = (src == 0).unsqueeze(-1).repeat(1, 1, self.out_dims)
        output = self.output_nn(output)  # simple linear
        if self.avg:
            output = output.masked_fill(mask, 0)
            output = output.sum(dim=1)/(~mask).sum(dim=1)
            output, logits = output.chunk(2, dim=-1)
            probability = torch.ones_like(output)
            probability[:, :logits.shape[-1]] = torch.sigmoid(logits)
            output = output * probability

        return output


# %%
if __name__ == '__main__':
    # For instance, assume d_model=512 for the transformer part,
    # and we wish to select 200 features from the raw element vector.
    model = CrabNet(out_dims=3, d_model=512, N=3, heads=4, feature_selector_k=200)
    print(model)
