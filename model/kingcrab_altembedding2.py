import numpy as np
import pandas as pd
import torch
from torch import nn

# Set seeds for reproducibility
RNG_SEED = 42
torch.manual_seed(RNG_SEED)
np.random.seed(RNG_SEED)
data_type_torch = torch.float32


def split_fraction(x, num_digits=3):
    """
    Splits each fractional value in x into its first `num_digits` decimal components.
    Each digit is scaled so that its value is roughly in the range [0.1, 1].

    For example, for x = 0.234 and num_digits=3:
      - First component: floor(0.234*10)=2 --> 0.2
      - Second component: floor(0.234*100) - floor(0.234*10)*10 = 3 --> scaled: (3/10)*10 = 0.3
      - Third component: floor(0.234*1000) - floor(0.234*100)*10 = 4 --> scaled: (4/10)*100 = 0.4
    """
    components = []
    for i in range(1, num_digits + 1):
        # Multiply x by 10^i
        temp = x * (10 ** i)
        # Extract the digit at the i-th decimal place
        digit = torch.floor(temp) - torch.floor(x * (10 ** (i - 1))) * 10
        # Scale: (digit/10) multiplied by 10^(i-1)
        scaled = (digit / 10) * (10 ** (i - 1))
        components.append(scaled.unsqueeze(-1))
    return torch.cat(components, dim=-1)


class MultiBranchFractionalEncoder(nn.Module):
    """
    Multi-branch fractional encoder:
      - The full fractional input (a scalar between 0 and 1) is fed into an MLP ("full branch").
      - The input is also split into its first few decimal components.
      - Each split digit is processed by its own small network.
      - The outputs from the full branch and the split branches are concatenated.
    
    The final output dimension is d_model.
    """
    def __init__(self, d_model, full_hidden_dim=96, split_hidden_dim=16, num_splits=3):
        super().__init__()
        self.num_splits = num_splits
        # Calculate the output dimension for the full branch so that:
        # d_model = d_full (full branch) + num_splits * split_hidden_dim (each digit branch)
        d_full = d_model - num_splits * split_hidden_dim
        self.full_mlp = nn.Sequential(
            nn.Linear(1, full_hidden_dim),
            nn.ReLU(),
            nn.Linear(full_hidden_dim, d_full)
        )
        # Create separate networks for each split digit.
        self.split_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, split_hidden_dim),
                nn.ReLU(),
                nn.Linear(split_hidden_dim, split_hidden_dim)
            ) for _ in range(num_splits)
        ])
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, num_elements) with fractional values in [0, 1].
        Returns:
            Tensor of shape (batch_size, num_elements, d_model) that concatenates
            the full branch embedding and the embeddings for each split digit.
        """
        # Process the full fractional value.
        x_unsqueezed = x.unsqueeze(-1)  # shape: (batch_size, num_elements, 1)
        full_emb = self.full_mlp(x_unsqueezed)  # shape: (batch_size, num_elements, d_full)
        
        # Split the fraction into its decimal components.
        split_values = split_fraction(x, num_digits=self.num_splits)  # shape: (batch_size, num_elements, num_splits)
        split_embs = []
        for i in range(self.num_splits):
            # Get the i-th digit (as a scalar) and pass through its own network.
            digit = split_values[..., i].unsqueeze(-1)  # shape: (batch_size, num_elements, 1)
            split_emb = self.split_mlps[i](digit)         # shape: (batch_size, num_elements, split_hidden_dim)
            split_embs.append(split_emb)
        
        # Concatenate the full branch embedding with all split branch embeddings.
        combined = torch.cat([full_emb] + split_embs, dim=-1)
        return combined


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


class Embedder(nn.Module):
    def __init__(self, d_model, compute_device=None):
        super().__init__()
        self.d_model = d_model
        self.compute_device = compute_device

        elem_dir = 'madani/data/element_properties'
        # Choose which element information the model receives:
        mat2vec = f'{elem_dir}/mat2vec.csv'
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


class Encoder(nn.Module):
    def __init__(self, d_model, N, heads, frac=False, attn=True, compute_device=None):
        super().__init__()
        self.d_model = d_model
        self.N = N
        self.heads = heads
        self.fractional = frac
        self.attention = attn
        self.compute_device = compute_device
        self.embed = Embedder(d_model=self.d_model, compute_device=self.compute_device)
        
        # Use the multi-branch fractional encoder.
        # For instance, with d_model = 512, we allocate 16 dims for each split (3 splits),
        # and the remaining 512 - 48 = 464 dims for the full branch.
        self.frac_encoder = MultiBranchFractionalEncoder(d_model=self.d_model,
                                                         full_hidden_dim=64,
                                                         split_hidden_dim=16,
                                                         num_splits=3)
        
        self.emb_scaler = nn.parameter.Parameter(torch.tensor([1.]))

        if self.attention:
            encoder_layer = nn.TransformerEncoderLayer(
                self.d_model, nhead=self.heads, dim_feedforward=2048, dropout=0.1)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.N)

    def forward(self, src, frac):
        x = self.embed(src) * 2**self.emb_scaler
        mask = frac.unsqueeze(dim=-1)
        mask = torch.matmul(mask, mask.transpose(-2, -1))
        mask[mask != 0] = 1
        src_mask = mask[:, 0] != 1

        # Get the multi-branch fractional embedding.
        frac_emb = self.frac_encoder(frac)  # shape: (batch, num_elements, d_model)

        if self.attention:
            # Incorporate the fractional information by adding the embedding.
            x_src = x + frac_emb
            x_src = x_src.transpose(0, 1)
            x = self.transformer_encoder(x_src, src_key_padding_mask=src_mask)
            x = x.transpose(0, 1)

        if self.fractional:
            x = x * frac.unsqueeze(2).repeat(1, 1, self.d_model)

        hmask = mask[:, :, 0:1].repeat(1, 1, self.d_model)
        if mask is not None:
            x = x.masked_fill(hmask == 0, 0)
        return x


class CrabNet(nn.Module):
    def __init__(self, out_dims=3, d_model=512, N=3, heads=4, compute_device=None, residual_nn='roost'):
        super().__init__()
        self.avg = True
        self.out_dims = out_dims
        self.d_model = d_model
        self.N = N
        self.heads = heads
        self.compute_device = compute_device
        self.encoder = Encoder(d_model=self.d_model, N=self.N, heads=self.heads, compute_device=self.compute_device)
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
