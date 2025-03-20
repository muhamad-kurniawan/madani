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
    Each component is scaled so that the values roughly lie in the range [0.1, 1].
    
    For example, for x = 0.234 and num_digits=3:
        - First component: floor(0.234*10)=2 --> 0.2 (already in [0.1,1])
        - Second component: second digit = floor(0.234*100) - floor(0.234*10)*10 = 3 --> scaled: 3/10*10 = 0.3
        - Third component: third digit = floor(0.234*1000) - floor(0.234*100)*10 = 4 --> scaled: 4/10*100 = 0.4
    """
    components = []
    for i in range(1, num_digits + 1):
        # Multiply x by 10^i
        temp = x * (10 ** i)
        # Get the digit at the i-th decimal place
        digit = torch.floor(temp) - torch.floor(x * (10 ** (i - 1))) * 10
        # Scale: divide by 10 and multiply by 10^(i-1)
        scaled = (digit / 10) * (10 ** (i - 1))
        components.append(scaled.unsqueeze(-1))
    return torch.cat(components, dim=-1)


class LearnableFractionalEncoder(nn.Module):
    """
    Learns an embedding for the fractional number by combining:
      1. An MLP-based embedding from the full fraction.
      2. A digit-split embedding that extracts and scales the first few decimal components.
    
    The two parts are concatenated to yield a final embedding of dimension d_model.
    """
    def __init__(self, d_model, hidden_dim=64, num_digits=3):
        super().__init__()
        self.num_digits = num_digits
        # We reserve (num_digits) dimensions for the split features.
        # The MLP will output (d_model - num_digits) so that after concatenation, the total dimension is d_model.
        mlp_output_dim = d_model - num_digits
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, mlp_output_dim)
        )
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, num_elements) containing fractional values (in [0,1]).
        Returns:
            Tensor of shape (batch_size, num_elements, d_model) where d_model = mlp_output_dim + num_digits.
        """
        # Compute the MLP-based embedding from the full fraction.
        x_unsqueezed = x.unsqueeze(-1)  # shape: (batch_size, num_elements, 1)
        mlp_emb = self.mlp(x_unsqueezed)  # shape: (batch_size, num_elements, d_model - num_digits)
        
        # Compute the split-digit embedding.
        split_emb = split_fraction(x, num_digits=self.num_digits)  # shape: (batch_size, num_elements, num_digits)
        
        # Concatenate along the last dimension.
        combined = torch.cat([mlp_emb, split_emb], dim=-1)  # shape: (batch_size, num_elements, d_model)
        return combined


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
        # Choose what element information the model receives:
        mat2vec = f'{elem_dir}/mat2vec.csv'  # element embedding
        # Other options commented out:
        # mat2vec = f'{elem_dir}/onehot.csv'
        # mat2vec = f'{elem_dir}/random_200.csv'
        # mat2vec = f'{elem_dir}/oliynyk.csv'
        # mat2vec = f'{elem_dir}/magpie.csv'

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
        
        # Replace the original sinusoidal fractional encoders with the new learnable fractional encoder.
        self.frac_encoder = LearnableFractionalEncoder(d_model=self.d_model, hidden_dim=64, num_digits=3)
        
        self.emb_scaler = nn.parameter.Parameter(torch.tensor([1.]))
        # (The original pos_scaler parameters are no longer needed; they are omitted in this version.)

        if self.attention:
            encoder_layer = nn.TransformerEncoderLayer(
                self.d_model, nhead=self.heads, dim_feedforward=2048, dropout=0.1)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.N)

    def forward(self, src, frac):
        # Compute element embeddings.
        x = self.embed(src) * 2**self.emb_scaler
        
        # Create a mask for nonzero fractions.
        mask = frac.unsqueeze(dim=-1)
        mask = torch.matmul(mask, mask.transpose(-2, -1))
        mask[mask != 0] = 1
        src_mask = mask[:, 0] != 1

        # Obtain the learnable fractional embedding.
        frac_emb = self.frac_encoder(frac)  # shape: (batch, num_elements, d_model)

        if self.attention:
            # Add the learnable fractional embedding to the element embeddings.
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
            # use the Roost residual network
            self.out_hidden = [1024, 512, 256, 128]
            self.output_nn = ResidualNetwork(self.d_model, self.out_dims, self.out_hidden)
        else:
            # use a simpler residual network
            self.out_hidden = [256, 128]
            self.output_nn = ResidualNetwork(self.d_model, self.out_dims, self.out_hidden)

    def forward(self, src, frac):
        output = self.encoder(src, frac)
        # Average the "element contribution" at the end.
        mask = (src == 0).unsqueeze(-1).repeat(1, 1, self.out_dims)
        output = self.output_nn(output)  # simple linear
        if self.avg:
            output = output.masked_fill(mask, 0)
            output = output.sum(dim=1) / (~mask).sum(dim=1)
            output, logits = output.chunk(2, dim=-1)
            probability = torch.ones_like(output)
            probability[:, :logits.shape[-1]] = torch.sigmoid(logits)
            output = output * probability
        return output
