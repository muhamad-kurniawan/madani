"""
CrabNet‑FuseMoE (stable‑b2 2025‑05‑07)
======================================
• Adds `self.compute_device` alias in `CrabNet.__init__` and a `@property`
  wrapper so external code expects either `model.device` or `model.compute_device`.
• No functional changes to MoE; previous vectorised gate remains.
"""

import numpy as np
import pandas as pd
import torch
from torch import nn

data_type_torch = torch.float32

def get_default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------
# Residual Network
# ------------------------------
class ResidualNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden):
        super().__init__()
        dims = [input_dim] + hidden
        self.fcs = nn.ModuleList([nn.Linear(dims[i], dims[i+1])
                                  for i in range(len(dims)-1)])
        self.res = nn.ModuleList([nn.Linear(dims[i], dims[i+1], bias=False)
                                   if dims[i]!=dims[i+1] else nn.Identity()
                                   for i in range(len(dims)-1)])
        self.act = nn.ModuleList([nn.LeakyReLU() for _ in range(len(dims)-1)])
        self.out = nn.Linear(dims[-1], output_dim)
    def forward(self,x):
        for f,r,a in zip(self.fcs,self.res,self.act):
            x = a(f(x))+r(x)
        return self.out(x)

# ------------------------------
# Multi‑descriptor embedder
# ------------------------------
class MultiDescriptorEmbedder(nn.Module):
    def __init__(self, d_model, elem_dir, device):
        super().__init__()
        self.device = device
        # files = {"mat2vec":"mat2vec.csv","magpie":"magpie.csv","oliy":"oliynyk.csv"}
        files = {"mat2vec":"mat2vec.csv","magpie":"mat2vec.csv","oliy":"mat2vec.csv"}

        self.embs, self.projs = nn.ModuleDict(), nn.ModuleDict()
        for k,f in files.items():
            arr = pd.read_csv(f"{elem_dir}/{f}",index_col=0).values
            feat = arr.shape[-1]
            pad = np.zeros((1,feat))
            weight = torch.as_tensor(np.vstack([pad,arr]),dtype=data_type_torch)
            self.embs[k]  = nn.Embedding.from_pretrained(weight,freeze=False)
            self.projs[k] = nn.Linear(feat,d_model)
        self.move_to_device()
    def move_to_device(self):
        for m in self.modules():
            m.to(self.device)
    def forward(self,Z):
        views = []
        for k in self.embs:
            e = self.embs[k](Z)
            views.append(self.projs[k](e))
        return views

# ------------------------------
# Fraction encoders (unchanged)
# ------------------------------
class FractionalEncoder(nn.Module):
    def __init__(self,d_model,resolution=5000,log10=False):
        super().__init__()
        self.d_half = d_model//2
        self.resolution = resolution
        self.log10 = log10
        x = torch.linspace(0,resolution-1,resolution).view(resolution,1)
        f = torch.linspace(0,self.d_half-1,self.d_half).view(1,self.d_half).repeat(resolution,1)
        pe = torch.zeros(resolution,self.d_half)
        pe[:,0::2]=torch.sin(x/torch.pow(50,2*f[:,0::2]/self.d_half))
        pe[:,1::2]=torch.cos(x/torch.pow(50,2*f[:,1::2]/self.d_half))
        self.register_buffer('pe',pe)
    def forward(self,r):
        if self.log10:
            r = torch.clamp(0.0025*(torch.log2(r))**2,max=1)
        r = torch.clamp(r,min=1/self.resolution)
        idx = torch.round(r*self.resolution).long()-1
        return self.pe[idx]

# ------------------------------
# MoE Fusion Layer (vectorised)
# ------------------------------
class MoEElementFusion(nn.Module):
    """Sparse Laplace‑gated MoE fusion that is **CUDA‑assert safe**.

    We avoid `torch.cdist` (which can trigger kernel asserts when fed NaNs or
    non‑contiguous `expand`ed tensors) and compute the squared Euclidean
    distance by broadcasting – rock‑solid on CPU/GPU.
    """
    def __init__(self,d_model,n_experts=16,top_k=4):
        super().__init__()
        self.d_model,self.top_k = d_model,top_k
        self.n_experts = n_experts
        self.expert_keys = nn.Parameter(torch.randn(n_experts,d_model))  # (n,d)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model,4*d_model),nn.GELU(),nn.Linear(4*d_model,d_model))
            for _ in range(n_experts)])
        self.routers = nn.ModuleList()  # one tiny router per view

    def add_router(self):
        self.routers.append(nn.Linear(self.d_model,self.n_experts))

    def _laplace_gate(self,h,router):
        """Return fused tensor for one view.
        h: (B,L,D) — element tokens of this view.
        router: small Linear producing per‑expert bias.
        """
        B,L,D = h.shape
        keys = self.expert_keys.to(h.device)                 # (n,d)
        # Broadcast subtraction → (B,L,n,d)
        diff2 = (h.unsqueeze(2) - keys)**2                   # auto‑broadcast
        dist2 = diff2.sum(-1)                                # (B,L,n)
        logits = -dist2 + router(h)                          # (B,L,n)
        topv,topi = logits.topk(self.top_k,-1)               # (B,L,k)
        w = torch.softmax(topv,-1)                           # (B,L,k)
        # Gather expert outputs
        exp_out = torch.stack([E(h) for E in self.experts],2)   # (B,L,n,d)
        sel = torch.gather(exp_out,2,topi.unsqueeze(-1).expand(-1,-1,-1,D))
        fused = (w.unsqueeze(-1)*sel).sum(2)                 # (B,L,D)
        return fused

    def forward(self,views):
        fused = 0
        for v,R in zip(views,self.routers):
            fused += self._laplace_gate(v,R)
        return fused

# ------------------------------
# Encoder with MoE or additive fusion
# ------------------------------
class Encoder(nn.Module):
    def __init__(self,d_model,N,heads,use_moe,n_experts,top_k,device):
        super().__init__()
        self.embedder = MultiDescriptorEmbedder(d_model,'madani/data/element_properties',device)
        self.use_moe = use_moe
        if use_moe:
            self.fuser = MoEElementFusion(d_model,n_experts,top_k)
            for _ in range(3):   # three views
                self.fuser.add_router()
        self.pe  = FractionalEncoder(d_model,5000,False)
        self.ple = FractionalEncoder(d_model,5000,True)
        enc_layer = nn.TransformerEncoderLayer(d_model,heads,batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer,N)
    def forward(self,Z,frac):
        views = self.embedder(Z)
        x = self.fuser(views) if self.use_moe else sum(views)
        x = x + torch.cat([self.pe(frac), self.ple(frac)],-1)
        x = self.encoder(x)
        return x

# ------------------------------
# CrabNet FuseMoE model
# ------------------------------
class CrabNet(nn.Module):
    def __init__(self,out_dims=3,d_model=512,N=3,heads=4,use_moe=True,n_experts=16,top_k=4,residual_nn='roost',device=None):
        super().__init__()
        self.device = get_default_device() if device is None else device
        self.encoder = Encoder(d_model,N,heads,use_moe,n_experts,top_k,self.device).to(self.device)
        hidden = [1024,512,256,128] if residual_nn=='roost' else [256,128]
        self.output_nn = ResidualNetwork(d_model,out_dims*2,hidden).to(self.device)
        self.avg = True
        self.out_dims = out_dims
        self.d_model = d_model
        self.N = N
        self.heads = heads
        # Alias for legacy trainer
        self.compute_device = self.device
    # expose property access
    @property
    def compute_device(self):
        return self.device
    @compute_device.setter
    def compute_device(self,v):
        self.device = v
    def forward(self,Z,frac):
        Z,frac = Z.to(self.device), frac.to(self.device)
        h = self.encoder(Z,frac)
        out = self.output_nn(h)                      # (B,L,2*out_dims)
        mask = (Z==0).unsqueeze(-1).expand_as(out)
        out = out.masked_fill(mask,0)
        out = out.sum(1)/(~mask).sum(1)              # mean over elements
        val,logit = out.chunk(2,-1)
        prob = torch.sigmoid(logit)
        return val*prob

# ------------------------------
if __name__=='__main__':
    model = CrabNet()
    print("Model device:",model.compute_device)
