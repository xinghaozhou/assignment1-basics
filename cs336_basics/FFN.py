import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Int
from torch import Tensor

from einops import rearrange, einsum

from cs336_basics.linear import Linear

class SiLU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, in_features):
        return torch.sigmoid(in_features) * in_features
    

class Softmax(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        max_val, _ = x.max(dim=self.dim, keepdim=True)
        exp_x = torch.exp(x - max_val)
        out = exp_x / exp_x.sum(dim=self.dim, keepdim=True)
        return out
    

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.w1_weight = nn.Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2_weight = nn.Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3_weight = nn.Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, 
                in_features: Float[Tensor, " ... d_model"]):

        w1x = self.w1_weight(in_features)
        w3x = self.w3_weight(in_features)

        silu = SiLU()
        w1x = silu(w1x)

        element_wise_mul = w1x * w3x

        out = self.w2_weight(element_wise_mul)

        return out