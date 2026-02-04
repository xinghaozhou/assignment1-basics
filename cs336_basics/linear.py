import torch
import torch.nn as nn
from einops import rearrange, einsum


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        # Superclass constructor
        super().__init__()

        # initialize the weight
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype)) # To make it memory effiency and no tranpose needed in forward

        # Formula
        std = 2/(in_features + out_features)
        nn.init.trunc_normal_(self.weight, mean=0, std=std, a=(-3*std), b = (3*std))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x @ self.weight.T
        return x
    