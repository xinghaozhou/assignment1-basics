import torch
import torch.nn as nn
from einops import rearrange, einsum

class RMSnorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        with torch.no_grad():
            self.learnable_weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype)) # gi


    def forward(self, x: torch.Tensor):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(1/self.d_model * torch.sum(x**2, dim=-1, keepdim=True) + self.eps)
        x = x / rms

        x = einsum(
            x, self.learnable_weight,
            "batch seq_len d_model, d_model -> batch seq_len d_model"
        )

        return x.to(in_dtype)