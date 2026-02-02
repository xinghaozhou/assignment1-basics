import torch
import torch.nn as nn

class Softmax(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        max_val, _ = x.max(dim=self.dim, keepdim=True)
        exp_x = torch.exp(x - max_val)
        out = exp_x / exp_x.sum(dim=self.dim, keepdim=True)
        return out
    