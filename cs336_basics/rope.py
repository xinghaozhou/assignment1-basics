import torch
import torch.nn as nn
from einops import rearrange, einsum

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, 
                 theta: float, 
                 d_k: int, 
                 max_seq_len: int, 
                 device: str | None = None,
                 dtype: torch.dtype | None = None):
        super().__init__()

        self.theta = torch.tensor(theta)
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device, dtype=dtype) / d_k))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
      


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # Sanity Check
        assert x.device == self.inv_freq.device

        # In case that token_position macthes the T
        T = x.size(-2)
        token_positions = token_positions[..., :T]

        device = x.device
        token_positions = token_positions[..., None].to(device)

        theta = (token_positions* self.inv_freq[None:, ])
        cos = torch.cos(theta)
        sin = torch.sin(theta)

        x1, x2 = x[..., ::2], x[..., 1::2] # here, it gets every calcualted (x) R@x(math) = x @ R.T = x @ R

        # Becuase x1/x2 * cos/sin is element-wise mul, O(d), which is equivalent to diagonal matrix mul.
        even_rot = x1 * cos - x2 * sin   # x0, x2, x4..
        odd_rot  = x1 * sin + x2 * cos   # x1, x3, x5..


        # back to [x0, x1, x2, ...]
        out = torch.stack([even_rot, odd_rot], dim=-1)  
        out = out.flatten(-2)    

        return out