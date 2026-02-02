import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange, einsum

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embeddings_dim, device=None, dtype=None):
        super().__init__()

        # Important: different from Linear Layer, we don't need linear transformation here
        self.num_embeddings = num_embeddings
        self.embeddings_dim = embeddings_dim
        self.learnable_weight = nn.Parameter(torch.empty(num_embeddings, embeddings_dim, device=device, dtype=dtype))
        

        with torch.no_grad():
            nn.init.trunc_normal_(self.learnable_weight, mean=0, std=1, a=-3, b=3)
        


    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids (b, s_length)
        # embedding matrix (vocab_size, d_model)

        token_ids = nn.functional.one_hot(token_ids, self.num_embeddings).to(self.learnable_weight.dtype) # token_ids = (B, S, vocab_size(0, 1))

        out = einsum(
            token_ids, self.learnable_weight,
            "batch seq_len vocab_size, vocab_size embedding_dim -> batch seq_len embedding_dim"
        )

        return out