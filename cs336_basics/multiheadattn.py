import torch
import torch.nn as nn
from einops import rearrange, einsum
from jaxtyping import Bool, Float, Int
from torch import Tensor

from cs336_basics.rope import RotaryPositionalEmbedding
from cs336_basics.softmax import Softmax
from cs336_basics.linear import Linear 

class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, 
                num_heads: int, 
                theta: float | None = None,
                max_seq_len: int | None = None,
                device= None,
                dtype = None):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # self.q_proj_weight = nn.Parameter(
        #     torch.empty(num_heads * self.d_k, d_model, device=device, dtype=dtype)
        # )
        # self.k_proj_weight = nn.Parameter(
        #     torch.empty(num_heads * self.d_k, d_model, device=device, dtype=dtype)
        # )
        # self.v_proj_weight = nn.Parameter(
        #     torch.empty(num_heads * self.d_k, d_model, device=device, dtype=dtype)
        # )
        # self.o_proj_weight = nn.Parameter(
        #     torch.empty(d_model, num_heads * self.d_k, device=device, dtype=dtype)
        # )

        self.q_proj_weight = nn.Linear(
            num_heads * self.d_k, d_model, device=device, dtype=dtype
        )
        self.k_proj_weight = nn.Linear(
            num_heads * self.d_k, d_model, device=device, dtype=dtype
        )
        self.v_proj_weight = nn.Linear(
            num_heads * self.d_k, d_model, device=device, dtype=dtype
        )
        self.o_proj_weight = nn.Linear(
            d_model, num_heads * self.d_k, device=device, dtype=dtype
        )
        

        if theta is not None and max_seq_len is not None:
            self.theta = theta
            self.max_seq_len = max_seq_len
            self.RoPE = RotaryPositionalEmbedding(theta=self.theta, d_k = self.d_k, max_seq_len=self.max_seq_len, device=device)

    def forward(self, 
                in_features: Float[Tensor, " ... sequence_length d_in"], 
                token_positions: Int[Tensor, " ... sequence_length"] | None = None,
               ):
        # QKV projection
        # 1. Pass through projection layer first
        # Q = in_features @ self.q_proj_weight.T
        # K = in_features @ self.k_proj_weight.T
        # V = in_features @ self.v_proj_weight.T

        Q = self.q_proj_weight(in_features)
        K = self.k_proj_weight(in_features)
        V = self.q_proj_weight(in_features)

        # 2. Rearrange to heads, d_k arrangment
        Q = rearrange(Q, "... T (h d_k) -> ... h T d_k", h=self.num_heads)
        K = rearrange(K, "... T (h d_k) -> ... h T d_k", h=self.num_heads)
        V = rearrange(V, "... T (h d_k) -> ... h T d_k", h=self.num_heads)

        # After projection, apply RoPE
        Q = self.RoPE(Q, token_positions)
        K = self.RoPE(K, token_positions)

        # attention
        scores = einsum(Q, K, "... h T_q d_k, ... h T_k d_k -> ... h T_q T_k")
        scores = scores / (self.d_k ** 0.5)

        mask = torch.triu(
        torch.full((scores.size(-2), scores.size(-1)), float("-inf"), device=scores.device),
            diagonal=1
        )   
        scores = scores + mask

        softmax = Softmax(dim=-1)
        attn = softmax(scores)

        heads = einsum(attn, V, "... h T_q T_k, ... h T_k d_k -> ... h T_q d_k")

        # concat heads
        out = rearrange(heads, "... h T d_k -> ... T (h d_k)")


        #out = out @ self.o_proj_weight.T
        out = self.o_proj_weight(out)
            
        return out
    