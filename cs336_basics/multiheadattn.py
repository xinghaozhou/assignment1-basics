import torch
import torch.nn as nn
from einops import rearrange, einsum
from jaxtyping import Bool, Float, Int
from torch import Tensor

from cs336_basics.rope import RotaryPositionalEmbedding
from cs336_basics.softmax import Softmax
from cs336_basics.linear import Linear 
from cs336_basics.scaled_dot_product_attention import scaled_dot_product_attention

class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, 
                num_heads: int, 
                use_rope: bool = False,
                theta: float | None = None,
                max_seq_len: int | None = None,
                device: str | None = None,
                dtype: torch.dtype | None = None):
        super().__init__()

        kwargs = {'device': device, 'dtype': dtype}

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_rope = use_rope

        self.q_proj_weight = Linear(
            self.d_k * num_heads, d_model, **kwargs
        )
        
        self.k_proj_weight = Linear(
            self.d_k * num_heads, d_model, **kwargs
        )
        self.v_proj_weight = Linear(
            self.d_k * num_heads, d_model, **kwargs
        )
        self.o_proj_weight = Linear(
            num_heads * self.d_k, d_model, **kwargs
        )

        if use_rope:
            self.theta = theta
            self.max_seq_len = max_seq_len
            self.RoPE = RotaryPositionalEmbedding(theta=self.theta, d_k = self.d_k, max_seq_len=self.max_seq_len, **kwargs)

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
        V = self.v_proj_weight(in_features)

        # 2. Rearrange to heads, d_k arrangment
        Q = rearrange(Q, "... T (h d_k) -> ... h T d_k", h=self.num_heads)
        K = rearrange(K, "... T (h d_k) -> ... h T d_k", h=self.num_heads)
        V = rearrange(V, "... T (h d_k) -> ... h T d_k", h=self.num_heads)

        # After projection, apply RoPE
        if self.use_rope:
            Q = self.RoPE(Q, token_positions)
            K = self.RoPE(K, token_positions)

        # attention
        scores = einsum(Q, K, "... h T_q d_k, ... h T_k d_k -> ... h T_q T_k")
        scores = scores / (self.d_k ** 0.5)

        mask = torch.triu(
        torch.ones((scores.size(-2), scores.size(-1)), device=scores.device),
            diagonal=1
        ) == 0   


        heads_attns = scaled_dot_product_attention(Q,K, V, mask)
        
        # concat heads
        out = rearrange(heads_attns, "... h T d_k -> ... T (h d_k)")

        #out = out @ self.o_proj_weight.T
        out = self.o_proj_weight(out)
            
        return out
    