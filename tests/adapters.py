from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from typing import Iterator, Iterable
import regex as re
import ast
import json
import math

import torch.nn as nn
from einops import rearrange, einsum
import torch

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        # Superclass constructor
        super().__init__()

        # initialize the weight
        self.learnable_weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype, device=device)) # To make it memory effiency and no tranpose needed in forward

        with torch.no_grad():
            # Formula
            std = 2/(in_features + out_features)
            nn.init.trunc_normal_(self.learnable_weight, mean=0, std=std, a=(-3*std), b = (3*std))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = einsum(x, self.learnable_weight,
                     "... d_in, d_out d_in -> ... d_out")
        return out


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to

    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """

    ln = Linear(d_in, d_out)

    if weights is not None:
        state = {"learnable_weight": weights} # Because here weights it not Dict, "Make it dict-like"
        ln.load_state_dict(state)

    return ln(in_features)

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


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer

    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """

    embeddings = Embedding(vocab_size, d_model)

    if weights is not None:
        state = {"learnable_weight": weights} # Because here weights it not Dict, "Make it dict-like"
        embeddings.load_state_dict(state)

    return embeddings(token_ids)
    

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        with torch.no_grad():
            self.w1_weight = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))
            self.w2_weight = nn.Parameter(torch.empty(d_model, d_ff, device=device, dtype=dtype))
            self.w3_weight = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))

    def forward(self, 
                in_features: Float[Tensor, " ... d_model"]):
        
        w1x = einsum(
            self.w1_weight, in_features,
            "d_ff d_model, ... d_model -> ... d_ff" 
        )

        w3x = einsum(
            self.w3_weight, in_features,
            "d_ff d_model, ... d_model -> ... d_ff"
        )

        silu = SiLU()
        w1x = silu(w1x)


        element_wise_mul = einsum(
            w1x, w3x,
            "... d_ff, ... d_ff -> ... d_ff"
        )

        out = einsum(
            self.w2_weight, element_wise_mul,
            "d_model d_ff, ... d_ff -> ... d_model"
        )

        return out

        




def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    # Example:
    # If your state dict keys match, you can use `load_state_dict()`
    # swiglu.load_state_dict(weights)
    # You can also manually assign the weights
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight
    

    swiglu = SwiGLU(d_model=d_model, d_ff=d_ff)

    with torch.no_grad():
        state = {
            "w1_weight": w1_weight,
            "w2_weight": w2_weight,
            "w3_weight": w3_weight
        }
        swiglu.load_state_dict(state)

    out = swiglu(in_features)
    
    return out


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    qtk = einsum(
        Q, K,
        "... queries d_k, ... keys d_k -> ... queries keys"
    )

    sqrtd = torch.sqrt(torch.tensor(Q.size(-1)))

    value = qtk / sqrtd
    value = value.masked_fill(~mask, float("-inf")) # Do masking first, Put -inf for those ~mask (True mask)

    max_v = value.max(dim=-1, keepdim=True).values

    softmax = torch.exp(value - max_v) / torch.exp(value - max_v).sum(dim=-1, keepdim=True)

    out = softmax * mask

    out = softmax @ V

    return out

    
class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, 
                num_heads: int, 
                theta: float | None = None,
                max_seq_len: int | None = None,
                device= None,
                dtype = None):
        super().__init__()

        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_proj_weight = nn.Parameter(
            torch.empty(num_heads * self.d_k, d_model, device=device, dtype=dtype)
        )
        self.k_proj_weight = nn.Parameter(
            torch.empty(num_heads * self.d_k, d_model, device=device, dtype=dtype)
        )
        self.v_proj_weight = nn.Parameter(
            torch.empty(num_heads * self.d_k, d_model, device=device, dtype=dtype)
        )
        self.o_proj_weight = nn.Parameter(
            torch.empty(d_model, num_heads * self.d_k, device=device, dtype=dtype)
        )

        if theta is not None and max_seq_len is not None:
            self.theta = theta
            self.max_seq_len = max_seq_len

    def forward(self, 
                in_features: Float[Tensor, " ... sequence_length d_in"], 
                token_positions: Int[Tensor, " ... sequence_length"] | None = None,
               ):
        # QKV projection
        # 1. Pass through projection layer first
        Q = in_features @ self.q_proj_weight.T
        K = in_features @ self.k_proj_weight.T
        V = in_features @ self.v_proj_weight.T

        # 2. Rearrange to heads, d_k arrangment
        Q = rearrange(Q, "... T (h d_k) -> ... h T d_k", h=self.num_heads)
        K = rearrange(K, "... T (h d_k) -> ... h T d_k", h=self.num_heads)
        V = rearrange(V, "... T (h d_k) -> ... h T d_k", h=self.num_heads)

        if token_positions is not None and self.theta is not None and self.max_seq_len is not None:
            RoPE = RotaryPositionalEmbedding(theta=self.theta, d_k = self.d_k, max_seq_len=self.max_seq_len)
            
            # After projection, apply RoPE
            Q = RoPE(Q, token_positions)
            K = RoPE(K, token_positions)


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


        out = out @ self.o_proj_weight.T
            
        return out



def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    multi_head = CausalMultiHeadSelfAttention(d_model=d_model, num_heads=num_heads)
    
    with torch.no_grad():
        state = {
            "q_proj_weight": q_proj_weight,
            "k_proj_weight": k_proj_weight,
            "v_proj_weight": v_proj_weight,
            "o_proj_weight": o_proj_weight
        }
        multi_head.load_state_dict(state)

    out = multi_head(in_features)

    return out

def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """

    multi_head = CausalMultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, theta=theta, max_seq_len=max_seq_len)

    with torch.no_grad():
        state = {
            "q_proj_weight": q_proj_weight,
            "k_proj_weight": k_proj_weight,
            "v_proj_weight": v_proj_weight,
            "o_proj_weight": o_proj_weight
        }
        multi_head.load_state_dict(state)

    out = multi_head(in_features, token_positions=token_positions)

    return out


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device= None):
        super().__init__()
        self.theta = torch.tensor(theta, device=device)
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2) / d_k))
        self.register_buffer("inv_freq", inv_freq)
      


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        token_positions = rearrange(token_positions, "... pos -> ... pos 1")

        theta = token_positions* self.inv_freq[None:, ] 
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


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    
    RoPE = RotaryPositionalEmbedding(theta=theta, d_k=d_k, max_seq_len=max_seq_len)
    
    out = RoPE(in_query_or_key, token_positions)

    return out

class transformer_block(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 num_heads: int, 
                 d_ff: int,
                 theta: float | None = None,
                 max_seq_len: int | None = None
                 ):
        super().__init__()

        self.theta = theta
        self.max_seq_len = max_seq_len

        self.attn = CausalMultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, theta=theta, max_seq_len=max_seq_len)
        self.ln1 = RMSnorm(d_model=d_model)
        self.ln2 = RMSnorm(d_model=d_model)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff)        


    def forward(self, 
                in_features: Float[Tensor, " batch sequence_length d_model"],
                token_positions: Float[Tensor, "... seq_length"] | None = None):

        if self.theta is not None and self.max_seq_len is not None:
            token_positions = torch.arange(0, in_features.size(1)) # Make token_position 

        # First Block
        pre_in_features_1 = in_features.clone()
        in_features = self.attn(self.ln1(in_features), token_positions=token_positions)
        in_features += pre_in_features_1


        # Second Block
        pre_in_features_2 = in_features.clone()
        in_features = self.ffn(self.ln2(in_features))
        in_features += pre_in_features_2

        return in_features
        # Position-wise feedforward



def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """

    tb = transformer_block(d_model=d_model, num_heads=num_heads, d_ff=d_ff, max_seq_len=max_seq_len, theta=theta)

    if weights is not None:
        key_map ={
            "attn.q_proj.weight": "attn.q_proj_weight",
            "attn.k_proj.weight": "attn.k_proj_weight",
            "attn.v_proj.weight": "attn.v_proj_weight",
            "attn.output_proj.weight": "attn.o_proj_weight",

            # layer norm / rms
            "ln1.weight": "ln1.learnable_weight",
            "ln2.weight": "ln2.learnable_weight",

            # ffn
            "ffn.w1.weight": "ffn.w1_weight",
            "ffn.w2.weight": "ffn.w2_weight",
            "ffn.w3.weight": "ffn.w3_weight",
        }

        new_weights = {key_map.get(k): v for k, v in weights.items()} # Mapping keys
        tb.load_state_dict(new_weights)

    return tb(in_features)


class transformer_lm(nn.Module):
    def __init__(self, 
                 vocab_size: int, 
                 d_model: int, 
                 context_length: int, 
                 num_layers: int, 
                 num_heads: int, 
                 d_ff: int,
                 rope_theta: float,):
        super().__init__()
        self.token_embedding = Embedding(num_embeddings=vocab_size, embeddings_dim=d_model) # Did one hot for me 

        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        
        for _ in range(num_layers):
            self.layers.append(transformer_block(d_model=d_model, num_heads=num_heads, d_ff=d_ff, theta=rope_theta, max_seq_len=context_length))

        self.ln_final = RMSnorm(d_model=d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.softmax = Softmax(dim=-1)

    def forward(self, in_indices: Int[Tensor, " batch_size sequence_length"]):



        x = self.token_embedding(in_indices) # [B, T] -> (one hot -> [B, T, vocab_size] @ [vocab_size, d_model]) = [B, T, d_model]

        for layer in self.layers:
            x = layer(x) # [B, T, d_model] -> Transformer layer = [B, T. d_model]

        x = self.ln_final(x) # [B, T, d_model] -> [B, T, d_model]

        x = self.lm_head(x) # [B, T, d_model] -> [B, T, vocab_size]

        # Noeed need for softmax here
        #x = self.softmax(x) # [B, T, vocab_size] -> [B, T, vocab_size] (prob) 

        return x


def build_key_map(n_layers: int):
    key_map = {}

    # embedding
    key_map["token_embeddings.weight"] = "token_embedding.learnable_weight"

    for i in range(n_layers):
        # attention
        key_map[f"layers.{i}.attn.q_proj.weight"] = f"layers.{i}.attn.q_proj_weight"
        key_map[f"layers.{i}.attn.k_proj.weight"] = f"layers.{i}.attn.k_proj_weight"
        key_map[f"layers.{i}.attn.v_proj.weight"] = f"layers.{i}.attn.v_proj_weight"
        key_map[f"layers.{i}.attn.output_proj.weight"] = f"layers.{i}.attn.o_proj_weight"

        # layer norm / rms
        key_map[f"layers.{i}.ln1.weight"] = f"layers.{i}.ln1.learnable_weight"
        key_map[f"layers.{i}.ln2.weight"] = f"layers.{i}.ln2.learnable_weight"

        # ffn
        key_map[f"layers.{i}.ffn.w1.weight"] = f"layers.{i}.ffn.w1_weight"
        key_map[f"layers.{i}.ffn.w2.weight"] = f"layers.{i}.ffn.w2_weight"
        key_map[f"layers.{i}.ffn.w3.weight"] = f"layers.{i}.ffn.w3_weight"

    key_map[f"ln_final.weight"] = "ln_final.learnable_weight"

    return key_map

def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\Theta$ parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """

    transformer = transformer_lm(vocab_size=vocab_size, d_model=d_model, context_length=context_length, num_layers=num_layers, num_heads=num_heads, d_ff=d_ff, rope_theta=rope_theta)

    if weights is not None:
        key_map = build_key_map(num_layers) # Here, because I use helper function for mapping the keys parameters
        new_weights = {key_map.get(k, k): v for k, v in weights.items()}

        transformer.load_state_dict(new_weights)


    return transformer(in_indices)
    


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


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    rms = RMSnorm(d_model, eps)

    if weights is not None:
        state = {"learnable_weight": weights}
        rms.load_state_dict(state)
    
    return rms(in_features)
        

class SiLU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, in_features):
        return torch.sigmoid(in_features) * in_features

def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """

    silu = SiLU()
    return silu(in_features)



def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """

    device = "cpu"
    dataset = torch.tensor(dataset).to(device)
    n = dataset.size(0)

    start_point = torch.randint(low = 0, high = n - context_length, size=(batch_size,)).to(device) # Sampling B starting points from [1, n-m)
    
    offset = torch.arange(context_length).to(device) # Make context_length offset

    idx = (start_point[:, None] + offset[None, :]).to(device) # Make [b, context_len] first pair
    
    x = dataset[idx]
    y = dataset[idx + 1]

    

    return (x, y)


class Softmax(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        max_val, _ = x.max(dim=self.dim, keepdim=True)
        exp_x = torch.exp(x - max_val)
        out = exp_x / exp_x.sum(dim=self.dim, keepdim=True)
        return out



def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    softmax = Softmax(dim)
    out = softmax(in_features)
    
    return out


class CrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):

        dtype = inputs.dtype

        targets_oh = nn.functional.one_hot(
            targets, num_classes=inputs.size(-1)
        ).to(dtype)

        # numerical stability
        inputs = inputs - inputs.max(dim=-1, keepdim=True).values

        # logsumexp (denominator)
        softmax_bottom = torch.log(torch.exp(inputs).sum(dim=-1))

        # target logit (numerator, already in log space)
        target_logit = (inputs * targets_oh).sum(dim=-1)

        # cross entropy
        out = softmax_bottom - target_logit

        return out.mean()



def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """

    ce = CrossEntropy()
    out = ce(inputs, targets)

    return out




def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """

    total_norm_sq = 0.0

    # calculate l2 norm for all parameters
    for p in parameters:
        if p.grad is not None:
            total_norm_sq += p.grad.norm(2).item() ** 2

    total_norm = total_norm_sq ** 0.5


    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + 1e-6)
        for p in parameters:
            if p.grad is not None:
                p.grad.mul_(scale) # in-place multiply, otherwise use copy_

    return parameters





def get_adamw_cls() -> Any:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    class AdamW(torch.optim.Optimizer):
        def __init__(self, params, betas, eps, weight_decay, lr=1e-3):
            self.weight_decay = weight_decay
            self.betas = betas
            self.eps = eps

            defaults = {
                "lr": lr
            }

            super().__init__(params, defaults)
            

        def step(self):
            beta1, beta2 = self.betas

            for group in self.param_groups: # Group = all hyperparameters used in this object

                for p in group["params"]: # Here, p = parameter = theta
                    if p.grad is None: # no backward() / frozen / not chosen MoE / zero grad
                        continue

                    if p not in self.state: # Because every parameter in AdamW has its own m/v/t, 
                        # It is the first time using this parameter, therefore do initialize
                        self.state[p] = {
                            "m": torch.zeros_like(p), 
                            "v": torch.zeros_like(p),
                            "t": 1,
                        } # Replace nn.Parameter

                    # Get the state
                    state = self.state[p] 
                    t = state.get("t", 0) # get value t at state
                    grad = p.grad.data # get grad at state

                    m = beta1 * state["m"] + (1 - beta1) * grad
                    v = beta2 * state["v"]  + (1 - beta2) * (grad ** 2)

                    bias_correction2 = 1 - beta2 ** t
                    bias_correction1 = 1 - beta1 ** t
                    alpha_t = group['lr'] * bias_correction2**(1/2) / bias_correction1

                    p.data = p.data - alpha_t * m / (torch.sqrt(v) + self.eps)

                    p.data = p.data - group["lr"] * self.weight_decay * p.data

                    # Needs to store back
                    state["t"] = t + 1
                    state["m"] = m
                    state["v"] = v


    return AdamW




def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    

    if it < warmup_iters:
        a_t = it / warmup_iters * max_learning_rate
    elif warmup_iters <= it and it <= cosine_cycle_iters:
        a_t = min_learning_rate + 1/2 * (1 + math.cos(((it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * torch.pi))) * (max_learning_rate - min_learning_rate)
    else:
        a_t = min_learning_rate

    return a_t

def save_checkpoint(model:torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike | BinaryIO | IO[bytes]):

    for i in range(1, iteration+1):
        model_params = model.state_dict()
        optim_params = optimizer.state_dict()

        params = {}
        for k, v in model_params.items():
            if k not in params:
                params[k] = v
            
        for k, v in optim_params.items():
            if k not in params:
                params[k] = v

        params['iteration'] = i

        torch.save(params, out)



def load_checkpoint(src: str | os.PathLike | BinaryIO | IO[bytes], model, optimizer):
    params = torch.load(src)
    model_params = {}
    optim_params = {}
    iteration = 0

    for k, v in params.items():
        if k == 'state' or k == 'param_groups':
            optim_params[k] = v
        elif k == 'iteration':
            iteration = v
        else:
            model_params[k] = v

    model.load_state_dict(model_params)
    optimizer.load_state_dict(optim_params)

    return iteration


        


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    save_checkpoint(model=model, optimizer=optimizer, iteration=iteration, out=out)


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    return load_checkpoint(src=src, model=model, optimizer=optimizer)

class Tokenizer:
    def __init__(self, 
                 vocab: dict[int, bytes], 
                 merges: list[tuple[bytes, bytes]], 
                 special_tokens=None):
        
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens else []

        # Make a reverse look up vocab (bytes -> id) 
        self.rev_vocab = {v: k for k, v in self.vocab.items()}

    @classmethod
    def from_files(cls, 
                   vocab_filepath:str,
                    merges_filepath:str, 
                    special_tokens=None):
        
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab = json.load(f)

        new_vocab: dict[tuple[bytes], int] = {}
        for k, v in vocab.items():
            v = v.encode('utf-8')
            new_vocab[k] = v 
        vocab = new_vocab


        with open(merges_filepath, "r", encoding="utf-8") as f:
            merges_txt = f.read().splitlines()
        
        merges = []
        for line in merges_txt:
            out = ast.literal_eval(line)
            merges.append(out)

        tokenizer = Tokenizer(vocab, merges, special_tokens)

        return tokenizer
    

    def encode(self, text: str) -> list[int]:
        # Split the text by each special token
        special_token_pat = re.compile(
            "(" + "|".join(
                re.escape(t)
                for t in sorted(self.special_tokens, key=len, reverse=True)
            ) + ")"
        )

        # First, need to check if the self.special_tokens is empty.
        # If no special tokens, do not use special_token_pat split, otherwise, it splits on character
        if not self.special_tokens:
            lines = [text]
        else:
            matches = re.split(special_token_pat, text)
            lines = matches # just for convenience 

        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        encoding_lst = []
        # Second, loop through each passage
        for line in lines:
            # If there is special tokens
            if line in self.special_tokens:
                line = line.encode('utf-8')
                encoding_lst.append(self.rev_vocab[line])
                continue

            # Make the passage as a list of <pre-token>
            matches = re.finditer(PAT, line)

            # For each pre-tok in the list, encode this 
            for match in matches:
                s = match.group()

                # For each elem, make it tuple-like bytes! 
                tok = s.encode('utf-8')
                tok = tuple(bytes([b]) for b in tok)

                for merge in self.merges:
                    new_tup = []

                    i = 0
                    while i < len(tok):
                        # In pre_merge, we look at the next byte, it might be merged or not, so we use *** +1 *** rather than actual len of merge bytes
                        if i + 1 < len(tok) and tok[i] == merge[0] and tok[i+1] == merge[1]: 
                            new_tup.append(tok[i] + tok[i+1]) 
                            i += 2 # Because after each merging, the elem merged together, we only need 2
                        else:
                            new_tup.append(tok[i])      
                            i += 1    

                    new_tup = tuple(new_tup)
                    tok = new_tup


                # After these, all the elem in tok will can be looked up through dict
                for t in tok:
                    encoding_lst.append(self.rev_vocab[t])
                
        return encoding_lst

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        special_token_pat = "|".join(re.escape(tok) for tok in self.special_tokens)
        
        encoding_lst = []

        for it in iterable:
            # First split the text by each special token
            special_token_pat = re.compile(
                "(" + "|".join(
                    re.escape(t)
                    for t in sorted(self.special_tokens, key=len, reverse=True)
                ) + ")"
            )

            matches = re.split(special_token_pat, it)
            lines = matches # just for convenience 

            for line in lines:
                # If there is special tokens, just skip
                if line in self.special_tokens:
                    line = line.encode('utf-8')
                    encoding_lst.append(self.rev_vocab[line])
                    continue

                matches = re.finditer(PAT, line)

                for match in matches:
                    s = match.group()

                    tok = s.encode('utf-8')
                    tok = tuple(bytes([b]) for b in tok)

                    for merge in self.merges:
                        new_tup = []

                        i = 0
                        while i < len(tok):
                            # In pre_merge, we look at the next byte, it might be merged or not, so we use *** +1 *** rather than actual len of merge bytes
                            if i + 1 < len(tok) and tok[i] == merge[0] and tok[i+1] == merge[1]: 
                                new_tup.append(tok[i] + tok[i+1]) 
                                i += 2 # Because after each merging, the elem merged together, we only need 2
                            else:
                                new_tup.append(tok[i])      
                                i += 1    

                        new_tup = tuple(new_tup)
                        tok = new_tup

                    # After these, all the elem in tok will can be looked up through dict
                    for t in tok:
                        encoding_lst.append(self.rev_vocab[t])

        return iter(encoding_lst)

    
    def decode(self, ids: list[int]) -> str:

        pre_dec: bytes = b""
        for id in ids:
            if id not in self.vocab:
                pre_dec += b"\xef\xbf\xbd"
            else:
                pre_dec+=self.vocab[id]

        output = pre_dec.decode("utf-8", errors="replace")

        return output
        
def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """

    return Tokenizer(vocab, merges, special_tokens)
   


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    import regex as re
    from multiprocessing import Pool

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    pre_tok: dict[str, int] = {} # This is a generics for dict

    # Pre-tokenization
    # Count the frequency of str in text

    with open(input_path, 'r', encoding="utf-8") as f:
        content = f.read()

        special_token_pat = "|".join(re.escape(tok) for tok in special_tokens)
        lines = re.split(special_token_pat, content)

    # Run multiprocessing
    # with Pool(processes=10) as pool:
    #     pre_tok_list = pool.map(pre_tokenization, lines)
    # pre_tok_list = pre_tokenization(lines)
    pre_tok_list = list(map(pre_tokenization, lines))

    pre_tok: dict[str, int] = {}

    for elem in pre_tok_list:
        for k, v in elem.items():
            pre_tok[k] = pre_tok.get(k, 0) + v

    tok: dict[tuple[bytes], int] = {}
    for k, v in pre_tok.items():
        k = k.encode('utf-8')
        t_b = tuple(bytes([b]) for b in k)
        tok[t_b] = v


    # Merge processing
    vocab: dict[int, bytes] = {}
    len_spec_tokens = len(special_tokens)

    # !!! Add special token bytes into vocab (special token should be removed from text)
    for i in range(len_spec_tokens):
        vocab[i] = special_tokens[i].encode('utf-8')

    # Add 0-255 bytes char
    for i in range(0, 256):
        vocab[i+len_spec_tokens] = bytes([i])

    # Check how many more vocab needed to be merged 
    lst = []
    while len(lst) < (vocab_size - 256 - len_spec_tokens): # Already have 256 bytes and special_tokens
        post_tok, mst_freq_bytes = merge(tok)
        tok = post_tok
        lst.append(mst_freq_bytes)

    # Use index, start from existing vocab and continue adding more...
    index = len(vocab)
    for i, merged_t in enumerate(lst):
        vocab[index + i] = merged_t[0] + merged_t[1]

    return vocab, lst

def merge(pre_merge: dict[tuple[bytes], int]
          )-> tuple[dict[tuple[bytes], int], tuple[bytes, bytes]]:
    """
        pre_merge: dict[(bytes, ...): freq]
        return:
            post_merge: dict[(merged bytes, ...): freq]
            mst_freq_btyes: tuple(bytes1, bytes2)    
    """

    count: dict[tuple[bytes], int] = {}
    post_merge: dict[tuple[bytes], int] = {}

    # Loop through all the bytes and find the most frequent pair in this iteration.
    for t, freq in pre_merge.items():
        for i in range(len(t)-1):
            pair = (t[i], t[i+1])
            count[pair] = count.get(pair, 0) + freq

    # Find the most frequent pair
    mst_freq_pair = max(
        count.items(),
        key=lambda item: (item[1], item[0])   # frequency first, then lexicographic
    )
    mst_freq_bytes = mst_freq_pair[0]
    tokA, tokB = mst_freq_bytes
    
    # To merge the mst_freq_bytes in key
    for k, v in pre_merge.items():
        new_tup = []
        
        i = 0
        while i < len(k):
            # In pre_merge, we look at the next byte, it might be merged or not, so we use *** +1 *** rather than actual len of merge bytes
            if i + 1 < len(k) and k[i] == tokA and k[i+1] == tokB: 
                new_tup.append(mst_freq_bytes[0] + mst_freq_bytes[1]) 
                i += 2 # Because after each merging, the elem merged together, we only need 2
            else:
                new_tup.append(k[i])      
                i += 1    

        new_tup = tuple(new_tup)
        post_merge[new_tup] = post_merge.get(new_tup, 0) + v # It might have existing key there

    return post_merge, mst_freq_bytes

def pre_tokenization(line: str) -> dict[str, int]:
    import regex as re
    
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    pre_tok: dict[str, int] = {} # This is a generics for dict
 
    if not line:
        return 
    
    matches = re.finditer(PAT, line)

    for match in matches:
        s = match.group()
        if s not in pre_tok:
            pre_tok[s] = 1
        else:
            pre_tok[s] += 1
        
    return pre_tok

