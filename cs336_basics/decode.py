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
import numpy as np

import torch.nn as nn
from einops import rearrange, einsum
import torch

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        # Superclass constructor
        super().__init__()

        # initialize the weight
        self.learnable_weight = nn.Parameter(torch.empty(out_features, in_features), device=device, dtype=dtype) # To make it memory effiency and no tranpose needed in forward

        with torch.no_grad():
            # Formula
            std = 2/(in_features + out_features)
            nn.init.trunc_normal_(self.learnable_weight, mean=0, std=std, a=(-3*std), b = (3*std))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = einsum(x, self.learnable_weight,
                     "... d_in, d_out d_in -> ... d_out")
        return out
    
class Embedding(nn.Module):
    def __init__(self, num_embeddings, embeddings_dim, device=None, dtype=None):
        super().__init__()

        # Important: different from Linear Layer, we don't need linear transformation here
        self.num_embeddings = num_embeddings
        self.embeddings_dim = embeddings_dim
        self.learnable_weight = nn.Parameter(torch.empty(num_embeddings, embeddings_dim), device=device, dtype=dtype)
        

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
    

class RMSnorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        with torch.no_grad():
            self.learnable_weight = nn.Parameter(torch.ones(d_model), device=device, dtype=dtype) # gi


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


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        with torch.no_grad():
            self.w1_weight = nn.Parameter(torch.empty(d_ff, d_model), device=device, dtype=dtype)
            self.w2_weight = nn.Parameter(torch.empty(d_model, d_ff), device=device, dtype=dtype)
            self.w3_weight = nn.Parameter(torch.empty(d_ff, d_model), device=device, dtype=dtype)

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
            torch.empty(num_heads * self.d_k, d_model), device=device, dtype=dtype
        )
        self.k_proj_weight = nn.Parameter(
            torch.empty(num_heads * self.d_k, d_model), device=device, dtype=dtype
        )
        self.v_proj_weight = nn.Parameter(
            torch.empty(num_heads * self.d_k, d_model), device=device, dtype=dtype
        )
        self.o_proj_weight = nn.Parameter(
            torch.empty(d_model, num_heads * self.d_k), device=device, dtype=dtype
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
    


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device= None):
        super().__init__()
        self.theta = torch.tensor(theta)
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2) / d_k))
        self.register_buffer("inv_freq", inv_freq, device=device)
      


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
    
def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    data = torch.load(dataset)

    n = data.size(0)

    start_point = torch.randint(low=0, high= n - context_length, size=(batch_size,), device=device)
    offset = torch.arange(0, context_length)

    first_pair = (start_point[:, None] + offset[None, :]).to(device) # Make [b, context_len] first pair
    second_pair = (start_point[:, None] + offset[None, :] + 1).to(device) # Make [b,  context_len] second pair

    return (first_pair, second_pair)



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



# Training Script
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--path", help="Input Prompt")
parser.add_argument("--prompt", help="Input Prompt")
parser.add_argument("--max_tokens", help="max number of generated tokens")
parser.add_argument("--temperature", help="temperature")
parser.add_argument("--top_p", help="Top-p sampling")

args = parser.parse_args()



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

class Softmax(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        max_val, _ = x.max(dim=self.dim, keepdim=True)
        exp_x = torch.exp(x - max_val)
        out = exp_x / exp_x.sum(dim=self.dim, keepdim=True)
        return out
    


def decode(
    model,
    prompt: torch.Tensor,      # [1, T] or [B, T]
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token_id: int = None,
):
    idx = prompt
    token_len = prompt.size(-1)
    softmax = Softmax(dim=-1)
    
    # Stop until <|endoftext|> or max token
    while token_len < max_new_tokens: 
        # input token to model
        logits = model(idx) # [B, sequence_length, vocab_size]

        # !!! We only use the last token logits (the one generated from model)
        logits = logits[..., -1, ...]
        logits /= temperature
        prob = softmax(logits)

        if top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(prob, dim=-1, descending=True) # descend
            cum_probs = torch.cumsum(sorted_probs, dim=-1) # cumulative sum

            mask = cum_probs > top_p  
            mask[..., 1:] = mask[..., :-1] # Keep the first that cum prob >= top_p and anything else get removed
            mask[..., 0] = False # The first get kept

            top_p_probs = sorted_probs * (~mask)
            top_p_probs /= top_p_probs.sum(dim=-1, keepdim=True)

            # !!! Sample the new token using multinomial from top_p token
            sample_indices = torch.multinomial(top_p_probs, 1)
            next_token = sorted_idx.gather(-1, sample_indices) # Get the next token ids from sorted_idx[sample_indcies]

        else:
            next_token = torch.multinomial(prob, 1)
        

        # If next_token is eos, stop
        if eos_token_id is not None:
            if (next_token == eos_token_id).all():
                break

        # update
        token_len += 1
        idx = torch.concat([idx, next_token], dim=-1)

    return idx


   



        
        


