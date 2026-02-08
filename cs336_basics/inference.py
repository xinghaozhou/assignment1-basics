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

from cs336_basics.softmax import Softmax

@torch.no_grad()
def generate(
    model,
    prompt: torch.Tensor,      # [1, T] or [B, T]
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token_id: int = 0,
):
    device = next(model.parameters()).device

    if prompt.dim() == 1: # if B dim does not exist
        prompt = prompt[None, ...]

    idx = prompt.to(device)

    token_len = 0
    softmax = Softmax(dim=-1)
    
    # Stop until <|endoftext|> or max token
    while token_len < max_new_tokens: 
        # input token to model
        logits = model(idx) # [B, sequence_length, vocab_size]

        # !!! We only use the last token logits (the one generated from model)
        logits = logits[:, -1:, :]
        logits /= temperature

        prob = softmax(logits)

        if top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(prob, dim=-1, descending=True) # descend
            cum_probs = torch.cumsum(sorted_probs, dim=-1) # cumulative sum

            mask = cum_probs > top_p  
            mask_copy = mask.clone()
            mask[..., 1:] = mask_copy[..., :-1] # Keep the first that cum prob >= top_p and anything else get removed
            mask[..., 0] = False # The first get kept

            top_p_probs = sorted_probs * (~mask)
            top_p_probs /= top_p_probs.sum(dim=-1, keepdim=True)

            # top_p_probs = [B, 1, V] -> [B, V]
            # sample_indices = [B, 1, V] -> [B, V]
            top_p_probs = top_p_probs.squeeze(1)
            sorted_idx = sorted_idx.squeeze(1)


            # !!! Sample the new token using multinomial from top_p token
            sample_indices = torch.multinomial(top_p_probs, 1)
            next_token = sorted_idx.gather(-1, sample_indices) # Get the next token ids from sorted_idx[sample_indcies]

        else:
            # prob = [B, 1, V]
            prob = prob.squeeze(1) # = prob = [B, V]
            next_token = torch.multinomial(prob, 1)
        
        # If next_token is eos, stop

        # Batch = 1
        if logits.size(0) == 1:
            if eos_token_id is not None:
                if (next_token.item() == eos_token_id):
                    break

        # Batch > 1
        else:
            if eos_token_id is not None:
                if (next_token.item() == eos_token_id).all():
                    break

        # update
        token_len += 1

        idx = torch.concat([idx, next_token], dim=-1)

    return idx


   
