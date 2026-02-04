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
import time

from cs336_basics.transformer import transformer_lm
from cs336_basics.adamW import AdamW
from cs336_basics.scheduler import lr_schedule
from cs336_basics.gradient_clip import gradient_clipping
from cs336_basics.loss import CrossEntropy
from cs336_basics.checkpoint import save_checkpoint, load_checkpoint
from cs336_basics.data_loading import get_batch
from cs336_basics.evaluation import evaluate


# Training Script
import argparse

parser = argparse.ArgumentParser()

# Device
parser.add_argument("--device", help="Which device", default='mps')

# Data
parser.add_argument("--train", help="Input training data", type=str)
parser.add_argument("--val", help="Input training data", type=str)
parser.add_argument("--val_step", help="num of steps to val", type=int)
parser.add_argument("--batch_size", help="Batch size", type=int)
parser.add_argument("--val_batch_size", help="Num of batches in val", type=int)
parser.add_argument("--context_length", help="Context length", type=int)
parser.add_argument("--lr", help="Maximum Learning Rate", type=float)

# Transformers parts
parser.add_argument("--vocab_size", help="Vocab size", type=int)
parser.add_argument("--d_model", help="dimension of embeddings", type=int)
parser.add_argument("--num_layers", help="num of transformer blocks", type=int)
parser.add_argument("--num_heads", help="num of heads in multi-head attention", type=int)
parser.add_argument("--d_ff", help="dimension of feed-forward network", type=int)
parser.add_argument("--rope_theta", help="theta in RoPE", type=float)

# Optimizer parts
parser.add_argument("--betas", help="betas of AdamW", type=float, default=(0.9, 0.95), nargs=2)
parser.add_argument("--eps", help="eps of AdamW", type=float)
parser.add_argument("--weight_decay", help="weight decay of AdamW", type=float)
parser.add_argument("--iteration", help="nums of iterations", type=int)

# lr rate schedule
parser.add_argument("--max_lr", help="max learning rate", type=float)
parser.add_argument("--min_lr", help="min learning rate", type=float)
parser.add_argument("--warmup_iters", help="warm up iteration", type=int)
parser.add_argument("--cosine_cycle_iters", help="cosine cycle iteration", type=int)
parser.add_argument("--max_l2_norm", help="maximum l2 norm in gradient clipping", type=float)

# Checkpoint
parser.add_argument("--resume_checkpoint", help="resume checkpoint", type=str)
parser.add_argument("--output", help="Output dir", type=str)
parser.add_argument("--save_step", help="step to save checkpoint", type=int)
parser.add_argument("--logging_step", help="step to log", type=int)

args = parser.parse_args()


def train():    
    import time as time
    from tqdm import tqdm

    device = args.device
    model = transformer_lm(args.vocab_size, args.d_model, args.context_length, args.num_layers, args.num_heads, args.d_ff, args.rope_theta).to(device)
    optim = AdamW(model.parameters(), betas=tuple(args.betas), eps=args.eps, weight_decay=args.weight_decay) # Put the model.params in Adam
    lr_sch = lr_schedule(max_learning_rate=args.max_lr, min_learning_rate=args.min_lr, warmup_iters=args.warmup_iters, cosine_cycle_iters=args.cosine_cycle_iters)
    grad_clip = gradient_clipping(args.max_l2_norm)

    training_set = np.memmap(args.train, mode='r', dtype=np.int32)
    val_set = np.memmap(args.val, mode='r', dtype=np.int32)

    ce = CrossEntropy()
    
    start = time.time()
    model.train() # Declare it is training now
    # Training loop 
    iteration = args.iteration
    for t in tqdm(range(iteration)):
        optim.zero_grad()

        x, gt = get_batch(training_set, args.batch_size, args.context_length, args.device)
        x = x.to(device)
        gt = gt.to(device)

        lr = lr_sch(t)

        for group in optim.param_groups:
            group["lr"] = lr

        y = model(x)
    
        loss = ce(y, gt)
        loss.backward()

        grad_clip(model.parameters())
        optim.step()


        if t % args.save_step == 0:
            output_dir = f"{args.output}_{t}"
            #save_checkpoint(model, optim, t, output_dir)

        if t % args.logging_step ==0:
            log_end_log = time.time()
            print(f"Iteration {t}| Time: {log_end_log - start:.2f}| Learning Rate {lr}| Train Loss: {(loss.item()):.4f}")

        if args.val_step and args.val is not None:
            if t % args.val_step == 0:
                val_loss = evaluate(model, val_set, args.val_batch_size, args.context_length, device)
                log_end_val = time.time()
                print(f"Iteration {t}| Time: {log_end_val - start:.2f}| Val Loss: {val_loss:.4f}")
                # Back to Train
                model.train()

    # output_dir = f"{args.output}_final"
    # save_checkpoint(model, optim, t, output_dir)
    
    if args.val is not None:
      model.eval()
      with torch.no_grad():
          x_val, gt_val = get_batch(val_set, args.batch_size, args.context_length, args.device)
          x_val.to(device)
          gt_val.to(device)

          y_val = model(x_val)

          loss_val = ce(y_val, gt_val)

          print(f"Iteration End | Val Loss: {(loss_val.item()):.4f}")


if __name__ == "__main__":
    train()

