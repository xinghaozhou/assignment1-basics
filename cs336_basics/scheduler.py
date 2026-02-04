import torch
import torch.nn as nn
import math

class lr_schedule(nn.Module):
    def __init__(self,
                 max_learning_rate: float, 
                 min_learning_rate: float,
                 warmup_iters=int, 
                 cosine_cycle_iters=int):
        super().__init__()
        self.max_learning_rate = max_learning_rate
        self.min_learning_rate = min_learning_rate
        self.warmup_iters = warmup_iters
        self.cosine_cycle_iters = cosine_cycle_iters


    def forward(self, it):
        if it < self.warmup_iters:
            a_t = it / self.warmup_iters * self.max_learning_rate
        elif self.warmup_iters <= it and it <= self.cosine_cycle_iters:
            a_t = self.min_learning_rate + 1/2 * (1 + math.cos(((it - self.warmup_iters) / (self.cosine_cycle_iters - self.warmup_iters) * torch.pi))) * (self.max_learning_rate - self.min_learning_rate)
        else:
            a_t = self.min_learning_rate

        return a_t