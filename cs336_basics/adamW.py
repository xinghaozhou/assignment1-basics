import torch
from collections.abc import Callable
from typing import Optional, Tuple

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, betas, eps, weight_decay, lr=1e-3):

        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay} # set up all init params

        super().__init__(params, defaults)
        

    def step(self, closure: Optional[Callable] = None) -> None:

        loss = None if closure is None else closure() # Passed in callable function, when using it call it.

        for group in self.param_groups: # Group = all hyperparameters used in this object
            lr = group["lr"]
            betas = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]: # Here, p = parameter = theta
                if p.grad is None: # no backward() / frozen / not chosen MoE / zero grad
                    continue

                # Because every parameter in AdamW has its own m/v/t, 
                state = self.state[p]

                 # It is the first time using this parameter, therefore do initialize
                t = state.get("t", 1)
                m = state.get("m", torch.zeros_like(p))
                v = state.get("v", torch.zeros_like(p))

                grad = p.grad.data # get grad at state

                m = betas[0] * m + (1 - betas[0]) * grad
                v = betas[1] * v  + (1 - betas[1]) * (grad ** 2)

                bias_correction2 = 1 - betas[1] ** t
                bias_correction1 = 1 - betas[0] ** t
                alpha_t = lr * bias_correction2**(1/2) / bias_correction1

                p.data = p.data - alpha_t * m / (torch.sqrt(v) + eps)

                p.data = p.data - lr * weight_decay * p.data

                # Needs to store back
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
