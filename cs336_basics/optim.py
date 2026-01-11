from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math



weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
opt = torch.optim.SGD([weights], lr=1e3)

for t in range(10):
    opt.zero_grad()
    loss = (weights**2).mean()
    print(loss.cpu().item())
    loss.backward()
    opt.step()
