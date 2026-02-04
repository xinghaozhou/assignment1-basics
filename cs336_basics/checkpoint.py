import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

def save_checkpoint(model:torch.nn.Module, 
                    optimizer: torch.optim.Optimizer, 
                    iteration: int, 
                    out: str | os.PathLike | BinaryIO | IO[bytes]):

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
