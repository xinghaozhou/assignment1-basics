import torch
import numpy as np
from cs336_basics.loss import CrossEntropy
from cs336_basics.data_loading import get_batch

@torch.no_grad()
def evaluate(
    model, data: np.memmap, 
    val_batch_size: int, 
    context_length: int, 
    device: str | None = None,
    dtype: torch.dtype | None = None
):
    
    kwargs = {'device': device, 'dtype': dtype}
    model.eval()
    ce = CrossEntropy()

    loss_val_cum = 0

    for _ in range(val_batch_size): # get mu.t
        x_val, gt_val = get_batch(data, val_batch_size, context_length, **kwargs)
        x_val = x_val.to(device)
        gt_val = gt_val.to(device)

        y_val = model(x_val)

        loss_val = ce(y_val, gt_val)
        loss_val_cum += loss_val

    return loss_val_cum / val_batch_size

