import torch
import numpy as np


def run_get_batch(
    data: np.memmap, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    N = len(data)

    ix = np.random.randint(
        0,
        N - context_length - 1,
        size=(batch_size,)
    )

    x = np.stack([data[int(i):int(i)+context_length] for i in ix])
    y = np.stack([data[int(i)+1:int(i)+1+context_length] for i in ix])

    x = torch.from_numpy(x).long().to(device)
    y = torch.from_numpy(y).long().to(device)

    return x, y
