import torch
import numpy as np
import numpy.typing as npt
from typing import Tuple


def get_batch(
    dataset: npt.NDArray, 
    batch_size: int, 
    context_length: int, 
    device: str | None = None, 
    dtype: torch.dtype | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    dataset = torch.tensor(dataset, device=device, dtype=dtype)
    n = dataset.size(0)

    start_point = torch.randint(low = 0, high = n - context_length, size=(batch_size,), device=device, dtype=dtype) # Sampling B starting points from [1, n-m)
    
    offset = torch.arange(context_length, device=device, dtype=dtype) # Make context_length offset

    idx = (start_point[:, None] + offset[None, :]) # Make [b, context_len] first pair
    
    x = dataset[idx]
    y = dataset[idx + 1]





    # N = len(data)

    # ix = np.random.randint(
    #     0,
    #     N - context_length - 1,
    #     size=(batch_size,)
    # )

    # x = np.stack([data[int(i):int(i)+context_length] for i in ix])
    # y = np.stack([data[int(i)+1:int(i)+1+context_length] for i in ix])

    # x = torch.from_numpy(x).long().to(device)
    # y = torch.from_numpy(y).long().to(device)

    return x, y

# def get_batch(
#     dataset: npt.NDArray,
#     batch_size: int,
#     context_length: int, device: str
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     begin = np.random.randint(0, len(dataset) - context_length, size=batch_size)

#     input = np.stack([dataset[bi : bi + context_length] for bi in begin])
#     label = np.stack([dataset[bi + 1 : bi + context_length + 1] for bi in begin])

#     input_tensor = torch.tensor(input, dtype=torch.long, device=device)
#     label_tensor = torch.tensor(label, dtype=torch.long, device=device)

#     return input_tensor, label_tensor

