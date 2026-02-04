import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
from einops import rearrange, einsum


def scaled_dot_product_attention(
        Q: Float[Tensor, " ... queries d_k"],
        K: Float[Tensor, " ... keys d_k"],
        V: Float[Tensor, " ... values d_v"],
        mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    qtk = einsum(
        Q, K,
        "... queries d_k, ... keys d_k -> ... queries keys"
    )

    sqrtd = torch.sqrt(torch.tensor(Q.size(-1)))

    value = qtk / sqrtd
    value = value.masked_fill(~mask, float("-inf")) # Do masking first, Put -inf for those ~mask (True mask)

    max_v = value.max(dim=-1, keepdim=True).values

    softmax = torch.exp(value - max_v) / torch.exp(value - max_v).sum(dim=-1, keepdim=True)

    out = softmax * mask

    out = softmax @ V

    return out