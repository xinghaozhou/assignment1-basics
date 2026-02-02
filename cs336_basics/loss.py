import torch
import torch.nn as nn
#from cs336_basics.softmax import softmax

class CrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):

        dtype = inputs.dtype

        targets_oh = nn.functional.one_hot(
            targets, num_classes=inputs.size(-1)
        ).to(dtype)

        # numerical stability
        inputs = inputs - inputs.max(dim=-1, keepdim=True).values

        # logsumexp (denominator)
        softmax_bottom = torch.log(torch.exp(inputs).sum(dim=-1))

        # target logit (numerator, already in log space)
        target_logit = (inputs * targets_oh).sum(dim=-1)

        # cross entropy
        out = softmax_bottom - target_logit

        return out.mean()




# class CrossEntropy(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
#         softmax = Softmax(dim=-1)

#         probs = softmax(inputs)
        
#         selected_probs = probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
#         return -torch.mean(selected_probs)

   
