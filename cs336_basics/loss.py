import torch
import torch.nn as nn
#from cs336_basics.softmax import softmax

class CrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        # numerical stability
        inputs = inputs - inputs.max(dim=-1, keepdim=True).values

        target_logit = torch.gather(inputs, -1, targets[..., None]).squeeze(-1)

        # logsumexp (denominator)
        softmax_bottom = torch.log(torch.exp(inputs).sum(dim=-1))

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

   
