import torch.nn as nn

class gradient_clipping(nn.Module):
    def __init__(self, max_l2_norm):
        super().__init__()
        self.max_l2_norm = max_l2_norm

    def forward(self, parameters):
        
        total_norm_sq = 0.0

        # calculate l2 norm for all parameters
        for p in parameters:
            if p.grad is not None:
                total_norm_sq += p.grad.norm(2).item() ** 2

        total_norm = total_norm_sq ** 0.5


        if total_norm > self.max_l2_norm:
            scale = self.max_l2_norm / (total_norm + 1e-6)
            for p in parameters:
                if p.grad is not None:
                    p.grad.mul_(scale) # in-place multiply, otherwise use copy_

        return parameters