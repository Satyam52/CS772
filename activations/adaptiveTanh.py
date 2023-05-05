import torch
import torch.nn as nn

class AdaptiveTanh(nn.Module):
    def __init__(self, n):
        super(AdaptiveTanh, self).__init__()
        self.alpha = nn.Parameter(torch.ones(n))
        self.beta = nn.Parameter(torch.ones(n))

    def forward(self, x):
        return self.alpha.view(-1, 1) * torch.tanh(self.beta.view(-1, 1) * x)