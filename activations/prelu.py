import torch.nn as nn
import torch

class PReLU(nn.Module):
    def __init__(self, alpha = 1.0):
        super(PReLU,self).__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.alpha.requiresGrad = True

    def forward(self, x):
        return torch.max(x, self.alpha*x)
    
    def get_param(self):
        return self.alpha