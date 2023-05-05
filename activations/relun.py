import torch.nn as nn
import torch

class ReLUN(nn.Module):
    def __init__(self, _min = 1.0):
        super(ReLUN,self).__init__()
        self._min = nn.Parameter(torch.tensor(_min))
        self._min.requiresGrad = True

    def forward(self, x):
        return torch.min(torch.relu(x), self._min)
    
    def get_param(self):
        return self._min