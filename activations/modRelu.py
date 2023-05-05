import torch
import torch.nn as nn

class ModReLU(nn.Module):
    def __init__(self, n):
        super(ModReLU, self).__init__()
        self.bias = nn.Parameter(torch.zeros(n))
        self.bias.requires_grad = True

    def forward(self, x):
        magnitude = torch.abs(x)
        phase = x / (magnitude + 1e-9)
        relu = nn.ReLU()
        magnitude = relu(magnitude + self.bias.view(-1, 1))

        return magnitude * phase
