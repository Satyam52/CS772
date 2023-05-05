import torch.nn as nn
import torch

class LearnedSiLU_1(nn.Module):
    def __init__(self, slope = 1.0):
        super(LearnedSiLU,self).__init__()
        self.slope = nn.Parameter(torch.tensor(slope))
        self.slope.requiresGrad = True

    def forward(self, x):
        #print(self.slope.is_cuda, x.is_cuda)
        return self.slope * x * torch.sigmoid(x)

class LearnedSiLU(nn.Module):
    def __init__(self, slope = 1.0, embedding_dim=1):
        super(LearnedSiLU,self).__init__()
        self.slope = nn.Parameter(torch.ones(embedding_dim)*slope)
        self.slope.requiresGrad = True

    def forward(self, x):
        #print(self.slope.is_cuda, x.is_cuda)
        #print(self.slope.shape, x.shape)
        return self.slope * x * torch.sigmoid(x)

    def get_param(self):
        return self.slope