import torch.nn as nn
import torch

class LearnedSwish(nn.Module):
    def __init__(self, beta = 1.0):
        super(LearnedSwish,self).__init__()
        self.beta = nn.Parameter(torch.tensor(beta))
        self.beta.requiresGrad = True

    def forward(self, x):
        #print(self.slope.is_cuda, x.is_cuda)
        return x * torch.sigmoid(self.beta*x)