import torch
import torch.nn as nn

class sigmoid(nn.Module):
    def __init__(self):
        super(sigmoid, self).__init__()
        
    def forward(self, x):
        return 1 / (1 + torch.exp(-(x)))

class softmax(nn.Module):
    def __init__(self):
        super(softmax, self).__init__()
    
    def forward(self, x):
        return torch.exp(x) / torch.sum(torch.exp(x))

class ReLU(nn.Module):
    def __init__(self):
        super(ReLU, self).__init__()
        
    def forward(self, x):
        return torch.maximum(x, torch.tensor(0.0))

class ReLU6(nn.Module):
    def __init__(self):
        super(ReLU6, self).__init__()
        
    def forward(self, x):
        return torch.minimum(torch.maximum(input, torch.tensor(0)), torch.tensor(6))
    
class GeLU(nn.Module):
    def __init__(self):
        super(GeLU, self).__init__()
    
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2 / torch.pi)) * (x + 0.044715 * x**3)))