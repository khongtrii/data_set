import numpy as np
import torch
import torch.nn as nn



class MAE(nn.Module):
    def __init__(self):
        super(MAE, self).__init__()
    def forward(self, y_pred, y_true):
        assert y_pred.shape == y_true.shape, "Predicted and Target must have the same shape."
        
        abs = torch.abs(y_true - y_pred)
        
        mae = torch.mean(abs)
        _ = mae.detach().numpy().round(4)
        
        return mae, _
    
"""
if __name__ == "__main__":
    
    mae = MAE()s
    
    predict = torch.tensor([2.5, 0.0, 2.1, 7.8], requires_grad=True)
    target = torch.tensor([3.0, -0.5, 2.0, 7.5])
    
    loss, _ = mae(predict, target)
    
    print("MAE: %.3f" % loss)
    
    loss.backward()
    print("Gradients:", predict.grad)
"""