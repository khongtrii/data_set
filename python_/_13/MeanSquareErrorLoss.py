import numpy as np
import torch
import torch.nn as nn

class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()
    def forward(self, y_pred, y_true):
        assert y_pred.shape == y_true.shape, "Predicted and Target must have the same shape."
        
        squared = torch.square(y_true - y_pred)
        
        mse = torch.mean(squared)
        _ = mse.detach().numpy().round(4)
        
        return mse, _
        # return mse
    
"""
if __name__ == "__main__":
    
    mse = MSE()
    
    predict = torch.tensor([2.5, 0.0, 2.1, 7.8], requires_grad=True)
    target = torch.tensor([3.0, -0.5, 2.0, 7.5])
    
    loss, _ = mse(predict, target)
    
    print("MSE: %.3f" % loss)
    
    loss.backward()
    print("Gradients:", predict.grad)
"""