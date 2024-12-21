import torch 
import torch.nn as nn
from functional import sigmoid

class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
    def forward(self, y_true, y_pred, weight = None):
        """
            x --> y_pred --> input
            y --> y_true --> target
        """
        
        assert y_true.shape == y_pred.shape, "Ensure that their shape matches."
        m = sigmoid()
        
        x = m(y_pred)
        y = y_true
        w = weight
        if w==None:
            bce = - (y * torch.log(x) + (1 - y) * torch.log(1 - x))
            bce = torch.mean(bce)
        else:
            assert y_true.shape == y_pred.shape == weight.shape, "Ensure that their shape matches."
            bce = - w * (y * torch.log(x) + (1 - y) * torch.log(1 - x))
            bce = torch.mean(bce)
        
        return bce
    
"""        
if __name__ == '__main__':
    
    y_true = torch.tensor([0, 1, 1, 0, 1], dtype=torch.float32)
    y_pred = torch.tensor([0.1, 0.9, 0.8, 0.2, 0.7], dtype=torch.float32, requires_grad=True)
    # weight = torch.rand_like(y_true, dtype=torch.float32)
    bce = BCELoss()
    loss = bce(y_true, y_pred)#, weight)
    
    print(f"BCE Loss: {loss.item()}")
    loss.backward()
    print(f"Gradients: {y_pred.grad}")
"""