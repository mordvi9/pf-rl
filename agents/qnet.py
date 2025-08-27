import torch 
import torch.nn as nn 

class QNet(nn.Module):
    def __init__(self, input_dim = 80, hidden = [256,256], output_dim = 4):
        super(QNet, self).__init__()
        self.il = nn.Linear(input_dim, hidden[0])
        self.hl = nn.Linear(hidden[0], hidden[1])
        self.out = nn.Linear(hidden[1], output_dim)

    def forward(self, x):
        x = torch.relu(self.il(x))
        x = torch.relu(self.hl(x))
        x = self.out(x)
        return x
