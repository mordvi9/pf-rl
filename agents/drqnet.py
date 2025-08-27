import torch.nn as nn 
import torch.nn.functional as F

class DRQN(nn.Module):
    def __init__(self, num_layers = 1, obs_dim = 9, hidden_dim = 128, action_dim = 4):
        super().__init__()

        self.input = nn.Linear(obs_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, action_dim)


    def forward(self, x, h, c):
        x = F.relu(self.input(x))
        x, (new_h, new_c) = self.lstm(x,(h,c))
        x = self.output(x)
        return x, new_h, new_c
    
