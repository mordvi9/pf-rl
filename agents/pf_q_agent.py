import torch
import os
import sys
import torch.nn.functional as F
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agents.qnet import QNet
from agents.replay_buffer import ReplayBuffer
import numpy as np
import copy

class PFQAgent:
    def __init__(self, state_dim, action_dim, lr = 0.0005, gamma = 0.99, epsilon = 1, epsilon_decay = 0.995, epsilon_min = 0.01, buffer_size = 10000):

        self.qnet = QNet(state_dim, hidden = [256,256], output_dim = action_dim )
        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr = lr)

        self.target_net = copy.deepcopy(self.qnet)
        self.update_target_every = 100
        self.train_steps = 0

        self.state_dim = state_dim 
        self.action_dim = action_dim 

        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon 
        self.epsilon_decay = epsilon_decay 
        self.epsilon_min = epsilon_min
        self.eps_backup = None

        self.replay = ReplayBuffer(buffer_size)

    def select_act(self, belief):
        if torch.rand(1).item() <= self.epsilon:
            return int(torch.randint(0,4,(1,)).item())
        tens = torch.from_numpy(belief).float().unsqueeze(0)
        with torch.no_grad():
            q_vals = self.qnet(tens)
        return q_vals.argmax(dim = 1).item()
                       
    def member(self, state, action, reward, next_state, done):
        self.replay.push((state, action, reward, next_state, done))
    
    def freeze_eps(self):
        if not hasattr(self, "_eps_backup"):
            self._eps_backup = self.epsilon     
            self.epsilon = 0.0                   

    def restore_eps(self):
        if hasattr(self, "_eps_backup"):
            self.epsilon = self._eps_backup      
            del self._eps_backup     

    def train(self, batch_size, step_num):
        if len(self.replay) < batch_size:
            return None
        
        batch = self.replay.sample(batch_size)
        states, actions, rewards, next_states, dones, masks = zip(*batch)

        states = torch.tensor(np.array(states), dtype = torch.float32)
        actions = torch.tensor(actions, dtype = torch.long)
        rewards = torch.tensor(rewards, dtype = torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype = torch.float32)
        dones = torch.tensor(dones, dtype = torch.float32)

        q_vals = self.qnet(states)
        q_s_a = q_vals.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            q_next = self.target_net(next_states)
            max_q_next = q_next.max(dim=1)[0]
            target_q = rewards + (1-dones) * self.gamma * max_q_next
        
        loss = F.mse_loss(q_s_a, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qnet.parameters(), max_norm=2.0)
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.update_target_every == 0:
            self.target_net.load_state_dict(self.qnet.state_dict())

        return loss.item()
    
    def decay_eps(self):
        if len(self.replay) > 5000:
            self.epsilon = max(self.epsilon_min, self.epsilon* self.epsilon_decay)

