import torch
import os
import sys
import torch.nn.functional as F
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agents.drqnet import DRQN
from agents.sequence_replay_buffer import SequenceReplayBuffer
import numpy as np
import copy
import random

class DRQAgent:
    def __init__(self, state_dim, obs_dim, action_dim, seq_len = 20, lr = 0.0005, gamma = 0.99, epsilon = 1, epsilon_decay = 0.98, epsilon_min = 0.1, buffer_size = 20000):
        self.qnet = DRQN(num_layers = 1, obs_dim = 9, hidden_dim = 64, action_dim = action_dim)
        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr = lr)

        self.target_net = copy.deepcopy(self.qnet)
        self.update_target_every = 1000
        self.train_steps = 0

        self.seq_len = seq_len
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.hidden_dim = 64 
        self.action_dim = action_dim

        self.gamma = gamma
        self.epsilon = epsilon 
        self.epsilon_decay = epsilon_decay 
        self.epsilon_min = epsilon_min

        self.replay = SequenceReplayBuffer(buffer_size, self.seq_len)
        self.current_episode = []

    def trimorfill(self,seq):
        if len(seq) > self.seq_len:
            return seq[-self.seq_len:]  
        else:
            return seq 
    
    def init_hidden(self, batch_size):
        h0 = torch.zeros([1, batch_size, self.hidden_dim])
        c0 = torch.zeros([1, batch_size, self.hidden_dim])
        return h0, c0

    def select_act(self, obs, h, c):
        q_vals, h, c = self.qnet(obs, h,c)        
        if torch.rand(1).item() <= self.epsilon:
            return torch.randint(0, self.action_dim, (1,)).item(), h, c
        else:
            q_vals = q_vals[0, -1]
            return q_vals.argmax().item(), h, c
                       
    def member(self, state, action, reward, next_state, done):
        self.current_episode.append((state, action, reward, next_state, done))
        if done:
            self.replay.push(self.current_episode)
            self.current_episode = []

    def freeze_eps(self):
        if not hasattr(self, "_eps_backup"):
            self._eps_backup = self.epsilon     
            self.epsilon = 0.0                   

    def restore_eps(self):
        if hasattr(self, "_eps_backup"):
            self.epsilon = self._eps_backup      
            del self._eps_backup     

    def train(self, batch_size, train_step):
        if len(self.replay) < batch_size:
            return None
        
        states, actions, rewards, next_states, dones, masks = self.replay.sample(batch_size)

        state_batch = torch.tensor(states, dtype=torch.float32)
        action_batch = torch.tensor(actions, dtype=torch.long)
        reward_batch = torch.tensor(rewards, dtype=torch.float32)
        next_state_batch = torch.tensor(next_states, dtype=torch.float32)
        done_batch = torch.tensor(dones, dtype=torch.float32)
        mask_batch = torch.tensor(masks, dtype=torch.float32)

        h0, c0 = self.init_hidden(batch_size)

        q_vals, h, c = self.qnet(state_batch, h0, c0)
        current_q = q_vals.gather(2, action_batch.unsqueeze(2)).squeeze(2)

        with torch.no_grad():
            target_q, h, c = self.target_net(next_state_batch, h0, c0)
            max_next_q = target_q.max(2)[0]
            targets = reward_batch + self.gamma * (1 - done_batch) * max_next_q

        loss = F.smooth_l1_loss(current_q * mask_batch, targets * mask_batch, reduction='sum')
        loss = loss / mask_batch.sum()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qnet.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.update_target_every == 0:
            self.target_net.load_state_dict(self.qnet.state_dict())

        return loss.item()
    
    def decay_eps(self):
        if len(self.replay) > 10000:
            self.epsilon = max(self.epsilon_min, self.epsilon* self.epsilon_decay)

