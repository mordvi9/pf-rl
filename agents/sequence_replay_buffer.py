import random
import numpy as np

class SequenceReplayBuffer:
    def __init__(self, max_episodes, seq_len):
        self.max_episodes = max_episodes
        self.seq_len = seq_len
        self.episodes = []

    def __len__(self):
        return len(self.episodes)

    def push(self, episode):
        if len(self.episodes) >= self.max_episodes:
            self.episodes.pop(0)
        self.episodes.append(list(episode))

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones, masks = [], [], [], [], [], []
        for i in range(batch_size):
            sample = random.choice(self.episodes)
            if len(sample) >= self.seq_len:
                seq = sample[-self.seq_len:]
                mask = np.ones(self.seq_len, dtype=bool)  
            else:
                pad_length = self.seq_len - len(sample)
                pad = [(np.zeros_like(sample[0][0]), 0, 0.0, np.zeros_like(sample[0][0]), False)] * pad_length
                seq = pad + sample
                mask = np.array([False] * pad_length + [True] * len(sample))
            
            s, a, r, ns, d = zip(*seq)
            states.append(np.stack(s))
            actions.append(np.stack(a))
            rewards.append(np.stack(r))
            next_states.append(np.stack(ns))
            dones.append(np.stack(d))
            masks.append(mask)
        
        return (np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones), np.array(masks))