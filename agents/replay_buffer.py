import random

class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = []
        self.max_size = max_size
    
    def push(self, transition):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)