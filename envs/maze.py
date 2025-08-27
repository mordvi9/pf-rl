import numpy as np
import gymnasium as gym
from gymnasium import spaces

class Maze(gym.Env):

    def __init__(self, noise_level):
        super().__init__()

        self.grid = np.array([
            [1,1,1,1,1,1,1,1,1,1,1],  # r0 border
            [1,0,0,0,0,0,0,0,0,0,1],  # r1 corridor   
            [1,0,1,1,0,0,0,1,1,0,1],  # r2 corridor
            [1,0,0,0,0,0,0,0,0,0,1],  # r3 corridor 
            [1,1,2,2,0,0,0,1,1,1,1],  # r4 fork 
            [1,1,2,0,0,0,0,0,0,1,1],  # r5 down the shared stem
            [1,2,2,0,2,0,0,1,1,1,1],  # r6 landmarks 
            [1,0,0,0,0,0,0,0,0,1,1],  # r7 corridor 
            [1,0,0,0,0,0,0,0,0,0,1],  # r8 corridor 
            [1,0,3,3,3,3,3,3,3,1,1],  # r9 left exit at (9,2), right exit at (9,8)
            [1,1,1,1,1,1,1,1,1,1,1],  # r10 border
        ], dtype=int)

        self.h, self.w = self.grid.shape
        self.noise_level = noise_level
        
        self.obs_space = spaces.Box(low = 0, high = 3, shape = (9,), dtype=np.int32)
        self.action_space = spaces.Discrete(4)
        self.base_grid = self.grid.copy()

        self.agent_pos = None
        self.exit_pos = (9,7) #if np.random.rand() < 0.5 else (9,7) 
        self.grid[9][tuple(self.exit_pos)[1]] = 0

        self.start_row = 1
        self.start_cols = [1,9]

        self.seen_landmark = False
        self.gamma = 0.99
        

    def reset(self, seed = None):
        super().reset(seed=seed)
        self.grid = self.base_grid.copy()
        self.exit_pos = (9,7) #if np.random.rand() < 0.5 else (9,7) 
        self.grid[9][tuple(self.exit_pos)[1]] = 0
        self.agent_pos = np.array([self.start_row, np.random.choice(self.start_cols)])
        self.seen_landmark = False

        obs = self.get_obs()
        info = {"agent_pos": self.agent_pos.tolist(), "exit_pos": self.exit_pos}
        return obs, info
    
    def step(self, action):
        #0 = UP, 1 = DOWN, 2 = RIGHT, 3 = LEFT
        slip = True if np.random.rand() < 0.01 else False

        if slip:
            action = np.random.randint(0, 4, dtype = int) 

        r, c = self.agent_pos
        
        if action == 0:
            r_new = r - 1
            c_new = c 
        elif action == 1:
            r_new = r + 1
            c_new = c 
        elif action == 2:
            r_new = r 
            c_new = c+1
        else:
            r_new = r
            c_new = c-1

        reward = -0.02
        done = False

        if not (0 <= r_new < self.h and 0 <= c_new < self.w and self.grid[r_new, c_new] == 0):
            self.agent_pos = np.array([r, c])
        else:
            self.agent_pos = np.array([r_new, c_new])

        def potential(self, r, c):
            return 1.4 * (r / (self.h - 1))
        
        pos_old = (r,c)
        pos_new = tuple(self.agent_pos)
        reward += self.gamma * potential(self, *pos_new) - potential(self, *pos_old)
    
        
        if pos_new[0] == 4 and not self.seen_landmark:
            reward += 0.2
            self.seen_landmark = True
        
        if pos_new == pos_old:
            reward -= 0.05

        if pos_new == self.exit_pos:
            reward = 1
            done = True

        truncated = False
        obs = self.get_obs()  
        info = {"agent_pos": self.agent_pos.tolist(), "exit_pos": self.exit_pos}

        reward = max(-1.0, min(1.0, reward))

        return obs, reward, done, truncated, info
        
    def get_obs(self):
        r, c = self.agent_pos
        r_max = min(r+1, self.h - 1)
        r_min = max(r-1, 0)
        c_max = min(c+1, self.w - 1)
        c_min = max(c-1, 0)

        obs = np.ones((3,3), dtype = int)
        actual = self.grid[r_min:r_max+1, c_min:c_max+1]

        row_start = 1 - (r-r_min)
        col_start = 1 - (c-c_min)

        obs[row_start:actual.shape[0] +row_start, col_start:actual.shape[1] + col_start] = actual

        if np.random.rand() < 0.5:
            obs = np.fliplr(obs) 
        
        noise_mask = (np.random.rand(3,3) < self.noise_level).astype(np.int32)
        noisy_obs = (obs + noise_mask) % 3

        return noisy_obs.flatten()

    def render(self):
        r,c, = self.agent_pos
        er, ec = self.exit_pos
        disp = []
        for i in range(self.h):
            row = ""
            for j in range(self.w):
                if (i,j) == (r,c):
                    row += "A"
                elif (i,j) == (er,ec):
                    row += "X"
                elif self.grid[i][j] == 1:
                    row += "="
                else:
                    row += " "
            disp.append(row)
        print("\n".join(disp))
        print()

                

