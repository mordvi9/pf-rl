import numpy as np


class ParticleFilter:
    def __init__(self, n, h, w, start_row, start_cols, grid, p_noise, c=-0.7):
        self.n = n
        self.h = h
        self.w = w
        self.start_row = start_row
        self.start_cols = start_cols
        self.grid = grid
        self.p_noise = p_noise
        self.c = c

        self.particles = np.zeros((n, 2), dtype = float) 
        self.weights = np.ones(n) / n 

        self.init_particles()        
    
    def init_particles(self):
        for k in range(self.n):
            start_row = float(self.start_row)
            free_corridor = np.argwhere(self.grid[self.start_row] == 0)[:, 0]  
            c = float(np.random.choice(free_corridor))
            self.particles[k] = (float(self.start_row), c)
        self.weights = np.ones(self.n) / self.n 

    def predict(self, action):

        dr = np.array([-1, +1, 0,  0], dtype = float)
        dc = np.array([ 0,  0, +1, -1], dtype = float)

        intend = np.stack([dr[action]*np.ones(self.n), dc[action]*np.ones(self.n)], axis = 1)
        noise = np.random.normal(loc=0.0, scale=0.1, size=(self.n, 2))

        new_particles = self.particles + intend + noise

        rows_f = np.clip(new_particles[:, 0], 0, self.h - 1)
        cols_f = np.clip(new_particles[:, 1], 0, self.w - 1)

        rows_i = np.floor(rows_f).astype(int)
        cols_i = np.floor(cols_f).astype(int)

        is_free = (self.grid[rows_i, cols_i] == 0)
        self.particles[is_free] = new_particles[is_free]

        return

    def update(self, obs_flat):
        obs_vec = obs_flat.astype(int)                    
        p = float(self.p_noise)
        p = min(max(p, 1e-6), 1.0 - 1e-6)

        pad = np.pad(self.grid, 1, constant_values=1)
        r = np.clip(self.particles[:,0].astype(int), 0, self.h-1) + 1
        c = np.clip(self.particles[:,1].astype(int), 0, self.w-1) + 1
        offsets = np.arange(3) - 1 
        rows = r[:, None, None] + offsets[None, :, None]
        cols = c[:, None, None] + offsets[None, None, :]

        actual = pad[rows, cols] 
        mirror = actual[:, :, ::-1] 
        perf_obs = actual.reshape(self.n, -1)
        mirror_obs = mirror.reshape(self.n, -1)

        #c = -0.7
        err_perf = self.c* ((obs_vec - perf_obs)** 2).sum(axis=1)
        err_mirror = self.c* ((obs_vec - mirror_obs)** 2).sum(axis=1)

        log_w = np.logaddexp(err_perf, err_mirror) - np.log(2.0)
        w = np.exp(log_w - log_w.max())
        self.weights = w / w.sum()

    def resample(self):
        ess = 1.0 / np.sum(self.weights**2)
        if ess < 0.5 * self.n:
            i = self.systematic_resample(self.weights)
            self.particles = self.particles[i]
            self.weights = np.ones(self.n) / self.n
    
    def systematic_resample(self, weights):
        n = len(weights)
        positions = (np.arange(n) + np.random.rand()) / n
        indexes = np.zeros(n, dtype=int)
        cumulative_sum = np.cumsum(weights)
        i = 0
        j = 0
        while i < n:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        return indexes

    def thompson_sampling(self):
        i = np.random.choice(self.n, p = self.weights)
        return self.particles[i]
    
    def get_belief(self):
        h = np.zeros((self.h, self.w), dtype=float)
        for k in range(self.n):
            r_cont, c_cont = self.particles[k]
            r_i = int(np.floor(r_cont))
            c_i = int(np.floor(c_cont))
            r_i = np.clip(r_i, 0, self.h - 1)
            c_i = np.clip(c_i, 0, self.w - 1)
            h[r_i, c_i] += self.weights[k]
        return h.flatten()

    def belief_avg(self):
        return np.average(self.particles, axis=0, weights=self.weights)
    
    def mean_belief(self):
        mean = self.belief_avg()
        variance = np.average((self.particles - mean)**2, axis=0, weights=self.weights)
        std = np.sqrt(variance)
        return np.concatenate([mean, std])  