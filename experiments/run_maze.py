import sys
import os
import hydra
from omegaconf import DictConfig
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.maze import Maze
from filters.pf import ParticleFilter
from agents.pf_q_agent import PFQAgent
from agents.drq_agent import DRQAgent
from trainer import Trainer


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    env = Maze(cfg.p_noise)
    part_filter = ParticleFilter(cfg.n_particles, env.h, env.w, env.start_row, env.start_cols, env.grid, cfg.p_noise)
    pf_agent = PFQAgent(state_dim = 4, action_dim = 4, lr=cfg.p_lr, gamma=cfg.p_gamma, epsilon=cfg.p_epsilon, epsilon_decay=cfg.p_epsilon_decay, epsilon_min=cfg.p_epsilon_min, buffer_size=cfg.p_buffer_size)
    drqn_agent = DRQAgent(state_dim = (env.h*env.w), obs_dim = 9, action_dim = 4, seq_len = cfg.seq_len, lr=cfg.d_lr, gamma=cfg.d_gamma, epsilon=cfg.d_epsilon, epsilon_decay=cfg.d_epsilon_decay, epsilon_min=cfg.d_epsilon_min, buffer_size=cfg.d_buffer_size)
    agent = pf_agent
    pf = part_filter
    type = "pf"
    trainer = Trainer(agent, env, type, pf, save_results = True, num_eps = cfg.num_eps, batch_size = cfg.batch_size, eval_freq = cfg.eval_freq, eval_size = cfg.eval_size, max_steps = cfg.max_steps, warmup = cfg.warmup)
    trainer.train()
    trainer.save_results(loss=True, qnet=True, entropy=True, ess = True)

if __name__ == "__main__":
    main()
