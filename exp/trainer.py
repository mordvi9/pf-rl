import torch 
import numpy as np 
import os, csv
import datetime as dt

class Trainer:
    def __init__(self, agent, env, type, pf = None, save_results=False, num_eps = 5000, batch_size = 64, eval_freq = 100, eval_size = 50, max_steps = 1000, warmup = 10, out_path="results/eval.csv"):
        self.agent = agent 
        self.env = env
        self.pf = pf 
        self.type = type
        self.out_path = out_path

        self.num_eps = num_eps
        self.batch_size = batch_size
        self.eval_freq = eval_freq
        self.eval_size = eval_size
        self.max_steps = max_steps
        self.warmup = warmup

        self.loss_log = []
        self.test_log = []
        self.ep_returns = []
        self.ep_steps = []
        self.eps_log = []

        self.entropy_log = []
        self.ess_hist = []

        self.global_step = 0

        self.ax = None

        self.save_res = save_results
        self.exp_id = type       
        self.run_seed = 0               
        self.res_out_path = f"results/{self.exp_id}_eval_seed{self.run_seed }.csv"
        self.loss_out_path = f"results/{self.exp_id}_loss_seed{self.run_seed }.npy"
        self.qnet_out_path = f"results/{self.exp_id}_qnet_seed{self.run_seed }.pt"
        self.entropy_out_path = f"results/{self.exp_id}_entropy_seed{self.run_seed }.npy"
        self.ess_out_path = f"results/{self.exp_id}_ess_seed{self.run_seed }.npy"

        if self.save_res:
            os.makedirs("results", exist_ok=True)
            os.makedirs(os.path.dirname(self.res_out_path), exist_ok=True)
            with open(self.res_out_path, "w", newline="") as f:
                csv.writer(f).writerow(["episode", "mean_reward", "mean_length", "timestamp"])

    def run_episode(self, render= False, ep=None, train= True):
        H = []
        obs, _ = self.env.reset()
        if self.type == "pf":
            self.pf.init_particles()
            belief = self.pf.mean_belief()
        elif self.type == "drqn":
            h, c = self.agent.init_hidden(batch_size = 1)
        else:
            self.pf.init_particles()
            belief = self.pf.mean_belief()
            h, c = self.agent.init_hidden(batch_size = 1)

        done = False
        total_reward = 0.0

        for t in range(self.max_steps):
            if train:
                self.global_step += 1
            if self.type == "pf":
                action = self.agent.select_act(belief)
            elif self.type == "drqn":
                s = torch.from_numpy(obs).float().view(1, 1, -1)
                action, h, c = self.agent.select_act(s, h, c)
                h = h.detach()
                c = c.detach()
            else:
                action, h, c = self.agent.select_act(belief, h, c)
                h = h.detach()
                c = c.detach()

            s_prime, reward, done, *_ = self.env.step(action)
            total_reward += reward

            if self.type == "pf":
                self.pf.predict(action)
                self.pf.update(s_prime)

            if self.type == "pf":
                if train:
                    ess = 1.0 / np.sum(self.pf.weights**2)
                    self.ess_hist.append((self.global_step, ess))
                    H_t = -np.sum(self.pf.weights*np.log(self.pf.weights+1e-12))
                    self.entropy_log.append((self.global_step, H_t))
            
                self.pf.resample()
                next_belief = self.pf.mean_belief()
            
            if train:
                if self.type == "pf" :
                    self.agent.member(belief, action, reward, next_belief, done)
                else:
                    self.agent.member(obs, action, reward, s_prime, done)

            warm = 100 if self.pf else self.batch_size 
            if len(self.agent.replay) >= warm:
                loss = self.agent.train(self.batch_size, ep)    
                if loss is not None:
                    self.loss_log.append((self.global_step,loss))
                #self.agent.decay_eps()

            obs = s_prime
            if self.type == "pf":
                belief = next_belief
            if done:
                break
        
        return total_reward, t + 1, H
    
    def evaluate(self, render=False):
        old_step = self.global_step
        self.agent.freeze_eps()
        eval_rewards = []
        eval_lengths = []
        for i in range(self.eval_size):
            rew, l, _ = self.run_episode(render, ep=None, train = False)
            eval_rewards.append(rew)
            eval_lengths.append(l)
        self.agent.restore_eps()
        mean_rew = np.mean(eval_rewards)
        mean_len = np.mean(eval_lengths)
        self.test_log.append((mean_rew, mean_len))
        self.global_step = old_step  
        return float(mean_rew), float(mean_len)

    def train(self):
        self.global_step = 0
        best_eval_reward = -float('inf') 
        for ep in range(self.num_eps):

            render = (ep % 50 == 0 and ep > 0)
            train_reward, train_len, H = self.run_episode(render, ep)

            if ep > self.warmup:
                self.ep_returns.append(train_reward)
                self.ep_steps.append(train_len)
                self.eps_log.append(self.agent.epsilon)
                if len(H) > 0:
                    self.entropy_log.append([ep, np.mean(H)])
                self.agent.decay_eps()

            if ep % self.eval_freq == 0:
                mean_r, mean_l = self.evaluate(render=False)
                timestamp = dt.datetime.now().isoformat()
                print(f"mean reward = {mean_r:.3f}; mean length = {mean_l:.3f}")

                if mean_r > best_eval_reward:
                    print(f"New best evaluation reward: {mean_r:.3f} (old best was {best_eval_reward:.3f}). Saving best model...")
                    best_eval_reward = mean_r
                    torch.save(self.agent.qnet.state_dict(), f"{self.exp_id}_best_model.pt")

                if self.save_res:
                    header = ["episode", "mean_reward", "mean_length", "timestamp"]
                    with open(self.res_out_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        if f.tell() == 0:                     
                            writer.writerow(header)
                        writer.writerow([ep, mean_r, mean_l, timestamp])
    
                

    def save_results(self, loss=False, qnet=False, entropy=False, ess= False):
        if loss:
            np.save(f"{self.loss_out_path}",  np.array(self.loss_log))
        if qnet:
            torch.save(self.agent.qnet.state_dict(), f"{self.qnet_out_path}")
        if entropy:
            np.save(f"{self.entropy_out_path}",  np.array(self.entropy_log))
        if ess:
            np.save(f"{self.ess_out_path}",  np.array(self.ess_hist))
