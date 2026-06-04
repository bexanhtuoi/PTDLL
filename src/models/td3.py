from __future__ import annotations

import numpy as np
import torch
from torch import nn, optim

from models.base import BaseModel, DeterministicActor, ReplayBuffer, TwinQNet


class TD3Agent(BaseModel):
    def __init__(
        self, lookback: int, n_assets: int, n_features: int = 10,
        lr: float = 3e-4, gamma: float = 0.99, tau: float = 0.005,
        policy_noise: float = 0.2, noise_clip: float = 0.4,
        policy_delay: int = 2, exploration_noise: float = 0.10,
        entropy_coef: float = 0.2,
        batch_size: int = 64, replay_capacity: int = 100000,
        device: str = "cpu",
    ):
        self.lookback = lookback
        self.n_assets = n_assets
        self.n_features = n_features
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.exploration_noise = exploration_noise
        self.entropy_coef = entropy_coef
        self.batch_size = batch_size
        self.total_steps = 0

        self.actor = DeterministicActor(lookback, n_assets, n_features).to(device)
        self.target_actor = DeterministicActor(lookback, n_assets, n_features).to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        for p in self.target_actor.parameters():
            p.requires_grad = False

        self.critic = TwinQNet(lookback, n_assets, n_features).to(device)
        self.target_critic = TwinQNet(lookback, n_assets, n_features).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        for p in self.target_critic.parameters():
            p.requires_grad = False

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr, weight_decay=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr, weight_decay=1e-5)

        self.buffer = ReplayBuffer(replay_capacity)

    def get_weights(self, state: np.ndarray) -> np.ndarray:
        self.actor.eval()
        with torch.no_grad():
            return self.actor(torch.from_numpy(state).unsqueeze(0).to(self.device)).squeeze(0).cpu().numpy()

    def _act(self, state: np.ndarray) -> np.ndarray:
        self.actor.eval()
        with torch.no_grad():
            state_t = torch.from_numpy(state).unsqueeze(0).to(self.device)
            weights = self.actor(state_t).squeeze(0).cpu().numpy()
            noise = np.random.normal(0, self.exploration_noise, size=weights.shape)
            weights = weights + noise
            weights = np.clip(weights, 0, None)
            total = weights.sum()
            if total > 1e-10:
                weights = weights / total
            else:
                weights = np.ones_like(weights) / len(weights)
            return weights

    def train_episode(self, env, start_idx: int | None = None, end_idx: int | None = None) -> tuple[float, dict]:
        state = env.reset(start_idx=start_idx, end_idx=end_idx)
        done = False
        while not done:
            action = self._act(state)
            next_state, reward, done, info = env.step(action)
            self.buffer.push(state.copy(), action.copy(), reward, next_state.copy(), done)
            state = next_state
            self.total_steps += 1
            if len(self.buffer) >= self.batch_size:
                s_b, a_b, r_b, ns_b, d_b = self.buffer.sample(self.batch_size, self.device)
                self._update_critic(s_b, a_b, r_b, ns_b, d_b)
                if self.total_steps % self.policy_delay == 0:
                    self._update_actor(s_b)
                    self._soft_update(self.critic, self.target_critic)
                    self._soft_update(self.actor, self.target_actor)
        metrics = env.evaluate_episode()
        return metrics.get("sharpe", 0.0), metrics

    def _update_critic(self, state, action, reward, next_state, done):
        with torch.no_grad():
            next_action = self.target_actor(next_state)
            noise = torch.normal(0, self.policy_noise, size=next_action.shape, device=self.device)
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            next_action = next_action + noise
            next_action = torch.clamp(next_action, 0, 1)
            row_sum = next_action.sum(dim=1, keepdim=True)
            next_action = next_action / (row_sum + 1e-10)
            q1, q2 = self.target_critic(next_state, next_action)
            q_target = torch.min(q1, q2)
            q_target = reward + self.gamma * (1 - done) * q_target
        q1, q2 = self.critic(state, action)
        critic_loss = nn.MSELoss()(q1, q_target) + nn.MSELoss()(q2, q_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def _update_actor(self, state):
        action = self.actor(state)
        q = self.critic.q1_forward(state, action)
        entropy = -(action * torch.log(action + 1e-10)).sum(dim=1).mean()
        actor_loss = -q.mean() - self.entropy_coef * entropy
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def _soft_update(self, net, target):
        with torch.no_grad():
            for p, tp in zip(net.parameters(), target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    def run_episode(self, env, start_idx: int | None = None, end_idx: int | None = None) -> dict:
        self.actor.eval()
        state = env.reset(start_idx=start_idx, end_idx=end_idx)
        done = False
        while not done:
            with torch.no_grad():
                weights = self.actor(torch.from_numpy(state).unsqueeze(0).to(self.device)).squeeze(0).cpu().numpy()
            next_state, reward, done, info = env.step(weights)
            state = next_state
        return env.evaluate_episode(env.compute_benchmarks())

    def state_dict(self) -> dict:
        return {
            "actor": self.actor.state_dict(),
            "target_actor": self.target_actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.actor.load_state_dict(state_dict["actor"])
        self.target_actor.load_state_dict(state_dict["target_actor"])
        self.critic.load_state_dict(state_dict["critic"])
        self.target_critic.load_state_dict(state_dict["target_critic"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state_dict["critic_optimizer"])
