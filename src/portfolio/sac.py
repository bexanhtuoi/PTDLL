from __future__ import annotations

import numpy as np
import torch
from torch import nn, optim

from portfolio.base import BaseModel, PolicyNet, ReplayBuffer, TwinQNet


class SACAgent(BaseModel):
    def __init__(
        self, lookback: int, n_assets: int, n_features: int = 10,
        lr: float = 3e-4, gamma: float = 0.99, tau: float = 0.005,
        alpha_mult: float = 15.0, target_entropy: float | None = None,
        batch_size: int = 64, replay_capacity: int = 100000,
        device: str = "cpu",
    ):
        self.lookback = lookback
        self.n_assets = n_assets
        self.n_features = n_features
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha_mult = alpha_mult
        self.batch_size = batch_size

        self.actor = PolicyNet(lookback, n_assets, n_features).to(device)
        self.critic = TwinQNet(lookback, n_assets, n_features).to(device)
        self.target_critic = TwinQNet(lookback, n_assets, n_features).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        for p in self.target_critic.parameters():
            p.requires_grad = False

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr, weight_decay=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr, weight_decay=1e-5)

        target_entropy = target_entropy if target_entropy is not None else -float(n_assets)
        self.log_alpha = torch.tensor(np.log(0.1), dtype=torch.float32, device=device, requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        self.target_entropy = target_entropy

        self.buffer = ReplayBuffer(replay_capacity)

    def predict(self, state: np.ndarray) -> np.ndarray:
        self.actor.eval()
        with torch.no_grad():
            logits = self.actor(torch.from_numpy(state).unsqueeze(0).to(self.device))
            return torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    def sample_action(self, state_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.actor(state_t)
        logits = torch.clamp(logits, -20, 20)
        probs = torch.softmax(logits, dim=1)
        probs = torch.clamp(probs, 1e-6, 1)
        dist = torch.distributions.Dirichlet(probs * self.alpha_mult + 1)
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def get_alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def train_ep(self, env, start_idx: int | None = None, end_idx: int | None = None) -> tuple[float, dict]:
        state = env.reset(start_idx=start_idx, end_idx=end_idx)
        done = False
        while not done:
            self.actor.eval()
            with torch.no_grad():
                state_t = torch.from_numpy(state).unsqueeze(0).to(self.device)
                action, _ = self.sample_action(state_t)
                action = action.squeeze(0).cpu().numpy()
            next_state, reward, done, info = env.step(action)
            self.buffer.push(state.copy(), action.copy(), reward, next_state.copy(), done)
            state = next_state
            if len(self.buffer) >= self.batch_size:
                s_b, a_b, r_b, ns_b, d_b = self.buffer.sample(self.batch_size, self.device)
                self.update_critic(s_b, a_b, r_b, ns_b, d_b)
                self.update_actor_alpha(s_b)
                self.soft_update()
        metrics = env.score_ep()
        return metrics.get("sharpe", 0.0), metrics

    def update_critic(self, state, action, reward, next_state, done):
        with torch.no_grad():
            next_action, next_log_prob = self.sample_action(next_state)
            q1_target, q2_target = self.target_critic(next_state, next_action)
            q_target = torch.min(q1_target, q2_target)
            alpha = self.get_alpha()
            q_target = reward + self.gamma * (1 - done) * (q_target - alpha * next_log_prob.unsqueeze(1))
        q1, q2 = self.critic(state, action)
        critic_loss = nn.MSELoss()(q1, q_target) + nn.MSELoss()(q2, q_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

    def update_actor_alpha(self, state):
        action, log_prob = self.sample_action(state)
        alpha = self.get_alpha()
        q1, q2 = self.critic(state, action)
        q = torch.min(q1, q2)
        actor_loss = (alpha * log_prob.unsqueeze(1) - q).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

    def soft_update(self):
        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    def state_dict(self) -> dict:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu().item(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.actor.load_state_dict(state_dict["actor"])
        self.critic.load_state_dict(state_dict["critic"])
        self.target_critic.load_state_dict(state_dict["target_critic"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state_dict["critic_optimizer"])
        self.log_alpha.data.fill_(state_dict["log_alpha"])
        self.alpha_optimizer.load_state_dict(state_dict["alpha_optimizer"])
