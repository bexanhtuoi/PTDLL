from __future__ import annotations

import numpy as np
import torch
from torch import nn, optim

from portfolio.base import BaseModel, PolicyNet, ValueNet


class PPOAgent(BaseModel):
    def __init__(
        self, lookback: int, n_assets: int, n_features: int = 10,
        lr: float = 3e-4, gamma: float = 0.99, gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2, entropy_coef: float = 0.05,
        k_epochs: int = 4, alpha_mult: float = 5.0, device: str = "cpu",
        weight_decay: float = 1e-5, actor_wd: float | None = None,
    ):
        self.lookback = lookback
        self.n_assets = n_assets
        self.n_features = n_features
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.k_epochs = k_epochs
        self.alpha_mult = alpha_mult
        self.actor_wd = actor_wd if actor_wd is not None else weight_decay

        self.policy = PolicyNet(lookback, n_assets, n_features).to(device)
        self.value = ValueNet(lookback, n_assets, n_features).to(device)
        self.optimizer = optim.Adam(
            list(self.policy.parameters()) + list(self.value.parameters()),
            lr=lr, weight_decay=weight_decay,
        )
        self.policy_opt = optim.Adam(self.policy.parameters(), lr=lr, weight_decay=self.actor_wd)
        self.value_opt = optim.Adam(self.value.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.96)

    def predict(self, state: np.ndarray) -> np.ndarray:
        self.policy.eval()
        with torch.no_grad():
            logits = self.policy(torch.from_numpy(state).unsqueeze(0).to(self.device))
            return torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    def probs(self, state_t: torch.Tensor) -> torch.Tensor:
        logits = self.policy(state_t)
        logits = torch.clamp(logits, -20, 20)
        return torch.softmax(logits, dim=1)

    def act(self, state: np.ndarray):
        state_t = torch.from_numpy(state).unsqueeze(0).to(self.device)
        probs = self.probs(state_t).squeeze(0)
        probs = torch.clamp(probs, 1e-6, 1)
        dist = torch.distributions.Dirichlet(probs * self.alpha_mult + 1)
        action = dist.rsample()
        return action.detach().cpu().numpy(), dist.log_prob(action), dist.entropy(), state_t

    def train_ep(self, env, start_idx: int | None = None, end_idx: int | None = None) -> tuple[float, dict]:
        state = env.reset(start_idx=start_idx, end_idx=end_idx)
        episode: list[tuple] = []
        done = False
        while not done:
            action, log_prob, _, state_t = self.act(state)
            next_state, reward, done, info = env.step(action)
            episode.append((state, action, reward, log_prob.item(), state_t))
            state = next_state
        self.update(episode)
        metrics = env.score_ep()
        return metrics.get("sharpe", 0.0), metrics

    def unpack_episode(self, episode: list[tuple]):
        states, actions, rewards, log_probs_old, state_ts = [], [], [], [], []
        for s, a, r, lp, st in episode:
            states.append(s); actions.append(a); rewards.append(r)
            log_probs_old.append(lp); state_ts.append(st)
        advantages, returns = self.compute_gae(rewards, state_ts)
        log_probs_old_t = torch.tensor(log_probs_old, device=self.device, dtype=torch.float32)
        return states, actions, advantages, returns, log_probs_old_t

    def surrogate_loss(self, state, action, advantage, return_, log_prob_old):
        state_t = torch.from_numpy(state).unsqueeze(0).to(self.device)
        action_t = torch.from_numpy(action).unsqueeze(0).to(self.device)
        probs = torch.clamp(self.probs(state_t), 1e-6, 1)
        dist = torch.distributions.Dirichlet(probs * self.alpha_mult + 1)
        new_log_prob = dist.log_prob(action_t.squeeze(0))
        entropy = dist.entropy()
        ratio = torch.exp(new_log_prob - log_prob_old)
        ratio = torch.clamp(ratio, 0.01, 100)
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantage
        policy_loss = -torch.min(surr1, surr2)
        value_pred = self.value(state_t).squeeze(-1)
        value_loss = 0.5 * (value_pred - return_) ** 2
        return policy_loss + 0.5 * value_loss - self.entropy_coef * entropy

    def update(self, episode: list[tuple]) -> float:
        self.policy.train()
        self.value.train()
        states, actions, advantages, returns, log_probs_old_t = self.unpack_episode(episode)
        total_loss = 0.0
        for _ in range(self.k_epochs):
            policy_loss = 0.0
            value_loss = 0.0
            for i, s in enumerate(states):
                state_t = torch.from_numpy(s).unsqueeze(0).to(self.device)
                action_t = torch.from_numpy(actions[i]).unsqueeze(0).to(self.device)
                probs = torch.clamp(self.probs(state_t), 1e-6, 1)
                dist = torch.distributions.Dirichlet(probs * self.alpha_mult + 1)
                new_log_prob = dist.log_prob(action_t.squeeze(0))
                entropy = dist.entropy()
                ratio = torch.exp(new_log_prob - log_probs_old_t[i])
                ratio = torch.clamp(ratio, 0.01, 100)
                surr1 = ratio * advantages[i]
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages[i]
                policy_loss = policy_loss + -torch.min(surr1, surr2) - self.entropy_coef * entropy
                v_pred = self.value(state_t).squeeze(-1)
                value_loss = value_loss + 0.5 * (v_pred - returns[i]) ** 2
            self.policy_opt.zero_grad()
            policy_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.policy_opt.step()
            self.value_opt.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value.parameters(), 0.5)
            self.value_opt.step()
            total_loss += (policy_loss + value_loss).item()
        self.scheduler.step()
        return total_loss / self.k_epochs

    @torch.no_grad()
    def compute_gae(self, rewards: list[float], state_ts: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        n = len(rewards)
        values = torch.zeros(n + 1, device=self.device, dtype=torch.float32)
        for i, s in enumerate(state_ts):
            values[i] = self.value(s).squeeze(-1)
        values[-1] = self.value(state_ts[-1]).squeeze(-1)
        advantages = torch.zeros(n, device=self.device, dtype=torch.float32)
        gae = 0.0
        for t in reversed(range(n)):
            delta = rewards[t] + self.gamma * values[t + 1] - values[t]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages[t] = gae
        returns = advantages + values[:n]
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def state_dict(self) -> dict:
        return {
            "policy": self.policy.state_dict(),
            "value": self.value.state_dict(),
            "policy_opt": self.policy_opt.state_dict(),
            "value_opt": self.value_opt.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.policy.load_state_dict(state_dict["policy"])
        self.value.load_state_dict(state_dict["value"])
        self.policy_opt.load_state_dict(state_dict["policy_opt"])
        self.value_opt.load_state_dict(state_dict["value_opt"])
        self.scheduler.load_state_dict(state_dict["scheduler"])
