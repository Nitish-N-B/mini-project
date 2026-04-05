"""
networks.py  —  SentryNet Phase 2
===================================
Policy and value neural networks for MAPPO training.
"""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


class PolicyNet(nn.Module):
    """Actor network mapping 20-dim drone observations to 3-dim actions."""

    def __init__(self, obs_dim: int = 20, act_dim: int = 3, hidden_dim: int = 128) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.mean_head = nn.Linear(hidden_dim, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        self._init_weights()

    def _init_weights(self) -> None:
        """Orthogonal initialization for stable PPO optimization."""
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2.0))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.zeros_(self.mean_head.bias)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return action distribution parameters for observations.

        Args:
            obs: Tensor of shape (batch, 20).

        Returns:
            mean: Tensor of shape (batch, 3), tanh-bounded to [-1, 1].
            std: Tensor of shape (batch, 3).
        """
        assert obs.ndim == 2 and obs.shape[-1] == self.obs_dim, f"Expected (batch, {self.obs_dim}), got {tuple(obs.shape)}"
        feat = self.net(obs)
        mean = torch.tanh(self.mean_head(feat))
        log_std = torch.clamp(self.log_std, -2.0, 2.0).unsqueeze(0).expand_as(mean)
        std = torch.exp(log_std)
        return mean, std

    def distribution(self, obs: torch.Tensor) -> Normal:
        """Build Normal action distribution for a batch of observations."""
        mean, std = self.forward(obs)
        return Normal(mean, std)

    def sample_action_tensor(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample actions for tensor observations.

        Args:
            obs: Tensor shape (batch, 20).
            deterministic: If True, return distribution mean.

        Returns:
            actions: Tensor shape (batch, 3), clipped to [-1, 1].
            log_probs: Tensor shape (batch,).
            entropy: Tensor shape (batch,).
        """
        dist = self.distribution(obs)
        if deterministic:
            actions = dist.mean
        else:
            actions = dist.rsample()
        actions = torch.clamp(actions, -1.0, 1.0)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return actions, log_probs, entropy

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute log-probabilities and entropy of provided actions.

        Args:
            obs: Tensor shape (batch, 20).
            actions: Tensor shape (batch, 3).
        """
        assert actions.ndim == 2 and actions.shape[-1] == self.act_dim, f"Expected (batch, {self.act_dim}), got {tuple(actions.shape)}"
        dist = self.distribution(obs)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_probs, entropy

    def get_action(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, float]:
        """Get one action for one observation.

        Args:
            obs: Numpy array or tensor shape (20,).
            deterministic: If True, returns mean action.

        Returns:
            action: np.ndarray shape (3,).
            log_prob: float.
        """
        if isinstance(obs, np.ndarray):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        elif isinstance(obs, torch.Tensor):
            obs_tensor = obs.float().unsqueeze(0) if obs.ndim == 1 else obs.float()
        else:
            raise TypeError(f"Unsupported observation type: {type(obs)}")

        assert obs_tensor.shape == (1, self.obs_dim), f"Expected (1, {self.obs_dim}), got {tuple(obs_tensor.shape)}"
        device = next(self.parameters()).device
        obs_tensor = obs_tensor.to(device)

        with torch.no_grad():
            action_t, log_prob_t, _ = self.sample_action_tensor(obs_tensor, deterministic=deterministic)

        action = action_t.squeeze(0).cpu().numpy().astype(np.float32)
        log_prob = float(log_prob_t.squeeze(0).cpu().item())
        return action, log_prob


class ValueNet(nn.Module):
    """Centralized critic mapping concatenated 60-dim observations to value."""

    def __init__(self, obs_dim: int = 60, hidden_dim: int = 128) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.value_head = nn.Linear(hidden_dim, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        """Orthogonal initialization for critic stability."""
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2.0))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, obs_all: torch.Tensor) -> torch.Tensor:
        """Return scalar values for centralized observations.

        Args:
            obs_all: Tensor shape (batch, 60).

        Returns:
            Tensor shape (batch, 1).
        """
        assert obs_all.ndim == 2 and obs_all.shape[-1] == self.obs_dim, f"Expected (batch, {self.obs_dim}), got {tuple(obs_all.shape)}"
        feat = self.net(obs_all)
        return self.value_head(feat)
