"""
rollout_buffer.py  —  SentryNet Phase 2
========================================
Rollout storage and GAE computation for MAPPO.
"""

from __future__ import annotations

from typing import Dict, Generator, Tuple

import numpy as np
import torch


class RolloutBuffer:
    """Stores transitions for all drone agents and computes GAE."""

    def __init__(
        self,
        n_steps: int = 2048,
        n_drones: int = 3,
        obs_dim: int = 20,
        act_dim: int = 3,
        gamma: float = 0.99,
        lam: float = 0.95,
        device: str = "cpu",
    ) -> None:
        self.n_steps = n_steps
        self.n_drones = n_drones
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.lam = lam
        self.device = device
        self.ptr = 0

        self.obs = np.zeros((n_steps, n_drones, obs_dim), dtype=np.float32)
        self.actions = np.zeros((n_steps, n_drones, act_dim), dtype=np.float32)
        self.rewards = np.zeros((n_steps, n_drones), dtype=np.float32)
        self.values = np.zeros((n_steps, n_drones), dtype=np.float32)
        self.log_probs = np.zeros((n_steps, n_drones), dtype=np.float32)
        self.dones = np.zeros((n_steps, n_drones), dtype=bool)

        self.advantages = np.zeros((n_steps, n_drones), dtype=np.float32)
        self.returns = np.zeros((n_steps, n_drones), dtype=np.float32)

    def add(
        self,
        step: int,
        obs_dict: Dict[str, np.ndarray],
        actions_dict: Dict[str, np.ndarray],
        rewards_dict: Dict[str, float],
        values_dict: Dict[str, float],
        log_probs_dict: Dict[str, float],
        dones_dict: Dict[str, bool],
    ) -> None:
        """Store one timestep using environment dict outputs."""
        assert 0 <= step < self.n_steps, f"Step {step} out of bounds [0, {self.n_steps})"
        drone_keys = [f"drone_{i}" for i in range(self.n_drones)]

        obs_arr = np.stack([obs_dict[k] for k in drone_keys]).astype(np.float32)  # (n_drones, obs_dim)
        act_arr = np.stack([actions_dict[k] for k in drone_keys]).astype(np.float32)  # (n_drones, act_dim)
        rew_arr = np.asarray([rewards_dict[k] for k in drone_keys], dtype=np.float32)  # (n_drones,)
        val_arr = np.asarray([values_dict[k] for k in drone_keys], dtype=np.float32)  # (n_drones,)
        lp_arr = np.asarray([log_probs_dict[k] for k in drone_keys], dtype=np.float32)  # (n_drones,)
        done_arr = np.asarray([dones_dict[k] for k in drone_keys], dtype=bool)  # (n_drones,)

        assert obs_arr.shape == (self.n_drones, self.obs_dim), f"obs shape mismatch: {obs_arr.shape}"
        assert act_arr.shape == (self.n_drones, self.act_dim), f"actions shape mismatch: {act_arr.shape}"

        self.obs[step] = obs_arr
        self.actions[step] = act_arr
        self.rewards[step] = rew_arr
        self.values[step] = val_arr
        self.log_probs[step] = lp_arr
        self.dones[step] = done_arr

        self.ptr = max(self.ptr, step + 1)

    def compute_gae(self, last_values: np.ndarray) -> None:
        """Compute GAE advantages and returns, then normalize advantages."""
        assert self.ptr > 0, "Cannot compute GAE on empty buffer"
        assert last_values.shape == (self.n_drones,), f"Expected last_values shape ({self.n_drones},), got {last_values.shape}"

        gae = np.zeros((self.n_drones,), dtype=np.float32)

        for t in reversed(range(self.ptr)):
            if t == self.ptr - 1:
                next_values = last_values
            else:
                next_values = self.values[t + 1]

            not_done = 1.0 - self.dones[t].astype(np.float32)  # (n_drones,)
            delta_t = self.rewards[t] + self.gamma * not_done * next_values - self.values[t]  # (n_drones,)
            gae = delta_t + self.gamma * self.lam * not_done * gae
            self.advantages[t] = gae

        self.returns[:self.ptr] = self.advantages[:self.ptr] + self.values[:self.ptr]

        adv = self.advantages[:self.ptr]
        adv_mean = adv.mean()
        adv_std = adv.std() + 1e-8
        self.advantages[:self.ptr] = (adv - adv_mean) / adv_std

    def get_batches(
        self,
        batch_size: int = 256,
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None, None]:
        """Yield random mini-batches for PPO update."""
        total = self.ptr * self.n_drones
        assert total > 0, "Buffer is empty"

        flat_obs = self.obs[:self.ptr].reshape(total, self.obs_dim)  # (total, obs_dim)
        flat_actions = self.actions[:self.ptr].reshape(total, self.act_dim)  # (total, act_dim)
        flat_log_probs = self.log_probs[:self.ptr].reshape(total)  # (total,)
        flat_advantages = self.advantages[:self.ptr].reshape(total)  # (total,)
        flat_returns = self.returns[:self.ptr].reshape(total)  # (total,)

        idx = np.random.permutation(total)

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            b = idx[start:end]
            yield (
                torch.as_tensor(flat_obs[b], dtype=torch.float32, device=self.device),
                torch.as_tensor(flat_actions[b], dtype=torch.float32, device=self.device),
                torch.as_tensor(flat_log_probs[b], dtype=torch.float32, device=self.device),
                torch.as_tensor(flat_advantages[b], dtype=torch.float32, device=self.device),
                torch.as_tensor(flat_returns[b], dtype=torch.float32, device=self.device),
            )

    def reset(self) -> None:
        """Reset buffer content and write pointer."""
        self.obs.fill(0.0)
        self.actions.fill(0.0)
        self.rewards.fill(0.0)
        self.values.fill(0.0)
        self.log_probs.fill(0.0)
        self.dones.fill(False)
        self.advantages.fill(0.0)
        self.returns.fill(0.0)
        self.ptr = 0
