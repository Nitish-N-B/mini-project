"""
mappo_trainer.py  —  SentryNet Phase 2
========================================
Full MAPPO trainer for shared-policy multi-drone learning.
"""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import torch
import torch.nn as nn

from border_env import BorderEnv
from networks import PolicyNet, ValueNet
from rollout_buffer import RolloutBuffer


class MAPPOTrainer:
    """Full MAPPO training loop with shared actor and centralized critic."""

    DEFAULT_CONFIG: Dict[str, Any] = {
        "lr": 3e-4,
        "gamma": 0.99,
        "lam": 0.95,
        "clip_eps": 0.2,
        "value_coef": 0.5,
        "entropy_coef": 0.01,
        "max_grad_norm": 10.0,
        "n_steps": 2048,
        "batch_size": 256,
        "n_epochs": 4,
        "total_steps": 1_000_000,
        "save_every": 50_000,
        "eval_every": 50_000,
        "use_wandb": True,
        "run_name": "system_A",
        "checkpoint_dir": "checkpoints",
        "seed": 0,
    }

    def __init__(self, env: BorderEnv, config: Optional[Dict[str, Any]] = None) -> None:
        self.env = env
        self.config = copy.deepcopy(self.DEFAULT_CONFIG)
        if config is not None:
            self.config.update(config)

        self.seed = int(self.config["seed"])
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.policy = PolicyNet(obs_dim=20, act_dim=3).to(self.device)
        self.value = ValueNet(obs_dim=60).to(self.device)

        lr = float(self.config["lr"])
        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.value_opt = torch.optim.Adam(self.value.parameters(), lr=lr)

        self.buffer = RolloutBuffer(
            n_steps=int(self.config["n_steps"]),
            n_drones=3,
            obs_dim=20,
            act_dim=3,
            gamma=float(self.config["gamma"]),
            lam=float(self.config["lam"]),
            device=self.device,
        )

        self.total_env_steps = 0

    @staticmethod
    def _drone_keys() -> List[str]:
        return ["drone_0", "drone_1", "drone_2"]

    def _obs_all_tensor(self, obs_dict: Dict[str, np.ndarray]) -> torch.Tensor:
        drone_obs = np.concatenate([obs_dict[k] for k in self._drone_keys()], axis=0).astype(np.float32)
        obs_all = torch.as_tensor(drone_obs, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, 60)
        assert obs_all.shape == (1, 60), f"Expected (1, 60), got {tuple(obs_all.shape)}"
        return obs_all

    @staticmethod
    def _mean_trust_from_info(info: Dict[str, Any]) -> float:
        trust = info.get("trust_scores", [])
        if not trust:
            return 0.0
        flat: List[float] = []
        for row in trust:
            if isinstance(row, (list, tuple)):
                flat.extend([float(x) for x in row])
        return float(np.mean(flat)) if flat else 0.0

    def collect_rollout(self) -> Dict[str, float]:
        """Collect one rollout and compute GAE-ready buffer targets."""
        self.buffer.reset()
        n_steps = int(self.config["n_steps"])

        obs, _ = self.env.reset(seed=self.seed + self.total_env_steps)

        rollout_rewards: List[float] = []
        rollout_captures = 0
        rollout_episode_count = 0
        steps_to_capture: List[int] = []
        trust_values: List[float] = []

        for step in range(n_steps):
            actions_dict: Dict[str, np.ndarray] = {}
            log_probs_dict: Dict[str, float] = {}
            values_dict: Dict[str, float] = {}

            with torch.no_grad():
                for k in self._drone_keys():
                    action, log_prob = self.policy.get_action(obs[k], deterministic=False)
                    actions_dict[k] = action
                    log_probs_dict[k] = float(log_prob)

                obs_all = self._obs_all_tensor(obs)
                v = float(self.value(obs_all).squeeze(0).squeeze(0).cpu().item())
                for k in self._drone_keys():
                    values_dict[k] = v

            actions_env: Dict[str, Any] = {k: actions_dict[k] for k in self._drone_keys()}
            actions_env["sensor_0"] = 1 if float(obs["sensor_0"][0]) > 0.5 else 0

            next_obs, rewards, term, trunc, info_all = self.env.step(actions_env)
            dones_dict = {k: bool(term[k] or trunc[k]) for k in self._drone_keys()}

            self.buffer.add(
                step=step,
                obs_dict=obs,
                actions_dict=actions_dict,
                rewards_dict={k: float(rewards[k]) for k in self._drone_keys()},
                values_dict=values_dict,
                log_probs_dict=log_probs_dict,
                dones_dict=dones_dict,
            )

            rollout_rewards.append(float(np.mean([rewards[k] for k in self._drone_keys()])))

            if any(dones_dict.values()):
                rollout_episode_count += 1
                info0 = info_all.get("drone_0", {})
                trust_values.append(self._mean_trust_from_info(info0))
                if bool(info0.get("captured", False)):
                    rollout_captures += 1
                    steps_to_capture.append(int(info0.get("step", 0)))
                next_obs, _ = self.env.reset()

            obs = next_obs

        with torch.no_grad():
            obs_all_last = self._obs_all_tensor(obs)
            last_v = float(self.value(obs_all_last).squeeze(0).squeeze(0).cpu().item())
            last_values = np.array([last_v, last_v, last_v], dtype=np.float32)

        self.buffer.compute_gae(last_values=last_values)

        capture_rate = float(rollout_captures / rollout_episode_count) if rollout_episode_count > 0 else 0.0
        mean_steps_to_capture = float(np.mean(steps_to_capture)) if steps_to_capture else float(500)
        mean_trust = float(np.mean(trust_values)) if trust_values else 0.0

        return {
            "mean_reward": float(np.mean(rollout_rewards)) if rollout_rewards else 0.0,
            "capture_rate": capture_rate,
            "mean_trust": mean_trust,
            "steps_to_capture": mean_steps_to_capture,
        }

    def update(self) -> Dict[str, float]:
        """Run PPO updates for actor and critic using collected rollout."""
        n_epochs = int(self.config["n_epochs"])
        batch_size = int(self.config["batch_size"])
        clip_eps = float(self.config["clip_eps"])
        entropy_coef = float(self.config["entropy_coef"])
        value_coef = float(self.config["value_coef"])
        max_grad_norm = float(self.config["max_grad_norm"])

        policy_losses: List[float] = []
        value_losses: List[float] = []
        entropy_values: List[float] = []
        approx_kls: List[float] = []

        total = self.buffer.ptr * self.buffer.n_drones
        obs_all_t = self.buffer.obs[: self.buffer.ptr].reshape(self.buffer.ptr, -1)  # (T, 60)
        obs_all_flat = np.repeat(obs_all_t, repeats=self.buffer.n_drones, axis=0)  # (T*3, 60)
        flat_returns = self.buffer.returns[: self.buffer.ptr].reshape(total)

        for _ in range(n_epochs):
            idx = np.random.permutation(total)

            for start in range(0, total, batch_size):
                end = min(start + batch_size, total)
                b = idx[start:end]

                obs_b = torch.as_tensor(
                    self.buffer.obs[: self.buffer.ptr].reshape(total, self.buffer.obs_dim)[b],
                    dtype=torch.float32,
                    device=self.device,
                )  # (batch, 20)
                act_b = torch.as_tensor(
                    self.buffer.actions[: self.buffer.ptr].reshape(total, self.buffer.act_dim)[b],
                    dtype=torch.float32,
                    device=self.device,
                )  # (batch, 3)
                old_log_b = torch.as_tensor(
                    self.buffer.log_probs[: self.buffer.ptr].reshape(total)[b],
                    dtype=torch.float32,
                    device=self.device,
                )  # (batch,)
                adv_b = torch.as_tensor(
                    self.buffer.advantages[: self.buffer.ptr].reshape(total)[b],
                    dtype=torch.float32,
                    device=self.device,
                )  # (batch,)
                ret_b = torch.as_tensor(flat_returns[b], dtype=torch.float32, device=self.device)  # (batch,)
                obs_all_b = torch.as_tensor(obs_all_flat[b], dtype=torch.float32, device=self.device)  # (batch, 60)

                new_log_b, entropy_b = self.policy.evaluate_actions(obs_b, act_b)
                ratio = torch.exp(new_log_b - old_log_b)
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()

                value_pred = self.value(obs_all_b).squeeze(-1)
                value_loss = 0.5 * torch.mean((value_pred - ret_b) ** 2)

                entropy_term = entropy_b.mean()
                total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_term

                self.policy_opt.zero_grad()
                self.value_opt.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_grad_norm)
                nn.utils.clip_grad_norm_(self.value.parameters(), max_grad_norm)
                self.policy_opt.step()
                self.value_opt.step()

                with torch.no_grad():
                    approx_kl = torch.mean(old_log_b - new_log_b).item()

                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropy_values.append(float(entropy_term.item()))
                approx_kls.append(float(approx_kl))

        return {
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "entropy": float(np.mean(entropy_values)) if entropy_values else 0.0,
            "approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
        }

    def evaluate(self, n_episodes: int = 50, p_drop_eval: float = 0.0) -> Dict[str, float]:
        """Evaluate policy deterministically on a separate environment."""
        eval_env = BorderEnv(
            use_pybullet=False,
            domain_rand=False,
            p_drop=p_drop_eval,
            p_spoof=0.0,
            use_trust=getattr(self.env, "use_trust", True),
            seed=self.seed,
        )

        captures = 0
        ep_rewards: List[float] = []
        ep_steps: List[int] = []
        trust_values: List[float] = []

        for ep in range(n_episodes):
            obs, _ = eval_env.reset(seed=self.seed + ep)
            done = False
            ep_reward = 0.0
            steps = 0

            while not done:
                actions_env: Dict[str, Any] = {}
                for k in self._drone_keys():
                    a, _ = self.policy.get_action(obs[k], deterministic=True)
                    actions_env[k] = a
                actions_env["sensor_0"] = 1 if float(obs["sensor_0"][0]) > 0.5 else 0

                obs, rewards, term, trunc, info = eval_env.step(actions_env)
                ep_reward += float(np.mean([rewards[k] for k in self._drone_keys()]))
                steps += 1
                done = any(term[k] or trunc[k] for k in self._drone_keys())

            info0 = info.get("drone_0", {})
            captures += int(bool(info0.get("captured", False)))
            trust_values.append(self._mean_trust_from_info(info0))
            ep_rewards.append(ep_reward)
            ep_steps.append(steps)

        eval_env.close()

        return {
            "capture_rate": float(captures / n_episodes),
            "mean_steps": float(np.mean(ep_steps)) if ep_steps else 0.0,
            "mean_reward": float(np.mean(ep_rewards)) if ep_rewards else 0.0,
            "mean_trust": float(np.mean(trust_values)) if trust_values else 0.0,
        }

    def save_checkpoint(self, step: int) -> None:
        """Persist training state to checkpoints/{run_name}/step_{step}.pt."""
        root = Path(str(self.config["checkpoint_dir"])) / str(self.config["run_name"])
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"step_{step}.pt"
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "value_state_dict": self.value.state_dict(),
                "policy_opt_state_dict": self.policy_opt.state_dict(),
                "value_opt_state_dict": self.value_opt.state_dict(),
                "step": step,
                "config": self.config,
                "seed": self.seed,
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        """Load model and optimizer state from checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy_state_dict"])
        self.value.load_state_dict(ckpt["value_state_dict"])
        self.policy_opt.load_state_dict(ckpt["policy_opt_state_dict"])
        self.value_opt.load_state_dict(ckpt["value_opt_state_dict"])
        self.total_env_steps = int(ckpt.get("step", 0))

    def train(self) -> None:
        """Run MAPPO training loop until configured total_steps."""
        use_wandb = bool(self.config["use_wandb"])
        wandb = None
        if use_wandb:
            try:
                import wandb as _wandb

                wandb = _wandb
                wandb.init(
                    project="sentrinet",
                    name=str(self.config["run_name"]),
                    config=self.config,
                    reinit=True,
                )
            except Exception:
                wandb = None

        n_steps = int(self.config["n_steps"])
        total_steps = int(self.config["total_steps"])
        save_every = int(self.config["save_every"])
        eval_every = int(self.config["eval_every"])

        while self.total_env_steps < total_steps:
            rollout_stats = self.collect_rollout()
            loss_stats = self.update()
            self.total_env_steps += n_steps

            metrics = {
                "train/reward": rollout_stats["mean_reward"],
                "train/capture_rate": rollout_stats["capture_rate"],
                "train/policy_loss": loss_stats["policy_loss"],
                "train/value_loss": loss_stats["value_loss"],
                "train/entropy": loss_stats["entropy"],
                "train/approx_kl": loss_stats["approx_kl"],
                "env/trust_mean": rollout_stats["mean_trust"],
                "env/drop_rate": float(getattr(self.env.channel, "p_drop", 0.0)),
                "env/battery_mean": float(np.mean(getattr(self.env, "battery", np.zeros((3,), dtype=np.float32)))),
            }

            if self.total_env_steps % eval_every == 0:
                eval_stats = self.evaluate(n_episodes=50, p_drop_eval=float(getattr(self.env.channel, "p_drop", 0.0)))
                metrics.update(
                    {
                        "eval/capture_rate": eval_stats["capture_rate"],
                        "eval/mean_steps": eval_stats["mean_steps"],
                        "eval/mean_reward": eval_stats["mean_reward"],
                        "eval/mean_trust": eval_stats["mean_trust"],
                    }
                )

            if wandb is not None:
                wandb.log(metrics, step=self.total_env_steps)

            print(
                f"[Step {self.total_env_steps:>8,}] "
                f"Capture={rollout_stats['capture_rate']*100:5.1f}% | "
                f"Reward={rollout_stats['mean_reward']:7.2f} | "
                f"PolicyLoss={loss_stats['policy_loss']:8.4f} | "
                f"ValueLoss={loss_stats['value_loss']:8.4f}"
            )

            if self.total_env_steps % save_every == 0:
                self.save_checkpoint(step=self.total_env_steps)

        self.save_checkpoint(step=self.total_env_steps)

        if wandb is not None:
            wandb.finish()

        print(
            f"Training completed for {self.config['run_name']} at {self.total_env_steps:,} steps. "
            f"Final checkpoint saved."
        )
