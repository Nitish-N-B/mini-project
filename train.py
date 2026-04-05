"""
train.py  —  SentryNet Phase 2
===============================
Main training entrypoint for Systems A, B, and C.
"""

from __future__ import annotations

import argparse
import random
from typing import Dict, List

import numpy as np
import torch

from border_env import BorderEnv
from mappo_trainer import MAPPOTrainer


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_env(system: str, seed: int) -> BorderEnv:
    """Create BorderEnv configured for the selected system."""
    if system == "A":
        return BorderEnv(
            use_pybullet=False,
            domain_rand=True,
            p_drop=0.0,
            p_spoof=0.0,
            use_trust=False,
            seed=seed,
        )
    if system == "B":
        return BorderEnv(
            use_pybullet=False,
            domain_rand=True,
            p_drop=0.2,
            p_spoof=0.0,
            use_trust=False,
            seed=seed,
        )
    if system == "C":
        return BorderEnv(
            use_pybullet=False,
            domain_rand=True,
            p_drop=0.2,
            p_spoof=0.0,
            use_trust=True,
            seed=seed,
        )
    raise ValueError(f"Unknown system '{system}'")


def train_one(system: str, seed: int, total_steps: int, use_wandb: bool) -> None:
    """Train one (system, seed) configuration."""
    set_seed(seed)
    env = build_env(system=system, seed=seed)
    run_name = f"system_{system}_seed{seed}"

    config: Dict[str, object] = {
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
        "total_steps": int(total_steps),
        "save_every": 50_000,
        "eval_every": 50_000,
        "use_wandb": bool(use_wandb),
        "run_name": run_name,
        "checkpoint_dir": "checkpoints",
        "seed": int(seed),
    }

    print(f"\n=== Training {run_name} for {total_steps:,} steps ===")
    trainer = MAPPOTrainer(env=env, config=config)
    trainer.train()
    env.close()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train SentryNet MAPPO systems")
    parser.add_argument("--system", choices=["A", "B", "C", "all"], default="all")
    parser.add_argument("--seed", type=int, default=None, help="Single seed; default runs seeds 0/1/2")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Multiple seeds, e.g. --seeds 0 1 2")
    parser.add_argument("--steps", type=int, default=1_000_000, help="Total steps per run")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--fast", action="store_true", help="Smoke test mode (10,000 steps)")
    return parser.parse_args()


def main() -> None:
    """Train one or more systems based on CLI options."""
    args = parse_args()

    steps = 10_000 if args.fast else int(args.steps)
    use_wandb = not bool(args.no_wandb)

    systems: List[str] = ["A", "B", "C"] if args.system == "all" else [args.system]
    if args.seeds is not None:
        seeds: List[int] = [int(s) for s in args.seeds]
    elif args.seed is not None:
        seeds = [int(args.seed)]
    else:
        seeds = [0, 1, 2]

    for system in systems:
        for seed in seeds:
            train_one(system=system, seed=seed, total_steps=steps, use_wandb=use_wandb)


if __name__ == "__main__":
    main()
