"""
evaluate.py  —  SentryNet Phase 2
==================================
Evaluate trained checkpoints and export full experiment CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from border_env import BorderEnv
from mappo_trainer import MAPPOTrainer


DROP_RATES: List[float] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]


def find_best_checkpoint(run_name: str) -> Path:
    """Select checkpoint with highest step number for a run."""
    run_dir = Path("checkpoints") / run_name
    assert run_dir.exists(), f"Checkpoint directory not found: {run_dir}"
    ckpts = sorted(run_dir.glob("step_*.pt"))
    if ckpts:
        return ckpts[-1]
    final = run_dir / "final.pt"
    assert final.exists(), f"No checkpoints found in {run_dir}"
    return final


def mean_trust_from_info(info: Dict[str, object]) -> float:
    """Compute mean trust scalar from nested trust lists."""
    trust = info.get("trust_scores", [])
    flat: List[float] = []
    if isinstance(trust, list):
        for row in trust:
            if isinstance(row, list):
                flat.extend([float(x) for x in row])
    return float(np.mean(flat)) if flat else 0.0


def evaluate_condition(
    system: str,
    seed: int,
    drop_rate: float,
    n_episodes: int,
) -> Dict[str, float]:
    """Evaluate one (system, seed, drop_rate) condition."""
    use_trust = system == "C"

    env = BorderEnv(
        use_pybullet=False,
        domain_rand=False,
        p_drop=drop_rate,
        p_spoof=0.0,
        use_trust=use_trust,
        seed=seed,
    )

    trainer = MAPPOTrainer(
        env=env,
        config={
            "use_wandb": False,
            "total_steps": 1,
            "run_name": f"system_{system}_seed{seed}",
            "seed": seed,
        },
    )

    ckpt = find_best_checkpoint(f"system_{system}_seed{seed}")
    trainer.load_checkpoint(str(ckpt))

    captures = 0
    steps_list: List[int] = []
    rewards_list: List[float] = []
    trust_list: List[float] = []
    battery_list: List[float] = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_reward = 0.0
        ep_steps = 0
        final_info: Dict[str, object] = {}

        while not done:
            actions = {}
            for i in range(3):
                a, _ = trainer.policy.get_action(obs[f"drone_{i}"], deterministic=True)
                actions[f"drone_{i}"] = a
            actions["sensor_0"] = 1 if float(obs["sensor_0"][0]) > 0.5 else 0

            obs, rew, term, trunc, info = env.step(actions)
            ep_reward += float(np.mean([rew[f"drone_{i}"] for i in range(3)]))
            ep_steps += 1
            done = any(term[f"drone_{i}"] or trunc[f"drone_{i}"] for i in range(3))
            final_info = info.get("drone_0", {})

        captures += int(bool(final_info.get("captured", False)))
        steps_list.append(ep_steps)
        rewards_list.append(ep_reward)
        trust_list.append(mean_trust_from_info(final_info))

        drone_pos = final_info.get("drone_pos")
        if isinstance(drone_pos, np.ndarray):
            # Battery is stored in env directly; this keeps requested column present.
            battery_list.append(float(np.mean(env.battery)))
        else:
            battery_list.append(float(np.mean(env.battery)))

    env.close()

    return {
        "system": system,
        "seed": seed,
        "drop_rate": drop_rate,
        "capture_rate": captures / n_episodes,
        "mean_steps": float(np.mean(steps_list)) if steps_list else 0.0,
        "mean_reward": float(np.mean(rewards_list)) if rewards_list else 0.0,
        "mean_trust": float(np.mean(trust_list)) if trust_list else 0.0,
        "mean_battery": float(np.mean(battery_list)) if battery_list else 0.0,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI args for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate SentryNet checkpoints")
    parser.add_argument("--system", choices=["A", "B", "C", "all"], default="all")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--fast", action="store_true", help="Run 20 episodes per condition")
    return parser.parse_args()


def main() -> None:
    """Run full condition sweep and save results CSV."""
    args = parse_args()
    systems = ["A", "B", "C"] if args.system == "all" else [args.system]
    episodes = 20 if args.fast else int(args.episodes)

    rows: List[Dict[str, float]] = []
    for system in systems:
        for seed in args.seeds:
            for drop in DROP_RATES:
                print(f"Evaluating system={system} seed={seed} drop={drop:.1f} ...")
                rows.append(evaluate_condition(system=system, seed=int(seed), drop_rate=float(drop), n_episodes=episodes))

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    out_csv = results_dir / "full_experiment.csv"
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Saved results to {out_csv}")


if __name__ == "__main__":
    main()
