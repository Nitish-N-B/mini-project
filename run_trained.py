"""
run_trained.py — Run a trained SentryNet policy
=================================================
Loads a MAPPO checkpoint and runs episodes with the trained policy.

Modes:
  --mode stats   : Run N episodes headless, print capture statistics (default)
  --mode visual  : Run with PyBullet 3D visualization

Usage:
  python run_trained.py --checkpoint checkpoints/system_C/final.pt
  python run_trained.py --checkpoint checkpoints/system_C/final.pt --mode visual
  python run_trained.py --checkpoint checkpoints/system_C/final.pt --episodes 100 --p_drop 0.3
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import time

from networks import PolicyNet
from border_env import BorderEnv

N_DRONES = 3


def resolve_checkpoint_path(checkpoint: str) -> str:
    """Resolve checkpoint input to a concrete .pt file path.

    Supports:
      - direct file path to .pt
      - directory path containing final.pt or step_*.pt
    """
    p = Path(checkpoint)
    if p.is_file():
        return str(p)

    if p.is_dir():
        final_ckpt = p / "final.pt"
        if final_ckpt.exists():
            return str(final_ckpt)
        step_ckpts = sorted(p.glob("step_*.pt"))
        if step_ckpts:
            return str(step_ckpts[-1])

    raise FileNotFoundError(
        f"Checkpoint not found: {checkpoint}. "
        "Pass a .pt file or a run directory containing final.pt / step_*.pt"
    )


def load_policy(checkpoint_path: str, device: str = "cpu") -> PolicyNet:
    """Load a trained PolicyNet from checkpoint."""
    checkpoint_path = resolve_checkpoint_path(checkpoint_path)
    policy = PolicyNet().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()
    step = ckpt.get("total_steps", ckpt.get("step", "?"))
    step_str = f"{step:,}" if isinstance(step, int) else str(step)
    print(f"[Loaded] {checkpoint_path} (trained for {step_str} steps)")
    return policy


def get_drone_actions(policy, obs_dict, device="cpu", deterministic=True):
    """Get actions for all drones from the trained policy."""
    action_dict = {}
    for i in range(N_DRONES):
        action, _ = policy.get_action(obs_dict[f"drone_{i}"], deterministic=deterministic)
        action_dict[f"drone_{i}"] = action

    # Sensor: reactive rule (trigger when alert detected)
    action_dict["sensor_0"] = 1 if obs_dict["sensor_0"][0] > 0.5 else 0
    return action_dict


# ─────────────────────────────────────────────────────────────────────────────
#  Stats mode — headless evaluation
# ─────────────────────────────────────────────────────────────────────────────
def run_stats(policy, args):
    """Run N episodes and print capture statistics."""
    env = BorderEnv(
        use_pybullet=False,
        domain_rand=True,
        p_drop=args.p_drop,
        p_spoof=args.p_spoof,
        use_trust=args.use_trust,
        seed=args.seed,
    )

    captures = 0
    total_steps = 0
    rewards_all = []

    print(f"\nRunning {args.episodes} episodes  |  p_drop={args.p_drop}  "
          f"p_spoof={args.p_spoof}  use_trust={args.use_trust}\n")

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        ep_reward = 0.0
        step = 0

        while env.agents:
            action_dict = get_drone_actions(policy, obs, deterministic=True)
            obs, rew, term, trunc, info = env.step(action_dict)
            ep_reward += sum(rew[f"drone_{i}"] for i in range(N_DRONES)) / N_DRONES
            step += 1

        captured = info.get("drone_0", {}).get("captured", False)
        if captured:
            captures += 1
        total_steps += step
        rewards_all.append(ep_reward)

        if ep % 20 == 0 or ep == args.episodes:
            rate = captures / ep * 100
            print(f"  Episode {ep:>4}/{args.episodes}  |  "
                  f"Captures: {captures:>3}  |  Rate: {rate:5.1f}%  |  "
                  f"Avg reward: {np.mean(rewards_all):>7.2f}  |  "
                  f"Avg length: {total_steps/ep:>5.0f}")

    rate = captures / args.episodes * 100
    print(f"\n{'='*55}")
    print(f"  RESULTS  ({args.episodes} episodes)")
    print(f"{'='*55}")
    print(f"  Capture rate : {rate:.1f}%  ({captures}/{args.episodes})")
    print(f"  Avg reward   : {np.mean(rewards_all):.2f}")
    print(f"  Avg length   : {total_steps / args.episodes:.0f} steps")
    print(f"{'='*55}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  Visual mode — PyBullet 3D rendering
# ─────────────────────────────────────────────────────────────────────────────
def run_visual(policy, args):
    """Run episodes with PyBullet visualization."""
    import pybullet as p
    import os
    import gym_pybullet_drones as g

    ASSETS_DIR = os.path.join(os.path.dirname(g.__file__), 'assets')
    CF2X_URDF  = os.path.join(ASSETS_DIR, 'cf2x.urdf')
    RACER_URDF = os.path.join(ASSETS_DIR, 'racer.urdf')

    SCALE = 8.0
    DRONE_COLORS = [
        [1.0, 0.2, 0.2, 1.0],
        [0.2, 1.0, 0.2, 1.0],
        [0.2, 0.4, 1.0, 1.0],
    ]

    env = BorderEnv(
        use_pybullet=True,
        render_mode='human',
        domain_rand=True,
        p_drop=args.p_drop,
        p_spoof=args.p_spoof,
        use_trust=args.use_trust,
    )

    episode = 0
    print(f"\nVisualizing trained policy  |  p_drop={args.p_drop}  "
          f"p_spoof={args.p_spoof}  use_trust={args.use_trust}")
    print("Press Ctrl+C to stop\n")

    try:
        while True:
            episode += 1
            obs, _ = env.reset()
            pb = env._pb

            # Camera & world
            p.resetDebugVisualizerCamera(28, 45, -35, [10, 10, 2], physicsClientId=pb)
            corners = [([0,0,0],[20,0,0]),([20,0,0],[20,20,0]),
                       ([20,20,0],[0,20,0]),([0,20,0],[0,0,0])]
            for a, b in corners:
                p.addUserDebugLine(a, b, [1,0.5,0], 3, physicsClientId=pb)

            # Load drone models
            drone_ids = []
            for i in range(3):
                did = p.loadURDF(CF2X_URDF, env.drone_pos[i].tolist(),
                                p.getQuaternionFromEuler([0,0,0]),
                                physicsClientId=pb, globalScaling=SCALE)
                if DRONE_COLORS[i]:
                    for link in range(-1, p.getNumJoints(did, physicsClientId=pb)):
                        p.changeVisualShape(did, link, rgbaColor=DRONE_COLORS[i],
                                           physicsClientId=pb)
                drone_ids.append(did)

            intruder_id = p.loadURDF(RACER_URDF, env.intruder_pos.tolist(),
                                    p.getQuaternionFromEuler([0,0,0]),
                                    physicsClientId=pb, globalScaling=SCALE+2)
            for link in range(-1, p.getNumJoints(intruder_id, physicsClientId=pb)):
                p.changeVisualShape(intruder_id, link, rgbaColor=[1,1,0,1],
                                   physicsClientId=pb)

            # Labels
            for i, did in enumerate(drone_ids):
                p.addUserDebugText(f'Hunter-{i}', [0,0,1.2],
                                  textColorRGB=DRONE_COLORS[i][:3],
                                  textSize=1.2, parentObjectUniqueId=did,
                                  physicsClientId=pb)
            p.addUserDebugText('INTRUDER', [0,0,1.5], textColorRGB=[1,1,0],
                              textSize=1.3, parentObjectUniqueId=intruder_id,
                              physicsClientId=pb)

            print(f"\n── Episode {episode} ──")
            print(f"   Intruder at: {env.intruder_pos.round(2)}  "
                  f"Speed: {env.intruder_speed:.1f} m/s  "
                  f"Wind: {env.wind_vec.round(1)} m/s")

            step = 0
            while env.agents:
                action_dict = get_drone_actions(policy, obs, deterministic=True)
                obs, rew, term, trunc, info = env.step(action_dict)
                step += 1

                # Update visual positions
                for i in range(3):
                    vx, vy = float(env.drone_vel[i][0]), float(env.drone_vel[i][1])
                    orn = p.getQuaternionFromEuler([
                        float(np.clip(-vy*0.12, -0.4, 0.4)),
                        float(np.clip( vx*0.12, -0.4, 0.4)), 0])
                    p.resetBasePositionAndOrientation(
                        drone_ids[i], env.drone_pos[i].tolist(), orn,
                        physicsClientId=pb)

                p.resetBasePositionAndOrientation(
                    intruder_id, env.intruder_pos.tolist(),
                    p.getQuaternionFromEuler([0,0,0]), physicsClientId=pb)

                time.sleep(0.02)

            captured = info.get("drone_0", {}).get("captured", False)
            result = "CAPTURED!" if captured else f"Timeout ({step} steps)"
            print(f"   Result: {result}")

            # Cleanup models
            for bid in drone_ids + [intruder_id]:
                try: p.removeBody(bid, physicsClientId=pb)
                except: pass

    except KeyboardInterrupt:
        print("\nStopped.")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a trained SentryNet policy")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to .pt checkpoint file")
    parser.add_argument("--mode", type=str, default="stats", choices=["stats", "visual"],
                        help="stats = headless evaluation, visual = PyBullet 3D")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Number of episodes (stats mode, default: 100)")
    parser.add_argument("--p_drop", type=float, default=0.0,
                        help="Packet drop rate during evaluation (default: 0.0)")
    parser.add_argument("--p_spoof", type=float, default=0.0,
                        help="Spoof rate during evaluation (default: 0.0)")
    parser.add_argument("--use_trust", action="store_true",
                        help="Enable trust-weighted aggregation")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed")

    args = parser.parse_args()

    policy = load_policy(args.checkpoint)

    if args.mode == "stats":
        run_stats(policy, args)
    else:
        run_visual(policy, args)
