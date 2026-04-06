# SentryNet

**Trust-Aware Adversarial Multi-Agent Reinforcement Learning for Autonomous Drone Border Surveillance**

> VI Semester Mini Project · Ramaiah Institute of Technology · CSE (AI & ML) + CSE (Cyber Security)  
> Guide: Dr. Siddesh G.M · Co-guide: Mr. Prashanth Reddy

---

## Overview

SentryNet is a research system that trains a team of cooperative drones to autonomously detect and intercept an intruder in a 3D simulated airspace — even when their radio communications are actively jammed or spoofed by an adversary.

The core research question: *can a lightweight, decentralised EMA-based trust scoring mechanism make cooperative drones robust against adversarial communication attacks, without cryptography?*

The system proves this empirically by comparing three training regimes across 54 evaluation conditions, showing that trust-aware MARL (System C) maintains a significantly higher intruder capture rate than standard MARL (System A) under packet drop rates ≥ 0.3.

---

## Demo

```
3 hunter drones (red / green / blue)  +  1 intruder drone (yellow)
20 × 20 × 10 m simulated airspace  ·  PyBullet physics  ·  real CF2X quadrotor models

python simulation_trial.py          # random-action baseline visualisation
python run_trained.py \
  --checkpoint checkpoints/system_C/final.pt \
  --mode visual --p_drop 0.3       # trained System C under 30% attack
```

---

## Architecture

```
BorderEnv (PettingZoo ParallelEnv)
│
├── 3 hunter drones    MAPPO agents  ·  Box(3) actions  ·  20-dim observations
├── 1 ground sensor    QMIX agent    ·  Discrete(2)     ·  4-dim observations
└── 1 intruder drone   Autonomous    ·  random walk biased toward border centre

Communication pipeline (runs every step):
  honest_msg → AdversarialChannel → TrustModule → TrustAggregator → obs[6:12]
```

### Observation space (20-dim per drone)

| Dims  | Content                              |
|-------|--------------------------------------|
| 0–2   | own position (x, y, z)               |
| 3–5   | own velocity (vx, vy, vz)            |
| 6–8   | trust-aggregated intruder position   |
| 9–11  | trust-aggregated intruder velocity   |
| 12    | ground sensor alert flag             |
| 13–15 | intruder position relative to drone (zeros outside 60° FOV / 8 m) |
| 16    | battery level [0, 1]                 |
| 17–19 | wind vector (m/s)                    |

### EMA trust scoring

```
On receive:  error    = ‖received_pos − true_pos‖
             accuracy = max(0,  1 − error / 5.0)
             τ        = 0.1 × accuracy  +  0.9 × τ

On drop:     τ = 0.95 × τ

Always:      τ = clip(τ, 0, 1)
```

Honest senders converge to τ ≈ 1.0. Adversarial senders reach τ < 0.3 within ~200 steps.

### Trust-weighted aggregation

```
agg = Σ(τⱼ · msgⱼ) / Σ(τⱼ)
```

Dropped messages are excluded from both numerator and denominator. No cryptography, no shared secrets — pure local arithmetic.

---

## The Experiment

Three training systems are compared to isolate the contribution of each component:

| System | p_drop | use_trust | Description                    |
|--------|--------|-----------|--------------------------------|
| A      | 0.0    | False     | Standard MARL — baseline       |
| B      | 0.2    | False     | Adversarial training only      |
| C      | 0.2    | True      | Trust-aware MARL ← research contribution |

All other hyperparameters are identical across systems. Evaluation runs 200 episodes per condition across 6 drop rates {0.0, 0.1, 0.2, 0.3, 0.4, 0.5} × 3 seeds = **54 conditions · 10,800 total episodes**.

**Key claim:** System C capture rate > System A capture rate by >20 percentage points at drop_rate ≥ 0.3 (p < 0.05).

---

## Project Structure

```
sentrinet/
│
├── border_env.py           PettingZoo ParallelEnv — 3D world, physics, comms pipeline
├── adversarial_channel.py  Bernoulli packet drop + Gaussian coordinate spoofing
├── trust_module.py         EMA per-sender trust scoring
├── trust_aggregator.py     Trust-weighted message aggregation
│
├── networks.py             PolicyNet (actor) + ValueNet (centralised critic)
├── rollout_buffer.py       GAE rollout storage for MAPPO
├── mappo_trainer.py        Full MAPPO training loop with WandB logging
│
├── train.py                CLI entry point — trains Systems A, B, C
├── evaluate.py             Generates results/full_experiment.csv
├── plot_results.py         Degradation curves and training plots
├── run_trained.py          Load checkpoint → headless stats or 3D visualisation
│
├── simulation_trial.py     Random-action 3D visualisation (Phase 1 demo)
├── diagnostic_3d.py        Environment health checks
│
├── tests/
│   └── test_phase1_3d.py   pytest suite (30+ tests)
│
├── checkpoints/            Saved model checkpoints (git-ignored)
└── results/                CSV and plots (git-ignored)
```

---

## Quickstart

### Prerequisites

```bash
# Python 3.10+
pip install pybullet pettingzoo gymnasium numpy pytest
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install wandb matplotlib pandas scipy seaborn

# Install gym-pybullet-drones (real CF2X drone models)
git clone https://github.com/utiasDSL/gym-pybullet-drones.git
cd gym-pybullet-drones && pip install -e . && cd ..
```

### Verify the environment

```bash
python diagnostic_3d.py
pytest tests/test_phase1_3d.py -v
```

Expected output:
```
── CHECK 1: Reset              ✓  4 agents, drone obs=(20,), sensor obs=(4,)
── CHECK 2: Domain randomisation  ✓  5 episodes with varied mass/wind/speed
── CHECK 3: Step loop          ✓  10 steps OK
── CHECK 4: Adversarial channel   ✓  empirical drop rate ≈ 0.50
── CHECK 5: Full episode       ✓  Episode ended at step 500
═══════════════════════════════════════
  ALL PHASE 1 (3D) CHECKS PASSED
```

### Run the 3D visualisation (random policy)

```bash
python simulation_trial.py
```

Controls in the PyBullet window: left-click drag to rotate, scroll to zoom, R to reset camera.

### Train

```bash
# Smoke test (10k steps, no WandB, fast)
python train.py --system A --fast --no-wandb --seed 0

# Full training — all 3 systems, all 3 seeds (9 runs × 1M steps)
python train.py --system all

# Single system
python train.py --system C --steps 1000000 --seeds 0 1 2
```

### Evaluate

```bash
python evaluate.py --system all --seeds 0 1 2 --episodes 200
```

Outputs `results/full_experiment.csv`.

### Plot results

```bash
python plot_results.py
```

Outputs four plots to `results/plots/`:
- `plot1_degradation_capture_rate.png` — main result
- `plot2_steps_to_capture.png`
- `plot3_trust_dynamics.png`
- `plot4_training_curves_reward.png`

### Run a trained policy

```bash
# Headless stats (100 episodes)
python run_trained.py \
  --checkpoint checkpoints/system_C_seed0/final.pt \
  --episodes 100 --p_drop 0.3

# 3D visualisation with trained policy
python run_trained.py \
  --checkpoint checkpoints/system_C_seed0/final.pt \
  --mode visual --p_drop 0.3 --use_trust
```

---

## Domain Randomisation

Every episode resets with fresh parameters to force policy generalisation:

| Parameter       | Range                        |
|-----------------|------------------------------|
| Drone mass      | ±18% of 0.027 kg (CF2X nominal) |
| Wind vector     | ±15 km/h per axis            |
| Sensor noise σ  | 0.0 – 0.3 m                  |
| Intruder speed  | 1.5 – 4.0 m/s                |

---

## Simulation Realism Features

All improvements are integrated in `border_env.py`:

- **Ornstein-Uhlenbeck wind turbulence** — stochastic gusts each step, not just episode-level randomisation
- **Distance-based packet drop** — channel quality degrades linearly with sender-receiver distance
- **Battery voltage sag** — thrust output scales with remaining battery level
- **FOV-limited detection** — intruder is only visible within 60° cone / 8 m range; zeros otherwise
- **Collision penalty** — −5.0 reward per colliding drone pair per step, forces spread formation
- **Distance-based reward shaping** — continuous approach signal, not just sparse capture reward

---

## Hyperparameters

| Parameter        | Value      |
|------------------|------------|
| Learning rate    | 3 × 10⁻⁴   |
| Discount γ       | 0.99       |
| GAE λ            | 0.95       |
| PPO clip ε       | 0.2        |
| Value coefficient| 0.5        |
| Entropy coefficient | 0.01    |
| Gradient clip    | 10.0       |
| Rollout length   | 2048 steps |
| Mini-batch size  | 256        |
| PPO epochs/rollout | 4        |
| Total steps      | 1,000,000  |

---

## Network Architecture

```
PolicyNet (actor — runs on each drone, deployed at inference)
  Linear(20, 128) → ReLU → Linear(128, 128) → ReLU → Linear(128, 3) → Tanh
  Learnable log_std parameter, clamped to [−2, 2]
  Orthogonal weight initialisation (gain = √2, output layer gain = 0.01)
  Output: 3-dim action in [−1, 1] = [Δx, Δy, Δz] normalised thrust

ValueNet (centralised critic — training only, discarded at deployment)
  Linear(60, 128) → ReLU → Linear(128, 128) → ReLU → Linear(128, 1)
  Input: all 3 drone observations concatenated (20 × 3 = 60-dim)
  Orthogonal initialisation (output layer gain = 1.0)
```

---

## Hardware Deployment Path

The trained PolicyNet exports to a ~20 KB ONNX file and runs at 200 Hz on the STM32 MCU inside a Crazyflie 2.x. Trust aggregation runs on the ground station over Crazyradio PA.

```
Stage 1  Simulation only          ← current state
Stage 2  Software-in-the-loop     inference on laptop, actions to sim
Stage 3  Hardware-in-the-loop     tethered Crazyflie, real motor commands
Stage 4  Free flight demo         3 drones + 1 intruder, live dashboard
```

Minimum hardware for Stage 4: 3× Crazyflie 2.x, 1× Crazyradio PA dongle, Flow Deck v2 per drone (~₹37,500 total).

---

## Tech Stack

| Layer            | Technology                                      |
|------------------|-------------------------------------------------|
| Physics          | PyBullet · gym-pybullet-drones (CF2X URDF)      |
| MARL interface   | PettingZoo ParallelEnv                          |
| RL framework     | PyTorch — custom MAPPO implementation           |
| Experiment tracking | Weights & Biases                             |
| Visualisation    | PyBullet GUI · Matplotlib                       |
| Testing          | pytest (30+ tests)                              |
| OS               | Windows 11 · Python 3.10+                       |

---

## Team

| Name                   | Roll No.     | Department         |
|------------------------|--------------|--------------------|
| Chetan S Siddannavar   | 1MS23CY012   | CSE (Cyber Security) |
| Gokul K.C              | 1MS23CY018   | CSE (Cyber Security) |
| Nitish N.B             | 1MS23CY044   | CSE (Cyber Security) |
| Goutham P              | 1MS24CY403   | CSE (Cyber Security) |

**Guide:** Dr. Siddesh G.M · **Co-guide:** Mr. Prashanth Reddy  
Ramaiah Institute of Technology, Bengaluru · VI Semester · Course: CYP67

---

## Research Context

SentryNet addresses a confirmed gap in the MARL literature: existing adversarial communication defences either require cryptographic infrastructure (impractical on resource-constrained edge drones) or assume a centralised trust server (unavailable in decentralised deployments). SentryNet's EMA mechanism is O(1) time and O(1) memory — the only state required is a single float per sender.

Target venues for publication: AAMAS 2026, IEEE TNNLS, Computers & Security.

---

## License

Academic project. All rights reserved. Contact the authors for reuse permissions.
