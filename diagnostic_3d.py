"""
diagnostic_3d.py
Run: python diagnostic_3d.py
"""
import numpy as np
from border_env import BorderEnv

G = "\033[92m✓\033[0m"
B = "\033[1m"
R = "\033[0m"

print(f"\n{B}── CHECK 1: Reset{R}")
env = BorderEnv(use_pybullet=False, domain_rand=True, seed=42)
obs, info = env.reset()
assert len(obs) == 4, "Should have 4 agents"
assert obs["drone_0"].shape == (20,), f"Got {obs['drone_0'].shape}"
assert obs["sensor_0"].shape == (4,)
print(f"  {G}  4 agents, drone obs=(20,), sensor obs=(4,)")

print(f"\n{B}── CHECK 2: Domain randomisation{R}")
for ep in range(5):
    env.reset()
    print(f"  {G}  ep{ep}: mass={env.drone_mass.round(4)}, "
          f"wind={env.wind_vec.round(2)}, "
          f"intruder_speed={env.intruder_speed:.2f}")

print(f"\n{B}── CHECK 3: Step loop{R}")
obs, _ = env.reset()
for _ in range(10):
    actions = {a: env.action_space(a).sample() for a in env.agents}
    obs, rew, term, trunc, info = env.step(actions)
d0_info = info["drone_0"]
print(f"  {G}  10 steps OK | trust={d0_info['trust_scores'][0]}")

print(f"\n{B}── CHECK 4: Adversarial channel (50% drop){R}")
env2 = BorderEnv(use_pybullet=False, p_drop=0.5, p_spoof=0.1, seed=0)
obs, _ = env2.reset()
for _ in range(100):
    actions = {a: env2.action_space(a).sample() for a in env2.agents}
    if not env2.agents: break
    env2.step(actions)
stats = env2.channel.get_stats()
dr = stats["empirical_drop_rate"]
assert 0.35 < dr < 0.65, f"Expected ~0.5 drop rate, got {dr:.3f}"
print(f"  {G}  empirical drop rate={dr:.3f} (expected ≈0.5)")

print(f"\n{B}── CHECK 5: Full episode (clean){R}")
env3 = BorderEnv(use_pybullet=False, domain_rand=False)
obs, _ = env3.reset()
done, step = False, 0
while not done:
    actions = {a: env3.action_space(a).sample() for a in env3.agents}
    _, rew, term, trunc, info = env3.step(actions)
    step += 1
    done = not env3.agents
print(f"  {G}  Episode ended at step {step}, "
      f"captured={info['drone_0']['captured']}")

print(f"\n\033[1m\033[92m{'═'*45}\033[0m")
print(f"\033[1m\033[92m  ALL PHASE 1 (3D) CHECKS PASSED\033[0m")
print(f"\033[1m\033[92m{'═'*45}\033[0m\n")