# visualize.py
# ─────────────────────────────────────────────────────────────────────────────
# SentryNet Phase 1 — Real-time 3D Visualisation
#
# Shows real CF2X quadrotor drones (scaled up) flying in the 20×20×10m
# border surveillance world. Drones act randomly for now — Phase 2 replaces
# the random actions with a trained MAPPO policy.
#
# Controls (in PyBullet window):
#   Left click + drag   → rotate view
#   Scroll wheel        → zoom in/out
#   Right click + drag  → pan
#   R                   → reset camera
#
# Run:
#   python visualize.py
#
# Options (edit the CONFIG block below):
#   SCALE         → drone visual size (8.0 = good default)
#   SPEED         → simulation speed (0.02 = real-time, 0.0 = max speed)
#   P_DROP        → packet drop rate (0.0 = clean, 0.5 = heavy attack)
#   P_SPOOF       → spoof rate (0.0 = clean, 0.1 = light spoofing)
#   DOMAIN_RAND   → True = different wind/mass every episode
# ─────────────────────────────────────────────────────────────────────────────

from border_env import BorderEnv
import numpy as np
import time
import pybullet as p
import os
import gym_pybullet_drones as g

# ═════════════════════════════════════════════════════════════════════════════
#  CONFIG — edit these to change behaviour
# ═════════════════════════════════════════════════════════════════════════════

SCALE       = 8.0    # drone visual scale  (try 6–15)
SPEED       = 0.02   # seconds per step    (0.02 = ~real-time, 0.0 = fastest)
P_DROP      = 0.0    # packet drop rate    (0.0 = no attack, 0.5 = heavy)
P_SPOOF     = 0.0    # spoof rate          (0.0 = no attack, 0.1 = light)
DOMAIN_RAND = True   # randomise per episode

# ═════════════════════════════════════════════════════════════════════════════
#  URDF paths
# ═════════════════════════════════════════════════════════════════════════════

ASSETS_DIR  = os.path.join(os.path.dirname(g.__file__), 'assets')
CF2X_URDF   = os.path.join(ASSETS_DIR, 'cf2x.urdf')    # hunter drones
RACER_URDF  = os.path.join(ASSETS_DIR, 'racer.urdf')   # intruder drone

# Hunter colors: red, green, blue
DRONE_COLORS = [
    [1.0, 0.2, 0.2, 1.0],
    [0.2, 1.0, 0.2, 1.0],
    [0.2, 0.4, 1.0, 1.0],
]

# ═════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═════════════════════════════════════════════════════════════════════════════

def load_drone(pb, urdf_path, position, color=None, scale=8.0):
    """Load a drone URDF at position, optionally tint it and scale it up."""
    body_id = p.loadURDF(
        urdf_path,
        basePosition=position,
        baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
        physicsClientId=pb,
        flags=p.URDF_USE_SELF_COLLISION,
        globalScaling=scale,
    )
    if color:
        num_joints = p.getNumJoints(body_id, physicsClientId=pb)
        for link_idx in range(-1, num_joints):
            p.changeVisualShape(
                body_id, link_idx,
                rgbaColor=color,
                physicsClientId=pb,
            )
    return body_id


def set_camera(pb):
    """Bird's-eye view of the full 20×20m world."""
    p.resetDebugVisualizerCamera(
        cameraDistance=28,
        cameraYaw=45,
        cameraPitch=-35,
        cameraTargetPosition=[10, 10, 2],
        physicsClientId=pb,
    )


def draw_world(pb):
    """Draw orange boundary lines and labels so the world is readable."""
    # Border perimeter
    corners = [
        ([0,  0,  0], [20,  0,  0]),
        ([20, 0,  0], [20, 20,  0]),
        ([20, 20, 0], [0,  20,  0]),
        ([0,  20, 0], [0,   0,  0]),
    ]
    for a, b in corners:
        p.addUserDebugLine(a, b,
                           lineColorRGB=[1, 0.5, 0],
                           lineWidth=3,
                           physicsClientId=pb)

    # Altitude ceiling lines at z=10
    for corner in [[0,0], [20,0], [20,20], [0,20]]:
        p.addUserDebugLine(
            [corner[0], corner[1], 0],
            [corner[0], corner[1], 10],
            lineColorRGB=[1, 0.5, 0],
            lineWidth=1,
            physicsClientId=pb,
        )

    # Labels
    p.addUserDebugText("BORDER CENTRE", [9.2, 10, 0.3],
                       textColorRGB=[1, 1, 0],
                       textSize=1.5, physicsClientId=pb)
    p.addUserDebugText("20m", [10, -0.5, 0],
                       textColorRGB=[0.8, 0.8, 0.8],
                       textSize=1.0, physicsClientId=pb)
    p.addUserDebugText("20m", [-0.5, 10, 0],
                       textColorRGB=[0.8, 0.8, 0.8],
                       textSize=1.0, physicsClientId=pb)
    p.addUserDebugText("10m altitude", [0.5, 0.5, 10],
                       textColorRGB=[0.8, 0.8, 0.8],
                       textSize=0.9, physicsClientId=pb)


def add_labels(pb, drone_ids, intruder_id):
    """Floating name tags above each drone."""
    names  = ['Hunter-0', 'Hunter-1', 'Hunter-2']
    colors = [[1, 0.4, 0.4], [0.4, 1, 0.4], [0.4, 0.6, 1]]
    for i, did in enumerate(drone_ids):
        p.addUserDebugText(
            names[i], [0, 0, 1.2],
            textColorRGB=colors[i],
            textSize=1.2,
            parentObjectUniqueId=did,
            physicsClientId=pb,
        )
    p.addUserDebugText(
        'INTRUDER ▶', [0, 0, 1.5],
        textColorRGB=[1, 1, 0],
        textSize=1.3,
        parentObjectUniqueId=intruder_id,
        physicsClientId=pb,
    )


def remove_bodies(pb, ids):
    """Safely remove pybullet bodies."""
    for bid in ids:
        try:
            p.removeBody(bid, physicsClientId=pb)
        except Exception:
            pass


# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════

print("=" * 55)
print("  SentryNet — Phase 1 Visualisation")
print("=" * 55)
print(f"  Drone scale   : {SCALE}x real size")
print(f"  Sim speed     : {SPEED}s/step")
print(f"  Attack        : drop={P_DROP}, spoof={P_SPOOF}")
print(f"  Domain rand   : {DOMAIN_RAND}")
print("=" * 55)
print("  Press Ctrl+C to stop\n")

env = BorderEnv(
    use_pybullet=True,
    render_mode='human',
    domain_rand=DOMAIN_RAND,
    p_drop=P_DROP,
    p_spoof=P_SPOOF,
)

episode = 0

try:
    while True:
        episode += 1
        obs, info = env.reset()
        pb = env._pb

        # ── Setup world ───────────────────────────────────────────────────
        set_camera(pb)
        draw_world(pb)

        # ── Load real drone models ────────────────────────────────────────
        drone_ids = []
        for i in range(3):
            did = load_drone(
                pb, CF2X_URDF,
                position=env.drone_pos[i].tolist(),
                color=DRONE_COLORS[i],
                scale=SCALE,
            )
            drone_ids.append(did)

        intruder_id = load_drone(
            pb, RACER_URDF,
            position=env.intruder_pos.tolist(),
            color=[1.0, 1.0, 0.0, 1.0],
            scale=SCALE + 2,    # slightly bigger so intruder stands out
        )

        add_labels(pb, drone_ids, intruder_id)

        # ── Print episode header ──────────────────────────────────────────
        print(f'\n{"─"*55}')
        print(f'  EPISODE {episode}')
        print(f'{"─"*55}')
        print(f'  Intruder at  : {env.intruder_pos.round(2)}')
        print(f'  Wind         : {env.wind_vec.round(2)} m/s')
        print(f'  Masses       : {env.drone_mass.round(4)} kg')
        print(f'  Intruder spd : {env.intruder_speed:.2f} m/s')
        print(f'{"─"*55}')

        # ── Episode loop ──────────────────────────────────────────────────
        step = 0
        while env.agents:
            # Random actions — Phase 2 replaces this with trained policy
            actions = {a: env.action_space(a).sample() for a in env.agents}
            obs, rew, term, trunc, info = env.step(actions)
            step += 1

            # ── Update drone positions and orientation ─────────────────
            for i in range(3):
                pos = env.drone_pos[i].tolist()

                # Tilt drone in direction of travel (looks realistic)
                vx  = float(env.drone_vel[i][0])
                vy  = float(env.drone_vel[i][1])
                roll  = float(np.clip(-vy * 0.12, -0.4, 0.4))
                pitch = float(np.clip( vx * 0.12, -0.4, 0.4))
                orn   = p.getQuaternionFromEuler([roll, pitch, 0])

                p.resetBasePositionAndOrientation(
                    drone_ids[i], pos, orn, physicsClientId=pb)

            # ── Update intruder position ───────────────────────────────
            p.resetBasePositionAndOrientation(
                intruder_id,
                env.intruder_pos.tolist(),
                [0, 0, 0, 1],
                physicsClientId=pb,
            )

            # ── Slow down simulation to watchable speed ────────────────
            if SPEED > 0:
                time.sleep(SPEED)

            # ── Terminal log every 100 steps ──────────────────────────
            if step % 100 == 0:
                dists = np.linalg.norm(
                    env.drone_pos - env.intruder_pos, axis=1).round(1)
                trust = env.trust_mods[0].get_trust_scores().round(2)
                print(f'  Step {step:3d} '
                      f'| Intruder: {env.intruder_pos.round(1)} '
                      f'| Dists: {dists} '
                      f'| Trust: {trust} '
                      f'| Batt: {env.battery.round(2)}')

        # ── Episode summary ───────────────────────────────────────────────
        captured = info['drone_0']['captured']
        print(f'{"─"*55}')
        print(f'  Episode {episode} ended '
              f'| Steps: {step} '
              f'| Captured: {captured}')

        # ── Clean up bodies before next episode ───────────────────────────
        remove_bodies(pb, drone_ids + [intruder_id])
        time.sleep(1.5)

except KeyboardInterrupt:
    print('\n\nStopped by user.')
    env.close()