"""
border_env.py  ─  SentryNet Phase 1 (3D)
==========================================
3D Border Surveillance Environment  |  gym-pybullet-drones + PettingZoo

World: 20×20×10 m airspace
Agents: 3 hunter drones (MAPPO) + 1 ground sensor (QMIX)
Target: 1 intruder drone (autonomous random-walk)

Observation per drone (20-dim):
  [own_x,y,z (3), own_vx,vy,vz (3), agg_x,y,z (3), agg_vx,vy,vz (3),
   sensor_alert (1), rel_x,y,z (3), battery (1), wind_x,y,z (3)]

Observation sensor (4-dim): [detected, noisy_x, noisy_y, noisy_z]

Action drones : Box(3) in [-1,1]     ← [Δx, Δy, Δz] thrust
Action sensor : Discrete(2)          ← 0=Idle, 1=Trigger

Install:
  pip install pybullet pettingzoo gymnasium numpy
  pip install git+https://github.com/utiasDSL/gym-pybullet-drones.git
"""

from __future__ import annotations
import os
import numpy as np
from typing import Optional, Dict, Any, Tuple

from gymnasium import spaces
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec

from adversarial_channel import AdversarialChannel
from trust_module        import TrustModule
from trust_aggregator    import TrustAggregator

# ── Optional deps ─────────────────────────────────────────────────────────────
try:
    import pybullet as p
    import pybullet_data
    _PYBULLET = True
except ImportError:
    _PYBULLET = False
    print("[SentryNet] pybullet not found → mock physics mode.")

try:
    import gym_pybullet_drones as _gpd_pkg
    _GPD = True
except ImportError:
    _GPD = False

# ── World constants ───────────────────────────────────────────────────────────
N_DRONES      = 3
MAX_STEPS     = 500
CAPTURE_R     = 2.0      # metres
CLOSE_R       = 5.0      # metres (partial reward)
WORLD_XY      = 20.0     # [0, WORLD_XY]
MAX_ALT       = 10.0
DT            = 0.05     # seconds per step
MASS_NOM      = 0.027    # kg (Crazyflie 2.x)
G             = 9.81
MAX_SPEED     = 5.0      # m/s
W1,W2,W3,W4  = 10.0, 0.1, 0.05, 5.0  # reward weights
DRONE_OBS_DIM = 20
SENSOR_OBS_DIM= 4
DRONE_ACT_DIM = 3


# ── Mock physics (no pybullet needed for testing) ─────────────────────────────
class _MockPhysics:
    DRAG = 0.08
    def __init__(self, n, mass, wind):
        self.n, self.mass, self.wind = n, mass.copy(), wind.copy()
        self.pos = np.zeros((n, 3))
        self.vel = np.zeros((n, 3))

    def reset(self, pos):
        self.pos = pos.copy()
        self.vel = np.zeros_like(pos)

    def step(self, thrust):
        for i in range(self.n):
            grav  = np.array([0., 0., -G * self.mass[i]])
            wind_f= self.wind * self.mass[i] * 0.08
            drag  = -self.DRAG * self.vel[i]
            acc   = (thrust[i] + grav + wind_f + drag) / self.mass[i]
            self.vel[i] = np.clip(self.vel[i] + acc*DT, -MAX_SPEED, MAX_SPEED)
            self.pos[i] = np.clip(
                self.pos[i] + self.vel[i]*DT,
                [0,0,0], [WORLD_XY, WORLD_XY, MAX_ALT]
            )

    @property
    def states(self):
        return self.pos.copy(), self.vel.copy()


# ── BorderEnv ─────────────────────────────────────────────────────────────────
class BorderEnv(ParallelEnv):
    """
    PettingZoo ParallelEnv for SentryNet 3D border surveillance.

    Quick start (mock physics, no pybullet needed):
        env = BorderEnv(use_pybullet=False)
        obs, info = env.reset()
        actions = {a: env.action_space(a).sample() for a in env.agents}
        obs, rew, term, trunc, info = env.step(actions)

    Full pybullet mode:
        env = BorderEnv(use_pybullet=True, render_mode="human")
    """

    metadata = {"render_modes": ["human","rgb_array"], "name": "sentrinet_v1"}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        use_pybullet: bool = True,
        domain_rand: bool = True,
        p_drop: float = 0.0,
        p_spoof: float = 0.0,
        spoof_std: float = 2.0,
        use_trust: bool = True,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.render_mode  = render_mode
        self.use_pybullet = use_pybullet and _PYBULLET
        self.domain_rand  = domain_rand
        self.use_trust    = use_trust
        self.rng          = np.random.default_rng(seed)

        self.possible_agents = [f"drone_{i}" for i in range(N_DRONES)] + ["sensor_0"]
        self.agents: list[str] = []

        # Spaces
        inf = np.inf
        self._obs_sp = {
            **{f"drone_{i}": spaces.Box(-inf, inf, (DRONE_OBS_DIM,), np.float32)
               for i in range(N_DRONES)},
            "sensor_0": spaces.Box(-inf, inf, (SENSOR_OBS_DIM,), np.float32),
        }
        self._act_sp = {
            **{f"drone_{i}": spaces.Box(-1., 1., (DRONE_ACT_DIM,), np.float32)
               for i in range(N_DRONES)},
            "sensor_0": spaces.Discrete(2),
        }

        # Comms layer
        self.channel    = AdversarialChannel(p_drop=p_drop, p_spoof=p_spoof,
                                              spoof_std=spoof_std, seed=seed)
        self.trust_mods = [TrustModule(n_senders=N_DRONES-1) for _ in range(N_DRONES)]
        self.aggregator = TrustAggregator(n_senders=N_DRONES-1, msg_dim=6)

        # State (init'd in reset)
        self.drone_pos   = np.zeros((N_DRONES, 3))
        self.drone_vel   = np.zeros((N_DRONES, 3))
        self.intruder_pos= np.zeros(3)
        self.intruder_vel= np.zeros(3)
        self.battery     = np.ones(N_DRONES)
        self.sensor_alert= 0
        self._noisy_int_pos  = np.zeros(3)
        self._agg_msgs       = np.zeros((N_DRONES, 6))
        self.drone_mass  = np.full(N_DRONES, MASS_NOM)
        self.wind_vec    = np.zeros(3)
        self.sensor_noise_std = 0.0
        self.intruder_speed   = 2.5
        self.step_count  = 0
        self._prev_dists = np.full(N_DRONES, WORLD_XY)  # for reward shaping

        # PyBullet handles
        self._pb   = None
        self._dids = []
        self._iid  = None
        self._mock: Optional[_MockPhysics] = None

    def observation_space(self, agent): return self._obs_sp[agent]
    def action_space(self, agent):      return self._act_sp[agent]

    # ── reset ──────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.agents      = self.possible_agents[:]
        self.step_count  = 0
        self.battery     = np.ones(N_DRONES)
        self.sensor_alert= 0

        self._domain_randomise()

        # Spawn positions
        self.drone_pos = np.array([
            [self.rng.uniform(2, 8),   self.rng.uniform(2, 8),   self.rng.uniform(3,6)],
            [self.rng.uniform(12, 18), self.rng.uniform(2, 8),   self.rng.uniform(3,6)],
            [self.rng.uniform(2, 8),   self.rng.uniform(12, 18), self.rng.uniform(3,6)],
        ], dtype=np.float64)
        self.drone_vel     = np.zeros((N_DRONES, 3))
        self.intruder_pos  = self._border_spawn()
        self.intruder_vel  = self._inward_vel(self.intruder_pos)
        self._noisy_int_pos= self.intruder_pos.copy()
        self._prev_dists   = np.linalg.norm(self.drone_pos - self.intruder_pos, axis=1)
        self._agg_msgs     = np.tile(
            np.concatenate([self.intruder_pos, self.intruder_vel]),
            (N_DRONES, 1)
        ).astype(np.float32)

        if self.use_pybullet:
            self._init_pybullet()
        else:
            self._mock = _MockPhysics(N_DRONES, self.drone_mass, self.wind_vec)
            self._mock.reset(self.drone_pos)

        for tm in self.trust_mods:
            tm.reset()
        self.channel.reset_stats()
        return self._obs_all(), {}

    def _domain_randomise(self):
        if not self.domain_rand:
            self.drone_mass       = np.full(N_DRONES, MASS_NOM)
            self.wind_vec         = np.zeros(3)
            self.sensor_noise_std = 0.0
            self.intruder_speed   = 2.5
            return
        # Mass ±18%
        self.drone_mass = self.rng.uniform(MASS_NOM*0.82, MASS_NOM*1.18, N_DRONES)
        # Wind up to 15 km/h
        wmax = 15.0 / 3.6
        self.wind_vec = self.rng.uniform(-wmax, wmax, 3)
        self.wind_vec[2] *= 0.25
        self.sensor_noise_std = float(self.rng.uniform(0.0, 0.30))
        self.intruder_speed   = float(self.rng.uniform(1.5, 4.0))

    # ── step ───────────────────────────────────────────────────────────────
    def step(self, actions: Dict[str, Any]):
        assert self.agents, "Episode done — call reset()."
        self.step_count += 1

        # Reward for sensor action should match the alert state visible at decision time.
        prev_sensor_alert = int(self.sensor_alert)

        thrust = self._to_thrust(actions)
        self._step_physics(thrust)
        self._step_intruder()
        self._step_sensor(actions.get("sensor_0", 0))
        self._comms_pipeline()

        effort = np.linalg.norm(thrust, axis=1) / (MASS_NOM * G)
        self.battery = np.clip(self.battery - 0.0005*effort, 0., 1.)

        rewards  = self._compute_rewards(actions, prev_sensor_alert)
        captured = self._captured()
        trunc    = self.step_count >= MAX_STEPS
        done     = captured or trunc
        if done:
            self.agents = []

        term  = {a: captured for a in self.possible_agents}
        trunc_ = {a: trunc   for a in self.possible_agents}
        info  = self._info(captured)
        return self._obs_all(), rewards, term, trunc_, info

    # ── physics ────────────────────────────────────────────────────────────
    def _to_thrust(self, actions) -> np.ndarray:
        T = np.zeros((N_DRONES, 3))
        for i in range(N_DRONES):
            act  = np.clip(np.asarray(
                actions.get(f"drone_{i}", np.zeros(3)), np.float64), -1., 1.)
            hov  = self.drone_mass[i] * G
            T[i, 2]  = hov * (1.0 + act[2]*0.50)
            T[i, :2] = hov * act[:2] * 0.35
            # Battery voltage sag under high thrust.
            scale = 0.6 + 0.4 * self.battery[i]
            T[i] *= scale
        return T

    def _step_physics(self, thrust):
        if self.use_pybullet and self._pb is not None:
            self._pb_step(thrust)
        else:
            self._mock.mass = self.drone_mass
            self._mock.wind = self.wind_vec
            self._mock.step(thrust)
            self.drone_pos, self.drone_vel = self._mock.states

        # Ornstein-Uhlenbeck wind turbulence process.
        theta, sigma = 0.1, 0.5
        self.wind_vec += (
            -theta * self.wind_vec * DT
            + sigma * np.sqrt(DT) * self.rng.normal(0.0, 1.0, 3)
        )
        self.wind_vec[2] *= 0.3
        self.wind_vec = np.clip(self.wind_vec, -6.0, 6.0)

    def _step_intruder(self):
        centre  = np.array([WORLD_XY/2, WORLD_XY/2, self.intruder_pos[2]])
        to_c    = centre - self.intruder_pos
        bias    = to_c / (np.linalg.norm(to_c)+1e-8) * 0.6
        noise   = np.array([*self.rng.uniform(-1,1,2), self.rng.uniform(-0.2,0.2)]) * 0.4
        d       = bias + noise
        self.intruder_vel = d / (np.linalg.norm(d)+1e-8) * self.intruder_speed
        self.intruder_pos = np.clip(
            self.intruder_pos + self.intruder_vel*DT,
            [0,0,0.5], [WORLD_XY, WORLD_XY, MAX_ALT]
        )

    def _step_sensor(self, action):
        noise = (self.rng.normal(0, self.sensor_noise_std, 3)
                 if self.sensor_noise_std > 0 else np.zeros(3))
        self._noisy_int_pos = self.intruder_pos + noise
        sensor_loc = np.array([WORLD_XY/2, WORLD_XY/2, 0.])
        self.sensor_alert = int(
            np.linalg.norm(self._noisy_int_pos[:2] - sensor_loc[:2]) < 8.0)

    # ── comms ──────────────────────────────────────────────────────────────
    def _comms_pipeline(self):
        honest = np.tile(
            np.concatenate([self.intruder_pos, self.intruder_vel]),
            (N_DRONES, 1)
        ).astype(np.float32)
        true_pos = self.intruder_pos.copy()
        base_drop = float(self.channel.p_drop)

        recv_msgs  = np.zeros((N_DRONES, N_DRONES, 6), np.float32)
        drop_masks = np.zeros((N_DRONES, N_DRONES), bool)

        for sender in range(N_DRONES):
            for receiver in range(N_DRONES):
                if receiver == sender:
                    continue
                dist = float(np.linalg.norm(self.drone_pos[sender] - self.drone_pos[receiver]))
                effective_drop = min(0.95, base_drop + 0.025 * dist)
                self.channel.set_drop_rate(effective_drop)
                msg = honest[sender][np.newaxis, :]
                recv, drops = self.channel.transmit(msg)
                recv_msgs[receiver, sender] = recv[0]
                drop_masks[receiver, sender] = drops[0]

        # Restore original configured drop rate after distance-aware transmission.
        self.channel.set_drop_rate(base_drop)

        agg = np.zeros((N_DRONES, 6), np.float32)
        for i in range(N_DRONES):
            snd    = [j for j in range(N_DRONES) if j != i]
            msgs_i = recv_msgs[i][snd]
            drp_i  = drop_masks[i][snd]
            if self.use_trust:
                scr_i  = self.trust_mods[i].get_trust_scores()
                self.trust_mods[i].update(msgs_i[:,:3], true_pos, drp_i)
            else:
                # Systems A/B: uniform weights (no trust mechanism)
                scr_i = np.ones(N_DRONES - 1, dtype=np.float64)
            agg[i] = self.aggregator.aggregate(msgs_i, scr_i, drp_i)
        self._agg_msgs = agg

    # ── rewards ────────────────────────────────────────────────────────────
    def _compute_rewards(self, actions, sensor_alert_for_reward: int) -> Dict[str, float]:
        dists    = np.linalg.norm(self.drone_pos - self.intruder_pos, axis=1)
        captured = self._captured()
        sec_fail = float(self.channel.get_stats()["empirical_spoof_rate"] > 0.05)
        # Min distance across team (encourages at least one drone to close in)
        min_dist = dists.min()
        rew: Dict[str, float] = {}
        for i in range(N_DRONES):
            r  = -W2                                 # time penalty (-0.1/step)
            r -= W3 * (1.0 - self.battery[i])        # energy cost
            r -= W4 * sec_fail                       # security penalty

            # ── Distance shaping (continuous pursuit signal) ──
            # Reward for getting closer vs previous step (Δdist)
            approach = self._prev_dists[i] - dists[i]  # positive when closing in
            r += 0.5 * approach

            # Proximity bonus (scales with closeness, not a cliff)
            if dists[i] < CLOSE_R:
                r += W1 * 0.1 * (1.0 - dists[i] / CLOSE_R)  # up to +1.0 at contact

            # Team coordination: bonus when min team distance is small
            if min_dist < CLOSE_R:
                r += 0.3 * (1.0 - min_dist / CLOSE_R)

            # Capture
            if captured:
                r += W1

            rew[f"drone_{i}"] = float(r)

        self._prev_dists = dists.copy()  # update for next step

        # Collision penalty between hunter drones.
        for i in range(N_DRONES):
            for j in range(i + 1, N_DRONES):
                if np.linalg.norm(self.drone_pos[i] - self.drone_pos[j]) < 1.5:
                    rew[f"drone_{i}"] -= 5.0
                    rew[f"drone_{j}"] -= 5.0

        alert  = int(sensor_alert_for_reward)
        action = int(actions.get("sensor_0", 0))
        rew["sensor_0"] = 1.0 if (alert and action==1) else \
                          0.05 if (not alert and action==0) else -0.5
        return rew

    def _captured(self) -> bool:
        return bool(np.any(
            np.linalg.norm(self.drone_pos - self.intruder_pos, axis=1) < CAPTURE_R))

    # ── observations ───────────────────────────────────────────────────────
    def _obs_all(self):
        obs = {f"drone_{i}": self._drone_obs(i) for i in range(N_DRONES)}
        obs["sensor_0"] = np.array(
            [float(self.sensor_alert), *self._noisy_int_pos], np.float32)
        return obs

    def _drone_obs(self, i) -> np.ndarray:
        rel = self.intruder_pos - self.drone_pos[i]
        dist = float(np.linalg.norm(rel))
        fwd = self.drone_vel[i] / (np.linalg.norm(self.drone_vel[i]) + 1e-8)
        cos_a = float(np.clip(np.dot(fwd, rel / (dist + 1e-8)), -1.0, 1.0))
        angle = float(np.degrees(np.arccos(cos_a)))
        detected = (dist < 8.0) and (angle < 60.0)
        intruder_rel = rel if detected else np.zeros(3, dtype=np.float64)

        return np.concatenate([
            self.drone_pos[i],                      # 3
            self.drone_vel[i],                      # 3
            self._agg_msgs[i],                      # 6
            [float(self.sensor_alert)],             # 1
            intruder_rel,                           # 3
            [self.battery[i]],                      # 1
            self.wind_vec,                          # 3
        ]).astype(np.float32)                       # = 20

    def _info(self, captured):
        base = dict(
            captured=captured, step=self.step_count,
            intruder_pos=self.intruder_pos.copy(),
            drone_pos=self.drone_pos.copy(),
            wind=self.wind_vec.copy(),
            drone_mass=self.drone_mass.copy(),
            sensor_noise_std=self.sensor_noise_std,
            intruder_speed=self.intruder_speed,
            trust_scores=[tm.get_trust_scores().tolist() for tm in self.trust_mods],
        )
        return {a: base for a in self.possible_agents}

    # ── helpers ────────────────────────────────────────────────────────────
    def _border_spawn(self):
        alt  = float(self.rng.uniform(1.5, 5.0))
        edge = int(self.rng.integers(0, 4))
        r    = float(self.rng.uniform(0, WORLD_XY))
        edges = [
            [r,        0.,       alt],
            [r,        WORLD_XY, alt],
            [0.,       r,        alt],
            [WORLD_XY, r,        alt],
        ]
        return np.array(edges[edge])

    def _inward_vel(self, pos):
        c = np.array([WORLD_XY/2, WORLD_XY/2, pos[2]])
        d = c - pos
        return d / (np.linalg.norm(d)+1e-8) * self.intruder_speed

    # ── pybullet ───────────────────────────────────────────────────────────
    def _init_pybullet(self):
        if not _PYBULLET: return
        if self._pb is not None:
            try: p.disconnect(self._pb)
            except: pass
        self._pb = p.connect(p.GUI if self.render_mode=="human" else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                  physicsClientId=self._pb)
        p.setGravity(self.wind_vec[0]*0.08, self.wind_vec[1]*0.08, -G,
                     physicsClientId=self._pb)
        p.setTimeStep(DT, physicsClientId=self._pb)
        p.loadURDF("plane.urdf", physicsClientId=self._pb)

        urdf = self._get_urdf()
        self._dids = []
        for i in range(N_DRONES):
            did = p.loadURDF(urdf,
                basePosition=self.drone_pos[i].tolist(),
                baseOrientation=p.getQuaternionFromEuler([0,0,0]),
                physicsClientId=self._pb)
            p.changeDynamics(did, -1, mass=float(self.drone_mass[i]),
                             physicsClientId=self._pb)
            self._dids.append(did)
        self._iid = p.loadURDF("sphere_small.urdf",
            basePosition=self.intruder_pos.tolist(),
            physicsClientId=self._pb)

    def _pb_step(self, thrust):
        for i, did in enumerate(self._dids):
            p.applyExternalForce(did, -1, thrust[i].tolist(), [0,0,0],
                                 p.WORLD_FRAME, physicsClientId=self._pb)
        p.stepSimulation(physicsClientId=self._pb)
        for i, did in enumerate(self._dids):
            pos, _ = p.getBasePositionAndOrientation(did, physicsClientId=self._pb)
            vel, _ = p.getBaseVelocity(did, physicsClientId=self._pb)
            self.drone_pos[i] = np.array(pos)
            self.drone_vel[i] = np.array(vel)
        p.resetBasePositionAndOrientation(self._iid,
            self.intruder_pos.tolist(), [0,0,0,1], physicsClientId=self._pb)

    def _get_urdf(self) -> str:
        if _GPD:
            import gym_pybullet_drones as g
            c = os.path.join(os.path.dirname(g.__file__), "assets", "cf2x.urdf")
            if os.path.exists(c): return c
        return "sphere_small.urdf"

    def render(self):
        if self.render_mode == "rgb_array" and _PYBULLET and self._pb:
            w, h = 640, 480
            vm = p.computeViewMatrix([WORLD_XY/2,-5,15],
                [WORLD_XY/2,WORLD_XY/2,0],[0,0,1],physicsClientId=self._pb)
            pm = p.computeProjectionMatrixFOV(60,w/h,0.1,100,physicsClientId=self._pb)
            _, _, rgb, _, _ = p.getCameraImage(w,h,vm,pm,physicsClientId=self._pb)
            return np.array(rgb,np.uint8)[:,:,:3]

    def close(self):
        if _PYBULLET and self._pb:
            try: p.disconnect(self._pb)
            except: pass
            self._pb = None


def make_aec_env(**kw): return parallel_to_aec(BorderEnv(**kw))