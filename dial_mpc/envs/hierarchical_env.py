import mujoco
import jax
import jax.numpy as jnp
from functools import partial
from dataclasses import dataclass

import torch
from tensordict.nn import TensorDictModule

from brax import math
from brax import envs as brax_envs
from brax.io import mjcf
from brax.base import System
from brax.mjx.base import State as PipelineState
from brax.envs.base import State

from dial_mpc.envs.base_env import BaseEnv, BaseEnvConfig
from dial_mpc.utils.io_utils import get_model_path
from dial_mpc.utils.td_mod_to_jax import td_module_to_jax, set_interaction_mode


# fmt: off
MJC_JOINTS = [
    'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint', 
    'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
    'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint',
    'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint', 
]

ISAAC_JOINTS = [
    'FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint', 
    'FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint', 
    'FL_calf_joint', 'FR_calf_joint', 'RL_calf_joint', 'RR_calf_joint'
]
# fmt: on

isaac2mjc = jnp.array([ISAAC_JOINTS.index(joint) for joint in MJC_JOINTS])
mjc2isaac = jnp.array([MJC_JOINTS.index(joint) for joint in ISAAC_JOINTS])


@dataclass
class HierarchicalEnvConfig(BaseEnvConfig):
    policy_path: str = None
    command_size: int = -1
    action_hist: int = -1
    kp = 25.
    kd = 0.5
    action_scale = 0.5


class HierarchicalEnv(BaseEnv):
    def __init__(self, config: HierarchicalEnvConfig):
        super().__init__(config)

        policy_td: TensorDictModule = torch.load(config.policy_path)
        policy_td.module[0].set_missing_tolerance(True)
        set_interaction_mode("mode")
        self.policy_jax, self.params = td_module_to_jax(policy_td)

        self._command_size = config.command_size
        self._action_hist = config.action_hist

        self._torso_idx = mujoco.mj_name2id(
            self.sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, "base"
        )
        self._init_q = jnp.array(self.sys.mj_model.keyframe("home").qpos)

        # fmt: off
        self.default_joint_pos_isaac = jnp.array(
            [0.1, -0.1, 0.1, -0.1,
             0.78, 0.78, 0.78, 0.78,
             -1.5, -1.5, -1.5, -1.5]
        )
        self.default_joint_pos_mjc = self.default_joint_pos_isaac[isaac2mjc]
        # fmt: on

    @property
    def action_size(self) -> int:
        return self._command_size

    def reset(self, rng: jax.Array) -> State:
        rng, key = jax.random.split(rng)

        pipeline_state = self.pipeline_init(self._init_q, jnp.zeros(self._nv))

        state_info = {
            "rng": rng,
            "is_init": jnp.array([1]),
            "adapt_hx": jnp.zeros(128),
            "action_buf": jnp.zeros((12, self._action_hist)),
            "applied_action": jnp.zeros(12),
            "vel_tar": jnp.array([1.5, 0.0, 0.0]),
        }

        obs = self._compute_obs(pipeline_state, state_info)
        reward, done = jnp.zeros(2)
        metrics = {}
        state = State(pipeline_state, obs, reward, done, metrics, state_info)
        return state
    
    @partial(jax.jit, static_argnums=(0,))
    def act2joint(self, act: jax.Array) -> jax.Array:
        joint_targets = self.default_joint_pos_mjc + act * self._config.action_scale
        joint_targets = jnp.clip(
            joint_targets,
            self.physical_joint_range[:, 0],
            self.physical_joint_range[:, 1],
        )
        return joint_targets

    @partial(jax.jit, static_argnums=(0,))
    def act2tau(self, act: jax.Array, pipline_state) -> jax.Array:
        joint_target = self.act2joint(act)

        q = pipline_state.qpos[7:]
        q = q[: len(joint_target)]
        qd = pipline_state.qvel[6:]
        qd = qd[: len(joint_target)]
        q_err = joint_target - q
        tau = self._config.kp * q_err - self._config.kd * qd

        tau = jnp.clip(
            tau, self.joint_torque_range[:, 0], self.joint_torque_range[:, 1]
        )
        return tau

    def step(self, state: State, action: jax.Array) -> State:
        td_jax = {
            "command": action,
            "policy": state.obs,
            "is_init": state.info["is_init"],
            "adapt_hx": state.info["adapt_hx"],
        }
        # check td_jax shape after vmap
        td_jax = {k: jnp.expand_dims(v, 0) for k, v in td_jax.items()}
        with torch.inference_mode():
            td_jax = self.policy_jax(td_jax, state_dict=self.params)
        
        print("td_jax_out", {k: v.shape for k, v in td_jax.items()})

        # update state info
        state.info["is_init"] = state.info["is_init"].at[:].set(0)
        state.info["adapt_hx"] = td_jax["next", "adapt_hx"].squeeze(0)
        action_isaac = td_jax["action"].squeeze(0)
        action_buf = state.info["action_buf"]
        # TODO: check roll order
        action_buf = jnp.roll(action_buf, 1, axis=1)
        action_buf = action_buf.at[:, 0].set(action_isaac)
        # check if need to assign back
        state.info["action_buf"] = action_buf
        state.info["applied_action"] = state.info["applied_action"] * 0.2 + action_isaac * 0.8

        # compute ctrls
        action_mjc = state.info["applied_action"][isaac2mjc]

        # physics step
        if self._config.leg_control == "position":
            ctrl = self.act2joint(action_mjc)
        elif self._config.leg_control == "torque":
            ctrl = self.act2tau(action_mjc, state.pipeline_state)
        pipeline_state = self.pipeline_step(state.pipeline_state, ctrl)

        # compute obs, reward, done
        obs = self._compute_obs(pipeline_state, state.info)
        reward = self._compute_reward(pipeline_state, state.info)
        done = self._compute_done(pipeline_state, state.info)

        metrics = {}
        state = State(pipeline_state, obs, reward, done, metrics, state.info)
        return state

    def _compute_obs(
        self, pipeline_state: PipelineState, state_info: dict
    ) -> jnp.ndarray:
        # projected gravity, jpos, jvel, action buf
        R = pipeline_state.x.rot[self._torso_idx - 1]
        projected_gravity = math.inv_rotate(jnp.array([0.0, 0.0, -1.0]), R)
        jpos = pipeline_state.qpos[7:][mjc2isaac]
        jvel = pipeline_state.qvel[6:][mjc2isaac]
        action_buf = state_info["action_buf"]
        obs = jnp.concatenate(
            [
                projected_gravity,
                jpos,
                jvel,
                # TODO: check buffer shape
                action_buf.flatten(),
            ]
        )
        return obs

    def _compute_reward(self, pipeline_state: PipelineState, state_info: dict) -> jnp.ndarray:
        x, xd = pipeline_state.x, pipeline_state.xd
        # reward vel
        vb = math.inv_rotate(
            xd.vel[self._torso_idx - 1], x.rot[self._torso_idx - 1]
        )
        reward_vel = -jnp.sum((vb[:2] - state_info["vel_tar"][:2]) ** 2)

        # stay upright reward
        vec_tar = jnp.array([0.0, 0.0, 1.0])
        vec = math.rotate(vec_tar, x.rot[0])
        reward_upright = -jnp.sum(jnp.square(vec - vec_tar))

        reward = (
            reward_vel * 1.0
            + reward_upright * 0.5
        )

        return reward

    def _compute_done(self, pipeline_state: PipelineState, state_info: dict) -> jnp.ndarray:
        up = jnp.array([0.0, 0.0, 1.0])
        joint_angles = pipeline_state.q[7:]
        done = jnp.dot(math.rotate(up, pipeline_state.x.rot[self._torso_idx - 1]), up) < 0
        done |= jnp.any(joint_angles < self.joint_range[:, 0])
        done |= jnp.any(joint_angles > self.joint_range[:, 1])
        done |= pipeline_state.x.pos[self._torso_idx - 1, 2] < 0.18
        done = done.astype(jnp.float32)
        return done


@dataclass
class Go2ForceEnvConfig(HierarchicalEnvConfig):
    pass


class Go2ForceEnv(HierarchicalEnv):
    def make_system(self, config: Go2ForceEnvConfig) -> System:
        model_path = get_model_path("unitree_go2", "mjx_scene_force.xml")
        sys = mjcf.load(model_path)
        sys = sys.tree_replace({"opt.timestep": config.timestep})
        return sys


brax_envs.register_environment("go2_force", Go2ForceEnv)
