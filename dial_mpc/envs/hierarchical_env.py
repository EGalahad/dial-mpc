from jax.numpy import ndarray
import mujoco
import jax
import jax.numpy as jnp
from functools import partial
from dataclasses import dataclass

import torch
from tensordict.nn import TensorDictModule
import active_adaptation

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
    command_controllable_size: int = -1
    action_hist: int = -1
    kp = 25.0
    kd = 0.5
    action_scale = 0.5


class HierarchicalEnv(BaseEnv):
    def __init__(self, config: HierarchicalEnvConfig):
        super().__init__(config)
        self._config: HierarchicalEnvConfig

        policy_td: TensorDictModule = torch.load(config.policy_path)
        policy_td.module[0].set_missing_tolerance(True)
        set_interaction_mode("mode")
        self.policy_jax, self.params = td_module_to_jax(policy_td)

        self._command_size = config.command_size
        self._command_controllable_size = config.command_controllable_size
        self._action_hist = config.action_hist

        # fmt: off
        self.default_joint_pos_isaac = jnp.array(
            [0.1, -0.1, 0.1, -0.1,
             0.78, 0.78, 0.78, 0.78,
             -1.5, -1.5, -1.5, -1.5]
        )
        self.default_joint_pos_mjc = self.default_joint_pos_isaac[isaac2mjc]
        # fmt: on

        self._torso_idx = mujoco.mj_name2id(
            self.sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, "base"
        )
        self._init_q = jnp.array(self.sys.mj_model.keyframe("home").qpos)
        self.physical_joint_range = self.physical_joint_range[:12]
        self.joint_range = self.physical_joint_range

    @property
    def action_size(self) -> int:
        return self._command_controllable_size

    def reset(self, rng: jax.Array) -> State:
        rng, key = jax.random.split(rng)

        pipeline_state = self.pipeline_init(self._init_q, jnp.zeros(self._nv))

        state_info = {
            "rng": rng,
            "command": jnp.zeros(self._command_size),
            "is_init": jnp.array([1]),
            "adapt_hx": jnp.zeros(128),
            "action_buf": jnp.zeros((12, self._action_hist)),
            "applied_action": jnp.zeros(12),
            "vel_tar": jnp.array([1.0, 0.0, 0.0]),
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

        q = pipline_state.qpos[7 : 7 + 12]
        q = q[: len(joint_target)]
        qd = pipline_state.qvel[6 : 6 + 12]
        qd = qd[: len(joint_target)]
        q_err = joint_target - q
        tau = self._config.kp * q_err - self._config.kd * qd

        tau = jnp.clip(
            tau, self.joint_torque_range[:, 0], self.joint_torque_range[:, 1]
        )
        return tau

    def step(self, state: State, action: jax.Array) -> State:
        command = state.info["command"]
        td_jax = {
            "command": command,
            "policy": state.obs,
            "is_init": state.info["is_init"],
            "adapt_hx": state.info["adapt_hx"],
        }
        # check td_jax shape after vmap
        td_jax = {k: jnp.expand_dims(v, 0) for k, v in td_jax.items()}
        with torch.inference_mode():
            td_jax = self.policy_jax(td_jax, state_dict=self.params)

        # update state info
        state.info["is_init"] = state.info["is_init"].at[:].set(0)
        state.info["adapt_hx"] = td_jax["next", "adapt_hx"].squeeze(0)
        action_isaac = td_jax["action"].squeeze(0)
        action_buf = state.info["action_buf"]
        action_buf = jnp.roll(action_buf, 1, axis=1)
        action_buf = action_buf.at[:, 0].set(action_isaac)
        # check if need to assign back
        state.info["action_buf"] = action_buf
        state.info["applied_action"] = (
            state.info["applied_action"] * 0.2 + action_isaac * 0.8
        )

        # compute ctrls and physics step
        action_mjc = state.info["applied_action"][isaac2mjc]
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
        body_quat = pipeline_state.x.rot[self._torso_idx - 1]
        projected_gravity = math.inv_rotate(jnp.array([0.0, 0.0, -1.0]), body_quat)
        jpos = pipeline_state.qpos[7 : 7 + 12][mjc2isaac]
        jvel = pipeline_state.qvel[6 : 6 + 12][mjc2isaac]
        action_buf = state_info["action_buf"]
        obs = jnp.concatenate(
            [
                projected_gravity,
                jpos,
                jvel,
                action_buf.flatten(),
            ]
        )
        return obs

    def _compute_reward(
        self, pipeline_state: PipelineState, state_info: dict
    ) -> jnp.ndarray:
        return jnp.zeros(shape=())

    def _compute_done(
        self, pipeline_state: PipelineState, state_info: dict
    ) -> jnp.ndarray:
        return jnp.zeros(shape=())


@dataclass
class Go2ForceEnvConfig(HierarchicalEnvConfig):
    command_size: int = 10
    command_controllable_size: int = 2
    action_hist: int = 3


class Go2ForceEnv(HierarchicalEnv):
    def make_system(self, config: Go2ForceEnvConfig) -> System:
        model_path = get_model_path("unitree_go2", "mjx_scene_force_with_setpoint.xml")
        sys = mjcf.load(model_path)
        sys = sys.tree_replace({"opt.timestep": config.timestep})
        return sys

    def reset(self, rng: jax.Array) -> System:
        state = super().reset(rng)
        additional_info = {
            "setpoint_pos_b": jnp.array([1.0, 0.0, 0.0]),
            "yaw_diff": jnp.array([0.0]),
            "kp": jnp.array([10.0, 10.0, 10.0]),
            "kd": jnp.array([6.0, 6.0, 6.0]),
            "virtual_mass": jnp.array([1.0]),
        }
        state.info.update(additional_info)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        # update state.info["command"]
        state.info["setpoint_pos_b"] = (
            state.info["setpoint_pos_b"].at[:2].set(action[:2])
        )
        command = jnp.concatenate(
            [
                state.info["setpoint_pos_b"][:2],
                state.info["yaw_diff"],
                state.info["kp"][:2] * state.info["setpoint_pos_b"][:2],
                state.info["kd"],
                state.info["kp"][2] * state.info["yaw_diff"],
                state.info["virtual_mass"],
            ]
        )
        state.info["command"] = state.info["command"].at[:].set(command)

        # Update setpoint position visualization
        body_pos = state.pipeline_state.x.pos[self._torso_idx - 1]
        body_quat = state.pipeline_state.x.rot[self._torso_idx - 1]
        setpoint_pos_b = jnp.array([action[0], action[1], 0.0])
        setpoint_pos = body_pos + math.rotate(setpoint_pos_b, body_quat)
        state = state.replace(
            pipeline_state=state.pipeline_state.replace(
                qpos=state.pipeline_state.qpos.at[-2:].set(setpoint_pos[:2])
            )
        )
        return super().step(state, action)

    def _compute_reward(
        self, pipeline_state: PipelineState, state_info: dict
    ) -> jnp.ndarray:
        x, xd = pipeline_state.x, pipeline_state.xd
        # reward vel
        vb = math.inv_rotate(xd.vel[self._torso_idx - 1], x.rot[self._torso_idx - 1])
        reward_vel = -jnp.sum((vb[:2] - state_info["vel_tar"][:2]) ** 2)

        # stay upright reward
        vec_tar = jnp.array([0.0, 0.0, 1.0])
        vec = math.rotate(vec_tar, x.rot[self._torso_idx - 1])
        reward_upright = -jnp.sum(jnp.square(vec - vec_tar))

        reward = reward_vel * 1.0 + reward_upright * 0.5

        return reward

    def _compute_done(
        self, pipeline_state: PipelineState, state_info: dict
    ) -> jnp.ndarray:
        up = jnp.array([0.0, 0.0, 1.0])
        joint_angles = pipeline_state.q[7 : 7 + 12]
        done = (
            jnp.dot(math.rotate(up, pipeline_state.x.rot[self._torso_idx - 1]), up) < 0
        )
        done |= jnp.any(joint_angles < self.joint_range[:, 0])
        done |= jnp.any(joint_angles > self.joint_range[:, 1])
        done |= pipeline_state.x.pos[self._torso_idx - 1, 2] < 0.18
        done = done.astype(jnp.float32)
        return done


@dataclass
class Hierarchical2RobotsEnvConfig(HierarchicalEnvConfig):
    pass


@dataclass
class Go2ForceTransportEnvConfig(Hierarchical2RobotsEnvConfig):
    command_size: int = 10
    command_controllable_size: int = 6
    action_hist: int = 3


class Hierarchical2RobotsEnv(HierarchicalEnv):
    """task of two robot transporting box by collaboratively apply force to the box, the robot is in [-1, 0, 0] and [1, 0, 0], the box is commanded to move in [0, 1, 0]"""

    def __init__(self, config: Hierarchical2RobotsEnvConfig):
        BaseEnv.__init__(self, config)
        self._config: Hierarchical2RobotsEnvConfig

        policy_td: TensorDictModule = torch.load(config.policy_path)
        policy_td.module[0].set_missing_tolerance(True)
        set_interaction_mode("mode")
        self.policy_jax, self.params = td_module_to_jax(policy_td)

        self._command_size = config.command_size
        self._command_controllable_size = config.command_controllable_size
        self._action_hist = config.action_hist

        # fmt: off
        self.default_joint_pos_isaac = jnp.array(
            [0.1, -0.1, 0.1, -0.1,
             0.78, 0.78, 0.78, 0.78,
             -1.5, -1.5, -1.5, -1.5]
        )
        self.default_joint_pos_mjc = self.default_joint_pos_isaac[isaac2mjc]
        # fmt: on

        self._torso_idx_1 = mujoco.mj_name2id(
            self.sys.mj_model, mujoco.mjtObj.mjOBJ_BODY, "base_1"
        )
        self._torso_idx_2 = mujoco.mj_name2id(
            self.sys.mj_model, mujoco.mjtObj.mjOBJ_BODY, "base_2"
        )
        self._box_idx = mujoco.mj_name2id(
            self.sys.mj_model, mujoco.mjtObj.mjOBJ_BODY, "box"
        )

        self.default_joint_pos_mjc = jnp.tile(
            self.default_joint_pos_mjc, (2, 1)
        ).reshape(-1)
        
        self._init_q = jnp.array(self.sys.mj_model.keyframe("home").qpos)
        self.physical_joint_range = jnp.concatenate(
            [
                self.sys.mj_model.jnt_range[self._torso_idx_1 : self._torso_idx_1 + 12],
                self.sys.mj_model.jnt_range[self._torso_idx_2 : self._torso_idx_2 + 12],
            ],
            axis=0,
        )
        self.joint_range = self.physical_joint_range

    @partial(jax.jit, static_argnums=(0,))
    def act2tau(self, act: jax.Array, pipline_state) -> jax.Array:
        joint_target = self.act2joint(act)

        q = jnp.concatenate(
            [pipline_state.qpos[7 : 7 + 12], pipline_state.qpos[19 + 7 : 19 + 7 + 12]],
            axis=0,
        )  # pipline_state.qpos[7:]
        qd = jnp.concatenate(
            [pipline_state.qvel[6 : 6 + 12], pipline_state.qvel[18 + 6 : 18 + 6 + 12]],
            axis=0,
        )  # pipline_state.qvel[6:]
        q_err = joint_target - q
        tau = self._config.kp * q_err - self._config.kd * qd

        tau = jnp.clip(
            tau, self.joint_torque_range[:, 0], self.joint_torque_range[:, 1]
        )
        return tau

    def reset(self, rng: jax.Array) -> State:
        rng, key = jax.random.split(rng)

        pipeline_state = self.pipeline_init(self._init_q, jnp.zeros(self._nv))

        state_info = {
            "rng": rng,
            "command_1": jnp.zeros(self._config.command_size),
            "command_2": jnp.zeros(self._config.command_size),
            "is_init_1": jnp.array([1]),
            "is_init_2": jnp.array([1]),
            "adapt_hx_1": jnp.zeros(128),
            "adapt_hx_2": jnp.zeros(128),
            "action_buf_1": jnp.zeros((12, self._config.action_hist)),
            "action_buf_2": jnp.zeros((12, self._config.action_hist)),
            "applied_action_1": jnp.zeros(12),
            "applied_action_2": jnp.zeros(12),
            "vel_tar": jnp.array([0.0, 0.0, 0.0]),
        }


        obs = self._compute_obs(pipeline_state, state_info)
        reward, done = jnp.zeros(2)
        metrics = {}
        state = State(pipeline_state, obs, reward, done, metrics, state_info)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        obs_1, obs_2 = jnp.split(state.obs, 2)
        td_jax_1 = {
            "command": state.info["command_1"],
            "policy": obs_1,
            "is_init": state.info["is_init_1"],
            "adapt_hx": state.info["adapt_hx_1"],
        }
        td_jax_2 = {
            "command": state.info["command_2"],
            "policy": obs_2,
            "is_init": state.info["is_init_2"],
            "adapt_hx": state.info["adapt_hx_2"],
        }
        td_jax_1 = {k: jnp.expand_dims(v, 0) for k, v in td_jax_1.items()}
        td_jax_2 = {k: jnp.expand_dims(v, 0) for k, v in td_jax_2.items()}
        with torch.inference_mode():
            td_jax_1 = self.policy_jax(td_jax_1, state_dict=self.params)
            td_jax_2 = self.policy_jax(td_jax_2, state_dict=self.params)

        # update state info
        state.info["is_init_1"] = state.info["is_init_1"].at[:].set(0)
        state.info["is_init_2"] = state.info["is_init_2"].at[:].set(0)
        state.info["adapt_hx_1"] = td_jax_1["next", "adapt_hx"].squeeze(0)
        state.info["adapt_hx_2"] = td_jax_2["next", "adapt_hx"].squeeze(0)
        action_buf_1 = state.info["action_buf_1"]
        action_buf_1 = jnp.roll(action_buf_1, 1, axis=1)
        action_buf_1 = action_buf_1.at[:, 0].set(td_jax_1["action"].squeeze(0))
        state.info["action_buf_1"] = action_buf_1

        action_buf_2 = state.info["action_buf_2"]
        action_buf_2 = jnp.roll(action_buf_2, 1, axis=1)
        action_buf_2 = action_buf_2.at[:, 0].set(td_jax_2["action"].squeeze(0))
        state.info["action_buf_2"] = action_buf_2

        state.info["applied_action_1"] = (
            state.info["applied_action_1"] * 0.2 + td_jax_1["action"].squeeze(0) * 0.8
        )
        state.info["applied_action_2"] = (
            state.info["applied_action_2"] * 0.2 + td_jax_2["action"].squeeze(0) * 0.8
        )

        # Compute ctrls and physics step
        action_mjc_1 = state.info["applied_action_1"][isaac2mjc]
        action_mjc_2 = state.info["applied_action_2"][isaac2mjc]
        action_mjc = jnp.concatenate([action_mjc_1, action_mjc_2])
        if self._config.leg_control == "position":
            ctrl = self.act2joint(action_mjc)
        elif self._config.leg_control == "torque":
            ctrl = self.act2tau(action_mjc, state.pipeline_state)
        pipeline_state = self.pipeline_step(state.pipeline_state, ctrl)

        # Compute obs, reward, done
        obs = self._compute_obs(pipeline_state, state.info)
        reward = self._compute_reward(pipeline_state, state.info)
        done = self._compute_done(pipeline_state, state.info)

        return State(
            obs=obs,
            reward=reward,
            done=done,
            info=state.info,
            pipeline_state=pipeline_state,
        )

    def _compute_obs(
        self, pipeline_state: PipelineState, state_info: dict
    ) -> jax.Array:
        body_quat_1 = pipeline_state.x.rot[self._torso_idx_1 - 1]
        jpos_1 = pipeline_state.qpos[7 : 7 + 12][mjc2isaac]
        jvel_1 = pipeline_state.qvel[6 : 6 + 12][mjc2isaac]
        action_buf_1 = state_info["action_buf_1"]

        body_quat_2 = pipeline_state.x.rot[self._torso_idx_2 - 1]
        projected_gravity_1 = math.inv_rotate(jnp.array([0.0, 0.0, -1.0]), body_quat_1)
        projected_gravity_2 = math.inv_rotate(jnp.array([0.0, 0.0, -1.0]), body_quat_2)
        jpos_2 = pipeline_state.qpos[19 + 7 : 19 + 7 + 12][mjc2isaac]
        jvel_2 = pipeline_state.qvel[18 + 6 : 18 + 6 + 12][mjc2isaac]
        action_buf_2 = state_info["action_buf_2"]

        obs = jnp.concatenate(
            [
                projected_gravity_1,
                jpos_1,
                jvel_1,
                action_buf_1.flatten(),
                projected_gravity_2,
                jpos_2,
                jvel_2,
                action_buf_2.flatten(),
            ]
        )
        return obs


@dataclass
class Go2ForceTransportEnvConfig(Hierarchical2RobotsEnvConfig):
    command_size: int = 10
    command_controllable_size: int = 6
    action_hist: int = 3


class Go2ForceTransportEnv(Hierarchical2RobotsEnv):
    """task of two robot transporting box by collaboratively apply force to the box, the robot is in [-1, 0, 0] and [1, 0, 0], the box is commanded to move in [0, 1, 0]"""
    def make_system(self, config: Go2ForceTransportEnvConfig) -> System:
        model_path = get_model_path(
            "unitree_go2", "mjx_scene_go2_force_transport_with_setpoint.xml"
        )
        sys = mjcf.load(model_path)
        sys = sys.tree_replace({"opt.timestep": config.timestep})
        return sys

    def reset(self, rng: jax.Array) -> State:
        state = super().reset(rng)
        additional_info = {
            "setpoint_pos_b_1": jnp.array([1.0, 0.0, 0.0]),
            "setpoint_pos_b_2": jnp.array([1.0, 0.0, 0.0]),
            "yaw_diff_1": jnp.array([0.0]),
            "yaw_diff_2": jnp.array([0.0]),
            "kp": jnp.array([10.0, 10.0, 10.0]),
            "kd": jnp.array([6.0, 6.0, 6.0]),
            "virtual_mass": jnp.array([1.0]),
        }
        state.info.update(additional_info)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        # update command
        state.info["setpoint_pos_b_1"] = (
            state.info["setpoint_pos_b_1"].at[:2].set(action[:2])
        )
        state.info["yaw_diff_1"] = state.info["yaw_diff_1"].at[:].set(action[2:3])
        state.info["setpoint_pos_b_2"] = (
            state.info["setpoint_pos_b_2"].at[:2].set(action[3:5])
        )
        state.info["yaw_diff_2"] = state.info["yaw_diff_2"].at[:].set(action[5:6])
        command_1 = jnp.concatenate(
            [
                state.info["setpoint_pos_b_1"][:2],
                state.info["yaw_diff_1"],
                state.info["kp"][:2] * state.info["setpoint_pos_b_1"][:2],
                state.info["kd"],
                state.info["kp"][2] * state.info["yaw_diff_1"],
                state.info["virtual_mass"],
            ]
        )
        command_2 = jnp.concatenate(
            [
                state.info["setpoint_pos_b_2"][:2],
                state.info["yaw_diff_2"],
                state.info["kp"][:2] * state.info["setpoint_pos_b_2"][:2],
                state.info["kd"],
                state.info["kp"][2] * state.info["yaw_diff_2"],
                state.info["virtual_mass"],
            ]
        )
        state.info["command_1"] = state.info["command_1"].at[:].set(command_1)
        state.info["command_2"] = state.info["command_2"].at[:].set(command_2)

        # update setpoint visualization
        body_pos_1 = state.pipeline_state.x.pos[self._torso_idx_1 - 1]
        body_quat_1 = state.pipeline_state.x.rot[self._torso_idx_1 - 1]
        setpoint_pos_b_1 = jnp.array([action[0], action[1], 0.0])
        setpoint_pos_1 = body_pos_1 + math.rotate(setpoint_pos_b_1, body_quat_1)
        state = state.replace(
            pipeline_state=state.pipeline_state.replace(
                qpos=state.pipeline_state.qpos.at[-4:-2].set(setpoint_pos_1[:2])
            )
        )
        body_pos_2 = state.pipeline_state.x.pos[self._torso_idx_2 - 1]
        body_quat_2 = state.pipeline_state.x.rot[self._torso_idx_2 - 1]
        setpoint_pos_b_2 = jnp.array([action[3], action[4], 0.0])
        setpoint_pos_2 = body_pos_2 + math.rotate(setpoint_pos_b_2, body_quat_2)
        state = state.replace(
            pipeline_state=state.pipeline_state.replace(
                qpos=state.pipeline_state.qpos.at[-2:].set(setpoint_pos_2[:2])
            )
        )

        return super().step(state, action)

    def _compute_reward(
        self, pipeline_state: PipelineState, state_info: dict
    ) -> jnp.ndarray:
        x, xd = pipeline_state.x, pipeline_state.xd
        # transport box reward
        reward_box_vel = -jnp.sum(
            (xd.vel[self._box_idx - 1][:2] - state_info["vel_tar"][:2]) ** 2
        )
        # close to box reward
        head_vec = jnp.array([0.285, 0.0, 0.0])
        pos_1 = x.pos[self._torso_idx_1 - 1, :2]
        R_1 = math.quat_to_3x3(x.rot[self._torso_idx_1 - 1])
        head_pos_1 = pos_1 + jnp.dot(R_1, head_vec)[:2]
        pos_2 = x.pos[self._torso_idx_2 - 1, :2]
        R_2 = math.quat_to_3x3(x.rot[self._torso_idx_2 - 1])
        head_pos_2 = pos_2 + jnp.dot(R_2, head_vec)[:2]
        box_pos = x.pos[self._box_idx - 1, :2]
        reward_box_1 = -jnp.sum(jnp.square(head_pos_1 - box_pos))
        reward_box_2 = -jnp.sum(jnp.square(head_pos_2 - box_pos))
        reward_box_close = reward_box_1 + reward_box_2
        # box ori
        y_vec = jnp.array([0.0, 1.0, 0.0])
        R_box = math.quat_to_3x3(x.rot[self._box_idx - 1])
        box_y_vec = jnp.dot(R_box, y_vec)
        reward_box_ori = -jnp.sum(jnp.square(box_y_vec - y_vec))
        # box height
        box_height = x.pos[self._box_idx - 1, 2]
        box_height = jnp.clip(box_height, 0.0, 0.35)
        reward_box_height = box_height

        reward = (
            # reward_gaits * 0.1
            # + reward_upright * 0.5
            # + reward_yaw * 0.3
            # + reward_height * 1.0
            +reward_box_vel * 2.0
            + reward_box_close * 2.0
            + reward_box_ori * 5.0
            + reward_box_height * 5.0
        )
        return reward

@dataclass
class Go2VelocityTransportEnvConfig(Hierarchical2RobotsEnvConfig):
    command_size: int = 4
    command_controllable_size: int = 6
    action_hist: int = 3

class Go2VelocityTransportEnv(Hierarchical2RobotsEnv):
    """Task of two robots transporting a box by collaboratively applying velocity commands. 
    The robots are positioned at [-1, 0, 0] and [1, 0, 0], and the box is commanded to move in [0, 1, 0]"""
    
    def make_system(self, config: Go2VelocityTransportEnvConfig) -> System:
        model_path = get_model_path(
            "unitree_go2", "mjx_scene_go2_force_transport_with_setpoint.xml"
        )
        sys = mjcf.load(model_path)
        sys = sys.tree_replace({"opt.timestep": config.timestep})
        return sys

    def reset(self, rng: jax.Array) -> State:
        state = super().reset(rng)
        additional_info = {
            "command_linvel_1": jnp.zeros(2),
            "command_angvel_1": jnp.zeros(1),
            "command_linvel_2": jnp.zeros(2),
            "command_angvel_2": jnp.zeros(1),
            "aux_input": jnp.zeros(1),
        }
        state.info.update(additional_info)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        # Update commands
        state.info["command_linvel_1"] = action[:2]
        state.info["command_angvel_1"] = action[2:3]
        state.info["command_linvel_2"] = action[3:5]
        state.info["command_angvel_2"] = action[5:6]
        
        # Construct commands for both robots
        command_1 = jnp.concatenate([
            state.info["command_linvel_1"],
            state.info["command_angvel_1"],
            state.info["aux_input"],
        ])
        command_2 = jnp.concatenate([
            state.info["command_linvel_2"],
            state.info["command_angvel_2"],
            state.info["aux_input"],
        ])
        
        state.info["command_1"] = state.info["command_1"].at[:].set(command_1)
        state.info["command_2"] = state.info["command_2"].at[:].set(command_2)

        # Update visualization (if needed)
        body_pos_1 = state.pipeline_state.x.pos[self._torso_idx_1 - 1]
        body_quat_1 = state.pipeline_state.x.rot[self._torso_idx_1 - 1]
        setpoint_pos_b_1 = jnp.array([action[0], action[1], 0.0])
        setpoint_pos_1 = body_pos_1 + math.rotate(setpoint_pos_b_1, body_quat_1)
        state = state.replace(
            pipeline_state=state.pipeline_state.replace(
                qpos=state.pipeline_state.qpos.at[-4:-2].set(setpoint_pos_1[:2])
            )
        )
        body_pos_2 = state.pipeline_state.x.pos[self._torso_idx_2 - 1]
        body_quat_2 = state.pipeline_state.x.rot[self._torso_idx_2 - 1]
        setpoint_pos_b_2 = jnp.array([action[3], action[4], 0.0])
        setpoint_pos_2 = body_pos_2 + math.rotate(setpoint_pos_b_2, body_quat_2)
        state = state.replace(
            pipeline_state=state.pipeline_state.replace(
                qpos=state.pipeline_state.qpos.at[-2:].set(setpoint_pos_2[:2])
            )
        )

        return super().step(state, action)

    def _compute_reward(
        self, pipeline_state: PipelineState, state_info: dict
    ) -> jnp.ndarray:
        x, xd = pipeline_state.x, pipeline_state.xd
        
        # Box velocity reward
        reward_box_vel = -jnp.sum(
            (xd.vel[self._box_idx - 1][:2] - state_info["vel_tar"][:2]) ** 2
        )
        
        # Robots close to box reward
        head_vec = jnp.array([0.285, 0.0, 0.0])
        pos_1 = x.pos[self._torso_idx_1 - 1, :2]
        R_1 = math.quat_to_3x3(x.rot[self._torso_idx_1 - 1])
        head_pos_1 = pos_1 + jnp.dot(R_1, head_vec)[:2]
        pos_2 = x.pos[self._torso_idx_2 - 1, :2]
        R_2 = math.quat_to_3x3(x.rot[self._torso_idx_2 - 1])
        head_pos_2 = pos_2 + jnp.dot(R_2, head_vec)[:2]
        box_pos = x.pos[self._box_idx - 1, :2]
        reward_box_1 = -jnp.sum(jnp.square(head_pos_1 - box_pos))
        reward_box_2 = -jnp.sum(jnp.square(head_pos_2 - box_pos))
        reward_box_close = reward_box_1 + reward_box_2
        
        # Box orientation reward
        y_vec = jnp.array([0.0, 1.0, 0.0])
        R_box = math.quat_to_3x3(x.rot[self._box_idx - 1])
        box_y_vec = jnp.dot(R_box, y_vec)
        reward_box_ori = -jnp.sum(jnp.square(box_y_vec - y_vec))
        
        # Box height reward
        box_height = x.pos[self._box_idx - 1, 2]
        box_height = jnp.clip(box_height, 0.0, 0.35)
        reward_box_height = box_height

        # Combine rewards
        reward = (
            reward_box_vel * 2.0
            + reward_box_close * 2.0
            + reward_box_ori * 5.0
            + reward_box_height * 5.0
        )
        return reward

brax_envs.register_environment("go2_force", Go2ForceEnv)
brax_envs.register_environment("go2_force_transport", Go2ForceTransportEnv)
brax_envs.register_environment("go2_vel_transport", Go2VelocityTransportEnv)