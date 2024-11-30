import os
import time
from dataclasses import dataclass
import importlib
import sys

import yaml
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import scienceplots
import art
import emoji

import jax
from jax import numpy as jnp
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline
import functools

from brax.io import html
import brax.envs as brax_envs

import dial_mpc.envs as dial_envs
from dial_mpc.utils.io_utils import get_example_path, load_dataclass_from_dict
from dial_mpc.examples import examples
from dial_mpc.core.dial_config import DialConfig

plt.style.use("science")

# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags

# jax.config.update("jax_disable_jit", True)

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")
jax.config.update("jax_explain_cache_misses", True)


class Converter:
    def __init__(self, ctrl_dt, Hnode, Hsample, nu):
        self.ctrl_dt = ctrl_dt
        self.Hnode = Hnode
        self.Hsample = Hsample
        self.nu = nu
        self.step_us = jnp.linspace(0, self.ctrl_dt * self.Hsample, self.Hsample + 1)
        self.step_nodes = jnp.linspace(0, self.ctrl_dt * self.Hsample, self.Hnode + 1)
        
        self.node2u: functools.partial # process (horizon)
        self.node2u_vmap = jax.jit(
            jax.vmap(self.node2u, in_axes=(1), out_axes=(1))
        )  # process (horizon, ctrl_dim)
        self.u2node_vmap = jax.jit(jax.vmap(self.u2node, in_axes=(1), out_axes=(1)))
        self.node2u_vvmap = jax.jit(
            jax.vmap(self.node2u_vmap, in_axes=(0))
        )  # process (batch, horizon, ctrl_dim)
        self.u2node_vvmap = jax.jit(jax.vmap(self.u2node_vmap, in_axes=(0)))

    @functools.partial(jax.jit, static_argnums=(0,))
    def node2u(self, nodes):
        spline = InterpolatedUnivariateSpline(self.step_nodes, nodes, k=2)
        us = spline(self.step_us)
        return us

    @functools.partial(jax.jit, static_argnums=(0,))
    def u2node(self, us):
        spline = InterpolatedUnivariateSpline(self.step_us, us, k=2)
        nodes = spline(self.step_nodes)
        return nodes

    @functools.partial(jax.jit, static_argnums=(0,))
    def shift(self, Y):
        u = self.node2u_vmap(Y)
        u = jnp.roll(u, -1, axis=0)
        u = u.at[-1].set(jnp.zeros(self.nu))
        Y = self.u2node_vmap(u)
        return Y

def rollout_us(step_env, state, us):
    def step(state, u):
        state = step_env(state, u)
        return state, (state.reward, state.pipeline_state)

    _, (rews, pipline_states) = jax.lax.scan(step, state, us)
    return rews, pipline_states


def main():
    art.tprint("LeCAR @ CMU\nDIAL-MPC", font="big", chr_ignore=True)
    parser = argparse.ArgumentParser()
    config_or_example = parser.add_mutually_exclusive_group(required=True)
    config_or_example.add_argument("--config", type=str, default=None)
    config_or_example.add_argument("--example", type=str, default=None)
    config_or_example.add_argument("--list-examples", action="store_true")
    parser.add_argument(
        "--custom-env",
        type=str,
        default=None,
        help="Custom environment to import dynamically",
    )
    args = parser.parse_args()

    if args.list_examples:
        print("Examples:")
        for example in examples:
            print(f"  {example}")
        return

    if args.custom_env is not None:
        sys.path.append(os.getcwd())
        importlib.import_module(args.custom_env)

    if args.example is not None:
        config_dict = yaml.safe_load(open(get_example_path(args.example + ".yaml")))
    else:
        config_dict = yaml.safe_load(open(args.config))

    dial_config = load_dataclass_from_dict(DialConfig, config_dict)
    rng = jax.random.PRNGKey(seed=dial_config.seed)

    # find env config
    env_config_type = dial_envs.get_config(dial_config.env_name)
    env_config = load_dataclass_from_dict(
        env_config_type, config_dict, convert_list_to_array=True
    )

    print(emoji.emojize(":rocket:") + "Creating environment")
    env: dial_envs.BaseEnv = brax_envs.get_environment(dial_config.env_name, config=env_config)
    reset_env = jax.jit(env.reset)
    step_env = jax.jit(env.step)

    converter = Converter(ctrl_dt=0.02, Hnode=dial_config.Hnode, Hsample=dial_config.Hsample, nu=env.sys.nu)

    rng, rng_reset = jax.random.split(rng)
    state_init = reset_env(rng_reset)
    Y0 = jnp.zeros([dial_config.Hnode + 1, env.sys.nu])
    
    @functools.partial(jax.jit, static_argnames=("sequence_length"))
    def loss_unwrapped(Y, env_state, sequence_length):
        us = converter.node2u_vmap(Y)
        
        def step(state, u):
            nstate = step_env(state, u)
            return nstate, (nstate.reward, nstate)

        final_state, (rewards, states)= jax.lax.scan(
            step, 
            env_state, 
            us, 
            length=sequence_length
        )

        return -jnp.mean(rewards), (states, rewards)

        
    loss = functools.partial(loss_unwrapped, sequence_length=converter.step_us.shape[0])
    loss_grad = jax.grad(loss, has_aux=True)
    
    print("Performing JIT for backward pass")
    compile_time_st = time.time()
    grads, (states, rewards) = loss_grad(Y0, state_init)
    compile_time = time.time() - compile_time_st
    print(f"Compiled in {compile_time:.2f} seconds")

    Nstep = dial_config.n_steps
    rews = []
    rews_plan = []
    rollout = []
    state = state_init
    us = []
    infos = []

    def diffusion_body(state_Y0, i):
        state, Y0 = state_Y0
        grad, (states, rewards) = loss_grad(Y0, state)
        grad = jnp.clip(grad, -dial_config.grad_clip, dial_config.grad_clip)
        Y0 = Y0 - dial_config.lr * grad
        return (state, Y0), rewards

    @functools.partial(jax.jit, static_argnames=("n_diffuse"))
    def optimize_once(state, Y0, n_diffuse):
        (state, Y0), rewardss = jax.lax.scan(diffusion_body, (state, Y0), jnp.arange(n_diffuse))
        return (state, Y0), rewardss

    with tqdm(range(Nstep), desc="Rollout") as pbar:
        for t in pbar:
            # forward single step
            env_state = step_env(state, Y0[0])
            rollout.append(env_state.pipeline_state)
            rews.append(env_state.reward)
            us.append(Y0[0])

            Y0 = converter.shift(Y0)

            n_diffuse = dial_config.Ndiffuse
            # if t == 0:
            #     n_diffuse = dial_config.Ndiffuse_init

            t0 = time.time()

            state = env_state
            (state, Y0), rewardss = optimize_once(state, Y0, n_diffuse)
            # for i in range(n_diffuse):
                # (state, Y0), rewards = diffusion_body((state, Y0), i)

            rews_plan.append(jnp.mean(rewards))
            info = {
                "rew": state.reward, 
                "freq": 1 / (time.time() - t0),
                "xbar": state.pipeline_state.x.pos,
            }
            infos.append(info)
            
            freq = 1 / (time.time() - t0)
            pbar.set_postfix({"rew": f"{env_state.reward:.2e}", "freq": f"{freq:.2f}"})

    rew = jnp.array(rews).mean()
    print(f"mean reward = {rew:.2e}")

    # save us
    # us = jnp.array(us)
    # jnp.save("./results/us.npy", us)

    # create result dir if not exist
    if not os.path.exists(dial_config.output_dir):
        os.makedirs(dial_config.output_dir)

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # plot rews_plan
    # plt.plot(rews_plan)
    # plt.savefig(os.path.join(dial_config.output_dir,
    #             f"{timestamp}_rews_plan.pdf"))

    # host webpage with flask
    print("Processing rollout for visualization")
    import flask

    app = flask.Flask(__name__)
    webpage = html.render(
        env.sys.tree_replace({"opt.timestep": env.dt}), rollout, 1080, True
    )

    # save the html file
    with open(
        os.path.join(dial_config.output_dir, f"{timestamp}_brax_visualization.html"),
        "w",
    ) as f:
        f.write(webpage)

    # save the rollout
    data = []
    xdata = []
    for i in range(len(rollout)):
        pipeline_state = rollout[i]
        data.append(
            jnp.concatenate(
                [
                    jnp.array([i]),
                    pipeline_state.qpos,
                    pipeline_state.qvel,
                    pipeline_state.ctrl,
                ]
            )
        )
        xdata.append(infos[i]["xbar"][-1])
    data = jnp.array(data)
    xdata = jnp.array(xdata)
    jnp.save(os.path.join(dial_config.output_dir, f"{timestamp}_states"), data)
    jnp.save(os.path.join(dial_config.output_dir, f"{timestamp}_predictions"), xdata)

    @app.route("/")
    def index():
        return webpage

    app.run(port=5000)


if __name__ == "__main__":
    main()