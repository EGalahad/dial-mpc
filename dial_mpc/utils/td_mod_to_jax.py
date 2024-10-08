import torch
import torch.nn as nn
import torch.distributions as D
import jax
import copy
from typing import List, Optional, Literal

from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictModuleBase, TensorDictSequential
import torch.utils
import torch.utils.dlpack
from torchrl.modules.tensordict_module.probabilistic import SafeProbabilisticModule
from torchrl.envs.transforms import Compose, ObservationNorm, CatTensors
from .torch2jax import Torchish, t2j_array

INTERACTION_MODE: Literal["random", "mean", "mode"] = "random"

def set_interaction_mode(mode: Literal["random", "mean", "mode"]):
    global INTERACTION_MODE
    INTERACTION_MODE = mode

# import jax.numpy as jnp
# jnp.set_printoptions(precision=4, linewidth=200)
# def gru_cell(input: Torchish, hx: Torchish, w_ih: Torchish, w_hh: Torchish, b_ih: Torchish = None, b_hh: Torchish = None) -> Torchish:
#     w_ir, w_iz, w_in = jnp.split(w_ih.T, 3, axis=1)
#     w_hr, w_hz, w_hn = jnp.split(w_hh.T, 3, axis=1)
#     if b_ih is not None:
#       b_ir, b_iz, b_in = jnp.split(b_ih.reshape(3, -1), 3)
#       b_hr, b_hz, b_hn = jnp.split(b_hh.reshape(3, -1), 3)
#       r = jax.nn.sigmoid(input @ w_ir + b_ir + hx @ w_hr + b_hr)
#       z = jax.nn.sigmoid(input @ w_iz + b_iz + hx @ w_hz + b_hz)
#       n = jax.nn.tanh(input @ w_in + b_in + r * (hx @ w_hn + b_hn))
#     else:
#       r = jax.nn.sigmoid(input @ w_ir + hx @ w_hr)
#       z = jax.nn.sigmoid(input @ w_iz + hx @ w_hz)
#       n = jax.nn.tanh(input @ w_in + r * (hx @ w_hn))
#     return (1 - z) * n + z * hx


# input_size = 8
# hidden_size = 16

# input_jnp = jax.random.normal(jax.random.PRNGKey(0), (1, input_size), dtype=jnp.float32)
# hx_jnp = jax.random.normal(jax.random.PRNGKey(1), (1, hidden_size), dtype=jnp.float32)
# w_ih_jnp = jax.random.normal(jax.random.PRNGKey(2), (3 * hidden_size, input_size), dtype=jnp.float32)
# w_hh_jnp = jax.random.normal(jax.random.PRNGKey(3), (3 * hidden_size, hidden_size), dtype=jnp.float32)
# b_ih_jnp = jax.random.normal(jax.random.PRNGKey(4), (3 * hidden_size,), dtype=jnp.float32)
# b_hh_jnp = jax.random.normal(jax.random.PRNGKey(5), (3 * hidden_size,), dtype=jnp.float32)

# input_torch = torch.utils.dlpack.from_dlpack(torch.from_dlpack(input_jnp))
# hx_torch = torch.utils.dlpack.from_dlpack(torch.from_dlpack(hx_jnp))
# w_ih_torch = torch.utils.dlpack.from_dlpack(torch.from_dlpack(w_ih_jnp))
# w_hh_torch = torch.utils.dlpack.from_dlpack(torch.from_dlpack(w_hh_jnp))
# b_ih_torch = torch.utils.dlpack.from_dlpack(torch.from_dlpack(b_ih_jnp))
# b_hh_torch = torch.utils.dlpack.from_dlpack(torch.from_dlpack(b_hh_jnp))

# out_torch = torch.gru_cell(input_torch, hx_torch, w_ih_torch, w_hh_torch, b_ih_torch, b_hh_torch)
# print(out_torch)

# out_jnp = gru_cell(input_jnp, hx_jnp, w_ih_jnp, w_hh_jnp, b_ih_jnp, b_hh_jnp)
# print(out_jnp)

# def print_dict_keys(pytree, depth=0):
#     if isinstance(pytree, dict):
#         print("  " * depth + "Dict keys:", list(pytree.keys()))
#         for v in pytree.values():
#             print_dict_keys(v, depth + 1)
#     elif isinstance(pytree, (list, tuple)):
#         print("  " * depth + f"{type(pytree).__name__}:")
#         for item in pytree:
#             print_dict_keys(item, depth + 1)
#     else:
#         print("  " * depth + f"Leaf: {type(pytree).__name__}")


def t2j_function(f):
    def foo(*args):
        out = f(*jax.tree_util.tree_map(Torchish, args))
        out = jax.tree_util.tree_map(lambda torchish: torchish.value, out)
        out = {unwrap_key(k): v for k, v in out.items()}
        return out

    return foo


def t2j_module(module):
    def f(x, state_dict={}):
        # We want to have a non-mutating API, so we need to copy the module before performing parameter surgery. Note that
        # doing this copy in `t2j_module` and outside of `f` is not sufficient: multiple calls to `f` should not step on
        # each others toes.
        m = copy.deepcopy(module)

        # Can't use torch.func.functional_call due to https://github.com/pytorch/pytorch/issues/110249
        assert state_dict.keys() == dict(m.state_dict()).keys()

        def visit(m, prefix):
            for name, _ in m.named_parameters(recurse=False):
                m._parameters[name] = Torchish(state_dict[".".join(prefix + [name])])

            for name, _ in m.named_buffers(recurse=False):
                m._buffers[name] = Torchish(state_dict[".".join(prefix + [name])])

            # NOTE: named_children() is the non-recursive version of named_modules()
            for name, child in m.named_children():
                visit(child, prefix=prefix + [name])

        # Replace parameters with Torchish objects
        visit(m, prefix=[])

        return t2j_function(m)(x)

    return f


class _TensorDictModule(nn.Module):
    def __init__(self, module: nn.Module, in_keys: list[str], out_keys: list[str]):
        super().__init__()
        self._nn_module = module
        self.in_keys = in_keys
        self.out_keys = out_keys

    def forward(self, td: dict):
        inputs = [td[td_key] for td_key in self.in_keys]
        outputs = self._nn_module(*inputs)

        if not isinstance(outputs, (tuple, list)):
            outputs = (outputs,)

        out_dict = {out_key: output for out_key, output in zip(self.out_keys, outputs)}
        td.update(out_dict)
        # print(self._nn_module, [(key, val.shape) for key, val in td.items()])
        return td


class _TensorDictSequential(nn.Module):
    def __init__(
        self, *modules: TensorDictModule, in_keys: list[str], out_keys: list[str]
    ) -> None:
        super().__init__()
        self._td_modules = nn.ModuleList(list(modules))
        self.in_keys = in_keys
        self.out_keys = out_keys

    def forward(self, td: dict):
        for module in self._td_modules:
            td = module(td)
            # print(module, [(key, val.shape) for key, val in td.items()])
        return td

class _Compose(nn.Module):
    def __init__(self, *transforms) -> None:
        super().__init__()
        self._transforms = nn.ModuleList(list(transforms))
    
    def forward(self, td: dict):
        for transform in self._transforms:
            td = transform(td)
        return td

class _ObservationNorm(nn.Module):
    def __init__(self, in_keys: list[str], out_keys: list[str], loc, scale):
        super().__init__()
        self.in_keys = in_keys
        self.out_keys = out_keys
        self.loc: torch.Tensor
        self.scale: torch.Tensor
        self.register_buffer("loc", loc)
        self.register_buffer("scale", scale)

    def forward(self, td: dict):
        for in_key, out_key in zip(self.in_keys, self.out_keys):
            try:
                td[out_key] = torch.div(td[in_key] - self.loc, self.scale)
            except KeyError:
                pass
        return td

class _SafeProbabilisticModule(nn.Module):
    def __init__(
        self,
        in_keys: list[str],
        out_keys: list[str],
        distribution_class: type[D.Distribution],
        distribution_kwargs: dict,
        return_log_prob: bool,
        log_prob_key: str = "sample_log_prob",
    ):
        super().__init__()
        self.in_keys = in_keys
        self.out_keys = out_keys
        self.dist_keys = in_keys

        self.distribution_class = distribution_class
        self.distribution_kwargs = distribution_kwargs
        self.interaction_mode = INTERACTION_MODE

        self.return_log_prob = return_log_prob
        self.log_prob_key = log_prob_key

    def forward(self, td: dict):
        dist_kwargs = {
            dist_key: td.get(td_key) for dist_key, td_key in zip(self.dist_keys, self.in_keys)
        }
        dist = self.distribution_class(**dist_kwargs, **self.distribution_kwargs)
        if INTERACTION_MODE == "random":
            out_tensors = dist.sample()
        elif INTERACTION_MODE == "mean":
            out_tensors = dist.mean
        elif INTERACTION_MODE == "mode":
            out_tensors = dist.mode
        if isinstance(out_tensors, TensorDict):
            td.update(out_tensors)
            if self.return_log_prob:
                td = dist.log_prob(td)
        else:
            if not isinstance(out_tensors, (list, tuple)):
                out_tensors = (out_tensors,)
            td.update(
                {key: value for key, value in zip(self.out_keys, out_tensors)}
            )
            if self.return_log_prob:
                log_prob = dist.log_prob(*out_tensors)
                td.update({self.log_prob_key: log_prob})
        return td

class _CatTensors(nn.Module):
    def __init__(self, in_keys: list[str], out_key: str) -> None:
        super().__init__()
        self.in_keys = in_keys
        self.out_key = out_key

    def forward(self, td: dict):
        td[self.out_key] = torch.cat([td[key] for key in self.in_keys], dim=-1)
        return td

def wrap_key(k):
    if isinstance(k, tuple):
        new_key = '/'.join(k)
    else:
        new_key = str(k)
    return new_key

def unwrap_key(k: str):
    if k.count('/') > 0:
        new_key = tuple(k.split('/'))
    else:
        new_key = k
    return new_key

def td_module_to_nn_module(tdmod: TensorDictModule | nn.Module) -> nn.Module:
    if isinstance(tdmod, SafeProbabilisticModule):
        return _SafeProbabilisticModule(
            list(map(wrap_key, tdmod.in_keys)),
            list(map(wrap_key, tdmod.out_keys)),
            tdmod.distribution_class,
            tdmod.distribution_kwargs,
            tdmod.return_log_prob,
            tdmod.log_prob_key,
        )

    if isinstance(tdmod, Compose):
        return _Compose(*[td_module_to_nn_module(_module) for _module in tdmod.transforms])
    if isinstance(tdmod, ObservationNorm):
        return _ObservationNorm(
            list(map(wrap_key, tdmod.in_keys)),
            list(map(wrap_key, tdmod.out_keys)),
            tdmod.loc,
            tdmod.scale,
        )
    if isinstance(tdmod, CatTensors):
        return _CatTensors(list(map(wrap_key, tdmod.in_keys)), wrap_key(tdmod.out_keys[0]))

    if isinstance(tdmod, TensorDictSequential):
        return _TensorDictSequential(
            *[td_module_to_nn_module(_module) for _module in tdmod.module],
            in_keys=list(map(wrap_key, tdmod.in_keys)),
            out_keys=list(map(wrap_key, tdmod.out_keys))
        )
    if isinstance(tdmod, TensorDictModuleBase):
        return _TensorDictModule(tdmod.module, list(map(wrap_key, tdmod.in_keys)), list(map(wrap_key, tdmod.out_keys)))
    raise NotImplementedError

def td_module_to_jax(module: TensorDictModule):
    module_nn = td_module_to_nn_module(module)
    module_jax = t2j_module(module_nn)
    params = {k: t2j_array(v) for k, v in module_nn.state_dict().items()}
    return module_jax, params
    

if __name__ == "__main__":
    td = TensorDict(
        {
            "command": torch.rand(10),
            "policy": torch.rand(63),
            "is_init": torch.zeros(1, dtype=torch.bool),
            "adapt_hx": torch.zeros(128),
        },
        [],
    ).unsqueeze(0)
    policy_torch = torch.load("/home/elijah/Documents/dial-mpc/policy-go2force-639.pt")
    policy_torch.module[0].set_missing_tolerance(True)

    td_jax = jax.tree_util.tree_map(t2j_array, td.to_dict())
    policy_jax, params = td_module_to_jax(policy_torch)

    out_jax = policy_jax(td_jax, state_dict=params)
    out_torch = policy_torch(td)

    print(out_jax.keys())

    for k, v in out_torch.items(True, True):
        print(k, v.shape)
        print(v.flatten()[:10])
        print(out_jax[k].flatten()[:10])