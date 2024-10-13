from typing import Any, Dict, Sequence, Tuple, Union, List
from dial_mpc.envs.unitree_h1_env import (
    UnitreeH1WalkEnvConfig,
    UnitreeH1PushCrateEnvConfig,
)
from dial_mpc.envs.unitree_go2_env import (
    UnitreeGo2EnvConfig,
    UnitreeGo2SeqJumpEnvConfig,
    UnitreeGo2CrateEnvConfig,
    UnitreeGo2PushEnvConfig,
    UnitreeGo2SlalomEnvConfig,
    UnitreeGo2TransportEnvConfig,
)
from .hierarchical_env import (
    Go2ForceEnvConfig,
    Go2ForceTransportEnvConfig,
    Go2VelocityTransportEnvConfig,
)

from .orca_env import OrcaWalkEnvConfig


_configs = {
    "unitree_h1_walk": UnitreeH1WalkEnvConfig,
    "unitree_h1_push_crate": UnitreeH1PushCrateEnvConfig,
    "unitree_go2_walk": UnitreeGo2EnvConfig,
    "unitree_go2_seq_jump": UnitreeGo2SeqJumpEnvConfig,
    "unitree_go2_crate_climb": UnitreeGo2CrateEnvConfig,
    "unitree_go2_box_push": UnitreeGo2PushEnvConfig,
    "unitree_go2_slalom": UnitreeGo2SlalomEnvConfig,
    "unitree_go2_box_transport": UnitreeGo2TransportEnvConfig,
    "go2_force": Go2ForceEnvConfig,
    "go2_force_transport": Go2ForceTransportEnvConfig,
    "go2_vel_transport": Go2VelocityTransportEnvConfig,
    "orca_walk": OrcaWalkEnvConfig,
}


def register_config(name: str, config: Any):
    _configs[name] = config


def get_config(name: str) -> Any:
    return _configs[name]
