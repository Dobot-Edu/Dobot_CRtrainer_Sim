"""Small pickle-compatible subset of LeRobot's async inference helpers.

The policy server runs in a separate Python environment.  Pickle stores the
module path of each dataclass, so the Isaac Sim client exposes these classes as
``lerobot.async_inference.helpers``.  The server then reconstructs them as its
native LeRobot classes without importing LeRobot's training stack into Isaac
Sim.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RemotePolicyConfig:
    policy_type: str
    pretrained_name_or_path: str
    lerobot_features: dict[str, dict[str, Any]]
    actions_per_chunk: int
    device: str = "cpu"
    rename_map: dict[str, str] = field(default_factory=dict)


@dataclass
class TimedData:
    timestamp: float
    timestep: int

    def get_timestamp(self) -> float:
        return self.timestamp

    def get_timestep(self) -> int:
        return self.timestep


@dataclass
class TimedObservation(TimedData):
    observation: dict[str, Any]
    must_go: bool = False

    def get_observation(self) -> dict[str, Any]:
        return self.observation


@dataclass
class TimedAction(TimedData):
    action: Any

    def get_action(self) -> Any:
        return self.action


def install_pickle_compatibility() -> None:
    """Register helper classes at the module path used by LeRobot 0.6.x."""

    lerobot_path = "lerobot"
    async_path = "lerobot.async_inference"
    helpers_path = f"{async_path}.helpers"

    lerobot_module = sys.modules.get(lerobot_path)
    if lerobot_module is None:
        lerobot_module = types.ModuleType(lerobot_path)
        lerobot_module.__path__ = []
        sys.modules[lerobot_path] = lerobot_module

    async_module = sys.modules.get(async_path)
    if async_module is None:
        async_module = types.ModuleType(async_path)
        async_module.__path__ = []
        sys.modules[async_path] = async_module
        setattr(lerobot_module, "async_inference", async_module)

    helpers_module = types.ModuleType(helpers_path)
    for cls in (RemotePolicyConfig, TimedData, TimedObservation, TimedAction):
        cls.__module__ = helpers_path
        setattr(helpers_module, cls.__name__, cls)

    sys.modules[helpers_path] = helpers_module
    setattr(async_module, "helpers", helpers_module)


__all__ = [
    "RemotePolicyConfig",
    "TimedAction",
    "TimedData",
    "TimedObservation",
    "install_pickle_compatibility",
]
