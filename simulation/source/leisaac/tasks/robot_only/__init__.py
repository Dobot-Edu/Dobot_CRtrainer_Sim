"""Minimal CR5A-only environment used for startup and teleoperation checks."""

import gymnasium as gym


gym.register(
    id="LeIsaac-CR5A-RobotOnly-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cr5a_robot_only_env_cfg:CR5ARobotOnlyEnvCfg",
    },
)


__all__ = []
