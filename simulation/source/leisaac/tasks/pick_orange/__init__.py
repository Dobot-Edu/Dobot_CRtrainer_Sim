import gymnasium as gym

gym.register(
    id='LeIsaac-CR5A-PickOrange-v0',
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cr5a_pick_orange_env_cfg:CR5APickOrangeEnvCfg",
    },
)


__all__ = []
