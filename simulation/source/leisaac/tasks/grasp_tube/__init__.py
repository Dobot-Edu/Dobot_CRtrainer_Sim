import gymnasium as gym

gym.register(
    id="LeIsaac-CR5A-GraspTube-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cr5a_grasp_tube_env_cfg:CR5AGraspTubeEnvCfg",
    },
)
