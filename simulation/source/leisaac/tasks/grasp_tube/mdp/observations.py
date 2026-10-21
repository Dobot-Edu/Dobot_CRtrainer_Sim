"""Placement check carried over from the reference x-trainer task3."""

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply


def object_in_container(
    env,
    object_cfg: SceneEntityCfg,
    container_cfg: SceneEntityCfg,
    x_range=(-0.20, 0.20),
    y_range=(-0.02, 0.02),
    z_range=(-0.05, 0.0),
    object_offset=(-0.08114, 0.13, -0.01572),
) -> torch.Tensor:
    """Check the tube's rotated local detection point against the rack zone."""
    obj = env.scene[object_cfg.name]
    rack = env.scene[container_cfg.name]
    offset = torch.as_tensor(object_offset, device=env.device, dtype=obj.data.root_pos_w.dtype)
    point = obj.data.root_pos_w + quat_apply(
        obj.data.root_quat_w, offset.expand(env.num_envs, -1)
    )
    relative = point - rack.data.root_pos_w
    return (
        (relative[:, 0] > x_range[0]) & (relative[:, 0] < x_range[1])
        & (relative[:, 1] > y_range[0]) & (relative[:, 1] < y_range[1])
        & (relative[:, 2] > z_range[0]) & (relative[:, 2] < z_range[1])
    )
