"""Three-tube completion rule from the reference x-trainer task3."""

import torch

from .observations import object_in_container


def task3_success(env, tube_objects_cfg, rack_cfg, object_offsets, x_range, y_range, z_range):
    done = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    for tube_cfg in tube_objects_cfg:
        done &= object_in_container(
            env,
            object_cfg=tube_cfg,
            container_cfg=rack_cfg,
            object_offset=object_offsets[tube_cfg.name],
            x_range=x_range,
            y_range=y_range,
            z_range=z_range,
        )
    return done
