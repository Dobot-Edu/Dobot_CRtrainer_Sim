"""Single CR5A arm facing the tubes and then the rack along +Y."""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from leisaac.tasks.pick_orange.cr5a_pick_orange_env_cfg import (
    CR5AActionsCfg,
    CR5APickOrangeEnvCfg,
    CR5APickOrangeSceneCfg,
    ObservationsCfg as CR5AObservationsCfg,
)
from leisaac.tasks.template import SingleArmTaskEnvCfg, SingleArmTerminationsCfg
from leisaac.utils.constant import ASSETS_ROOT
from leisaac.utils.general_assets import parse_usd_and_create_subassets

from . import mdp


TUBE_USD_PATH = str(Path(ASSETS_ROOT) / "scenes" / "grasp_tube" / "training_env.usd")
TUBE_NAMES = ("newTube_01", "newTube_02", "newTube_03")
TASK_OBJECT_NAMES = (*TUBE_NAMES, "rack")
OBJECT_OFFSET = (-0.08114, 0.13, -0.01572)
SCENE_HEIGHT = 0.8
SCENE_Y_OFFSET = -0.50


@configclass
class CR5AGraspTubeSceneCfg(CR5APickOrangeSceneCfg):
    scene: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Scene",
        spawn=sim_utils.UsdFileCfg(usd_path=TUBE_USD_PATH),
    )


@configclass
class ObservationsCfg(CR5AObservationsCfg):
    @configclass
    class SubtaskCfg(ObsGroup):
        put_in_rack_tube01 = ObsTerm(
            func=mdp.object_in_container,
            params={"object_cfg": SceneEntityCfg("newTube_01"), "container_cfg": SceneEntityCfg("rack"), "object_offset": OBJECT_OFFSET},
        )
        put_in_rack_tube02 = ObsTerm(
            func=mdp.object_in_container,
            params={"object_cfg": SceneEntityCfg("newTube_02"), "container_cfg": SceneEntityCfg("rack"), "object_offset": OBJECT_OFFSET},
        )
        put_in_rack_tube03 = ObsTerm(
            func=mdp.object_in_container,
            params={"object_cfg": SceneEntityCfg("newTube_03"), "container_cfg": SceneEntityCfg("rack"), "object_offset": OBJECT_OFFSET},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class TerminationsCfg(SingleArmTerminationsCfg):
    success = DoneTerm(
        func=mdp.task3_success,
        params={
            "tube_objects_cfg": [SceneEntityCfg(name) for name in TUBE_NAMES],
            "rack_cfg": SceneEntityCfg("rack"),
            "x_range": (-0.20, 0.20),
            "y_range": (-0.02, 0.02),
            "z_range": (-0.05, 0.0),
            "object_offsets": {name: OBJECT_OFFSET for name in TUBE_NAMES},
        },
    )


@configclass
class CR5AGraspTubeEnvCfg(SingleArmTaskEnvCfg):
    scene: CR5AGraspTubeSceneCfg = CR5AGraspTubeSceneCfg(env_spacing=8.0)
    actions: CR5AActionsCfg = CR5AActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    dynamic_reset_gripper_effort_limit: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.rerender_on_reset = True
        self.sim.dt = 1.0 / 120.0
        self.sim.physx.enable_ccd = True
        # The robot USD includes a pedestal (~0.8 m high). Lift the low task3
        # scene to its tabletop, then keep registered rigid-body reset poses in sync.
        self.scene.scene.init_state.pos = (0.0, SCENE_Y_OFFSET, SCENE_HEIGHT)
        self.scene.robot.init_state.pos = (0.60, -0.95, 0.10)
        self.scene.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)
        parse_usd_and_create_subassets(
            TUBE_USD_PATH, self, specific_name_list=list(TASK_OBJECT_NAMES)
        )
        for name in TASK_OBJECT_NAMES:
            cfg = getattr(self.scene, name)
            x, y, z = cfg.init_state.pos
            cfg.init_state.pos = (x, y + SCENE_Y_OFFSET, z + SCENE_HEIGHT)
        self.viewer.eye = (1.7, -2.1, 2.0)
        self.viewer.lookat = (0.6, -0.35, 0.9)

    use_teleop_device = CR5APickOrangeEnvCfg.use_teleop_device
    preprocess_device_action = CR5APickOrangeEnvCfg.preprocess_device_action
