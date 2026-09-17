"""CR5A + AG95 without a task scene or task objects."""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg, OffsetCfg
from isaaclab.utils import configclass

from leisaac.assets.robots.cr5a import CR5A_AG95_CFG, CR5A_CONTROL_JOINTS, CR5A_EE_FRAME_LINK
from leisaac.tasks.template import mdp
from leisaac.tasks.pick_orange.cr5a_pick_orange_env_cfg import (
    CR5AActionsCfg,
    CR5APickOrangeEnvCfg,
)
from leisaac.tasks.template import (
    SingleArmObservationsCfg,
    SingleArmTaskEnvCfg,
    SingleArmTaskSceneCfg,
    SingleArmTerminationsCfg,
)


@configclass
class CR5ARobotOnlySceneCfg(SingleArmTaskSceneCfg):
    """Only the robot, a ground plane, and the inherited dome light."""

    scene = AssetBaseCfg(
        # GroundPlaneCfg is spawned once globally and does not accept the
        # per-environment regex path used by cloned robot assets.
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    robot = CR5A_AG95_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Link6",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{CR5A_EE_FRAME_LINK}",
                name="flange",
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{CR5A_EE_FRAME_LINK}",
                name="grasp_center",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.155)),
            ),
        ],
    )

    wrist = None
    front = None


@configclass
class CR5ARobotOnlyObservationsCfg(SingleArmObservationsCfg):
    @configclass
    class PolicyCfg(SingleArmObservationsCfg.PolicyCfg):
        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=CR5A_CONTROL_JOINTS, preserve_order=True)},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=CR5A_CONTROL_JOINTS, preserve_order=True)},
        )
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=CR5A_CONTROL_JOINTS, preserve_order=True)},
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=CR5A_CONTROL_JOINTS, preserve_order=True)},
        )
        joint_pos_target = ObsTerm(
            func=mdp.joint_pos_target,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=CR5A_CONTROL_JOINTS, preserve_order=True)},
        )
        wrist = None
        front = None

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class CR5ARobotOnlyTerminationsCfg(SingleArmTerminationsCfg):
    time_out = None


@configclass
class CR5ARobotOnlyEnvCfg(SingleArmTaskEnvCfg):
    scene: CR5ARobotOnlySceneCfg = CR5ARobotOnlySceneCfg(env_spacing=4.0)
    actions: CR5AActionsCfg = CR5AActionsCfg()
    observations: CR5ARobotOnlyObservationsCfg = CR5ARobotOnlyObservationsCfg()
    terminations: CR5ARobotOnlyTerminationsCfg = CR5ARobotOnlyTerminationsCfg()
    dynamic_reset_gripper_effort_limit: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.1)
        self.scene.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)
        self.viewer.eye = (1.8, -2.0, 1.4)
        self.viewer.lookat = (0.0, 0.0, 0.8)

    def use_teleop_device(self, teleop_device):
        if teleop_device not in ("cr5a_keyboard", "cr5a_master"):
            raise ValueError("CR5A RobotOnly supports cr5a_keyboard and cr5a_master")
        self.task_type = teleop_device

    def preprocess_device_action(self, action, teleop_device):
        # Keep the same CR5A/AG95 mapping as PickOrange without importing any
        # task objects or scene USD layers.
        return CR5APickOrangeEnvCfg.preprocess_device_action(self, action, teleop_device)
