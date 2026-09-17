import torch

import isaaclab.envs.mdp as base_mdp
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import FrameTransformerCfg, OffsetCfg, TiledCameraCfg
from isaaclab.utils import configclass

from leisaac.assets.robots.cr5a import (
    AG95_ACTION_JOINTS,
    AG95_ACTION_FACTORS,
    AG95_MAIN_JOINT,
    CR5A_AG95_CFG,
    CR5A_ARM_JOINTS,
    CR5A_CONTROL_JOINTS,
    CR5A_EE_FRAME_LINK,
    CR5A_HOME,
)
from leisaac.assets.scenes.kitchen import KITCHEN_WITH_ORANGE_CFG, KITCHEN_WITH_ORANGE_USD_PATH
from leisaac.utils.general_assets import parse_usd_and_create_subassets

from . import mdp
from ..template import SingleArmTaskEnvCfg, SingleArmTaskSceneCfg, SingleArmObservationsCfg, SingleArmTerminationsCfg


TASK_OBJECT_NAMES = ("Plate", "Orange001", "Orange002", "Orange003")


@configclass
class CR5APickOrangeSceneCfg(SingleArmTaskSceneCfg):
    """The original PickOrange scene with only the robot asset replaced."""

    scene: AssetBaseCfg = KITCHEN_WITH_ORANGE_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")
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

    # Reuse the two Camera prims authored in cr5a_ag95_mimic.usd.  With
    # spawn=None Isaac Lab creates render products for the existing cameras
    # without replacing their authored transforms or lens parameters.
    wrist = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Link6/wrist_camera_preview",
        spawn=None,
        data_types=["rgb"],
        width=640,
        height=480,
        # The environment advances one 1/60 s simulation step per recorded
        # control step.  A 1/30 s sensor period therefore duplicated every RGB
        # frame.  Zero updates the camera on every environment step.
        update_period=0.0,
    )
    front = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/front_camera_preview",
        spawn=None,
        data_types=["rgb"],
        width=640,
        height=480,
        update_period=0.0,
    )


@configclass
class CR5AActionsCfg:
    arm_action = base_mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=CR5A_ARM_JOINTS, scale=1.0,
        use_default_offset=False, preserve_order=True,
    )
    gripper_action = base_mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=AG95_ACTION_JOINTS, scale=1.0,
        use_default_offset=False, preserve_order=True,
    )


@configclass
class ObservationsCfg(SingleArmObservationsCfg):
    @configclass
    class PolicyCfg(SingleArmObservationsCfg.PolicyCfg):
        # Record only the independently controlled DOFs: six CR5A joints plus
        # the single AG95 mimic driver.  Full articulation state is still kept
        # by the recorder's states/initial_state terms for exact replay.
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

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class SubtaskCfg(ObsGroup):
        pick_orange001 = ObsTerm(func=mdp.orange_grasped, params={"object_cfg": SceneEntityCfg("Orange001"), "gripper_joint_name": AG95_MAIN_JOINT, "grasp_threshold": 0.08})
        put_orange001_to_plate = ObsTerm(func=mdp.put_orange_to_plate, params={"object_cfg": SceneEntityCfg("Orange001"), "plate_cfg": SceneEntityCfg("Plate"), "gripper_joint_name": AG95_MAIN_JOINT, "grasp_threshold": 0.08})
        pick_orange002 = ObsTerm(func=mdp.orange_grasped, params={"object_cfg": SceneEntityCfg("Orange002"), "gripper_joint_name": AG95_MAIN_JOINT, "grasp_threshold": 0.08})
        put_orange002_to_plate = ObsTerm(func=mdp.put_orange_to_plate, params={"object_cfg": SceneEntityCfg("Orange002"), "plate_cfg": SceneEntityCfg("Plate"), "gripper_joint_name": AG95_MAIN_JOINT, "grasp_threshold": 0.08})
        pick_orange003 = ObsTerm(func=mdp.orange_grasped, params={"object_cfg": SceneEntityCfg("Orange003"), "gripper_joint_name": AG95_MAIN_JOINT, "grasp_threshold": 0.08})
        put_orange003_to_plate = ObsTerm(func=mdp.put_orange_to_plate, params={"object_cfg": SceneEntityCfg("Orange003"), "plate_cfg": SceneEntityCfg("Plate"), "gripper_joint_name": AG95_MAIN_JOINT, "grasp_threshold": 0.08})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class TerminationsCfg(SingleArmTerminationsCfg):
    success = DoneTerm(func=mdp.task_done, params={
        "oranges_cfg": [SceneEntityCfg("Orange001"), SceneEntityCfg("Orange002"), SceneEntityCfg("Orange003")],
        "plate_cfg": SceneEntityCfg("Plate"),
        "rest_joint_names": CR5A_ARM_JOINTS,
        "rest_joint_positions": CR5A_HOME,
    })


@configclass
class CR5APickOrangeEnvCfg(SingleArmTaskEnvCfg):
    scene: CR5APickOrangeSceneCfg = CR5APickOrangeSceneCfg(env_spacing=8.0)
    actions: CR5AActionsCfg = CR5AActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    # AG95 uses the fixed effort limits declared in the CR5A articulation
    # config.  The old SO101 mass-based gripper limiter is not part of this
    # single-CR5A task.
    dynamic_reset_gripper_effort_limit: bool = False

    def __post_init__(self):
        super().__post_init__()
        # Refresh RTX sensors immediately after reset so the first recorded
        # frame belongs to the new episode instead of the previous one.
        self.rerender_on_reset = True
        # Rotate the complete rack assembly (rack, CR5A and AG95) 180 degrees
        # around the world Z axis. Quaternion ordering is wxyz.
        self.scene.robot.init_state.pos = (2.0, -1.0, 0.1)
        self.scene.robot.init_state.rot = (0.0, 0.0, 0.0, 1.0)
        # scene.usd is the single source of truth for task-object poses.
        # parse_usd_and_create_subassets() copies each authored world pose into
        # the corresponding Isaac Lab asset's default state.
        parse_usd_and_create_subassets(
            KITCHEN_WITH_ORANGE_USD_PATH,
            self,
            specific_name_list=list(TASK_OBJECT_NAMES),
        )
        self.viewer.eye = (0.0, -3.0, 2.0)
        self.viewer.lookat = (1.2, -1.5, 0.9)

    def use_teleop_device(self, teleop_device):
        if teleop_device not in ("cr5a_keyboard", "bi_keyboard", "cr5a_master"):
            raise ValueError(
                "CR5A PickOrange supports cr5a_keyboard, bi_keyboard and cr5a_master"
            )
        self.task_type = teleop_device

    def preprocess_device_action(self, action, teleop_device):
        gripper_action_joints = AG95_ACTION_JOINTS
        output = torch.zeros((teleop_device.env.num_envs, 6 + len(gripper_action_joints)), device=teleop_device.env.device)
        if action.get("cr5a_selftest_hold"):
            robot = teleop_device.env.scene["robot"]
            names = robot.data.joint_names
            arm_ids = [names.index(name) for name in CR5A_ARM_JOINTS]
            gripper_ids = [names.index(name) for name in gripper_action_joints]
            output[:, :6] = robot.data.joint_pos[:, arm_ids]
            output[:, 6:] = robot.data.joint_pos[:, gripper_ids]
            return output
        raw = torch.as_tensor(action["joint_state"], dtype=torch.float32, device=output.device)
        if action.get("cr5a_master"):
            # The master already provides a 1:1 relative joint target.  Do not
            # add another per-frame slew limit here: the former 0.01 rad limit
            # capped tracking at roughly 0.6 rad/s and made the robot lag far
            # behind the hand.  The articulation actuator retains its own
            # 3.14 rad/s velocity limit for safety.
            output[:, :6] = raw[:6].unsqueeze(0).expand(output.shape[0], -1)
            close = torch.clamp(torch.as_tensor(action["gripper_closed"], device=output.device), 0.0, 1.0)
        else:
            output[:, :6] = raw[8:14] + torch.tensor(CR5A_HOME, device=output.device)
            close = torch.clamp(raw[15] / 0.04, 0.0, 1.0)
        # Only the mimic reference joint is commanded.  The other seven AG95
        # joints follow through the PhysX mimic relationships authored in USD.
        output[:, 6] = close * AG95_ACTION_FACTORS[0]
        return output
