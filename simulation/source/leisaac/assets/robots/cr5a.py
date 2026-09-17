import math
import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


_PROJECT_ROOT = Path(os.environ.get("CR5_SIM_ROOT") or Path(__file__).resolve().parents[5])
_DEFAULT_ASSET = _PROJECT_ROOT / "assets" / "robot" / "cr5a_ag95" / "usd" / "cr5a_ag95_mimic.usd"
CR5A_AG95_USD_PATH = os.environ.get("CR5A_AG95_USD_PATH", str(_DEFAULT_ASSET))

CR5A_HOME = [math.radians(v) for v in (88.0, -7.0, 98.0, 0.0, -90.0, -90.0)]
CR5A_ARM_JOINTS = [f"joint{i}" for i in range(1, 7)]
AG95_JOINTS = [
    "gripper_finger1_joint",
    "gripper_finger2_joint",
    "gripper_finger1_finger_joint",
    "gripper_finger2_finger_joint",
    "gripper_finger1_inner_knuckle_joint",
    "gripper_finger2_inner_knuckle_joint",
    "gripper_finger1_finger_tip_joint",
    "gripper_finger2_finger_tip_joint",
]

# The canonical USD authors the remaining seven AG95 joints as PhysX mimic
# joints.  Only the reference joint is independently commanded and exposed to
# teleoperation/recording; the full joint list remains available as simulator
# state for exact replay.
AG95_DRIVER_JOINTS = ["gripper_finger1_joint"]
AG95_MIMIC_JOINTS = [name for name in AG95_JOINTS if name not in AG95_DRIVER_JOINTS]
AG95_ACTION_JOINTS = AG95_DRIVER_JOINTS
AG95_ACTION_FACTORS = (0.6524,)
AG95_MAIN_JOINT = AG95_DRIVER_JOINTS[0]
CR5A_CONTROL_JOINTS = [*CR5A_ARM_JOINTS, *AG95_ACTION_JOINTS]
CR5A_EE_FRAME_LINK = "gripper_finger1_knuckle_link"

CR5A_AG95_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=CR5A_AG95_USD_PATH,
        # The imported CR5A USD is position-controlled.  Keep gravity off until
        # its measured drive gains/effort limits are available; otherwise the
        # arm collapses as soon as an Isaac Lab task starts stepping physics.
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            fix_root_link=True,
            enabled_self_collisions=True,
            # The AG95 contains several light four-bar links.  Extra solver
            # iterations keep their revolute pivots tight under contact load.
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=8,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            **dict(zip(CR5A_ARM_JOINTS, CR5A_HOME)),
            **{name: 0.0 for name in AG95_JOINTS},
        },
    ),
    actuators={
        "cr5a_arm": ImplicitActuatorCfg(
            joint_names_expr=CR5A_ARM_JOINTS,
            effort_limit_sim=150.0,
            velocity_limit_sim=3.14,
            stiffness=800.0,
            damping=80.0,
        ),
        "ag95_driver": ImplicitActuatorCfg(
            joint_names_expr=AG95_DRIVER_JOINTS,
            # Match the 1000 Nm limit carried by the source AG95 URDF.  The
            # previous 100 Nm / 300 Nm-rad drive visibly lagged, letting the
            # independently imported four-bar links separate under load.
            effort_limit_sim=1000.0,
            velocity_limit_sim=2.0,
            stiffness=5000.0,
            damping=200.0,
        ),
        # Register the follower DOFs with Isaac Lab without applying a second
        # drive.  Their motion comes entirely from the USD's PhysX mimic graph.
        "ag95_mimic_passthrough": ImplicitActuatorCfg(
            joint_names_expr=AG95_MIMIC_JOINTS,
            effort_limit_sim=1.0,
            velocity_limit_sim=10.0,
            stiffness=0.0,
            damping=0.0,
        ),
    },
)
