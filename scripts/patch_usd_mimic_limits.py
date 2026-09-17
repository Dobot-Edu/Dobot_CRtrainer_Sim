"""Add finite PhysX limits to AG95 mimic revolute joints in a USD asset.

Isaac Sim's URDF importer can author the PhysX mimic API while omitting the
child-joint limits. PhysX rejects such a joint at articulation creation time.
This script authors the limits into the USD root layer without flattening the
asset or changing its references.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

try:
    from pxr import Sdf, Usd, UsdPhysics
    _PATCH_APP = None
except ModuleNotFoundError:
    # Isaac Sim's pip environment exposes pxr after Kit starts. Bootstrap a
    # small headless app so this utility also works from plain conda Python.
    from isaacsim import SimulationApp

    _PATCH_APP = SimulationApp({"headless": True})
    from pxr import Sdf, Usd, UsdPhysics


JOINT_LIMITS_DEG = {
    # The source URDF limits are in radians; USD physics revolute limits use
    # degrees.
    "gripper_finger1_joint": (0.0, math.degrees(0.6524)),
    "gripper_finger2_joint": (0.0, math.degrees(0.6524)),
    "gripper_finger1_finger_joint": (0.0, math.degrees(0.29775)),
    "gripper_finger2_finger_joint": (0.0, math.degrees(0.29775)),
    "gripper_finger1_inner_knuckle_joint": (0.0, math.degrees(0.9750)),
    "gripper_finger2_inner_knuckle_joint": (0.0, math.degrees(0.9750)),
    "gripper_finger1_finger_tip_joint": (0.0, math.degrees(0.9750)),
    "gripper_finger2_finger_tip_joint": (0.0, math.degrees(0.9750)),
}


def patch_asset(path: Path) -> list[str]:
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"Could not open USD: {path}")

    root = stage.GetRootLayer()
    stage.SetEditTarget(root)
    found: list[str] = []
    for prim in stage.Traverse():
        name = prim.GetName()
        if name not in JOINT_LIMITS_DEG:
            continue
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            raise RuntimeError(f"{prim.GetPath()} is not a PhysicsRevoluteJoint")
        joint = UsdPhysics.RevoluteJoint(prim)
        lower, upper = JOINT_LIMITS_DEG[name]
        joint.CreateLowerLimitAttr().Set(lower)
        joint.CreateUpperLimitAttr().Set(upper)
        if name != "gripper_finger1_joint":
            # Isaac Sim's importer occasionally writes the gearing/offset
            # attributes but omits referenceJoint on the first linkage in a
            # chain.  PhysX requires exactly one target for every mimic joint.
            root_path = prim.GetPath().GetParentPath().GetParentPath()
            reference = root_path.AppendPath("joints/gripper_finger1_joint")
            rel = prim.GetRelationship("physxMimicJoint:rotY:referenceJoint")
            if not rel:
                rel = prim.CreateRelationship("physxMimicJoint:rotY:referenceJoint")
            rel.SetTargets([Sdf.Path(reference)])
        found.append(str(prim.GetPath()))

    missing = sorted(set(JOINT_LIMITS_DEG) - {Path(p).name for p in found})
    if missing:
        raise RuntimeError(f"Joints not found in {path}: {', '.join(missing)}")
    root.Save()
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("usd", type=Path)
    args = parser.parse_args()
    for path in [args.usd]:
        print(f"patching {path}")
        for prim_path in patch_asset(path):
            print(f"  patched {prim_path}")


if __name__ == "__main__":
    try:
        main()
    finally:
        if _PATCH_APP is not None:
            _PATCH_APP.close()
