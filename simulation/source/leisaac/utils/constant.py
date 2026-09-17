import os
from pathlib import Path


def _detect_project_root() -> Path:
    """Resolve the standalone CR5_Sim root without depending on GitPython."""
    configured = os.environ.get("CR5_SIM_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[4]


def _resolve_assets_root() -> str:
    """Return env override if provided, otherwise default assets directory."""
    env_root = os.environ.get("LEISAAC_ASSETS_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve().as_posix()

    return (_detect_project_root() / "assets").resolve().as_posix()


ASSETS_ROOT = _resolve_assets_root()

SINGLE_ARM_JOINT_NAMES = [
    "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"
]
