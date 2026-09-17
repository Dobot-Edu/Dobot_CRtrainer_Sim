"""Validate that the files required by the CR5A task are real assets."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROBOT_ASSET_ROOT = ROOT / "assets" / "robot" / "cr5a_ag95"
REQUIRED = (
    ROBOT_ASSET_ROOT / "usd" / "cr5a_ag95_mimic.usd",
    ROBOT_ASSET_ROOT / "urdf" / "xtrainer_cr5a_ag95_mimic.urdf",
    ROOT / "assets" / "scenes" / "kitchen_with_orange" / "scene.usd",
)


def main() -> int:
    missing = [path for path in REQUIRED if not path.exists()]
    empty = [path for path in REQUIRED if path.exists() and path.stat().st_size == 0]
    # A non-empty top-level USD is not sufficient: its configuration layers
    # and referenced meshes must also be present. Empty Git/LFS placeholders
    # otherwise make Isaac Sim fail during scene construction with little
    # indication of the real cause.
    dependency_files = [
        *ROBOT_ASSET_ROOT.glob("usd/**/*.usd"),
        *ROBOT_ASSET_ROOT.glob("meshes/**/*.stl"),
        *ROBOT_ASSET_ROOT.glob("meshes/**/*.STL"),
    ]
    empty_dependencies = [path for path in dependency_files if path.stat().st_size == 0]
    if missing or empty or empty_dependencies:
        print("CR5A asset check: FAIL")
        for path in missing:
            print(f"  missing: {path}")
        for path in empty:
            print(f"  empty placeholder: {path}")
        for path in empty_dependencies:
            print(f"  empty dependency: {path}")
        print("请从完整的 XTrainer_CR5A_IsaacSim5/Dobot_Sim 资产目录补齐文件后重试。")
        return 1
    print("CR5A asset check: PASS")
    for path in REQUIRED:
        print(f"  {path.relative_to(ROOT)} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
