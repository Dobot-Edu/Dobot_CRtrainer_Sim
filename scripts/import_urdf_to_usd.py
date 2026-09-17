"""Import a CR5A/AG95 URDF into a standalone USD with Isaac Sim.

Run this script with the Isaac Sim Python environment.  The output is written
to a new file; existing USD assets are never overwritten unless the caller
explicitly chooses the same output path.
"""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import URDF to USD via Isaac Sim")
    parser.add_argument("urdf", type=Path, help="Input URDF path")
    parser.add_argument("output", type=Path, help="Output USD path")
    parser.add_argument("--headless", action="store_true", help="Run without opening a viewport")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    urdf_path = args.urdf.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": args.headless})
    try:
        print("[CR5_Sim] Isaac Sim app started; loading URDF importer", flush=True)
        # Isaac Sim 5.0 ships the importer as an optional extension.  Enable it
        # before importing its Python module.
        import omni.kit.app

        extension_manager = omni.kit.app.get_app().get_extension_manager()
        extension_manager.set_extension_enabled_immediate("isaacsim.asset.importer.urdf", True)
        simulation_app.update()
        try:
            from isaacsim.asset.importer.urdf import _urdf
        except ImportError:
            # Compatibility with older Isaac Sim installations.
            try:
                from omni.isaac.urdf import _urdf
            except ImportError:
                from omni.importer.urdf import _urdf
        import omni.usd
        from pxr import Usd

        print(f"[CR5_Sim] URDF importer module: {_urdf.__file__}", flush=True)
        importer = _urdf.acquire_urdf_interface()
        config = _urdf.ImportConfig()
        config.merge_fixed_joints = False
        config.convex_decomp = False
        config.import_inertia_tensor = True
        config.fix_base = True
        config.make_default_prim = True
        config.self_collision = False
        config.create_physics_scene = True
        config.distance_scale = 1.0
        config.density = 0.0
        if hasattr(_urdf, "UrdfJointTargetType"):
            config.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION

        print(f"[CR5_Sim] Importing URDF: {urdf_path}", flush=True)
        asset_root = str(urdf_path.parent)
        asset_name = urdf_path.name
        robot = importer.parse_urdf(asset_root, asset_name, config)
        if robot is None:
            raise RuntimeError(f"Isaac Sim URDF parser failed for {urdf_path}")
        # Isaac Sim 5.0 requires the parsed UrdfRobot plus the asset root/name;
        # passing only a URDF filename is the old API and raises TypeError.
        prim_path = importer.import_robot(
            asset_root,
            asset_name,
            robot,
            config,
            str(output_path),
            False,
        )
        if not prim_path:
            raise RuntimeError(f"Isaac Sim URDF importer failed for {urdf_path}")

        simulation_app.update()
        # The importer writes directly to ``output_path`` and may leave the
        # current anonymous stage unchanged. Re-open the exported file so the
        # default prim is authored on the actual output layer.
        stage = Usd.Stage.Open(str(output_path))
        if stage is None:
            raise RuntimeError(f"Could not open exported USD: {output_path}")
        imported_prim = stage.GetPrimAtPath(prim_path)
        if not imported_prim or not imported_prim.IsValid():
            # Some Isaac Sim 5.0 builds return a path without the /World
            # prefix even though the imported prim is authored below /World.
            imported_prim = stage.GetPrimAtPath(f"/World{prim_path}")
        if not imported_prim or not imported_prim.IsValid():
            stem = urdf_path.stem
            candidates = [
                prim
                for prim in stage.GetPseudoRoot().GetChildren()
                if prim.IsValid() and prim.GetName() in {stem, "World"}
            ]
            if candidates and candidates[0].GetName() == "World":
                children = list(candidates[0].GetChildren())
                imported_prim = children[0] if children else candidates[0]
            elif candidates:
                imported_prim = candidates[0]
        if not imported_prim or not imported_prim.IsValid():
            raise RuntimeError(f"Importer returned invalid prim path: {prim_path}")
        # Isaac Lab references robot USDs through the root layer's defaultPrim.
        # The 5.0 importer does not always author it when a destination path is
        # supplied, which otherwise produces an unresolved <defaultPrim>
        # reference during environment composition.
        stage.SetDefaultPrim(imported_prim)
        root_layer = stage.GetRootLayer()
        root_layer.Save()
        if root_layer.identifier != str(output_path) and not root_layer.Export(str(output_path)):
            raise RuntimeError(f"Could not save USD: {output_path}")
        print(f"[CR5_Sim] Exported USD: {output_path}", flush=True)
        print(f"[CR5_Sim] Imported prim: {prim_path}", flush=True)
    except BaseException:
        print("[CR5_Sim] URDF export failed with exception:", flush=True)
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
