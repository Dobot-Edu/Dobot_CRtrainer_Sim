"""Run a minimal CR5A articulation smoke test without tasks or teleoperation."""

from __future__ import annotations

import argparse
import os
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_USD = PROJECT_ROOT / "assets" / "robot" / "cr5a_ag95" / "usd" / "cr5a_ag95_mimic.usd"

parser = argparse.ArgumentParser(description="CR5A articulation smoke test")
parser.add_argument("--usd_path", type=Path, default=DEFAULT_USD)
parser.add_argument("--steps", type=int, default=120)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

usd_path = args.usd_path.expanduser().resolve()
os.environ["CR5_SIM_ROOT"] = str(PROJECT_ROOT)
os.environ["CR5A_AG95_USD_PATH"] = str(usd_path)

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from leisaac.assets.robots.cr5a import CR5A_AG95_CFG


def main() -> None:
    if not usd_path.is_file():
        raise FileNotFoundError(usd_path)
    if args.steps <= 0:
        raise ValueError("--steps must be positive")

    print(f"[CR5_Check] USD: {usd_path}", flush=True)
    sim = SimulationContext(sim_utils.SimulationCfg(device=args.device))
    sim.set_camera_view((1.8, -2.0, 1.4), (0.0, 0.0, 0.8))

    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/ground", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0)
    light_cfg.func("/World/Light", light_cfg)

    robot_cfg = CR5A_AG95_CFG.copy()
    robot_cfg.prim_path = "/World/Robot"
    print("[CR5_Check] Creating Articulation", flush=True)
    robot = Articulation(robot_cfg)
    stage = __import__("omni.usd", fromlist=["get_context"]).get_context().get_stage()
    from pxr import UsdPhysics

    interesting = [
        (
            str(prim.GetPath()),
            prim.GetTypeName(),
            bool(prim.HasAPI(UsdPhysics.RigidBodyAPI)),
        )
        for prim in stage.Traverse()
        if any(token in prim.GetName().lower() for token in ("base", "root_joint", "link6", "gripper"))
    ]
    print(f"[CR5_Check] Relevant prims: {interesting}", flush=True)

    print("[CR5_Check] Before sim.reset()", flush=True)
    sim.reset()
    print("[CR5_Check] After sim.reset()", flush=True)

    joint_names = list(robot.data.joint_names)
    print(f"[CR5_Check] Joint count: {len(joint_names)}", flush=True)
    print(f"[CR5_Check] Joints: {joint_names}", flush=True)

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = torch.zeros_like(joint_pos)
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.reset()

    sim_dt = sim.get_physics_dt()
    for step in range(1, args.steps + 1):
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_dt)
        if step == 1 or step % 30 == 0 or step == args.steps:
            finite = bool(torch.isfinite(robot.data.joint_pos).all().item())
            print(
                f"[CR5_Check] Step {step}/{args.steps}: "
                f"running={simulation_app.is_running()} finite={finite}",
                flush=True,
            )
            if not finite:
                raise RuntimeError("non-finite joint positions detected")

    print("[CR5_Check] PASS: articulation reset and stepping succeeded", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        print("[CR5_Check] FAIL", flush=True)
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
