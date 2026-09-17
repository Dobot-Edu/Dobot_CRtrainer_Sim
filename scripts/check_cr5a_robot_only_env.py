"""Smoke-test the RobotOnly Isaac Lab environment without teleoperation devices."""

from __future__ import annotations

import argparse
import os
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_USD = PROJECT_ROOT / "assets" / "robot" / "cr5a_ag95" / "usd" / "cr5a_ag95_mimic.usd"

parser = argparse.ArgumentParser(description="CR5A RobotOnly environment smoke test")
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
import gymnasium as gym
import leisaac.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main() -> None:
    if not usd_path.is_file():
        raise FileNotFoundError(usd_path)
    if args.steps <= 0:
        raise ValueError("--steps must be positive")

    task_name = "LeIsaac-CR5A-RobotOnly-v0"
    print(f"[CR5_EnvCheck] USD: {usd_path}", flush=True)
    print("[CR5_EnvCheck] Parsing environment config", flush=True)
    env_cfg = parse_env_cfg(task_name, device=args.device, num_envs=1)
    env_cfg.use_teleop_device("cr5a_keyboard")
    env_cfg.recorders = None

    print("[CR5_EnvCheck] Before gym.make", flush=True)
    env = gym.make(task_name, cfg=env_cfg).unwrapped
    print("[CR5_EnvCheck] After gym.make", flush=True)

    print(f"[CR5_EnvCheck] Action space: {env.action_space}", flush=True)
    if env.action_manager.total_action_dim != 7:
        raise RuntimeError(
            f"expected 7 action dimensions (6 arm + 1 gripper), got "
            f"{env.action_manager.total_action_dim}"
        )
    env.reset()
    print("[CR5_EnvCheck] After env.reset", flush=True)

    action = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
    print(f"[CR5_EnvCheck] Action shape: {tuple(action.shape)}", flush=True)
    for step in range(1, args.steps + 1):
        env.step(action)
        if step == 1 or step % 30 == 0 or step == args.steps:
            finite = bool(torch.isfinite(env.scene["robot"].data.joint_pos).all().item())
            print(
                f"[CR5_EnvCheck] Step {step}/{args.steps}: "
                f"running={simulation_app.is_running()} finite={finite}",
                flush=True,
            )
            if not finite:
                raise RuntimeError("non-finite joint positions detected")

    print("[CR5_EnvCheck] PASS: RobotOnly env reset and stepping succeeded", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        print("[CR5_EnvCheck] FAIL", flush=True)
        traceback.print_exc()
        raise
    finally:
        if "env" in globals() and globals()["env"] is not None:
            globals()["env"].close()
        simulation_app.close()
