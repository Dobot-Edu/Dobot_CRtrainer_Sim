"""Run the trained CR5A ACT policy in Isaac Sim through LeRobot's gRPC server."""

from __future__ import annotations

import argparse
import multiprocessing
import os
import time
from pathlib import Path


if multiprocessing.get_start_method(allow_none=True) != "spawn":
    multiprocessing.set_start_method("spawn", force=True)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("CR5_SIM_ROOT", str(PROJECT_ROOT))
os.environ.setdefault("LEISAAC_ASSETS_ROOT", str(PROJECT_ROOT / "assets"))


from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Evaluate the CR5A ACT policy in Isaac Sim")
parser.add_argument("--task", default="LeIsaac-CR5A-PickOrange-v0")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--step_hz", type=int, default=30)
parser.add_argument("--episode_length_s", type=float, default=120.0)
parser.add_argument(
    "--eval_rounds",
    type=int,
    default=0,
    help="Number of episodes. Zero runs until the window closes.",
)
parser.add_argument("--policy_host", default="127.0.0.1")
parser.add_argument("--policy_port", type=int, default=5555)
parser.add_argument("--policy_type", choices=["act"], default="act")
parser.add_argument("--policy_device", default="cuda")
parser.add_argument("--policy_action_horizon", type=int, default=10)
parser.add_argument("--policy_rpc_timeout_s", type=float, default=5.0)
parser.add_argument("--policy_setup_timeout_s", type=float, default=180.0)
parser.add_argument("--policy_checkpoint_path", default=None)
parser.add_argument(
    "--policy_language_instruction",
    default="Pick all three oranges and place them on the plate",
    help="Retained for protocol compatibility; ACT does not consume language.",
)
parser.add_argument(
    "--max_arm_step",
    type=float,
    default=0.30,
    help="Maximum change of one arm target per control step in radians; <=0 disables it.",
)
parser.add_argument(
    "--max_gripper_step",
    type=float,
    default=0.35,
    help="Maximum change of the AG95 driver target per control step; <=0 disables it.",
)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(enable_cameras=True, device="cpu")
args_cli = parser.parse_args()

if args_cli.step_hz <= 0:
    parser.error("--step_hz must be positive")
if args_cli.policy_action_horizon <= 0:
    parser.error("--policy_action_horizon must be positive")
if not args_cli.policy_checkpoint_path:
    parser.error("--policy_checkpoint_path is required")


app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app


import carb
import gymnasium as gym
import omni
import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils import parse_env_cfg

import leisaac.tasks  # noqa: F401 - registers the Gym task
from leisaac.assets.robots.cr5a import CR5A_CONTROL_JOINTS
from leisaac.policy import CR5ALeRobotPolicyClient


class RateLimiter:
    def __init__(self, hz: int) -> None:
        self.period = 1.0 / hz
        self.last_time = time.monotonic()

    def reset(self) -> None:
        self.last_time = time.monotonic()

    def sleep(self, env: ManagerBasedRLEnv) -> None:
        deadline = self.last_time + self.period
        while time.monotonic() < deadline:
            time.sleep(min(0.01, deadline - time.monotonic()))
            env.sim.render()
        self.last_time = max(deadline, time.monotonic())


class ResetController:
    def __init__(self) -> None:
        app_window = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = app_window.get_keyboard()
        self._subscription = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            self._on_keyboard_event,
        )
        self.requested = False

    def _on_keyboard_event(self, event, *_) -> bool:
        if event.type == carb.input.KeyboardEventType.KEY_PRESS and event.input.name == "R":
            self.requested = True
        return True

    def consume(self) -> bool:
        requested = self.requested
        self.requested = False
        return requested

    def close(self) -> None:
        if self._subscription is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._subscription)
            self._subscription = None


def _control_joint_ids(env: ManagerBasedRLEnv) -> list[int]:
    names = env.scene["robot"].data.joint_names
    missing = [name for name in CR5A_CONTROL_JOINTS if name not in names]
    if missing:
        raise RuntimeError(f"Robot is missing policy joints: {missing}")
    return [names.index(name) for name in CR5A_CONTROL_JOINTS]


def _current_control_state(env: ManagerBasedRLEnv, joint_ids: list[int]) -> torch.Tensor:
    return env.scene["robot"].data.joint_pos[:, joint_ids].detach().clone()


def _safe_action_chunk(
    env: ManagerBasedRLEnv,
    actions: torch.Tensor,
    joint_ids: list[int],
) -> torch.Tensor:
    """Validate, joint-limit clamp, and slew-limit one policy action chunk."""

    current = _current_control_state(env, joint_ids)
    hold = current.unsqueeze(0).repeat(args_cli.policy_action_horizon, 1, 1)

    if actions.ndim != 3 or actions.shape[1:] != (1, len(joint_ids)):
        print(f"[Safety] Invalid action shape {tuple(actions.shape)}; holding position.")
        return hold
    actions = actions.to(device=env.device, dtype=torch.float32)
    if not torch.isfinite(actions).all():
        print("[Safety] Policy produced NaN or Inf; holding position.")
        return hold

    robot = env.scene["robot"]
    limits = getattr(robot.data, "soft_joint_pos_limits", None)
    if limits is None:
        lower = torch.full_like(current[0], -torch.inf)
        upper = torch.full_like(current[0], torch.inf)
    else:
        lower = limits[0, joint_ids, 0]
        upper = limits[0, joint_ids, 1]
        lower = torch.where(torch.isfinite(lower), lower, torch.full_like(lower, -torch.inf))
        upper = torch.where(torch.isfinite(upper), upper, torch.full_like(upper, torch.inf))

    max_step = torch.tensor(
        [args_cli.max_arm_step] * 6 + [args_cli.max_gripper_step],
        dtype=torch.float32,
        device=env.device,
    )
    slew_enabled = max_step > 0.0

    safe_steps = []
    previous = current[0]
    for raw_step in actions[:, 0, :]:
        target = torch.minimum(torch.maximum(raw_step, lower), upper)
        delta = target - previous
        bounded_delta = torch.where(
            slew_enabled,
            torch.minimum(torch.maximum(delta, -max_step), max_step),
            delta,
        )
        target = torch.minimum(torch.maximum(previous + bounded_delta, lower), upper)
        safe_steps.append(target)
        previous = target
    return torch.stack(safe_steps, dim=0).unsqueeze(1)


def _policy_observation(observation: dict) -> dict:
    if "policy" not in observation:
        raise KeyError(f"Environment observation has no 'policy' group: {list(observation)}")
    required = {"joint_pos", "wrist", "front"}
    missing = required - observation["policy"].keys()
    if missing:
        raise KeyError(f"Policy observation is missing: {sorted(missing)}")
    return observation["policy"]


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else int(time.time())
    env_cfg.episode_length_s = args_cli.episode_length_s
    env_cfg.recorders = None
    if args_cli.eval_rounds <= 0 and hasattr(env_cfg.terminations, "time_out"):
        env_cfg.terminations.time_out = None

    env: ManagerBasedRLEnv | None = None
    policy: CR5ALeRobotPolicyClient | None = None
    controller: ResetController | None = None
    try:
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        joint_ids = _control_joint_ids(env)
        camera_infos = {
            "wrist": (int(env.cfg.scene.wrist.height), int(env.cfg.scene.wrist.width)),
            "front": (int(env.cfg.scene.front.height), int(env.cfg.scene.front.width)),
        }
        policy = CR5ALeRobotPolicyClient(
            host=args_cli.policy_host,
            port=args_cli.policy_port,
            camera_infos=camera_infos,
            pretrained_name_or_path=args_cli.policy_checkpoint_path,
            policy_type=args_cli.policy_type,
            actions_per_chunk=args_cli.policy_action_horizon,
            device=args_cli.policy_device,
            rpc_timeout_s=args_cli.policy_rpc_timeout_s,
            setup_timeout_s=args_cli.policy_setup_timeout_s,
            task_description=args_cli.policy_language_instruction,
        )
        controller = ResetController()
        rate_limiter = RateLimiter(args_cli.step_hz)

        observation, _ = env.reset()
        policy.reset(_policy_observation(observation)["joint_pos"])
        rate_limiter.reset()

        completed = 0
        successes = 0
        while simulation_app.is_running() and (
            args_cli.eval_rounds <= 0 or completed < args_cli.eval_rounds
        ):
            print(f"[Evaluation] Episode {completed + 1} started. Press R to reset.")
            outcome: str | None = None

            while simulation_app.is_running() and outcome is None:
                if controller.consume():
                    outcome = "manual reset"
                    break

                # Isaac Lab stores tensors produced during env.step() in its
                # asset data buffers.  Running the environment under
                # torch.inference_mode() turns those buffers into inference
                # tensors, which cannot be updated in-place by the next reset.
                # The neural network lives in the separate LeRobot process, so
                # the Isaac client does not need any autograd context here.
                policy_obs = _policy_observation(observation)
                action_chunk = policy.get_action(policy_obs)
                action_chunk = _safe_action_chunk(env, action_chunk, joint_ids)

                for action in action_chunk:
                    if controller.consume():
                        outcome = "manual reset"
                        break
                    observation, _, terminated, timed_out, _ = env.step(action)
                    rate_limiter.sleep(env)
                    if bool(terminated[0]):
                        outcome = "success"
                        break
                    if bool(timed_out[0]):
                        outcome = "timeout"
                        break

            if not simulation_app.is_running():
                break

            completed += 1
            if outcome == "success":
                successes += 1
            print(
                f"[Evaluation] Episode {completed}: {outcome}. "
                f"Success rate {successes}/{completed} = {successes / completed:.1%}"
            )

            if args_cli.eval_rounds > 0 and completed >= args_cli.eval_rounds:
                break
            observation, _ = env.reset()
            # Reset local fallback state and force the next fresh observation to
            # be used.  No action from the preceding episode is reused.
            policy.reset(_policy_observation(observation)["joint_pos"])
            rate_limiter.reset()

        if completed:
            print(f"[Evaluation] Final success rate: {successes}/{completed} = {successes / completed:.1%}")
    finally:
        if controller is not None:
            controller.close()
        if policy is not None:
            policy.close()
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
