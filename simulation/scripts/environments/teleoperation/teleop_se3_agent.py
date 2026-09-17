"""Launch the CR5A Isaac Lab task with a keyboard or one physical master hand."""

from __future__ import annotations

import argparse
import multiprocessing
import os
import re
import time
from pathlib import Path

import torch


if multiprocessing.get_start_method(allow_none=True) != "spawn":
    multiprocessing.set_start_method("spawn", force=True)


parser = argparse.ArgumentParser(description="CR5A single-arm teleoperation")
parser.add_argument(
    "--task",
    default=None,
    help="Registered task ID. Omit it to start the lightweight CR5A-only environment.",
)
parser.add_argument(
    "--teleop_device",
    default="cr5a_master",
    choices=["cr5a_master", "cr5a_keyboard", "bi_keyboard"],
)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--sensitivity", type=float, default=1.0)
parser.add_argument("--step_hz", type=int, default=30)
parser.add_argument("--record", action="store_true")
parser.add_argument("--dataset_file", default="../datasets/cr5a_teleop.hdf5")
parser.add_argument(
    "--dataset_layout",
    choices=["per_episode", "single_file"],
    default="per_episode",
    help="Write one task-named HDF5 per episode or keep all demos in one file.",
)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--num_demos", type=int, default=0)
parser.add_argument("--quality", action="store_true")
parser.add_argument(
    "--cr5_root",
    default=None,
    help="Override the CR5_Sim root used for assets and master-hand config.",
)


class RateLimiter:
    def __init__(self, hz: int):
        if hz <= 0:
            raise ValueError("step_hz must be positive")
        self.period = 1.0 / hz
        self.last_time = time.monotonic()

    def sleep(self, env) -> None:
        deadline = self.last_time + self.period
        while time.monotonic() < deadline:
            time.sleep(min(0.01, deadline - time.monotonic()))
            env.sim.render()
        self.last_time = max(deadline, time.monotonic())


def _safe_filename(value: str) -> str:
    """Return a filesystem-safe task name without changing its identity."""
    result = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return result or "episode"


def _next_available_single_file(path: Path) -> Path:
    """Avoid startup failure and overwrite when a legacy dataset exists."""
    if not path.exists():
        return path
    suffix = path.suffix or ".hdf5"
    stem = path.stem if path.suffix else path.name
    for index in range(1_000_000):
        candidate = path.with_name(f"{stem}_{index:06d}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find an available dataset filename beside: {path}")


def _set_manual_termination(env, success: bool) -> None:
    """Mark the current episode before the reset requested by R/N."""
    if not hasattr(env, "termination_manager"):
        return
    from isaaclab.managers import TerminationTermCfg

    value = bool(success)
    env.termination_manager.set_term_cfg(
        "success",
        TerminationTermCfg(
            func=lambda current_env: torch.full(
                (current_env.num_envs,), value, dtype=torch.bool, device=current_env.device
            )
        ),
    )
    env.termination_manager.compute()


def main() -> None:
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    project_root = Path(args.cr5_root or Path(__file__).resolve().parents[4]).resolve()
    os.environ.setdefault("CR5_SIM_ROOT", str(project_root))
    os.environ.setdefault("LEISAAC_ASSETS_ROOT", str(project_root / "assets"))

    app_launcher = AppLauncher(vars(args))
    simulation_app = app_launcher.app
    print(
        "[CR5_Sim] AppLauncher returned "
        f"running={simulation_app.is_running()} "
        f"exiting={simulation_app.is_exiting()}"
    )
    env = None
    device = None
    try:
        import gymnasium as gym
        from isaaclab.managers import DatasetExportMode
        from isaaclab_tasks.utils import parse_env_cfg

        import leisaac.tasks  # noqa: F401
        from leisaac.devices import CR5AKeyboard
        from leisaac.enhance.managers import EnhanceDatasetExportMode, StreamingRecorderManager

        task_name = args.task or "LeIsaac-CR5A-RobotOnly-v0"
        env_cfg = parse_env_cfg(task_name, device=args.device, num_envs=args.num_envs)
        env_cfg.use_teleop_device(args.teleop_device)
        env_cfg.seed = args.seed if args.seed is not None else int(time.time())
        # Teleoperation uses explicit R/N key presses to finish an episode.
        # Leaving the task timeout or automatic success term active would
        # reset the scene without giving the recorder the intended label.
        if hasattr(env_cfg.terminations, "time_out"):
            env_cfg.terminations.time_out = None
        if hasattr(env_cfg.terminations, "success"):
            env_cfg.terminations.success = None
        if args.quality:
            env_cfg.sim.render.antialiasing_mode = "FXAA"
            env_cfg.sim.render.rendering_mode = "quality"

        dataset_path = Path(args.dataset_file).expanduser()
        if not dataset_path.is_absolute():
            dataset_path = (project_root / "simulation" / dataset_path).resolve()
        if args.record:
            if args.dataset_layout == "per_episode":
                if args.num_envs != 1:
                    raise ValueError("per_episode HDF5 recording requires --num_envs=1")
                if args.resume:
                    print("[CR5_Sim] per_episode mode always continues at the next free index; --resume is not needed.")
                env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL
                env_cfg.recorders.dataset_export_dir_path = str(dataset_path.parent)
                env_cfg.recorders.dataset_filename = _safe_filename(task_name)
                print(
                    "[CR5_Sim] Per-episode dataset pattern: "
                    f"{dataset_path.parent / (_safe_filename(task_name) + '_NNNNNN.hdf5')}"
                )
            else:
                if args.resume:
                    if not dataset_path.exists():
                        raise FileNotFoundError(f"--resume requires an existing dataset: {dataset_path}")
                    env_cfg.recorders.dataset_export_mode = EnhanceDatasetExportMode.EXPORT_ALL_RESUME
                else:
                    available_path = _next_available_single_file(dataset_path)
                    if available_path != dataset_path:
                        print(
                            f"[CR5_Sim] Dataset exists; recording to a new file instead: {available_path}"
                        )
                    dataset_path = available_path
                    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL
                env_cfg.recorders.dataset_export_dir_path = str(dataset_path.parent)
                env_cfg.recorders.dataset_filename = dataset_path.stem
            # Keep a live success term so the recorder can distinguish the
            # value set by _set_manual_termination() on the next reset.
            from isaaclab.managers import TerminationTermCfg

            env_cfg.terminations.success = TerminationTermCfg(
                func=lambda current_env: torch.zeros(
                    current_env.num_envs, dtype=torch.bool, device=current_env.device
                )
            )
        else:
            env_cfg.recorders = None

        print(f"[CR5_Sim] Creating task environment: {task_name}")
        env = gym.make(task_name, cfg=env_cfg).unwrapped
        print("[CR5_Sim] Task environment created")
        if args.record:
            del env.recorder_manager
            env.recorder_manager = StreamingRecorderManager(
                env_cfg.recorders,
                env,
                env_name=task_name,
                dataset_layout=args.dataset_layout,
            )
            env.recorder_manager.flush_steps = 100
            env.recorder_manager.compression = "lzf"

        print(f"[CR5_Sim] Before device init: {args.teleop_device}", flush=True)
        if args.teleop_device == "cr5a_master":
            from leisaac.devices import CR5AMaster

            device = CR5AMaster(env, project_root=project_root)
        else:
            device = CR5AKeyboard(env, sensitivity=0.06 * args.sensitivity)
        print("[CR5_Sim] After device init", flush=True)

        reset_requested = False
        success_requested = False

        def request_reset() -> None:
            nonlocal reset_requested
            reset_requested = True

        def request_success() -> None:
            nonlocal reset_requested, success_requested
            success_requested = True
            reset_requested = True

        device.add_callback("R", request_reset)
        device.add_callback("N", request_success)
        print(device)
        print(
            "[CR5_Sim] Before env.initialize "
            f"has_initialize={hasattr(env, 'initialize')}",
            flush=True,
        )
        if hasattr(env, "initialize"):
            env.initialize()
        print("[CR5_Sim] After env.initialize", flush=True)
        print(
            "[CR5_Sim] Before reset "
            f"running={simulation_app.is_running()} "
            f"exiting={simulation_app.is_exiting()}",
            flush=True,
        )
        env.reset()
        device.reset()
        print(
            "[CR5_Sim] Before Kit update "
            f"running={simulation_app.is_running()} "
            f"exiting={simulation_app.is_exiting()}",
            flush=True,
        )
        # Pump one Kit frame after environment construction.  This makes the
        # lifecycle state explicit on Isaac Sim 5.x and avoids entering the
        # teleoperation loop before the application has processed its startup
        # events.
        simulation_app.update()
        print(
            "[CR5_Sim] Entering teleoperation loop "
            f"running={simulation_app.is_running()} "
            f"exiting={simulation_app.is_exiting()}",
            flush=True,
        )
        rate_limiter = RateLimiter(args.step_hz)
        resume_recorded_count = 0
        if args.record and args.resume and args.dataset_layout == "single_file":
            resume_recorded_count = env.recorder_manager._dataset_file_handler.get_num_episodes()
            print(f"Resume recording from existing dataset file with {resume_recorded_count} episodes.")
        recorded_count = resume_recorded_count

        while simulation_app.is_running():
            with torch.inference_mode():
                actions = device.advance()
                if reset_requested:
                    if args.record:
                        _set_manual_termination(env, success_requested)
                    env.reset()
                    if args.record:
                        # N temporarily replaces the live success term with a
                        # constant True so record_pre_reset() labels exactly
                        # this episode as successful.  Restore False after the
                        # reset; otherwise every step of the next episode is a
                        # success termination and creates another HDF5 file.
                        _set_manual_termination(env, False)
                    device.reset()
                    reset_requested = False
                    success_requested = False
                    if args.record and hasattr(env.recorder_manager, "exported_successful_episode_count"):
                        recorded_count = resume_recorded_count + int(
                            env.recorder_manager.exported_successful_episode_count
                        )
                        print(f"Recorded successful episodes: {recorded_count}")
                        if args.num_demos and recorded_count >= args.num_demos:
                            break
                elif actions is None:
                    env.render()
                else:
                    env.step(actions)
            rate_limiter.sleep(env)
    finally:
        if device is not None and hasattr(device, "close"):
            device.close()
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
