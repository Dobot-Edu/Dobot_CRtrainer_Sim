#!/usr/bin/env python3
"""Convert CR5A Isaac Lab recordings to a local LeRobot dataset.

The input may be either an old aggregate HDF5 file containing multiple
``/data/demo_*`` groups or a directory containing the newer one-file-per-
episode recordings.  Incomplete and failed episodes are skipped by default.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

try:
    import h5py
except ImportError as exc:  # pragma: no cover - environment-dependent
    raise SystemExit("Missing dependency: install h5py in the LeRobot environment.") from exc

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - environment-dependent
    raise SystemExit("Missing dependency: install numpy in the LeRobot environment.") from exc


STATE_KEYS = (
    "obs/policy/joint_pos",
    "obs/joint_pos",
    "joint_pos",
)
ACTION_KEYS = (
    "processed_actions",
    "actions",
    "obs/policy/actions",
)
WRIST_KEYS = (
    "obs/policy/wrist",
    "obs/wrist",
    "wrist",
)
FRONT_KEYS = (
    "obs/policy/front",
    "obs/front",
    "front",
)
CONTROL_AXIS_NAMES = (
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "gripper_finger1_joint",
)


@dataclass(frozen=True)
class EpisodeRef:
    file_path: Path
    demo_name: str
    success: bool
    length: int
    state_key: str
    action_key: str
    wrist_key: str
    front_key: str
    state_width: int
    wrist_shape: tuple[int, int, int]
    front_shape: tuple[int, int, int]

    @property
    def label(self) -> str:
        return f"{self.file_path.name}:{self.demo_name}"


def _natural_key(value: str) -> tuple:
    return tuple(int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value))


def _discover_hdf5_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in {".hdf5", ".h5"}:
            raise ValueError(f"Input file is not HDF5: {input_path}")
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input does not exist: {input_path}")
    files = [*input_path.rglob("*.hdf5"), *input_path.rglob("*.h5")]
    files = sorted(set(files), key=lambda path: _natural_key(str(path.relative_to(input_path))))
    if not files:
        raise FileNotFoundError(f"No .hdf5 or .h5 files found under: {input_path}")
    return files


def _dataset_at(group: h5py.Group, key: str) -> h5py.Dataset | None:
    try:
        value = group[key]
    except KeyError:
        return None
    return value if isinstance(value, h5py.Dataset) else None


def _resolve_dataset_key(group: h5py.Group, candidates: Sequence[str], override: str | None) -> str:
    keys = (override,) if override else candidates
    for key in keys:
        if key and _dataset_at(group, key) is not None:
            return key
    requested = override if override else ", ".join(candidates)
    raise KeyError(f"none of these dataset paths exists: {requested}")


def _parse_indices(value: str) -> tuple[int, ...]:
    try:
        indices = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("indices must be comma-separated integers") from exc
    if len(indices) != 7 or len(set(indices)) != 7 or min(indices, default=-1) < 0:
        raise argparse.ArgumentTypeError("exactly seven distinct non-negative indices are required")
    return indices


def _rgb_shape(shape: tuple[int, ...]) -> tuple[int, int, int]:
    frame_shape = tuple(shape[1:])
    while len(frame_shape) > 3 and frame_shape[0] == 1:
        frame_shape = frame_shape[1:]
    if len(frame_shape) != 3:
        raise ValueError(f"expected frame shape HWC or CHW, got {frame_shape}")
    if frame_shape[-1] in (3, 4):
        height, width, _ = frame_shape
    elif frame_shape[0] in (3, 4):
        _, height, width = frame_shape
    else:
        raise ValueError(f"expected 3 or 4 image channels, got {frame_shape}")
    return int(height), int(width), 3


def _normalize_rgb(frame: np.ndarray) -> np.ndarray:
    image = np.asarray(frame)
    while image.ndim > 3 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 3:
        raise ValueError(f"expected HWC or CHW image, got shape {image.shape}")
    if image.shape[-1] in (3, 4):
        pass
    elif image.shape[0] in (3, 4):
        image = np.transpose(image, (1, 2, 0))
    else:
        raise ValueError(f"expected 3 or 4 image channels, got shape {image.shape}")
    image = image[..., :3]
    if np.issubdtype(image.dtype, np.floating):
        finite_max = float(np.nanmax(image)) if image.size else 0.0
        if finite_max <= 1.0:
            image = image * 255.0
    return np.ascontiguousarray(np.nan_to_num(image, nan=0.0).clip(0, 255).astype(np.uint8))


def _read_success(group: h5py.Group) -> bool | None:
    if "success" not in group.attrs:
        return None
    value = np.asarray(group.attrs["success"])
    if value.size != 1:
        raise ValueError(f"success attribute must contain one value, got shape {value.shape}")
    return bool(value.reshape(-1)[0])


def _inspect_demo(
    file_path: Path,
    demo_name: str,
    group: h5py.Group,
    args: argparse.Namespace,
) -> EpisodeRef:
    success = _read_success(group)
    if success is None:
        raise ValueError("missing success attribute (probably an interrupted episode)")

    state_key = _resolve_dataset_key(group, STATE_KEYS, args.state_key)
    action_key = _resolve_dataset_key(group, ACTION_KEYS, args.action_key)
    wrist_key = _resolve_dataset_key(group, WRIST_KEYS, args.wrist_key)
    front_key = _resolve_dataset_key(group, FRONT_KEYS, args.front_key)
    datasets = {
        "state": group[state_key],
        "action": group[action_key],
        "wrist": group[wrist_key],
        "front": group[front_key],
    }
    lengths = {name: int(dataset.shape[0]) for name, dataset in datasets.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"frame counts differ: {lengths}")
    length = lengths["state"]
    if length <= 0:
        raise ValueError("episode contains no frames")

    state = datasets["state"]
    action = datasets["action"]
    if state.ndim != 2:
        raise ValueError(f"state must be rank 2, got {state.shape}")
    if action.ndim != 2 or action.shape[1] != 7:
        raise ValueError(f"action must have shape (frames, 7), got {action.shape}")
    if state.shape[1] < 7:
        raise ValueError(f"state has fewer than seven values: {state.shape}")
    if state.shape[1] != 7 and max(args.legacy_state_indices) >= state.shape[1]:
        raise ValueError(
            f"legacy state indices {args.legacy_state_indices} do not fit state shape {state.shape}"
        )

    return EpisodeRef(
        file_path=file_path,
        demo_name=demo_name,
        success=success,
        length=length,
        state_key=state_key,
        action_key=action_key,
        wrist_key=wrist_key,
        front_key=front_key,
        state_width=int(state.shape[1]),
        wrist_shape=_rgb_shape(tuple(datasets["wrist"].shape)),
        front_shape=_rgb_shape(tuple(datasets["front"].shape)),
    )


def _scan_episodes(
    files: Sequence[Path], args: argparse.Namespace
) -> tuple[list[EpisodeRef], list[str], int, int, int]:
    accepted: list[EpisodeRef] = []
    anomalies: list[str] = []
    total = 0
    skipped_failed = 0
    skipped_incomplete = 0

    for file_path in files:
        try:
            with h5py.File(file_path, "r") as stream:
                if "data" not in stream or not isinstance(stream["data"], h5py.Group):
                    anomalies.append(f"{file_path.name}: missing /data group")
                    continue
                data_group = stream["data"]
                demo_names = sorted(data_group.keys(), key=_natural_key)
                if not demo_names:
                    anomalies.append(f"{file_path.name}: /data contains no demos")
                    continue
                for demo_name in demo_names:
                    total += 1
                    value = data_group[demo_name]
                    if not isinstance(value, h5py.Group):
                        anomalies.append(f"{file_path.name}:{demo_name}: not a group")
                        continue
                    try:
                        success = _read_success(value)
                        if success is None:
                            skipped_incomplete += 1
                            anomalies.append(
                                f"{file_path.name}:{demo_name}: missing success attribute; skipped as incomplete"
                            )
                            continue
                        if not success and not args.include_failed:
                            skipped_failed += 1
                            continue
                        accepted.append(_inspect_demo(file_path, demo_name, value, args))
                    except (KeyError, TypeError, ValueError) as exc:
                        anomalies.append(f"{file_path.name}:{demo_name}: {exc}")
        except OSError as exc:
            anomalies.append(f"{file_path.name}: cannot open HDF5: {exc}")

    if accepted:
        expected_wrist = accepted[0].wrist_shape
        expected_front = accepted[0].front_shape
        shape_valid: list[EpisodeRef] = []
        for episode in accepted:
            if episode.wrist_shape != expected_wrist or episode.front_shape != expected_front:
                anomalies.append(
                    f"{episode.label}: camera size differs; wrist={episode.wrist_shape}, "
                    f"front={episode.front_shape}, expected wrist={expected_wrist}, front={expected_front}"
                )
            else:
                shape_valid.append(episode)
        accepted = shape_valid

    return accepted, anomalies, total, skipped_failed, skipped_incomplete


def _guarded_remove_output(output: Path) -> None:
    resolved = output.resolve()
    protected = {Path.cwd().resolve(), Path.home().resolve(), Path(resolved.anchor)}
    if resolved in protected or len(resolved.parts) < 3:
        raise ValueError(f"refusing to remove unsafe output path: {resolved}")
    shutil.rmtree(resolved)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="HDF5 file or directory of episode HDF5 files")
    parser.add_argument("--output", required=True, type=Path, help="local LeRobot dataset directory")
    parser.add_argument("--repo-id", default="local/cr5a_pick_orange", help="LeRobot repository identifier")
    parser.add_argument("--task", default="Pick the oranges and place them on the plate.")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--include-failed", action="store_true", help="also convert episodes with success=False")
    parser.add_argument("--strict", action="store_true", help="stop before conversion if any malformed episode is found")
    parser.add_argument("--overwrite", action="store_true", help="remove the exact output directory if it already exists")
    parser.add_argument("--no-video", action="store_true", help="store PNG image sequences instead of MP4 videos")
    parser.add_argument("--image-writer-threads", type=int, default=8)
    parser.add_argument(
        "--legacy-state-indices",
        type=_parse_indices,
        default=(0, 1, 2, 3, 4, 5, 6),
        help="seven indices selected from legacy state vectors wider than 7 (default: 0,1,2,3,4,5,6)",
    )
    parser.add_argument("--state-key", help="override the HDF5 state dataset path inside each demo")
    parser.add_argument("--action-key", help="override the HDF5 action dataset path inside each demo")
    parser.add_argument("--wrist-key", help="override the HDF5 wrist RGB dataset path inside each demo")
    parser.add_argument("--front-key", help="override the HDF5 front RGB dataset path inside each demo")
    return parser


def _print_scan_report(
    files: Sequence[Path],
    accepted: Sequence[EpisodeRef],
    anomalies: Sequence[str],
    total: int,
    skipped_failed: int,
    skipped_incomplete: int,
) -> None:
    print("\nHDF5 scan summary")
    print(f"  files:              {len(files)}")
    print(f"  demos found:        {total}")
    print(f"  demos selected:     {len(accepted)}")
    print(f"  failed skipped:     {skipped_failed}")
    print(f"  incomplete skipped: {skipped_incomplete}")
    print(f"  selected frames:    {sum(episode.length for episode in accepted)}")
    if anomalies:
        print(f"  warnings/errors:    {len(anomalies)}")
        for message in anomalies:
            print(f"    - {message}")


def main() -> int:
    args = _build_parser().parse_args()
    if args.fps <= 0:
        raise SystemExit("--fps must be positive")
    if args.image_writer_threads < 0:
        raise SystemExit("--image-writer-threads cannot be negative")

    input_path = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    try:
        files = _discover_hdf5_files(input_path)
        accepted, anomalies, total, skipped_failed, skipped_incomplete = _scan_episodes(files, args)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    _print_scan_report(files, accepted, anomalies, total, skipped_failed, skipped_incomplete)

    if args.strict and anomalies:
        raise SystemExit("Strict mode: conversion stopped because the scan found malformed episodes.")
    if not accepted:
        raise SystemExit("No valid episodes selected; no LeRobot dataset was created.")

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise SystemExit(
            "LeRobot is not installed. From this repository, run: pip install -e ./lerobot-main"
        ) from exc

    source_files_in_output = []
    for file_path in files:
        try:
            file_path.relative_to(output)
        except ValueError:
            continue
        source_files_in_output.append(file_path)
    if source_files_in_output:
        raise SystemExit(
            "Output directory contains input HDF5 files; refusing to risk deleting source data: "
            f"{output}"
        )
    if output.exists():
        if not args.overwrite:
            raise SystemExit(f"Output already exists: {output}. Choose another path or pass --overwrite.")
        try:
            _guarded_remove_output(output)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

    image_dtype = "image" if args.no_video else "video"
    wrist_shape = accepted[0].wrist_shape
    front_shape = accepted[0].front_shape
    axes = {"axes": list(CONTROL_AXIS_NAMES)}
    features = {
        "observation.state": {"dtype": "float32", "shape": (7,), "names": axes},
        "action": {"dtype": "float32", "shape": (7,), "names": axes},
        "observation.images.wrist": {
            "dtype": image_dtype,
            "shape": wrist_shape,
            "names": ["height", "width", "channels"],
        },
        "observation.images.front": {
            "dtype": image_dtype,
            "shape": front_shape,
            "names": ["height", "width", "channels"],
        },
    }
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        root=output,
        robot_type="cr5a_ag95",
        features=features,
        use_videos=not args.no_video,
        image_writer_threads=args.image_writer_threads,
    )

    wide_state_widths: set[int] = set()
    converted_frames = 0
    converted_episodes = 0
    try:
        for episode in accepted:
            with h5py.File(episode.file_path, "r") as stream:
                group = stream["data"][episode.demo_name]
                state_data = group[episode.state_key]
                action_data = group[episode.action_key]
                wrist_data = group[episode.wrist_key]
                front_data = group[episode.front_key]
                for frame_index in range(episode.length):
                    state = np.asarray(state_data[frame_index], dtype=np.float32).reshape(-1)
                    if episode.state_width != 7:
                        wide_state_widths.add(episode.state_width)
                        state = state[list(args.legacy_state_indices)]
                    action = np.asarray(action_data[frame_index], dtype=np.float32).reshape(7)
                    dataset.add_frame(
                        {
                            "observation.state": state,
                            "action": action,
                            "observation.images.wrist": _normalize_rgb(wrist_data[frame_index]),
                            "observation.images.front": _normalize_rgb(front_data[frame_index]),
                            "task": args.task,
                        }
                    )
                dataset.save_episode()
            converted_episodes += 1
            converted_frames += episode.length
            print(
                f"[{converted_episodes}/{len(accepted)}] converted {episode.label} "
                f"({episode.length} frames)"
            )
    finally:
        dataset.finalize()

    if wide_state_widths:
        print(
            "Legacy state vectors were reduced to seven controls using indices "
            f"{args.legacy_state_indices}; source widths: {sorted(wide_state_widths)}"
        )
    print("\nConversion complete")
    print(f"  output:   {output}")
    print(f"  episodes: {converted_episodes}")
    print(f"  frames:   {converted_frames}")
    print(f"  fps:      {args.fps}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
