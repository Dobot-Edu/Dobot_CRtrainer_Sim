from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "Dobot_Init"))

from master_teleop.collection.workflow import (  # noqa: E402
    calibration_from_raw,
    create_master,
    format_six,
    update_runtime_master,
    wait_master_sample,
)
from master_teleop.core.config import load_master_project_config  # noqa: E402
from master_teleop.core.sync_trace import (  # noqa: E402
    write_calibration_report,
    write_master_pose_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect or replace the single right master-hand zero offsets."
    )
    parser.add_argument(
        "--config-dir",
        default=str(ROOT / "Dobot_Init" / "master_teleop" / "config"),
    )
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument(
        "--target-deg",
        nargs=6,
        type=float,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        help="Explicit CR5A joint target; defaults to the configured home pose.",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Write new offsets. Without this flag the command is read-only.",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Require typing CALIBRATE before writing the runtime file.",
    )
    parser.add_argument("--no-save", action="store_true", help="Calculate without saving.")
    args = parser.parse_args()
    if args.samples < 3:
        parser.error("--samples must be at least 3")
    if (args.confirm or args.no_save) and not args.calibrate:
        parser.error("--confirm and --no-save require --calibrate")

    config = load_master_project_config(args.config_dir)
    target_deg = [
        float(value)
        for value in (
            args.target_deg
            or config.get("master_hand", {}).get("robot_pose_init", {}).get(
                "home_joint_deg", [90, 0, 90, 0, -90, -90]
            )
        )
    ]
    if len(target_deg) != 6:
        raise ValueError("calibration target must contain six joint angles")

    master = create_master(config)
    try:
        master.connect()
        sample = wait_master_sample(master)
        if master.profile is None:
            raise RuntimeError("master profile was not identified")
        print(f"master profile: {master.profile.name}")
        print(f"target CR5A joints: {format_six(target_deg, 2)} deg")
        raw_rows: list[list[int]] = []
        calibrated_rows: list[list[float]] = []
        for _ in range(args.samples):
            sample = wait_master_sample(master, timeout_s=1.0)
            if len(sample.raw_positions) < 7 or len(sample.joint_positions_rad) != 6:
                raise RuntimeError("master sample is missing six joints or seven raw positions")
            raw_rows.append(list(sample.raw_positions[:7]))
            calibrated_rows.append(list(sample.joint_positions_rad))
            time.sleep(0.02)
        averaged_raw = [statistics.fmean(row[index] for row in raw_rows) for index in range(7)]
        current_rad = [statistics.fmean(row[index] for row in calibrated_rows) for index in range(6)]
        current_deg = [math.degrees(value) for value in current_rad]
        relative_deg = [_wrapped_degree_delta(current, target) for current, target in zip(current_deg, target_deg)]
        print(f"current calibrated master: {format_six(current_deg, 2)} deg")
        print(f"relative to target: {format_six(relative_deg, 2)} deg")

        if not args.calibrate:
            report = write_master_pose_report(
                ROOT / "Data_Collections" / "master_teleop" / "diagnostics",
                profile_name=master.profile.name,
                reference_joint_deg=target_deg,
                current_joint_rad=current_rad,
                relative_joint_deg=relative_deg,
                averaged_raw_positions=averaged_raw,
                sample_count=len(raw_rows),
                loaded_joint_offsets_rad=master.profile.joint_offsets_rad,
                joint_signs=master.profile.joint_signs,
            )
            print(f"read-only diagnostic: {report}")
            return

        print("请保持主手不动，并确认它处于目标零位、夹爪完全张开。")
        if args.confirm and input("输入 CALIBRATE 继续: ").strip() != "CALIBRATE":
            raise SystemExit("Calibration cancelled; no file was changed")
        calibration = calibration_from_raw(
            averaged_raw,
            target_deg,
            master.profile.joint_signs,
            master.config.position_units_per_pi,
        )
        print(f"joint_offsets_rad: {calibration['joint_offsets_rad']}")
        saved_path = None
        if not args.no_save:
            saved_path = update_runtime_master(
                {"side": master.profile.name, "profiles": {master.profile.name: calibration}},
                path=Path(args.config_dir) / "master_runtime.yaml",
            )
            print(f"saved: {saved_path}")
        report = write_calibration_report(
            ROOT / "Data_Collections" / "master_teleop" / "diagnostics",
            profile_name=master.profile.name,
            target_deg=target_deg,
            sample_count=len(raw_rows),
            first_calibrated_rad=calibrated_rows[0],
            averaged_raw_positions=averaged_raw,
            joint_signs=master.profile.joint_signs,
            position_units_per_pi=master.config.position_units_per_pi,
            calibration=calibration,
            saved_config_path=saved_path,
        )
        print(f"calibration diagnostic: {report}")
    finally:
        master.close()


def _wrapped_degree_delta(target: float, reference: float) -> float:
    return (float(target) - float(reference) + 180.0) % 360.0 - 180.0


if __name__ == "__main__":
    main()
