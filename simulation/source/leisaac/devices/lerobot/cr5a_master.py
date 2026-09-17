from collections.abc import Callable
from pathlib import Path
import os
import sys
import time

import carb
import numpy as np
import omni.appwindow

from ..device_base import Device


class CR5AMaster(Device):
    """Calibrated single right X-Trainer master for CR5A joint teleoperation."""

    def __init__(self, env, project_root: str | Path | None = None):
        super().__init__(env)
        project = Path(project_root or os.environ.get("CR5_SIM_ROOT") or Path(__file__).resolve().parents[5])
        init_root = project / "Dobot_Init"
        if not init_root.exists():
            raise FileNotFoundError(f"CR5_Sim Dobot_Init directory not found: {init_root}")
        sys.path.insert(0, str(init_root))

        from master_teleop.collection.workflow import (
            create_master,
            probe_master,
            serial_candidates,
            wait_master_sample,
        )
        from master_teleop.core.config import load_master_project_config
        from master_teleop.control.button_controller import MasterButtonController

        try:
            config = load_master_project_config(str(init_root / "master_teleop" / "config"))
            master = create_master(config)
        except (FileNotFoundError, ValueError) as exc:
            raise RuntimeError(
                "Master-hand configuration is incomplete. Run "
                "Data_Collections/master_teleop/collection/1_find_port.py first."
            ) from exc
        try:
            master.connect()
            wait_master_sample(master)
        except Exception as configured_error:
            master.close()
            values = config["master_hand"]
            master = None
            last_error = configured_error
            for device in serial_candidates([values.get("device", "")]):
                for baud in values.get("baud_rate_candidates", [values.get("baud_rate", 2000000)]):
                    try:
                        master, _ = probe_master(values, device, int(baud), "right")
                        break
                    except Exception as exc:
                        last_error = exc
                if master is not None:
                    break
            if master is None:
                raise RuntimeError(f"No calibrated right master found: {last_error}") from last_error

        if master.profile is None or master.profile.name != "right":
            master.close()
            raise RuntimeError("The connected master is not the calibrated right master")
        self.master = master
        self._diagnostic_only = os.environ.get("CR5A_MASTER_DIAGNOSTIC", "0") == "1"
        self._selftest_hold = os.environ.get("CR5A_MASTER_SELFTEST_HOLD", "0") == "1"
        button_cfg = config.get("master_hand", {}).get("buttons", {})
        self.buttons = MasterButtonController(
            short_press_max_s=float(button_cfg.get("short_press_max_s", 0.5)),
            long_press_min_s=float(button_cfg.get("long_press_min_s", 1.0)),
            initial_locked=bool(button_cfg.get("initial_locked", True)),
        )
        self.master.set_locked(self.buttons.state.master_locked)
        self._started = False
        self._reference_pending = False
        self._master_reference = None
        self._robot_reference = None
        # The master-hand driver exposes a normalized continuous gripper value.
        # Keep analog mapping as the default so the simulated AG95 follows the
        # trigger position; binary hysteresis remains available for old setups.
        self._gripper_closed_state = False
        gripper_cfg = config.get("master_hand", {}).get("gripper", {})
        self._gripper_mode = str(gripper_cfg.get("mode", "analog")).strip().lower()
        if self._gripper_mode not in {"analog", "binary"}:
            raise ValueError("master_hand.gripper.mode must be analog or binary")
        self._gripper_close_threshold = float(gripper_cfg.get("close_threshold", 0.50))
        self._gripper_open_threshold = float(gripper_cfg.get("open_threshold", 0.20))
        names = list(env.scene["robot"].data.joint_names)
        self._arm_joint_indices = [names.index(f"joint{i}") for i in range(1, 7)]
        print(f"[CR5A master] articulation joint order={names}")
        print(f"[CR5A master] CR5A joint indices={self._arm_joint_indices}")
        arm_term = env.action_manager._terms.get("arm_action")
        if arm_term is not None:
            action_ids = list(arm_term._joint_ids)
            print(f"[CR5A master] arm action indices={action_ids}")
            print(f"[CR5A master] arm action order={[names[index] for index in action_ids]}")
        self._reset_state = False
        self._additional_callbacks = {}
        # Do not turn an OS/Kit key-repeat sequence into many one-frame demos.
        # This latch intentionally survives device.reset(); only KEY_RELEASE
        # arms the corresponding lifecycle key again.
        self._lifecycle_keys_down: set[str] = set()
        self._closed = False

        appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = appwindow.get_keyboard()
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)
        print(f"[CR5A master] connected profile={master.profile.name}; press B to enable")

    def _on_keyboard_event(self, event, *args):
        input_value = event.input
        name = input_value if isinstance(input_value, str) else input_value.name
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if name in {"R", "N"}:
                if name in self._lifecycle_keys_down:
                    return True
                self._lifecycle_keys_down.add(name)
            if name == "B":
                self._started = True
                self._reset_state = False
                self._reference_pending = True
            elif name == "R":
                self._started = False
                self._reset_state = True
                if "R" in self._additional_callbacks:
                    self._additional_callbacks["R"]()
            elif name == "N":
                self._started = False
                self._reset_state = True
                if "N" in self._additional_callbacks:
                    self._additional_callbacks["N"]()
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE and name in {"R", "N"}:
            self._lifecycle_keys_down.discard(name)
        return True

    def input2action(self):
        if self._selftest_hold:
            return {
                "reset": False,
                "started": True,
                "cr5a_master": True,
                "cr5a_selftest_hold": True,
                "joint_state": np.zeros(6, dtype=np.float32),
                "gripper_closed": np.float32(0.0),
            }
        if self._diagnostic_only:
            return {"reset": False, "started": False, "cr5a_master": True}
        reset = self._reset_state
        if reset:
            self._reset_state = False
            return {"reset": True, "started": False, "cr5a_master": True}
        if not self._started:
            return {"reset": False, "started": False, "cr5a_master": True}
        sample = self.master.latest()
        if sample is None or not sample.tracking_valid:
            return {"reset": False, "started": False, "cr5a_master": True}
        master_joints = np.asarray(sample.joint_positions_rad, dtype=np.float32)
        if self._reference_pending:
            self._master_reference = master_joints.copy()
            self._robot_reference = (
                self.env.scene["robot"].data.joint_pos[0, self._arm_joint_indices]
                .detach().cpu().numpy().astype(np.float32)
            )
            self._reference_pending = False
            print(f"[CR5A master] clutch reference master={self._master_reference.tolist()}")
            print(f"[CR5A master] clutch reference robot={self._robot_reference.tolist()}")
        button_state = self.buttons.update(sample, time.monotonic())
        for event in button_state.events:
            if event.action == "master_lock":
                self.master.set_locked(bool(event.value))
                print(f"[CR5A master] {'LOCKED' if event.value else 'UNLOCKED'}")
        if button_state.master_locked or self._master_reference is None:
            return {"reset": False, "started": False, "cr5a_master": True}
        delta = np.arctan2(
            np.sin(master_joints - self._master_reference),
            np.cos(master_joints - self._master_reference),
        )
        target_joints = self._robot_reference + delta
        trigger_pressed = float(np.clip(1.0 - float(sample.gripper_open), 0.0, 1.0))
        if self._gripper_mode == "binary":
            if trigger_pressed >= self._gripper_close_threshold:
                self._gripper_closed_state = True
            elif trigger_pressed <= self._gripper_open_threshold:
                self._gripper_closed_state = False
            gripper_value = float(self._gripper_closed_state)
        else:
            gripper_value = trigger_pressed
        return {
            "reset": False,
            "started": True,
            "cr5a_master": True,
            "joint_state": target_joints.astype(np.float32),
            # Kept under the existing field name for task compatibility; the
            # value is continuous in analog mode and binary only in legacy mode.
            "gripper_closed": np.float32(gripper_value),
        }

    def reset(self):
        self._reset_state = False

    def close(self):
        """Release the leader torque and the Isaac Sim keyboard subscription."""
        if self._closed:
            return
        self._closed = True
        subscription = getattr(self, "_keyboard_sub", None)
        if subscription is not None:
            try:
                self._input.unsubscribe_to_keyboard_events(self._keyboard, subscription)
            finally:
                self._keyboard_sub = None
        master = getattr(self, "master", None)
        if master is not None:
            master.close()

    def add_callback(self, key: str, func: Callable):
        self._additional_callbacks[key] = func

    def __del__(self):
        try:
            self.close()
        except Exception:
            # Destructors run during interpreter shutdown, when SDK cleanup
            # errors cannot be reported or recovered from reliably.
            pass
