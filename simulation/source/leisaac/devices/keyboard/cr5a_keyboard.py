"""Keyboard teleoperation for the single CR5A arm.

The legacy ``BiKeyboard`` emitted a 16-DoF X-Trainer command.  This adapter
keeps the right-arm slice of that command for compatibility with the CR5A task,
while making it explicit that only one arm is simulated.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable

import carb
import numpy as np
import omni

from ..device_base import Device


class CR5AKeyboard(Device):
    """Drive CR5A joint deltas with the right-hand keyboard layout."""

    def __init__(self, env, sensitivity: float = 0.06):
        super().__init__(env)
        self.sensitivity = float(sensitivity)
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        # Omniverse may emit repeated KEY_PRESS events while a key is held.
        # Keep lifecycle keys latched until KEY_RELEASE so one physical press
        # can create at most one reset/episode boundary.
        self._lifecycle_keys_down: set[str] = set()
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )
        self._velocity = np.zeros(16, dtype=np.float32)
        self._state = np.zeros(16, dtype=np.float32)
        self._z_pressed = False
        self.started = False
        self._reset_state = False
        self._additional_callbacks: dict[str, Callable] = {}
        self._key_to_index = {
            "U": 8,
            "I": 9,
            "O": 10,
            "J": 11,
            "K": 12,
            "L": 13,
        }

    def __del__(self):
        try:
            self.close()
        except Exception:
            # Isaac Sim may already be shutting down when the object is
            # collected, so keyboard unsubscription is best-effort here.
            pass

    def close(self):
        input_interface = getattr(self, "_input", None)
        subscription = getattr(self, "_keyboard_sub", None)
        keyboard = getattr(self, "_keyboard", None)
        if input_interface is not None and subscription is not None:
            input_interface.unsubscribe_to_keyboard_events(keyboard, subscription)
            self._keyboard_sub = None

    def __str__(self) -> str:
        return (
            "CR5A keyboard controller (single arm)\n"
            "  J1..J6: U/I/O/J/K/L, reverse: hold Z\n"
            "  gripper: H, start: B, reset: R, success/reset: N"
        )

    def input2action(self):
        reset = self._reset_state
        if reset:
            self._reset_state = False
            return {"reset": True, "started": False, "keyboard": True}
        if not self.started:
            return {"reset": False, "started": False, "keyboard": True}
        self._state += self._velocity
        self._state[8:14] = np.clip(self._state[8:14], -np.pi, np.pi)
        self._state[14:16] = np.clip(self._state[14:16], 0.0, 0.04)
        return {
            "reset": False,
            "started": True,
            "keyboard": True,
            "joint_state": self._state.copy(),
        }

    def reset(self):
        self.started = False
        self._velocity.fill(0.0)
        self._state.fill(0.0)
        self._z_pressed = False

    def add_callback(self, key: str, func: Callable):
        self._additional_callbacks[key] = func

    def _on_keyboard_event(self, event, *args):
        input_value = event.input
        name = input_value if isinstance(input_value, str) else input_value.name
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if name in {"R", "N"}:
                if name in self._lifecycle_keys_down:
                    return True
                self._lifecycle_keys_down.add(name)
            if name == "Z":
                self._z_pressed = True
                return True
            if name == "B":
                self.started = True
                self._reset_state = False
            elif name == "R":
                self.started = False
                self.reset()
                self._reset_state = True
                callback = self._additional_callbacks.get("R")
                if callback:
                    callback()
            elif name == "N":
                self.started = False
                self.reset()
                self._reset_state = True
                callback = self._additional_callbacks.get("N")
                if callback:
                    callback()
            elif name in self._key_to_index:
                direction = -1.0 if self._z_pressed else 1.0
                self._velocity[self._key_to_index[name]] = direction * self.sensitivity
            elif name == "H":
                direction = -1.0 if self._z_pressed else 1.0
                self._velocity[14:16] = direction * self.sensitivity * 0.1
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if name in {"R", "N"}:
                self._lifecycle_keys_down.discard(name)
            elif name == "Z":
                self._z_pressed = False
            elif name in self._key_to_index:
                self._velocity[self._key_to_index[name]] = 0.0
            elif name == "H":
                self._velocity[14:16] = 0.0
        return True
