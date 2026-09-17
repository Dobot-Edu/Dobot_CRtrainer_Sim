"""Teleoperation devices shipped with the CR5A-only simulation."""

from .device_base import Device, DeviceBase
from .keyboard.cr5a_keyboard import CR5AKeyboard

__all__ = ["Device", "DeviceBase", "CR5AKeyboard", "CR5AMaster"]


def __getattr__(name):
    # Hardware support imports omni.appwindow, which is unavailable in the
    # headless Kit. Load it only when a physical master is actually selected.
    if name == "CR5AMaster":
        from .lerobot.cr5a_master import CR5AMaster

        return CR5AMaster
    raise AttributeError(name)
