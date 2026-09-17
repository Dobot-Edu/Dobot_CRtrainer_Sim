#!/usr/bin/env bash

# Unified CR5A launcher.  It is safe to source: a failed Isaac Sim process
# prints its status and returns to the caller's shell instead of closing it.
set -u

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$ROOT/constants.py"
PYTHON_BIN="${CR5_SIM_PYTHON:-python}"

if [[ ! -f "$CONFIG_FILE" ]]; then
  printf '[CR5_Sim] Missing configuration: %s\n' "$CONFIG_FILE" >&2
  return 2 2>/dev/null || exit 2
fi

# Read and validate the small, trusted Python configuration without sourcing
# Python syntax into Bash.  Values are shell-quoted by shlex.quote().
if ! CONFIG_VALUES="$("$PYTHON_BIN" - "$CONFIG_FILE" <<'PY'
import runpy
import shlex
import sys
from pathlib import Path

cfg = runpy.run_path(sys.argv[1])
required = ("usd", "teleop_mode", "save_mode", "task")
missing = [name for name in required if name not in cfg]
if missing:
    raise SystemExit("constants.py missing: " + ", ".join(missing))

teleop_mode = cfg["teleop_mode"]
task = cfg["task"]
if teleop_mode not in {"cr5a_master", "keyboard"}:
    raise SystemExit("teleop_mode must be cr5a_master or keyboard")
if task not in {"only_robot", "pick_orange"}:
    raise SystemExit("task must be only_robot or pick_orange")
if not isinstance(cfg["save_mode"], bool):
    raise SystemExit("save_mode must be True or False")
dataset_layout = cfg.get("dataset_layout", "per_episode")
if dataset_layout not in {"per_episode", "single_file"}:
    raise SystemExit("dataset_layout must be per_episode or single_file")

values = {
    "USD": cfg["usd"],
    "TELEOP_MODE": teleop_mode,
    "SAVE_MODE": "1" if cfg["save_mode"] else "0",
    "TASK": task,
    "DEVICE": cfg.get("device", "cpu"),
    "NUM_ENVS": cfg.get("num_envs", 1),
    "ENABLE_CAMERAS": "1" if cfg.get("enable_cameras", True) else "0",
    "DATASET_FILE": cfg.get("dataset_file", "datasets/cr5a_teleop.hdf5"),
    "DATASET_LAYOUT": dataset_layout,
}
for key, value in values.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
)"; then
  printf '[CR5_Sim] Could not read %s\n' "$CONFIG_FILE" >&2
  return 2 2>/dev/null || exit 2
fi
eval "$CONFIG_VALUES"

export CR5_SIM_ROOT="$ROOT"
export LEISAAC_ASSETS_ROOT="$ROOT/assets"
export PYTHONPATH="$ROOT/simulation/source:$ROOT/Dobot_Init:${PYTHONPATH:-}"
export DISPLAY="${DISPLAY:-:0}"

if [[ "$USD" = /* ]]; then
  USD_PATH="$USD"
else
  USD_PATH="$ROOT/$USD"
fi
if [[ ! -f "$USD_PATH" ]]; then
  printf '[CR5_Sim] USD does not exist: %s\n' "$USD_PATH" >&2
  return 2 2>/dev/null || exit 2
fi
export CR5A_AG95_USD_PATH="$USD_PATH"

case "$TELEOP_MODE" in
  cr5a_master) TELEOP_DEVICE="cr5a_master" ;;
  keyboard) TELEOP_DEVICE="cr5a_keyboard" ;;
esac
case "$TASK" in
  only_robot) TASK_ID="LeIsaac-CR5A-RobotOnly-v0" ;;
  pick_orange) TASK_ID="LeIsaac-CR5A-PickOrange-v0" ;;
esac

ARGS=(
  "--task=$TASK_ID"
  "--teleop_device=$TELEOP_DEVICE"
  "--num_envs=$NUM_ENVS"
  "--device=$DEVICE"
)
if [[ "$ENABLE_CAMERAS" = "1" || "$ENABLE_CAMERAS" = "True" || "$ENABLE_CAMERAS" = "true" ]]; then
  ARGS+=("--enable_cameras")
fi
if [[ "$SAVE_MODE" = "1" ]]; then
  if [[ "$DATASET_FILE" = /* ]]; then
    DATASET_PATH="$DATASET_FILE"
  else
    DATASET_PATH="$ROOT/$DATASET_FILE"
  fi
  mkdir -p "$(dirname -- "$DATASET_PATH")"
    ARGS+=("--record" "--dataset_file=$DATASET_PATH" "--dataset_layout=$DATASET_LAYOUT")
fi

printf '[CR5_Sim] USD: %s\n' "$USD_PATH"
printf '[CR5_Sim] Task: %s | teleop: %s | device: %s | record: %s | dataset: %s\n' \
  "$TASK_ID" "$TELEOP_DEVICE" "$DEVICE" "$([[ "$SAVE_MODE" = "1" ]] && echo on || echo off)" "$DATASET_LAYOUT"

if [[ "${CR5_SIM_DRY_RUN:-0}" = "1" ]]; then
  printf '[CR5_Sim] Dry run; would execute:'
  printf ' %q' "$PYTHON_BIN" "$ROOT/simulation/scripts/environments/teleoperation/teleop_se3_agent.py" "${ARGS[@]}"
  printf '\n'
  export CR5_SIM_LAST_EXIT_CODE=0
  if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    return 0
  fi
  exit 0
fi

"$PYTHON_BIN" "$ROOT/simulation/scripts/environments/teleoperation/teleop_se3_agent.py" "${ARGS[@]}"
status=$?
printf '\n[CR5_Sim] Isaac Sim process exited with code %d.\n' "$status"
export CR5_SIM_LAST_EXIT_CODE="$status"
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  return 0
fi
exit "$status"
