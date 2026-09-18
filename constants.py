"""Single source of truth for the CR5_Sim launcher.

Edit the values in this file, then run ``source ./activate_sim.sh`` on Linux.
Paths are relative to the repository root unless they are absolute.
"""

# Canonical CR5A + AG95 asset.  Alternative USD experiments are archived
# under backup/ and are not selectable through the normal launcher.
usd = "assets/robot/cr5a_ag95/usd/cr5a_ag95_mimic.usd"

# "cr5a_master" for the physical master hand, or "keyboard" for keyboard input.
teleop_mode = "keyboard"

# Set True to export HDF5 demonstrations.
save_mode = True

# "only_robot" is the lightweight validation task; "pick_orange" loads the kitchen.
task = "only_robot"

# Stable 5090D baseline.  Change only for an intentional CUDA re-test.
device = "cpu"
num_envs = 1
enable_cameras = True

# Used only when save_mode is True.
dataset_file = "datasets/cr5a_teleop.hdf5"

# "per_episode" writes one task-name-plus-index HDF5 per R/N boundary.
# "single_file" keeps the legacy /data/demo_0, demo_1, ... layout.
dataset_layout = "per_episode"
