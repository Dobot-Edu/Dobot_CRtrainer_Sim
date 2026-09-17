import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(vars(args))
simulation_app = app_launcher.app

from omni.usd import get_context

usd_path = Path(
    "/home/dobot/dsw_ws/CR5_Sim/assets/robot/cr5a_ag95/usd/cr5a_ag95_mimic.usd"
).resolve()

print(f"Opening USD: {usd_path}")
get_context().open_stage(str(usd_path))

while simulation_app.is_running():
    simulation_app.update()

simulation_app.close()
