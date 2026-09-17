# master_teleop

这里放单主手模式的发现、零位标定和按钮检查脚本。仿真 HDF5 数据不放在这里，而统一写入仓库根目录的 `datasets/`。

脚本顺序：

1. `1_find_port.py`：发现稳定串口、波特率和右手 profile。
2. `2_get_offset.py`：先只读检查，再用 `--calibrate --confirm` 写零位偏置。
3. `3_just_buttonA.py`：检查 A/B 按键、锁定状态和当前六关节角。

这些脚本通过 `Dobot_Init/master_teleop` 加载主手驱动，不再依赖外部 `Dobot_CRtrainer`。
