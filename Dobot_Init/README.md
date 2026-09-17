# Dobot_Init

这里是仿真所需的真实单右主手初始化层：Dynamixel 读取、右手 profile、按键状态机、配置和本地 SDK。它只读取主手，不连接或移动真实 CR5A follower。

标定脚本位于 `Data_Collections/master_teleop/collection/`，运行时配置位于 `master_teleop/config/`。不要把真实串口路径写进 Python；串口发现和标定会写入 `master_runtime.yaml`。
