# CR5_Sim：X-Trainer 单主手控制 Isaac Sim 中的 CR5A

本仓库把原来的三个工程拆成一个面向单臂 CR5A 的工作区：从 X-Trainer 双臂主从系统中抽出一个真实主手，只驱动 Isaac Sim 里的 CR5A + AG95。真机 CR5A follower、双臂 X-Trainer 仿真、Quest/VR、训练和缓存都不属于本仓库的运行链路。

## 目录

```text
CR5_Sim/
├── assets/
│   ├── robot/cr5a_ag95/             # CR5A + AG95 的 USD、URDF、meshes
│   ├── scenes/kitchen_with_orange/  # PickOrange 场景
│   └── docs/                        # 说明图片
├── simulation/
│   ├── source/                      # 精简后的 Isaac Lab/LeIsaac 包
│   └── scripts/environments/teleoperation/teleop_se3_agent.py
├── Dobot_Init/master_teleop/        # 主手驱动、配置、标定运行时覆盖和 SDK
├── Data_Collections/master_teleop/  # 串口发现、零位标定、按键测试脚本
├── datasets/                        # 录制的 HDF5 文件
├── scripts/check_assets.py          # 启动前资产检查
├── scripts/check_cr5a_articulation.py    # 不带任务的 articulation/PhysX 检查
├── scripts/check_cr5a_robot_only_env.py  # RobotOnly + action manager 检查
├── constants.py                    # USD、任务、遥操方式和录制开关
├── 启动仿真.sh                      # 唯一启动入口
└── backup/                         # 历史启动脚本和实验文件
```

当前注册 `LeIsaac-CR5A-RobotOnly-v0` 和 `LeIsaac-CR5A-PickOrange-v0`，设备是 `cr5a_master` 和 `cr5a_keyboard`。启动配置统一写在根目录 `constants.py`：`task="only_robot"` 使用 RobotOnly 最小环境，`task="pick_orange"` 加载厨房和橙子场景；`teleop_mode` 选择主手或键盘。主手代码不再从外部 `Dobot_CRtrainer` 目录动态导入。

## 重要前置检查

如果检查结果为 `FAIL`，请从同事的完整 Linux 资产目录或 Git LFS 工作区补齐同名文件，再继续下面的步骤。不要用空文件启动仿真。

```bash
cd /path/to/CR5_Sim
python scripts/check_assets.py
```

检查必须输出 `CR5A asset check: PASS`，且至少包含：

- `assets/robot/cr5a_ag95/usd/cr5a_ag95_mimic.usd`
- `assets/robot/cr5a_ag95/urdf/xtrainer_cr5a_ag95_mimic.urdf`
- `assets/scenes/kitchen_with_orange/scene.usd`

USD、URDF 和 `meshes/` 必须一起复制，不能只拿一个 USD。

## 环境安装

源工程 README 的已验证组合是 Isaac Sim 4.5 + Isaac Lab 0.47.1；同事交接记录使用了 `Dobot-Sim5.0` 环境。仓库内的 Isaac Lab 源码版本是 2.2.1，`pyproject.toml` 的可选依赖写的是 2.3.0，因此这两个版本标记需要和实际 Python 环境核对。当前没有把 Isaac Sim 5.1 作为已验证组合，不能因为能打开窗口就认为任务 API、PhysX 和 USD 行为完全兼容。

请以机器上已经安装并能启动的 Isaac Lab 环境为准，不要同时启动两个 Isaac Sim。先不要为了黑屏直接重装 CUDA；Vulkan 日志已经识别到 NVIDIA GPU，黑屏也可能来自 USD 依赖缺失、版本组合不匹配或窗口显示问题。下面的命令假定该环境中的 Python 能导入 `isaaclab`。

```bash
cd /path/to/CR5_Sim
conda activate Dobot-Sim5.0
pip install -e simulation/source

export CR5_SIM_ROOT="$PWD"
export LEISAAC_ASSETS_ROOT="$CR5_SIM_ROOT/assets"
export PYTHONPATH="$CR5_SIM_ROOT/simulation/source:$CR5_SIM_ROOT/Dobot_Init:${PYTHONPATH:-}"
```

如果使用的是不同环境，可用 `CR5_SIM_PYTHON=/完整路径/python` 指定启动脚本使用的 Python。

## 连接真实单右主手

1. 给主手上电并连接 USB。主手是 X-Trainer 的右手配置，串口优先使用 `/dev/serial/by-id/` 下的稳定路径。
2. 确认当前用户有串口权限：

   ```bash
   ls -l /dev/serial/by-id/
   groups                         # 应包含 dialout
   sudo usermod -aG dialout "$USER" # 只需首次执行；随后重新登录
   ```

3. 发现串口、波特率和右手 ID，并写入 `Dobot_Init/master_teleop/config/master_runtime.yaml`：

   ```bash
   python Data_Collections/master_teleop/collection/1_find_port.py
   ```

   也可以只探测指定端口而不保存：

   ```bash
   python Data_Collections/master_teleop/collection/1_find_port.py \
       --port /dev/serial/by-id/你的设备 --no-save
   ```

4. 检查主手按键和当前角度。这个脚本不会连接或移动真实 CR5A：

   ```bash
   python Data_Collections/master_teleop/collection/3_just_buttonA.py
   ```

主手启动时默认锁定。A 短按切换锁定/解锁，B 短按和扳机状态会被读取；遥操进程退出时会释放主手扭矩，避免设备持续发热。需要在标定过程中保持机械锁定时，使用下面的 `3_just_buttonA.py`，它会按标定流程保留锁定状态。

## 主手零位标定

标定目标取 `master_hand.yaml` 中真实 CR5A 的标定姿态，当前为 `[90, 0, 90, 0, -90, -90]` 度。把主手物理姿态摆到与这个姿态对应的位置，夹爪完全张开并保持不动。仿真任务本身的初始姿态为 `[88, -7, 98, 0, -90, -90]` 度；启动时按 `B` 会建立“当前对当前”的相对参考，因此两组 home 数值不同不会造成机械臂跳变。

先做只读检查，确认采样稳定且误差方向正确：

```bash
python Data_Collections/master_teleop/collection/2_get_offset.py
```

确认姿态无误后才写入运行时覆盖：

```bash
python Data_Collections/master_teleop/collection/2_get_offset.py --calibrate --confirm
```

命令会要求输入 `CALIBRATE`，然后把新偏置写入 `Dobot_Init/master_teleop/config/master_runtime.yaml`。基准配置 `master_hand.yaml` 不会被覆盖，便于回滚。诊断 JSONL/JSON 文件写入 `Data_Collections/master_teleop/diagnostics/`。

更换主手、更换电机零位或发现六关节整体偏移时重新标定。普通的每次启动不需要重新标定。

## 启动 Isaac Sim 和遥操

启动脚本会调用 Isaac Lab 的 `AppLauncher`，由同一个 Python 进程打开 Isaac Sim 窗口，不需要先手工打开另一个 Isaac Sim 实例。图形桌面/SSH 场景需要提前设置正确的 `DISPLAY`，例如 `export DISPLAY=:1`。

### 5090D 短期稳定配置

当前已经验证的 5090D 组合是 Isaac Sim 5.0 + Isaac Lab 0.45.9 + CPU 物理。`torch.cuda` 和 Vulkan 可以识别 RTX 5090，但 Isaac Sim 5.0 的 GPU PhysX pipeline 在该显卡上仍可能初始化失败。因此短期遥操使用 CPU 物理、GPU rendering。需要复测 CUDA 时，把 `constants.py` 的 `device` 改为 `"cuda"`，不作为稳定基线。

```bash
conda activate Dobot-Sim5.0-5090
cd ~/dsw_ws/CR5_Sim
export DISPLAY=:0
export XAUTHORITY=/run/user/1000/gdm/Xauthority
export CR5_SIM_ROOT="$PWD"
export LEISAAC_ASSETS_ROOT="$PWD/assets"
export PYTHONPATH="$PWD/simulation/source:$PWD/Dobot_Init:${PYTHONPATH:-}"

source ./启动仿真.sh
```

脚本读取 `constants.py`，默认使用 `--device=cpu` 和 `--enable_cameras`。这里的 `--enable_cameras` 只用于选择可见的 rendering Kit；当前 RobotOnly 任务没有启用相机传感器。脚本可以用 `source` 或 `bash` 执行，结束后会回到当前 shell，并打印退出码。

要单独复测 CUDA 物理路径时，使用：

```bash
sed -i 's/device = "cpu"/device = "cuda"/' constants.py
source ./启动仿真.sh
```

### 真实主手控制仿真 CR5A

```bash
cd /path/to/CR5_Sim
export DISPLAY=:1
source ./启动仿真.sh
```

上述命令由 `constants.py` 决定启动轻量的 RobotOnly 或 PickOrange 环境、键盘或主手设备、以及 CPU/CUDA。默认使用 CPU 物理作为稳定基线。5090D 上建议保持 `device = "cpu"`。

等价命令：

```bash
python simulation/scripts/environments/teleoperation/teleop_se3_agent.py \
  --task=LeIsaac-CR5A-RobotOnly-v0 \
  --teleop_device=cr5a_master \
  --num_envs=1 \
  --device=cpu \
  --enable_cameras
```

等价的 PickOrange 场景命令：

```bash
python simulation/scripts/environments/teleoperation/teleop_se3_agent.py \
  --task=LeIsaac-CR5A-PickOrange-v0 \
  --teleop_device=cr5a_master \
  --num_envs=1 \
  --device=cpu \
  --enable_cameras
```

仿真窗口出现后，先按键盘 `B` 建立“当前主手姿态 -> 当前仿真 CR5A 姿态”的相对参考，再用主手 A 短按解锁，移动主手即可跟随。再次短按 A 会锁定主手跟随。默认扳机使用连续 `0.0 ~ 1.0` 的 AG95 模拟量映射，完全按下才到最大闭合位置。键盘 `R` 失败并重置，`N` 成功并重置。

### 分层验证命令

不要直接用完整遥操脚本判断 USD 是否有效。建议按以下顺序验证：

1. 直接用 `isaac sim` 打开 USD，只验证 GUI、材质和 USD 基本解析。
2. 运行最小 articulation 检查，只验证 PhysX articulation 的创建、reset 和 stepping：

   ```bash
   python -u scripts/check_cr5a_articulation.py \
     --steps=120 --device=cpu --headless
   ```

   预期输出 `Joint count: 14` 和 `PASS: articulation reset and stepping succeeded`。
3. 运行 RobotOnly 环境检查，加入 FrameTransformer、action manager 和 `env.reset()`，但不连接任何遥操设备：

   ```bash
   python -u scripts/check_cr5a_robot_only_env.py \
     --steps=120 --device=cpu --headless
   ```

   预期 action space 为 `(1, 7)`，并输出 `PASS: RobotOnly env reset and stepping succeeded`。
4. 上述两步通过后，再运行键盘遥操，最后才连接真实主手。

单独打开 USD 能成功，不代表 Isaac Lab 任务一定能启动。任务还会创建 articulation、FrameTransformer、actuator、action manager 并执行环境 reset；这些步骤中的路径或关节配置错误会在窗口正常出现后退出。

### 当前数采是否包含相机

当前 `LeIsaac-CR5A-PickOrange-v0` 已启用 USD 中的两路 Camera prim：腕部相机 `Link6/wrist_camera_preview` 和外部相机 `front_camera_preview`。两路均按 `640x480`、RGB、30 FPS 配置，并作为策略观测写入 HDF5。启动录制时仍需带 `--enable_cameras`。

当前遥操和录制 action 固定为 7 维：六个 CR5A 关节加一个 AG95 主动关节。完整 `states` 仍可包含七个 mimic 从动关节，以便精确恢复仿真状态；这些从动值不是独立控制量。

相机数据量很大，当前只保存 RGB，不保存 depth 或 segmentation。

### 不连接主手时的键盘检查

```bash
cd /path/to/CR5_Sim
source ./启动仿真.sh
```

键位：`U/I/O/J/K/L` 控制 J1-J6，按住 `Z` 反向，`H` 控制夹爪，`B` 开始，`R` 重置，`N` 成功并重置。兼容旧命令的 `--teleop_device=bi_keyboard` 别名仍可用，但它现在只控制 CR5A 单臂。

### 录制 HDF5

把 `constants.py` 中的 `save_mode` 改成 `True`。默认的
`dataset_layout = "per_episode"` 会在 `dataset_file` 所在目录中按任务名逐轮写入：

```text
LeIsaac-CR5A-PickOrange-v0_000000.hdf5
LeIsaac-CR5A-PickOrange-v0_000001.hdf5
LeIsaac-CR5A-PickOrange-v0_000002.hdf5
```

启动时会扫描已有编号并自动使用下一个空闲编号，不会覆盖旧数据，也不会因为文件已存在而退出。每个文件内部只有 `/data/demo_0`。需要旧版“一个 HDF5 包含多个 demo”的布局时，把 `dataset_layout` 改成 `"single_file"`；若指定文件已存在且未使用 `--resume`，程序会自动选择带六位编号的新文件。

```bash
python simulation/scripts/environments/teleoperation/teleop_se3_agent.py \
  --task=LeIsaac-CR5A-PickOrange-v0 \
  --teleop_device=cr5a_master \
  --record \
  --dataset_layout=per_episode \
  --dataset_file=../datasets/cr5a_master_pick_orange.hdf5
```

从 `R`/`N` 重置时，录制器分别记录失败/成功 episode。`--resume` 只用于 `single_file` 模式；`per_episode` 总是自动续号。不要让两个进程同时写同一个数据目录。

策略观测中的 `joint_pos`、`joint_vel`、相对关节状态和关节目标均按固定顺序保存 7 维：J1-J6 加 AG95 主 mimic 关节。`states` 与 `initial_state` 仍保留完整 articulation 状态，供精确回放使用。

### 转换为 LeRobot 数据集

`scripts/convert_cr5_hdf5_to_lerobot.py` 同时支持旧版单个多-demo HDF5 和新版逐 episode HDF5 目录。转换前会检查成功标记、状态、动作及两路 RGB 的帧数和形状；默认只转换 `success=True` 的 episode，并以 30 FPS 写为 LeRobot 0.6.1 数据集。

LeRobot 0.6.1 要求 Python 3.12，建议使用与 Isaac Sim 分开的环境：

```bash
cd /path/to/CR5_Sim
conda create -n cr5_lerobot python=3.12 -y
conda activate cr5_lerobot
pip install -e ./lerobot-main
pip install h5py
```

转换整个逐 episode 目录：

```bash
python scripts/convert_cr5_hdf5_to_lerobot.py \
  --input datasets \
  --output datasets/lerobot/cr5a_pick_orange \
  --repo-id local/cr5a_pick_orange
```

`--input` 也可以直接指向旧的聚合 HDF5。若确实需要保留 `success=False` 的轮次，加 `--include-failed`；输出目录已存在时脚本默认停止，确认要重建后才使用 `--overwrite`。新版 7 维 `joint_pos` 会直接转换；旧数据的状态向量若超过 7 维，默认选择索引 `0,1,2,3,4,5,6`，即 J1-J6 和 AG95 主关节，并在结果中打印提示。若旧文件关节顺序不同，用 `--legacy-state-indices` 显式指定。

转换完成后可以先用 ACT 做一轮训练验证：

```bash
lerobot-train \
  --dataset.repo_id=local/cr5a_pick_orange \
  --dataset.root=datasets/lerobot/cr5a_pick_orange \
  --policy.type=act \
  --output_dir=outputs/train/cr5a_pick_orange_act \
  --policy.device=cuda
```

## 配置和路径覆盖

- `Dobot_Init/master_teleop/config/master_hand.yaml`：右主手 ID、波特率、夹爪 `analog`/`binary` 模式、按钮阈值和基准偏置。
- `Dobot_Init/master_teleop/config/master_runtime.yaml`：串口发现/标定生成的运行时覆盖，不提交真实机器专属路径。
- `CR5_SIM_ROOT`：仓库根目录，默认由启动脚本自动推断。
- `LEISAAC_ASSETS_ROOT`：资产根目录，默认是 `CR5_SIM_ROOT/assets`。
- `CR5A_AG95_USD_PATH`：仅供诊断脚本临时覆盖，正常启动固定使用 canonical mimic USD。
当前运行资产、遥操方式和任务由 `constants.py` 决定。示例配置为：

```python
usd = "assets/robot/cr5a_ag95/usd/cr5a_ag95_mimic.usd"
teleop_mode = "cr5a_master"
save_mode = False
task = "only_robot"
dataset_layout = "per_episode"
```

外部 action 是 6 个 CR5A 关节加 1 个主动夹爪关节，共 7 维。旧版 origin、DH 和历史重导出资产只保存在 `backup/archived_assets/`，不参与正常启动。
如果以后重新从 URDF 导出 USD，请在 Isaac Sim 5.0 的本地图形桌面终端运行：

```bash
python scripts/import_urdf_to_usd.py \
  assets/robot/cr5a_ag95/urdf/xtrainer_cr5a_ag95_mimic.urdf \
  /tmp/cr5a_ag95_mimic_candidate.usd \
  --headless
```

先把候选文件放在活动资产目录之外。只有它通过纯机器人 CPU 测试后，才替换
`usd/cr5a_ag95_mimic.usd`。`patch_usd_mimic_limits.py` 仅用于诊断性
补写限位，不建议对新导出文件继续叠加未知补丁。

`启动仿真.sh` 可以用 `source` 执行。Isaac Sim 结束或崩溃后，脚本会打印退出码并返回当前终端，不会用 `exec` 替换交互 shell；退出码同时保存在 `CR5_SIM_LAST_EXIT_CODE` 中。旧脚本和历史实验文件仅存放在 `backup/`。

## 常见问题

**`No calibrated right master found`**：先运行 `1_find_port.py`，确认没有其他进程占用串口，检查 `master_runtime.yaml` 的 `side: right` 和 `dialout` 权限。

**按 B 后没有跟随**：B 必须在 Isaac Sim 窗口获得焦点时按下；随后 A 短按解锁主手。确认启动参数是 `LeIsaac-CR5A-PickOrange-v0` + `cr5a_master`。

**环境在 `Starting the simulation` 后退出，或提示 `FrameTransformer ... No matching prims`**：这是 USD 层级与任务配置不一致。当前完整 mimic USD 中可用于 frame transform 的刚体是：

```text
/World/Robot/Link6
/World/Robot/gripper_finger1_knuckle_link
```

`cr5a_base` 和 `gripper_base_link` 在该 USD 中是普通 Xform，不能直接作为 `FrameTransformerCfg` 的 source 或 target。

**仿真窗口打不开**：检查 `DISPLAY`、`XAUTHORITY`、NVIDIA 驱动和 Isaac Lab 环境；不要用系统 WindowsApps 的 Python，必须使用能导入 `isaaclab` 的环境 Python。键盘遥操不要使用 `--headless`，必须使用可见的 rendering Kit。

**日志出现 `PhysX GPU solver pipeline failed` 或 `NVTT CUDA driver not found`**：这表示 GPU PhysX 或 NVTT 纹理工具初始化失败并发生软件回退，不等同于 PyTorch 一定是 CPU 版。先在启动仿真的同一个 Python 环境执行：

```bash
nvidia-smi
python -c "import torch, importlib.metadata as m; print('torch=', torch.__version__); print('torch_cuda=', torch.version.cuda); print('cuda_available=', torch.cuda.is_available()); print('gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None); print('isaaclab=', m.version('isaaclab'))"
ldconfig -p | grep 'libcuda.so.1'
```

然后用无窗口键盘任务区分窗口问题和任务/资产问题：

```bash
python simulation/scripts/environments/teleoperation/teleop_se3_agent.py \
  --task=LeIsaac-CR5A-PickOrange-v0 \
  --teleop_device=cr5a_keyboard \
  --num_envs=1 \
  --device=cuda \
  --headless
```

`--headless` 按设计不会弹出 Isaac Sim 窗口；它只用于确认 Python、GPU、任务注册和 USD/PhysX 初始化是否成功。键盘遥操在 headless 模式下也没有可接收按键的仿真窗口，因此验证完成后请用 `Ctrl-C` 结束进程。若该命令持续运行且没有 `ERROR`/`Traceback`，通常说明任务已经启动。

如果 headless 可以运行，优先检查 `DISPLAY`、Xorg/SSH 会话和 Isaac Sim 用户缓存（可在 Isaac Sim 安装目录执行一次 `./isaac-sim.sh --reset-user`），不要先重装 CUDA。如果 headless 也在 USD/PhysX 初始化阶段失败，先修复完整资产和 Isaac Sim/Isaac Lab 版本配对。当前仓库中的 CR5A 资产检查会把 `usd/**/*.usd` 和 `meshes/**/*.stl` 下的 0 字节文件报告为 `FAIL`；不要用只有顶层 USD 或 URDF、缺少 meshes 的目录启动。

**资产找不到或 USD 解析失败**：先执行 `python scripts/check_assets.py`。如果报告 0 字节，占位文件必须从完整资产源补齐。

**夹爪只有开/关两档**：确认 `Dobot_Init/master_teleop/config/master_hand.yaml` 中为 `mode: analog`，并重启遥操进程。`binary` 模式才会使用 `close_threshold`/`open_threshold` 做滞回二值化。主手的模拟量会被映射到主动 AG95 关节的 `0.0 ~ 0.6524 rad`。

**夹爪方向或状态异常**：确认使用的是 `cr5a_master`，不要把旧双臂 `xtrainerleader` 设备混入本仓库；mimic USD 只有 `gripper_finger1_joint` 接收外部 action，其余关节由 PhysX mimic 关系跟随。

## 当前验证状态

目录迁移、CR5A-only 任务注册、路径解耦、mimic USD 集成、FrameTransformer 路径修复、主手驱动与标定脚本的 Python 编译检查均已完成。运行资产现已收敛为 `cr5a_ag95_mimic.usd`，action manager 固定为 7 维；PickOrange 从 `scene.usd` 读取盘子和橙子的默认位姿，reset 时直接恢复这些位姿，不再随机微调。以上整理需要在 Isaac Sim 5.0/Linux 环境重新执行 articulation、RobotOnly、键盘、真实主手、逐 episode 录制和 LeRobot 转换测试。当前稳定物理基线仍为 CPU；CUDA 只作为单独复测路径。

## 后续扩展入口

### 新增任务场景

1. 把场景 USD 和依赖纹理放到 `assets/scenes/<scene_name>/`。
2. 在 `simulation/source/leisaac/assets/scenes/` 增加场景路径和 `AssetBaseCfg`，或直接在任务 SceneCfg 中定义。
3. 新建 `simulation/source/leisaac/tasks/<task_name>/`，至少包含 `__init__.py` 和 `<task_name>_env_cfg.py`。
4. 在该 task 的 `__init__.py` 用 `gym.register()` 注册唯一 ID；观察、动作、奖励、终止条件和随机化分别放到 env cfg 或 `mdp/` 下。
5. 用 `--task=新注册的ID` 启动。任务的 `use_teleop_device()` 负责声明它支持哪些遥操设备。

### 新增遥操设备

- 真实主手读取和标定属于 `Dobot_Init/master_teleop`，不要把串口读写放进 task。
- Isaac Sim 内的设备适配放在 `simulation/source/leisaac/devices/`，例如键盘放 `keyboard/`，硬件主手放 `lerobot/`，并在 `devices/__init__.py` 导出。
- 设备输出的 action 字典如何变成机器人关节目标，由任务的 `preprocess_device_action()` 处理。新增任务若仍是 CR5A 六关节 + AG95，可以复用 `CR5AMaster`；若关节或动作布局不同，应新增设备或新的 action preprocessor。
- `Data_Collections/master_teleop/` 只放串口发现、标定、按钮检查和采集辅助脚本；它不是 task 或遥操设备实现目录。录制入口仍是 `simulation/scripts/environments/teleoperation/teleop_se3_agent.py`，数据文件统一写入 `datasets/`。
