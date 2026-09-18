# CR5A Pick Orange

本项目使用 X-Trainer 单主手遥操作 Isaac Sim 中的 Dobot CR5A + AG95，完成三颗橙子的抓取、放置、数据采集、ACT 训练与闭环推理。

## 任务说明

**任务目标**：依次抓取场景中的 3 颗橙子，并全部放入目标盘子。

**机器人**：Dobot CR5A + DH-Robotics AG95

**观测**：7 维关节状态，以及 `wrist`、`front` 两路 640 × 480 RGB 图像。

**动作**：6 个 CR5A 关节目标和 1 个 AG95 主动关节目标，共 7 维。

**控制频率**：30 Hz

## 环境安装

仿真环境需要 Linux、NVIDIA 驱动、Isaac Sim 5.0 和与之匹配的 Isaac Lab。项目已在 `Dobot-Sim5.0-5090` 环境中验证，RTX 5090D 使用 CPU PhysX 与 GPU 渲染。

```bash
cd /path/to/CR5_Sim
conda activate Dobot-Sim5.0-5090
python -m pip install --no-deps -e simulation/source
```

训练、转换和策略服务使用独立的 LeRobot 环境：

```bash
conda activate lerobot
python -m pip install -e ./lerobot-main
python -m pip install h5py
```

启动前检查核心资产：

```bash
python scripts/check_assets.py
```

预期输出 `CR5A asset check: PASS`。

## 下载数据集和模型

```bash
conda activate lerobot

hf download Dobot-Official/CR5A-Pick_Orange \
  --repo-type dataset \
  --local-dir lerobot_datasets/cr5a_pick_orange_v1

hf download Dobot-Official/CR5A-act-Pick_Orange \
  --local-dir pretrained_model
```

- 数据集：[Dobot-Official/CR5A-Pick_Orange](https://huggingface.co/datasets/Dobot-Official/CR5A-Pick_Orange)
- ACT 模型：[Dobot-Official/CR5A-act-Pick_Orange](https://huggingface.co/Dobot-Official/CR5A-act-Pick_Orange)

## 启动遥操作与采集

在 `constants.py` 中选择任务、控制方式和是否录制：

```python
teleop_mode = "keyboard"      # 或 "cr5a_master"
save_mode = False             # 录制时设为 True
task = "pick_orange"
device = "cpu"
dataset_layout = "per_episode"
```

启动仿真：

```bash
conda activate Dobot-Sim5.0-5090
cd /path/to/CR5_Sim
source ./activate_sim.sh
```

主手模式下，仿真窗口出现后按 `B` 建立相对参考，再短按主手 `A` 解锁。键盘 `R` 将当前轮次标记为失败并重置，`N` 将当前轮次标记为成功并重置。

`save_mode = True` 时，每个 episode 会单独保存到 `datasets/`，已有文件不会被覆盖。

## 转换为 LeRobot 数据集

```bash
conda activate lerobot

python scripts/convert_cr5_hdf5_to_lerobot.py \
  --input datasets \
  --output lerobot_datasets/cr5a_pick_orange_v1 \
  --repo-id local/cr5a_pick_orange \
  --strict
```

转换脚本默认只保留 `success=True` 的 episode，并检查状态、动作及两路 RGB 的帧数是否一致。

## 训练 ACT

```bash
conda activate lerobot

lerobot-train \
  --dataset.repo_id=local/cr5a_pick_orange \
  --dataset.root=lerobot_datasets/cr5a_pick_orange_v1 \
  --dataset.eval_split=0.1 \
  --dataset.return_uint8=true \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir=outputs/train/cr5a_pick_orange_act_v1 \
  --job_name=cr5a_pick_orange_act_v1 \
  --steps=50000 \
  --batch_size=16 \
  --num_workers=8 \
  --log_freq=100 \
  --eval_steps=2000 \
  --max_eval_samples=512 \
  --save_freq=10000 \
  --wandb.enable=false
```

## 启动策略推理

使用两个终端。先启动 LeRobot 策略服务：

```bash
cd /path/to/CR5_Sim
conda activate lerobot

python -m lerobot.async_inference.policy_server \
  --host=127.0.0.1 \
  --port=5555 \
  --fps=30
```

服务端开始监听后，在第二个终端启动 Isaac Sim：

```bash
cd /path/to/CR5_Sim
conda activate Dobot-Sim5.0-5090

python simulation/scripts/evaluation/policy_inference.py \
  --task=LeIsaac-CR5A-PickOrange-v0 \
  --eval_rounds=10 \
  --step_hz=30 \
  --policy_host=127.0.0.1 \
  --policy_port=5555 \
  --policy_action_horizon=10 \
  --policy_device=cuda \
  --device=cpu \
  --enable_cameras \
  --policy_checkpoint_path=pretrained_model
```

推理过程中可按 `R` 手动重置当前 episode。

## 项目结构

```text
CR5_Sim/
├── assets/             # CR5A、AG95 和 Pick Orange 场景资产
├── simulation/         # Isaac Lab 任务、遥操作与推理代码
├── Dobot_Init/         # X-Trainer 主手驱动与配置
├── Data_Collections/   # 串口发现、标定和按键检查
├── scripts/            # 资产检查与 HDF5 转换工具
├── lerobot-main/       # 本项目验证使用的 LeRobot 版本
├── constants.py        # 启动配置
└── activate_sim.sh     # 仿真与数据采集入口
```

本项目基于 [embodied-dobot/x-trainer](https://github.com/embodied-dobot/x-trainer)、Isaac Lab、LeIsaac 和 LeRobot 开发。
