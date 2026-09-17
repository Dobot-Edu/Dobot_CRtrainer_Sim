# 资产说明

- `robot/cr5a_ag95/`：CR5A + AG95 的 canonical mimic USD、源 URDF 和网格。
- `scenes/kitchen_with_orange/`：PickOrange 场景及其贴图。
- `docs/`：项目说明图片。

正常运行只读取 `robot/cr5a_ag95/usd/cr5a_ag95_mimic.usd`。旧版 origin、DH 和历史重导出资产已移到仓库根目录 `backup/archived_assets/`，不要从活动配置引用它们。

如果 canonical USD、源 URDF 或 mesh 显示为 0 字节，说明资产传输不完整，不能启动 Isaac Sim。请从完整的 Linux 资产目录补齐后，再运行 `python scripts/check_assets.py`。
