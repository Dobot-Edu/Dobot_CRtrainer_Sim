# CR5A + AG95

唯一运行入口是 `usd/cr5a_ag95_mimic.usd`。它由
`urdf/xtrainer_cr5a_ag95_mimic.urdf` 导出，夹爪包含一个主动关节和七个
PhysX mimic 从动关节。

遥操作和录制 action 只暴露六个 CR5A 关节加一个 AG95 主动关节，共 7 维。
完整仿真 state 可以包含七个 mimic 从动关节，用于精确恢复物理状态；它们不是独立控制量。

旧 origin、DH 和重导出实验资产统一保存在仓库根目录 `backup/`，正常启动和资产检查
不会读取这些文件。

USD、URDF 与 `meshes/` 必须一起保留，不能只复制一个 USD 文件，否则相对网格引用会失效。
