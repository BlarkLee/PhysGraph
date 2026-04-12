# AGENTS 协作约定（PhysGraph 改造）

更新时间：2026-04-12

## 1) 长期目标与原则
- 目标：将 PhysGraph 第一阶段锁定在 `OakInk + 点轨迹对象表征`，单手优先。
- 数据集口径（新增）：第一阶段唯一数据集为 `OakInk`，`DexYCB` 不再纳入阶段执行、对照与结论依据。
- 原则：先验证可行性与稳定性，再推进高新颖性结构扩展。
- 决策门禁：仅当 `A2` 或 `A3` 稳定优于 `A0`，才进入方案 B（anchor tokens）。

## 2) 第一阶段改动边界
- 必改：
  - `main/dataset/*`（OakInk 短序列筛选/适配与 factory）
  - `physgraph_envs/lib/envs/tasks/dexhandmanip_sh.py`（点轨迹观测/奖励）
  - 单手链路最小配套（`TASK_MAP` 与 `train.py` 注册）
- 尽量不改：
  - `dexhandmanip_bih.py`
  - 主网络大规模重构

## 3) 实验命名与门禁规则
- 命名固定：`A0_pose_baseline`、`A1_ptpos`、`A2_ptpos_ptflow`、`A3_ptpos_ptflow_region_geom`
- 当前执行口径：先在 OakInk 短序列上做 A0 冒烟，再推进 A1/A2/A3。
- 每次实验至少记录：配置差异、成功率/失败率、点误差、数值异常率
- 回退规则：
  - 早期持续崩溃 -> 回退 `hand_base + pt_pos`
  - A2/A3 不优于 A0 -> 暂缓结构升级，优先排查数据对齐与奖励权重

## 4) 文档维护（长期规则）
- 每个里程碑完成后同步更新：
  - `docs/next_step_plan.md`
  - `docs/environment_baseline.md`
  - `docs/project_brief.md`
- 阶段决策变化时同步更新：
  - `docs/Physgraph改造.md`
  - `AGENTS.md`
