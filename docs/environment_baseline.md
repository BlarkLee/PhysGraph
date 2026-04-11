# 当前代码环境基线与改造接口契约

更新时间：2026-04-11

## 1. 当前基线（与改造直接相关）

## 1.1 数据层
- 文件：`main/dataset/factory.py`
- 现状：
  - `dataset_type()` 通过 index 前缀判别数据集；
  - 已支持 `oakink2/grabdemo/...`，未内置 DexYCB 分支。

## 1.2 环境层（单手主入口）
- 文件：`physgraph_envs/lib/envs/tasks/dexhandmanip_sh.py`
- 现状：
  - `target` 观测包含 `delta_manip_obj_pos/quat/vel/ang_vel` 等 pose 主导字段；
  - 奖励函数 `compute_imitation_reward` 仍以 object pose/vel 误差为核心。

## 1.3 边界约束
- 第一阶段只改：
  - `main/dataset/*`
  - `dexhandmanip_sh.py`
- 第一阶段尽量不改：
  - `dexhandmanip_bih.py`
  - 大规模图结构重构。

## 2. 第一阶段接口契约（执行口径）

## 2.1 Dataset adapter 最小返回字段（DexYCB）
- 必需字段（训练对齐）：
  - `wrist_pos`
  - `wrist_rot`
  - `wrist_velocity`
  - `wrist_angular_velocity`
  - `mano_joints`
  - `mano_joints_velocity`
  - `obj_trajectory`（兼容旧逻辑）
  - `obj_points_t`（新增：时序锚点）
  - `obj_points_target_t`（新增：目标锚点）
  - `obj_points_mask_t`（新增：可见性/置信度）
- 兼容原则：
  - 第一阶段允许 `obj_trajectory` 与点轨迹并存；
  - 点轨迹字段缺失时可回退 pose 路径（仅用于调试，不用于最终对照结论）。

## 2.2 环境 `target` 观测新增字段（A 方案）
- 新增优先级：
  1. `delta_obj_points`（当前点到目标点偏差）
  2. `delta_obj_points_flow`（点流偏差）
  3. `obj_points_mask`（加权/屏蔽）
- 保留字段：
  - 现有 wrist/joints 相关项全部保留；
  - object pose 字段保留开关，便于 A0 对照。

## 2.3 奖励开关契约（课程化）
- A1 开启：
  - `r_hand_base`
  - `r_pt_pos`
- A2 增强：
  - `r_pt_pos + r_pt_flow`
- A3 增强：
  - `r_pt_pos + r_pt_flow + r_region_prox + r_geom`
- 第一阶段默认不强开：
  - `r_region_false`
  - `r_force_unexp`
  - `r_force_excess`（仅可轻量保底）

## 3. 文件改造触点清单（第一阶段）
- `main/dataset/factory.py`
  - 增加 DexYCB dataset_type 分支与注册路径。
- `main/dataset/*`
  - 新增 DexYCB 数据读取与缓存生成逻辑。
- `physgraph_envs/lib/envs/tasks/dexhandmanip_sh.py`
  - `compute_observations()`：拼接点轨迹目标字段；
  - `compute_reward()` 与 `compute_imitation_reward()`：接入 A1/A2/A3 奖励项；
  - 增加回退开关（pose vs point）。

## 4. 验收与回归口径
- 每轮实验至少记录：
  - 配置差异；
  - 成功率/失败率；
  - 点误差（可见性加权）；
  - 数值异常（速度/力爆炸）。
- 必跑对照：
  - `A0_pose_baseline`
  - `A1_ptpos`
  - `A2_ptpos_ptflow`
  - `A3_ptpos_ptflow_region_geom`

## 5. 风险观察点（执行时重点盯）
- 坐标系错位：点误差不降或剧烈震荡。
- 奖励过强：早期探索被压死，成功率接近 0。
- 计算开销过高：吞吐骤降，训练不稳定。

## 6. 文档同步规则
- M1/M2/M3 完成后必须同步更新：
  - `docs/next_step_plan.md`
  - `docs/environment_baseline.md`
  - `docs/project_brief.md`

## 当前执行版（Execution Version）

### 已落地接口变更
- DexYCB 索引：`d<seq>@<stage>`
- 新增 dataset：`dexycb_rh/lh`
- 新增点轨迹字段：`obj_points_t / obj_points_target_t / obj_points_mask_t`
- 单手链路：
  - `ResDexHandRH` / `ResDexHandLH`
  - `res_rh_*` / `res_lh_*` 训练注册

### 点轨迹配置开关
- `pointTrackK`
- `usePointTarget`
- `usePtFlow`
- `useRegionGeom`
- `poseFallback`
- `wHandBaseReward/wPtPosReward/wPtFlowReward/wRegionGeomReward`
- `ptPosBeta/ptFlowBeta`

### 当前建议验证顺序
1. 数据字段与 shape 检查（20 条样本）。
2. 点轨迹与目标点可视化核对（至少 5 条序列）。
3. 64 env 冒烟 2k steps。
4. A0 -> A1 -> A2 -> A3 三种子对照。
