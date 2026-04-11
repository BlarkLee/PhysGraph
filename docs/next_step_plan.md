# PhysGraph 改造执行里程碑（压缩执行版）

更新时间：2026-04-11

## 0. 执行总则
- 主线：先 `A 方案`（单 object token + 点轨迹奖励），稳定后再进 `B 方案`。
- 范围：第一阶段仅单手（`dexhandmanip_sh.py`）。
- 闸门：仅当 `A2` 或 `A3` 稳定优于 `A0`，才进入 anchor token 扩展。

## 1. 里程碑总览
| 里程碑 | 目标 | 预计时长 | 主要改动 | 产出 | 通过标准 |
|---|---|---|---|---|---|
| M0 | 规格冻结与数据对齐设计 | 1-2 天 | 文档与接口定义 | 字段映射表、点轨迹 schema | 字段、坐标系、评估口径全部确认 |
| M1 | DexYCB 数据接入与离线缓存 | 2-4 天 | `main/dataset/*` | Dataset adapter + factory 注册 + 缓存脚本 | 能稳定产出训练可读样本 |
| M2 | A1/A2 最小训练闭环 | 3-5 天 | `dexhandmanip_sh.py` | `pt_pos/pt_flow` 观测与奖励接入 | 训练不崩，点误差下降 |
| M3 | A0-A3 对照与闸门决策 | 4-7 天 | 配置/脚本/记录文档 | 对照表 + 结论 | 满足升级 B 的闸门或给出回退结论 |

## 2. 里程碑细化

## M0：规格冻结（必须先过）
- 任务：
  - 确认 DexYCB 到当前管线的最小必要字段。
  - 确认第一阶段点轨迹来源（推荐：mesh 锚点离线变换）。
  - 确认评估指标：成功率/失败率/点误差/异常率。
- 交付：
  - `docs/Physgraph改造.md`（研究决策文档）已完成。
  - `docs/environment_baseline.md` 补充接口契约。
- 退出条件（DoD）：
  - 无未决“关键未知”阻塞 M1 开始。

## M1：数据接入与缓存
- 任务：
  - 新增 DexYCB adapter，接入 `main/dataset/factory.py`。
  - 生成离线点轨迹缓存（含可见性/置信度）。
  - 逐帧可视化核对坐标系一致性。
- 交付：
  - 数据读取类、factory 注册、缓存产物规范。
- 退出条件（DoD）：
  - 任意样本可稳定输出训练所需字段；
  - `当前锚点 vs 目标锚点` 可视化无明显坐标错位。

## M2：最小训练闭环（A1/A2）
- 任务：
  - 在 `target` 中接入点轨迹相关字段（保留旧字段回退开关）。
  - MVP 奖励先启用：`r_pt_pos`，再加 `r_pt_flow`。
  - 保留手部主干：`r_hand_base`。
- 交付：
  - A1 配置：`A1_ptpos`
  - A2 配置：`A2_ptpos_ptflow`
- 退出条件（DoD）：
  - 训练连续运行无早期持续崩溃；
  - 点误差曲线有下降趋势；
  - 无持续速度/力爆炸。

## M3：对照实验与闸门
- 任务：
  - 跑齐并记录 `A0/A1/A2/A3`。
  - A3 为课程化增强版：`pt_pos + pt_flow + region_prox + geom`。
  - 汇总差异配置、成功率/失败率、点误差、异常率。
- 交付：
  - 可复现实验记录表 + 升级结论。
- 退出条件（DoD）：
  - 若 `A2/A3` 稳定优于 `A0`：进入 B 方案设计；
  - 否则：维持 A，优先排查数据对齐/奖励权重/锚点质量。

## 3. 闸门与回退规则
- G1（稳定性闸门）：
  - 若训练早期持续崩溃，立即回退到 `hand_base + pt_pos`。
- G2（效果闸门）：
  - 若 A2/A3 不优于 A0，不推进 B。
- G3（算力闸门）：
  - 若开销过高，先降 `K`，再增加离线缓存中间量。

## 4. 未来 72 小时执行清单
1. 冻结 M1 输入输出字段（DexYCB adapter 的返回字典规范）。
2. 完成锚点缓存最小脚本与可视化校验。
3. 在 `dexhandmanip_sh.py` 预留 `pt_pos` 观测/奖励接口（不启用复杂接触项）。

## 5. 文档联动要求
- 每个里程碑结束后必须同步：
  - `docs/next_step_plan.md`：状态与下一步；
  - `docs/environment_baseline.md`：接口与配置变化；
  - `docs/project_brief.md`：目标与阶段结论（简版）。

## 当前执行版（Execution Version）

### 里程碑状态
| 里程碑 | 目标 | 状态 | 核心产出 |
|---|---|---|---|
| M1 | DexYCB 数据接入 | 已完成 | `dataset_type` 新增 `d` 前缀，`dexycb_rh/lh` 适配器，点轨迹字段输出 |
| M2 | 点轨迹观测/奖励接入 | 已完成（首版） | `dexhandmanip_sh.py` 支持 `usePointTarget/usePtFlow/useRegionGeom` |
| M2.5 | 单手链路配套 | 已完成 | `TASK_MAP` 注册 RH/LH，`train.py` 注册 `res_rh/lh_*` |
| M3 | A0-A3 对照实验 | 待执行 | 三种子对照、门禁结论 |
| M4 | 文档与结论同步 | 进行中 | 关键文档已统一并更新 |

### 本周执行清单
1. 先跑 `A0_pose_baseline`（3 seeds）。
2. 跑 `A1_ptpos`。
3. 跑 `A2_ptpos_ptflow`。
4. 跑 `A3_ptpos_ptflow_region_geom`。
5. 产出 go/no-go。

### 示例命令
- A0: `python main/rl/train.py task=ResDexHand rl_train=ResDexHandPPO side=RH experiment=A0_pose_baseline task.env.usePointTarget=False`
- A1: `python main/rl/train.py task=ResDexHand rl_train=ResDexHandPPO side=RH experiment=A1_ptpos task.env.usePointTarget=True task.env.usePtFlow=False task.env.useRegionGeom=False task.env.poseFallback=True`
- A2: `python main/rl/train.py task=ResDexHand rl_train=ResDexHandPPO side=RH experiment=A2_ptpos_ptflow task.env.usePointTarget=True task.env.usePtFlow=True task.env.useRegionGeom=False task.env.poseFallback=True`
- A3: `python main/rl/train.py task=ResDexHand rl_train=ResDexHandPPO side=RH experiment=A3_ptpos_ptflow_region_geom task.env.usePointTarget=True task.env.usePtFlow=True task.env.useRegionGeom=True task.env.poseFallback=True`
