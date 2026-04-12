# PhysGraph 改造执行里程碑（压缩执行版）

更新时间：2026-04-12

## 变更说明（2026-04-12）
- 阶段数据集主线锁定为 `OakInk`（第一阶段唯一数据集）。
- `DexYCB` 路径从第一阶段执行计划中移除，不再作为实验输入或对照来源。
- 执行原则改为：复用原项目 OakInk 数据链路与奖励主干，仅做短序列优先与入口配置增强。
- A0-A3 命名与门禁规则不变。

## 0. 执行总则
- 主线：先 `A 方案`（单 object token + 点轨迹奖励），稳定后再进 `B 方案`。
- 范围：第一阶段仅单手（`dexhandmanip_sh.py`）。
- 闸门：仅当 `A2` 或 `A3` 稳定优于 `A0`，才进入 anchor token 扩展。

## 0.1 当前阶段目标（OakInk-only）
- 目标：在 `OakInk` 短序列上验证 `A0->A1->A2->A3` 的稳定闭环与效果趋势。
- 关注点：优先保证可训练、可复现、可解释，不引入跨数据集变量。
- 相比旧口径的变化：不再并行维护 DexYCB 路径，减少“数据语义差异”对阶段结论的干扰。

## 1. 里程碑总览
| 里程碑 | 目标 | 预计时长 | 主要改动 | 产出 | 通过标准 |
|---|---|---|---|---|---|
| M0 | 规格冻结与数据对齐设计 | 1-2 天 | 文档与接口定义 | 字段映射表、点轨迹 schema | 字段、坐标系、评估口径全部确认 |
| M1 | OakInk 短序列执行链路 | 1-2 天 | `main/dataset/*` + 配置 | 短序列索引筛选 + 自动/手动索引入口 | A0 冒烟可稳定跑通 |
| M2 | A1/A2 最小训练闭环 | 3-5 天 | `dexhandmanip_sh.py` | `pt_pos/pt_flow` 观测与奖励接入 | 训练不崩，点误差下降 |
| M3 | A0-A3 对照与闸门决策 | 4-7 天 | 配置/脚本/记录文档 | 对照表 + 结论 | 满足升级 B 的闸门或给出回退结论 |

## 2. 里程碑细化

## M0：规格冻结（必须先过）
- 任务：
  - 确认 OakInk 到当前管线的最小必要字段。
  - 确认第一阶段点轨迹来源（推荐：mesh 锚点离线变换）。
  - 确认评估指标：成功率/失败率/点误差/异常率。
- 交付：
  - `docs/Physgraph改造.md`（研究决策文档）已完成。
  - `docs/environment_baseline.md` 补充接口契约。
- 退出条件（DoD）：
  - 无未决“关键未知”阻塞 M1 开始。

## M1：OakInk 短序列执行链路
- 任务：
  - 复用现有 OakInk adapter，不重写数据读取主逻辑。
  - 增加短序列筛选工具，输出 `hash@stage` 索引。
  - 在单手任务入口支持自动短序列选择（可被手动 `dataIndices` 覆盖）。
- 交付：
  - 可复用短序列筛选工具与 A0 冒烟配置口径。
- 退出条件（DoD）：
  - A0 在短序列上稳定运行，无早期持续崩溃；
  - 数据字段无缺失导致的链路中断。

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
1. 用 `python main/dataset/oakink2_shortlist.py` 产出候选短序列索引。
2. 先跑 `A0_pose_baseline`（OakInk 短序列）验证闭环稳定。
3. 在 A0 稳定基础上推进 A1/A2/A3 对照。

## 5. 文档联动要求
- 每个里程碑结束后必须同步：
  - `docs/next_step_plan.md`：状态与下一步；
  - `docs/environment_baseline.md`：接口与配置变化；
  - `docs/project_brief.md`：目标与阶段结论（简版）。

## 当前执行版（Execution Version）

### 里程碑状态
| 里程碑 | 目标 | 状态 | 核心产出 |
|---|---|---|---|
| M1 | OakInk 短序列执行链路 | 已完成（入口增强） | 新增短序列筛选工具与自动短序列入口 |
| M2 | 点轨迹观测/奖励接入 | 已完成（首版） | `dexhandmanip_sh.py` 支持 `usePointTarget/usePtFlow/useRegionGeom` |
| M2.5 | 单手链路配套 | 已完成 | `TASK_MAP` 注册 RH/LH，`train.py` 注册 `res_rh/lh_*` |
| M3 | A0-A3 对照实验 | 待执行 | 三种子对照、门禁结论 |
| M4 | 文档与结论同步 | 进行中 | 关键文档已统一并更新 |

### 本周执行清单
1. 先跑 `A0_pose_baseline`（OakInk 短序列，1-3 seeds）。
2. 跑 `A1_ptpos`。
3. 跑 `A2_ptpos_ptflow`。
4. 跑 `A3_ptpos_ptflow_region_geom`。
5. 产出 go/no-go。

### A0-A3 最小实验矩阵（OakInk-only）
| 维度 | 设定 |
|---|---|
| 数据集 | OakInk-V2（唯一） |
| 侧别 | RH（主矩阵），LH 可选复核 |
| 短序列策略 | `dataIndices=[oakink_auto_short]`, `oakink_short_topk=1`, `oakink_short_max_frames=180` |
| 训练资源 | `num_envs=512`, `headless=true` |
| 训练步数 | `max_iterations=1200`, `early_stop_epochs=1200` |
| seeds | `42`, `142`, `242` |
| 实验组 | `A0_pose_baseline`, `A1_ptpos`, `A2_ptpos_ptflow`, `A3_ptpos_ptflow_region_geom` |
| 总运行数 | 12（4 组 x 3 seeds） |

### 记录字段（每次运行最少）
1. 元信息：`experiment`、`seed`、`side`、`dexhand`、`oakink_short_*`、代码提交号。
2. 结果指标：成功率、失败率、episode 平均 reward、收敛轮次。
3. 点相关：`reward_pt_pos`、`reward_pt_flow`、`reward_region_geom`（未开启时记录为 0）。
4. 稳定性：`error_buf` 比例、NaN/Inf 发生次数、速度/力异常率（若日志可得）。
5. 对照结论：相对 A0 的均值差、方差、是否满足门禁（A2/A3 稳定优于 A0）。

### 目的（为什么要跑最小矩阵）
- 不是追求单次最高分，而是验证“点轨迹路线在同一数据域内是否稳定优于 pose baseline”。
- 用 3 seeds 控制随机性，避免单 seed 偶然结果导致错误路线决策。
- 为是否进入方案 B（anchor tokens）提供可复现、可审计的门禁证据。

### 示例命令
1. 短序列筛选：`python main/dataset/oakink2_shortlist.py --side right --topk 8 --max-frames 180`
2. 单次模板（A2 示例）：`python main/rl/train.py task=ResDexHand rl_train=ResDexHandPPO side=RH dexhand=inspire headless=true test=false num_envs=512 max_iterations=1200 early_stop_epochs=1200 seed=42 experiment=A2_ptpos_ptflow_s42 dataIndices=[oakink_auto_short] auto_oakink_short=True oakink_short_topk=1 oakink_short_max_frames=180 oakink_data_dir=data/OakInk-v2 oakink_skip=2 task.env.usePointTarget=True task.env.usePtFlow=True task.env.useRegionGeom=False task.env.poseFallback=True`
3. 批量脚本：`powershell -ExecutionPolicy Bypass -File main/rl/run_a0_a3_oakink.ps1 -Mode gate`
4. 汇总脚本（仅汇总，不重复训练）：`powershell -ExecutionPolicy Bypass -File main/rl/run_a0_a3_oakink.ps1 -Mode gate -SkipTrain`
5. 自动产出门禁文件：
   - `runs/analysis/a0_a3_run_metrics.csv`
   - `runs/analysis/a0_a3_group_summary.csv`
   - `runs/analysis/a0_a3_gate_decision.csv`
   - `runs/analysis/a0_a3_summary.md`
6. 手工模板：
   - `docs/templates/a0_a3_results_template.csv`
   - `docs/templates/a0_a3_results_template.md`

### BiH Point-Track Contrast (optional, current data reality)
1. Goal: verify whether point-track rewards/targets improve BiH performance on the same OakInk index.
2. Keep index fixed (for example `083f7@0`) to avoid data-selection noise.
3. Minimal contrast:
   - `BiH_A0_pose_s{seed}`: `usePointTarget=False`, `usePtFlow=False`, `useRegionGeom=False`
   - `BiH_A3_ptpos_ptflow_region_s{seed}`: `usePointTarget=True`, `usePtFlow=True`, `useRegionGeom=True`
4. Suggested seeds: `42,142,242`.
5. Recommended command path: `main/rl/run_a0_a3_oakink.ps1 -Side BiH -DataIndex <hash@stage>`.
