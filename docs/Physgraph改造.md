# PhysGraph 改造研究方向（2026-04-11）

## 0. 决策更新（2026-04-12）
- 数据集路线调整：第一阶段数据集锁定为 OakInk（唯一执行数据源）。
- DexYCB 从第一阶段执行面移除，不再纳入实验输入与对照结论。
- 执行原则：优先复用原 OakInk 代码路径与训练思路，只做短序列优先与入口配置增强。
- A0/A1/A2/A3 命名、指标记录与 A2/A3 对 A0 的门禁规则保持不变。

## 1. 问题定义
- 目标：将当前 PhysGraph 管线聚焦到 `OakInk + 点轨迹对象表征`。
- 约束：第一阶段只做单手，优先可训练稳定，再讨论高新颖性结构升级。
- 核心诉求：不再以估计 6D pose 作为主表征，而是直接用稀疏点（或点轨迹）表达物体运动与目标。

## 2. 已确认事实（来自你的文档 + 当前代码）
- 代码边界（第一阶段）：
  - 必改：`main/dataset/*`（OakInk 短序列筛选与入口增强）。
  - 必改：`physgraph_envs/lib/envs/tasks/dexhandmanip_sh.py`（观测/奖励接入点轨迹）。
  - 尽量不改：`dexhandmanip_bih.py`、大规模网络重构。
- 当前环境现状（已读代码）：
  - `main/dataset/factory.py` 已支持 `oakink2`，可直接复用。
  - `dexhandmanip_sh.py` 的 `target` 仍以 `delta_manip_obj_pos/quat/vel` 为主，包含 `bps` 特征。
  - 现有 `compute_imitation_reward` 仍是 wrist/joints + object pose/vel + force 组合，不是点轨迹主导。
- 实验门禁（AGENTS.md）：
  - 必做 A0/A1/A2/A3 命名与指标记录；
  - A2/A3 未稳定优于 A0，不进入 anchor-token 方案 B。

## 3. 假设与关键未知
### 3.1 假设
  - OakInk 字段足以支持你需要的最小数据项：手部轨迹 + 物体状态 + 图像/深度可用于点生成。
- 第一阶段可接受“离线生成点轨迹缓存”，而不是把视频生成/点跟踪/深度提升塞进 RL 在线循环。

### 3.2 未知（会实质影响设计）
- OakInk 到当前仿真坐标系的对齐复杂度（相机系/世界系/手系切换）。
- 点轨迹来源优先级：`mesh锚点投影` vs `RGBD重建` vs `视频预测轨迹`。
- 锚点数 `K` 与训练稳定性/性能上限的平衡区间。
- region/contact 奖励在你任务上的“最早可开启强度”。

## 4. 公开方案扫描（论文 + GitHub + HF）

## 4.1 与本项目直接相关的主线
- PhysGraph（arXiv 2603.01436）：图 token + 结构偏置 + 接触建模骨架，适合作为“策略壳”保留。  
  https://arxiv.org/abs/2603.01436
- ManipTrans（CVPR 2025）：明确给出 hand imitation + object following + contact force 的稳定奖励范式。  
  https://arxiv.org/abs/2503.21860  
  https://github.com/ManipTrans/ManipTrans
- Dex4D（arXiv 2602.15828）：以 point tracks 驱动 dexterous policy 的直接先例。  
  https://arxiv.org/abs/2602.15828
- DexPoint（CoRL 2022）：点云对象表征 + RL + sim2real，可借鉴对象编码与训练组织方式。  
  https://proceedings.mlr.press/v205/qin23a.html  
  https://github.com/yzqin/dexpoint-release

## 4.2 数据与工具链可得性
- OakInk 数据与原始工程链路（当前项目已内置）。  
- 点跟踪工具（CoTracker3，代码+模型+HF数据）：
  - https://arxiv.org/abs/2410.11831
  - https://github.com/facebookresearch/co-tracker
  - https://hf.co/facebook/cotracker3
  - https://hf.co/datasets/facebook/CoTracker3_Kubric
- 深度估计工具（Depth Anything V2）：
  - https://arxiv.org/abs/2406.09414
  - https://github.com/DepthAnything/Depth-Anything-V2
  - https://hf.co/depth-anything/Depth-Anything-V2-Large

## 4.3 成熟度判断
- 成熟可复用：
  - OakInk 数据读取与标注体系（当前仓库已接入）；
  - 点跟踪/深度模型工具链；
  - ManipTrans 的稳定训练主干思想。
- 部分成熟：
  - 点轨迹直接驱动灵巧手奖励（已有 Dex4D 方向，但生态不如 pose 管线成熟）。
- 尚无“现成一体化成熟方案”：
  - 在 PhysGraph 结构内，把 OakInk + 点轨迹 + 区域接触奖励整合成稳定训练闭环。

## 4.4 执行边界（补充）
- 第一阶段所有训练与评估仅基于 OakInk。
- 任何 DexYCB 相关脚本、适配器或缓存不作为本阶段里程碑交付要求。

## 5. 候选路线（3 选）

## 方案 A：保守（单 object token，不改主图）
- 做法：
  - 保持现有 token 结构；
  - 用 `K` 点轨迹误差（`pt_pos/pt_flow`）替代 object pose 主奖励；
  - 点对经轻量编码器压成 object feature 回填。
- 预期收益：最快验证“点表征是否优于 pose 基线”。
- 复杂度：低。
- 依赖：OakInk 短序列筛选 + 点缓存 + `dexhandmanip_sh.py` 奖励替换。
- 失败模式：提升有限，可能只验证到“可用”而非“显著更好”。
- 验证路径：A0→A1→A2（你的现有命名正好匹配）。
- 新颖性潜力：中等偏低。
- 新颖性风险：低（容易被视为工程迁移）。

## 方案 B：中间（少量 anchor token + 动态边）
- 做法：
  - 引入少量 anchor token（如 8/16）；
  - fingertip 与近邻 anchor 建立动态 proximity/contact edges；
  - 保留原 bias 机制，做增量扩展。
- 预期收益：更贴近 PhysGraph 精神，接触表达能力明显提升。
- 复杂度：中。
- 依赖：A 稳定结果、token 映射与 bias 对齐改造。
- 失败模式：训练不稳定、调参周期拉长。
- 验证路径：A2/A3 稳定优于 A0 后再切入。
- 新颖性潜力：高于 A。
- 新颖性风险：中。

## 方案 C：激进（全 anchor 图 + 全量接触几何/力约束）
- 做法：
  - 对象全面锚点化，区域/几何/非目标接触/力稳定全开启。
- 预期收益：表达上限最高。
- 复杂度：高。
- 依赖：完善点质量、课程化权重、稳定训练工程。
- 失败模式：探索早期崩溃，成本高，迭代慢。
- 验证路径：仅在 B 成熟后尝试。
- 新颖性潜力：最高。
- 新颖性风险：最高（高失败概率）。

## 6. 横向对比
| 方案 | 工程可行性 | 训练稳定性 | 上限潜力 | 新颖性 | 推荐时机 |
|---|---:|---:|---:|---:|---|
| A | 5 | 4 | 3 | 2.5 | 立即 |
| B | 3.5 | 3 | 4.5 | 4 | A 通过后 |
| C | 2 | 2 | 5 | 5 | B 稳定后 |

## 7. 新颖性风险评估（你这个课题真正的“可发点”）
- 若只做 A：价值在“高质量复现+迁移”，论文新颖性有限，但能快速建立可信基线。
- 若做到 B 并稳定优于 A0/A2：具备更强研究说服力，因为体现了 PhysGraph 风格与点轨迹接触建模的组合优势。
- 若直接 C：理论很强，但高风险导致周期不可控，不建议首发路线。

## 8. 推荐下一步（编码前决策版）

## 8.1 推荐路线
- 先执行方案 A（单手 + 最小改动）；
- 同时在接口层预留 B 的锚点 token 扩展位；
- 明确闸门：仅当 A2 或 A3 稳定优于 A0，再进入 B。

## 8.2 首个原型应做什么
- 原型：`A1_ptpos`（先不引入复杂 region/force 惩罚），打通从 OakInk 到 `pt_pos reward` 的闭环。
- 目标：确认训练不崩、能收敛、点误差下降趋势正常。

## 8.3 首先要测什么
- 必测：
  - 成功率/失败率；
  - 点误差（可见性加权）；
  - 数值稳定性（速度/力异常率）。
- 建议加：
  - 奖励项分量曲线（便于看哪一项压死探索）。

## 8.4 开始编码前必须先拍板的一个决策
- 决策项：`第一阶段点轨迹来源` 只选一种主线（推荐优先级）：
  1. 仿真 mesh 锚点离线变换（推荐）；
  2. RGBD 重建锚点（次选）；
  3. 视频生成+跟踪+深度提升（仅作真实部署链路，不做第一阶段训练阻塞）。

## 9. 与现有文档的关系
- `docs/solution_landscape.md`：保留作为外部方案地图。
- `docs/approach_comparison.md`：保留作为 A/B/C 版本对比。
- `docs/next_step_plan.md`：作为执行计划与里程碑跟踪。
- 本文：作为“研究决策总文档”，用于编码前对齐方向与风险。

## 当前执行版（Execution Version）

### 已实现能力
- 数据层：
  - `<hash5>@<stage>` OakInk 索引解析
  - OakInk 短序列筛选脚本与自动入口
  - 点轨迹字段：`obj_points_t / obj_points_target_t / obj_points_mask_t`
- 环境层：
  - `dexhandmanip_sh.py` 支持点轨迹观测与奖励开关
  - 保留 pose 路径用于 A0 对照与回归
- 训练链路：
  - 注册 `ResDexHandRH/LH`
  - 注册 `res_rh/lh_*` 网络与模型名

### 执行策略（A0-A3）
- A0：pose baseline
- A1：`pt_pos`
- A2：`pt_pos + pt_flow`
- A3：`pt_pos + pt_flow + region_geom`

### 风险与缓解
- 坐标系风险（最高）：先做可视化对齐，再长训。
- 奖励风险：按 A1->A2->A3 课程化启用，不提前强开复杂项。
- 工程风险：保持 `poseFallback`，确保可随时回归 A0。

### 下一步
1. 执行 A0-A3 三种子对照并汇总指标。
2. 输出 go/no-go（是否进入方案 B）决策。
3. 若 no-go，按“数据对齐 -> 奖励权重 -> 点采样质量”顺序排查。
