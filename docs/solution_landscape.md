# 公开方案扫描与可复用资源地图（执行关联版）

更新时间：2026-04-12

## 1. 核心公开路线
- PhysGraph（策略骨架）：  
  https://arxiv.org/abs/2603.01436
- ManipTrans（稳定奖励主干）：  
  https://arxiv.org/abs/2503.21860  
  https://github.com/ManipTrans/ManipTrans
- Dex4D（点轨迹策略直接先例）：  
  https://arxiv.org/abs/2602.15828
- DexPoint（点云 RL 先例）：  
  https://proceedings.mlr.press/v205/qin23a.html  
  https://github.com/yzqin/dexpoint-release

## 2. 数据与工具链可得性（OakInk-only 执行口径）
- OakInk-V2：
  - 官方站点：https://oakink.net/v2/
  - 代码仓库：https://github.com/oakink/OakInk2
- 点跟踪：
  - CoTracker3 论文：https://arxiv.org/abs/2410.11831
  - 代码：https://github.com/facebookresearch/co-tracker
  - HF 模型：https://hf.co/facebook/cotracker3
  - HF 数据：https://hf.co/datasets/facebook/CoTracker3_Kubric
- 深度估计：
  - Depth Anything V2 论文：https://arxiv.org/abs/2406.09414
  - 代码：https://github.com/DepthAnything/Depth-Anything-V2
  - HF 模型：https://hf.co/depth-anything/Depth-Anything-V2-Large

## 3. 成熟度判断
- 成熟可复用：
  - OakInk 数据读取、分段与标注链路；
  - 点跟踪与深度工具；
  - ManipTrans 的稳定 imitation/reward 主干思路。
- 部分成熟：
  - 点轨迹直接驱动灵巧手奖励（可行，但工程耦合较高）。
- 仍缺成熟一体化：
  - PhysGraph 框架内的 `OakInk + 点轨迹 + 区域接触奖励` 端到端稳定集成方案。

## 4. 对当前执行计划的直接含义
- M1：围绕 OakInk 做短序列筛选与自动入口，不再维护 DexYCB 路径。
- M2：点轨迹生成走离线预处理，不阻塞 RL 训练主循环。
- M3：先评估 A0-A3，再决定是否进入 B（anchor token）。

## 5. 非阻塞增强项（第二阶段再做）
- TAPIR：https://arxiv.org/abs/2306.08637
- 更强轨迹/深度模型替换与融合。
