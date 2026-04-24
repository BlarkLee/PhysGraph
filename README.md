# PhysGraph Adaptation (OakInk Point-Track Ablations)

This repository is an adaptation of the original [PhysGraph](https://github.com/BlarkLee/PhysGraph) project for staged ablation studies.

Current stage focus:
- Dataset scope: OakInk-v2 only (phase-1)
- Task scope: single-hand first, bimanual as contrast
- Method scope: point-track representation and reward ablations (`A0` to `A3`, plus optional `BiH B0-B3`)

## Project Status

- This is a research adaptation branch, not an official mirror of upstream PhysGraph.
- The repository is actively used for controlled ablation experiments and engineering validation.
- Main experiment goal: determine whether point-track reward components provide stable gains over baseline settings.

## Relation to Upstream PhysGraph

- Upstream paper/repo introduces PhysGraph for bimanual dexterous hand-tool-object manipulation.
- This adaptation keeps the training/environment backbone where practical and modifies:
  - dataset entry and shortlist workflow for OakInk-only phase-1
  - point-track observation/reward controls
  - ablation scripts and result summarization utilities
- Upstream contribution attribution remains with the original PhysGraph authors.

## What Is Implemented Here

### 1) OakInk-only execution path (phase-1)
- OakInk shortlist utility: `main/dataset/oakink2_shortlist.py`
- Auto shortlist options in task config and train commands
- Reduced phase-1 variance by fixing dataset domain to OakInk

### 2) Point-track ablation controls
Config flags (task env):
- `usePointTarget`
- `usePtFlow`
- `useRegionGeom`
- `poseFallback`
- `pointTrackK`
- reward weights and betas (`w*`, `ptPosBeta`, `ptFlowBeta`)

### 3) A0-A3 experiment matrix
- `A0_pose_baseline`
- `A1_ptpos`
- `A2_ptpos_ptflow`
- `A3_ptpos_ptflow_region_geom`

Batch script:
- `main/rl/run_a0_a3_oakink.ps1`

Outputs:
- `runs/analysis/a0_a3_run_metrics.csv`
- `runs/analysis/a0_a3_group_summary.csv`
- `runs/analysis/a0_a3_gate_decision.csv`
- `runs/analysis/a0_a3_summary.md`

## Repository Layout (Key Paths)

- `main/dataset/`:
  OakInk dataset adapters, shortlist utility, and data prep utilities
- `main/rl/`:
  train entry, batch ablation script, summary script
- `main/cfg/`:
  task and RL config overrides used by ablations
- `physgraph_envs/lib/envs/tasks/`:
  single-hand and bimanual task logic, point-track reward integration
- `docs/`:
  project brief, baseline docs, execution plan, result templates

## Setup

Follow the upstream PhysGraph dependency stack for IsaacGym + Python 3.8 environment, then install this repo:

```bash
pip install -r requirements.txt
pip install -e .
```

Notes:
- IsaacGym Preview 4 is required.
- OakInk-v2 data should be prepared under `data/OakInk-v2`.
- If object assets/URDF preprocessing is missing, refer to utilities under `physgraph_envs/lib/utils/`.

## Quick Start

### 1) (Optional) shortlist OakInk sequences
```bash
python main/dataset/oakink2_shortlist.py --side right --topk 8 --max-frames 180
```

### 2) Run A0-A3 batch
```powershell
powershell -ExecutionPolicy Bypass -File main/rl/run_a0_a3_oakink.ps1 -Mode gate
```

### 3) Summary-only rerun
```powershell
powershell -ExecutionPolicy Bypass -File main/rl/run_a0_a3_oakink.ps1 -Mode gate -SkipTrain
```

### 4) Optional bimanual contrast
```powershell
powershell -ExecutionPolicy Bypass -File main/rl/run_a0_a3_oakink.ps1 -Mode gate -Side BiH -DataIndex 083f7@0
```

## Reproducibility and Reporting

Recommended minimum reporting per run:
- config delta vs baseline
- seed and checkpoint
- success/fail rates
- reward and strict metrics (`Et`, `Er`, `Ej`, `Eft`)
- numerical anomaly counts (`NaN/Inf`, if available)

Recommended comparison protocol:
- compare groups using matched seeds
- report mean/std across seeds
- avoid mixing best checkpoints from different seeds for fair visual comparison

## Citation

If you use the original PhysGraph method, cite the upstream work:

```bibtex
@misc{physgraph,
  title={PhysGraph: Physically-Grounded Graph-Transformer Policies for Bimanual Dexterous Hand-Tool-Object Manipulation},
  author={Runfa Blark Li and David Kim and Xinshuang Liu and Keito Suzuki and Dwait Bhatt and Nikola Raicevic and Xin Lin and Ki Myung Brian Lee and Nikolay Atanasov and Truong Nguyen},
  year={2026},
  eprint={2603.01436},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2603.01436}
}
```

If this adaptation branch is used in a report/paper, also state:
- it is derived from PhysGraph
- exact commit hash used
- local ablation protocol and deviations from upstream

## License and Acknowledgement

- Please follow the license terms in this repository and the upstream PhysGraph repository.
- Thanks to OakInk-v2 and related open-source tooling used in data and training pipelines.

