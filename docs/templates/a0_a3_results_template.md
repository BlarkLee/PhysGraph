# A0-A3 Results Template (OakInk-only)

## Experiment Scope
- Dataset: `OakInk-v2`
- Side: `RH` (LH optional replay)
- Seeds: `42, 142, 242`
- Groups: `A0_pose_baseline`, `A1_ptpos`, `A2_ptpos_ptflow`, `A3_ptpos_ptflow_region_geom`
- Total runs: `12`

## Run-Level Checklist
1. Fill one row per run in CSV.
2. Keep `status` in `{ok, missing_run_dir, missing_checkpoint, failed, pending}`.
3. Required metrics: `success_rate`, `fail_rate`, `reward`.
4. Recommended metrics: `err_trans_et`, `err_rot_er`, `err_joint_ej`, `err_ft_eft`.
5. Numerical safety: `nan_inf_count`, `numeric_anomaly_rate`.

## Group Summary
| group | runs_expected | runs_valid | mean_sr | std_sr | mean_fr | mean_reward | mean_Et | nan_inf_total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A0_pose_baseline | 3 |  |  |  |  |  |  |  |
| A1_ptpos | 3 |  |  |  |  |  |  |  |
| A2_ptpos_ptflow | 3 |  |  |  |  |  |  |  |
| A3_ptpos_ptflow_region_geom | 3 |  |  |  |  |  |  |  |

## Gate Decision
- Rule:
  - `mean_sr(candidate) >= mean_sr(A0) + sr_margin`
  - `mean_fr(candidate) <= mean_fr(A0) + fr_margin`
  - `std_sr(candidate) <= max_std_sr`
- Candidate: `A2` or `A3`
- Decision: `{{GO_OR_NO_GO_OR_PENDING}}`
- Evidence: `{{AUTO_FILLED_OR_MANUAL_NOTE}}`
