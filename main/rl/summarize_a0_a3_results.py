#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import math
import re
from pathlib import Path
from statistics import mean, pstdev


DEFAULT_GROUPS = [
    "A0_pose_baseline",
    "A1_ptpos",
    "A2_ptpos_ptflow",
    "A3_ptpos_ptflow_region_geom",
]

GROUP_FLAGS = {
    "A0_pose_baseline": (False, False, False),
    "A1_ptpos": (True, False, False),
    "A2_ptpos_ptflow": (True, True, False),
    "A3_ptpos_ptflow_region_geom": (True, True, True),
}


def maybe_num(value, missing=-math.inf):
    return value if value is not None else missing


def parse_number(token):
    if token is None:
        return None
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", token)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def extract_ckpt_metrics(path):
    name = path.stem

    def get(tag):
        m = re.search(rf"_{tag}_([^_]+)", name)
        return parse_number(m.group(1)) if m else None

    ep = get("ep")
    sr = get("sr")
    fr = get("fr")
    rew = get("rew")
    er = get("Er")
    et = get("Et")
    ej = get("Ej")
    eft = get("Eft")

    return {
        "checkpoint_name": path.name,
        "checkpoint_path": str(path.as_posix()),
        "epoch": int(ep) if ep is not None else None,
        "success_rate": sr,
        "fail_rate": fr,
        "reward": rew,
        "err_rot_er": er,
        "err_trans_et": et,
        "err_joint_ej": ej,
        "err_ft_eft": eft,
    }


def pick_representative_ckpt(ckpt_metrics):
    if not ckpt_metrics:
        return None

    # Prefer terminal checkpoints: highest epoch, then quality tie-breakers.
    with_epoch = [x for x in ckpt_metrics if x["epoch"] is not None]
    if with_epoch:
        return max(
            with_epoch,
            key=lambda x: (
                x["epoch"],
                maybe_num(x["success_rate"]),
                maybe_num(x["reward"]),
            ),
        )

    with_sr = [x for x in ckpt_metrics if x["success_rate"] is not None]
    if with_sr:
        return max(
            with_sr,
            key=lambda x: (
                x["success_rate"],
                maybe_num(x["reward"]),
            ),
        )

    with_rew = [x for x in ckpt_metrics if x["reward"] is not None]
    if with_rew:
        return max(
            with_rew,
            key=lambda x: (
                x["reward"],
                x["epoch"] if x["epoch"] is not None else -1,
            ),
        )

    return sorted(ckpt_metrics, key=lambda x: x["checkpoint_name"])[-1]


def pick_best_run(matches):
    best = None
    for run_dir in matches:
        nn_dir = run_dir / "nn"
        ckpts = []
        if nn_dir.exists():
            ckpts.extend(nn_dir.glob("*.pth"))
            if not ckpts:
                ckpts.extend([p for p in nn_dir.iterdir() if p.is_file()])
        parsed = [extract_ckpt_metrics(p) for p in ckpts]
        chosen = pick_representative_ckpt(parsed)
        if chosen is None:
            continue
        candidate = {
            "run_dir": run_dir,
            "chosen": chosen,
            "score": (
                maybe_num(chosen["epoch"], -1),
                maybe_num(chosen["success_rate"]),
                maybe_num(chosen["reward"]),
                run_dir.stat().st_mtime,
            ),
        }
        if best is None or candidate["score"] > best["score"]:
            best = candidate
    return best


def infer_gate_groups(groups):
    if "A0_pose_baseline" in groups:
        baseline = "A0_pose_baseline"
        candidates = [g for g in ("A2_ptpos_ptflow", "A3_ptpos_ptflow_region_geom") if g in groups]
        return baseline, candidates
    if "BiH_B0_hand_base_only_nopose" in groups:
        baseline = "BiH_B0_hand_base_only_nopose"
        candidates = [g for g in ("BiH_B2_ptpos_ptflow_nopose", "BiH_B3_ptpos_ptflow_region_nopose") if g in groups]
        return baseline, candidates
    if not groups:
        return None, []
    return groups[0], groups[1:]


def list_candidate_runs(runs_root):
    out = {}
    if not runs_root.exists():
        return out
    for d in runs_root.iterdir():
        if not d.is_dir():
            continue
        if "__" not in d.name:
            continue
        exp = d.name.split("__", 1)[0]
        out.setdefault(exp, []).append(d)
    for exp in out:
        out[exp].sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return out


def fmt(x, nd=6):
    if x is None:
        return ""
    return f"{x:.{nd}f}"


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def summarize_group(rows):
    valid = [r for r in rows if r["status"] == "ok" and r["success_rate"] is not None]
    sr = [r["success_rate"] for r in valid]
    fr = [r["fail_rate"] for r in valid if r["fail_rate"] is not None]
    rw = [r["reward"] for r in valid if r["reward"] is not None]
    et = [r["err_trans_et"] for r in valid if r["err_trans_et"] is not None]
    nan_inf = [r["nan_inf_count"] for r in valid if r["nan_inf_count"] is not None]

    def smean(values):
        return mean(values) if values else None

    def sstd(values):
        return pstdev(values) if len(values) > 1 else 0.0 if len(values) == 1 else None

    return {
        "runs_expected": len(rows),
        "runs_valid": len(valid),
        "mean_success_rate": smean(sr),
        "std_success_rate": sstd(sr),
        "mean_fail_rate": smean(fr),
        "std_fail_rate": sstd(fr),
        "mean_reward": smean(rw),
        "std_reward": sstd(rw),
        "mean_err_trans_et": smean(et),
        "std_err_trans_et": sstd(et),
        "sum_nan_inf_count": int(sum(nan_inf)) if nan_inf else 0,
    }


def decide_gate(group_stats, baseline_group, candidate_groups, sr_margin, fr_margin, max_std_sr):
    if baseline_group is None:
        return ("PENDING", "No groups provided for gate.")
    if baseline_group not in group_stats:
        return ("PENDING", f"Missing baseline summary: {baseline_group}.")

    baseline = group_stats[baseline_group]
    if baseline["runs_valid"] == 0:
        return ("PENDING", f"{baseline_group} has no valid runs with success_rate.")
    if not candidate_groups:
        return ("PENDING", "No candidate groups configured for gate.")

    for g in candidate_groups:
        if g not in group_stats:
            continue
        st = group_stats[g]
        if st["runs_valid"] < st["runs_expected"]:
            continue
        if st["mean_success_rate"] is None or st["mean_fail_rate"] is None:
            continue
        if baseline["mean_fail_rate"] is None:
            continue

        stable = (st["std_success_rate"] or 0.0) <= max_std_sr
        better = st["mean_success_rate"] >= baseline["mean_success_rate"] + sr_margin
        not_worse_fail = st["mean_fail_rate"] <= baseline["mean_fail_rate"] + fr_margin

        if stable and better and not_worse_fail:
            reason = (
                f"{g} passes gate: mean_sr={st['mean_success_rate']:.4f} "
                f"({baseline_group}={baseline['mean_success_rate']:.4f}), std_sr={st['std_success_rate']:.4f}, "
                f"mean_fr={st['mean_fail_rate']:.4f} ({baseline_group}={baseline['mean_fail_rate']:.4f})."
            )
            return ("PASS", reason)

    reason = (
        f"No candidate satisfies gate against {baseline_group}. "
        f"Rule: mean_sr >= {baseline_group} + {sr_margin}, "
        f"mean_fr <= {baseline_group} + {fr_margin}, std_sr <= {max_std_sr}."
    )
    return ("FAIL", reason)


def write_markdown(path, run_rows, summary_rows, gate_status, gate_reason, args, baseline_group):
    path.parent.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("# Experiment Result Summary (OakInk-only)\n")
        f.write(f"- Generated at: {now}\n")
        f.write(f"- Runs root: `{args.runs_root}`\n")
        f.write(f"- Seeds: `{','.join(str(s) for s in args.seeds)}`\n")
        f.write(
            f"- Gate baseline: `{baseline_group}`\n"
        )
        f.write(
            f"- Gate rule: `mean_sr >= baseline+{args.sr_margin}`, `mean_fr <= baseline+{args.fr_margin}`, "
            f"`std_sr <= {args.max_std_sr}`\n\n"
        )

        f.write("## Gate Decision\n")
        f.write(f"- Status: **{gate_status}**\n")
        f.write(f"- Reason: {gate_reason}\n\n")

        f.write("## Group Summary\n")
        f.write("| group | expected | valid | mean_sr | std_sr | mean_fr | mean_reward | mean_Et | nan_inf_total |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in summary_rows:
            f.write(
                f"| {r['group']} | {r['runs_expected']} | {r['runs_valid']} | {fmt(r['mean_success_rate'], 4)} "
                f"| {fmt(r['std_success_rate'], 4)} | {fmt(r['mean_fail_rate'], 4)} "
                f"| {fmt(r['mean_reward'], 4)} | {fmt(r['mean_err_trans_et'], 4)} | {r['sum_nan_inf_count']} |\n"
            )
        f.write("\n")

        f.write("## Run-Level Metrics\n")
        f.write(
            "| group | seed | status | success_rate | fail_rate | reward | Et | Er | Ej | Eft | run_dir | checkpoint |\n"
        )
        f.write("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|\n")
        for r in run_rows:
            f.write(
                f"| {r['group']} | {r['seed']} | {r['status']} | {fmt(r['success_rate'], 4)} "
                f"| {fmt(r['fail_rate'], 4)} | {fmt(r['reward'], 4)} | {fmt(r['err_trans_et'], 4)} "
                f"| {fmt(r['err_rot_er'], 4)} | {fmt(r['err_joint_ej'], 4)} | {fmt(r['err_ft_eft'], 4)} "
                f"| {r['run_dir']} | {r['checkpoint_name']} |\n"
            )


def parse_args():
    p = argparse.ArgumentParser(description="Summarize A0-A3 OakInk-only experiment results.")
    p.add_argument("--runs-root", default="runs", help="Root directory of training runs.")
    p.add_argument("--analysis-dir", default="runs/analysis", help="Output directory for summary artifacts.")
    p.add_argument("--groups", default=",".join(DEFAULT_GROUPS), help="Comma-separated experiment groups.")
    p.add_argument("--seeds", default="42,142,242", help="Comma-separated seeds.")
    p.add_argument("--sr-margin", type=float, default=0.0, help="Gate threshold: candidate mean_sr >= A0 + sr_margin")
    p.add_argument("--fr-margin", type=float, default=0.02, help="Gate threshold: candidate mean_fr <= A0 + fr_margin")
    p.add_argument("--max-std-sr", type=float, default=0.08, help="Gate threshold: candidate std_sr <= max_std_sr")
    return p.parse_args()


def main():
    args = parse_args()
    args.groups = [x.strip() for x in args.groups.split(",") if x.strip()]
    args.seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    runs_root = Path(args.runs_root)
    analysis_dir = Path(args.analysis_dir)
    candidates = list_candidate_runs(runs_root)

    run_rows = []
    group_to_rows = {g: [] for g in args.groups}

    for group in args.groups:
        use_pt, use_flow, use_region = GROUP_FLAGS.get(group, (None, None, None))
        for seed in args.seeds:
            exp = f"{group}_s{seed}"
            matches = candidates.get(exp, [])

            row = {
                "group": group,
                "seed": seed,
                "experiment": exp,
                "status": "missing_run_dir",
                "run_dir": "",
                "checkpoint_name": "",
                "checkpoint_path": "",
                "epoch": None,
                "reward": None,
                "success_rate": None,
                "fail_rate": None,
                "err_rot_er": None,
                "err_trans_et": None,
                "err_joint_ej": None,
                "err_ft_eft": None,
                "nan_inf_count": 0,
                "numeric_anomaly_rate": None,
                "usePointTarget": use_pt,
                "usePtFlow": use_flow,
                "useRegionGeom": use_region,
            }

            if matches:
                best_run = pick_best_run(matches)
                if best_run:
                    row["run_dir"] = best_run["run_dir"].as_posix()
                    row.update(best_run["chosen"])
                    row["status"] = "ok"
                else:
                    row["run_dir"] = matches[0].as_posix()
                    row["status"] = "missing_checkpoint"

            run_rows.append(row)
            group_to_rows[group].append(row)

    run_csv_fields = [
        "group",
        "seed",
        "experiment",
        "status",
        "run_dir",
        "checkpoint_name",
        "checkpoint_path",
        "epoch",
        "reward",
        "success_rate",
        "fail_rate",
        "err_trans_et",
        "err_rot_er",
        "err_joint_ej",
        "err_ft_eft",
        "nan_inf_count",
        "numeric_anomaly_rate",
        "usePointTarget",
        "usePtFlow",
        "useRegionGeom",
    ]
    run_csv_path = analysis_dir / "a0_a3_run_metrics.csv"
    write_csv(run_csv_path, run_csv_fields, run_rows)

    group_stats = {}
    summary_rows = []
    for group in args.groups:
        st = summarize_group(group_to_rows[group])
        group_stats[group] = st
        summary_rows.append({"group": group, **st})

    summary_csv_fields = [
        "group",
        "runs_expected",
        "runs_valid",
        "mean_success_rate",
        "std_success_rate",
        "mean_fail_rate",
        "std_fail_rate",
        "mean_reward",
        "std_reward",
        "mean_err_trans_et",
        "std_err_trans_et",
        "sum_nan_inf_count",
    ]
    summary_csv_path = analysis_dir / "a0_a3_group_summary.csv"
    write_csv(summary_csv_path, summary_csv_fields, summary_rows)

    baseline_group, candidate_groups = infer_gate_groups(args.groups)
    gate_status, gate_reason = decide_gate(
        group_stats, baseline_group, candidate_groups, args.sr_margin, args.fr_margin, args.max_std_sr
    )
    gate_csv_path = analysis_dir / "a0_a3_gate_decision.csv"
    write_csv(
        gate_csv_path,
        [
            "gate_status",
            "gate_reason",
            "sr_margin",
            "fr_margin",
            "max_std_sr",
            "baseline_group",
            "candidate_groups",
            "generated_at",
        ],
        [
            {
                "gate_status": gate_status,
                "gate_reason": gate_reason,
                "sr_margin": args.sr_margin,
                "fr_margin": args.fr_margin,
                "max_std_sr": args.max_std_sr,
                "baseline_group": baseline_group or "",
                "candidate_groups": ",".join(candidate_groups),
                "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
            }
        ],
    )

    md_path = analysis_dir / "a0_a3_summary.md"
    write_markdown(md_path, run_rows, summary_rows, gate_status, gate_reason, args, baseline_group)

    print(f"[summary] run-level csv: {run_csv_path.as_posix()}")
    print(f"[summary] group csv: {summary_csv_path.as_posix()}")
    print(f"[summary] gate csv: {gate_csv_path.as_posix()}")
    print(f"[summary] markdown: {md_path.as_posix()}")
    print(f"[summary] gate_status: {gate_status}")


if __name__ == "__main__":
    main()
