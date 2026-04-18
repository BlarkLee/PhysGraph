#!/usr/bin/env python3
"""
Pick best checkpoints for specific seeds and build a portable package for test=true runs.
"""

import argparse
import csv
import datetime as dt
import json
import math
import re
import shutil
from pathlib import Path


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
        marker = f"_{tag}_"
        i = name.find(marker)
        if i < 0:
            return None
        tail = name[i + len(marker) :]
        return parse_number(tail)

    ep = get("ep")
    sr = get("sr")
    fr = get("fr")
    rew = get("rew")

    return {
        "checkpoint_name": path.name,
        "checkpoint_path": str(path.as_posix()),
        "epoch": int(ep) if ep is not None else None,
        "success_rate": sr,
        "fail_rate": fr,
        "reward": rew,
        "mtime": path.stat().st_mtime,
    }


def ckpt_score(item):
    # "Best" policy: higher SR > lower FR > higher reward > higher epoch > newer file.
    return (
        maybe_num(item.get("success_rate")),
        -maybe_num(item.get("fail_rate")),
        maybe_num(item.get("reward")),
        maybe_num(item.get("epoch"), -1),
        item.get("mtime", -1),
        item.get("checkpoint_name", ""),
    )


def pick_best_ckpt_from_run(run_dir):
    nn_dir = run_dir / "nn"
    if not nn_dir.exists():
        return None

    ckpts = list(nn_dir.glob("*.pth"))
    if not ckpts:
        ckpts = [p for p in nn_dir.iterdir() if p.is_file()]
    if not ckpts:
        return None

    parsed = [extract_ckpt_metrics(p) for p in ckpts]
    return max(parsed, key=ckpt_score)


def find_run_candidates(runs_root, experiment_name):
    if not runs_root.exists():
        return []
    return sorted(
        [d for d in runs_root.iterdir() if d.is_dir() and d.name.startswith(f"{experiment_name}__")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def pick_best_for_seed(runs_root, group, seed):
    exp = f"{group}_s{seed}"
    runs = find_run_candidates(runs_root, exp)
    if not runs:
        return {"seed": seed, "experiment": exp, "status": "missing_run_dir"}

    best = None
    for run_dir in runs:
        chosen = pick_best_ckpt_from_run(run_dir)
        if chosen is None:
            continue
        candidate = {
            "seed": seed,
            "experiment": exp,
            "status": "ok",
            "run_dir": run_dir,
            "chosen": chosen,
            "score": ckpt_score(chosen) + (run_dir.stat().st_mtime,),
        }
        if best is None or candidate["score"] > best["score"]:
            best = candidate

    if best is None:
        return {
            "seed": seed,
            "experiment": exp,
            "status": "missing_checkpoint",
            "run_dir": runs[0],
        }
    return best


def bool_str(flag):
    return "True" if flag else "False"


def write_test_command_files(out_dir, group, selected_rows, args):
    use_pt, use_flow, use_region = GROUP_FLAGS.get(group, (True, True, False))
    if args.use_point_target is not None:
        use_pt = args.use_point_target
    if args.use_pt_flow is not None:
        use_flow = args.use_pt_flow
    if args.use_region_geom is not None:
        use_region = args.use_region_geom
    sh_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Run from PhysGraph repo root on lab host.",
        "",
    ]
    ps1_lines = [
        "Param()",
        "",
        "# Run from PhysGraph repo root on lab host.",
        "",
    ]

    for row in selected_rows:
        seed = row["seed"]
        ckpt_rel = Path("runs/seed_compare_packages") / out_dir.name / "checkpoints" / f"seed{seed}" / row["checkpoint_name"]
        ckpt_unix = f"{args.lab_repo_root}/{ckpt_rel.as_posix()}"
        ckpt_win = f"$RepoRoot\\{str(ckpt_rel).replace('/', '\\')}"
        exp_name = f"{group}_s{seed}_testcmp"

        common_items = [
            "task=ResDexHand",
            "rl_train=ResDexHandPPO",
            f"side={args.side}",
            f"dexhand={args.dexhand}",
            "headless=true",
            "test=true",
            f"seed={seed}",
            f"experiment={exp_name}",
            "dataIndices=[oakink_auto_short]",
            "auto_oakink_short=True",
            "oakink_short_topk=1",
            "oakink_short_max_frames=180",
            f"oakink_data_dir={args.oakink_data_dir}",
            "oakink_skip=2",
            f"task.env.usePointTarget={bool_str(use_pt)}",
            f"task.env.usePtFlow={bool_str(use_flow)}",
            f"task.env.useRegionGeom={bool_str(use_region)}",
            "task.env.poseFallback=True",
        ]

        sh_cmd = "python main/rl/train.py " + " ".join(common_items + [f"checkpoint={ckpt_unix}"])
        sh_lines.extend([f"# seed={seed}", sh_cmd, ""])

        ps_cmd_items = [
            "python",
            "main/rl/train.py",
            *common_items,
            f"checkpoint={ckpt_win}",
        ]
        ps1_lines.append(f"# seed={seed}")
        ps1_lines.append("& " + " ".join([f'"{x}"' if " " in x else x for x in ps_cmd_items]))
        ps1_lines.append("")

    sh_path = out_dir / "run_test_true_compare.sh"
    sh_path.write_text("\n".join(sh_lines), encoding="utf-8", newline="\n")
    ps1_path = out_dir / "run_test_true_compare.ps1"
    ps1_header = ["$RepoRoot = (Get-Location).Path", ""]
    ps1_path.write_text("\n".join(ps1_header + ps1_lines), encoding="utf-8", newline="\n")


def write_manifest_csv(path, rows):
    fields = [
        "group",
        "seed",
        "experiment",
        "status",
        "run_dir",
        "checkpoint_name",
        "checkpoint_path",
        "success_rate",
        "fail_rate",
        "reward",
        "epoch",
        "packaged_checkpoint",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args():
    p = argparse.ArgumentParser(description="Package best checkpoints for selected seeds.")
    p.add_argument("--group", required=True, help="Experiment group prefix, e.g. A2_ptpos_ptflow.")
    p.add_argument("--seeds", default="42,242", help="Comma-separated seeds. Default: 42,242")
    p.add_argument("--runs-root", default="runs", help="Training runs root.")
    p.add_argument("--out-root", default="runs/seed_compare_packages", help="Output package root.")
    p.add_argument("--side", default="RH", help="side override for test command template.")
    p.add_argument("--dexhand", default="inspire", help="dexhand override for test command template.")
    p.add_argument(
        "--use-point-target",
        default=None,
        choices=["True", "False"],
        help="Override task.env.usePointTarget in generated test commands.",
    )
    p.add_argument(
        "--use-pt-flow",
        default=None,
        choices=["True", "False"],
        help="Override task.env.usePtFlow in generated test commands.",
    )
    p.add_argument(
        "--use-region-geom",
        default=None,
        choices=["True", "False"],
        help="Override task.env.useRegionGeom in generated test commands.",
    )
    p.add_argument("--oakink-data-dir", default="data/OakInk-v2", help="OakInk directory used on lab host.")
    p.add_argument(
        "--lab-repo-root",
        default="/path/to/PhysGraph",
        help="Absolute PhysGraph repo path on lab host, used in shell command template.",
    )
    p.add_argument("--archive", action="store_true", help="Also generate a .zip archive next to package dir.")
    p.add_argument("--allow-missing", action="store_true", help="Do not fail when any seed run is missing.")
    args = p.parse_args()
    args.use_point_target = None if args.use_point_target is None else (args.use_point_target == "True")
    args.use_pt_flow = None if args.use_pt_flow is None else (args.use_pt_flow == "True")
    args.use_region_geom = None if args.use_region_geom is None else (args.use_region_geom == "True")
    return args


def main():
    args = parse_args()
    runs_root = Path(args.runs_root)
    out_root = Path(args.out_root)
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    pkg_name = f"{args.group}_seeds_{'-'.join(str(s) for s in seeds)}_{ts}"
    out_dir = out_root / pkg_name
    ckpt_root = out_dir / "checkpoints"
    cfg_root = out_dir / "run_configs"
    ckpt_root.mkdir(parents=True, exist_ok=True)
    cfg_root.mkdir(parents=True, exist_ok=True)

    selected = [pick_best_for_seed(runs_root, args.group, seed) for seed in seeds]

    missing = [r for r in selected if r.get("status") != "ok"]
    if missing and not args.allow_missing:
        details = ", ".join(f"seed={r['seed']}:{r['status']}" for r in missing)
        raise RuntimeError(f"Missing runs/checkpoints: {details}. Re-run with --allow-missing to continue.")

    manifest_rows = []
    packaged_rows = []
    for row in selected:
        seed = row["seed"]
        out_ckpt_rel = ""
        run_dir = row.get("run_dir")
        run_dir_s = run_dir.as_posix() if isinstance(run_dir, Path) else ""

        manifest = {
            "group": args.group,
            "seed": seed,
            "experiment": row.get("experiment", ""),
            "status": row.get("status", ""),
            "run_dir": run_dir_s,
            "checkpoint_name": "",
            "checkpoint_path": "",
            "success_rate": "",
            "fail_rate": "",
            "reward": "",
            "epoch": "",
            "packaged_checkpoint": "",
        }

        if row.get("status") == "ok":
            chosen = row["chosen"]
            src_ckpt = Path(chosen["checkpoint_path"])
            dst_dir = ckpt_root / f"seed{seed}"
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst_ckpt = dst_dir / src_ckpt.name
            shutil.copy2(src_ckpt, dst_ckpt)
            out_ckpt_rel = dst_ckpt.relative_to(out_dir).as_posix()

            src_cfg = Path(run_dir) / "config.yaml"
            if src_cfg.exists():
                shutil.copy2(src_cfg, cfg_root / f"seed{seed}_config.yaml")

            manifest.update(
                {
                    "checkpoint_name": chosen.get("checkpoint_name", ""),
                    "checkpoint_path": chosen.get("checkpoint_path", ""),
                    "success_rate": chosen.get("success_rate", ""),
                    "fail_rate": chosen.get("fail_rate", ""),
                    "reward": chosen.get("reward", ""),
                    "epoch": chosen.get("epoch", ""),
                    "packaged_checkpoint": out_ckpt_rel,
                }
            )
            packaged_rows.append(
                {
                    "seed": seed,
                    "checkpoint_name": chosen.get("checkpoint_name", ""),
                }
            )

        manifest_rows.append(manifest)

    write_manifest_csv(out_dir / "selected_checkpoints.csv", manifest_rows)
    write_test_command_files(out_dir, args.group, packaged_rows, args)

    manifest_json = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "group": args.group,
        "seeds": seeds,
        "runs_root": runs_root.as_posix(),
        "package_dir": out_dir.as_posix(),
        "selection_policy": "max(success_rate, -fail_rate, reward, epoch, checkpoint_mtime)",
        "rows": manifest_rows,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest_json, ensure_ascii=False, indent=2), encoding="utf-8")

    readme_lines = [
        "# Seed Checkpoint Package",
        "",
        f"- Group: `{args.group}`",
        f"- Seeds: `{','.join(str(s) for s in seeds)}`",
        f"- Generated at: `{manifest_json['generated_at']}`",
        "",
        "## Content",
        "- `checkpoints/seed*/`: selected checkpoints",
        "- `run_configs/seed*_config.yaml`: source run config snapshot (if exists)",
        "- `selected_checkpoints.csv`: selected run/checkpoint metadata",
        "- `manifest.json`: machine-readable package metadata",
        "- `run_test_true_compare.sh` / `run_test_true_compare.ps1`: test=true command templates",
        "",
        "## Usage",
        "1. Copy this package directory (or its zip) to target lab-host path.",
        "2. On lab host, enter PhysGraph repo root and adjust checkpoint path / oakink_data_dir if needed.",
        "3. Run one of the generated command files to compare seeds under test=true.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8", newline="\n")

    zip_path = ""
    if args.archive:
        zip_path = shutil.make_archive(str(out_dir), "zip", root_dir=str(out_root), base_dir=pkg_name)

    print(f"[package] created: {out_dir.as_posix()}")
    print(f"[package] csv: {(out_dir / 'selected_checkpoints.csv').as_posix()}")
    print(f"[package] manifest: {(out_dir / 'manifest.json').as_posix()}")
    if zip_path:
        print(f"[package] archive: {Path(zip_path).as_posix()}")


if __name__ == "__main__":
    main()
