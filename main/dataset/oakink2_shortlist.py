import argparse
import ast
import json
import os
import pickle
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple


@dataclass
class OakInkStageStat:
    index: str
    seq_hash: str
    stage: int
    length: int
    anno_path: str
    has_retargeted: bool


def _extract_seq_hash(anno_path: str) -> str:
    name = os.path.splitext(os.path.basename(anno_path))[0]
    parts = name.split("_")
    if len(parts) > 5 and len(parts[5]) >= 5:
        return parts[5][:5]
    return name[-5:]


def _to_range(value) -> Optional[Tuple[int, int]]:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    begin, end = int(value[0]), int(value[1])
    return begin, end


def _intersection(a: Tuple[int, int], b: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    begin = max(a[0], b[0])
    end = min(a[1], b[1])
    if begin >= end:
        return None
    return begin, end


def _count_frames_in_range(frame_ids: Sequence[int], valid_range: Tuple[int, int]) -> int:
    begin, end = valid_range
    return sum(1 for fid in frame_ids if begin <= fid <= end)


def _iter_stage_ranges(program_info_path: str) -> Iterable[Tuple[int, Optional[Tuple[int, int]], Optional[Tuple[int, int]]]]:
    with open(program_info_path, "r", encoding="utf-8") as f:
        raw_info = json.load(f)

    for stage, k in enumerate(raw_info.keys()):
        try:
            seg_pair = ast.literal_eval(k)
        except (SyntaxError, ValueError):
            continue
        if not isinstance(seg_pair, (list, tuple)) or len(seg_pair) != 2:
            continue
        left_range = _to_range(seg_pair[0])
        right_range = _to_range(seg_pair[1])
        yield stage, left_range, right_range


def collect_oakink_stage_stats(
    *,
    data_dir: str = "data/OakInk-v2",
    side: str = "right",
    skip: int = 2,
    retarget_root: Optional[str] = None,
) -> List[OakInkStageStat]:
    if side not in ("left", "right"):
        raise ValueError(f"Invalid side: {side}")
    if skip <= 0:
        raise ValueError(f"skip must be > 0, got {skip}")

    anno_dir = os.path.join(data_dir, "anno_preview")
    program_dir = os.path.join(data_dir, "program", "program_info")
    if not os.path.isdir(anno_dir):
        raise FileNotFoundError(f"OakInk anno directory not found: {anno_dir}")
    if not os.path.isdir(program_dir):
        raise FileNotFoundError(f"OakInk program directory not found: {program_dir}")

    side_is_right = side == "right"
    stage_stats: List[OakInkStageStat] = []
    anno_files = sorted(
        [os.path.join(anno_dir, n) for n in os.listdir(anno_dir) if n.endswith(".pkl")]
    )

    for anno_path in anno_files:
        stem = os.path.splitext(os.path.basename(anno_path))[0]
        program_info_path = os.path.join(program_dir, f"{stem}.json")
        if not os.path.exists(program_info_path):
            continue

        with open(anno_path, "rb") as f:
            anno = pickle.load(f)

        frame_ids = anno.get("mocap_frame_id_list", [])
        if not frame_ids:
            continue
        frame_ids = frame_ids[::skip]
        seq_hash = _extract_seq_hash(anno_path)

        for stage, left_range, right_range in _iter_stage_ranges(program_info_path):
            primary = right_range if side_is_right else left_range
            secondary = left_range if side_is_right else right_range
            if primary is None:
                continue
            valid_range = primary if secondary is None else _intersection(primary, secondary)
            if valid_range is None:
                continue

            length = _count_frames_in_range(frame_ids, valid_range)
            if length <= 0:
                continue

            has_retargeted = False
            if retarget_root:
                retarget_path = os.path.join(retarget_root, f"{stem}@{stage}.pkl")
                has_retargeted = os.path.exists(retarget_path)

            stage_stats.append(
                OakInkStageStat(
                    index=f"{seq_hash}@{stage}",
                    seq_hash=seq_hash,
                    stage=stage,
                    length=length,
                    anno_path=anno_path,
                    has_retargeted=has_retargeted,
                )
            )

    return stage_stats


def select_oakink_short_indices(
    *,
    data_dir: str = "data/OakInk-v2",
    side: str = "right",
    skip: int = 2,
    topk: int = 1,
    max_frames: Optional[int] = None,
    require_retargeted: bool = False,
    retarget_root: Optional[str] = None,
) -> List[OakInkStageStat]:
    if topk <= 0:
        return []

    stage_stats = collect_oakink_stage_stats(
        data_dir=data_dir,
        side=side,
        skip=skip,
        retarget_root=retarget_root,
    )
    if max_frames is not None:
        stage_stats = [s for s in stage_stats if s.length <= max_frames]
    if require_retargeted:
        stage_stats = [s for s in stage_stats if s.has_retargeted]

    stage_stats.sort(key=lambda s: (s.length, s.index))
    return stage_stats[:topk]


def main():
    parser = argparse.ArgumentParser(description="Select short OakInk sequence-stage indices.")
    parser.add_argument("--data-dir", default="data/OakInk-v2")
    parser.add_argument("--side", choices=["left", "right"], default="right")
    parser.add_argument("--skip", type=int, default=2)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=180)
    parser.add_argument("--require-retargeted", action="store_true")
    parser.add_argument("--retarget-root", default="")
    args = parser.parse_args()

    retarget_root = args.retarget_root or None
    selected = select_oakink_short_indices(
        data_dir=args.data_dir,
        side=args.side,
        skip=args.skip,
        topk=args.topk,
        max_frames=args.max_frames,
        require_retargeted=args.require_retargeted,
        retarget_root=retarget_root,
    )

    indices = [s.index for s in selected]
    print("indices:", indices)
    if selected:
        print("details:")
        for s in selected:
            print(
                f"  {s.index} | len={s.length} | retargeted={s.has_retargeted} | anno={os.path.basename(s.anno_path)}"
            )


if __name__ == "__main__":
    main()
