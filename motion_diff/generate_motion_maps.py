"""
Pre-compute the explicit motion priors M_t (motion difference maps) for a whole dataset.

    # DAUB-R      (lambda = 0.5, data5 ... data22 folders, *.bmp)
    python motion_diff/generate_motion_maps.py --dataset DAUB \
        --root datasets/DAUB --out datasets/motion_difference_map_DAUB

    # IRDST-H     (lambda = 0.5, numeric sequence folders, *.bmp)
    python motion_diff/generate_motion_maps.py --dataset IRDST-H \
        --root datasets/IRDST-H/images --out datasets/motion_difference_map_IRDST-H

    # IRSTD-UAV   (lambda = 1.0 -> compensated differencing only, numeric sequence folders, *.png)
    python motion_diff/generate_motion_maps.py --dataset IRSTD-UAV \
        --root datasets/IRSTD-UAV/images --out datasets/motion_difference_map_IRSTD-UAV

Output layout: <out>/<sequence>/<frame_id:04d>.<bmp|png>, one 8-bit map per frame.
The first / last frame of a sequence (no t-1 / t+1) reuse the map of their neighbour.
Use `--lam` to reproduce the differencing ablation of Table VIII (0 = raw, 0.5 = fusion, 1 = compensated).
"""
import argparse
import os
import re
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from explicit_motion_modeling import PRESETS, motion_prior_three_frames, motion_prior_to_uint8, replace  # noqa: E402

DATASET_IO = {
    # (input extensions, output extension, only folders that match this regex)
    "DAUB": ((".bmp", ".jpg", ".jpeg", ".png"), ".bmp", r"^data\d+$"),
    "IRDST-H": ((".bmp", ".jpg", ".jpeg", ".png"), ".bmp", r"^\d+$"),
    "IRSTD-UAV": ((".png",), ".png", r"^\d+$"),
}


def _frame_id(filename: str):
    m = re.search(r"\d+", os.path.splitext(os.path.basename(filename))[0])
    return int(m.group()) if m else None


def list_sequence(folder: str, exts):
    items = []
    for f in os.listdir(folder):
        if os.path.splitext(f)[1].lower() in exts and os.path.isfile(os.path.join(folder, f)):
            fid = _frame_id(f)
            if fid is not None:
                items.append((fid, os.path.join(folder, f)))
    items.sort(key=lambda x: x[0])
    return items


def process_sequence(seq_dir: str, out_dir: str, cfg, exts, out_ext: str, lam):
    items = list_sequence(seq_dir, exts)
    if len(items) < 3:
        print(f"  [skip] {seq_dir}: need at least 3 frames, found {len(items)}")
        return 0
    os.makedirs(out_dir, exist_ok=True)
    ids = [fid for fid, _ in items]

    done = 0
    for i in range(1, len(items) - 1):
        frames = [cv2.imread(items[j][1]) for j in (i - 1, i, i + 1)]
        if any(f is None for f in frames):
            print(f"  [warn] failed to read a frame around id {ids[i]}, skipped")
            continue
        m_t, _, _, _ = motion_prior_three_frames(frames[0], frames[1], frames[2], cfg, lam)
        cv2.imwrite(os.path.join(out_dir, f"{ids[i]:04d}{out_ext}"), motion_prior_to_uint8(m_t))
        done += 1
        if done % 100 == 0:
            print(f"  {done}/{len(items) - 2} frames")

    # boundary frames: copy the map of the closest interior frame
    for src_id, dst_id in ((ids[1], ids[0]), (ids[-2], ids[-1])):
        src = os.path.join(out_dir, f"{src_id:04d}{out_ext}")
        dst = os.path.join(out_dir, f"{dst_id:04d}{out_ext}")
        if os.path.exists(src) and src != dst:
            cv2.imwrite(dst, cv2.imread(src, cv2.IMREAD_GRAYSCALE))
    return done


def main():
    p = argparse.ArgumentParser(description="Generate explicit motion priors (motion difference maps)")
    p.add_argument("--dataset", required=True, choices=list(PRESETS))
    p.add_argument("--root", required=True, help="directory that contains the sequence folders")
    p.add_argument("--out", required=True, help="output root for the motion difference maps")
    p.add_argument("--lam", type=float, default=None,
                   help="override the compensated-difference weight lambda (default: dataset preset)")
    p.add_argument("--gamma", type=float, default=None, help="override the Mexican-hat mixing weight gamma")
    p.add_argument("--sequences", nargs="*", default=None, help="only process these sequence folders")
    args = p.parse_args()

    cfg = PRESETS[args.dataset]
    if args.gamma is not None:
        cfg = replace(cfg, gamma=args.gamma)
    exts, out_ext, folder_regex = DATASET_IO[args.dataset]

    folders = [f for f in os.listdir(args.root)
               if os.path.isdir(os.path.join(args.root, f)) and re.match(folder_regex, f)]
    if args.sequences:
        folders = [f for f in folders if f in set(args.sequences)]
    folders.sort(key=lambda f: int(re.search(r"\d+", f).group()))
    if not folders:
        raise SystemExit(f"no sequence folder matching {folder_regex!r} found under {args.root}")

    lam = cfg.lam if args.lam is None else args.lam
    print(f"dataset={args.dataset}  lambda={lam}  gamma={cfg.gamma}  sequences={len(folders)}")
    total = 0
    for folder in folders:
        print(f"Processing {folder}")
        total += process_sequence(os.path.join(args.root, folder), os.path.join(args.out, folder), cfg, exts, out_ext, lam)
    print(f"All done: {total} motion maps written to {args.out}")


if __name__ == "__main__":
    main()
