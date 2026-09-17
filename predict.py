"""
Single-frame inference demo.

    python predict.py --dataset DAUB --weights logs/DAUB_best.pth \
        --image datasets/DAUB/data21/482.bmp \
        --motion_root datasets/motion_difference_map_DAUB \
        --out results/pred_482.png

The previous frame (t-1) is located automatically next to `--image`; the motion difference map
M_t is read from `<motion_root>/<sequence>/<frame_id>` (see motion_diff/ for how to generate it).
Set `--motion_root ''` to run the detector without the explicit motion prior.
"""
import argparse
import importlib
import os

import cv2
import numpy as np
import torch
from PIL import Image

from utils.utils import cvtColor, get_classes, preprocess_input, resize_image
from utils.utils_bbox import decode_outputs, non_max_suppression

CONFIGS = {
    # module, default input (H, W), motion pad value, frame id width for the motion map, motion ext
    "DAUB": ("nets.MPR_DAUB", [512, 512], 128, (4,), ".bmp", "utils.dataloader_for_DAUB"),
    "IRDST-H": ("nets.MPR_IRDSTH", [512, 512], 0, (4,), ".bmp", "utils.dataloader_for_IRDSTH"),
    "IRSTD-UAV": ("nets.MPR_IRSTD_UAV", [512, 640], 128, (8, 4), ".png", "utils.dataloader_for_IRSTD_UAV"),
}
NUM_FRAME = 2


def parse_args():
    p = argparse.ArgumentParser(description="MPR-Net single-frame inference")
    p.add_argument("--dataset", required=True, choices=list(CONFIGS))
    p.add_argument("--weights", required=True)
    p.add_argument("--image", required=True, help="path of frame t, e.g. datasets/DAUB/data21/482.bmp")
    p.add_argument("--motion_root", default="", help="root of the motion difference maps ('' = no motion prior)")
    p.add_argument("--classes_path", default="model_data/classes.txt")
    p.add_argument("--input_shape", type=int, nargs=2, default=None, metavar=("H", "W"))
    p.add_argument("--confidence", type=float, default=0.5)
    p.add_argument("--nms_iou", type=float, default=0.65)
    p.add_argument("--out", default="results/pred.png")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def frame_paths(image_path):
    """<dir>/<id>.<ext> -> [<dir>/<id-1>.<ext>, <dir>/<id>.<ext>] (zero-padding width preserved)."""
    dir_path = os.path.dirname(image_path)
    stem, ext = os.path.splitext(os.path.basename(image_path))
    width = len(stem) if stem.startswith("0") and len(stem) > 1 else 0
    index = int(stem)
    return [os.path.join(dir_path, f"{max(i, 0):0{width}d}{ext}") for i in range(index - NUM_FRAME + 1, index + 1)]


def load_motion(image_path, motion_root, input_shape, pad_value, id_widths, ext, preprocess_motion_diff):
    h, w = input_shape
    seq = os.path.basename(os.path.dirname(image_path))
    image_id = int(os.path.splitext(os.path.basename(image_path))[0])
    iw, ih = Image.open(image_path).size
    scale = min(w / iw, h / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    dx, dy = (w - nw) // 2, (h - nh) // 2

    maps = []
    for k in range(NUM_FRAME):
        fid = max(image_id - k, 0)
        m = None
        for width in id_widths:
            path = os.path.join(motion_root, seq, f"{fid:0{width}d}{ext}")
            if os.path.exists(path):
                m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                break
        if m is None:
            print(f"[warn] motion map for frame {fid} not found under {os.path.join(motion_root, seq)}, using zeros")
            m = np.zeros((h, w), dtype=np.uint8)
        else:
            m = cv2.resize(m, (nw, nh), interpolation=cv2.INTER_LINEAR)
            padded = np.full((h, w), pad_value, dtype=np.uint8)
            padded[dy:dy + nh, dx:dx + nw] = m
            m = padded
        maps.append(m)
    maps = np.array(maps[::-1])
    maps = np.array([preprocess_motion_diff(m.astype(np.float32)) for m in maps], dtype=np.float32)  # (T, H, W, 3)
    return torch.from_numpy(np.transpose(maps, (3, 0, 1, 2))[None])  # (1, 3, T, H, W)


if __name__ == "__main__":
    args = parse_args()
    module_name, default_shape, pad_value, id_widths, motion_ext, loader_module = CONFIGS[args.dataset]
    input_shape = args.input_shape or default_shape
    MPR = importlib.import_module(module_name).MPR
    preprocess_motion_diff = importlib.import_module(loader_module).preprocess_motion_diff

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    class_names, num_classes = get_classes(args.classes_path)

    net = MPR(num_classes, num_frame=NUM_FRAME)
    state_dict = torch.load(args.weights, map_location=device)
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("target_encoder.clip_model")}
    net.load_state_dict(state_dict, strict=False)
    net = net.eval().to(device)

    # ---- frames t-1, t ----
    images = [Image.open(p) for p in frame_paths(args.image)]
    image_shape = np.array(np.shape(images[0])[0:2])
    images = [cvtColor(im) for im in images]
    data = [resize_image(im, (input_shape[1], input_shape[0]), True) for im in images]
    data = [np.transpose(preprocess_input(np.array(im, dtype="float32")), (2, 0, 1)) for im in data]
    frames = torch.from_numpy(np.expand_dims(np.stack(data, axis=1), 0)).to(device)

    motion = None
    if args.motion_root:
        motion = load_motion(args.image, args.motion_root, input_shape, pad_value, id_widths, motion_ext,
                             preprocess_motion_diff).to(device)

    with torch.no_grad():
        outputs = net(frames, motion_diffs=motion)
        outputs = decode_outputs(outputs, input_shape)
        outputs = non_max_suppression(outputs, num_classes, input_shape, image_shape, True,
                                      conf_thres=args.confidence, nms_thres=args.nms_iou)

    canvas = cv2.cvtColor(np.array(images[-1]), cv2.COLOR_RGB2BGR)
    if outputs[0] is None:
        print("No target detected.")
    else:
        labels = outputs[0][:, 6].astype("int32")
        scores = outputs[0][:, 4] * outputs[0][:, 5]
        boxes = outputs[0][:, :4]
        for c, s, (top, left, bottom, right) in zip(labels, scores, boxes):
            x1, y1, x2, y2 = int(left), int(top), int(right), int(bottom)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 1)
            cv2.putText(canvas, f"{class_names[c]} {s:.2f}", (x1, max(y1 - 3, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
            print(f"{class_names[c]} {s:.3f} [{x1}, {y1}, {x2}, {y2}]")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cv2.imwrite(args.out, canvas)
    print(f"Saved to {args.out}")
