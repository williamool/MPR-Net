"""
Evaluate MPR-Net on the IRSTD-UAV test set (COCO protocol: mAP50 / Precision / Recall / F1).

    python test_IRSTD-UAV.py \
        --weights     logs/IRSTD-UAV_best.pth \
        --coco_json   datasets/IRSTD-UAV/val_coco.json \
        --images_root datasets/IRSTD-UAV/images \
        --motion_root datasets/motion_difference_map_IRSTD-UAV

Inference follows the paper: 512x640 (HxW) letterbox, confidence >= 0.001, NMS IoU 0.65.
"""
import argparse
import json
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from nets.MPR_IRSTD_UAV import MPR
from utils.dataloader_for_IRSTD_UAV import preprocess_motion_diff
from utils.utils import cvtColor, get_classes, preprocess_input, resize_image, show_config
from utils.utils_bbox import decode_outputs, non_max_suppression

# grey value used to pad the letterboxed motion difference map (must match the training loader)
MOTION_PAD_VALUE = 128
NUM_FRAME = 2  # frames t-1 and t are fed to the network; only frame t is used by the detector


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate MPR-Net on IRSTD-UAV")
    p.add_argument("--weights", default="logs/IRSTD-UAV_best.pth")
    p.add_argument("--coco_json", default="datasets/IRSTD-UAV/val_coco.json", help="COCO-format ground truth")
    p.add_argument("--images_root", default="datasets/IRSTD-UAV/images", help="root joined with the COCO file_name")
    p.add_argument("--motion_root", default="datasets/motion_difference_map_IRSTD-UAV",
                   help="pre-computed motion difference maps ('' = run without the motion prior)")
    p.add_argument("--classes_path", default="model_data/classes.txt")
    p.add_argument("--input_shape", type=int, nargs=2, default=[512, 640], metavar=("H", "W"))
    p.add_argument("--confidence", type=float, default=0.001)
    p.add_argument("--nms_iou", type=float, default=0.65)
    p.add_argument("--out_dir", default="results/IRSTD-UAV")
    p.add_argument("--map_mode", type=int, default=0, choices=[0, 1, 2],
                   help="0: predict + evaluate, 1: predict only, 2: evaluate an existing eval_results.json")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


class Detector(object):
    def __init__(self, args):
        self.input_shape = args.input_shape
        self.confidence = args.confidence
        self.nms_iou = args.nms_iou
        self.letterbox_image = True
        self.motion_diff_root = args.motion_root if args.motion_root else None
        self.cuda = torch.cuda.is_available() and not args.cpu
        self.class_names, self.num_classes = get_classes(args.classes_path)

        self.net = MPR(self.num_classes, num_frame=NUM_FRAME)
        device = torch.device("cuda" if self.cuda else "cpu")
        state_dict = torch.load(args.weights, map_location=device)
        # the frozen CLIP text encoder is (re-)loaded at run time, drop its tensors if present
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("target_encoder.clip_model")}
        missing, unexpected = self.net.load_state_dict(state_dict, strict=False)
        missing = [k for k in missing if not k.startswith("target_encoder.clip_model")]
        if missing or unexpected:
            print(f"[warn] missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''} | "
                  f"unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
        self.net = self.net.eval()
        if self.cuda:
            self.net = nn.DataParallel(self.net).cuda()
        print(f"{args.weights} model, and classes loaded.")
        show_config(weights=args.weights, input_shape=self.input_shape, confidence=self.confidence,
                    nms_iou=self.nms_iou, motion_root=self.motion_diff_root, cuda=self.cuda)

    def detect_image(self, image_id, images, results, image_path=None, clsid2catid=None):
        image_shape = np.array(np.shape(images[0])[0:2])
        images = [cvtColor(image) for image in images]
        image_data = [resize_image(im, (self.input_shape[1], self.input_shape[0]), self.letterbox_image) for im in images]
        image_data = [np.transpose(preprocess_input(np.array(im, dtype="float32")), (2, 0, 1)) for im in image_data]
        image_data = np.expand_dims(np.stack(image_data, axis=1), 0)  # (1, 3, T, H, W)

        motion_diff_data = None
        if self.motion_diff_root is not None and image_path is not None:
            motion_diff_data = self.load_motion_diff_data(image_path)
            if motion_diff_data is not None:
                motion_diff_data = np.expand_dims(motion_diff_data, 0)  # (1, 3, T, H, W)

        with torch.no_grad():
            images_tensor = torch.from_numpy(image_data)
            if self.cuda:
                images_tensor = images_tensor.cuda()
            if motion_diff_data is not None:
                motion_tensor = torch.from_numpy(motion_diff_data)
                if self.cuda:
                    motion_tensor = motion_tensor.cuda()
                outputs = self.net(images_tensor, motion_diffs=motion_tensor)
            else:
                outputs = self.net(images_tensor)
            outputs = decode_outputs(outputs, self.input_shape)
            outputs = non_max_suppression(outputs, self.num_classes, self.input_shape, image_shape,
                                          self.letterbox_image, conf_thres=self.confidence, nms_thres=self.nms_iou)
            if outputs[0] is None:
                return results
            top_label = np.array(outputs[0][:, 6], dtype="int32")
            top_conf = outputs[0][:, 4] * outputs[0][:, 5]
            top_boxes = outputs[0][:, :4]

        for i, c in enumerate(top_label):
            top, left, bottom, right = top_boxes[i]
            results.append({
                "image_id": int(image_id),
                "category_id": clsid2catid[c],
                "bbox": [float(left), float(top), float(right - left), float(bottom - top)],
                "score": float(top_conf[i]),
            })
        return results

    @staticmethod
    def _resolve_motion_diff_path(folder, frame_id):
        """IRSTD-UAV frames use 8-digit ids, motion maps are usually saved with 4-digit ids; try both."""
        for name in (f"{frame_id:08d}.png", f"{frame_id:04d}.png", f"{frame_id}.png"):
            p = os.path.join(folder, name)
            if os.path.exists(p):
                return p
        return None

    def load_motion_diff_data(self, image_path):
        """Load M_t for frames t-1 and t with exactly the same letterbox geometry as the RGB frames."""
        h, w = self.input_shape
        path_parts = image_path.replace("\\", "/").split("/")
        data_folder, image_filename = path_parts[-2], path_parts[-1]
        image_id = int(image_filename.split(".")[0])

        try:
            iw, ih = Image.open(image_path).size
        except Exception:
            iw, ih = w, h
        scale = min(w / iw, h / ih)
        nw, nh = int(iw * scale), int(ih * scale)
        dx, dy = (w - nw) // 2, (h - nh) // 2

        motion_diff_folder = os.path.join(self.motion_diff_root, data_folder)
        motion_diff_data = []
        for k in range(NUM_FRAME):
            frame_id = max(image_id - k, 0)
            motion_diff_path = self._resolve_motion_diff_path(motion_diff_folder, frame_id)
            motion_diff = cv2.imread(motion_diff_path, cv2.IMREAD_GRAYSCALE) if motion_diff_path else None
            if motion_diff is None:
                motion_diff = np.zeros((h, w), dtype=np.uint8)
            else:
                motion_diff = cv2.resize(motion_diff, (nw, nh), interpolation=cv2.INTER_LINEAR)
                padded = np.full((h, w), MOTION_PAD_VALUE, dtype=np.uint8)
                padded[dy:dy + nh, dx:dx + nw] = motion_diff
                motion_diff = padded
            motion_diff_data.append(motion_diff)

        motion_diff_data = np.array(motion_diff_data[::-1])  # oldest -> newest, (T, H, W)
        processed = [preprocess_motion_diff(m.astype(np.float32)) for m in motion_diff_data]  # (T, H, W, 3)
        return np.transpose(np.array(processed, dtype=np.float32), (3, 0, 1, 2))  # (3, T, H, W)


def get_history_imgs(path):
    """<dir>/<00000012>.png -> [<dir>/00000011.png, <dir>/00000012.png]"""
    dir_path = os.path.dirname(path)
    stem, ext = os.path.splitext(os.path.basename(path))
    index = int(stem)
    return [os.path.join(dir_path, f"{max(i, 0):08d}{ext}") for i in range(index - NUM_FRAME + 1, index + 1)]


def coco_prf1(cocoEval):
    """P / R / F1 at IoU = 0.5 derived from the COCO precision-recall curve (same protocol as the paper)."""
    precision_50 = cocoEval.eval["precision"][0, :, 0, 0, -1]  # IoU=0.5, all recall thr., cat 0, area all, maxDets 100
    recall_50 = cocoEval.eval["recall"][0, 0, 0, -1]
    valid = precision_50[:int(recall_50 * 100)]
    precision = float(np.mean(valid)) if len(valid) else 0.0
    f1 = 2 * recall_50 * precision / (recall_50 + precision + 1e-12)
    return precision, float(recall_50), float(f1), precision_50


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    result_json = os.path.join(args.out_dir, "eval_results.json")

    # pycocotools requires an 'info' field
    with open(args.coco_json, "r", encoding="utf-8") as f:
        coco_data = json.load(f)
    temp_json_path = None
    coco_gt_path = args.coco_json
    if "info" not in coco_data:
        coco_data["info"] = {"description": "Dataset", "version": "1.0", "year": 2024}
        temp_json_path = os.path.join(args.out_dir, "_gt_with_info.json")
        with open(temp_json_path, "w", encoding="utf-8") as f:
            json.dump(coco_data, f)
        coco_gt_path = temp_json_path

    cocoGt = COCO(coco_gt_path)
    ids = sorted(set(cocoGt.getImgIds()) & set(cocoGt.imgToAnns.keys()))
    clsid2catid = cocoGt.getCatIds()

    if args.map_mode in (0, 1):
        detector = Detector(args)
        results = []
        for image_id in tqdm(ids, desc="Predicting"):
            image_path = os.path.join(args.images_root, cocoGt.loadImgs(image_id)[0]["file_name"])
            images = [Image.open(p) for p in get_history_imgs(image_path)]
            results = detector.detect_image(image_id, images, results, image_path=image_path, clsid2catid=clsid2catid)
        with open(result_json, "w") as f:
            json.dump(results, f)

    if args.map_mode in (0, 2):
        cocoDt = cocoGt.loadRes(result_json)
        cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        precision, recall, f1, precision_curve = coco_prf1(cocoEval)
        np.savetxt(os.path.join(args.out_dir, "pr_curve_iou50.txt"), precision_curve, fmt="%.6f")
        print("mAP50: %.4f, Precision: %.4f, Recall: %.4f, F1: %.4f" % (cocoEval.stats[1], precision, recall, f1))
        with open(os.path.join(args.out_dir, "metrics.txt"), "w") as f:
            f.write("weights: %s\nmAP50: %.4f\nPrecision: %.4f\nRecall: %.4f\nF1: %.4f\n" % (
                args.weights, cocoEval.stats[1], precision, recall, f1))

    if temp_json_path and os.path.exists(temp_json_path):
        os.remove(temp_json_path)
