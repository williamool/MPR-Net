"""
Sequence data loader for IRSTD-UAV.

Annotation txt format (one frame per line):
    <path>/images/<sequence>/<frame_id>.png x1,y1,x2,y2,cls [x1,y1,x2,y2,cls ...]
Images and motion difference maps are PNG (640x512 by default). The part of the stored path after
`/images/` is re-based onto `images_root`, so annotation files from another machine can be reused.
"""
import os
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
import torch

import cv2


def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    image = image.convert("RGB")
    return image


def preprocess(image):
    image -= np.array([0.485, 0.456, 0.406])
    image /= np.array([0.229, 0.224, 0.225])
    return image


def preprocess_motion_diff(image):
    if len(image.shape) == 2:
        image = np.stack([image, image, image], axis=-1)
    elif len(image.shape) == 3 and image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    image = image.astype(np.float32)
    image /= 255.0
    return image


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


def augmentation(images, boxes, h, w, hue=0.1, sat=0.7, val=0.4, flip_flag=None):
    if flip_flag is None:
        flip_flag = rand() < 0.5

    if flip_flag:
        for i in range(len(images)):
            images[i] = (
                Image.fromarray(images[i].astype("uint8"))
                .convert("RGB")
                .transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            )
        for i in range(len(boxes)):
            boxes[i][[0, 2]] = w - boxes[i][[2, 0]]

    for i in range(len(images)):
        images[i] = images[i].astype(np.float32) / 255.0

    return np.array(images, dtype=np.float32), np.array(boxes, dtype=np.float32), flip_flag


def augmentation_motion_diff(motion_diffs, flip_flag):
    if flip_flag:
        for i in range(len(motion_diffs)):
            motion_diffs[i] = np.fliplr(motion_diffs[i])

    processed_diffs = []
    for i in range(len(motion_diffs)):
        processed_diffs.append(preprocess_motion_diff(motion_diffs[i]))
    return np.array(processed_diffs, dtype=np.float32)


class seqDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        image_size,
        num_frame=2,
        type="train",
        motion_diff_root=None,
        images_root="datasets/IRSTD-UAV/images",
    ):
        super(seqDataset, self).__init__()
        self.dataset_path = dataset_path
        self.img_idx = []
        self.anno_idx = []
        self.type = type
        if isinstance(image_size, (list, tuple)):
            self.input_h, self.input_w = int(image_size[0]), int(image_size[1])
        else:
            self.input_h = self.input_w = int(image_size)
        self.num_frame = num_frame
        self.motion_diff_root = motion_diff_root
        self.images_root = images_root.rstrip("/")
        self.txt_path = dataset_path
        self.aug = type == "train"

        with open(self.txt_path, encoding="utf-8") as f:
            data_lines = f.readlines()
            self.length = len(data_lines)
            for line in data_lines:
                line = line.strip().split()
                if not line:
                    continue
                raw_path = line[0]
                resolved = self.resolve_image_path(raw_path)
                self.img_idx.append(resolved)
                if len(line) > 1:
                    self.anno_idx.append(
                        np.array([np.array(list(map(int, box.split(",")))) for box in line[1:]])
                    )
                else:
                    self.anno_idx.append(np.empty((0, 5), dtype=np.float32))

    def resolve_image_path(self, raw_path):
        raw_path = raw_path.strip().replace("\\", "/")
        if os.path.isfile(raw_path):
            return raw_path
        rel = None
        if "/images/" in raw_path:
            rel = raw_path.split("/images/", 1)[1].lstrip("/")
        if rel is None:
            return os.path.join(self.images_root, os.path.basename(raw_path))
        return os.path.join(self.images_root, rel)

    def _seq_dir_from_image_path(self, image_path):
        """.../images/13/00000000.png -> 13"""
        image_path = image_path.replace("\\", "/")
        parent = os.path.dirname(image_path)
        return os.path.basename(parent)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        images, box = self.get_data(index)
        flip_flag = None
        if self.aug:
            images, box, flip_flag = augmentation(
                images, box, self.input_h, self.input_w, hue=0.1, sat=0.7, val=0.4
            )
        else:
            flip_flag = False
            images = images.astype(np.float32) / 255.0

        images = np.transpose(preprocess(images), (3, 0, 1, 2))

        if len(box) != 0:
            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + (box[:, 2:4] / 2)

        motion_diffs = None
        if self.motion_diff_root is not None:
            motion_diffs = self.get_motion_diff_data(index, flip_flag if self.aug else False)
            motion_diffs = np.transpose(motion_diffs, (3, 0, 1, 2))

        if self.type == "train":
            multi_box = self.get_boxes_data(index)
            if flip_flag:
                w_img = self.input_w
                for box_item in multi_box:
                    if len(box_item) != 0:
                        box_item[:, [0, 2]] = w_img - box_item[:, [2, 0]]
            for box_item in multi_box:
                if len(box_item) != 0:
                    box_item[:, 2:4] = box_item[:, 2:4] - box_item[:, 0:2]
                    box_item[:, 0:2] = box_item[:, 0:2] + (box_item[:, 2:4] / 2)
        else:
            multi_box = None

        return images, box, None, multi_box, None, motion_diffs

    def get_boxes_data(self, index):
        h, w = self.input_h, self.input_w
        with open(self.txt_path, encoding="utf-8") as f:
            data_lines = f.readlines()

        boxes_frames = []
        for id in range(self.num_frame):
            idx = max(index - id, 0)
            line = data_lines[idx].strip().split()
            resolved = self.resolve_image_path(line[0])
            img = Image.open(resolved)
            iw, ih = img.size
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2

            if len(line) > 1:
                label_data = np.array(
                    [np.array(list(map(int, box.split(",")))) for box in line[1:]],
                    dtype=np.float32,
                )
            else:
                label_data = np.empty((0, 5), dtype=np.float32)

            if label_data.size > 0 and id == 0:
                label_data[:, [0, 2]] = label_data[:, [0, 2]] * nw / iw + dx
                label_data[:, [1, 3]] = label_data[:, [1, 3]] * nh / ih + dy
                label_data[:, 0:2][label_data[:, 0:2] < 0] = 0
                label_data[:, 2][label_data[:, 2] > w] = w
                label_data[:, 3][label_data[:, 3] > h] = h

            boxes_frames.append(label_data)

        return [np.array(b, dtype=np.float32) for b in boxes_frames[::-1]]

    def get_data(self, index):
        image_data = []
        h, w = self.input_h, self.input_w
        file_name = self.img_idx[index]
        dirname = os.path.dirname(file_name)
        basename = os.path.basename(file_name)
        stem, ext = os.path.splitext(basename)
        image_id = int(stem)

        label_data = self.anno_idx[index].copy()

        for id in range(self.num_frame):
            frame_id = max(image_id - id, 0)
            frame_path = os.path.join(dirname, f"{frame_id:08d}{ext}")
            img = Image.open(frame_path)
            img = cvtColor(img)
            iw, ih = img.size
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2

            img = img.resize((nw, nh), Image.Resampling.BICUBIC)
            new_img = Image.new("RGB", (w, h), (128, 128, 128))
            new_img.paste(img, (dx, dy))
            image_data.append(np.array(new_img, np.float32))

            if len(label_data) > 0 and id == 0:
                np.random.shuffle(label_data)
                label_data[:, [0, 2]] = label_data[:, [0, 2]] * nw / iw + dx
                label_data[:, [1, 3]] = label_data[:, [1, 3]] * nh / ih + dy
                label_data[:, 0:2][label_data[:, 0:2] < 0] = 0
                label_data[:, 2][label_data[:, 2] > w] = w
                label_data[:, 3][label_data[:, 3] > h] = h
                box_w = label_data[:, 2] - label_data[:, 0]
                box_h = label_data[:, 3] - label_data[:, 1]
                label_data = label_data[np.logical_and(box_w > 1, box_h > 1)]

        image_data = np.array(image_data[::-1])
        label_data = np.array(label_data, dtype=np.float32)
        return image_data, label_data

    def get_motion_diff_data(self, index, flip_flag=False):
        motion_diff_data = []
        h, w = self.input_h, self.input_w
        file_name = self.img_idx[index]
        basename = os.path.basename(file_name)
        stem, ext = os.path.splitext(basename)
        image_id = int(stem)
        seq_dir = self._seq_dir_from_image_path(file_name)

        ref_path = file_name
        try:
            ref_img = Image.open(ref_path)
            iw, ih = ref_img.size
        except Exception:
            iw, ih = w, h

        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        dx = (w - nw) // 2
        dy = (h - nh) // 2

        motion_seq_root = os.path.join(self.motion_diff_root, seq_dir)

        def resolve_motion_diff_path(seq_root, fid):
            """IRSTD-UAV frames use 8-digit ids, motion maps are usually saved with 4-digit ids; try both."""
            candidates = [
                os.path.join(seq_root, f"{fid:08d}.png"),
                os.path.join(seq_root, f"{fid:04d}.png"),
                os.path.join(seq_root, f"{fid}.png"),
            ]
            for p in candidates:
                if os.path.exists(p):
                    return p
            return candidates[0]

        for id in range(self.num_frame):
            frame_id = max(image_id - id, 0)
            motion_diff_path = resolve_motion_diff_path(motion_seq_root, frame_id)
            if not os.path.exists(motion_diff_path):
                motion_diff = np.zeros((h, w, 3), dtype=np.uint8)
            else:
                motion_diff = cv2.imread(motion_diff_path, cv2.IMREAD_GRAYSCALE)
                if motion_diff is None:
                    motion_diff = np.zeros((h, w, 3), dtype=np.uint8)
                else:
                    motion_diff = cv2.resize(motion_diff, (nw, nh), interpolation=cv2.INTER_LINEAR)
                    new_motion_diff = np.full((h, w), 128, dtype=np.uint8)
                    new_motion_diff[dy : dy + nh, dx : dx + nw] = motion_diff
                    motion_diff = np.stack([new_motion_diff, new_motion_diff, new_motion_diff], axis=-1)

            motion_diff_data.append(motion_diff)

        motion_diff_data = np.array(motion_diff_data[::-1])

        if self.aug:
            motion_diff_data = augmentation_motion_diff(motion_diff_data, flip_flag)
        else:
            # do not write in place into a uint8 array: floats in [0,1] would be truncated to 0
            motion_diff_data = np.stack(
                [preprocess_motion_diff(motion_diff_data[i].astype(np.float32)) for i in range(len(motion_diff_data))],
                axis=0,
            ).astype(np.float32)

        return motion_diff_data


def dataset_collate(batch):
    images = []
    bboxes = []
    captions = []
    multi_boxes = []
    relations = []
    motion_diffs = []

    for item in batch:
        if len(item) == 6:
            img, box, caption, multi_box, relation, motion_diff = item
        else:
            img, box, caption, multi_box, relation = item
            motion_diff = None

        images.append(img)
        bboxes.append(box)
        captions.append(caption)
        multi_boxes.append(multi_box)
        relations.append(relation)
        if motion_diff is not None:
            motion_diffs.append(motion_diff)

    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    bboxes = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in bboxes]

    if len(motion_diffs) > 0:
        motion_diffs = torch.from_numpy(np.array(motion_diffs)).type(torch.FloatTensor)
    else:
        motion_diffs = None

    return images, bboxes, captions, multi_boxes, relations, motion_diffs
