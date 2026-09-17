"""
Sequence data loader for DAUB-R.

Annotation txt format (one frame per line):
    <path>/<sequence>/<frame_id>.bmp x1,y1,x2,y2,cls [x1,y1,x2,y2,cls ...]
Images are letterbox-resized to the configured input size (grey padding 128) and the motion
difference maps are resized/padded with exactly the same geometry so that they stay aligned.
"""
import os

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data.dataset import Dataset


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


def resolve_image_path(raw_path, images_root=None):
    """
    Map the image path stored in the annotation txt onto the local dataset root.
    If `images_root` is given, only the trailing `<sequence>/<frame>.bmp` part of the stored path is kept,
    so annotation files written on another machine can be reused without editing them.
    """
    raw_path = raw_path.strip().replace("\\", "/")
    if images_root is None or os.path.isfile(raw_path):
        return raw_path
    rel = "/".join(raw_path.split("/")[-2:])
    return os.path.join(images_root, rel)


class seqDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        image_size,
        num_frame=2,
        type="train",
        motion_diff_root=None,
        images_root=None,
    ):
        super(seqDataset, self).__init__()
        self.dataset_path = dataset_path
        self.img_idx = []
        self.anno_idx = []
        self.type = type
        if isinstance(image_size, (list, tuple)):
            self.input_h, self.input_w = int(image_size[0]), int(image_size[1])
        else:
            s = int(image_size)
            self.input_h = self.input_w = s
        self.num_frame = num_frame
        self.motion_diff_root = motion_diff_root
        self.images_root = images_root
        self.txt_path = dataset_path
        self.aug = type == "train"

        with open(self.txt_path, encoding="utf-8") as f:
            data_lines = f.readlines()
            self.length = len(data_lines)
            for line in data_lines:
                line = line.strip().split()
                if not line:
                    continue
                self.img_idx.append(resolve_image_path(line[0], self.images_root))
                if len(line) > 1:
                    self.anno_idx.append(
                        np.array([np.array(list(map(int, box.split(",")))) for box in line[1:]])
                    )
                else:
                    self.anno_idx.append(np.empty((0, 5), dtype=np.float32))

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
        file_name = self.img_idx[index]
        image_id = int(file_name.split("/")[-1][:-4])
        image_path = file_name.replace(file_name.split("/")[-1], "")
        img = Image.open(image_path + "%d.bmp" % image_id)
        iw, ih = img.size
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        dx = (w - nw) // 2
        dy = (h - nh) // 2

        boxes_frames = []
        with open(self.txt_path, encoding="utf-8") as f:
            data_lines = f.readlines()

        for id in range(self.num_frame):
            idx = max(index - id, 0)
            line = data_lines[idx].strip().split()
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
        image_id = int(file_name.split("/")[-1][:-4])
        image_path = file_name.replace(file_name.split("/")[-1], "")
        label_data = self.anno_idx[index].copy()

        for id in range(self.num_frame):
            img = Image.open(image_path + "%d.bmp" % max(image_id - id, 0))
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
        normalized_path = file_name.replace("\\", "/")
        path_parts = normalized_path.split("/")

        if len(path_parts) >= 2:
            data_folder = path_parts[-2]
            image_filename = path_parts[-1]
            image_id = int(image_filename.split(".")[0])
            image_path = file_name.replace(image_filename, "")
        else:
            raise ValueError(f"Invalid path format: {file_name}")

        try:
            img = Image.open(image_path + "%d.bmp" % image_id)
            iw, ih = img.size
        except Exception:
            iw, ih = w, h

        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        dx = (w - nw) // 2
        dy = (h - nh) // 2

        motion_diff_folder = os.path.join(self.motion_diff_root, data_folder)

        for id in range(self.num_frame):
            frame_id = max(image_id - id, 0)
            motion_diff_path = os.path.join(motion_diff_folder, f"{frame_id:04d}.bmp")
            if not os.path.exists(motion_diff_path):
                z = np.zeros((h, w), dtype=np.uint8)
                motion_diff = np.stack([z, z, z], axis=-1)
            else:
                motion_diff = cv2.imread(motion_diff_path, cv2.IMREAD_GRAYSCALE)
                if motion_diff is None:
                    z = np.zeros((h, w), dtype=np.uint8)
                    motion_diff = np.stack([z, z, z], axis=-1)
                else:
                    motion_diff = cv2.resize(motion_diff, (nw, nh), interpolation=cv2.INTER_LINEAR)
                    new_motion_diff = np.full((h, w), 128, dtype=np.uint8)
                    new_motion_diff[dy : dy + nh, dx : dx + nw] = motion_diff
                    motion_diff = np.stack(
                        [new_motion_diff, new_motion_diff, new_motion_diff], axis=-1
                    )

            motion_diff_data.append(motion_diff)

        motion_diff_data = np.array(motion_diff_data[::-1])

        if self.aug:
            motion_diff_data = augmentation_motion_diff(motion_diff_data, flip_flag)
        else:
            motion_diff_data = np.stack(
                [
                    preprocess_motion_diff(motion_diff_data[i].astype(np.float32))
                    for i in range(len(motion_diff_data))
                ],
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
