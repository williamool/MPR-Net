"""
Train MPR-Net on DAUB-R.

    python train_DAUB.py \
        --train_txt  datasets/DAUB/train.txt \
        --val_txt    datasets/DAUB/val.txt \
        --images_root datasets/DAUB \
        --motion_root datasets/motion_difference_map_DAUB \
        --pretrained model_data/pre_trained_backbone.pth \
        --save_dir   logs/DAUB

All hyper-parameters default to the setting used in the paper
(512x512 letterbox input, 100 epochs, batch 8, SGD lr 1e-2, momentum 0.937, wd 5e-4, cosine decay).
"""
import argparse
import datetime
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.MPR_DAUB import MPR
from nets.training import ModelEMA, YOLOLoss, get_lr_scheduler, set_optimizer_lr, weights_init
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader_for_DAUB import dataset_collate, seqDataset
from utils.utils import get_classes, show_config
from utils.utils_fit import fit_one_epoch


def parse_args():
    p = argparse.ArgumentParser(description="Train MPR-Net on DAUB-R")
    # ---------------- data ----------------
    p.add_argument("--train_txt", default="datasets/DAUB/train.txt")
    p.add_argument("--val_txt", default="datasets/DAUB/val.txt")
    p.add_argument("--images_root", default="datasets/DAUB",
                   help="dataset root containing data5/, data6/, ... (used to re-base the paths stored in the txt)")
    p.add_argument("--motion_root", default="datasets/motion_difference_map_DAUB",
                   help="root of the pre-computed motion difference maps (see motion_diff/); "
                        "set to '' to train without the explicit motion prior")
    p.add_argument("--classes_path", default="model_data/classes.txt")
    # ---------------- model ----------------
    p.add_argument("--pretrained", default="model_data/pre_trained_backbone.pth",
                   help="COCO-pretrained YOLOX-s weights used to initialise the CSPDarknet backbone ('' = from scratch)")
    p.add_argument("--input_shape", type=int, nargs=2, default=[512, 512], metavar=("H", "W"))
    # ---------------- optimisation ----------------
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--momentum", type=float, default=0.937)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--seed", type=int, default=2023)
    # ---------------- logging ----------------
    p.add_argument("--save_dir", default="logs/DAUB")
    p.add_argument("--save_period", type=int, default=1)
    return p.parse_args()


def load_pretrained_backbone(model, weight_path, device):
    """Copy every tensor whose name and shape match (backbone / neck / head of YOLOX-s)."""
    model_dict = model.state_dict()
    pretrained_dict = torch.load(weight_path, map_location=device)
    load_key, no_load_key, temp_dict = [], [], {}
    for k, v in pretrained_dict.items():
        if k in model_dict and np.shape(model_dict[k]) == np.shape(v):
            temp_dict[k] = v
            load_key.append(k)
        else:
            no_load_key.append(k)
    model_dict.update(temp_dict)
    model.load_state_dict(model_dict)
    print(f"Load weights {weight_path}: {len(load_key)} tensors loaded, {len(no_load_key)} skipped "
          f"(MPR-specific modules are trained from scratch).")


if __name__ == "__main__":
    args = parse_args()
    Cuda = torch.cuda.is_available()
    device = torch.device("cuda" if Cuda else "cpu")
    motion_diff_root = args.motion_root if args.motion_root else None
    if motion_diff_root is not None and not os.path.isdir(motion_diff_root):
        raise FileNotFoundError(f"motion difference maps not found: {motion_diff_root}")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    class_names, num_classes = get_classes(args.classes_path)

    # ---------------- model ----------------
    model = MPR(num_classes=num_classes, num_frame=2)
    weights_init(model)
    if args.pretrained:
        load_pretrained_backbone(model, args.pretrained, device)

    # YOLOX loss (lambda_iou = 5, lambda_obj = 1, lambda_cls = 1) on the single stride-8 level
    yolo_loss = YOLOLoss(num_classes, args.fp16, strides=[8])

    time_str = datetime.datetime.strftime(datetime.datetime.now(), "%Y_%m_%d_%H_%M_%S")
    log_dir = os.path.join(args.save_dir, "loss_" + time_str)
    loss_history = LossHistory(log_dir, model, input_shape=args.input_shape)

    scaler = torch.cuda.amp.GradScaler() if args.fp16 else None

    model_train = model.train()
    if Cuda:
        model_train = nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    ema = ModelEMA(model_train)

    # ---------------- data ----------------
    with open(args.train_txt, encoding="utf-8") as f:
        train_lines = f.readlines()
    with open(args.val_txt, encoding="utf-8") as f:
        val_lines = f.readlines()
    num_train, num_val = len(train_lines), len(val_lines)

    show_config(
        classes_path=args.classes_path, pretrained=args.pretrained, input_shape=args.input_shape,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, momentum=args.momentum,
        weight_decay=args.weight_decay, lr_decay_type="cos", save_dir=log_dir,
        num_workers=args.num_workers, num_train=num_train, num_val=num_val,
        motion_root=motion_diff_root,
    )

    batch_size = args.batch_size
    nbs = 64
    lr_limit_max, lr_limit_min = 5e-2, 5e-4
    Init_lr_fit = min(max(batch_size / nbs * args.lr, lr_limit_min), lr_limit_max)
    Min_lr_fit = min(max(batch_size / nbs * args.lr * 0.01, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    pg0, pg1, pg2 = [], [], []  # bn weights / conv weights (with decay) / biases
    for k, v in model.named_modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d) or "bn" in k:
            pg0.append(v.weight)
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)
    optimizer = optim.SGD(pg0, Init_lr_fit, momentum=args.momentum, nesterov=True)
    optimizer.add_param_group({"params": pg1, "weight_decay": args.weight_decay})
    optimizer.add_param_group({"params": pg2})

    lr_scheduler_func = get_lr_scheduler("cos", Init_lr_fit, Min_lr_fit, args.epochs)

    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size
    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError("The dataset is too small to continue training.")

    train_dataset = seqDataset(args.train_txt, args.input_shape, 2, "train",
                               motion_diff_root=motion_diff_root, images_root=args.images_root)
    val_dataset = seqDataset(args.val_txt, args.input_shape, 2, "val",
                             motion_diff_root=motion_diff_root, images_root=args.images_root)

    gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=args.num_workers,
                     pin_memory=True, drop_last=True, collate_fn=dataset_collate)
    gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=args.num_workers,
                         pin_memory=True, drop_last=True, collate_fn=dataset_collate)

    # In-training mAP evaluation is disabled (it does not feed the motion prior);
    # evaluate the saved checkpoints with test_DAUB.py instead.
    eval_callback = EvalCallback(model, args.input_shape, class_names, num_classes, val_lines, log_dir, Cuda,
                                 eval_flag=False, period=args.epochs)

    # ---------------- train ----------------
    for epoch in range(args.epochs):
        gen.dataset.epoch_now = epoch
        gen_val.dataset.epoch_now = epoch
        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
        fit_one_epoch(model_train, model, ema, yolo_loss, loss_history, eval_callback, optimizer, epoch,
                      epoch_step, epoch_step_val, gen, gen_val, args.epochs, Cuda, args.fp16, scaler,
                      args.save_period, log_dir, local_rank=0)

    loss_history.writer.close()
