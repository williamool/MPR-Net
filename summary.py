"""
Model complexity: Params / FLOPs / forward FPS.

    python summary.py --dataset DAUB        # 512x512  -> 8.8M params, 43.5 GFLOPs in the paper
    python summary.py --dataset IRSTD-UAV   # 512x640  -> 8.8M params, 54.4 GFLOPs in the paper

Notes
* Params exclude the frozen CLIP ViT-B/32 text encoder, which is only used to build the motion
  prototypes; FLOPs are measured after the prototypes have been cached (eval mode), i.e. they
  correspond to the per-frame detection cost.
* FLOPs = 2 x MACs (thop reports MACs).
"""
import argparse
import importlib
import time

import torch
from thop import clever_format, profile

DATASETS = {
    "DAUB": ("nets.MPR_DAUB", [512, 512]),
    "IRDST-H": ("nets.MPR_IRDSTH", [512, 512]),
    "IRSTD-UAV": ("nets.MPR_IRSTD_UAV", [512, 640]),
}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="DAUB", choices=list(DATASETS))
    p.add_argument("--input_shape", type=int, nargs=2, default=None, metavar=("H", "W"))
    p.add_argument("--fps_iters", type=int, default=100, help="0 to skip the FPS measurement")
    args = p.parse_args()

    module_name, default_shape = DATASETS[args.dataset]
    input_shape = args.input_shape or default_shape
    MPR = importlib.import_module(module_name).MPR

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MPR(num_classes=1, num_frame=2).to(device).eval()

    frames = torch.randn(1, 3, 2, input_shape[0], input_shape[1]).to(device)
    motion = torch.rand(1, 3, 2, input_shape[0], input_shape[1]).to(device)

    with torch.no_grad():
        model(frames, motion)  # warm-up: loads CLIP and caches the motion prototypes
        flops, _ = profile(model, inputs=(frames, motion), verbose=False)
    flops = flops * 2

    params = sum(p.numel() for n, p in model.named_parameters() if not n.startswith("target_encoder.clip_model"))
    flops_str, params_str = clever_format([flops, params], "%.3f")
    print(f"Dataset: {args.dataset}  input (HxW): {input_shape[0]}x{input_shape[1]}")
    print(f"Total GFLOPs: {flops_str}")
    print(f"Total Params (w/o frozen CLIP text encoder): {params_str}")

    if args.fps_iters > 0:
        with torch.no_grad():
            for _ in range(10):
                model(frames, motion)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(args.fps_iters):
                model(frames, motion)
            if device.type == "cuda":
                torch.cuda.synchronize()
        print(f"Forward FPS ({device.type}, batch 1): {args.fps_iters / (time.time() - t0):.1f}")
