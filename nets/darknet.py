#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
#
# CSPDarknet backbone (YOLOX) with the MPR-Net Motion-Guided Feature Enhancement (MFE) module:
#   stem -> dark2 -> [HFE branch + motion-guided gated fusion] -> dark3 -> dark4 -> dark5

import torch
from torch import nn

from .high_pass import HFP, MotionDiffGatedFusion

class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = SiLU()
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    elif name == "sigmoid":
        module = nn.Sigmoid()
    elif name == False or name is None:
        # no activation -> identity
        module = nn.Identity()
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module

class Focus(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
        super().__init__()
        self.in_channels = in_channels
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        patch_top_left  = x[...,  ::2,  ::2]
        patch_bot_left  = x[..., 1::2,  ::2]
        patch_top_right = x[...,  ::2, 1::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat((patch_top_left, patch_bot_left, patch_top_right, patch_bot_right,), dim=1,)
        return self.conv(x)

class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"):
        super().__init__()
        pad         = (ksize - 1) // 2
        self.conv   = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=pad, groups=groups, bias=bias)
        self.bn     = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.act    = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(in_channels, in_channels, ksize=ksize, stride=stride, groups=in_channels, act=act,)
        self.pconv = BaseConv(in_channels, out_channels, ksize=1, stride=1, groups=1, act=act)

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)

class SPPBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1      = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        self.m          = nn.ModuleList([nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2) for ks in kernel_sizes])
        conv2_channels  = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2      = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5, depthwise=False, act="silu",):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y

class CSPLayer(nn.Module):
    def __init__(self, in_channels, out_channels, n=1, shortcut=True, expansion=0.5, depthwise=False, act="silu",):
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  
        self.conv1  = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2  = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3  = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        module_list = [Bottleneck(hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act) for _ in range(n)]
        self.m      = nn.Sequential(*module_list)

    def forward(self, x):
        
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)

class CSPDarknet(nn.Module):
    """CSPDarknet backbone with the Motion-Guided Feature Enhancement (MFE) module at the dark2 stage."""

    def __init__(
        self,
        dep_mul,
        wid_mul,
        out_features=("stem", "dark2", "dark3", "dark4", "dark5"),
        depthwise=False,
        act="silu",
        in_channels=3,
    ):
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        Conv = DWConv if depthwise else BaseConv
        base_channels = int(wid_mul * 64)
        base_depth = max(round(dep_mul * 3), 1)

        self.stem = Focus(in_channels, base_channels, ksize=3, act=act)
        c_dark2 = base_channels * 2

        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(base_channels * 2, base_channels * 2, n=base_depth, depthwise=depthwise, act=act),
        )

        self.hfp_dark2 = HFP(c_dark2, ratio=(0.25, 0.25), patch=(8, 8), isdct=True)
        self.motion_gate_dark2 = MotionDiffGatedFusion(channels=c_dark2)
        self.alpha = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

        self.stem_motion = Focus(in_channels, base_channels, ksize=3, act=act)
        self.dark2_motion = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(base_channels * 2, base_channels * 2, n=base_depth, depthwise=depthwise, act=act),
        )
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(base_channels * 4, base_channels * 4, n=base_depth * 3, depthwise=depthwise, act=act),
        )
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(base_channels * 8, base_channels * 8, n=base_depth * 3, depthwise=depthwise, act=act),
        )
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
            CSPLayer(base_channels * 16, base_channels * 16, n=base_depth, shortcut=False, depthwise=depthwise, act=act),
        )

    def forward(self, x, motion=None):
        """
        x      : current infrared frame, [B, 3, H, W]
        motion : explicit motion prior M_t (motion difference map), [B, 3, H, W] or None
        """
        outputs = {}
        x_rgb = self.stem(x)
        outputs["stem"] = x_rgb

        # visual feature F_v at the dark2 stage
        x_d2 = self.dark2(x_rgb)

        # High-Frequency Enhancement branch -> F_h  (Eq. 6-11)
        Fh = self.hfp_dark2(x_d2)

        if motion is not None:
            # motion branch shares the stem/dark2 topology with the visual backbone -> F_m
            m1 = self.stem_motion(motion)
            m2 = self.dark2_motion(m1)
            # Motion-Guided Gated Fusion -> F_s  (Eq. 12)
            Fs = self.motion_gate_dark2(x_d2, m2)
            # Dual-branch fusion: F_enh = F_v + alpha * F_h + beta * F_s  (Eq. 13)
            x = x_d2 + self.alpha * Fh + self.beta * Fs
        else:
            x = x_d2 + self.alpha * Fh

        outputs["dark2"] = x

        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}


if __name__ == '__main__':
    print(CSPDarknet(1, 1))