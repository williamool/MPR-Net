# ------------------------------------------------------------------#
# Motion-Guided Feature Enhancement (MFE) building blocks
#
# HFP  (High-Frequency Enhancement branch, Sec. III-B-2), adapted from HS-FPN
#      https://arxiv.org/abs/2412.10116
# ├── DctSpatialInteraction (spatial path, Eq. 9)
# ├── DctChannelInteraction (channel path, Eq. 10)
# └── HFP                   (fusion + 3x3 conv + GroupNorm, Eq. 11)
#
# MotionDiffGatedFusion (Motion-Guided Gated Fusion, Sec. III-B-3, Eq. 12)
# ------------------------------------------------------------------#

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch_dct as DCT
except ImportError:
    raise ImportError("Please install torch_dct: pip install torch-dct")

__all__ = ["DctSpatialInteraction", "DctChannelInteraction", "HFP", "MotionDiffGatedFusion"]


def _conv1x1(in_ch, out_ch, groups=1, bias=False):
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, groups=groups, bias=bias)


def _conv3x3(in_ch, out_ch, padding=1):
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=padding)


# ------------------------------------------------------------------#
# Spatial Path of HFP
# Only p1&p2 use dct to extract high_frequency response
# ------------------------------------------------------------------#
class DctSpatialInteraction(nn.Module):
    def __init__(self, in_channels, ratio, isdct=True):
        super(DctSpatialInteraction, self).__init__()
        self.ratio = ratio
        self.isdct = isdct  # True when in p1&p2, False when in p3&p4
        if not self.isdct:
            self.spatial1x1 = nn.Sequential(
                _conv1x1(in_channels, 1, bias=False)
            )

    def forward(self, x):
        _, _, h0, w0 = x.size()
        if not self.isdct:
            return x * torch.sigmoid(self.spatial1x1(x))
        # ----- isdct=True: DCT-masked spatial path (Eq. 6-9) -----
        idct = DCT.dct_2d(x, norm="ortho")
        weight = self._compute_weight(h0, w0, self.ratio).to(x.device)
        weight = weight.view(1, h0, w0).expand_as(idct)
        dct = idct * weight
        dct_ = DCT.idct_2d(dct, norm="ortho")
        return x * dct_

    def _compute_weight(self, h, w, ratio):
        h0 = int(h * ratio[0])
        w0 = int(w * ratio[1])
        weight = torch.ones((h, w), requires_grad=False)
        weight[:h0, :w0] = 0
        return weight


# ------------------------------------------------------------------#
# Channel Path of HFP
# Only p1&p2 use dct to extract high_frequency response
# ------------------------------------------------------------------#
class DctChannelInteraction(nn.Module):
    def __init__(self, in_channels, patch, ratio, isdct=True):
        super(DctChannelInteraction, self).__init__()
        self.in_channels = in_channels
        self.h = patch[0]
        self.w = patch[1]
        self.ratio = ratio
        self.isdct = isdct
        # groups=32 requires in_channels divisible by 32
        self.channel1x1 = nn.Sequential(
            _conv1x1(in_channels, in_channels, groups=32, bias=False),
        )
        self.channel2x1 = nn.Sequential(
            _conv1x1(in_channels, in_channels, groups=32, bias=False),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        n, c, h, w = x.size()
        if not self.isdct:
            amaxp = F.adaptive_max_pool2d(x, output_size=(1, 1))
            aavgp = F.adaptive_avg_pool2d(x, output_size=(1, 1))
            channel = self.channel1x1(self.relu(amaxp)) + self.channel1x1(
                self.relu(aavgp)
            )
            return x * torch.sigmoid(self.channel2x1(channel))

        # ----- isdct=True: DCT-masked channel path (Eq. 6-8, 10) -----
        idct = DCT.dct_2d(x, norm="ortho")
        weight = self._compute_weight(h, w, self.ratio).to(x.device)
        weight = weight.view(1, h, w).expand_as(idct)
        dct = idct * weight
        dct_ = DCT.idct_2d(dct, norm="ortho")

        amaxp = F.adaptive_max_pool2d(dct_, output_size=(self.h, self.w))
        aavgp = F.adaptive_avg_pool2d(dct_, output_size=(self.h, self.w))
        amaxp = torch.sum(self.relu(amaxp), dim=[2, 3]).view(n, c, 1, 1)
        aavgp = torch.sum(self.relu(aavgp), dim=[2, 3]).view(n, c, 1, 1)

        channel = self.channel1x1(amaxp) + self.channel1x1(aavgp)
        return x * torch.sigmoid(self.channel2x1(channel))

    def _compute_weight(self, h, w, ratio):
        h0 = int(h * ratio[0])
        w0 = int(w * ratio[1])
        weight = torch.ones((h, w), requires_grad=False)
        weight[:h0, :w0] = 0
        return weight


# ------------------------------------------------------------------#
# High Frequency Perception Module HFP
# ------------------------------------------------------------------#
class HFP(nn.Module):
    def __init__(
        self,
        in_channels,
        ratio,
        patch=(8, 8),
        isdct=True,
        num_groups=32,
    ):
        super(HFP, self).__init__()
        self.spatial = DctSpatialInteraction(in_channels, ratio=ratio, isdct=isdct)
        self.channel = DctChannelInteraction(
            in_channels, patch=patch, ratio=ratio, isdct=isdct
        )
        # in_channels should be divisible by num_groups (e.g. 32)
        self.out = nn.Sequential(
            _conv3x3(in_channels, in_channels),
            nn.GroupNorm(num_groups, in_channels),
        )

    def forward(self, x):
        spatial = self.spatial(x)
        channel = self.channel(x)
        return self.out(spatial + channel)


# ------------------------------------------------------------------#
# Motion-Guided Gated Fusion (MGGF), Eq. (12) in the paper
# Motion feature F_m generates a spatial gate A_s and a channel gate A_c
# (CBAM-style) that modulate the visual feature F_v -> F_s
# ------------------------------------------------------------------#
class MotionDiffGatedFusion(nn.Module):
    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        spatial_kernel: int = 7,
        alpha_init: float = 1.0,
        beta_init: float = 1.0,
        learnable_scale: bool = True,
    ):
        super().__init__()
        padding = spatial_kernel // 2

        self.spatial_conv = nn.Conv2d(
            2, 1, kernel_size=spatial_kernel, padding=padding, bias=True
        )

        hidden = max(channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
        )

        # Conv + GroupNorm (same as HFP)
        num_groups = min(32, channels)
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, channels),
        )

    def forward(self, F_rgb: torch.Tensor, F_motion: torch.Tensor, return_gates: bool = False):
        # Spatial gate A_s: [B,1,H,W]
        max_map = torch.max(F_motion, dim=1, keepdim=True)[0]
        avg_map = torch.mean(F_motion, dim=1, keepdim=True)
        spatial_in = torch.cat([max_map, avg_map], dim=1)
        A_s = torch.sigmoid(self.spatial_conv(spatial_in))

        # Channel gate A_c: [B,C,1,1]
        avgp = F.adaptive_avg_pool2d(F_motion, 1)
        maxp = F.adaptive_max_pool2d(F_motion, 1)
        A_c = torch.sigmoid(self.mlp(avgp) + self.mlp(maxp))

        F_rgb_s = F_rgb * A_s
        F_rgb_c = F_rgb * A_c
        F_out = self.fuse_conv(F_rgb_s + F_rgb_c)

        if return_gates:
            return F_out, A_s, A_c
        return F_out
