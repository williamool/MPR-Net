"""
MPR-Net for IRDST-H.

Identical architecture to nets/MPR_IRSTD_UAV.py / nets/MPR_DAUB.py; only the coarse textual
prompts used to initialise the motion prototypes differ between datasets.
"""
import torch
import torch.nn as nn

from .darknet import BaseConv, CSPDarknet, CSPLayer
from .graph_conv import GraphUpdate, update_features_with_gcn


class Feature_Extractor(nn.Module):
    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_features=("dark3", "dark4", "dark5"),
        in_channels=[256, 512, 1024],
        depthwise=False,
        act="silu",
        input_channels=3,
    ):
        super().__init__()
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act, in_channels=input_channels)
        self.in_features = in_features
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act)
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )
        self.reduce_conv1 = BaseConv(int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act)
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

    def forward(self, input, motion=None):
        out_features = self.backbone(input, motion)
        feat1, feat2, feat3 = [out_features[f] for f in self.in_features]
        P5 = self.lateral_conv0(feat3)
        P5_upsample = self.upsample(P5)
        P5_upsample = torch.cat([P5_upsample, feat2], 1)
        P5_upsample = self.C3_p4(P5_upsample)
        P4 = self.reduce_conv1(P5_upsample)
        P4_upsample = self.upsample(P4)
        P4_upsample = torch.cat([P4_upsample, feat1], 1)
        P3_out = self.C3_p3(P4_upsample)
        return P3_out


class TargetTextEncoder(nn.Module):
    def __init__(self, num_classes=1, learnable_tokens=7, embed_dim=512):
        super().__init__()
        self.learnable_ctx = nn.Parameter(torch.randn(num_classes, learnable_tokens, embed_dim) * 0.02)
        self.clip_model = None
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.register_buffer("_cached_text_embeddings", None, persistent=False)
        self._cached_class_names = None
        self._register_ema_compatible_params()

    def _register_ema_compatible_params(self):
        if "learnable_ctx" not in self._parameters:
            self.register_parameter("learnable_ctx", self.learnable_ctx)

    def clear_text_cache(self):
        self._cached_text_embeddings = None
        self._cached_class_names = None

    def train(self, mode=True):
        # learnable_ctx changes during training, so an inference cache must not
        # survive a switch back to train mode.
        if mode:
            self.clear_text_cache()
        return super().train(mode)

    def _load_from_state_dict(self, *args, **kwargs):
        # Loading another detector checkpoint invalidates the cached prototype.
        self.clear_text_cache()
        return super()._load_from_state_dict(*args, **kwargs)

    def encode_target_classes(self, class_names):
        class_names_key = tuple(class_names)
        if (
            not self.training
            and self._cached_text_embeddings is not None
            and self._cached_class_names == class_names_key
        ):
            return self._cached_text_embeddings

        try:
            import clip
        except ImportError:
            raise ImportError(
                "Please install CLIP: pip install git+https://github.com/ultralytics/CLIP.git"
            )

        if self.clip_model is None:
            self.clip_model = clip.load("ViT-B/32")[0]

        model = self.clip_model
        device = next(model.parameters()).device
        dtype = model.dtype

        text_token = torch.cat([clip.tokenize(p) for p in class_names]).to(device)
        tokenized_prompts = model.token_embedding(text_token)[:, :61, :].type(dtype)

        learnable_ctx = self.learnable_ctx.to(device).to(dtype)
        if learnable_ctx.shape[0] == 1 and len(class_names) > 1:
            learnable_ctx = learnable_ctx.repeat(len(class_names), 1, 1)
        x = torch.cat([tokenized_prompts, learnable_ctx], dim=1)

        x = x + model.positional_embedding.type(dtype)
        x = x.permute(1, 0, 2)
        x = model.transformer(x)
        x = x.permute(1, 0, 2)

        x = model.ln_final(x).type(dtype)
        txt_feats = (
            x[torch.arange(x.shape[0]), text_token.argmax(dim=-1)] @ model.text_projection
        )

        txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
        final_embeddings = txt_feats.reshape(-1, len(class_names), txt_feats.shape[-1])
        if not self.training:
            self._cached_text_embeddings = final_embeddings.detach()
            self._cached_class_names = class_names_key
        return final_embeddings


class ScoreCompute(nn.Module):
    def __init__(self, c1, c2, num_heads=1, embed_dim=128, guide_dim=512, scale=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.img_conv = (
            BaseConv(c1, embed_dim, ksize=1, stride=1, act=False)
            if c1 != embed_dim
            else None
        )
        self.text_linear = nn.Linear(guide_dim, embed_dim)
        self.bias = nn.Parameter(torch.zeros(num_heads))
        self.proj_conv = BaseConv(c1, c2, ksize=3, stride=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, num_heads, 1, 1)) if scale else 1.0

    def forward(self, img_feat, text_feat):
        bs, _, h, w = img_feat.shape

        if text_feat.dtype != self.text_linear.weight.dtype:
            text_feat = text_feat.to(self.text_linear.weight.dtype)

        text_feat = self.text_linear(text_feat)
        text_feat = text_feat.view(bs, -1, self.num_heads, self.head_dim)

        img_embed = self.img_conv(img_feat) if self.img_conv is not None else img_feat
        img_embed = img_embed.view(bs, self.num_heads, self.head_dim, h, w)

        attn_weight = torch.einsum("bmchw,bnmc->bmhwn", img_embed, text_feat)
        attn_weight = torch.logsumexp(attn_weight, dim=-1)
        attn_weight = attn_weight / (self.head_dim ** 0.5)
        attn_weight = attn_weight + self.bias[None, :, None, None]
        attn_weight = attn_weight.sigmoid() * self.scale
        score_map = attn_weight.mean(dim=1, keepdim=False)

        img_proj = self.proj_conv(img_feat)
        img_proj = img_proj.view(bs, self.num_heads, -1, h, w)
        img_proj = img_proj * attn_weight.unsqueeze(2)

        attn_feat = img_proj.view(bs, -1, h, w)
        return attn_feat, score_map


class DBaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, dilation=1, bias=False):
        super(DBaseConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAMBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )
        self.sigmoid_channel = nn.Sigmoid()
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        avgout = self.mlp(self.avg_pool(x))
        maxout = self.mlp(self.max_pool(x))
        scale_channel = self.sigmoid_channel(avgout + maxout)
        x = x * scale_channel
        avgout_spatial = torch.mean(x, dim=1, keepdim=True)
        maxout_spatial, _ = torch.max(x, dim=1, keepdim=True)
        scale_spatial = torch.cat([avgout_spatial, maxout_spatial], dim=1)
        scale_spatial = self.sigmoid_spatial(self.conv_spatial(scale_spatial))
        x = x * scale_spatial
        return x


class CleanFusionModule(nn.Module):
    def __init__(self, channels=256, reduction=16):
        super(CleanFusionModule, self).__init__()
        self.channels = channels
        mid_channels = channels

        self.pre_conv = BaseConv(channels * 4, mid_channels, 1, 1)
        self.cbam = CBAMBlock(mid_channels, reduction=reduction, kernel_size=7)
        self.main_conv = BaseConv(mid_channels, mid_channels, 3, 1)
        self.dilated1 = DBaseConv(mid_channels, mid_channels, 3, 1, dilation=1, padding=1)
        self.dilated2 = DBaseConv(mid_channels, mid_channels, 3, 1, dilation=2, padding=2)
        self.dilated3 = DBaseConv(mid_channels, mid_channels, 3, 1, dilation=4, padding=4)
        self.dilated_fuse = BaseConv(mid_channels * 3, mid_channels, 1, 1)
        self.merge_conv = BaseConv(mid_channels * 2, mid_channels, 3, 1)
        self.residual_conv = nn.Sequential(
            nn.Conv2d(channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(mid_channels),
        )
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.out_conv = BaseConv(mid_channels, mid_channels, 3, 1)

    def forward(self, x, x_enh):
        diff_feat = x_enh - x
        corr_feat = x_enh * x
        fused = torch.cat([x, x_enh, diff_feat, corr_feat], dim=1)
        fused = self.pre_conv(fused)
        fused = self.cbam(fused)
        main_feat = self.main_conv(fused)
        d1 = self.dilated1(fused)
        d2 = self.dilated2(fused)
        d3 = self.dilated3(fused)
        ms_feat = self.dilated_fuse(torch.cat([d1, d2, d3], dim=1))
        fusion_feat = self.merge_conv(torch.cat([main_feat, ms_feat], dim=1))
        out = self.alpha * fusion_feat + self.beta * self.residual_conv(x)
        out = self.out_conv(out)
        return [out]


class YOLOXHead(nn.Module):
    def __init__(self, num_classes, width=1.0, in_channels=[16, 32, 64], act="silu"):
        super().__init__()
        Conv = BaseConv

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()

        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(
                    in_channels=int(in_channels[i] * width),
                    out_channels=int(256 * width),
                    ksize=1,
                    stride=1,
                    act=act,
                )
            )
            self.cls_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.cls_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )

            self.reg_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.obj_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )

    def forward(self, inputs):
        outputs = []
        for k, x in enumerate(inputs):
            x = self.stems[k](x)
            cls_feat = self.cls_convs[k](x)
            cls_output = self.cls_preds[k](cls_feat)
            reg_feat = self.reg_convs[k](x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)
            output = torch.cat([reg_output, obj_output, cls_output], 1)
            outputs.append(output)
        return outputs


class MPR(nn.Module):
    """MPR-Net = MFE-enhanced CSPDarknet + PAFPN neck + Prototype-Guided Graph Reasoning + YOLOX head."""

    def __init__(self, num_classes, num_frame=2, text_input_dim=20 * 300):
        super(MPR, self).__init__()
        # num_frame / text_input_dim are kept for signature compatibility with the training scripts.
        _ = num_frame
        _ = text_input_dim

        self.backbone = Feature_Extractor(0.33, 0.50, input_channels=3)
        self.head = YOLOXHead(num_classes=num_classes, width=1.0, in_channels=[128], act="silu")

        self.target_encoder = TargetTextEncoder(num_classes=4, learnable_tokens=16, embed_dim=512)
        self.target_classes = [
            "a small bright moving target in infrared image",
            "a tiny hot flying object in thermal background",
            "a small moving target in infrared scene",
            "a weak small target with low contrast",
        ]

        self.c = 128
        self.attn = ScoreCompute(self.c, self.c, guide_dim=512, embed_dim=128, num_heads=1)
        self.feat_fusion = BaseConv(2 * self.c, self.c, ksize=1, stride=1, act=False)
        self.graph_update = GraphUpdate(self.c, 64, self.c)
        self.clean_fusion = CleanFusionModule(channels=self.c, reduction=16)

    def forward(self, inputs, motion_diffs=None):
        """
        inputs       : infrared frames  [B, 3, T, H, W]; the last frame (t) is the one detected.
        motion_diffs : explicit motion priors M_t [B, 3, T, H, W] (or None to disable the motion branch).
        Returns the decoupled YOLOX head outputs (a list with one stride-8 prediction map).
        """
        rgb_frame = inputs[:, :, -1, :, :]
        motion_frame = motion_diffs[:, :, -1, :, :] if motion_diffs is not None else None

        # ---- Motion-Guided Feature Enhancement + backbone / neck  ->  fused feature F_f ----
        f_feats = self.backbone(rgb_frame, motion_frame)

        # ---- Prototype-Guided Graph Reasoning ----
        # (1) motion prototypes P from coarse text prompts + learnable context tokens (Eq. 14-15)
        text_feat = self.target_encoder.encode_target_classes(self.target_classes)
        if text_feat.shape[0] != f_feats.shape[0]:
            text_feat = text_feat.expand(f_feats.shape[0], -1, -1)

        # (2) prototype-guided score map S and prototype-enhanced feature F_p (Eq. 16-24)
        attn_feat, score_map = self.attn(f_feats, text_feat)
        fused_feat = self.feat_fusion(torch.cat([f_feats, attn_feat], 1))

        # (3) Top-K graph construction + graph-based feature propagation  ->  F_r (Eq. 25-27)
        fused_feat = update_features_with_gcn(
            fused_feat, score_map, self.graph_update, k_ratio=0.002, similarity_threshold=0.5
        )

        # ---- joint modelling of F_f / F_r (Eq. 31-32) and detection head ----
        feats = self.clean_fusion(f_feats, fused_feat)[0]
        return self.head([feats])
