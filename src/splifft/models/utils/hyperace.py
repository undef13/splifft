"""HyperACE segmentation backbones for BS-RoFormer mask heads.

These modules are compatibility shims for unwa variants trained in msst
(`hyperace_v1`, `hyperace_v2`, and `large_inst_v2` head behavior). They are
kept separate from the core transformer stack because they are used only by a
small subset of checkpoints.

See: <https://huggingface.co/pcunwa/BS-Roformer-HyperACE> and <https://arxiv.org/abs/2506.17733>
"""  # this is largely lifted from unwa without much modification
# TODO figure out ways to simplify or unify

from __future__ import annotations

# ruff: noqa: ANN001,ANN201,E741
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .tfc_tdf import TfcTdf, instance_norm_factory, silu_factory


def autopad(
    k: int | tuple[int, int],
    p: int | tuple[int, int] | None = None,
) -> int | tuple[int, int]:
    if p is None:
        p = k // 2 if isinstance(k, int) else (k[0] // 2, k[1] // 2)
    return p


def build_hyperace_tfc_tdf(
    in_c: int,
    c: int,
    l: int,
    f: int,
    bn: int = 4,
) -> TfcTdf:
    return TfcTdf(
        in_channels=in_c,
        out_channels=c,
        num_blocks=l,
        f_bins=f,
        bottleneck_factor=bn,
        norm_factory=instance_norm_factory,
        act_factory=silu_factory,
    )


class Conv(nn.Module):
    def __init__(
        self,
        c1: int,
        c2: int,
        k: int | tuple[int, int] = 1,
        s: int | tuple[int, int] = 1,
        p: int | tuple[int, int] | None = None,
        g: int = 1,
        act: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.InstanceNorm2d(c2, affine=True, eps=1e-8)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x: Tensor) -> Any:
        return self.act(self.bn(self.conv(x)))


class DSConv(nn.Module):
    def __init__(
        self,
        c1: int,
        c2: int,
        k: int | tuple[int, int] = 3,
        s: int | tuple[int, int] = 1,
        p: int | tuple[int, int] | None = None,
        act: bool = True,
    ):
        super().__init__()
        self.dwconv = nn.Conv2d(c1, c1, k, s, autopad(k, p), groups=c1, bias=False)
        self.pwconv = nn.Conv2d(c1, c2, 1, 1, 0, bias=False)
        self.bn = nn.InstanceNorm2d(c2, affine=True, eps=1e-8)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x: Tensor) -> Any:
        return self.act(self.bn(self.pwconv(self.dwconv(x))))


class DS_Bottleneck(nn.Module):
    def __init__(self, c1: int, c2: int, k: int = 3, shortcut: bool = True):
        super().__init__()
        c_ = c1
        self.dsconv1 = DSConv(c1, c_, k=3, s=1)
        self.dsconv2 = DSConv(c_, c2, k=k, s=1)
        self.shortcut = shortcut and c1 == c2

    def forward(self, x: Tensor) -> Any:
        if self.shortcut:
            return x + self.dsconv2(self.dsconv1(x))
        return self.dsconv2(self.dsconv1(x))


class DS_C3k(nn.Module):
    def __init__(self, c1: int, c2: int, n: int = 1, k: int = 3, e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        self.m = nn.Sequential(*[DS_Bottleneck(c_, c_, k=k, shortcut=True) for _ in range(n)])

    def forward(self, x: Tensor) -> Any:
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class DS_C3k2(nn.Module):
    def __init__(self, c1: int, c2: int, n: int = 1, k: int = 3, e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.m = DS_C3k(c_, c_, n=n, k=k, e=1.0)
        self.cv2 = Conv(c_, c2, 1, 1)

    def forward(self, x: Tensor) -> Any:
        x_ = self.cv1(x)
        x_ = self.m(x_)
        return self.cv2(x_)


class AdaptiveHyperedgeGeneration(nn.Module):
    def __init__(self, in_channels: int, num_hyperedges: int, num_heads: int = 8):
        super().__init__()
        self.num_hyperedges = num_hyperedges
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads

        self.global_proto = nn.Parameter(torch.randn(num_hyperedges, in_channels))

        self.context_mapper = nn.Linear(2 * in_channels, num_hyperedges * in_channels, bias=False)

        self.query_proj = nn.Linear(in_channels, in_channels, bias=False)

        self.scale = self.head_dim**-0.5

    def forward(self, x: Tensor) -> Tensor:
        b, n, c = x.shape

        f_avg = F.adaptive_avg_pool1d(x.permute(0, 2, 1), 1).squeeze(-1)
        f_max = F.adaptive_max_pool1d(x.permute(0, 2, 1), 1).squeeze(-1)
        f_ctx = torch.cat((f_avg, f_max), dim=1)

        delta_p = self.context_mapper(f_ctx).view(b, self.num_hyperedges, c)
        p = self.global_proto.unsqueeze(0) + delta_p

        z = self.query_proj(x)

        z = z.view(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        p = p.view(b, self.num_hyperedges, self.num_heads, self.head_dim).permute(0, 2, 3, 1)

        sim = (z @ p) * self.scale

        s_bar = sim.mean(dim=1)

        a = F.softmax(s_bar.permute(0, 2, 1), dim=-1)

        return a


class HypergraphConvolution(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.W_e = nn.Linear(in_channels, in_channels, bias=False)
        self.W_v = nn.Linear(in_channels, out_channels, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: Tensor, a: Tensor) -> Any:
        f_m = torch.bmm(a, x)
        f_m = self.act(self.W_e(f_m))

        x_out = torch.bmm(a.transpose(1, 2), f_m)
        x_out = self.act(self.W_v(x_out))

        return x + x_out


class AdaptiveHypergraphComputation(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_hyperedges: int = 8,
        num_heads: int = 8,
    ):
        super().__init__()
        self.adaptive_hyperedge_gen = AdaptiveHyperedgeGeneration(
            in_channels, num_hyperedges, num_heads
        )
        self.hypergraph_conv = HypergraphConvolution(in_channels, out_channels)

    def forward(self, x: Tensor) -> Any:
        b, _, h, w = x.shape
        x_flat = x.flatten(2).permute(0, 2, 1)

        a = self.adaptive_hyperedge_gen(x_flat)

        x_out_flat = self.hypergraph_conv(x_flat, a)

        x_out = x_out_flat.permute(0, 2, 1).view(b, -1, h, w)
        return x_out


class C3AH(nn.Module):
    def __init__(
        self,
        c1: int,
        c2: int,
        num_hyperedges: int = 8,
        num_heads: int = 8,
        e: float = 0.5,
    ):
        super().__init__()
        c_ = int(c1 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.ahc = AdaptiveHypergraphComputation(c_, c_, num_hyperedges, num_heads)
        self.cv3 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x: Tensor) -> Any:
        x_lateral = self.cv1(x)
        x_ahc = self.ahc(self.cv2(x))
        return self.cv3(torch.cat((x_ahc, x_lateral), dim=1))


class HyperACE(nn.Module):
    def __init__(
        self,
        in_channels: list[int],
        out_channels: int,
        num_hyperedges: int = 8,
        num_heads: int = 8,
        k: int = 2,
        l: int = 1,
        c_h: float = 0.5,
        c_l: float = 0.25,
    ):
        super().__init__()

        c2, c3, c4, c5 = in_channels
        c_mid = c4

        self.fuse_conv = Conv(c2 + c3 + c4 + c5, c_mid, 1, 1)

        self.c_h = int(c_mid * c_h)
        self.c_l = int(c_mid * c_l)
        self.c_s = c_mid - self.c_h - self.c_l
        assert self.c_s > 0, "Channel split error"

        self.high_order_branch = nn.ModuleList(
            [C3AH(self.c_h, self.c_h, num_hyperedges, num_heads, e=1.0) for _ in range(k)]
        )
        self.high_order_fuse = Conv(self.c_h * k, self.c_h, 1, 1)

        self.low_order_branch = nn.Sequential(
            *[DS_C3k(self.c_l, self.c_l, n=1, k=3, e=1.0) for _ in range(l)]
        )

        self.final_fuse = Conv(self.c_h + self.c_l + self.c_s, out_channels, 1, 1)

    def forward(self, x: list[Tensor]) -> Any:
        b2, b3, b4, b5 = x

        _, _, h4, w4 = b4.shape

        b2_resized = F.interpolate(b2, size=(h4, w4), mode="bilinear", align_corners=False)
        b3_resized = F.interpolate(b3, size=(h4, w4), mode="bilinear", align_corners=False)
        b5_resized = F.interpolate(b5, size=(h4, w4), mode="bilinear", align_corners=False)

        x_b = self.fuse_conv(torch.cat((b2_resized, b3_resized, b4, b5_resized), dim=1))

        x_h, x_l, x_s = torch.split(x_b, [self.c_h, self.c_l, self.c_s], dim=1)

        x_h_outs = [m(x_h) for m in self.high_order_branch]
        x_h_fused = self.high_order_fuse(torch.cat(x_h_outs, dim=1))

        x_l_out = self.low_order_branch(x_l)

        y = self.final_fuse(torch.cat((x_h_fused, x_l_out, x_s), dim=1))

        return y


class GatedFusion(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, in_channels, 1, 1))

    def forward(self, f_in: Tensor, h: Tensor) -> Any:
        if f_in.shape[1] != h.shape[1]:
            raise ValueError(f"Channel mismatch: f_in={f_in.shape}, h={h.shape}")
        return f_in + self.gamma * h


class BackboneHyperAceV1(nn.Module):
    def __init__(self, in_channels: int = 256, base_channels: int = 64, base_depth: int = 3):
        super().__init__()
        c2 = base_channels
        c3 = 256
        c4 = 384
        c5 = 512
        c6 = 768

        self.stem = DSConv(in_channels, c2, k=3, s=(2, 1), p=1)

        self.p2 = nn.Sequential(
            DSConv(c2, c3, k=3, s=(2, 1), p=1),
            DS_C3k2(c3, c3, n=base_depth),
        )

        self.p3 = nn.Sequential(
            DSConv(c3, c4, k=3, s=(2, 1), p=1),
            DS_C3k2(c4, c4, n=base_depth * 2),
        )

        self.p4 = nn.Sequential(
            DSConv(c4, c5, k=3, s=(2, 1), p=1),
            DS_C3k2(c5, c5, n=base_depth * 2),
        )

        self.p5 = nn.Sequential(
            DSConv(c5, c6, k=3, s=(2, 1), p=1),
            DS_C3k2(c6, c6, n=base_depth),
        )

        self.out_channels = [c3, c4, c5, c6]

    def forward(self, x: Tensor) -> list[Tensor]:
        x = self.stem(x)
        x2 = self.p2(x)
        x3 = self.p3(x2)
        x4 = self.p4(x3)
        x5 = self.p5(x4)
        return [x2, x3, x4, x5]


class BackboneHyperAceV2(nn.Module):
    def __init__(self, in_channels: int = 256, base_channels: int = 64, base_depth: int = 3):
        super().__init__()
        c2 = base_channels
        c3 = 256
        c4 = 384
        c5 = 512
        c6 = 768

        self.stem = DSConv(in_channels, c2, k=3, s=(2, 1), p=1)

        self.p2 = nn.Sequential(
            DSConv(c2, c3, k=3, s=(2, 1), p=1),
            DS_C3k2(c3, c3, n=base_depth),
        )

        self.p3 = nn.Sequential(
            DSConv(c3, c4, k=3, s=(2, 1), p=1),
            DS_C3k2(c4, c4, n=base_depth * 2),
        )

        self.p4 = nn.Sequential(
            DSConv(c4, c5, k=3, s=2, p=1),
            DS_C3k2(c5, c5, n=base_depth * 2),
        )

        self.p5 = nn.Sequential(
            DSConv(c5, c6, k=3, s=2, p=1),
            DS_C3k2(c6, c6, n=base_depth),
        )

        self.out_channels = [c3, c4, c5, c6]

    def forward(self, x: Tensor) -> list[Tensor]:
        x = self.stem(x)
        x2 = self.p2(x)
        x3 = self.p3(x2)
        x4 = self.p4(x3)
        x5 = self.p5(x4)
        return [x2, x3, x4, x5]


class DecoderHyperAce(nn.Module):
    def __init__(
        self, encoder_channels: list[int], hyperace_out_c: int, decoder_channels: list[int]
    ):
        super().__init__()
        c_p2, c_p3, c_p4, c_p5 = encoder_channels
        c_d2, c_d3, c_d4, c_d5 = decoder_channels

        self.h_to_d5 = Conv(hyperace_out_c, c_d5, 1, 1)
        self.h_to_d4 = Conv(hyperace_out_c, c_d4, 1, 1)
        self.h_to_d3 = Conv(hyperace_out_c, c_d3, 1, 1)
        self.h_to_d2 = Conv(hyperace_out_c, c_d2, 1, 1)

        self.fusion_d5 = GatedFusion(c_d5)
        self.fusion_d4 = GatedFusion(c_d4)
        self.fusion_d3 = GatedFusion(c_d3)
        self.fusion_d2 = GatedFusion(c_d2)

        self.skip_p5 = Conv(c_p5, c_d5, 1, 1)
        self.skip_p4 = Conv(c_p4, c_d4, 1, 1)
        self.skip_p3 = Conv(c_p3, c_d3, 1, 1)
        self.skip_p2 = Conv(c_p2, c_d2, 1, 1)

        self.up_d5 = DS_C3k2(c_d5, c_d4, n=1)
        self.up_d4 = DS_C3k2(c_d4, c_d3, n=1)
        self.up_d3 = DS_C3k2(c_d3, c_d2, n=1)

        self.final_d2 = DS_C3k2(c_d2, c_d2, n=1)

    def forward(self, enc_feats: list[Tensor], h_ace: Tensor) -> Any:
        p2, p3, p4, p5 = enc_feats

        d5 = self.skip_p5(p5)
        h_d5 = self.h_to_d5(F.interpolate(h_ace, size=d5.shape[2:], mode="bilinear"))
        d5 = self.fusion_d5(d5, h_d5)

        d5_up = F.interpolate(d5, size=p4.shape[2:], mode="bilinear")
        d4_skip = self.skip_p4(p4)
        d4 = self.up_d5(d5_up) + d4_skip

        h_d4 = self.h_to_d4(F.interpolate(h_ace, size=d4.shape[2:], mode="bilinear"))
        d4 = self.fusion_d4(d4, h_d4)

        d4_up = F.interpolate(d4, size=p3.shape[2:], mode="bilinear")
        d3_skip = self.skip_p3(p3)
        d3 = self.up_d4(d4_up) + d3_skip

        h_d3 = self.h_to_d3(F.interpolate(h_ace, size=d3.shape[2:], mode="bilinear"))
        d3 = self.fusion_d3(d3, h_d3)

        d3_up = F.interpolate(d3, size=p2.shape[2:], mode="bilinear")
        d2_skip = self.skip_p2(p2)
        d2 = self.up_d3(d3_up) + d2_skip

        h_d2 = self.h_to_d2(F.interpolate(h_ace, size=d2.shape[2:], mode="bilinear"))
        d2 = self.fusion_d2(d2, h_d2)

        d2_final = self.final_d2(d2)

        return d2_final


class FreqPixelShuffleV1(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, scale: int = 2):
        super().__init__()
        self.scale = scale
        self.conv = DSConv(in_channels, out_channels * scale, k=3, s=1, p=1)

    def forward(self, x: Tensor) -> Any:
        x = self.conv(x)
        b, c_r, h, w = x.shape
        out_c = c_r // self.scale

        x = x.view(b, out_c, self.scale, h, w)

        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(b, out_c, h, w * self.scale)

        return x


class ProgressiveUpsampleHeadV1(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, target_bins: int = 1025):
        super().__init__()
        self.target_bins = target_bins

        c = in_channels

        self.block1 = FreqPixelShuffleV1(c, c, scale=2)
        self.block2 = FreqPixelShuffleV1(c, c // 2, scale=2)
        self.block3 = FreqPixelShuffleV1(c // 2, c // 2, scale=2)
        self.block4 = FreqPixelShuffleV1(c // 2, c // 4, scale=2)

        self.final_conv = nn.Conv2d(c // 4, out_channels, kernel_size=1, bias=False)

    def forward(self, x: Tensor) -> Any:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        if x.shape[-1] != self.target_bins:
            x = F.interpolate(
                x,
                size=(x.shape[2], self.target_bins),
                mode="bilinear",
                align_corners=False,
            )

        x = self.final_conv(x)
        return x


class FreqPixelShuffleV2(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, scale: int, f: int):
        super().__init__()
        self.scale = scale
        self.conv = DSConv(in_channels, out_channels * scale)
        self.out_conv = build_hyperace_tfc_tdf(out_channels, out_channels, 2, f)

    def forward(self, x: Tensor) -> Any:
        x = self.conv(x)
        b, c_r, h, w = x.shape
        out_c = c_r // self.scale

        x = x.view(b, out_c, self.scale, h, w)

        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(b, out_c, h, w * self.scale)

        return self.out_conv(x)


class ProgressiveUpsampleHeadV2(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, target_bins: int = 1025, in_bands: int = 62
    ):
        super().__init__()
        self.target_bins = target_bins

        c = in_channels

        self.block1 = FreqPixelShuffleV2(c, c // 2, scale=2, f=in_bands * 2)
        self.block2 = FreqPixelShuffleV2(c // 2, c // 4, scale=2, f=in_bands * 4)
        self.block3 = FreqPixelShuffleV2(c // 4, c // 8, scale=2, f=in_bands * 8)
        self.block4 = FreqPixelShuffleV2(c // 8, c // 16, scale=2, f=in_bands * 16)

        self.final_conv = nn.Conv2d(
            c // 16,
            out_channels,
            kernel_size=3,
            stride=1,
            padding="same",
            bias=False,
        )

    def forward(self, x: Tensor) -> Any:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        if x.shape[-1] != self.target_bins:
            x = F.interpolate(
                x,
                size=(x.shape[2], self.target_bins),
                mode="bilinear",
                align_corners=False,
            )

        x = self.final_conv(x)
        return x


class SegmModelHyperAceV1(nn.Module):
    def __init__(
        self,
        in_bands: int = 62,
        in_dim: int = 256,
        out_bins: int = 1025,
        out_channels: int = 4,
        base_channels: int = 64,
        base_depth: int = 2,
        num_hyperedges: int = 16,
        num_heads: int = 8,
    ):
        super().__init__()

        self.backbone = BackboneHyperAceV1(
            in_channels=in_dim,
            base_channels=base_channels,
            base_depth=base_depth,
        )
        enc_channels = self.backbone.out_channels
        c2, c3, c4, c5 = enc_channels

        hyperace_in_channels = enc_channels
        hyperace_out_channels = c4
        self.hyperace = HyperACE(
            hyperace_in_channels,
            hyperace_out_channels,
            num_hyperedges,
            num_heads,
            k=3,
            l=2,
        )

        decoder_channels = [c2, c3, c4, c5]
        self.decoder = DecoderHyperAce(enc_channels, hyperace_out_channels, decoder_channels)

        self.upsample_head = ProgressiveUpsampleHeadV1(
            in_channels=decoder_channels[0],
            out_channels=out_channels,
            target_bins=out_bins,
        )

    def forward(self, x: Tensor) -> Any:
        h, _ = x.shape[2:]

        enc_feats = self.backbone(x)

        h_ace_feats = self.hyperace(enc_feats)

        dec_feat = self.decoder(enc_feats, h_ace_feats)

        feat_time_restored = F.interpolate(
            dec_feat,
            size=(h, dec_feat.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )

        out = self.upsample_head(feat_time_restored)

        return out


class SegmModelHyperAceV2(nn.Module):
    def __init__(
        self,
        in_bands: int = 62,
        in_dim: int = 256,
        out_bins: int = 1025,
        out_channels: int = 4,
        base_channels: int = 64,
        base_depth: int = 2,
        num_hyperedges: int = 32,
        num_heads: int = 8,
    ):
        super().__init__()

        self.backbone = BackboneHyperAceV2(
            in_channels=in_dim,
            base_channels=base_channels,
            base_depth=base_depth,
        )
        enc_channels = self.backbone.out_channels
        c2, c3, c4, c5 = enc_channels

        hyperace_in_channels = enc_channels
        hyperace_out_channels = c4
        self.hyperace = HyperACE(
            hyperace_in_channels,
            hyperace_out_channels,
            num_hyperedges,
            num_heads,
            k=2,
            l=1,
        )

        decoder_channels = [c2, c3, c4, c5]
        self.decoder = DecoderHyperAce(enc_channels, hyperace_out_channels, decoder_channels)

        self.upsample_head = ProgressiveUpsampleHeadV2(
            in_channels=decoder_channels[0],
            out_channels=out_channels,
            target_bins=out_bins,
            in_bands=in_bands,
        )

    def forward(self, x: Tensor) -> Any:
        h, _ = x.shape[2:]

        enc_feats = self.backbone(x)

        h_ace_feats = self.hyperace(enc_feats)

        dec_feat = self.decoder(enc_feats, h_ace_feats)

        feat_time_restored = F.interpolate(
            dec_feat,
            size=(h, dec_feat.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )

        out = self.upsample_head(feat_time_restored)

        return out
