"""MDX23C TFC-TDF Network.

- Architecture: TFC-TDF v3 (Time-Frequency Convolution with Time-Distributed Fully-connected)
- Original: https://github.com/kuielab/sdx23
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from . import ModelParamsLike

if TYPE_CHECKING:
    from .. import types as t


@dataclass
class MDX23CParams(ModelParamsLike):
    chunk_size: t.ChunkSize
    output_stem_names: tuple[t.ModelOutputStemName, ...]
    dim_f: t.Gt0[int]
    """The size of the frequency dimension fed into the network. 
    Usually smaller than `n_fft // 2 + 1`."""
    num_subbands: t.Gt0[int]
    num_scales: t.Gt0[int]
    scale: tuple[t.Gt0[int], ...]
    """Downscaling factor per scale."""
    num_blocks_per_scale: t.Gt0[int]
    hidden_channels: t.Gt0[int]
    """Base number of channels."""
    growth: t.Gt0[int]
    """Channel growth per scale."""
    bottleneck_factor: t.Gt0[int]
    norm_type: Literal["BatchNorm", "InstanceNorm"] | str = "InstanceNorm"
    act_type: Literal["gelu", "relu", "elu"] | str = "gelu"
    stereo: bool = True

    @property
    def input_type(self) -> t.ModelInputType:
        return "spectrogram"

    @property
    def output_type(self) -> t.ModelOutputType:
        return "spectrogram"

    @property
    def inference_archetype(self) -> t.InferenceArchetype:
        return "frequency_masking"


def get_norm(norm_type: str, channels: int) -> nn.Module:
    if norm_type == "BatchNorm":
        return nn.BatchNorm2d(channels)
    elif norm_type == "InstanceNorm":
        return nn.InstanceNorm2d(channels, affine=True)
    elif "GroupNorm" in norm_type:
        g = int(norm_type.replace("GroupNorm", ""))
        return nn.GroupNorm(num_groups=g, num_channels=channels)
    return nn.Identity()


def get_act(act_type: str) -> nn.Module:
    if act_type == "gelu":
        return nn.GELU()
    elif act_type == "relu":
        return nn.ReLU()
    elif act_type.startswith("elu"):
        try:
            alpha = float(act_type.replace("elu", ""))
        except ValueError:
            alpha = 1.0
        return nn.ELU(alpha)
    raise ValueError(f"unknown activation: {act_type}")


class Upscale(nn.Module):
    def __init__(
        self,
        in_c: int,
        out_c: int,
        scale: tuple[int, int],
        norm_type: str,
        act_type: str,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            get_norm(norm_type, in_c),
            get_act(act_type),
            nn.ConvTranspose2d(
                in_channels=in_c,
                out_channels=out_c,
                kernel_size=scale,
                stride=scale,
                bias=False,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)  # type: ignore


class Downscale(nn.Module):
    def __init__(
        self,
        in_c: int,
        out_c: int,
        scale: tuple[int, int],
        norm_type: str,
        act_type: str,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            get_norm(norm_type, in_c),
            get_act(act_type),
            nn.Conv2d(
                in_channels=in_c,
                out_channels=out_c,
                kernel_size=scale,
                stride=scale,
                bias=False,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)  # type: ignore


class TFC_TDF_Block(nn.Module):
    def __init__(
        self,
        in_c: int,
        c: int,
        f: int,
        bn: int,
        norm_type: str,
        act_type: str,
    ):
        super().__init__()
        self.tfc1 = nn.Sequential(
            get_norm(norm_type, in_c),
            get_act(act_type),
            nn.Conv2d(in_c, c, 3, 1, 1, bias=False),
        )
        self.tdf = nn.Sequential(
            get_norm(norm_type, c),
            get_act(act_type),
            nn.Linear(f, f // bn, bias=False),
            get_norm(norm_type, c),
            get_act(act_type),
            nn.Linear(f // bn, f, bias=False),
        )
        self.tfc2 = nn.Sequential(
            get_norm(norm_type, c),
            get_act(act_type),
            nn.Conv2d(c, c, 3, 1, 1, bias=False),
        )
        self.shortcut = nn.Conv2d(in_c, c, 1, 1, 0, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        s = self.shortcut(x)
        x = self.tfc1(x)
        x = x + self.tdf(x)
        x = self.tfc2(x)
        x = x + s
        return x


class TFC_TDF(nn.Module):
    def __init__(
        self,
        in_c: int,
        c: int,
        blocks_per_scale: int,
        f: int,
        bn: int,
        norm_type: str,
        act_type: str,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                TFC_TDF_Block(in_c if i == 0 else c, c, f, bn, norm_type, act_type)
                for i in range(blocks_per_scale)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class MDX23C(nn.Module):
    def __init__(self, cfg: MDX23CParams):
        super().__init__()
        self.cfg = cfg
        if len(cfg.scale) != 2:
            raise ValueError(f"expected `scale` to have 2 elements, got {cfg.scale}")
        scale_2d = (cfg.scale[0], cfg.scale[1])
        self.num_target_instruments = len(cfg.output_stem_names)
        self.audio_channels = 2 if cfg.stereo else 1
        self.num_subbands = cfg.num_subbands

        dim_c = self.num_subbands * self.audio_channels * 2
        n = cfg.num_scales
        blocks_per_scale = cfg.num_blocks_per_scale
        c = cfg.hidden_channels
        g = cfg.growth
        bn = cfg.bottleneck_factor
        f = cfg.dim_f // self.num_subbands

        self.first_conv = nn.Conv2d(dim_c, c, 1, 1, 0, bias=False)

        self.encoder_blocks = nn.ModuleList()
        for _ in range(n):
            block = nn.Module()
            block.tfc_tdf = TFC_TDF(c, c, blocks_per_scale, f, bn, cfg.norm_type, cfg.act_type)
            block.downscale = Downscale(c, c + g, scale_2d, cfg.norm_type, cfg.act_type)
            f = f // scale_2d[1]
            c += g
            self.encoder_blocks.append(block)

        self.bottleneck_block = TFC_TDF(c, c, blocks_per_scale, f, bn, cfg.norm_type, cfg.act_type)

        self.decoder_blocks = nn.ModuleList()
        for _ in range(n):
            block = nn.Module()
            block.upscale = Upscale(c, c - g, scale_2d, cfg.norm_type, cfg.act_type)
            f = f * scale_2d[1]
            c -= g
            block.tfc_tdf = TFC_TDF(2 * c, c, blocks_per_scale, f, bn, cfg.norm_type, cfg.act_type)
            self.decoder_blocks.append(block)

        self.final_conv = nn.Sequential(
            nn.Conv2d(c + dim_c, c, 1, 1, 0, bias=False),
            get_act(cfg.act_type),
            nn.Conv2d(c, self.num_target_instruments * dim_c, 1, 1, 0, bias=False),
        )

    def cac2cws(self, x: Tensor) -> Tensor:
        return rearrange(x, "b c (k f) t -> b (c k) f t", k=self.num_subbands)

    def cws2cac(self, x: Tensor) -> Tensor:
        return rearrange(x, "b (c k) f t -> b c (k f) t", k=self.num_subbands)

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: input spectrogram (B, F*S, T, 2)
        :return: output spectrogram (B, N, F*S, T, 2)
        """
        b, fs, t, ri = x.shape
        if ri != 2:
            raise ValueError(f"expected final complex axis of size 2, got {ri}")
        f_full = fs // self.audio_channels
        if fs % self.audio_channels != 0:
            raise ValueError(
                f"expected frequency-channel axis divisible by audio channels ({self.audio_channels}), "
                f"got {fs}"
            )

        x_in = rearrange(
            x,
            "b (f s) t ri -> b (s ri) f t",
            s=self.audio_channels,
            ri=2,
        )
        x_in = x_in[..., : self.cfg.dim_f, :]
        mix = x_in = self.cac2cws(x_in)
        first_conv_out = x_in = self.first_conv(x_in)
        x_in = rearrange(x_in, "b c f t -> b c t f")

        encoder_outputs = []
        for block in self.encoder_blocks:
            x_in = block.tfc_tdf(x_in)  # type: ignore
            encoder_outputs.append(x_in)
            x_in = block.downscale(x_in)  # type: ignore

        x_in = self.bottleneck_block(x_in)

        for block in self.decoder_blocks:
            x_in = block.upscale(x_in)  # type: ignore
            x_in = torch.cat([x_in, encoder_outputs.pop()], 1)
            x_in = block.tfc_tdf(x_in)  # type: ignore

        x_in = rearrange(x_in, "b c t f -> b c f t")
        x_in = x_in * first_conv_out
        x_in = self.final_conv(torch.cat([mix, x_in], 1))
        x_in = self.cws2cac(x_in)

        x_in = rearrange(
            x_in,
            "b (n c) f t -> b n c f t",
            n=self.num_target_instruments,
        )

        if f_full > self.cfg.dim_f:
            pad_size = f_full - self.cfg.dim_f
            x_in = torch.nn.functional.pad(x_in, (0, 0, 0, pad_size))

        x_in = rearrange(
            x_in,
            "b n (s ri) f t -> b n (f s) t ri",
            s=self.audio_channels,
            ri=2,
        )

        return x_in
