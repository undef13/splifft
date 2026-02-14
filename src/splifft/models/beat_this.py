"""Beat This! Beat Tracker.

- See: https://arxiv.org/abs/2407.21658
- Adapted from: <https://github.com/CPJKU/beat_this>.
- License: MIT
"""
# Copyright (c) 2024 Institute of Computational Perception, JKU Linz, Austria

from __future__ import annotations

import contextlib
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import Tensor, nn

from . import ModelParamsLike
from .bs_roformer import (
    Attention,
    FeedForward,
    RotaryEmbedding,
    Transformer,
)

if TYPE_CHECKING:
    from .. import types as t
    from ..config import TorchDtype


@dataclass
class BeatThisParams(ModelParamsLike):
    chunk_size: t.ChunkSize
    output_stem_names: tuple[t.ModelOutputStemName, ...]
    spect_dim: t.Gt0[int] = 128
    transformer_dim: t.Gt0[int] = 512
    ff_mult: t.Gt0[int] = 4
    n_layers: t.Gt0[int] = 6
    head_dim: t.Gt0[int] = 32
    stem_dim: t.Gt0[int] = 32
    dropout_frontend: t.Dropout = 0.1
    dropout_transformer: t.Dropout = 0.2
    sum_head: bool = True
    partial_transformers: bool = True
    rotary_embed_dtype: TorchDtype | None = None
    transformer_residual_dtype: TorchDtype | None = None
    log_mel_hop_length: int = 441
    """The hop length of the log mel spectrogram.

    !!! warning
        This **must** match the `hop_length` in the `LogMelConfig` to ensure the rotary embeddings
        are sized correctly for the sequence length.
    """

    @property
    def input_type(self) -> t.ModelInputType:
        return "spectrogram"

    @property
    def output_type(self) -> t.ModelOutputType:
        return "logits"


class PartialFTTransformer(nn.Module):
    """Takes a (batch, channels, freqs, time) input, applies self-attention and
    a feed-forward block once across frequencies and once across time."""

    def __init__(
        self,
        dim: int,
        dim_head: int,
        n_head: int,
        rotary_embed_f: RotaryEmbedding,
        rotary_embed_t: RotaryEmbedding,
        dropout: float,
    ):
        super().__init__()
        self.attnF = Attention(
            dim, heads=n_head, dim_head=dim_head, dropout=dropout, rotary_embed=rotary_embed_f
        )
        self.ffF = FeedForward(dim, dropout=dropout)
        self.attnT = Attention(
            dim, heads=n_head, dim_head=dim_head, dropout=dropout, rotary_embed=rotary_embed_t
        )
        self.ffT = FeedForward(dim, dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:
        b = len(x)
        x = rearrange(x, "b c f t -> (b t) f c")
        x = x + self.attnF(x)
        x = x + self.ffF(x)
        x = rearrange(x, "(b t) f c -> (b f) t c", b=b)
        x = x + self.attnT(x)
        x = x + self.ffT(x)
        x = rearrange(x, "(b f) t c -> b c f t", b=b)
        return x


class SumHead(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.beat_downbeat_lin = nn.Linear(input_dim, 2)

    def forward(self, x: Tensor) -> Tensor:
        beat_downbeat = self.beat_downbeat_lin(x)
        beat, downbeat = rearrange(beat_downbeat, "b t c -> c b t", c=2)

        # aggregate beats and downbeats prediction
        # autocast to float16 disabled to avoid numerical issues causing NaNs
        device_type = beat.device.type
        if device_type != "mps" and torch.amp.is_autocast_available(device_type):  # type: ignore
            disable_autocast = torch.autocast(device_type, enabled=False)
        else:
            disable_autocast = contextlib.nullcontext()

        with disable_autocast:
            beat = beat.float() + downbeat.float()
        return torch.stack([beat, downbeat], dim=0)  # (2, b, t)


class Head(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.beat_downbeat_lin = nn.Linear(input_dim, 2)

    def forward(self, x: Tensor) -> Tensor:
        beat_downbeat = self.beat_downbeat_lin(x)
        beat, downbeat = rearrange(beat_downbeat, "b t c -> c b t", c=2)
        return torch.stack([beat, downbeat], dim=0)


class BeatThis(nn.Module):
    def __init__(self, cfg: BeatThisParams):
        super().__init__()

        max_frames = cfg.chunk_size // cfg.log_mel_hop_length
        rotary_embed_t = RotaryEmbedding(
            seq_len=max_frames,  # by default 1500 frames * 441 = 661500 samples
            dim_head=cfg.head_dim,
            dtype=cfg.rotary_embed_dtype,
        )

        # NOTE: removed rearrange from original impl to standardise input as (B, F, T)
        stem = nn.Sequential(
            OrderedDict(
                bn1d=nn.BatchNorm1d(cfg.spect_dim),
                add_channel=Rearrange("b f t -> b 1 f t"),
                conv2d=nn.Conv2d(
                    in_channels=1,
                    out_channels=cfg.stem_dim,
                    kernel_size=(4, 3),
                    stride=(4, 1),
                    padding=(0, 1),
                    bias=False,
                ),
                bn2d=nn.BatchNorm2d(cfg.stem_dim),
                activation=nn.GELU(),
            )
        )

        spect_dim = cfg.spect_dim // 4
        dim = cfg.stem_dim
        frontend_blocks = []
        for _ in range(3):
            rotary_embed_f = RotaryEmbedding(
                seq_len=spect_dim,
                dim_head=cfg.head_dim,
                dtype=cfg.rotary_embed_dtype,
            )
            frontend_blocks.append(
                nn.Sequential(
                    OrderedDict(
                        partial=(
                            PartialFTTransformer(
                                dim=dim,
                                dim_head=cfg.head_dim,
                                n_head=dim // cfg.head_dim,
                                rotary_embed_f=rotary_embed_f,
                                rotary_embed_t=rotary_embed_t,
                                dropout=cfg.dropout_frontend,
                            )
                            if cfg.partial_transformers
                            else nn.Identity()
                        ),
                        conv2d=nn.Conv2d(
                            in_channels=dim,
                            out_channels=dim * 2,
                            kernel_size=(2, 3),
                            stride=(2, 1),
                            padding=(0, 1),
                            bias=False,
                        ),
                        norm=nn.BatchNorm2d(dim * 2),
                        activation=nn.GELU(),
                    )
                )
            )
            dim *= 2
            spect_dim //= 2

        self.frontend = nn.Sequential(
            OrderedDict(
                stem=stem,
                blocks=nn.Sequential(*frontend_blocks),
                concat=Rearrange("b c f t -> b t (c f)"),
                linear=nn.Linear(dim * spect_dim, cfg.transformer_dim),
            )
        )

        n_heads = cfg.transformer_dim // cfg.head_dim
        # TODO check if this is really equivalent
        self.transformer_blocks = Transformer(
            dim=cfg.transformer_dim,
            depth=cfg.n_layers,
            dim_head=cfg.head_dim,
            heads=n_heads,
            attn_dropout=cfg.dropout_transformer,
            ff_dropout=cfg.dropout_transformer,
            ff_mult=cfg.ff_mult,
            norm_output=True,
            rotary_embed=rotary_embed_t,
            transformer_residual_dtype=cfg.transformer_residual_dtype,
        )

        if cfg.sum_head:
            self.task_heads = SumHead(cfg.transformer_dim)
        else:
            self.task_heads = Head(cfg.transformer_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: Input spectrogram (B, F, T)
        :return: Logits (2, B, T) -> [Beats, Downbeats]
        """
        x = self.frontend(x)
        x = self.transformer_blocks(x)
        x = self.task_heads(x)
        return x
