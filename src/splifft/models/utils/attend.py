from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor, einsum, nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from . import log_once, parse_version

if TYPE_CHECKING:
    from torch._C import _SDPBackend
logger = logging.getLogger(__name__)


class Attend(nn.Module):
    def __init__(
        self, dropout: float = 0.0, flash: bool = False, scale: float | None = None
    ) -> None:
        super().__init__()
        self.scale = scale
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        self.flash = flash
        assert not (flash and parse_version(torch.__version__) < (2, 0, 0)), (
            "expected pytorch >= 2.0.0 to use flash attention"
        )

        # determine efficient attention configs for cuda and cpu
        self.cpu_backends = [
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.MATH,
        ]
        self.cuda_backends: list[_SDPBackend] | None = None

        if not torch.cuda.is_available() or not flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device("cuda"))
        device_version = parse_version(f"{device_properties.major}.{device_properties.minor}")

        if device_version >= (8, 0):
            if os.name == "nt":
                cuda_backends = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
                log_once(logger, f"windows detected, using {cuda_backends=}")
            else:
                cuda_backends = [SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]
                log_once(logger, f"gpu compute capability >= 8.0, using {cuda_backends=}")
        else:
            cuda_backends = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
            log_once(logger, f"gpu compute capability < 8.0, using {cuda_backends=}")

        self.cuda_backends = cuda_backends

    def flash_attn(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        _, _heads, _q_len, _, _k_len, is_cuda, _device = (
            *q.shape,
            k.shape[-2],
            q.is_cuda,
            q.device,
        )  # type: ignore

        if self.scale is not None:
            default_scale = q.shape[-1] ** -0.5
            q = q * (self.scale / default_scale)

        backends = self.cuda_backends if is_cuda else self.cpu_backends
        # pytorch 2.0 flash attn: q, k, v, mask, dropout, softmax_scale
        with sdpa_kernel(backends=backends):  # type: ignore
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout if self.training else 0.0
            )

        return out

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """
        einstein notation

        - b: batch
        - h: heads
        - n, i, j: sequence length (base sequence length, source, target)
        - d: feature dimension
        """
        _q_len, _k_len, _device = q.shape[-2], k.shape[-2], q.device

        scale = self.scale or q.shape[-1] ** -0.5

        if self.flash:
            return self.flash_attn(q, k, v)

        # similarity
        sim = einsum("b h i d, b h j d -> b h i j", q, k) * scale

        # attention
        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # aggregate values
        out = einsum("b h i j, b h j d -> b h i d", attn, v)

        return out
