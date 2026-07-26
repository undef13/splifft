from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from . import log_once, parse_version

if TYPE_CHECKING:
    from torch._C import _SDPBackend

logger = logging.getLogger(__name__)

try:
    from sageattention import sageattn  # type: ignore

    _has_sage_attention = True
except ImportError:
    _has_sage_attention = False


AttentionBackend = Literal["math", "sdpa", "sage"]


class Attend(nn.Module):
    def __init__(
        self,
        dropout: float = 0.0,
        backend: AttentionBackend = "math",
        scale: float | None = None,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        self.backend = backend

        if self.backend == "sage" and not _has_sage_attention:
            log_once(logger, "sageattention requested but not found. falling back to sdpa.")
            self.backend = "sdpa"

        if self.backend == "sdpa" and parse_version(torch.__version__) < (2, 0, 0):
            log_once(logger, "sdpa requested but pytorch < 2.0.0. falling back to math.")
            self.backend = "math"

        self.use_sage = self.backend == "sage"
        self.use_sdpa = self.backend == "sdpa"

        self.cpu_backends = [
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.MATH,
        ]
        # Selected lazily: modules built on CPU (e.g. under a Modal memory
        # snapshot, where no GPU is attached) may later be moved to CUDA.
        self.cuda_backends: list[_SDPBackend] | None = None

    def _select_cuda_backends(self) -> list[_SDPBackend]:
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

        return cuda_backends

    def forward(self, q: Tensor, k: Tensor, v: Tensor, *, is_causal: bool = False) -> Tensor:
        """
        einstein notation

        - b: batch
        - h: heads
        - i, j: sequence length (source, target)
        - d: feature dimension
        """
        if self.use_sage:
            # NOTE: sage requires (is_causal and q.shape[-2] != k.shape[-2]) and q.shape[-1] in (64, 96, 128)
            # and q.dtype in (torch.float16, torch.bfloat16) but we just fail through for now.
            return sageattn(  # type: ignore
                q, k, v, tensor_layout="HND", is_causal=is_causal, sm_scale=self.scale
            )

        if self.use_sdpa:
            is_cuda = q.is_cuda
            if is_cuda and self.cuda_backends is None:
                self.cuda_backends = self._select_cuda_backends()
            backends = self.cuda_backends if is_cuda else self.cpu_backends
            with sdpa_kernel(backends=backends):  # type: ignore
                return F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=is_causal,
                    scale=self.scale,
                )

        scale = self.scale or q.shape[-1] ** -0.5

        # similarity
        sim = torch.einsum("b h i d, b h j d -> b h i j", q, k) * scale

        if is_causal:
            i, j = q.shape[-2], k.shape[-2]
            causal_mask = torch.ones((i, j), device=q.device, dtype=torch.bool).triu_(
                diagonal=j - i + 1
            )
            sim.masked_fill_(causal_mask, torch.finfo(sim.dtype).min)

        # attention
        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # aggregate values
        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)

        return out
