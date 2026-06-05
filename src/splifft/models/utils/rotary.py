from __future__ import annotations

import torch
from einops import rearrange, repeat
from torch import Tensor, nn

# NOTE: in MT3/YourMT3+, they use partial rotary embeddings and the pixel
# variant instead of the language variant. we do not yet support it


def rotate_half(x: Tensor) -> Tensor:
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


def apply_rotary_embedding(
    x: Tensor,
    *,
    cos: Tensor,
    sin: Tensor,
    sum_dtype: torch.dtype | None = None,
) -> Tensor:
    term1 = x * cos
    term2 = rotate_half(x) * sin
    if sum_dtype is None:
        return term1 + term2
    return (term1.to(sum_dtype) + term2.to(sum_dtype)).to(x.dtype)


class RotaryEmbedding(nn.Module):
    """A performance-oriented version of RoPE for source-separation models.

    Unlike `lucidrains`' implementation which computes embeddings JIT during the
    forward pass and caches calls with the same or shorter sequence length,
    we simply compute them AOT as persistent buffers. To keep the computational
    graph clean, we do not support dynamic sequence lengths, learned frequencies
    or length extrapolation.
    """

    def __init__(
        self,
        seq_len: int,
        dim_head: int,
        *,
        dtype: torch.dtype | None = None,
        theta: int = 10000,
        device: torch.device | None = None,
    ):
        super().__init__()
        if dim_head % 2 != 0:
            raise ValueError(f"rotary dimension must be even, got {dim_head}")
        base_freqs = 1.0 / (
            theta ** (torch.arange(0, dim_head, 2, device=device).float() / dim_head)
        )
        positions = torch.arange(seq_len, dtype=base_freqs.dtype, device=base_freqs.device)
        freqs = torch.einsum("i,j->ij", positions, base_freqs)
        freqs = repeat(freqs, "... d -> ... (d r)", r=2)

        # COMPAT: the original implementation does not generate the embeddings
        # on the fly, but serialises them in fp16. there are some tiny
        # differences:
        # |                     |   from weights  |   generated    |
        # | ------------------- | --------------- | -------------- |
        # | cos_emb_time:971,22 | -0.99462890625  | -0.994140625   |
        # | cos_emb_time:971,23 | -0.99462890625  | -0.994140625   |
        # | sin_emb_time:727,12 | -0.457763671875 | -0.4580078125  |
        # | sin_emb_time:727,13 | -0.457763671875 | -0.4580078125  |
        # | sin_emb_time:825,4  | -0.8544921875   | -0.85400390625 |
        # | sin_emb_time:825,5  | -0.8544921875   | -0.85400390625 |
        target_dtype = torch.float16 if dtype is None else dtype
        self.register_buffer("cos_emb", freqs.cos().to(target_dtype), persistent=True)
        self.register_buffer("sin_emb", freqs.sin().to(target_dtype), persistent=True)
        self.cos_emb: torch.Tensor
        self.sin_emb: torch.Tensor

    def embeddings(self, x: Tensor, *, start_index: int = 0) -> tuple[Tensor, Tensor]:
        seq_len = x.shape[-2]
        end_index = start_index + seq_len
        if end_index > self.cos_emb.shape[0]:
            raise ValueError(
                f"requested rotary positions up to {end_index}, "
                f"but only {self.cos_emb.shape[0]} are available"
            )
        cos = self.cos_emb[start_index:end_index].unsqueeze(0).unsqueeze(0).to(x.device, x.dtype)
        sin = self.sin_emb[start_index:end_index].unsqueeze(0).unsqueeze(0).to(x.device, x.dtype)
        return cos, sin

    def forward(self, x: Tensor, *, start_index: int = 0) -> Tensor:
        cos, sin = self.embeddings(x, start_index=start_index)
        # NOTE: original impl performed addition between two f32s but it comes
        # with 30% slowdown. we eliminate it so the addition is performed in the
        # query / key dtype instead.
        return apply_rotary_embedding(x, cos=cos, sin=sin)
