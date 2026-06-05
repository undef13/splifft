from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

if TYPE_CHECKING:
    from ... import types as t


class Stft(nn.Module):
    """A custom STFT implementation using 1D convolutions to ensure compatibility with CoreML."""

    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        win_length: int,
        window_fn: Callable[[int], Tensor],
        conv_dtype: torch.dtype | None,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.conv_dtype = conv_dtype

        window = window_fn(self.win_length)

        dft_mat = torch.fft.fft(torch.eye(self.n_fft, device=window.device))
        dft_mat_T = dft_mat.T

        real_kernels = dft_mat_T.real[
            : self.win_length, : (self.n_fft // 2 + 1)
        ] * window.unsqueeze(-1)
        imag_kernels = dft_mat_T.imag[
            : self.win_length, : (self.n_fft // 2 + 1)
        ] * window.unsqueeze(-1)

        # (out_channels, in_channels, kernel_size)
        self.register_buffer("real_conv_weight", real_kernels.T.unsqueeze(1).to(self.conv_dtype))
        self.register_buffer("imag_conv_weight", imag_kernels.T.unsqueeze(1).to(self.conv_dtype))

    def forward(self, x: Tensor) -> t.ComplexSpectrogram:
        b, s, t = x.shape
        x = x.reshape(b * s, 1, t).to(self.conv_dtype)

        padding = self.n_fft // 2
        x = F.pad(x, (padding, padding), "reflect")

        # NOTE: keeping the real / imag passes separate.
        # a single concatenated conv is exact in f16 on CUDA but diverges in f32
        real_part = F.conv1d(x, self.real_conv_weight, stride=self.hop_length)  # type: ignore
        imag_part = F.conv1d(x, self.imag_conv_weight, stride=self.hop_length)  # type: ignore
        spec = torch.stack((real_part, imag_part), dim=-1)  # (b*s, f, t_frames, c=2)

        _bs, f, t_frames, c = spec.shape
        spec = spec.view(b, s, f, t_frames, c)

        return spec  # type: ignore


class IStft(nn.Module):
    """A simple wrapper around torch.istft with a hacky workaround for MPS.

    TODO: implement a proper workaround.
    """

    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        win_length: int,
        window_fn: Callable[[int], Tensor] = torch.hann_window,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window_fn(self.win_length)

    def forward(
        self, spec: t.ComplexSpectrogram, length: int | None = None
    ) -> t.RawAudioTensor | t.NormalizedAudioTensor:
        device = spec.device
        is_mps = device.type == "mps"
        window = self.window.to(device)
        # see https://github.com/lucidrains/BS-RoFormer/issues/47
        # this would introduce a breaking change.
        # spec = spec.index_fill(1, torch.tensor(0, device=spec.device), 0.)  # type: ignore
        spec_complex = torch.view_as_complex(spec)

        try:
            audio = torch.istft(
                spec_complex,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=window,
                return_complex=False,
                length=length,
            )
        except RuntimeError:
            audio = torch.istft(
                spec_complex.cpu() if is_mps else spec_complex,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=window.cpu() if is_mps else window,
                return_complex=False,
                length=length,
            ).to(device)

        return audio  # type: ignore
