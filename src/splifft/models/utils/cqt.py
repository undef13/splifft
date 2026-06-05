"""Utilities for Constant-Q transform.

1. Schörkhuber, C. and Klapuri, A., 2010. Constant-Q transform toolbox for
   music processing. In Proc. 7th Sound and Music Computing Conf.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, NamedTuple, Sequence, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from typing_extensions import assert_never

if TYPE_CHECKING:
    from ... import types as t


def _cqt_window_values(
    *,
    window: t.WindowShape,
    local_index: Tensor,
    lengths: Tensor,
    dtype: torch.dtype,
) -> Tensor:
    local = local_index.to(dtype=dtype)
    denom = lengths.unsqueeze(1).to(dtype=dtype)
    phase = 2 * torch.pi * local / denom
    if window == "hann":
        return 0.5 - 0.5 * torch.cos(phase)
    elif window == "hamming":
        return 0.54 - 0.46 * torch.cos(phase)
    assert_never(window)


class RegularCqtKernels(NamedTuple):
    kernels: Tensor
    fft_length: int
    sqrt_lengths: Tensor
    frequencies: Tensor


def _create_regular_cqt_kernels(
    *,
    Q: float,
    fs: t.SampleRate,
    fmin: t.Gt0[float],
    n_bins: t.Gt0[int],
    bins_per_octave: t.Gt0[int],
    norm: int = 1,
    window: t.WindowShape = "hann",
    gamma: t.Ge0[float] = 0.0,
    dtype: torch.dtype = torch.float32,
) -> RegularCqtKernels:
    freq_indices = torch.arange(n_bins, dtype=dtype)
    freqs = fmin * (2.0 ** (freq_indices / float(bins_per_octave)))
    if float(freqs.max()) > fs / 2:
        raise ValueError(f"top cqt bin {float(freqs.max())} exceeds nyquist for sample_rate={fs}")

    alpha = 2.0 ** (1.0 / bins_per_octave) - 1.0
    lengths = torch.ceil(Q * fs / (freqs + gamma / alpha))
    max_len = int(lengths.max().item())
    fft_len = 1 << int(math.ceil(math.log2(max_len)))

    lengths_i = lengths.to(dtype=torch.int64)
    half_center = torch.full((n_bins,), fft_len / 2.0, dtype=dtype)
    starts = torch.ceil(half_center - lengths / 2.0).to(torch.int64) - (lengths_i % 2)

    time_index = torch.arange(fft_len, dtype=torch.int64).unsqueeze(0)
    local_index = time_index - starts.unsqueeze(1)
    valid = (local_index >= 0) & (local_index < lengths_i.unsqueeze(1))

    window_vals = _cqt_window_values(
        window=window,
        local_index=local_index,
        lengths=lengths_i,
        dtype=dtype,
    )
    window_vals = torch.where(valid, window_vals, torch.zeros_like(window_vals))

    neg_half = torch.div(-lengths_i, 2, rounding_mode="floor").unsqueeze(1)
    centered_n = (local_index + neg_half).to(dtype=dtype)
    phase = (2 * torch.pi / float(fs)) * freqs.unsqueeze(1) * centered_n
    signal = window_vals * torch.exp(1j * phase) / lengths.unsqueeze(1)
    signal = torch.where(valid, signal, torch.zeros_like(signal))

    if norm:
        denom = torch.linalg.vector_norm(signal, ord=norm, dim=1, keepdim=True).clamp_min(1e-12)
        signal = signal / denom

    kernels = signal.to(dtype=torch.complex64)
    return RegularCqtKernels(kernels, fft_len, lengths.sqrt().unsqueeze(-1), freqs)


def _resolve_centered_fmin(
    fmin: t.Gt0[float], *, bins_per_semitone: t.Gt0[int], center_bins: bool
) -> float:
    if not center_bins:
        return fmin
    return fmin / 2 ** ((bins_per_semitone - 1) / (12 * bins_per_semitone))


class RegularCQT(nn.Module):
    def __init__(
        self,
        *,
        sr: t.SampleRate,
        hop_length: t.HopSize,
        fmin: t.Gt0[float],
        n_bins: t.Gt0[int],
        bins_per_octave: t.Gt0[int],
        gamma: t.Ge0[float],
        center: bool,
        filter_scale: t.Gt0[float] = 1.0,
        window: t.WindowShape = "hann",
        norm: int = 1,
    ):
        super().__init__()
        self.n_bins = n_bins

        Q = filter_scale / (2 ** (1 / bins_per_octave) - 1)
        kernel_bank = _create_regular_cqt_kernels(
            Q=Q,
            fs=sr,
            fmin=fmin,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
            norm=norm,
            window=window,
            gamma=gamma,
        )

        self.register_buffer("sqrt_lengths", kernel_bank.sqrt_lengths, persistent=False)
        self.register_buffer("kernels", kernel_bank.kernels, persistent=False)
        self.register_buffer("frequencies", kernel_bank.frequencies, persistent=False)

        padding = kernel_bank.fft_length // 2 if center else 0
        self.conv = nn.Conv1d(
            1,
            2 * n_bins,
            kernel_size=kernel_bank.fft_length,
            stride=hop_length,
            padding=padding,
            padding_mode="reflect",
            bias=False,
        )
        self._init_weights()

    @torch.no_grad()
    def _init_weights(self) -> None:
        kernels = cast(Tensor, self.kernels).to(self.conv.weight.device)
        weights = torch.cat((kernels.real, -kernels.imag), dim=0).unsqueeze(1)
        self.conv.weight.copy_(weights)
        self.conv.weight.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3:
            raise ValueError(f"expected shape (batch,time) or (batch,1,time), got {tuple(x.shape)}")

        cqt = self.conv(x).view(x.size(0), 2, self.n_bins, -1)
        cqt = cqt * self.sqrt_lengths.to(cqt.device)
        return cast(Tensor, cqt.permute(0, 2, 3, 1))


class HarmonicRegularCQT(nn.Module):
    def __init__(
        self,
        *,
        harmonics: Sequence[t.Gt0[int]],
        sr: t.SampleRate,
        hop_length: t.HopSize,
        fmin: t.Gt0[float],
        bins_per_semitone: t.Gt0[int],
        n_bins: t.Gt0[int],
        center_bins: bool,
        gamma: t.Ge0[float],
        center: bool,
        filter_scale: t.Gt0[float] = 1.0,
        window: t.WindowShape = "hann",
    ):
        super().__init__()
        fmin = _resolve_centered_fmin(
            fmin,
            bins_per_semitone=bins_per_semitone,
            center_bins=center_bins,
        )

        self.cqt_layers = nn.ModuleList(
            [
                RegularCQT(
                    sr=sr,
                    hop_length=hop_length,
                    fmin=h * fmin,
                    n_bins=n_bins,
                    bins_per_octave=12 * bins_per_semitone,
                    gamma=gamma,
                    center=center,
                    filter_scale=filter_scale,
                    window=window,
                )
                for h in harmonics
            ]
        )

    def forward(self, audio_waveforms: Tensor) -> Tensor:
        return torch.stack([cqt(audio_waveforms) for cqt in self.cqt_layers], dim=1)


def _periodic_hann(window_length: int) -> np.ndarray:
    indices = np.arange(window_length)
    return 0.5 - 0.5 * np.cos(2 * np.pi * indices / window_length)


def _symmetric_hamming(window_length: int) -> np.ndarray:
    indices = np.arange(window_length)
    return 0.54 - 0.46 * np.cos(2 * np.pi * indices / (window_length - 1))


def _firwin2(numtaps: int, freq: np.ndarray, gain: np.ndarray) -> np.ndarray:
    nyquist = 1.0
    nfreqs = 1 + 2 ** int(np.ceil(np.log2(numtaps)))

    mesh = np.linspace(0.0, nyquist, nfreqs)
    response = np.interp(mesh, freq, gain)
    shift = np.exp(-(numtaps - 1) / 2.0 * 1j * np.pi * mesh / nyquist)
    response = response * shift

    window = _symmetric_hamming(numtaps)
    return np.fft.irfft(response)[:numtaps] * window


def _create_recursive_lowpass_filter(
    *,
    band_center: float,
    kernel_length: int,
    transition_bandwidth: float,
) -> Tensor:
    passband_max = band_center / (1 + transition_bandwidth)
    stopband_min = band_center * (1 + transition_bandwidth)
    key_frequencies = np.array([0.0, passband_max, stopband_min, 1.0], dtype=np.float64)
    gains = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float64)
    return torch.tensor(_firwin2(kernel_length, key_frequencies, gains), dtype=torch.float32)


class RecursiveCqtKernels(NamedTuple):
    kernels: np.ndarray
    fft_length: int


def _create_recursive_cqt_kernels(
    *,
    Q: float,
    fs: t.SampleRate,
    fmin: t.Gt0[float],
    n_bins: t.Gt0[int],
    bins_per_octave: t.Gt0[int],
    norm: int,
) -> RecursiveCqtKernels:
    fft_length = 2 ** int(np.ceil(np.log2(np.ceil(Q * fs / fmin))))
    freqs = fmin * 2.0 ** (np.r_[0:n_bins] / float(bins_per_octave))
    kernels = np.zeros((int(n_bins), int(fft_length)), dtype=np.complex64)

    for index in range(int(n_bins)):
        freq = freqs[index]
        kernel_length = np.ceil(Q * fs / freq)
        start = int(np.ceil(fft_length / 2.0 - kernel_length / 2.0)) - int(kernel_length % 2)
        window = _periodic_hann(int(kernel_length))
        signal = window * np.exp(
            np.r_[-kernel_length // 2 : kernel_length // 2] * 1j * 2 * np.pi * freq / fs
        )
        signal = signal / kernel_length
        if norm:
            signal = signal / np.linalg.norm(signal, norm)
        kernels[index, start : start + int(kernel_length)] = signal

    return RecursiveCqtKernels(kernels, fft_length)


def _downsampling_by_n(x: Tensor, filter_kernel: Tensor, factor: int) -> Tensor:
    padding = (filter_kernel.shape[-1] - 1) // 2
    padded = F.pad(x, (padding, padding))
    return F.conv1d(padded, filter_kernel.view(1, 1, -1), stride=factor)


def _get_recursive_cqt_complex(
    x: Tensor,
    cqt_kernels_real: Tensor,
    cqt_kernels_imag: Tensor,
    *,
    hop_length: int,
    padding: int,
) -> Tensor:
    padded = F.pad(x, (padding, padding), mode="reflect")
    real = F.conv1d(padded, cqt_kernels_real, stride=hop_length)
    imag = -F.conv1d(padded, cqt_kernels_imag, stride=hop_length)
    return torch.stack((real, imag), dim=-1)


def _early_downsample_count(
    *, nyquist_hz: float, filter_cutoff_hz: float, hop_length: int, n_octaves: int
) -> int:
    count_one = max(0, int(np.ceil(np.log2(0.85 * nyquist_hz / filter_cutoff_hz)) - 1) - 1)
    count_two = max(0, int(np.ceil(np.log2(hop_length))) - n_octaves + 1)
    return min(count_one, count_two)


class RecursiveCQT(nn.Module):
    """Recursive octave-decimation CQT.

    This follows the octave structure in [1], but like nnAudio it uses FIR
    anti-alias filtering and per-frame reflect padding instead of the paper's
    forward-reverse Butterworth decimation filter and global zero-padding scheme
    """

    def __init__(
        self,
        *,
        sr: t.SampleRate,
        hop_length: t.HopSize,
        fmin: t.Gt0[float],
        n_bins: t.Gt0[int],
        bins_per_octave: t.Gt0[int],
        center: bool,
        filter_scale: t.Gt0[float] = 1.0,
        basis_norm: int = 1,
    ):
        super().__init__()

        self.sample_rate = sr
        self.hop_length = hop_length
        self.fmin = fmin
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave
        self.center = center

        Q = float(filter_scale) / (2 ** (1 / bins_per_octave) - 1)
        n_filters = min(bins_per_octave, n_bins)
        self.n_octaves = int(np.ceil(float(n_bins) / bins_per_octave))

        fmin_top = fmin * 2 ** (self.n_octaves - 1)
        remainder = n_bins % bins_per_octave
        if remainder == 0:
            fmax_top = fmin_top * 2 ** ((bins_per_octave - 1) / bins_per_octave)
        else:
            fmax_top = fmin_top * 2 ** ((remainder - 1) / bins_per_octave)
        fmin_top = fmax_top / 2 ** (1 - 1 / bins_per_octave)

        if fmax_top > sr / 2:
            raise ValueError(f"top CQT bin {fmax_top} exceeds Nyquist for sample_rate={sr}")

        window_bandwidth = 1.5
        filter_cutoff = fmax_top * (1 + 0.5 * window_bandwidth / Q)
        if (
            _early_downsample_count(
                nyquist_hz=sr // 2,
                filter_cutoff_hz=filter_cutoff,
                hop_length=hop_length,
                n_octaves=self.n_octaves,
            )
            != 0
        ):
            raise NotImplementedError(
                "RecursiveCQT early downsampling is not implemented for this frontend"
            )

        kernel_bank = _create_recursive_cqt_kernels(
            Q=Q,
            fs=sr,
            fmin=fmin_top,
            n_bins=n_filters,
            bins_per_octave=bins_per_octave,
            norm=basis_norm,
        )
        self.n_fft = kernel_bank.fft_length

        freqs = fmin * 2.0 ** (np.r_[0:n_bins] / float(bins_per_octave))
        lengths = np.ceil(Q * sr / freqs)
        self.register_buffer(
            "frequencies", torch.tensor(freqs, dtype=torch.float32), persistent=False
        )

        self.register_buffer(
            "lengths", torch.tensor(lengths, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "cqt_kernels_real",
            torch.tensor(np.real(kernel_bank.kernels), dtype=torch.float32).unsqueeze(1),
            persistent=False,
        )
        self.register_buffer(
            "cqt_kernels_imag",
            torch.tensor(np.imag(kernel_bank.kernels), dtype=torch.float32).unsqueeze(1),
            persistent=False,
        )
        self.register_buffer(
            "lowpass_filter",
            _create_recursive_lowpass_filter(
                band_center=0.5, kernel_length=256, transition_bandwidth=0.001
            ),
            persistent=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3:
            raise ValueError(f"expected shape (batch,time) or (batch,1,time), got {tuple(x.shape)}")

        cqt_kernels_real = cast(Tensor, self.cqt_kernels_real)
        cqt_kernels_imag = cast(Tensor, self.cqt_kernels_imag)
        lowpass_filter = cast(Tensor, self.lowpass_filter)
        lengths = cast(Tensor, self.lengths)

        padding = self.n_fft // 2 if self.center else 0
        hop = self.hop_length
        cqt = _get_recursive_cqt_complex(
            x,
            cqt_kernels_real,
            cqt_kernels_imag,
            hop_length=hop,
            padding=padding,
        )

        x_down = x
        for _ in range(self.n_octaves - 1):
            hop //= 2
            x_down = _downsampling_by_n(x_down, lowpass_filter, 2)
            next_octave = _get_recursive_cqt_complex(
                x_down,
                cqt_kernels_real,
                cqt_kernels_imag,
                hop_length=hop,
                padding=padding,
            )
            cqt = torch.cat((next_octave, cqt), dim=1)

        cqt = cqt[:, -self.n_bins :, :]
        cqt = cqt * torch.sqrt(lengths.view(1, -1, 1, 1).to(cqt.device))
        return cqt


class HarmonicRecursiveCQT(nn.Module):
    def __init__(
        self,
        *,
        harmonics: Sequence[t.Gt0[int]],
        sr: t.SampleRate,
        hop_length: t.HopSize,
        fmin: t.Gt0[float],
        bins_per_semitone: t.Gt0[int],
        n_bins: t.Gt0[int],
        center_bins: bool,
        center: bool,
    ):
        super().__init__()
        fmin = _resolve_centered_fmin(
            fmin,
            bins_per_semitone=bins_per_semitone,
            center_bins=center_bins,
        )

        self.cqt_layers = nn.ModuleList(
            [
                RecursiveCQT(
                    sr=sr,
                    hop_length=hop_length,
                    fmin=h * fmin,
                    n_bins=n_bins,
                    bins_per_octave=12 * bins_per_semitone,
                    center=center,
                )
                for h in harmonics
            ]
        )

    def forward(self, audio_waveforms: Tensor) -> Tensor:
        return torch.stack([cqt(audio_waveforms) for cqt in self.cqt_layers], dim=1)
