"""Reusable, pure algorithmic components for inference and training."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Callable,
    Generic,
    Iterator,
    Protocol,
    TypeVar,
    assert_never,
    cast,
)

import torch
import torch.nn.functional as F
from annotated_types import Gt
from einops import rearrange
from torch import Tensor, nn

from . import types as t
from .models.utils.stft import IStft, Stft

if TYPE_CHECKING:
    from typing import Mapping, Sequence

    from .config import (
        DerivedStemsConfig,
        FeatureExtractionConfig,
        MaskingConfig,
        StemName,
        StftConfig,
    )


_AudioTensorLike = TypeVar("_AudioTensorLike")


@dataclass
class Audio(Generic[_AudioTensorLike]):
    data: _AudioTensorLike
    """This should either be an [raw][splifft.types.RawAudioTensor] or a
    [normalized][splifft.types.NormalizedAudioTensor] audio tensor."""
    sample_rate: t.SampleRate


#
# normalization
#


@dataclass
class NormalizationStats:
    """Statistics for [normalizing](https://en.wikipedia.org/wiki/Feature_scaling#Standardization_(Z-score_Normalization))
    and denormalizing audio.
    """

    mean: float
    r"""Mean $\mu$ of the mixture"""
    std: Annotated[float, Gt(0)]
    r"""Standard deviation $\sigma$ of the mixture"""


@dataclass
class NormalizedAudio:
    """Container for normalized audio and its original stats."""

    audio: Audio[t.NormalizedAudioTensor]  # NOTE: composition over inheritance.
    stats: NormalizationStats


def normalize_audio(audio: Audio[t.RawAudioTensor]) -> NormalizedAudio:
    """Preprocess the raw audio in the time domain to have a mean of 0 and a std of 1
    before passing it to the model.

    Operates on the mean of the [channels][splifft.types.Channels].
    """
    mono_audio = audio.data.mean(dim=0)
    mean = float(mono_audio.mean())
    std = float(mono_audio.std())

    if std <= 1e-8:  # silent audio
        return NormalizedAudio(
            audio=Audio(data=t.NormalizedAudioTensor(audio.data), sample_rate=audio.sample_rate),
            stats=NormalizationStats(mean, 1.0),
        )

    normalized_data = (audio.data - mean) / std
    return NormalizedAudio(
        audio=Audio(data=t.NormalizedAudioTensor(normalized_data), sample_rate=audio.sample_rate),
        stats=NormalizationStats(mean, std),
    )


def denormalize_audio(
    audio_data: t.NormalizedAudioTensor, stats: NormalizationStats
) -> t.RawAudioTensor:
    """Take the model output and restore them to their original loudness."""
    return t.RawAudioTensor((audio_data * stats.std) + stats.mean)


#
# chunking
#


def generate_chunks(
    audio_data: t.RawAudioTensor | t.NormalizedAudioTensor,
    chunk_size: t.ChunkSize,
    hop_size: t.HopSize,
    batch_size: t.BatchSize,
    *,
    padding_mode: t.PaddingMode = "reflect",
) -> Iterator[t.PaddedChunkedAudioTensor]:
    """Generates batches of overlapping chunks from an audio tensor.

    :return: An iterator that yields batches of chunks of shape (B, C, chunk_T).
    """
    padding = chunk_size - hop_size
    padded_audio = F.pad(audio_data, (padding, padding), mode=padding_mode)

    padded_len = padded_audio.shape[-1]
    rem = (padded_len - chunk_size) % hop_size
    if rem != 0:
        final_pad = hop_size - rem
        padded_audio = F.pad(padded_audio, (0, final_pad), mode="constant", value=0)

    unfolded = padded_audio.unfold(
        dimension=-1, size=chunk_size, step=hop_size
    )  # (C, num_chunks, chunk_size)

    num_chunks = unfolded.shape[1]
    unfolded = unfolded.permute(1, 0, 2)  # (num_chunks, C, chunk_size)

    for i in range(0, num_chunks, batch_size):
        yield t.PaddedChunkedAudioTensor(unfolded[i : i + batch_size])


def stitch_chunks(
    processed_chunks: Sequence[t.SeparatedChunkedTensor],
    num_stems: t.NumModelStems,
    chunk_size: t.ChunkSize,
    hop_size: t.HopSize,
    target_num_samples: t.Samples,
    *,
    window: t.WindowTensor,
) -> t.RawSeparatedTensor:
    r"""Stitches processed audio chunks back together using the [overlap-add method](https://en.wikipedia.org/wiki/Overlap%E2%80%93add_method).

    Reconstructs the full audio signal from a sequence of overlapping, processed chunks. Ensures
    that the sum of all overlapping windows is constant at every time step:
    $\sum_{m=-\infty}^{\infty} w[n - mH] = C$ where $H$ is the [hop size][splifft.types.HopSize].
    """
    all_chunks = torch.cat(tuple(processed_chunks), dim=0)
    total_chunks, _N, num_channels, _chunk_T = all_chunks.shape
    windowed_chunks = all_chunks * window.view(1, 1, 1, -1)

    # folding: (B, N * C * chunk_T) -> (1, N * C * chunk_T, total_chunks)
    reshaped_for_fold = windowed_chunks.permute(1, 2, 3, 0).reshape(
        1, num_stems * num_channels * chunk_size, total_chunks
    )

    total_length = (total_chunks - 1) * hop_size + chunk_size

    folded = F.fold(
        reshaped_for_fold,
        output_size=(1, total_length),
        kernel_size=(1, chunk_size),
        stride=(1, hop_size),
    )  # (1, N * C, 1, total_length)
    stitched = folded.view(num_stems, num_channels, total_length)

    # normalization for overlap-add
    windows_to_fold = window.expand(total_chunks, 1, chunk_size)
    reshaped_windows_for_fold = windows_to_fold.permute(1, 2, 0).reshape(
        1, chunk_size, total_chunks
    )
    norm_window = F.fold(
        reshaped_windows_for_fold,
        output_size=(1, total_length),
        kernel_size=(1, chunk_size),
        stride=(1, hop_size),
    ).squeeze(0)

    norm_window.clamp_min_(1e-8)  # for edges where the window sum might be zero
    stitched /= norm_window

    padding = chunk_size - hop_size
    if padding > 0:
        stitched = stitched[..., padding:-padding]

    return t.RawSeparatedTensor(stitched[..., :target_num_samples])


def aggregate_logits(
    processed_chunks: Sequence[t.LogitsTensor],
    starts: Sequence[int],
    full_size: int,
    chunk_size: int,
    num_stems: int,
    *,
    trim_margin: int = 0,
    overlap_mode: t.OverlapMode = "keep_first",
) -> t.LogitsTensor:
    """Stitches time-series logits (split/aggregate strategy).

    This is a 1:1 map of beat_this's aggregation behavior:
    - trim `trim_margin` frames from each chunk side
    - write into a full-size buffer
    - in `keep_first` mode, process chunks in reverse so earlier chunks
      overwrite later ones
    """
    all_chunks = torch.cat(tuple(processed_chunks), dim=0)
    total_chunks, _, chunk_len_frames = all_chunks.shape

    if len(starts) != total_chunks:
        raise ValueError(f"expected {total_chunks=} starts, got {len(starts)}")
    if chunk_len_frames != chunk_size:
        raise ValueError(f"expected {chunk_size=} but got chunk length {chunk_len_frames}")
    if trim_margin * 2 >= chunk_len_frames:
        raise ValueError(f"{trim_margin=} is too large for {chunk_len_frames=}")

    buffer = torch.full(
        (num_stems, full_size), -1000.0, device=all_chunks.device, dtype=all_chunks.dtype
    )

    if overlap_mode == "keep_first":
        indices = range(total_chunks - 1, -1, -1)
    elif overlap_mode == "keep_last":
        indices = range(total_chunks)
    else:
        assert_never(overlap_mode)

    for i in indices:
        chunk = all_chunks[i]
        chunk_valid = chunk[:, trim_margin : chunk_len_frames - trim_margin]
        start = starts[i] + trim_margin
        end = starts[i] + chunk_size - trim_margin

        clipped_start = max(0, start)
        clipped_end = min(end, full_size)
        if clipped_start >= clipped_end:
            continue

        src_start = clipped_start - start
        src_end = src_start + (clipped_end - clipped_start)
        buffer[:, clipped_start:clipped_end] = chunk_valid[:, src_start:src_end]

    return t.LogitsTensor(buffer)


def aggregate_sequence_chunks(
    processed_chunks: Sequence[Tensor],
    starts: Sequence[int],
    full_size: int,
    chunk_size: int,
    *,
    trim_margin: int = 0,
    overlap_mode: t.OverlapMode = "keep_first",
) -> Tensor:
    """Aggregate generic time-major chunk outputs.

    Each `processed_chunks[i]` must have shape `(chunk_time, ...)` where `...` can
    contain any additional feature dimensions (for example bins for `activations`).
    """
    if not processed_chunks:
        raise ValueError("expected at least one chunk")

    chunk_len_frames = int(processed_chunks[0].shape[0])
    if trim_margin * 2 >= chunk_len_frames:
        raise ValueError(f"{trim_margin=} is too large for {chunk_len_frames=}")
    if len(starts) != len(processed_chunks):
        raise ValueError(f"expected {len(processed_chunks)=} starts, got {len(starts)}")
    if chunk_len_frames != chunk_size:
        raise ValueError(f"expected {chunk_size=} but got chunk length {chunk_len_frames}")

    tail_shape = tuple(processed_chunks[0].shape[1:])
    buffer = processed_chunks[0].new_zeros((full_size, *tail_shape))

    if overlap_mode == "keep_first":
        indices = range(len(processed_chunks) - 1, -1, -1)
    elif overlap_mode == "keep_last":
        indices = range(len(processed_chunks))
    else:
        assert_never(overlap_mode)

    for i in indices:
        chunk = processed_chunks[i]
        if tuple(chunk.shape[1:]) != tail_shape:
            raise ValueError(
                f"all stream chunks must have identical non-time dimensions, got {chunk.shape[1:]} and {tail_shape}"
            )

        chunk_valid = chunk[trim_margin : chunk_len_frames - trim_margin]
        start = starts[i] + trim_margin
        end = starts[i] + chunk_size - trim_margin

        clipped_start = max(0, start)
        clipped_end = min(end, full_size)
        if clipped_start >= clipped_end:
            continue

        src_start = clipped_start - start
        src_end = src_start + (clipped_end - clipped_start)
        buffer[clipped_start:clipped_end] = chunk_valid[src_start:src_end]

    return buffer


def pad_dim(tensor: Tensor, *, dim: int, pad: tuple[int, int], value: float = 0.0) -> Tensor:
    """Pad an arbitrary tensor on a specific dimension.

    This avoids relying on `F.pad`'s reverse-dimension argument ordering.
    """
    left, right = pad
    if left < 0 or right < 0:
        raise ValueError(f"expected non-negative pad widths, got left={left}, right={right}")
    if left == 0 and right == 0:
        return tensor

    rank = tensor.ndim
    resolved_dim = dim if dim >= 0 else rank + dim
    if resolved_dim < 0 or resolved_dim >= rank:
        raise IndexError(f"dim out of range for rank-{rank} tensor: {dim}")

    pieces: list[Tensor] = []
    if left:
        left_shape = list(tensor.shape)
        left_shape[resolved_dim] = left
        pieces.append(tensor.new_full(left_shape, fill_value=value))

    pieces.append(tensor)

    if right:
        right_shape = list(tensor.shape)
        right_shape[resolved_dim] = right
        pieces.append(tensor.new_full(right_shape, fill_value=value))

    return torch.cat(pieces, dim=resolved_dim)


def split_sequence_tensor(
    sequence: Tensor,
    chunk_size: int,
    *,
    trim_margin: int = 0,
    avoid_short_end: bool = True,
) -> tuple[list[Tensor], list[int]]:
    """Split a time-major sequence tensor into overlapping chunks.

    `sequence` must be shaped `(time, ...)`, where `...` can be any feature tail.
    """
    full_size = sequence.shape[0]
    if (step := chunk_size - 2 * trim_margin) <= 0:
        raise ValueError(
            f"expected chunk_size - 2*trim_margin > 0, got {chunk_size=}, {trim_margin=}"
        )
    if not (starts := list(range(-trim_margin, full_size - trim_margin, step))):
        starts = [-trim_margin]
    if avoid_short_end and full_size > step:
        starts[-1] = full_size - (chunk_size - trim_margin)

    chunks: list[Tensor] = []
    for start in starts:
        src_start = max(start, 0)
        src_end = min(start + chunk_size, full_size)
        left = max(0, -start)
        right = max(0, start + chunk_size - full_size)

        chunk = sequence[src_start:src_end]
        if left > 0 or right > 0:
            chunk = pad_dim(chunk, dim=0, pad=(left, right), value=0.0)
        chunks.append(chunk)

    return chunks, starts


def apply_mask(
    spec_for_masking: t.ComplexSpectrogram,
    mask_batch: t.ComplexSpectrogram,
    mask_add_sub_dtype: torch.dtype | None,
    mask_out_dtype: torch.dtype | None,
) -> t.SeparatedSpectrogramTensor:
    """Applies a complex mask to a spectrogram.

    While this can be simply replaced by a complex multiplication and `torch.view_as_complex`,
    CoreML does not support it: https://github.com/apple/coremltools/issues/2003 so we handroll our
    own.
    """
    spec_real = spec_for_masking[..., 0]
    spec_imag = spec_for_masking[..., 1]
    mask_real = mask_batch[..., 0]
    mask_imag = mask_batch[..., 1]

    # see: 14385, 14401, 14392, 14408
    ac = spec_real * mask_real
    bd = spec_imag * mask_imag
    ad = spec_real * mask_imag
    bc = spec_imag * mask_real

    # see: 509, 506, 505, 504, 741, 747
    out_real = ac.to(mask_add_sub_dtype) - bd.to(mask_add_sub_dtype)
    out_imag = ad.to(mask_add_sub_dtype) + bc.to(mask_add_sub_dtype)

    # see: 503, 501
    separated_spec = torch.stack([out_real, out_imag], dim=-1).to(mask_out_dtype)
    return t.SeparatedSpectrogramTensor(separated_spec)


def get_model_floating_dtype(model: nn.Module) -> torch.dtype | None:
    """Infer floating input dtype from the model's first floating parameter."""
    first_param = next(model.parameters(), None)
    if first_param is None:
        return None
    return first_param.dtype if first_param.is_floating_point() else None


def to_model_device(
    tensor: Tensor,
    *,
    model_device: torch.device,
    model_floating_dtype: torch.dtype | None,
) -> Tensor:
    """Move tensor to model device while preserving model floating dtype compatibility."""
    if model_floating_dtype is not None and tensor.is_floating_point():
        return tensor.to(device=model_device, dtype=model_floating_dtype)
    return tensor.to(device=model_device)


#
# handle different i/o types
#


class ModelWaveformToWaveform(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        preprocess: t.PreprocessFn,
        postprocess: t.PostprocessFn,
        *,
        io_device: torch.device,
        model_device: torch.device,
    ):
        super().__init__()
        self.model = model
        self.preprocess = preprocess
        self.postprocess = postprocess
        self.io_device = io_device
        self.model_device = model_device
        self.model_input_dtype = get_model_floating_dtype(self.model)

    def forward(
        self, waveform_chunk: t.RawAudioTensor | t.NormalizedAudioTensor
    ) -> t.SeparatedChunkedTensor | t.LogitsTensor:
        model_waveform_chunk = cast(
            t.RawAudioTensor | t.NormalizedAudioTensor,
            to_model_device(
                waveform_chunk,
                model_device=self.model_device,
                model_floating_dtype=self.model_input_dtype,
            ),
        )
        preprocessed_input = self.preprocess(model_waveform_chunk)
        model_output = self.model(*preprocessed_input)
        postprocessed = self.postprocess(model_output, *preprocessed_input)
        if isinstance(postprocessed, Tensor):
            return cast(
                t.SeparatedChunkedTensor | t.LogitsTensor,
                postprocessed.to(self.io_device),
            )
        return postprocessed


def create_w2w_model(
    model: nn.Module,
    model_input_type: t.ModelInputType,
    model_output_type: t.ModelOutputType,
    stft_cfg: StftConfig | None,
    num_channels: t.Channels,
    chunk_size: t.ChunkSize,
    masking_cfg: MaskingConfig,
    *,
    io_device: torch.device,
    model_device: torch.device,
) -> ModelWaveformToWaveform:
    needs_stft = model_input_type == "spectrogram" or model_input_type == "waveform_and_spectrogram"
    needs_istft = model_output_type == "spectrogram_mask" or model_output_type == "spectrogram"

    if (needs_stft or needs_istft) and stft_cfg is None:
        raise ValueError(
            "expected stft config for models that operate on spectrograms, but found `None`."
        )

    preprocess: t.PreprocessFn = lambda chunk: (chunk,)  # noqa: E731
    postprocess: t.PostprocessFn = lambda model_output, *_: model_output  # noqa: E731

    if needs_stft:
        assert stft_cfg is not None
        conv_dtype = stft_cfg.conv_dtype
        if model_device.type == "cpu" and conv_dtype == torch.float16:
            conv_dtype = torch.float32

        stft_module = Stft(
            n_fft=stft_cfg.n_fft,
            hop_length=stft_cfg.hop_length,
            win_length=stft_cfg.win_length,
            window_fn=lambda win_len: _get_window_fn(stft_cfg.window_shape, win_len, model_device),
            conv_dtype=conv_dtype,
        ).to(model_device)
        if model_input_type == "spectrogram":
            preprocess = _create_stft_preprocessor(stft_module)
        elif model_input_type == "waveform_and_spectrogram":
            preprocess = _create_hybrid_preprocessor(stft_module)
        else:
            raise NotImplementedError(f"unsupported input type for stft: {model_input_type}")

    if needs_istft:
        assert stft_cfg is not None
        istft_module = IStft(
            n_fft=stft_cfg.n_fft,
            hop_length=stft_cfg.hop_length,
            win_length=stft_cfg.win_length,
            window_fn=lambda win_len: _get_window_fn(stft_cfg.window_shape, win_len, model_device),
        ).to(model_device)

        add_sub_dtype = masking_cfg.add_sub_dtype
        out_dtype = masking_cfg.out_dtype
        if model_device.type == "cpu":
            if add_sub_dtype == torch.float16:
                add_sub_dtype = torch.float32
            if out_dtype == torch.float16:
                out_dtype = torch.float32

        postprocess = _create_spec_postprocessor(
            istft_module,
            num_channels,
            chunk_size,
            add_sub_dtype,
            out_dtype,
            model_output_type,
        )
    return ModelWaveformToWaveform(
        model,
        preprocess,
        postprocess,
        io_device=io_device,
        model_device=model_device,
    )


def _create_stft_preprocessor(
    stft_module: Stft,
) -> Callable[[t.RawAudioTensor | t.NormalizedAudioTensor], tuple[t.ComplexSpectrogram]]:
    def _preprocess(
        chunk_batch: t.RawAudioTensor | t.NormalizedAudioTensor,
    ) -> tuple[t.ComplexSpectrogram]:
        spec_batch = stft_module(chunk_batch)
        b, s, f, t_frames, _ = spec_batch.shape
        model_input = spec_batch.permute(0, 2, 1, 3, 4).reshape(b, f * s, t_frames, 2)
        return (model_input,)

    return _preprocess


def _create_hybrid_preprocessor(
    stft_module: Stft,
) -> Callable[[t.RawAudioTensor | t.NormalizedAudioTensor], t.HybridModelInput]:
    def _preprocess(chunk_batch: t.RawAudioTensor | t.NormalizedAudioTensor) -> t.HybridModelInput:
        spec_batch = stft_module(chunk_batch)
        spec_batch_rearranged = rearrange(spec_batch, "b s f t c -> b (f s) t c")
        return (spec_batch_rearranged, chunk_batch)

    return _preprocess


def _create_spec_postprocessor(
    istft_module: IStft,
    num_channels: t.Channels,
    chunk_size: t.ChunkSize,
    mask_add_sub_dtype: torch.dtype | None,
    mask_out_dtype: torch.dtype | None,
    model_output_type: t.ModelOutputType,
) -> Callable[[t.ComplexSpectrogram, t.ComplexSpectrogram], t.SeparatedChunkedTensor]:
    def _postprocess(
        model_output: t.ComplexSpectrogram, spec_chunk: t.ComplexSpectrogram
    ) -> t.SeparatedChunkedTensor:
        separated_spec: t.SeparatedSpectrogramTensor

        if model_output_type == "spectrogram_mask":
            separated_spec = apply_mask(
                t.ComplexSpectrogram(spec_chunk.unsqueeze(1)),
                model_output,
                mask_add_sub_dtype,
                mask_out_dtype,
            )
        elif model_output_type == "spectrogram":
            separated_spec = t.SeparatedSpectrogramTensor(model_output)
        else:
            raise ValueError(f"Unsupported model output type: {model_output_type}")

        separated_spec = rearrange(separated_spec, "b n (f s) t c -> (b n s) f t c", s=num_channels)
        # COMPAT: note that istft is NOT part of the graph. 14454 implies fp16 but because
        # torch ComplexHalf is experimental, we explicitly cast to f32.
        separated_wave_chunk = istft_module(separated_spec.to(torch.float32), length=chunk_size)
        separated_wave_chunk_ = rearrange(
            separated_wave_chunk,
            "(b n s) t -> b n s t",
            b=spec_chunk.shape[0],
            s=num_channels,
        )
        return t.SeparatedChunkedTensor(separated_wave_chunk_)

    return _postprocess


class LogMelSpect(nn.Module):
    """Computes the log-mel spectrogram of a waveform."""

    def __init__(
        self,
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        n_mels: int,
        f_min: float = 0.0,
        f_max: float | None = None,
        mel_scale: str = "slaney",
        normalized: bool | str = "frame_length",
        power: float = 1.0,
        log_multiplier: float = 1000.0,
    ):
        super().__init__()
        import torchaudio.transforms as T

        self.spect_class = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            mel_scale=mel_scale,
            normalized=normalized,
            power=power,
        )
        self.log_multiplier = log_multiplier

    def forward(self, x: Tensor) -> t.LogMelSpectrogram:
        """
        :param x: Waveform tensor of shape (batch, channels, time) or (batch, time)
        :return: Log-Mel spectrogram of shape (batch, channels, n_mels, time)
        """
        if x.ndim == 2:
            x = x.unsqueeze(1)
        mel_spec = self.spect_class(x)
        return torch.log1p(self.log_multiplier * mel_spec)  # type: ignore


def to_log_magnitude(x: Tensor, *, epsilon: float = 1e-8) -> Tensor:
    """Convert complex or real spectrogram-like tensors to dB log-magnitude."""
    if x.shape[-1] == 2:
        x = torch.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2)
    else:
        x = x.abs()
    return x.clamp_min(epsilon).log10().mul(20)


def _cqt_window_values(
    *,
    window: str,
    local_index: Tensor,
    lengths: Tensor,
    dtype: torch.dtype,
) -> Tensor:
    """Vectorized periodic windows matching SciPy `get_window(..., fftbins=True)` semantics."""
    local = local_index.to(dtype=dtype)
    denom = lengths.unsqueeze(1).to(dtype=dtype)
    phase = 2 * torch.pi * local / denom
    if window == "hann":
        return 0.5 - 0.5 * torch.cos(phase)
    if window == "hamming":
        return 0.54 - 0.46 * torch.cos(phase)
    raise ValueError(f"unsupported CQT window={window!r}")


def create_cqt_kernels(
    *,
    Q: float,
    fs: int,
    fmin: float,
    n_bins: int,
    bins_per_octave: int,
    norm: int = 1,
    window: str = "hann",
    fmax: float | None = None,
    gamma: float = 0.0,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, int, Tensor, Tensor]:
    """Create time-domain CQT kernels using only PyTorch ops.

    This mirrors the nnAudio-style kernel generation used by PESTO but avoids
    SciPy so `splifft` can keep a minimal dependency surface.
    """
    del fmax
    freq_indices = torch.arange(n_bins, device=device, dtype=dtype)
    freqs = fmin * (2.0 ** (freq_indices / float(bins_per_octave)))

    alpha = 2.0 ** (1.0 / bins_per_octave) - 1.0
    lengths = torch.ceil(Q * fs / (freqs + gamma / alpha))
    max_len = int(lengths.max().item())
    fft_len = 1 << int(math.ceil(math.log2(max_len)))

    lengths_i = lengths.to(dtype=torch.int64)
    half_center = torch.full((n_bins,), fft_len / 2.0, device=device, dtype=dtype)
    starts = torch.ceil(half_center - lengths / 2.0).to(torch.int64) - (lengths_i % 2)

    time_index = torch.arange(fft_len, device=device, dtype=torch.int64).unsqueeze(0)
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
    return kernels, fft_len, lengths.sqrt().unsqueeze(-1), freqs


class SequenceFeatureExtractor(Protocol):
    """Protocol for sequence feature extractors.

    Required contract:
    - input: `(B, C, T)`
    - output: `(B, seq_len, feature_dim)`
    """

    hop_length_samples: int
    stage_name: str

    def __call__(self, x: Tensor) -> Tensor: ...


def _ensure_btf(x: Tensor, *, source: str) -> Tensor:
    if x.ndim != 3:
        raise ValueError(f"{source} must output `(B,T,F)`, got shape={tuple(x.shape)}")
    return x


class IdentitySequenceFeatureExtractor(nn.Module, SequenceFeatureExtractor):
    hop_length_samples = 1
    stage_name = "sequence_features"

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected shape (B,C,T), got {tuple(x.shape)}")
        if x.shape[1] != 1:
            raise ValueError(
                f"identity sequence extractor expects mono input with C=1, got shape={tuple(x.shape)}"
            )
        return _ensure_btf(x.transpose(1, 2), source="identity extractor")


class LogMelSequenceFeatureExtractor(nn.Module, SequenceFeatureExtractor):
    stage_name = "mel"

    def __init__(self, mel: LogMelSpect, *, hop_length_samples: int):
        super().__init__()
        self.mel = mel
        self.hop_length_samples = hop_length_samples

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected shape (B,C,T), got {tuple(x.shape)}")
        if x.shape[1] != 1:
            raise ValueError(
                f"mel extractor expects mono input with C=1, got shape={tuple(x.shape)}"
            )
        # TODO dont do this
        x_mono = x.mean(dim=1, keepdim=True)
        mel = self.mel(x_mono).squeeze(1)
        return _ensure_btf(mel.transpose(1, 2), source="mel extractor")


class CqtSequenceFeatureExtractor(nn.Module, SequenceFeatureExtractor):
    stage_name = "cqt"

    def __init__(
        self,
        hcqt: HarmonicCQT,
        *,
        hop_length_samples: int,
        log_epsilon: float,
    ):
        super().__init__()
        self.hcqt = hcqt
        self.hop_length_samples = hop_length_samples
        self.log_epsilon = log_epsilon

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected shape (B,C,T), got {tuple(x.shape)}")
        if x.shape[1] != 1:
            raise ValueError(
                f"cqt extractor expects mono input with C=1, got shape={tuple(x.shape)}"
            )
        x_mono = x.mean(dim=1)
        hcqt_output = self.hcqt(x_mono)
        cqt = to_log_magnitude(hcqt_output, epsilon=self.log_epsilon).permute(0, 3, 1, 2)
        b, t_len, harmonics, bins = cqt.shape
        return _ensure_btf(cqt.reshape(b, t_len, harmonics * bins), source="cqt extractor")


def create_sequence_feature_extractor(
    feature_cfg: FeatureExtractionConfig | None,
    *,
    sample_rate: t.SampleRate,
    device: torch.device,
) -> SequenceFeatureExtractor:
    if feature_cfg is None:
        return IdentitySequenceFeatureExtractor()

    if feature_cfg.kind == "mel":
        mel = LogMelSpect(
            sample_rate=feature_cfg.sample_rate,
            n_fft=feature_cfg.n_fft,
            hop_length=feature_cfg.hop_length,
            n_mels=feature_cfg.n_mels,
            f_min=feature_cfg.f_min,
            f_max=feature_cfg.f_max,
            mel_scale=feature_cfg.mel_scale,
            normalized=feature_cfg.normalized,
            power=feature_cfg.power,
            log_multiplier=feature_cfg.log_multiplier,
        ).to(device)
        return LogMelSequenceFeatureExtractor(mel=mel, hop_length_samples=feature_cfg.hop_length)

    if feature_cfg.kind == "cqt":
        hop_length_samples = int(round(feature_cfg.hop_size_ms * sample_rate / 1000))
        hcqt = HarmonicCQT(
            sr=sample_rate,
            hop_length=hop_length_samples,
            harmonics=feature_cfg.harmonics,
            fmin=feature_cfg.fmin,
            fmax=feature_cfg.fmax,
            bins_per_semitone=feature_cfg.bins_per_semitone,
            n_bins=feature_cfg.n_bins,
            center_bins=feature_cfg.center_bins,
            gamma=feature_cfg.gamma,
            center=feature_cfg.center,
        ).to(device)
        return CqtSequenceFeatureExtractor(
            hcqt=hcqt,
            hop_length_samples=hop_length_samples,
            log_epsilon=feature_cfg.log_epsilon,
        )

    raise ValueError(f"unsupported feature extractor kind: {feature_cfg.kind}")


class CQT(nn.Module):
    """Constant-Q transform layer (complex output) implemented via `Conv1d`."""

    def __init__(
        self,
        *,
        sr: t.SampleRate,
        hop_length: int,
        fmin: float,
        fmax: float | None,
        n_bins: int,
        bins_per_octave: int,
        gamma: float,
        center: bool,
        window: str = "hann",
        norm: int = 1,
    ):
        super().__init__()
        self.n_bins = n_bins

        Q = 1.0 / (2 ** (1 / bins_per_octave) - 1)
        kernels, kernel_width, sqrt_lengths, _freqs = create_cqt_kernels(
            Q=Q,
            fs=sr,
            fmin=fmin,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
            norm=norm,
            window=window,
            fmax=fmax,
            gamma=gamma,
            device=torch.device("cpu"),
        )

        self.register_buffer("sqrt_lengths", sqrt_lengths, persistent=False)
        self.register_buffer("kernel_real_imag", kernels, persistent=False)

        padding = kernel_width // 2 if center else 0
        self.conv = nn.Conv1d(
            1,
            2 * n_bins,
            kernel_size=kernel_width,
            stride=hop_length,
            padding=padding,
            padding_mode="reflect",
            bias=False,
        )
        self._init_weights()

    @torch.no_grad()
    def _init_weights(self) -> None:
        kernels = cast(Tensor, self.kernel_real_imag).to(self.conv.weight.device)
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


class HarmonicCQT(nn.Module):
    """Harmonic CQT computed by stacking one CQT per harmonic multiplier."""

    def __init__(
        self,
        *,
        harmonics: Sequence[int],
        sr: t.SampleRate,
        hop_length: int,
        fmin: float,
        fmax: float | None,
        bins_per_semitone: int,
        n_bins: int,
        center_bins: bool,
        gamma: float,
        center: bool,
    ):
        super().__init__()
        if center_bins:
            fmin = fmin / 2 ** ((bins_per_semitone - 1) / (12 * bins_per_semitone))

        self.cqt_kernels = nn.ModuleList(
            [
                CQT(
                    sr=sr,
                    hop_length=hop_length,
                    fmin=h * fmin,
                    fmax=fmax,
                    n_bins=n_bins,
                    bins_per_octave=12 * bins_per_semitone,
                    gamma=gamma,
                    center=center,
                )
                for h in harmonics
            ]
        )

    def forward(self, audio_waveforms: Tensor) -> Tensor:
        return torch.stack([cqt(audio_waveforms) for cqt in self.cqt_kernels], dim=1)


def _get_window_fn(name: str, win_length: int, device: torch.device) -> t.WindowTensor:
    # intentionally keeping it sealed and simple, not using getattr().
    fn: Callable[[int], Tensor]
    if name == "hann":
        fn = torch.hann_window
    elif name == "hamming":
        fn = torch.hamming_window
    else:
        raise ValueError(f"unknown window function: {name}")

    return t.WindowTensor(fn(win_length, device=device))


#
# stem postprocessing
#
def derive_stems(
    separated_stems: Mapping[t.ModelOutputStemName, t.RawAudioTensor],
    mixture_input: t.RawAudioTensor,
    stem_rules: DerivedStemsConfig,
) -> dict[StemName, t.RawAudioTensor]:
    """
    It is the caller's responsibility to ensure that all tensors are aligned and have the same shape.

    !!! note
        Mixture input and separated stems must first be [denormalized][splifft.core.denormalize_audio].
    """
    stems = {
        "mixture": t.RawAudioTensor(mixture_input),  # for subtraction
        **separated_stems,
    }

    for derived_name, rule in stem_rules.items():
        if rule.operation == "subtract":
            # pydantic should have already validated that the stem names exist so safe to index directly
            minuend = stems[rule.stem_name]
            subtrahend = stems[rule.by_stem_name]
            stems[derived_name] = t.RawAudioTensor(minuend - subtrahend)
        elif rule.operation == "sum":
            to_sum = tuple(stems[s] for s in rule.stem_names)
            stems[derived_name] = t.RawAudioTensor(torch.stack(to_sum).sum(dim=0))

    stems.pop("mixture", None)
    return stems


#
# misc
#


def str_to_torch_dtype(value: Any) -> torch.dtype:
    if not isinstance(value, str):
        raise TypeError(f"expected dtype to be a string, got {value} (type {type(value)})")
    try:
        dtype = getattr(torch, value)
    except AttributeError:
        raise ValueError(f"`{value}` cannot be found under the `torch` namespace")
    if not isinstance(dtype, torch.dtype):
        raise TypeError(f"expected {dtype} to be a dtype but it is a {type(dtype)}")
    return dtype
