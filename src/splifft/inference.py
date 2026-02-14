"""High level orchestrator for model inference"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import torch
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from torch import nn

from . import types as t
from .core import (
    LogMelSpect,
    _get_window_fn,
    aggregate_logits,
    create_w2w_model,
    denormalize_audio,
    derive_stems,
    generate_chunks,
    normalize_audio,
    split_spectrogram,
    stitch_chunks,
)

if TYPE_CHECKING:
    from .config import ChunkingConfig, Config, LogMelConfig, MaskingConfig, StemName, StftConfig
    from .core import Audio, NormalizationStats
    from .models import (
        ModelParamsLike,
    )


def run_inference_on_file(
    mixture: Audio[t.RawAudioTensor],
    config: Config,
    model: nn.Module,
    model_params_concrete: ModelParamsLike,
) -> dict[StemName, t.RawAudioTensor] | dict[str, t.LogitsTensor]:
    """Runs the full source separation pipeline on a single audio file."""
    mixture_data: t.RawAudioTensor | t.NormalizedAudioTensor = mixture.data
    mixture_stats: NormalizationStats | None = None
    if config.inference.normalize_input_audio:
        norm_audio = normalize_audio(mixture)
        mixture_data = norm_audio.audio.data
        mixture_stats = norm_audio.stats

    # HACK we dispatch to two very similar functions, which results in code duplication
    # but is unavoidable because:
    # - demucs: waveform -> split -> model -> stitch -> waveform
    # - bs: waveform -> split -> stft -> model -> istft -> stitch -> waveform
    # - beatthis: waveform -> log-mel -> **split** -> model -> **aggregate** -> logits
    # notice that the splitting and aggregation is done in spectrogram-space
    # and has almost completely different logic. so we hardcode it for now.
    # maybe in the future if we can somehow unify it, revisit.
    if config.chunking.strategy == "spectrogram":
        logits_data = _separate_beatthis(
            mixture_data=mixture_data,
            chunk_cfg=config.chunking,
            model=model,
            batch_size=config.inference.batch_size,
            num_model_stems=len(config.model.output_stem_names),
            chunk_size=config.model.chunk_size,
            model_input_type=model_params_concrete.input_type,
            log_mel_cfg=config.log_mel,
            use_autocast_dtype=config.inference.use_autocast_dtype,
        )
        # NOTE: we do not need to support derived stems or do tta for logits
        stems = {}
        for i, name in enumerate(config.model.output_stem_names):
            stems[name] = t.LogitsTensor(logits_data[i])
        return stems

    separated_data = _separate_waveform(
        mixture_data=mixture_data,
        chunk_cfg=config.chunking,
        model=model,
        batch_size=config.inference.batch_size,
        num_model_stems=len(config.model.output_stem_names),
        chunk_size=config.model.chunk_size,
        model_input_type=model_params_concrete.input_type,
        model_output_type=model_params_concrete.output_type,
        stft_cfg=config.stft,
        masking_cfg=config.masking,
        use_autocast_dtype=config.inference.use_autocast_dtype,
    )

    denormalized_stems: dict[t.ModelOutputStemName, t.RawAudioTensor] = {}
    for i, stem_name in enumerate(config.model.output_stem_names):
        stem_data = separated_data[i, ...]
        if mixture_stats is not None:
            stem_data = denormalize_audio(
                audio_data=t.NormalizedAudioTensor(stem_data),
                stats=mixture_stats,
            )
        denormalized_stems[stem_name] = t.RawAudioTensor(stem_data)

    if config.inference.apply_tta:
        raise NotImplementedError

    output_stems = denormalized_stems
    if config.derived_stems:
        output_stems = derive_stems(
            denormalized_stems,
            mixture.data,
            config.derived_stems,
        )

    return output_stems


def _progress_columns(
    batch_size: t.BatchSize,
    device: torch.device,
    use_autocast_dtype: torch.dtype | None,
) -> tuple[ProgressColumn, ...]:
    dtype_str = f" • {use_autocast_dtype}" if use_autocast_dtype else ""
    info_text = f"[cyan](bs=[bold]{batch_size}[/bold] • {device.type}{dtype_str})[/cyan]"
    return (
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        TextColumn(info_text),
    )


def _separate_waveform(
    mixture_data: t.RawAudioTensor | t.NormalizedAudioTensor,
    chunk_cfg: ChunkingConfig,
    model: nn.Module,
    batch_size: t.BatchSize,
    num_model_stems: t.NumModelStems,
    chunk_size: t.ChunkSize,
    model_input_type: t.ModelInputType,
    model_output_type: t.ModelOutputType,
    stft_cfg: StftConfig | None,
    masking_cfg: MaskingConfig,
    *,
    use_autocast_dtype: torch.dtype | None = None,
) -> t.RawSeparatedTensor:
    """Chunk waveform -> model -> overlap-add stitch."""
    if model_output_type == "logits":
        raise ValueError(
            f"logits models must use `{_separate_beatthis.__qualname__}`, not `{_separate_waveform.__qualname__}`"
        )

    device = mixture_data.device
    original_num_samples = mixture_data.shape[-1]
    hop_size = int(chunk_size * (1 - chunk_cfg.overlap_ratio))
    window = _get_window_fn(chunk_cfg.window_shape, chunk_size, device)

    padded_length = original_num_samples + 2 * (chunk_size - hop_size)
    num_chunks = max(0, (padded_length - chunk_size) // hop_size + 1)
    total_batches = math.ceil(num_chunks / batch_size)

    chunk_generator = generate_chunks(
        audio_data=mixture_data,
        chunk_size=chunk_size,
        hop_size=hop_size,
        batch_size=batch_size,
        padding_mode=chunk_cfg.padding_mode,
    )

    model_w2w = create_w2w_model(
        model=model,
        model_input_type=model_input_type,
        model_output_type=model_output_type,
        stft_cfg=stft_cfg,
        num_channels=mixture_data.shape[0],
        chunk_size=chunk_size,
        masking_cfg=masking_cfg,
    )

    separated_chunks: list[t.SeparatedChunkedTensor] = []
    with Progress(
        *_progress_columns(batch_size, device, use_autocast_dtype), transient=True
    ) as progress:
        task = progress.add_task("processing chunks...", total=total_batches)

        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=device.type,
                enabled=use_autocast_dtype is not None,
                dtype=use_autocast_dtype,
            ),
        ):
            for chunk_batch in chunk_generator:
                separated_batch = model_w2w(chunk_batch)
                separated_chunks.append(cast(t.SeparatedChunkedTensor, separated_batch))
                progress.update(task, advance=1)

    return stitch_chunks(
        processed_chunks=separated_chunks,
        num_stems=num_model_stems,
        chunk_size=chunk_size,
        hop_size=hop_size,
        target_num_samples=original_num_samples,
        window=t.WindowTensor(window),
    )


def _separate_beatthis(
    mixture_data: t.RawAudioTensor | t.NormalizedAudioTensor,
    chunk_cfg: ChunkingConfig,
    model: nn.Module,
    batch_size: t.BatchSize,
    num_model_stems: t.NumModelStems,
    chunk_size: t.ChunkSize,
    model_input_type: t.ModelInputType,
    log_mel_cfg: LogMelConfig | None,
    *,
    use_autocast_dtype: torch.dtype | None = None,
) -> t.LogitsTensor:
    """Full-audio log-mel -> split spectrogram -> model -> aggregate logits."""
    if log_mel_cfg is None:
        raise ValueError("expected log_mel config for logits model, but found `None`.")
    if model_input_type != "spectrogram":
        raise ValueError(f"expected spectrogram input for logits model, got {model_input_type}")

    device = mixture_data.device
    mel_preprocessor = LogMelSpect(
        sample_rate=log_mel_cfg.sample_rate,
        n_fft=log_mel_cfg.n_fft,
        hop_length=log_mel_cfg.hop_length,
        n_mels=log_mel_cfg.n_mels,
        f_min=log_mel_cfg.f_min,
        f_max=log_mel_cfg.f_max,
        mel_scale=log_mel_cfg.mel_scale,
        normalized=log_mel_cfg.normalized,
        power=log_mel_cfg.power,
        log_multiplier=log_mel_cfg.log_multiplier,
    ).to(device)

    # hardcoding mono for BeatThis for now
    if mixture_data.shape[0] > 1:
        mixture_mono = mixture_data.mean(dim=0, keepdim=True)
    else:
        mixture_mono = mixture_data

    with (
        torch.inference_mode(),
        torch.autocast(
            device_type=device.type,
            enabled=use_autocast_dtype is not None,
            dtype=use_autocast_dtype,
        ),
    ):
        # (B=1, C=1, F, T) -> (T, F)
        full_spect = mel_preprocessor(mixture_mono).squeeze(0).squeeze(0).transpose(0, 1)

    frame_chunk_size = chunk_size // log_mel_cfg.hop_length
    if frame_chunk_size <= 0:
        raise ValueError(
            f"invalid frame chunk size computed from {chunk_size=} and {log_mel_cfg.hop_length=}"
        )
    trim_margin = chunk_cfg.trim_margin or 0
    spect_chunks, starts = split_spectrogram(
        t.LogMelSpectrogram(full_spect),
        chunk_size=frame_chunk_size,
        trim_margin=trim_margin,
        avoid_short_end=True,
    )

    total_batches = math.ceil(len(spect_chunks) / batch_size)
    logits_chunks: list[t.LogitsTensor] = []
    with Progress(
        *_progress_columns(batch_size, device, use_autocast_dtype), transient=True
    ) as progress:
        task = progress.add_task("processing chunks...", total=total_batches)

        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=device.type,
                enabled=use_autocast_dtype is not None,
                dtype=use_autocast_dtype,
            ),
        ):
            for i in range(0, len(spect_chunks), batch_size):
                batch_chunks = spect_chunks[i : i + batch_size]
                batch_tensors = [torch.as_tensor(chunk) for chunk in batch_chunks]
                # (B, T, F) -> (B, F, T) for BeatThis model input
                model_input = torch.stack(batch_tensors, dim=0).transpose(1, 2)
                logits = model(model_input)
                # model returns (Stems, Batch, Time), we need (Batch, Stems, Time) for aggregation
                logits = logits.permute(1, 0, 2)
                logits_chunks.append(t.LogitsTensor(logits))
                progress.update(task, advance=1)

    return aggregate_logits(
        processed_chunks=logits_chunks,
        starts=starts,
        full_size=full_spect.shape[0],
        chunk_size=frame_chunk_size,
        num_stems=num_model_stems,
        trim_margin=trim_margin,
        overlap_mode=chunk_cfg.overlap_mode or "keep_first",
    )
