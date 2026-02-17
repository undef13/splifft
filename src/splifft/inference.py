"""Public inference APIs."""
# put heavy logic inside core.py, this file should just wire up the components together.

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generator, Literal, TypeAlias, cast

import torch
from torch import nn

from . import PATH_REGISTRY_DEFAULT, core
from . import types as t

if TYPE_CHECKING:
    from pathlib import Path

    from .config import Config, ConfigOverrides, IntoConfig, StemName
    from .models import ModelParamsLike


SUPPORTED_MODELS: dict[str, tuple[str, str]] = {
    "bs_roformer": ("splifft.models.bs_roformer", "BSRoformer"),
    "mel_roformer": ("splifft.models.bs_roformer", "BSRoformer"),
    "mdx23c": ("splifft.models.mdx23c", "MDX23C"),
    "beat_this": ("splifft.models.beat_this", "BeatThis"),
}


def _resolve_runtime_dtype(
    dtype: torch.dtype | None,
    *,
    device: torch.device,
    for_autocast: bool = False,
) -> torch.dtype | None:
    """Return a device-safe dtype fallback for inference-time casting/autocast."""
    if dtype is None:
        return None
    if device.type == "cpu" and dtype == torch.float16:
        # CPU fp16 kernels are incomplete/slow; keep numerics in a broadly supported dtype.
        return torch.bfloat16 if for_autocast else torch.float32
    return dtype


def resolve_model_entrypoint(
    model_type: t.ModelType,
    module_name: str | None,
    class_name: str | None,
) -> tuple[str, str]:
    if module_name is not None and class_name is not None:
        return module_name, class_name
    try:
        return SUPPORTED_MODELS[model_type]
    except KeyError as e:
        raise ValueError(
            f"could not resolve model entrypoint for model_type={model_type!r}; "
            "provide both module and class explicitly"
        ) from e


@dataclass(frozen=True)
class ChunkProcessed:
    batch_index: int
    total_batches: int


@dataclass(frozen=True)
class Stage:
    stage: str
    total_batches: int | None = field(kw_only=True, default=None)

    @dataclass(frozen=True)
    class Started:
        stage: str
        total_batches: int | None = None

    @dataclass(frozen=True)
    class Completed:
        stage: str

    @property
    def started(self) -> Started:
        return Stage.Started(stage=self.stage, total_batches=self.total_batches)

    @property
    def completed(self) -> Completed:
        return Stage.Completed(stage=self.stage)

    def __enter__(self) -> Stage:
        return self

    def __exit__(self, *_: object) -> None:
        return None


@dataclass(frozen=True)
class InferenceOutput:
    outputs: dict[StemName, t.RawAudioTensor] | dict[str, t.LogitsTensor]
    # `sample_rate` exists because if the user passed in a bare path to an audio file,
    # we need to know the sample rate to write the output files.
    sample_rate: t.SampleRate


InferenceEvent: TypeAlias = Stage.Started | ChunkProcessed | Stage.Completed | InferenceOutput


@dataclass(frozen=True)
class InferenceEngine:
    config: Config
    model: nn.Module
    model_params_concrete: ModelParamsLike

    @classmethod
    def from_pretrained(
        cls,
        *,
        config: IntoConfig,
        checkpoint_path: t.StrPath,
        overrides: ConfigOverrides = (),
        device: torch.device | str | None = None,
        module_name: str | None = None,
        class_name: str | None = None,
        package_name: str | None = None,
    ) -> InferenceEngine:
        from .config import into_config
        from .io import load_weights
        from .models import ModelMetadata

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        runtime_device = torch.device(device)

        config = into_config(config, overrides=overrides)
        resolved_module, resolved_class = resolve_model_entrypoint(
            config.model_type, module_name, class_name
        )
        metadata = ModelMetadata.from_module(
            module_name=resolved_module,
            model_cls_name=resolved_class,
            model_type=config.model_type,
            package=package_name,
        )

        model_params = config.model.to_concrete(metadata.params)
        full_output_stems = tuple(config.model.output_stem_names)
        requested_stems = tuple(config.inference.requested_stems or full_output_stems)

        state_dict_transform = None
        if requested_stems != full_output_stems:
            # optional model-level optimization contract: models can choose to
            # provide a stem-selection plan that may mutate params and checkpoint
            # loading. if absent, we keep full model outputs and discard
            # unrelated stems immediately after each forward pass.
            from .models import SupportsStemSelection

            if isinstance(metadata.model, SupportsStemSelection):
                plan = metadata.model.__splifft_stem_selection_plan__(model_params, requested_stems)
                model_params = plan.model_params
                state_dict_transform = plan.state_dict_transform

        model = metadata.model(model_params)
        if (forced_dtype := config.inference.force_weights_dtype) is not None:
            model = model.to(_resolve_runtime_dtype(forced_dtype, device=runtime_device))
        model = load_weights(
            model,
            checkpoint_path,
            device=runtime_device,
            state_dict_transform=state_dict_transform,
        ).eval()

        # maybe we should to an explicit try_compile() method while emitting events but eh.
        # we shuold probably log since it can take extremely long
        if (compile_cfg := config.inference.compile_model) is not None:
            compiled_model = torch.compile(
                model,
                fullgraph=compile_cfg.fullgraph,
                dynamic=compile_cfg.dynamic,
                mode=compile_cfg.mode,
            )
            model = cast(nn.Module, compiled_model)

        return cls(config=config, model=model, model_params_concrete=model_params)

    @classmethod
    def from_registry(
        cls,
        model_id: str,
        *,
        device: torch.device | str | None = None,
        overrides: ConfigOverrides = (),
        fetch_if_missing: bool = True,
        force_overwrite_config: bool = False,
        force_overwrite_model: bool = False,
        registry_path: Path = PATH_REGISTRY_DEFAULT,
    ) -> InferenceEngine:
        from .config import Registry
        from .io import get_model_paths

        model_paths = get_model_paths(
            model_id,
            fetch_if_missing=fetch_if_missing,
            force_overwrite_config=force_overwrite_config,
            force_overwrite_model=force_overwrite_model,
            registry=Registry.from_file(registry_path),
        )
        return cls.from_pretrained(
            config=model_paths.path_config,
            checkpoint_path=model_paths.path_checkpoint,
            overrides=overrides,
            device=device,
        )

    def _requested_output_stem_names(self) -> tuple[t.ModelOutputStemName, ...]:
        return (
            self.config.model.output_stem_names
            if self.config.inference.requested_stems is None
            else self.config.inference.requested_stems
        )

    def _requested_output_stem_indices(self) -> tuple[int, ...]:
        requested = self._requested_output_stem_names()
        runtime_output_stem_names = self.model_params_concrete.output_stem_names
        missing = tuple(name for name in requested if name not in runtime_output_stem_names)
        if missing:
            raise ValueError(
                f"requested stems {missing!r} are unavailable in runtime output stems {runtime_output_stem_names!r}"
            )
        return tuple(runtime_output_stem_names.index(stem_name) for stem_name in requested)

    def _autocast_context(self, device: torch.device) -> contextlib.AbstractContextManager[object]:
        if (is_autocast_available := getattr(torch.amp, "is_autocast_available", None)) is None:
            return contextlib.nullcontext()
        if not is_autocast_available(device.type):
            return contextlib.nullcontext()
        if (
            autocast_dtype := _resolve_runtime_dtype(
                self.config.inference.use_autocast_dtype,
                device=device,
                for_autocast=True,
            )
        ) is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type=device.type, dtype=autocast_dtype)

    def to_audio_tensor(
        self,
        mixture: t.StrPath | t.BytesPath | t.RawAudioTensor | core.Audio[t.RawAudioTensor],
    ) -> core.Audio[t.RawAudioTensor]:
        if isinstance(mixture, core.Audio):
            return mixture
        elif isinstance(mixture, torch.Tensor):
            return core.Audio(
                data=t.RawAudioTensor(mixture),
                sample_rate=self.config.audio_io.target_sample_rate,
            )
        else:
            from .io import read_audio

            device = next(self.model.parameters()).device
            return read_audio(
                mixture,  # type: ignore[arg-type]
                self.config.audio_io.target_sample_rate,
                self.config.audio_io.force_channels,
                device=device,
            )

    def run(
        self,
        mixture: t.StrPath | t.BytesPath | t.RawAudioTensor | core.Audio[t.RawAudioTensor],
    ) -> InferenceOutput:
        for event in self.stream(mixture):
            if isinstance(event, InferenceOutput):
                return event
        raise RuntimeError("inference stream finished without outputs")

    def stream(
        self,
        mixture: t.StrPath | t.BytesPath | t.RawAudioTensor | core.Audio[t.RawAudioTensor],
    ) -> Generator[InferenceEvent, None, None]:
        archetype = self.config.validate_inference_contract(self.model_params_concrete)

        audio_tensor = self.to_audio_tensor(mixture)
        mixture_data: t.RawAudioTensor | t.NormalizedAudioTensor = audio_tensor.data
        mixture_stats: core.NormalizationStats | None = None

        if self.config.normalization.enabled:
            with Stage("normalize") as s:
                yield s.started
                normalized = core.normalize_audio(audio_tensor)
                mixture_data = normalized.audio.data
                mixture_stats = normalized.stats
                yield s.completed

        if archetype == "event_detection":
            requested_stems = self._requested_output_stem_names()
            requested_stem_indices = self._requested_output_stem_indices()
            logits = yield from self._stream_event_detection(
                mixture_data,
                output_indices=requested_stem_indices,
                num_stems=len(requested_stems),
            )
            outputs: dict[str, t.LogitsTensor] = {}
            for i, name in enumerate(requested_stems):
                outputs[name] = t.LogitsTensor(logits[i])
            yield InferenceOutput(outputs=outputs, sample_rate=audio_tensor.sample_rate)
            return

        requested_stems = self._requested_output_stem_names()
        requested_stem_indices = self._requested_output_stem_indices()
        separated_data = yield from self._stream_waveform_pipeline(
            mixture_data,
            archetype,
            output_indices=requested_stem_indices,
            num_stems=len(requested_stems),
        )

        denormalized_stems: dict[t.ModelOutputStemName, t.RawAudioTensor] = {}
        with Stage("collect_outputs") as s:
            yield s.started
            for i, stem_name in enumerate(requested_stems):
                stem_data = separated_data[i, ...]
                if mixture_stats is not None:
                    stem_data = core.denormalize_audio(
                        audio_data=t.NormalizedAudioTensor(stem_data),
                        stats=mixture_stats,
                    )
                denormalized_stems[stem_name] = t.RawAudioTensor(stem_data)
            yield s.completed

        output_stems: dict[StemName, t.RawAudioTensor] = denormalized_stems
        if derived_stems_cfg := self.config.derived_stems:
            with Stage("derive_stems") as s:
                yield s.started
                output_stems = core.derive_stems(
                    denormalized_stems,
                    audio_tensor.data,
                    derived_stems_cfg,
                )
                yield s.completed

        yield InferenceOutput(outputs=output_stems, sample_rate=audio_tensor.sample_rate)

    def _stream_waveform_pipeline(
        self,
        mixture_data: t.RawAudioTensor | t.NormalizedAudioTensor,
        archetype: Literal["standard_end_to_end", "frequency_masking"],
        *,
        output_indices: tuple[int, ...],
        num_stems: int,
    ) -> Generator[InferenceEvent, None, t.RawSeparatedTensor]:
        if (chunk_cfg := self.config.waveform_chunking) is None:
            raise ValueError("missing `waveform_chunking`")

        output_type = self.model_params_concrete.output_type
        if output_type == "logits":
            raise ValueError("waveform pipeline cannot run logits models")

        stft_cfg = self.config.stft if archetype == "frequency_masking" else None
        device = mixture_data.device
        chunk_size = self.config.model.chunk_size
        hop_size = int(chunk_size * (1 - chunk_cfg.overlap_ratio))
        window = core._get_window_fn(chunk_cfg.window_shape, chunk_size, device)

        original_num_samples = mixture_data.shape[-1]
        padding = chunk_size - hop_size
        padded_length = original_num_samples + 2 * padding
        rem = (padded_length - chunk_size) % hop_size
        if rem != 0:
            padded_length += hop_size - rem
        num_chunks = max(0, (padded_length - chunk_size) // hop_size + 1)
        total_batches = math.ceil(num_chunks / self.config.inference.batch_size)

        chunk_generator = core.generate_chunks(
            audio_data=mixture_data,
            chunk_size=chunk_size,
            hop_size=hop_size,
            batch_size=self.config.inference.batch_size,
            padding_mode=chunk_cfg.padding_mode,
        )

        model_w2w = core.create_w2w_model(
            model=self.model,
            model_input_type=self.model_params_concrete.input_type,
            model_output_type=output_type,
            stft_cfg=stft_cfg,
            num_channels=mixture_data.shape[0],
            chunk_size=chunk_size,
            masking_cfg=self.config.masking,
        )

        separated_chunks: list[t.SeparatedChunkedTensor] = []
        selection_index_tensor: torch.Tensor | None = None
        with (
            torch.inference_mode(),
            self._autocast_context(device),
        ):
            batch_idx = 0
            for chunk_batch in chunk_generator:
                separated_batch: torch.Tensor = model_w2w(chunk_batch)
                if (
                    selection_index_tensor is None
                    and len(output_indices) != separated_batch.shape[1]
                ):
                    selection_index_tensor = torch.tensor(
                        output_indices, device=separated_batch.device
                    )
                if selection_index_tensor is not None:
                    separated_batch = separated_batch.index_select(
                        dim=1,
                        index=selection_index_tensor,
                    )
                separated_chunks.append(cast(t.SeparatedChunkedTensor, separated_batch))
                batch_idx += 1
                yield ChunkProcessed(batch_index=batch_idx, total_batches=total_batches)

        with Stage("stitch") as s:
            yield s.started
            stitched = core.stitch_chunks(
                processed_chunks=separated_chunks,
                num_stems=num_stems,
                chunk_size=chunk_size,
                hop_size=hop_size,
                target_num_samples=original_num_samples,
                window=t.WindowTensor(window),
            )
            yield s.completed
        return stitched

    def _stream_event_detection(
        self,
        mixture_data: t.RawAudioTensor | t.NormalizedAudioTensor,
        *,
        output_indices: tuple[int, ...],
        num_stems: int,
    ) -> Generator[InferenceEvent, None, t.LogitsTensor]:
        if (log_mel_cfg := self.config.log_mel) is None:
            raise ValueError("missing `log_mel`")
        if (chunk_cfg := self.config.logmel_chunking) is None:
            raise ValueError("missing `logmel_chunking`")

        device = mixture_data.device
        with Stage("log_mel") as s:
            yield s.started
            mel_preprocessor = core.LogMelSpect(
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

            mixture_mono = (
                mixture_data.mean(dim=0, keepdim=True)
                if mixture_data.shape[0] > 1
                else mixture_data
            )
            with (
                torch.inference_mode(),
                self._autocast_context(device),
            ):
                full_spect = mel_preprocessor(mixture_mono).squeeze(0).squeeze(0).transpose(0, 1)
            yield s.completed

        if (frame_chunk_size := self.config.model.chunk_size // log_mel_cfg.hop_length) <= 0:
            raise ValueError(
                f"invalid frame chunk size computed from {self.config.model.chunk_size=} and {log_mel_cfg.hop_length=}"
            )

        spect_chunks, starts = core.split_spectrogram(
            t.LogMelSpectrogram(full_spect),
            chunk_size=frame_chunk_size,
            trim_margin=chunk_cfg.trim_margin,
            avoid_short_end=chunk_cfg.avoid_short_end,
        )

        total_batches = math.ceil(len(spect_chunks) / self.config.inference.batch_size)
        logits_chunks: list[t.LogitsTensor] = []
        selection_index_tensor: torch.Tensor | None = None
        with (
            torch.inference_mode(),
            self._autocast_context(device),
        ):
            batch_idx = 0
            for i in range(0, len(spect_chunks), self.config.inference.batch_size):
                batch_chunks = spect_chunks[i : i + self.config.inference.batch_size]
                batch_tensors = [torch.as_tensor(chunk) for chunk in batch_chunks]
                model_input = torch.stack(batch_tensors, dim=0).transpose(1, 2)
                logits: torch.Tensor = self.model(model_input)
                logits = logits.permute(1, 0, 2)
                if selection_index_tensor is None and len(output_indices) != logits.shape[1]:
                    selection_index_tensor = torch.tensor(output_indices, device=logits.device)
                if selection_index_tensor is not None:
                    logits = logits.index_select(dim=1, index=selection_index_tensor)
                logits_chunks.append(t.LogitsTensor(logits))
                batch_idx += 1
                yield ChunkProcessed(batch_index=batch_idx, total_batches=total_batches)

        with Stage("aggregate_logits") as s:
            yield s.started
            logits = core.aggregate_logits(
                processed_chunks=logits_chunks,
                starts=starts,
                full_size=full_spect.shape[0],
                chunk_size=frame_chunk_size,
                num_stems=num_stems,
                trim_margin=chunk_cfg.trim_margin,
                overlap_mode=chunk_cfg.overlap_mode,
            )
            yield s.completed
        return logits
