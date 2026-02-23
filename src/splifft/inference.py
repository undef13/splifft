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
    "pesto": ("splifft.models.pesto", "Pesto"),
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


def _default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_device(value: torch.device | str | None, *, field_name: str) -> torch.device:
    if value is None:
        return _default_device()
    try:
        return torch.device(value)
    except Exception as e:
        raise ValueError(f"invalid {field_name} value: {value!r}") from e


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
    outputs: dict[StemName, t.RawAudioTensor] | dict[str, torch.Tensor]
    # `sample_rate` exists because if the user passed in a bare path to an audio file,
    # we need to know the sample rate to write the output files.
    sample_rate: t.SampleRate


InferenceEvent: TypeAlias = Stage.Started | ChunkProcessed | Stage.Completed | InferenceOutput


@dataclass(frozen=True)
class InferenceEngine:
    config: Config
    model: nn.Module
    model_params_concrete: ModelParamsLike
    model_device: torch.device
    io_device: torch.device
    model_input_dtype: torch.dtype | None

    @classmethod
    def from_pretrained(
        cls,
        *,
        config: IntoConfig,
        checkpoint_path: t.StrPath,
        overrides: ConfigOverrides = (),
        model_device: torch.device | str | None = None,
        io_device: torch.device | str | None = None,
        module_name: str | None = None,
        class_name: str | None = None,
        package_name: str | None = None,
    ) -> InferenceEngine:
        from .config import into_config
        from .io import load_weights
        from .models import ModelMetadata

        config = into_config(config, overrides=overrides)

        model_device_resolved = _resolve_device(
            model_device or config.inference.model_device,
            field_name="inference.model_device",
        )
        io_device_resolved = _resolve_device(
            io_device or config.inference.io_device,
            field_name="inference.io_device",
        )
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
            model = model.to(_resolve_runtime_dtype(forced_dtype, device=model_device_resolved))
        model = load_weights(
            model,
            checkpoint_path,
            device=model_device_resolved,
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

        return cls(
            config=config,
            model=model,
            model_params_concrete=model_params,
            model_device=model_device_resolved,
            io_device=io_device_resolved,
            model_input_dtype=core.get_model_floating_dtype(model),
        )

    @classmethod
    def from_registry(
        cls,
        model_id: str,
        *,
        model_device: torch.device | str | None = None,
        io_device: torch.device | str | None = None,
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
            model_device=model_device,
            io_device=io_device,
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

    def _adapt_input_channels(
        self,
        mixture_data: t.RawAudioTensor | t.NormalizedAudioTensor,
    ) -> Generator[InferenceEvent, None, t.RawAudioTensor | t.NormalizedAudioTensor]:
        required = self.model_params_concrete.input_channels
        num_channels = mixture_data.shape[0]

        if required == "mono":
            with Stage("downmix_to_mono") as s:
                yield s.started
                if num_channels != 1:
                    downmixed = mixture_data.mean(dim=0, keepdim=True)
                    if self.config.normalization.enabled:
                        mixture_data = t.NormalizedAudioTensor(downmixed)
                    else:
                        mixture_data = t.RawAudioTensor(downmixed)
                yield s.completed
            return mixture_data

        if required == "stereo":
            if num_channels != 2:
                raise ValueError(
                    f"model expects stereo input (2 channels), but received {num_channels} channel(s)"
                )
            return mixture_data

        raise ValueError(f"unsupported model input channel contract: {required!r}")

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

            return read_audio(
                mixture,  # type: ignore[arg-type]
                self.config.audio_io.target_sample_rate,
                self.config.audio_io.force_channels,
                device=self.io_device,
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
        raw_mixture_data = t.RawAudioTensor(audio_tensor.data.to(self.io_device))
        mixture_data: t.RawAudioTensor | t.NormalizedAudioTensor = raw_mixture_data
        mixture_stats: core.NormalizationStats | None = None

        if self.config.normalization.enabled:
            with Stage("normalize") as s:
                yield s.started
                normalized = core.normalize_audio(
                    core.Audio(data=raw_mixture_data, sample_rate=audio_tensor.sample_rate)
                )
                mixture_data = normalized.audio.data
                mixture_stats = normalized.stats
                yield s.completed

        mixture_data = yield from self._adapt_input_channels(mixture_data)

        if archetype == "sequence_labeling":
            requested_stems = self._requested_output_stem_names()
            requested_stem_indices = self._requested_output_stem_indices()
            sequence_outputs = yield from self._stream_sequence_labeling(
                mixture_data,
                requested_stems=requested_stems,
                output_indices=requested_stem_indices,
            )
            yield InferenceOutput(outputs=sequence_outputs, sample_rate=audio_tensor.sample_rate)
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
                    raw_mixture_data,
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
        if output_type in {"logits", "multi_stream"}:
            raise ValueError("waveform pipeline cannot run sequence models")

        stft_cfg = (
            self.config.transform
            if archetype == "frequency_masking"
            and self.config.transform is not None
            and self.config.transform.kind == "stft"
            else None
        )
        io_device = mixture_data.device
        chunk_size = self.config.model.chunk_size
        hop_size = int(chunk_size * (1 - chunk_cfg.overlap_ratio))
        window = core._get_window_fn(chunk_cfg.window_shape, chunk_size, io_device)

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
            io_device=self.io_device,
            model_device=self.model_device,
        )

        separated_chunks: list[t.SeparatedChunkedTensor] = []
        selection_index_tensor: torch.Tensor | None = None
        with (
            torch.inference_mode(),
            self._autocast_context(self.model_device),
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

    def _prepare_sequence_model_input(
        self,
        batch_sequence: torch.Tensor,
    ) -> torch.Tensor:
        if self.model_params_concrete.input_type == "waveform":
            return batch_sequence.squeeze(-1)
        return batch_sequence

    def _stream_sequence_labeling(
        self,
        mixture_data: t.RawAudioTensor | t.NormalizedAudioTensor,
        *,
        requested_stems: tuple[t.ModelOutputStemName, ...],
        output_indices: tuple[int, ...],
    ) -> Generator[InferenceEvent, None, dict[str, torch.Tensor]]:
        if (chunk_cfg := self.config.sequence_chunking) is None:
            raise ValueError("missing `sequence_chunking`")

        io_device = mixture_data.device
        sequence_input = mixture_data.unsqueeze(0)

        sequence_feature_extractor = core.create_sequence_feature_extractor(
            self.config.transform,
            sample_rate=self.config.audio_io.target_sample_rate,
            device=io_device,
        )

        with Stage(sequence_feature_extractor.stage_name) as s:
            yield s.started
            with (
                torch.inference_mode(),
                self._autocast_context(self.model_device),
            ):
                full_sequence = sequence_feature_extractor(sequence_input).squeeze(0)
            yield s.completed
        hop_length_samples = sequence_feature_extractor.hop_length_samples

        if (frame_chunk_size := self.config.model.chunk_size // hop_length_samples) <= 0:
            raise ValueError(
                "invalid frame chunk size computed from "
                f"{self.config.model.chunk_size=} and {hop_length_samples=}"
            )

        sequence_chunks, starts = core.split_sequence_tensor(
            full_sequence,
            chunk_size=frame_chunk_size,
            trim_margin=chunk_cfg.trim_margin,
            avoid_short_end=chunk_cfg.avoid_short_end,
        )

        total_batches = math.ceil(len(sequence_chunks) / self.config.inference.batch_size)
        stream_chunks: dict[str, list[torch.Tensor]] = {}
        selection_index_tensor: torch.Tensor | None = None
        with (
            torch.inference_mode(),
            self._autocast_context(self.model_device),
        ):
            batch_idx = 0
            for i in range(0, len(sequence_chunks), self.config.inference.batch_size):
                batch_chunks = sequence_chunks[i : i + self.config.inference.batch_size]
                batch_tensors = [torch.as_tensor(chunk) for chunk in batch_chunks]
                sequence_batch = torch.stack(batch_tensors, dim=0)
                model_input = self._prepare_sequence_model_input(sequence_batch)
                model_input = core.to_model_device(
                    model_input,
                    model_device=self.model_device,
                    model_floating_dtype=self.model_input_dtype,
                )
                model_output = self.model(model_input)

                if isinstance(model_output, dict):
                    for stream_name, stream_batch in model_output.items():
                        stream_batch = stream_batch.to(io_device)
                        for stream_chunk in stream_batch.unbind(dim=0):
                            stream_chunks.setdefault(stream_name, []).append(stream_chunk)
                else:
                    logits = model_output.to(io_device).permute(1, 0, 2)
                    if selection_index_tensor is None and len(output_indices) != logits.shape[1]:
                        selection_index_tensor = torch.tensor(output_indices, device=logits.device)
                    if selection_index_tensor is not None:
                        logits = logits.index_select(dim=1, index=selection_index_tensor)
                    for stream_idx, stem_name in enumerate(requested_stems):
                        stream_batch = logits[:, stream_idx, :]
                        for stream_chunk in stream_batch.unbind(dim=0):
                            stream_chunks.setdefault(stem_name, []).append(stream_chunk)

                batch_idx += 1
                yield ChunkProcessed(batch_index=batch_idx, total_batches=total_batches)

        with Stage("aggregate_sequence_outputs") as s:
            yield s.started
            aggregated: dict[str, torch.Tensor] = {}
            for stream_name in requested_stems:
                if stream_name not in stream_chunks:
                    raise ValueError(
                        f"requested stream {stream_name!r} not produced by model output keys {tuple(stream_chunks)}"
                    )
                aggregated[stream_name] = core.aggregate_sequence_chunks(
                    processed_chunks=stream_chunks[stream_name],
                    starts=starts,
                    full_size=full_sequence.shape[0],
                    chunk_size=frame_chunk_size,
                    trim_margin=chunk_cfg.trim_margin,
                    overlap_mode=chunk_cfg.overlap_mode,
                )
            yield s.completed
        return aggregated
