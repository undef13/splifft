"""Public inference APIs."""
# put heavy logic inside core.py, this file should just wire up the components together.

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generator, Literal, NamedTuple, TypeAlias, cast

import torch
from torch import nn
from typing_extensions import assert_never

from . import PATH_REGISTRY_DEFAULT, core
from . import types as t
from .models import HasSequenceWindowFrames

if TYPE_CHECKING:
    from pathlib import Path

    from .config import Config, ConfigOverrides, IntoConfig, StemName
    from .models import ModelParamsLike


class ModelEntrypoint(NamedTuple):
    module_name: str
    class_name: str


SUPPORTED_MODELS: dict[str, ModelEntrypoint] = {
    "bs_roformer": ModelEntrypoint("splifft.models.bs_roformer", "BSRoformer"),
    "mel_roformer": ModelEntrypoint("splifft.models.bs_roformer", "BSRoformer"),
    "mdx23c": ModelEntrypoint("splifft.models.mdx23c", "MDX23C"),
    "beat_this": ModelEntrypoint("splifft.models.beat_this", "BeatThis"),
    "pesto": ModelEntrypoint("splifft.models.pesto", "Pesto"),
    "basic_pitch": ModelEntrypoint("splifft.models.basic_pitch", "BasicPitch"),
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
) -> ModelEntrypoint:
    if module_name is not None and class_name is not None:
        return ModelEntrypoint(module_name, class_name)
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
class SeparatedBlock:
    """A finalized, denormalized span of separated audio, emitted in temporal order.

    Shaped `(num_requested_stems, channels, samples)` on the io device. Consecutive
    blocks concatenate to the full separated track.
    """

    data: torch.Tensor


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


InferenceEvent: TypeAlias = (
    Stage.Started | ChunkProcessed | SeparatedBlock | Stage.Completed | InferenceOutput
)


class _PreparedMixture(NamedTuple):
    audio: core.Audio[t.RawAudioTensor]
    raw_data: t.RawAudioTensor
    model_input: t.RawAudioTensor | t.NormalizedAudioTensor
    stats: core.NormalizationStats | None


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
        resolved_entrypoint = resolve_model_entrypoint(config.model_type, module_name, class_name)
        metadata = ModelMetadata.from_module(
            module_name=resolved_entrypoint.module_name,
            model_cls_name=resolved_entrypoint.class_name,
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

        assert_never(required)

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

    def _prepare_mixture(
        self,
        mixture: t.StrPath | t.BytesPath | t.RawAudioTensor | core.Audio[t.RawAudioTensor],
    ) -> Generator[InferenceEvent, None, _PreparedMixture]:
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
        return _PreparedMixture(audio_tensor, raw_mixture_data, mixture_data, mixture_stats)

    def stream_blocks(
        self,
        mixture: t.StrPath | t.BytesPath | t.RawAudioTensor | core.Audio[t.RawAudioTensor],
    ) -> Generator[InferenceEvent, None, None]:
        """Stream finalized [`SeparatedBlock`][splifft.inference.SeparatedBlock]s in
        temporal order without materializing the full separated tensor.

        Intended for streaming encoders; waveform pipelines only.
        """
        archetype = self.config.validate_inference_contract(self.model_params_concrete)
        if archetype == "sequence_labeling":
            raise ValueError("block streaming requires a waveform pipeline")
        if self.config.derived_stems:
            raise ValueError("block streaming does not support derived stems")

        prepared = yield from self._prepare_mixture(mixture)
        yield from self._stream_separated_blocks(
            prepared.model_input,
            archetype,
            output_indices=self._requested_output_stem_indices(),
            stats=prepared.stats,
        )

    def stream(
        self,
        mixture: t.StrPath | t.BytesPath | t.RawAudioTensor | core.Audio[t.RawAudioTensor],
    ) -> Generator[InferenceEvent, None, None]:
        archetype = self.config.validate_inference_contract(self.model_params_concrete)
        prepared = yield from self._prepare_mixture(mixture)

        if archetype == "sequence_labeling":
            requested_stems = self._requested_output_stem_names()
            sequence_outputs = yield from self._stream_sequence_labeling(
                prepared.model_input,
                requested_stems=requested_stems,
                output_indices=self._requested_output_stem_indices(),
            )
            yield InferenceOutput(outputs=sequence_outputs, sample_rate=prepared.audio.sample_rate)
            return

        # NOTE: this single destination tensor requires contiguous allocation (e.g. ~4.4GB for a
        # 4 min track with 53 stems in fp32). in the future, we should stream and flush these blocks
        # directly to disk, but implementing streamed encoders and decoders will come with extreme
        # complexity. `torchcodec` streamed encoding is only available in v0.13 (2026-05-21),
        # requiring torch>=2.11 so we defer it now until we understand more about ffmpeg.
        requested_stems = self._requested_output_stem_names()
        original_num_samples = prepared.raw_data.shape[-1]
        stitched: torch.Tensor | None = None
        stitched_offset = 0
        for event in self._stream_separated_blocks(
            prepared.model_input,
            archetype,
            output_indices=self._requested_output_stem_indices(),
            stats=prepared.stats,
        ):
            if not isinstance(event, SeparatedBlock):
                yield event
                continue
            block = event.data
            if stitched is None:
                stitched = torch.empty(
                    (block.shape[0], block.shape[1], original_num_samples),
                    device=self.io_device,
                    dtype=block.dtype,
                )
            end = stitched_offset + int(block.shape[-1])
            stitched[..., stitched_offset:end].copy_(block)
            stitched_offset = end

        if stitched is None:
            raise RuntimeError("waveform inference produced no stitched output")
        assert stitched_offset == original_num_samples

        output_stems: dict[StemName, t.RawAudioTensor] = {
            stem_name: t.RawAudioTensor(stitched[i, ...])
            for i, stem_name in enumerate(requested_stems)
        }
        if derived_stems_cfg := self.config.derived_stems:
            with Stage("derive_stems") as s:
                yield s.started
                output_stems = core.derive_stems(
                    output_stems,
                    prepared.raw_data,
                    derived_stems_cfg,
                )
                yield s.completed

        yield InferenceOutput(outputs=output_stems, sample_rate=prepared.audio.sample_rate)

    def _stream_separated_blocks(
        self,
        mixture_data: t.RawAudioTensor | t.NormalizedAudioTensor,
        archetype: Literal["standard_end_to_end", "frequency_masking"],
        *,
        output_indices: tuple[int, ...],
        stats: core.NormalizationStats | None,
    ) -> Generator[InferenceEvent, None, None]:
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

        selection_index_tensor = (
            torch.tensor(output_indices, device=self.io_device)
            if output_indices != tuple(range(len(self.model_params_concrete.output_stem_names)))
            else None
        )
        stitcher = core.StreamingStitcher(
            chunk_size=t.ChunkSize(chunk_size),
            hop_size=t.HopSize(hop_size),
            target_num_samples=t.Samples(original_num_samples),
            window=t.WindowTensor(window),
        )

        def _finalize(block: torch.Tensor) -> SeparatedBlock:
            if stats is not None:
                block = core.denormalize_audio(t.NormalizedAudioTensor(block), stats)
            return SeparatedBlock(data=block)

        with (
            torch.inference_mode(),
            self._autocast_context(self.model_device),
        ):
            batch_idx = 0
            for chunk_batch in chunk_generator:
                separated_batch: torch.Tensor = model_w2w(chunk_batch)
                if selection_index_tensor is not None:
                    separated_batch = separated_batch.index_select(
                        dim=1,
                        index=selection_index_tensor,
                    )
                for chunk in separated_batch.unbind(0):
                    if (block := stitcher.step(chunk)) is not None:
                        yield _finalize(block)
                batch_idx += 1
                yield ChunkProcessed(batch_index=batch_idx, total_batches=total_batches)

        if (tail := stitcher.flush()) is not None:
            yield _finalize(tail)

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

        if not isinstance(self.model_params_concrete, HasSequenceWindowFrames):
            raise TypeError(
                "spectrogram sequence-labeling models must implement HasSequenceWindowFrames"
            )

        frame_chunk_size = int(self.model_params_concrete.sequence_window_frames)
        split_sequence = core.split_sequence_tensor(
            full_sequence,
            chunk_size=frame_chunk_size,
            trim_margin=chunk_cfg.trim_margin,
            avoid_short_end=chunk_cfg.avoid_short_end,
        )

        total_batches = math.ceil(len(split_sequence.chunks) / self.config.inference.batch_size)
        stream_chunks: dict[str, list[torch.Tensor]] = {}
        selection_index_tensor = (
            torch.tensor(output_indices, device=io_device)
            if output_indices != tuple(range(len(self.model_params_concrete.output_stem_names)))
            else None
        )

        with (
            torch.inference_mode(),
            self._autocast_context(self.model_device),
        ):
            batch_idx = 0
            for i in range(0, len(split_sequence.chunks), self.config.inference.batch_size):
                batch_chunks = split_sequence.chunks[i : i + self.config.inference.batch_size]
                batch_tensors = [torch.as_tensor(chunk) for chunk in batch_chunks]
                sequence_batch = torch.stack(batch_tensors, dim=0)
                model_input = self._prepare_sequence_model_input(sequence_batch)
                model_input = core.to_model_device(
                    model_input,
                    model_device=self.model_device,
                    model_floating_dtype=self.model_input_dtype,
                )
                model_output = self.model(model_input)

                if isinstance(model_output, tuple):
                    for stem_name, output_index in zip(
                        requested_stems, output_indices, strict=True
                    ):
                        stream_batch = model_output[output_index].to(io_device)
                        for stream_chunk in stream_batch.unbind(dim=0):
                            stream_chunks.setdefault(stem_name, []).append(stream_chunk)
                elif isinstance(model_output, torch.Tensor):
                    logits = model_output.to(io_device).permute(1, 0, 2)
                    if selection_index_tensor is not None:
                        logits = logits.index_select(dim=1, index=selection_index_tensor)
                    for stream_idx, stem_name in enumerate(requested_stems):
                        stream_batch = logits[:, stream_idx, :]
                        for stream_chunk in stream_batch.unbind(dim=0):
                            stream_chunks.setdefault(stem_name, []).append(stream_chunk)
                else:
                    raise TypeError(
                        "sequence-labeling models must return either Tensor logits or "
                        "an tuple of stream tensors ordered according to output stem names"
                    )

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
                    starts=split_sequence.starts,
                    full_size=full_sequence.shape[0],
                    chunk_size=frame_chunk_size,
                    trim_margin=chunk_cfg.trim_margin,
                    overlap_mode=chunk_cfg.overlap_mode,
                )
            yield s.completed
        return aggregated
