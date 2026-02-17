"""Configuration"""

from __future__ import annotations

import copy
import json
from typing import (
    Annotated,
    Any,
    Hashable,
    Literal,
    Sequence,
    TypeAlias,
    TypeVar,
    Union,
    get_args,
)

# NOTE: we are not using typing.TYPE_CHECKING because pydantic relies on that
import torch
from annotated_types import Len
from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Discriminator,
    Field,
    GetCoreSchemaHandler,
    GetPydanticSchema,
    StringConstraints,
    TypeAdapter,
    model_validator,
)
from pydantic_core import CoreSchema, PydanticCustomError, core_schema
from typing_extensions import Self

from . import types as t
from .core import str_to_torch_dtype
from .models import ModelParamsLike, ModelParamsLikeT


def _get_torch_dtype_schema(_source_type: Any, _handler: GetCoreSchemaHandler) -> CoreSchema:
    return core_schema.json_or_python_schema(
        json_schema=core_schema.chain_schema(
            [
                core_schema.str_schema(),
                core_schema.no_info_plain_validator_function(str_to_torch_dtype),
            ]
        ),
        python_schema=core_schema.union_schema(
            [
                core_schema.is_instance_schema(torch.dtype),
                core_schema.no_info_plain_validator_function(str_to_torch_dtype),
            ]
        ),
        serialization=core_schema.plain_serializer_function_ser_schema(
            lambda dtype: str(dtype).split(".")[-1]
        ),
    )


TorchDtype: TypeAlias = Annotated[torch.dtype, GetPydanticSchema(_get_torch_dtype_schema)]

_PYDANTIC_STRICT_CONFIG = ConfigDict(strict=True, extra="forbid")

_Item = TypeVar("_Item", bound=Hashable)


def _to_tuple(sequence: Sequence[_Item]) -> tuple[_Item, ...]:
    # this is so json arrays are converted to tuples
    return tuple(sequence)


Tuple = Annotated[tuple[_Item, ...], BeforeValidator(_to_tuple)]


def _validate_unique_sequence(sequence: Sequence[_Item]) -> Sequence[_Item]:
    # e.g. to ensure there are no duplicate stem names
    if len(sequence) != len(set(sequence)):
        raise PydanticCustomError("unique_sequence", "Sequence must contain unique items")
    return sequence


_S = TypeVar("_S")
NonEmptyUnique = Annotated[
    _S,
    Len(min_length=1),
    AfterValidator(_validate_unique_sequence),
    Field(json_schema_extra={"unique_items": True}),
]

ModelInputStemName: TypeAlias = Literal["mixture"]
ModelOutputStemName: TypeAlias = Annotated[t.ModelOutputStemName, StringConstraints(min_length=1)]
_INPUT_STEM_NAMES = get_args(ModelInputStemName)


# NOTE: the ideal case would be to use an ADT whose variants are known at "compile" time, e.g. in Rust:
# enum ModelConfig {
#     BsRoformer { param_x: ..., param_y: ... },
#     Demucs { param_x: ..., params_z: ... },
# }
# but downstream users may want to register their own models with different configurations,
# so a discriminated enum wouldn't work here.
# so, we effectively let Config.model_config be dyn ModelParamsLike (i.e. dict[str, Any])
# and defer the validation of the model configuration until it is actually needed instead of doing it eagerly.
class LazyModelConfig(BaseModel):
    """A lazily validated model configuration.

    Note that it is not guaranteed to be fully valid until `to_concrete` is called.
    """

    chunk_size: t.ChunkSize
    output_stem_names: NonEmptyUnique[Tuple[ModelOutputStemName]]
    # NOTE: inference.requested_stems takes precedence if it is configured.

    def to_concrete(
        self,
        model_params: type[ModelParamsLikeT],
        *,
        pydantic_config: ConfigDict = ConfigDict(extra="forbid"),
    ) -> ModelParamsLikeT:
        """Validate against a real set of model parameters and convert to it.

        :raises pydantic.ValidationError: if extra fields are present in the model parameters
            that doesn't exist in the concrete model parameters.
        """
        # input_type and output_type are inconfigurable anyway
        # TODO: use lru cache to avoid recreating the TypeAdapter in a hot loop but dict isn't hashable
        ta = TypeAdapter(
            type(
                f"{model_params.__name__}Validator",
                (model_params,),
                {"__pydantic_config__": pydantic_config},
            )  # needed for https://docs.pydantic.dev/latest/errors/usage_errors/#type-adapter-config-unused
        )  # type: ignore
        # types defined within `TYPE_CHECKING` blocks will be forward references, so we need rebuild
        ta.rebuild(_types_namespace={"TorchDtype": TorchDtype, "t": t})
        model_params_concrete: ModelParamsLikeT = ta.validate_python(self.model_dump())
        return model_params_concrete

    @property
    def stem_names(self) -> tuple[ModelInputStemName | ModelOutputStemName, ...]:
        """Returns the model's input and output stem names."""
        return (*_INPUT_STEM_NAMES, *self.output_stem_names)

    model_config = ConfigDict(
        strict=True, extra="allow"
    )  # extra fields are not validated until `to_concrete`


class StftConfig(BaseModel):
    """configuration for the short-time fourier transform."""

    n_fft: t.FftSize = 2048
    hop_length: t.HopSize = 512
    win_length: t.FftSize = 2048
    window_shape: t.WindowShape = "hann"
    normalized: bool = False
    conv_dtype: TorchDtype | None = None
    """The data type used for the `conv1d` buffers."""

    model_config = _PYDANTIC_STRICT_CONFIG


class LogMelConfig(BaseModel):
    """Configuration for Log-Mel Spectrogram."""

    n_fft: t.FftSize
    hop_length: t.HopSize
    n_mels: t.Gt0[int]
    sample_rate: t.SampleRate
    f_min: float = 0.0
    f_max: float | None = None
    mel_scale: Literal["htk", "slaney"] = "slaney"
    normalized: bool | Literal["frame_length"] = "frame_length"
    power: float = 1.0
    log_multiplier: float = 1000.0

    model_config = _PYDANTIC_STRICT_CONFIG


class AudioIOConfig(BaseModel):
    target_sample_rate: t.SampleRate = 44100
    force_channels: t.Channels | None = 2
    """Whether to force mono or stereo audio input. If None, keep original."""

    model_config = _PYDANTIC_STRICT_CONFIG


class TorchCompileConfig(BaseModel):
    fullgraph: bool = True
    dynamic: bool = True
    mode: Literal["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"] = (
        "reduce-overhead"
    )


class InferenceConfig(BaseModel):
    batch_size: t.BatchSize = 8
    requested_stems: NonEmptyUnique[Tuple[ModelOutputStemName]] | None = None
    """Optional subset of model output stems to compute and emit.

    When provided, inference only computes/keeps these model outputs in the
    configured order. Depending on the model architecture, this may allow
    specific stem-specific weights to be not loaded at all.

    For example, the BS Roformer architecture has a shared backbone followed by
    multiple MLP heads, so setting this parameter can effectively patch out the
    unrelated heads.
    """
    force_weights_dtype: TorchDtype | None = None
    use_autocast_dtype: TorchDtype | None = None
    compile_model: TorchCompileConfig | None = None

    model_config = _PYDANTIC_STRICT_CONFIG


class NormalizationConfig(BaseModel):
    enabled: bool = False

    model_config = _PYDANTIC_STRICT_CONFIG


class WaveformChunkingConfig(BaseModel):
    method: Literal["overlap_add_windowed"] = "overlap_add_windowed"
    overlap_ratio: t.OverlapRatio = 0.5
    window_shape: t.WindowShape = "hann"
    padding_mode: t.PaddingMode = "reflect"

    model_config = _PYDANTIC_STRICT_CONFIG


class LogMelChunkingConfig(BaseModel):
    trim_margin: t.TrimMargin = 0
    overlap_mode: t.OverlapMode = "keep_first"
    avoid_short_end: bool = True

    model_config = _PYDANTIC_STRICT_CONFIG


class MaskingConfig(BaseModel):
    add_sub_dtype: TorchDtype | None = None
    out_dtype: TorchDtype | None = None

    model_config = _PYDANTIC_STRICT_CONFIG


DerivedStemName: TypeAlias = Annotated[str, StringConstraints(min_length=1)]
"""The name of a derived stem, e.g. `vocals_minus_drums`."""
StemName: TypeAlias = Union[ModelOutputStemName, DerivedStemName]
"""A name of a stem, either a model output stem or a derived stem."""


class SubtractConfig(BaseModel):
    operation: Literal["subtract"]
    stem_name: StemName
    by_stem_name: StemName

    model_config = _PYDANTIC_STRICT_CONFIG


class SumConfig(BaseModel):
    operation: Literal["sum"]
    stem_names: NonEmptyUnique[Tuple[StemName]]

    model_config = _PYDANTIC_STRICT_CONFIG


DerivedStemRule: TypeAlias = Annotated[Union[SubtractConfig, SumConfig], Discriminator("operation")]
DerivedStemsConfig: TypeAlias = dict[DerivedStemName, DerivedStemRule]


class OutputConfig(BaseModel):
    stem_names: Literal["all"] | NonEmptyUnique[Tuple[StemName]] = "all"
    file_format: t.FileFormat = "flac"
    bit_rate: t.BitRate | None = None
    """Output bit rate for lossy formats. The default is chosen by FFmpeg."""

    model_config = _PYDANTIC_STRICT_CONFIG


class ConfigOverrideError(ValueError):
    """Raised when one or more override *strings* are syntactically invalid."""


def parse_override_value(value: str) -> Any:
    """Parse a CLI override value into a Python object.

    Shell-like quoting/escaping semantics beyond what your shell and JSON provide
    are not supported.
    """
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def parse_config_override(override: str) -> tuple[tuple[str, ...], Any]:
    """Parse one override entry of the form `<dot.path>=<value>`. Empty value
    is interpreted as an empty string.

    Missing `=` or empty path segments like `a..b=1` or `.a=1` are considered syntax errors
    """
    key, sep, raw_value = override.partition("=")
    if sep == "":
        raise ConfigOverrideError(
            "invalid override `"
            f"{override}`: expected `<dot.path>=<value>`, e.g. `inference.batch_size=2`"
        )

    path = tuple(part.strip() for part in key.split("."))
    if not path or any(part == "" for part in path):
        raise ConfigOverrideError(
            f"invalid override path `{key}` in `{override}`: empty path segment is not allowed"
        )

    return path, parse_override_value(raw_value)


def set_path_value(mut_config_dict: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    """Set a nested value in-place, creating missing dictionaries as needed."""
    current = mut_config_dict
    for key in path[:-1]:
        next_node = current.get(key)
        if not isinstance(next_node, dict):
            next_node = {}
            current[key] = next_node
        current = next_node

    current[path[-1]] = value


ConfigOverrides: TypeAlias = Sequence[str]
"""`<dot.path>=<value>` form, e.g. `inference.batch_size=2`

- automatic creation of missing nested dictionaries while applying overrides
- validation is performed *after* all overrides are applied

Not supported: list index addressing in paths (e.g. `a.0.b=1` is treated as string keys)
"""


def apply_config_overrides(
    mut_config_dict: dict[str, Any], overrides: ConfigOverrides
) -> dict[str, Any]:
    for override in overrides:
        path, value = parse_config_override(override)
        set_path_value(mut_config_dict, path, value)
    return mut_config_dict


def load_config_dict(path: t.StrPath | t.BytesPath) -> dict[str, Any]:
    with open(path, "rb") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"expected top-level JSON object in config file, got {type(data).__name__}")
    return data


# if we were to implement a model registry (which we shouldn't need)
# heavily consider https://peps.python.org/pep-0487/#subclass-registration


class Config(BaseModel):
    identifier: str
    """Unique identifier for this configuration"""
    model_type: t.ModelType
    model: LazyModelConfig
    stft: StftConfig | None = None
    log_mel: LogMelConfig | None = None
    audio_io: AudioIOConfig = Field(default_factory=AudioIOConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)
    waveform_chunking: WaveformChunkingConfig | None = None
    logmel_chunking: LogMelChunkingConfig | None = None
    masking: MaskingConfig = Field(default_factory=MaskingConfig)
    derived_stems: DerivedStemsConfig | None = None
    output: OutputConfig = Field(default_factory=OutputConfig)
    experimental: dict[str, Any] | None = None
    """Any extra experimental configurations outside of the `splifft` core."""

    # NOTE: stft config can be none for models that operate on raw waveforms
    # they are checked in the processing step instead.
    @model_validator(mode="after")
    def check_derived_stems(self) -> Self:
        if self.derived_stems is None:
            return self

        base_output_stems = tuple(self.inference.requested_stems or self.model.output_stem_names)
        existing_stem_names: list[StemName] = [*_INPUT_STEM_NAMES, *base_output_stems]
        for derived_stem_name, definition in self.derived_stems.items():
            if derived_stem_name in existing_stem_names:
                raise PydanticCustomError(
                    "derived_stem_name_conflict",
                    "Derived stem `{derived_stem_name}` must not conflict with existing stem names: `{existing_stem_names}`",
                    {
                        "derived_stem_name": derived_stem_name,
                        "existing_stem_names": existing_stem_names,
                    },
                )
            required_stems: tuple[StemName, ...] = tuple()
            if isinstance(definition, SubtractConfig):
                required_stems = (definition.stem_name, definition.by_stem_name)
            elif isinstance(definition, SumConfig):
                required_stems = definition.stem_names
            for stem_name in required_stems:
                if stem_name not in existing_stem_names:
                    raise PydanticCustomError(
                        "invalid_derived_stem",
                        "Derived stem `{derived_stem_name}` requires stem `{stem_name}` but is not found in `{existing_stem_names}`",
                        {
                            "derived_stem_name": derived_stem_name,
                            "stem_name": stem_name,
                            "existing_stem_names": existing_stem_names,
                        },
                    )
            existing_stem_names.append(derived_stem_name)
        return self

    @model_validator(mode="after")
    def check_requested_stems(self) -> Self:
        if self.inference.requested_stems is None:
            return self

        model_stem_names = set(self.model.output_stem_names)
        for stem_name in self.inference.requested_stems:
            if stem_name in model_stem_names:
                continue
            raise PydanticCustomError(
                "invalid_target_stem",
                "Target stem `{stem_name}` is not found in model output stems: `{model_stem_names}`",
                {
                    "stem_name": stem_name,
                    "model_stem_names": self.model.output_stem_names,
                },
            )

        return self

    def validate_inference_contract(self, model_params: ModelParamsLike) -> t.InferenceArchetype:
        archetype = model_params.inference_archetype
        input_type = model_params.input_type
        output_type = model_params.output_type

        if archetype == "standard_end_to_end":
            if self.waveform_chunking is None:
                raise ValueError("`waveform_chunking` is required for standard_end_to_end pipeline")
            if output_type != "waveform":
                raise ValueError(
                    f"standard_end_to_end expects waveform model output, got {output_type}"
                )
        elif archetype == "frequency_masking":
            if self.stft is None:
                raise ValueError("`stft` is required for frequency_masking pipeline")
            if self.waveform_chunking is None:
                raise ValueError("`waveform_chunking` is required for frequency_masking pipeline")
            if input_type not in {"spectrogram", "waveform_and_spectrogram"}:
                raise ValueError(
                    f"frequency_masking expects spectrogram-like model input, got {input_type}"
                )
            if output_type not in {"spectrogram_mask", "spectrogram"}:
                raise ValueError(
                    f"frequency_masking expects spectrogram-like model output, got {output_type}"
                )
        elif archetype == "event_detection":
            if self.log_mel is None:
                raise ValueError("`log_mel` is required for event_detection pipeline")
            if self.logmel_chunking is None:
                raise ValueError("`logmel_chunking` is required for event_detection pipeline")
            if input_type != "spectrogram":
                raise ValueError(
                    f"event_detection expects spectrogram model input, got {input_type}"
                )
            if output_type != "logits":
                raise ValueError(f"event_detection expects logits model output, got {output_type}")
        else:
            raise ValueError(f"unknown inference archetype: {archetype}")

        if output_type == "logits" and self.output.file_format != "npy":
            raise ValueError(
                f"logits models require output.file_format='npy', got {self.output.file_format!r}"
            )
        if output_type != "logits" and self.output.file_format == "npy":
            raise ValueError("waveform/spectrogram models cannot use output.file_format='npy'")

        return archetype

    @classmethod
    def from_file(
        cls,
        path: t.StrPath | t.BytesPath,
        *,
        overrides: ConfigOverrides = (),
    ) -> Config:
        """Load config JSON from disk, optionally applying CLI-style overrides."""
        config_dict = load_config_dict(path)
        if overrides:
            apply_config_overrides(config_dict, overrides)
        return cls.model_validate(config_dict)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,  # for .model
        strict=True,
        extra="forbid",
    )


IntoConfig: TypeAlias = Config | dict[str, Any] | t.StrPath | t.BytesPath


def into_config(config: IntoConfig, *, overrides: ConfigOverrides = ()) -> Config:
    """Convert various config inputs into a validated [`Config`].

    If `overrides` is non-empty, they are applied before validation.
    Caller-owned objects are not mutated.
    """
    if isinstance(config, Config):
        if not overrides:
            return config
        config_dict = config.model_dump(mode="python")
        apply_config_overrides(config_dict, overrides)
        return Config.model_validate(config_dict)

    if isinstance(config, dict):
        config_dict = copy.deepcopy(config)
        if overrides:
            apply_config_overrides(config_dict, overrides)
        return Config.model_validate(config_dict)

    return Config.from_file(config, overrides=overrides)


#
# registry
#


class Model(BaseModel):
    authors: list[str]
    purpose: (
        Literal[
            "separation",
            "denoise",
            "debleed",
            "dereverb",
            "decrowd",
            "beat_tracking",
        ]
        | str
    )
    architecture: Literal["bs_roformer", "mel_roformer", "mdx23c", "scnet", "beat_this"] | str
    config_id: str | None = None
    """The default configuration identifier (filename stem) to use if one is not provided.
    Files are expected to be in `data/config`.

    If None, the model is not officially supported.
    """
    created_at: str | None = None
    """ISO8601 date, time is optional (e.g. YYYY-MM-DD)"""
    output: NonEmptyUnique[list[t.Instrument]] = Field(default_factory=list)
    status: Literal["tested"] | None = None
    metrics: list[Metrics] = Field(default_factory=list)
    description: list[Comment] = Field(default_factory=list)
    resources: list[Resource] = Field(default_factory=list)
    model_size: int | None = None
    """Model size in bytes, if available."""


class Metrics(BaseModel):
    values: dict[t.Instrument, dict[t.Metric, float]] = Field(default_factory=dict)
    source: Literal["mvsep"] | str | None = None  #  e.g. mvsep quality checker link


class Resource(BaseModel):
    kind: Literal[
        "model_ckpt",
        "config_msst",
        "arxiv",
        "other",
    ]
    url: str
    digest: str | None = None


class Comment(BaseModel):
    content: list[str]
    """Condensed informative points of the model (lowercase)"""
    author: str | None = None


class Registry(dict[t.Identifier, Model]):
    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return core_schema.no_info_after_validator_function(cls, handler(dict[t.Identifier, Model]))

    @classmethod
    def from_file(cls, path: t.StrPath | t.BytesPath) -> Registry:
        with open(path, "r") as f:
            data = f.read()
        ta = TypeAdapter(cls)
        return ta.validate_json(data)
