"""Source separation models."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar, runtime_checkable

from torch import nn

if TYPE_CHECKING:
    from .. import types as t


@runtime_checkable
class ModelParamsLike(Protocol):
    """A trait that must be implemented to be considered a model parameter.
    Note that `input_type` and `output_type` belong to a model's definition
    and does not allow modification via the configuration dictionary."""

    chunk_size: t.ChunkSize
    output_stem_names: tuple[t.ModelOutputStemName, ...]

    @property
    def input_type(self) -> t.ModelInputType: ...
    @property
    def output_type(self) -> t.ModelOutputType: ...
    @property
    def inference_archetype(self) -> t.InferenceArchetype: ...


ModelT = TypeVar("ModelT", bound=nn.Module)
ModelParamsLikeT = TypeVar("ModelParamsLikeT", bound=ModelParamsLike)


@dataclass
class ModelMetadata(Generic[ModelT, ModelParamsLikeT]):
    """Metadata about a model, including its type, parameter class, and model class."""

    model_type: t.ModelType
    params: type[ModelParamsLikeT]
    model: type[ModelT]

    @classmethod
    def from_module(
        cls,
        module_name: str,
        model_cls_name: str,
        *,
        model_type: t.ModelType,
        package: str | None = None,
    ) -> ModelMetadata[nn.Module, ModelParamsLike]:
        """
        Dynamically import a model named `X` and its parameter dataclass `XParams` under a
        given module name (e.g. `splifft.models.bs_roformer`).

        :param model_cls_name: The name of the model class to import, e.g. `BSRoformer`.
        :param module_name: The name of the module to import, e.g. `splifft.models.bs_roformer`.
        :param model_type: The type of the model, e.g. `bs_roformer`.
        :param package: The package to use as the anchor point from which to resolve the relative import.
        to an absolute import. This is only required when performing a relative import.
        """
        _loc = f"{module_name=} under {package=}"
        try:
            module = importlib.import_module(module_name, package)
        except ImportError as e:
            raise ValueError(f"failed to find or import module for {_loc}") from e

        params_cls_name = f"{model_cls_name}Params"
        model_cls = getattr(module, model_cls_name, None)
        params_cls = getattr(module, params_cls_name, None)
        if model_cls is None or params_cls is None:
            raise AttributeError(
                f"expected to find a class named `{params_cls_name}` in {_loc}, but it was not found."
            )

        return ModelMetadata(
            model_type=model_type,
            model=model_cls,
            params=params_cls,
        )
