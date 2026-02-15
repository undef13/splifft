from typing import Any

import pytest
from pydantic import ValidationError

from splifft import types as t
from splifft.config import Config, LazyModelConfig
from splifft.models import ModelParamsLike

#
# model
#

MODEL_CONFIG_BASE = {
    "chunk_size": 8192,
    "output_stem_names": ["vocals", "drums", "bass", "other"],
}


def test_model_config_lazy_required() -> None:
    model_config = LazyModelConfig.model_validate(MODEL_CONFIG_BASE)
    assert isinstance(model_config, LazyModelConfig)
    assert model_config.chunk_size == 8192
    assert model_config.output_stem_names == ("vocals", "drums", "bass", "other")


def test_model_config_invalid_stems() -> None:
    with pytest.raises(ValidationError):
        LazyModelConfig.model_validate(
            {
                **MODEL_CONFIG_BASE,
                "output_stem_names": ["vocals", "vocals"],
            }
        )
    with pytest.raises(ValidationError):
        LazyModelConfig.model_validate(
            {
                **MODEL_CONFIG_BASE,
                "output_stem_names": [],
            }
        )


MODEL_CONFIG_EXTRA = {
    **MODEL_CONFIG_BASE,
    "param_1": "13",
    "param_2": 13,
}


def test_model_config_lazy_extra_fields() -> None:
    lazy_model_config = LazyModelConfig.model_validate(MODEL_CONFIG_EXTRA)
    assert isinstance(lazy_model_config, LazyModelConfig)
    # arbitrary fields are allowed and not validated until `to_concrete` is called
    assert getattr(lazy_model_config, "param_1") == "13"
    assert getattr(lazy_model_config, "param_2") == 13


def test_model_config_to_concrete_extra_fields() -> None:
    # simulate a user creating their own model and validate it
    from dataclasses import dataclass

    @dataclass
    class MyModelParam(ModelParamsLike):
        chunk_size: int
        output_stem_names: tuple[str, ...]
        param_1: str
        param_2: int

        @property
        def input_type(self) -> t.ModelInputType:
            return "waveform"

        @property
        def output_type(self) -> t.ModelOutputType:
            return "waveform"

        @property
        def inference_archetype(self) -> t.InferenceArchetype:
            return "standard_end_to_end"

    lazy_model_config = LazyModelConfig.model_validate(MODEL_CONFIG_EXTRA)
    model_config = lazy_model_config.to_concrete(MyModelParam)
    assert isinstance(model_config, MyModelParam)
    assert model_config.param_1 == "13"
    assert model_config.param_2 == 13

    lazy_model_config_extra = LazyModelConfig.model_validate(
        {**MODEL_CONFIG_EXTRA, "param_3": (13,)}
    )
    assert isinstance(lazy_model_config_extra, LazyModelConfig)  # lazy allows extra invalid fields
    # extra fields are now not allowed on `to_concrete`
    # this would also mean that if MyModelConfig doesn't conform to ModelParamLike, it will raise an error
    # when `to_concrete` is called
    with pytest.raises(ValidationError):
        lazy_model_config_extra.to_concrete(MyModelParam)


def test_config_required() -> None:
    config_data: dict[str, Any] = {
        # simulate user forgetting to define a model id and stem names
        "model_type": "roformer",
        "model": MODEL_CONFIG_BASE,
    }
    with pytest.raises(ValidationError):
        Config.model_validate(config_data)

    config_data = {
        "identifier": "roformer",
        **config_data,
    }
    config = Config.model_validate(config_data)
    assert isinstance(config, Config)


#
# stems
#


@pytest.mark.parametrize(
    "derived_stems, expected_success",
    [
        (
            {
                "drumless": {
                    "operation": "subtract",
                    "stem_name": "mixture",
                    "by_stem_name": "drums",
                },
            },
            True,
        ),
        (
            {"drum_and_bass": {"operation": "sum", "stem_names": ["drums", "bass"]}},
            True,
        ),
        (
            {
                "__fail_non_existent": {
                    "operation": "sum",
                    "stem_names": ["drums", "non_existent_stem"],
                },
            },
            False,  # non-existent stem name
        ),
        (
            {"drums": {"operation": "sum", "stem_names": ["drums", "bass"]}},
            False,  # derived stem name conflicts with existing stem name
        ),
    ],
)
def test_config_stem(derived_stems: dict[str, Any], expected_success: bool) -> None:
    config_data: dict[str, Any] = {
        "identifier": "test_stem",
        "model_type": "roformer",
        "model": MODEL_CONFIG_BASE,
        "derived_stems": derived_stems,
    }
    if expected_success:
        Config.model_validate(config_data)
    else:
        with pytest.raises(ValidationError):
            Config.model_validate(config_data)


#
# integration test: try a real configuration file
#


@pytest.fixture
def config_roformer() -> Config:
    from splifft import PATH_CONFIG

    config = Config.from_file(PATH_CONFIG / "bs_roformer.json")
    assert isinstance(config.model, LazyModelConfig)
    return config


def test_config_roformer_concrete(config_roformer: Config) -> None:
    from splifft.models.bs_roformer import BSRoformerParams

    model_config = config_roformer.model.to_concrete(BSRoformerParams)
    assert isinstance(model_config, BSRoformerParams)


#
# static typing test
#


def test_lazy_model_config_protocol() -> None:
    # this is just to ensure we correctly implemented the ModelParamLike protocol for LazyModelConfig.
    from splifft.models import ModelParamsLike

    class _ModelParam(LazyModelConfig):
        input_type: t.ModelInputType
        output_type: t.ModelOutputType
        inference_archetype: t.InferenceArchetype

    assert set(_ModelParam.__pydantic_fields__) == ModelParamsLike.__protocol_attrs__  # type: ignore
