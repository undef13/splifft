# ruff: noqa: E402
from pathlib import Path

import torch

PATH_CONFIG = Path("data/config/bs_roformer.json")
PATH_CKPT = Path("data/models/roformer-fp16.pt")
PATH_MIXTURE = Path("data/audio/input/3BFTio5296w.flac")

# 1. parse + validate a JSON *without* having to import a particular pytorch model.
from splifft.config import Config

config = Config.from_file(PATH_CONFIG)

# 2. we now want to *lock in* the configuration to a specific model.
from splifft.models import ModelMetadata
from splifft.models.bs_roformer import BSRoformer, BSRoformerParams

metadata = ModelMetadata(model_type="bs_roformer", params=BSRoformerParams, model=BSRoformer)
model_params = config.model.to_concrete(metadata.params)

# 3. `metadata` acts as a model builder
from splifft.io import load_weights

model = metadata.model(model_params)
model = load_weights(model, PATH_CKPT, device="cpu")

# 4. load audio and run inference by passing dependencies explicitly.
from splifft.inference import InferenceEngine
from splifft.io import read_audio

mixture = read_audio(
    PATH_MIXTURE,
    config.audio_io.target_sample_rate,
    config.audio_io.force_channels,
)
engine = InferenceEngine(
    config=config,
    model=model,
    model_params_concrete=model_params,
    model_device=next(model.parameters()).device,
    io_device=torch.device("cpu"),
)
result = engine.run(mixture)

print(list(result.outputs.keys()))
