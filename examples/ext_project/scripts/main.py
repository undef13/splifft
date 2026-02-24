# ruff: noqa: E402
# --8<-- [start:config]
from pathlib import Path

import torch

from splifft.config import Config

config = Config.from_file(Path("path/to/my_model_config.json"))
# --8<-- [end:config]
# --8<-- [start:model]
from my_library.models import my_model_metadata

metadata = my_model_metadata()
my_model_params = config.model.to_concrete(metadata.params)
model = metadata.model(my_model_params)
# --8<-- [end:model]
# --8<-- [start:inference]
from splifft.inference import InferenceEngine
from splifft.io import load_weights, read_audio

checkpoint_path = Path("path/to/my_model.pt")
model = load_weights(model, checkpoint_path, device="cpu")

mixture = read_audio(
    Path("path/to/mixture.wav"), config.audio_io.target_sample_rate, config.audio_io.force_channels
)
engine = InferenceEngine(
    config=config,
    model=model,
    model_params_concrete=my_model_params,
    model_device=next(model.parameters()).device,
    io_device=torch.device("cpu"),
)
result = engine.run(mixture)

print(f"{list(result.outputs.keys())=}")
# --8<-- [end:inference]
