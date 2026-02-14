"""Operations for reading and writing to disk. All side effects should go here."""
# actually i might move this into core... Config.from_file are side effects anyways

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torchcodec.decoders import AudioDecoder

from . import types as t
from .core import Audio

if TYPE_CHECKING:
    from .models import ModelT


def read_audio(
    file: t.StrPath,
    target_sr: t.SampleRate,
    target_channels: int | None,
    device: torch.device | None = None,
) -> Audio[t.RawAudioTensor]:
    """Loads, resamples and converts channels."""
    decoder = AudioDecoder(source=file, sample_rate=target_sr, num_channels=target_channels)
    samples = decoder.get_all_samples()
    waveform = samples.data.to(device)

    return Audio(t.RawAudioTensor(waveform), samples.sample_rate)


# NOTE: torchaudio.save is simple enough and a wrapper is not needed.


#
# model loading
#


def load_weights(
    model: ModelT,
    checkpoint_file: t.StrPath | bytes,
    device: torch.device | str,
    *,
    strict: bool = False,
) -> ModelT:
    """Load the weights from a checkpoint into the given model.

    Handles standard PyTorch checkpoints and PyTorch Lightning checkpoints (stripping `model.` prefix).
    """

    state_dict = torch.load(checkpoint_file, map_location=device, weights_only=True)

    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            new_state_dict[key[6:]] = value
        else:
            new_state_dict[key] = value
    state_dict = new_state_dict

    # TODO: DataParallel and `module.` prefix
    model.load_state_dict(state_dict, strict=strict)
    # NOTE: do not torch.compile here!

    return model.to(device)
