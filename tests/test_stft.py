from typing import Callable

import pytest
import torch
import torchaudio.transforms as T

from splifft.models.utils.stft import Stft


@pytest.mark.parametrize("n_fft", [1024, 2049])
@pytest.mark.parametrize("hop_length", [256, 300])
@pytest.mark.parametrize("window_fn", [torch.hann_window, torch.hamming_window])
def test_stft_close_to_torchaudio(
    n_fft: int,
    hop_length: int,
    window_fn: Callable[[int], torch.Tensor],
) -> None:
    torch.manual_seed(13)
    waveform = torch.randn(2, 1, 16384)

    stft = Stft(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window_fn=window_fn,
        conv_dtype=None,
    )
    spec = torch.view_as_complex(stft(waveform).contiguous())

    reference = T.Spectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window_fn=window_fn,
        normalized=False,
        power=None,
        center=True,
    )(waveform)

    torch.testing.assert_close(spec, reference, rtol=0.0, atol=3e-4)
