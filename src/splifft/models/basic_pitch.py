"""ICASSP 2022 Basic Pitch. Raw multi-stream outputs only, no symbolic decoding.

See: <https://github.com/spotify/basic-pitch>, <https://arxiv.org/abs/2203.09893>
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import ModelParamsLike

if TYPE_CHECKING:
    from .. import types as t


@dataclass
class BasicPitchParams(ModelParamsLike):
    chunk_size: t.ChunkSize
    output_stem_names: tuple[t.ModelOutputStemName, ...]

    n_semitones: t.Gt0[int] = 88
    contour_bins_per_semitone: t.Gt0[int] = 3
    cqt_bins_per_semitone: t.Gt0[int] = 3
    cqt_n_bins: t.Gt0[int] = 372
    stack_harmonics: tuple[t.Gt0[float], ...] = (0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)

    @property
    def input_channels(self) -> t.ModelInputChannels:
        return "mono"

    @property
    def input_type(self) -> t.ModelInputType:
        return "spectrogram"

    @property
    def output_type(self) -> t.ModelOutputType:
        return "multi_stream"

    @property
    def inference_archetype(self) -> t.InferenceArchetype:
        return "sequence_labeling"


class HarmonicStacking(nn.Module):
    def __init__(
        self,
        *,
        bins_per_semitone: int,
        harmonics: tuple[float, ...],
        n_output_freqs: int,
    ):
        super().__init__()
        self.n_output_freqs = n_output_freqs
        self.shifts = [int(round(12.0 * bins_per_semitone * math.log2(h))) for h in harmonics]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (B, T, F)
        :return: (B, H, T, F_out)
        """
        stacked: list[torch.Tensor] = []
        for shift in self.shifts:
            if shift == 0:
                shifted = x
            elif shift > 0:
                shifted = F.pad(x[:, :, shift:], (0, shift))
            else:
                shifted = F.pad(x[:, :, :shift], (-shift, 0))
            stacked.append(shifted[:, :, : self.n_output_freqs])
        return torch.stack(stacked, dim=1)


class BasicPitch(nn.Module):
    def __init__(self, cfg: BasicPitchParams):
        super().__init__()
        self.cfg = cfg
        self.n_contour_bins = cfg.n_semitones * cfg.contour_bins_per_semitone

        self.hs = HarmonicStacking(
            bins_per_semitone=cfg.cqt_bins_per_semitone,
            harmonics=cfg.stack_harmonics,
            n_output_freqs=self.n_contour_bins,
        )

        num_in_channels = len(cfg.stack_harmonics)
        self.conv_contour = nn.Sequential(
            nn.Conv2d(num_in_channels, 8, kernel_size=(3, 39), padding="same"),
            nn.BatchNorm2d(8, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=5, padding="same"),
            nn.Sigmoid(),
        )
        self.conv_note = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=(1, 3)),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=(7, 3), padding="same"),
            nn.Sigmoid(),
        )
        self.conv_onset_pre = nn.Sequential(
            nn.Conv2d(num_in_channels, 32, kernel_size=5, stride=(1, 3)),
            nn.BatchNorm2d(32, eps=0.001),
            nn.ReLU(),
        )
        self.conv_onset_post = nn.Sequential(
            nn.Conv2d(33, 1, kernel_size=3, stride=1, padding="same"),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"expected `(B,T,F)` input, got {tuple(x.shape)}")
        if x.shape[-1] != self.cfg.cqt_n_bins:
            raise ValueError(f"expected feature dim {self.cfg.cqt_n_bins}, got {x.shape[-1]}")

        cqt = self.hs(x)

        contour = self.conv_contour(cqt)

        contour_for_note = F.pad(contour, (2, 2, 3, 3))
        note = self.conv_note(contour_for_note)

        cqt_for_onset = F.pad(cqt, (1, 1, 2, 2))
        onset_pre = self.conv_onset_pre(cqt_for_onset)
        onset_in = torch.cat((note, onset_pre), dim=1)
        onset = self.conv_onset_post(onset_in)

        contour_out = contour.squeeze(1)
        note_out = note.squeeze(1)
        onset_out = onset.squeeze(1)

        return {
            "onset": onset_out,
            "note": note_out,
            "contour": contour_out,
        }
