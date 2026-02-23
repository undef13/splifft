"""PESTO: Pitch Estimation with Self-supervised Transposition-equivariant Objective.

See: <https://github.com/SonyCSLParis/pesto>, <https://arxiv.org/abs/2309.02265>
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Callable, Literal, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import ModelParamsLike

if TYPE_CHECKING:
    from .. import types as t


@dataclass
class PestoParams(ModelParamsLike):
    chunk_size: t.ChunkSize
    output_stem_names: tuple[t.ModelOutputStemName, ...]

    reduction: Literal["argmax", "mean", "alwa"] = "alwa"
    convert_to_freq: bool = True

    crop_freq_bins_bottom: t.Ge0[int] = 16
    crop_freq_bins_top: t.Ge0[int] = 16

    # encoder params (mir-1k_g7 defaults)
    n_chan_input: t.Gt0[int] = 1
    n_chan_layers: tuple[t.Gt0[int], ...] = (40, 30, 30, 10, 3)
    n_prefilt_layers: t.Gt0[int] = 3
    prefilt_kernel_size: t.Gt0[int] = 39
    residual: bool = True
    n_bins_in: t.Gt0[int] = 219
    output_dim: t.Gt0[int] = 384
    activation_fn: Literal["relu", "silu", "leaky"] = "leaky"
    a_lrelu: t.Ge0[float] = 0.3
    p_dropout: t.Dropout = 0.2

    bins_per_semitone: t.Gt0[int] = 3

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


class ToeplitzLinear(nn.Conv1d):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(
            in_channels=1,
            out_channels=1,
            kernel_size=in_features + out_features - 1,
            padding=out_features - 1,
            bias=False,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(input.unsqueeze(-2)).squeeze(-2)


class Resnet1d(nn.Module):
    """Compact 1D CNN used by PESTO to decode HCQT frames into activations."""

    def __init__(
        self,
        *,
        n_chan_input: int = 1,
        n_chan_layers: tuple[int, ...] = (40, 30, 30, 10, 3),
        n_prefilt_layers: int = 3,
        prefilt_kernel_size: int = 39,
        residual: bool = True,
        n_bins_in: int = 219,
        output_dim: int = 384,
        activation_fn: Literal["relu", "silu", "leaky"] = "leaky",
        a_lrelu: float = 0.3,
        p_dropout: float = 0.2,
    ):
        super().__init__()

        activation_layer: Callable[[], nn.Module]
        if activation_fn == "relu":
            activation_layer = nn.ReLU
        elif activation_fn == "silu":
            activation_layer = nn.SiLU
        elif activation_fn == "leaky":
            activation_layer = partial(nn.LeakyReLU, negative_slope=a_lrelu)
        else:
            raise ValueError(f"unsupported activation_fn={activation_fn!r}")

        n_ch = list(n_chan_layers)
        if len(n_ch) < 5:
            n_ch.append(1)

        self.layernorm = nn.LayerNorm(normalized_shape=[n_chan_input, n_bins_in])

        prefilt_padding = prefilt_kernel_size // 2
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=n_chan_input,
                out_channels=n_ch[0],
                kernel_size=prefilt_kernel_size,
                padding=prefilt_padding,
                stride=1,
            ),
            activation_layer(),
            nn.Dropout(p=p_dropout),
        )
        self.n_prefilt_layers = n_prefilt_layers
        self.prefilt_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=n_ch[0],
                        out_channels=n_ch[0],
                        kernel_size=prefilt_kernel_size,
                        padding=prefilt_padding,
                        stride=1,
                    ),
                    activation_layer(),
                    nn.Dropout(p=p_dropout),
                )
                for _ in range(n_prefilt_layers - 1)
            ]
        )
        self.residual = residual

        conv_layers: list[nn.Module] = []
        for i in range(len(n_ch) - 1):
            conv_layers.extend(
                [
                    nn.Conv1d(
                        in_channels=n_ch[i],
                        out_channels=n_ch[i + 1],
                        kernel_size=1,
                        padding=0,
                        stride=1,
                    ),
                    activation_layer(),
                    nn.Dropout(p=p_dropout),
                ]
            )
        self.conv_layers = nn.Sequential(*conv_layers)

        self.flatten = nn.Flatten(start_dim=1)
        self.fc = ToeplitzLinear(n_bins_in * n_ch[-1], output_dim)
        self.final_norm = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layernorm(x)

        x = self.conv1(x)
        for i in range(0, self.n_prefilt_layers - 1):
            prefilt_layer = self.prefilt_layers[i]
            if self.residual:
                x = prefilt_layer(x) + x
            else:
                x = prefilt_layer(x)

        x = self.conv_layers(x)
        x = self.flatten(x)
        y_pred = self.fc(x)
        return cast(torch.Tensor, self.final_norm(y_pred))


class ConfidenceClassifier(nn.Module):
    """Frame-level voiced/unvoiced confidence head."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv1d(1, 1, 39, stride=3)
        self.linear = nn.Linear(72, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        geometric_mean = x.log().mean(dim=-1, keepdim=True).exp()
        arithmetic_mean = x.mean(dim=-1, keepdim=True).clamp_(min=1e-8)
        flatness = geometric_mean / arithmetic_mean

        x = F.relu(self.conv(x.unsqueeze(1)).squeeze(1))
        return torch.sigmoid(self.linear(torch.cat((x, flatness), dim=-1))).squeeze(-1)


def reduce_activations(activations: torch.Tensor, reduction: str = "alwa") -> torch.Tensor:
    """Reduce per-bin probabilities to scalar pitch per frame."""
    device = activations.device
    num_bins = int(activations.size(-1))

    bps, rem = divmod(num_bins, 128)
    if rem != 0:
        raise ValueError(f"expected output_dim to be divisible by 128, got {num_bins}")

    if reduction == "argmax":
        pred = activations.argmax(dim=-1)
        return pred.float() / bps

    all_pitches = torch.arange(num_bins, dtype=torch.float, device=device).div_(bps)
    if reduction == "mean":
        return torch.matmul(activations, all_pitches)

    if reduction == "alwa":
        center_bin = activations.argmax(dim=-1, keepdim=True)
        window = torch.arange(1, 2 * bps, device=device) - bps
        indices = (center_bin + window).clamp_(min=0, max=num_bins - 1)
        cropped_activations = activations.gather(-1, indices)
        cropped_pitches = all_pitches.unsqueeze(0).expand_as(activations).gather(-1, indices)
        return (cropped_activations * cropped_pitches).sum(dim=-1) / cropped_activations.sum(dim=-1)

    raise ValueError(f"unknown reduction={reduction!r}")


class Pesto(nn.Module):
    """PESTO inference head over externally computed HCQT features.

    Input contract: tensor of shape `(batch, time, feature_dim)` where
    `feature_dim = harmonics * freq_bins` in dB log-magnitude HCQT.
    """

    def __init__(self, cfg: PestoParams):
        super().__init__()
        self.cfg = cfg
        self.encoder = Resnet1d(
            n_chan_input=cfg.n_chan_input,
            n_chan_layers=cfg.n_chan_layers,
            n_prefilt_layers=cfg.n_prefilt_layers,
            prefilt_kernel_size=cfg.prefilt_kernel_size,
            residual=cfg.residual,
            n_bins_in=cfg.n_bins_in,
            output_dim=cfg.output_dim,
            activation_fn=cfg.activation_fn,
            a_lrelu=cfg.a_lrelu,
            p_dropout=cfg.p_dropout,
        )
        self.confidence = ConfidenceClassifier()

        self.register_buffer("shift", torch.zeros((), dtype=torch.float), persistent=True)

    def _crop_cqt(self, x: torch.Tensor) -> torch.Tensor:
        start = self.cfg.crop_freq_bins_bottom
        end = -self.cfg.crop_freq_bins_top if self.cfg.crop_freq_bins_top > 0 else None
        return x[..., start:end]

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(
                f"expected `(batch,time,feature_dim)` input, got shape={tuple(x.shape)}"
            )

        total_bins = (
            self.cfg.n_bins_in + self.cfg.crop_freq_bins_bottom + self.cfg.crop_freq_bins_top
        )
        expected_feature_dim = self.cfg.n_chan_input * total_bins
        if x.shape[-1] != expected_feature_dim:
            raise ValueError(
                "invalid PESTO feature dimension: "
                f"expected {expected_feature_dim} (= n_chan_input * (n_bins_in + crop_bottom + crop_top)), "
                f"got {x.shape[-1]}"
            )

        batch_size, num_frames, _feature_dim = x.shape
        x = x.view(batch_size, num_frames, self.cfg.n_chan_input, total_bins)
        x = x.flatten(0, 1)  # (B*T, H, F)

        # match reference implementation: convert dB back to linear energy and derive
        # confidence + volume from pre-crop HCQT bins.
        energy = x.mul(torch.log(torch.tensor(10.0, device=x.device, dtype=x.dtype)) / 10.0).exp()
        confidence_energy = energy.mean(dim=1)
        volume = energy.sum(dim=-1).mean(dim=-1)
        confidence = self.confidence(confidence_energy)

        x = self._crop_cqt(x)
        activations = self.encoder(x)

        activations = activations.view(batch_size, num_frames, activations.size(-1))
        confidence = confidence.view(batch_size, num_frames)
        volume = volume.view(batch_size, num_frames)

        shift_tensor = cast(torch.Tensor, self.shift)
        shift_bins = int(torch.round(shift_tensor * self.cfg.bins_per_semitone).item())
        activations = activations.roll(-shift_bins, dims=-1)

        pitch = reduce_activations(activations, reduction=self.cfg.reduction)
        if self.cfg.convert_to_freq:
            pitch = 440 * 2 ** ((pitch - 69) / 12)

        return {
            "pitch": pitch,
            "confidence": confidence,
            "volume": volume,
            "activations": activations,
        }
