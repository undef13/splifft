"""Time-Frequency Convolutions and Time-Distributed Fully-connected (TFC-TDF)

See: <https://arxiv.org/pdf/2306.09382>
"""

from __future__ import annotations

from collections.abc import Callable

from torch import Tensor, nn

NormFactory = Callable[[int], nn.Module]
ActFactory = Callable[[], nn.Module]


def instance_norm_factory(channels: int) -> nn.Module:
    return nn.InstanceNorm2d(channels, affine=True, eps=1e-8)


def silu_factory() -> nn.Module:
    return nn.SiLU()


class TfcTdfBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        f_bins: int,
        bottleneck_factor: int,
        *,
        norm_factory: NormFactory,
        act_factory: ActFactory,
    ):
        super().__init__()

        self.tfc1 = nn.Sequential(
            norm_factory(in_channels),
            act_factory(),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
        )
        self.tdf = nn.Sequential(
            norm_factory(out_channels),
            act_factory(),
            nn.Linear(f_bins, f_bins // bottleneck_factor, bias=False),
            norm_factory(out_channels),
            act_factory(),
            nn.Linear(f_bins // bottleneck_factor, f_bins, bias=False),
        )
        self.tfc2 = nn.Sequential(
            norm_factory(out_channels),
            act_factory(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
        )
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        s = self.shortcut(x)
        x = self.tfc1(x)
        x = x + self.tdf(x)
        x = self.tfc2(x)
        x = x + s
        return x


class TfcTdf(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        f_bins: int,
        bottleneck_factor: int,
        *,
        norm_factory: NormFactory,
        act_factory: ActFactory,
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                TfcTdfBlock(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    f_bins=f_bins,
                    bottleneck_factor=bottleneck_factor,
                    norm_factory=norm_factory,
                    act_factory=act_factory,
                )
                for i in range(num_blocks)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        return x
