"""Band-Split RoPE Transformer

- BS-RoFormer: https://arxiv.org/abs/2309.02612
- Mel-RoFormer: https://arxiv.org/abs/2409.04702

This implementation merges the two versions found in
[`lucidrains`'s implementation](https://github.com/lucidrains/BS-RoFormer)
However, there are several inconsistencies:

- `MLP` was defined differently in each file, one that has `depth - 1` hidden layers and one that
  has `depth` layers.
- `BSRoformer` applies one final RMSNorm after the entire stack of transformer layers, while the
  `MelBandRoformer` applies an RMSNorm at the end of *each* axial transformer block (time_transformer,
  freq_transformer, etc.) and has no final normalization layer.

Since fixing the three inconsistencies upstream is too big of a breaking change, we inherit them to
maintain compatability with community-trained models.
See: https://github.com/lucidrains/BS-RoFormer/issues/48.

To avoid dependency bloat, we do not:

- depend on `rotary_embeddings_torch`
- implement [`hyper_connections`](https://arxiv.org/abs/2409.19606)
- implement [learned value residual learning](https://doi.org/10.18653/v1%2F2025.acl-long.1375)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Literal, cast

import torch
import torch.nn.functional as F
from einops import pack, rearrange, reduce, repeat, unpack
from einops.layers.torch import Rearrange
from torch import Tensor, nn
from torch.nn import Module, ModuleList
from torch.utils.checkpoint import checkpoint

from . import ModelParamsLike

if TYPE_CHECKING:
    from .. import types as t
    from ..config import TorchDtype

# fmt: off
DEFAULT_FREQS_PER_BANDS = (
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    12, 12, 12, 12, 12, 12, 12, 12,
    24, 24, 24, 24, 24, 24, 24, 24,
    48, 48, 48, 48, 48, 48, 48, 48,
    128, 129
)
# fmt: on


@dataclass
class FixedBandsConfig:
    kind: Literal["fixed"]
    freqs_per_bands: tuple[t.Gt0[int], ...] = field(default_factory=lambda: DEFAULT_FREQS_PER_BANDS)


@dataclass
class MelBandsConfig:
    kind: Literal["mel"]
    stft_n_fft: t.Gt0[int] = 2048
    num_bands: t.Gt0[int] = 60
    sample_rate: t.Gt0[int] = 44100


@dataclass
class BaselineMaskEstimatorConfig:
    kind: Literal["baseline"] = "baseline"


@dataclass
class AxialRefinerLargeV2MaskEstimatorConfig:
    """unwa large-v2 head. Adds a small axial transformer refiner inside the mask head."""

    kind: Literal["axial_refiner_large_v2"]
    axial_refiner_depth: t.Gt0[int] = 4


@dataclass
class HyperAceResidualV1MaskEstimatorConfig:
    """unwa HyperACE v1 residual head compatibility config."""

    kind: Literal["hyperace_residual_v1"]
    num_hyperedges: t.Gt0[int] | None = None
    num_heads: t.Gt0[int] = 8


@dataclass
class HyperAceResidualV2MaskEstimatorConfig:
    """UNWA HyperACE v2 residual head compatibility config."""

    kind: Literal["hyperace_residual_v2"]
    num_hyperedges: t.Gt0[int] | None = None
    num_heads: t.Gt0[int] = 8


MaskEstimatorConfig = (
    BaselineMaskEstimatorConfig
    | AxialRefinerLargeV2MaskEstimatorConfig
    | HyperAceResidualV1MaskEstimatorConfig
    | HyperAceResidualV2MaskEstimatorConfig
)


@dataclass
class BSRoformerParams(ModelParamsLike):
    chunk_size: t.ChunkSize
    output_stem_names: tuple[t.ModelOutputStemName, ...]
    dim: t.Gt0[int]
    depth: t.Gt0[int]
    band_config: FixedBandsConfig | MelBandsConfig
    stft_hop_length: t.HopSize = 512
    stereo: bool = True
    time_transformer_depth: t.Gt0[int] = 1
    freq_transformer_depth: t.Gt0[int] = 1
    linear_transformer_depth: t.Ge0[int] = 0
    dim_head: int = 64
    heads: t.Gt0[int] = 8
    attn_dropout: t.Dropout = 0.0
    ff_dropout: t.Dropout = 0.0
    ff_mult: t.Gt0[int] = 4
    flash_attn: bool = True
    norm_output: bool = False
    """Note that in `lucidrains`' implementation, this is set to
    False for `bs_roformer` but True for `mel_roformer`!!"""
    mask_estimator_depth: t.Gt0[int] = 2
    """The number of hidden layers of the MLP is `mask_estimator_depth - 1`, that is:

- depth = 1: (dim_in, dim_out)
- depth = 2: (dim_in, dim_hidden, dim_out)

Note that in `lucidrains`' implementation of **mel-band roformers**, the number of hidden layers
is incorrectly set as `mask_estimator_depth`. This includes popular models like kim-vocals and
all models that use `zfturbo`'s music-source-separation training.

If you are migrating a mel-band roformer's `zfturbo` configuration, **increment** the mask_estimator
depth by 1.
    """
    mlp_expansion_factor: t.Gt0[int] = 4
    mask_estimator: MaskEstimatorConfig = field(default_factory=BaselineMaskEstimatorConfig)
    use_torch_checkpoint: bool = False
    sage_attention: bool = False
    use_shared_bias: bool = False  # COMPAT: weights are all zeros anyways, disabling by default
    skip_connection: bool = False  # NOTE: not yet implemented
    rms_norm_eps: t.Ge0[float] | None = None
    rotary_embed_dtype: TorchDtype | None = None
    transformer_residual_dtype: TorchDtype | None = None
    debug: bool = False
    """Whether to check for nan/inf in model outputs. Keep it off for [torch.compile][]."""

    @property
    def input_type(self) -> t.ModelInputType:
        return "spectrogram"

    @property
    def output_type(self) -> t.ModelOutputType:
        return "spectrogram_mask"

    @property
    def inference_archetype(self) -> t.InferenceArchetype:
        return "frequency_masking"


def l2norm(t: Tensor) -> Tensor:
    return F.normalize(t, dim=-1, p=2)


class RMSNorm(Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return F.normalize(x, dim=-1) * self.scale * self.gamma  # type: ignore


class RMSNormWithEps(Module):
    def __init__(self, dim: int, eps: float = 5.960464477539063e-08):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        l2_norm = torch.linalg.norm(x, dim=-1, keepdim=True)
        denom = torch.maximum(l2_norm, torch.full_like(l2_norm, self.eps))
        normalized_x = x / denom
        return normalized_x * self.scale * self.gamma  # type: ignore


def rms_norm(dim: int, eps: float | None) -> RMSNorm | RMSNormWithEps:
    if eps is None:
        return RMSNorm(dim)
    return RMSNormWithEps(dim, eps)


# attention


class RotaryEmbedding(nn.Module):
    """A performance-oriented version of RoPE.

    Unlike `lucidrains`' implementation which compute embeddings JIT during the
    forward pass and caches calls with the same or shorter sequence length,
    we simply compute them AOT as persistent buffers. To keep the computational
    graph clean, we do not support dynamic sequence lengths, learned frequencies
    or length extrapolation.
    """

    def __init__(
        self, seq_len: int, dim_head: int, *, dtype: torch.dtype | None, theta: int = 10000
    ):
        super().__init__()
        # COMPAT: the original implementation does not generate the embeddings
        # on the fly, but serialises them in fp16. there are some tiny
        # differences:
        # |                     |   from weights  |   generated    |
        # | ------------------- | --------------- | -------------- |
        # | cos_emb_time:971,22 | -0.99462890625  | -0.994140625   |
        # | cos_emb_time:971,23 | -0.99462890625  | -0.994140625   |
        # | sin_emb_time:727,12 | -0.457763671875 | -0.4580078125  |
        # | sin_emb_time:727,13 | -0.457763671875 | -0.4580078125  |
        # | sin_emb_time:825,4  | -0.8544921875   | -0.85400390625 |
        # | sin_emb_time:825,5  | -0.8544921875   | -0.85400390625 |
        freqs = 1.0 / (theta ** (torch.arange(0, dim_head, 2).float() / dim_head))
        t = torch.arange(seq_len)
        freqs = torch.einsum("i,j->ij", t, freqs)  # (seq_len, dim / 2)
        freqs = repeat(freqs, "... d -> ... (d r)", r=2)  # (seq_len, dim)
        self.cos_emb = freqs.cos().to(dtype)
        self.sin_emb = freqs.sin().to(dtype)

    def rotate_half(self, x: Tensor) -> Tensor:
        x = rearrange(x, "... (d r) -> ... d r", r=2)
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        return rearrange(x, "... d r -> ... (d r)")

    def forward(self, x: Tensor) -> Tensor:
        # x is (batch_eff, heads, seq_len_for_rotation, dim_head)
        cos_b = self.cos_emb.unsqueeze(0).unsqueeze(0).to(x.device, x.dtype)
        sin_b = self.sin_emb.unsqueeze(0).unsqueeze(0).to(x.device, x.dtype)

        term1 = x * cos_b
        term2 = self.rotate_half(x) * sin_b

        # NOTE: original impl performed addition between two f32s but it comes with 30% slowdown
        # we eliminate it so the addition is performed between two f16s (according to __init__).
        return term1 + term2


class FeedForward(Module):
    def __init__(
        self, dim: int, mult: int = 4, dropout: float = 0.0, rms_norm_eps: float | None = None
    ):
        super().__init__()
        dim_inner = int(dim * mult)
        # NOTE: in the paper: RMSNorm -> FC -> Tanh -> FC -> GLU
        self.net = nn.Sequential(
            rms_norm(dim, eps=rms_norm_eps),
            nn.Linear(dim, dim_inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_inner, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return cast(Tensor, self.net(x))


class Attention(Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        shared_qkv_bias: nn.Parameter | None = None,
        shared_out_bias: nn.Parameter | None = None,
        rotary_embed: RotaryEmbedding | None = None,
        flash: bool = True,
        sage_attention: bool = False,
        rms_norm_eps: float | None = None,
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head**-0.5
        dim_inner = heads * dim_head

        self.rotary_embed = rotary_embed

        if sage_attention:
            from .utils.attend_sage import AttendSage

            self.attend = AttendSage(flash=flash, dropout=dropout)
        else:
            from .utils.attend import Attend

            self.attend = Attend(flash=flash, dropout=dropout)  # type: ignore

        self.norm = rms_norm(dim, eps=rms_norm_eps)
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias=(shared_qkv_bias is not None))
        if shared_qkv_bias is not None:
            self.to_qkv.bias = shared_qkv_bias

        self.to_gates = nn.Linear(dim, heads)

        self.to_out = nn.Sequential(
            nn.Linear(dim_inner, dim, bias=(shared_out_bias is not None)),
            nn.Dropout(dropout),
        )
        if shared_out_bias is not None:
            self.to_out[0].bias = shared_out_bias

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)

        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, "b n (qkv h d) -> qkv b h n d", qkv=3, h=self.heads)

        if self.rotary_embed is not None:
            q = self.rotary_embed(q)
            k = self.rotary_embed(k)

        out = self.attend(q, k, v)

        gates = self.to_gates(x)
        gate_act = gates.sigmoid()

        out = out * rearrange(gate_act, "b n h -> b h n 1")

        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return cast(Tensor, out)


class LinearAttention(Module):
    """
    this flavor of linear attention proposed in https://arxiv.org/abs/2106.09681 by El-Nouby et al.
    """

    def __init__(
        self,
        *,
        dim: int,
        dim_head: int = 32,
        heads: int = 8,
        scale: int = 8,
        flash: bool = False,
        dropout: float = 0.0,
        sage_attention: bool = False,
        rms_norm_eps: float | None = None,
    ):
        super().__init__()
        dim_inner = dim_head * heads
        self.norm = rms_norm(dim, eps=rms_norm_eps)

        self.to_qkv = nn.Sequential(
            nn.Linear(dim, dim_inner * 3, bias=False),
            Rearrange("b n (qkv h d) -> qkv b h d n", qkv=3, h=heads),
        )

        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))

        if sage_attention:
            from .utils.attend_sage import AttendSage

            self.attend = AttendSage(scale=scale, dropout=dropout, flash=flash)
        else:
            from .utils.attend import Attend

            self.attend = Attend(scale=scale, dropout=dropout, flash=flash)  # type: ignore

        self.to_out = nn.Sequential(
            Rearrange("b h d n -> b n (h d)"), nn.Linear(dim_inner, dim, bias=False)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)

        q, k, v = self.to_qkv(x)

        q, k = map(l2norm, (q, k))
        q = q * self.temperature.exp()

        out = self.attend(q, k, v)

        return cast(Tensor, self.to_out(out))


class Transformer(Module):
    def __init__(
        self,
        *,
        dim: int,
        depth: int,
        dim_head: int = 64,
        heads: int = 8,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        ff_mult: int = 4,
        norm_output: bool = True,
        rotary_embed: RotaryEmbedding | None = None,
        flash_attn: bool = True,
        linear_attn: bool = False,
        sage_attention: bool = False,
        shared_qkv_bias: nn.Parameter | None = None,
        shared_out_bias: nn.Parameter | None = None,
        rms_norm_eps: float | None = None,
        transformer_residual_dtype: torch.dtype | None = None,  # COMPAT: float32, see 265
    ):
        super().__init__()
        self.layers = ModuleList([])

        for _ in range(depth):
            attn: LinearAttention | Attention
            if linear_attn:
                attn = LinearAttention(
                    dim=dim,
                    dim_head=dim_head,
                    heads=heads,
                    dropout=attn_dropout,
                    flash=flash_attn,
                    sage_attention=sage_attention,
                    rms_norm_eps=rms_norm_eps,
                )
            else:
                attn = Attention(
                    dim=dim,
                    dim_head=dim_head,
                    heads=heads,
                    dropout=attn_dropout,
                    shared_qkv_bias=shared_qkv_bias,
                    shared_out_bias=shared_out_bias,
                    rotary_embed=rotary_embed,
                    flash=flash_attn,
                    sage_attention=sage_attention,
                    rms_norm_eps=rms_norm_eps,
                )

            ff = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout, rms_norm_eps=rms_norm_eps)
            self.layers.append(ModuleList([attn, ff]))
        self.transformer_residual_dtype = transformer_residual_dtype

        self.norm = rms_norm(dim, eps=rms_norm_eps) if norm_output else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            block = cast(ModuleList, layer)
            attn = block[0]
            ff = block[1]
            attn_out = attn(x)
            if self.transformer_residual_dtype is not None:
                x = (
                    attn_out.to(self.transformer_residual_dtype)
                    + x.to(self.transformer_residual_dtype)
                ).to(x.dtype)
            else:
                x = attn_out + x

            ff_out = ff(x)
            if self.transformer_residual_dtype is not None:
                x = (
                    ff_out.to(self.transformer_residual_dtype)
                    + x.to(self.transformer_residual_dtype)
                ).to(x.dtype)
            else:
                x = ff_out + x
        return cast(Tensor, self.norm(x))


# bandsplit module


class BandSplit(Module):
    def __init__(self, dim: int, dim_inputs: tuple[int, ...], rms_norm_eps: float | None = None):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_features = ModuleList([])

        for dim_in in dim_inputs:
            net = nn.Sequential(rms_norm(dim_in, rms_norm_eps), nn.Linear(dim_in, dim))
            self.to_features.append(net)

    def forward(self, x: Tensor) -> Tensor:
        x_split = torch.split(x, list(self.dim_inputs), dim=-1)
        outs = []
        for split_input, to_feature_net in zip(x_split, self.to_features):
            split_output = to_feature_net(split_input)
            outs.append(split_output)
        return torch.stack(outs, dim=-2)


def mlp(
    dim_in: int,
    dim_out: int,
    dim_hidden: int | None = None,
    depth: int = 1,
    activation: type[Module] = nn.Tanh,
) -> nn.Sequential:
    dim_hidden_ = dim_hidden or dim_in

    net: list[Module] = []
    # NOTE: in lucidrain's impl, `bs_roformer` has `depth - 1` but `mel_roformer` has `depth`
    num_hidden_layers = depth - 1
    dims = (dim_in, *((dim_hidden_,) * num_hidden_layers), dim_out)

    for ind, (layer_dim_in, layer_dim_out) in enumerate(zip(dims[:-1], dims[1:])):
        is_last = ind == (len(dims) - 2)

        net.append(nn.Linear(layer_dim_in, layer_dim_out))

        if is_last:
            continue

        net.append(activation())

    return nn.Sequential(*net)


def _build_band_to_freq_mlps(
    *,
    dim: int,
    dim_inputs: tuple[int, ...],
    depth: int,
    mlp_expansion_factor: int,
) -> ModuleList:
    dim_hidden = dim * mlp_expansion_factor
    to_freqs = ModuleList()
    for dim_in in dim_inputs:
        to_freqs.append(
            nn.Sequential(
                mlp(dim, dim_in * 2, dim_hidden=dim_hidden, depth=depth),
                nn.GLU(dim=-1),
            )
        )
    return to_freqs


class MaskEstimator(Module):
    def __init__(
        self,
        dim: int,
        dim_inputs: tuple[int, ...],
        depth: int,
        mlp_expansion_factor: int,
    ):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_freqs = _build_band_to_freq_mlps(
            dim=dim,
            dim_inputs=dim_inputs,
            depth=depth,
            mlp_expansion_factor=mlp_expansion_factor,
        )

    def forward(self, x: Tensor) -> Tensor:
        x_unbound = x.unbind(dim=-2)

        outs = []

        for band_features, mlp_net in zip(x_unbound, self.to_freqs):
            freq_out = mlp_net(band_features)
            outs.append(freq_out)

        return torch.cat(outs, dim=-1)


class AxialRefinerLargeV2MaskEstimator(Module):
    def __init__(
        self,
        dim: int,
        dim_inputs: tuple[int, ...],
        mlp_depth: int,
        mlp_expansion_factor: int,
        axial_refiner_depth: int,
        t_frames: int,
        num_bands: int,
        rotary_embed_dtype: torch.dtype | None,
    ):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_freqs = _build_band_to_freq_mlps(
            dim=dim,
            dim_inputs=dim_inputs,
            depth=mlp_depth,
            mlp_expansion_factor=mlp_expansion_factor,
        )

        self.layers = ModuleList([])

        heads = 8
        dim_head = 64

        time_rotary_embed = RotaryEmbedding(
            seq_len=t_frames,
            dim_head=dim_head,
            dtype=rotary_embed_dtype,
        )
        freq_rotary_embed = RotaryEmbedding(
            seq_len=num_bands,
            dim_head=dim_head,
            dtype=rotary_embed_dtype,
        )

        for _ in range(axial_refiner_depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Transformer(
                            dim=dim,
                            depth=1,
                            heads=heads,
                            dim_head=dim_head,
                            attn_dropout=0.0,
                            ff_dropout=0.0,
                            flash_attn=True,
                            norm_output=False,
                            rotary_embed=time_rotary_embed,
                            sage_attention=False,
                        ),
                        Transformer(
                            dim=dim,
                            depth=1,
                            heads=heads,
                            dim_head=dim_head,
                            attn_dropout=0.0,
                            ff_dropout=0.0,
                            flash_attn=True,
                            norm_output=False,
                            rotary_embed=freq_rotary_embed,
                            sage_attention=False,
                        ),
                    ]
                )
            )

        self.norm = RMSNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        for transformer_block in self.layers:
            block = cast(ModuleList, transformer_block)
            time_transformer, freq_transformer = block

            x = rearrange(x, "b t f d -> b f t d")
            x, ps = pack([x], "* t d")

            x = time_transformer(x)

            (x,) = unpack(x, ps, "* t d")
            x = rearrange(x, "b f t d -> b t f d")
            x, ps = pack([x], "* f d")

            x = freq_transformer(x)

            (x,) = unpack(x, ps, "* f d")

        x = self.norm(x)

        x_unbound = x.unbind(dim=-2)

        outs = []

        for band_features, mlp_net in zip(x_unbound, self.to_freqs):
            freq_out = mlp_net(band_features)
            outs.append(freq_out)

        return torch.cat(outs, dim=-1)


class HyperAceResidualMaskEstimator(Module):
    def __init__(
        self,
        dim: int,
        dim_inputs: tuple[int, ...],
        depth: int,
        mlp_expansion_factor: int,
        segm: nn.Module,
    ):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_freqs = _build_band_to_freq_mlps(
            dim=dim,
            dim_inputs=dim_inputs,
            depth=depth,
            mlp_expansion_factor=mlp_expansion_factor,
        )

        self.segm = segm

    def forward(self, x: Tensor) -> Tensor:
        y = rearrange(x, "b t f c -> b c t f")
        y = self.segm(y)
        y = rearrange(y, "b c t f -> b t (f c)")

        x_unbound = x.unbind(dim=-2)
        outs = []
        for band_features, mlp_net in zip(x_unbound, self.to_freqs):
            freq_out = mlp_net(band_features)
            outs.append(freq_out)

        return cast(Tensor, torch.cat(outs, dim=-1) + y)


class BSRoformer(Module):
    def __init__(self, cfg: BSRoformerParams):
        super().__init__()
        self.stereo = cfg.stereo
        self.audio_channels = 2 if cfg.stereo else 1
        self.num_stems = len(cfg.output_stem_names)
        self.use_torch_checkpoint = cfg.use_torch_checkpoint
        self.skip_connection = cfg.skip_connection

        self.layers = ModuleList([])

        self.shared_qkv_bias: nn.Parameter | None = None
        self.shared_out_bias: nn.Parameter | None = None
        if cfg.use_shared_bias:
            dim_inner = cfg.heads * cfg.dim_head
            self.shared_qkv_bias = nn.Parameter(torch.ones(dim_inner * 3))
            self.shared_out_bias = nn.Parameter(torch.ones(cfg.dim))

        transformer = partial(
            Transformer,
            dim=cfg.dim,
            heads=cfg.heads,
            dim_head=cfg.dim_head,
            attn_dropout=cfg.attn_dropout,
            ff_dropout=cfg.ff_dropout,
            ff_mult=cfg.ff_mult,
            flash_attn=cfg.flash_attn,
            norm_output=cfg.norm_output,
            sage_attention=cfg.sage_attention,
            shared_qkv_bias=self.shared_qkv_bias,
            shared_out_bias=self.shared_out_bias,
            rms_norm_eps=cfg.rms_norm_eps,
            transformer_residual_dtype=cfg.transformer_residual_dtype,
        )

        t_frames = cfg.chunk_size // cfg.stft_hop_length + 1  # e.g. 588800 // 512 + 1 = 1151
        time_rotary_embed = RotaryEmbedding(
            seq_len=t_frames, dim_head=cfg.dim_head, dtype=cfg.rotary_embed_dtype
        )

        if is_mel := isinstance(cfg.band_config, MelBandsConfig):
            from torchaudio.functional import melscale_fbanks

            mel_cfg = cfg.band_config
            num_bands = mel_cfg.num_bands
            freqs = mel_cfg.stft_n_fft // 2 + 1
            mel_filter_bank = melscale_fbanks(
                n_freqs=freqs,
                f_min=0.0,
                f_max=float(mel_cfg.sample_rate / 2),
                n_mels=num_bands,
                sample_rate=mel_cfg.sample_rate,
                norm="slaney",
                mel_scale="slaney",
            ).T
            # TODO: adopt https://github.com/lucidrains/BS-RoFormer/issues/47
            mel_filter_bank[0, 0] = 1.0
            mel_filter_bank[-1, -1] = 1.0

            freqs_per_band_mask = mel_filter_bank > 0
            assert freqs_per_band_mask.any(dim=0).all(), (
                "all frequencies must be covered by at least one band"
            )

            repeated_freq_indices = repeat(torch.arange(freqs), "f -> b f", b=num_bands)
            freq_indices = repeated_freq_indices[freqs_per_band_mask]
            if self.stereo:
                freq_indices = repeat(freq_indices, "f -> f s", s=2)
                freq_indices = freq_indices * 2 + torch.arange(2)
                freq_indices = rearrange(freq_indices, "f s -> (f s)")
            self.register_buffer("freq_indices", freq_indices, persistent=False)
            self.register_buffer("freqs_per_band_mask", freqs_per_band_mask, persistent=False)

            num_freqs_per_band = reduce(freqs_per_band_mask, "b f -> b", "sum")
            num_bands_per_freq = reduce(freqs_per_band_mask, "b f -> f", "sum")

            self.register_buffer("num_freqs_per_band", num_freqs_per_band, persistent=False)
            self.register_buffer("num_bands_per_freq", num_bands_per_freq, persistent=False)

        elif isinstance(cfg.band_config, FixedBandsConfig):
            num_freqs_per_band = torch.tensor(cfg.band_config.freqs_per_bands)
            num_bands = len(cfg.band_config.freqs_per_bands)
        else:
            raise TypeError(f"unknown band config: {cfg.band_config}")
        self.is_mel = is_mel

        freq_rotary_embed = RotaryEmbedding(
            seq_len=num_bands, dim_head=cfg.dim_head, dtype=cfg.rotary_embed_dtype
        )

        for _ in range(cfg.depth):
            tran_modules = []
            if cfg.linear_transformer_depth > 0:
                tran_modules.append(
                    transformer(depth=cfg.linear_transformer_depth, linear_attn=True)
                )
            tran_modules.append(
                transformer(depth=cfg.time_transformer_depth, rotary_embed=time_rotary_embed)
            )
            tran_modules.append(
                transformer(depth=cfg.freq_transformer_depth, rotary_embed=freq_rotary_embed)
            )
            self.layers.append(nn.ModuleList(tran_modules))

        self.final_norm = (
            rms_norm(cfg.dim, eps=cfg.rms_norm_eps) if not self.is_mel else nn.Identity()
        )

        freqs_per_bands_with_complex = tuple(
            2 * f * self.audio_channels for f in num_freqs_per_band.tolist()
        )

        self.band_split = BandSplit(
            dim=cfg.dim,
            dim_inputs=freqs_per_bands_with_complex,
            rms_norm_eps=cfg.rms_norm_eps,
        )

        self.mask_estimators = nn.ModuleList([])

        def build_hyperace(config: MaskEstimatorConfig) -> nn.Module:
            if isinstance(config, HyperAceResidualV1MaskEstimatorConfig):
                from .utils.hyperace import SegmModelHyperAceV1

                return SegmModelHyperAceV1(
                    in_bands=len(freqs_per_bands_with_complex),
                    in_dim=cfg.dim,
                    out_bins=sum(freqs_per_bands_with_complex) // 4,
                    num_hyperedges=config.num_hyperedges or 16,
                    num_heads=config.num_heads,
                )
            if isinstance(config, HyperAceResidualV2MaskEstimatorConfig):
                from .utils.hyperace import SegmModelHyperAceV2

                return SegmModelHyperAceV2(
                    in_bands=len(freqs_per_bands_with_complex),
                    in_dim=cfg.dim,
                    out_bins=sum(freqs_per_bands_with_complex) // 4,
                    num_hyperedges=config.num_hyperedges or 32,
                    num_heads=config.num_heads,
                )
            raise TypeError(f"mask estimator is not hyperace-based: {config}")

        def build_mask_estimator(config: MaskEstimatorConfig) -> nn.Module:
            if isinstance(config, BaselineMaskEstimatorConfig):
                return MaskEstimator(
                    dim=cfg.dim,
                    dim_inputs=freqs_per_bands_with_complex,
                    depth=cfg.mask_estimator_depth,
                    mlp_expansion_factor=cfg.mlp_expansion_factor,
                )

            if isinstance(config, AxialRefinerLargeV2MaskEstimatorConfig):
                return AxialRefinerLargeV2MaskEstimator(
                    dim=cfg.dim,
                    dim_inputs=freqs_per_bands_with_complex,
                    mlp_depth=cfg.mask_estimator_depth,
                    mlp_expansion_factor=cfg.mlp_expansion_factor,
                    axial_refiner_depth=config.axial_refiner_depth,
                    t_frames=t_frames,
                    num_bands=num_bands,
                    rotary_embed_dtype=cfg.rotary_embed_dtype,
                )

            if isinstance(
                config,
                HyperAceResidualV1MaskEstimatorConfig | HyperAceResidualV2MaskEstimatorConfig,
            ):
                return HyperAceResidualMaskEstimator(
                    dim=cfg.dim,
                    dim_inputs=freqs_per_bands_with_complex,
                    depth=cfg.mask_estimator_depth,
                    mlp_expansion_factor=cfg.mlp_expansion_factor,
                    segm=build_hyperace(config),
                )

            raise TypeError(f"unknown mask_estimator config: {config}")

        for _ in range(len(cfg.output_stem_names)):
            self.mask_estimators.append(build_mask_estimator(cfg.mask_estimator))

        self.debug = cfg.debug

    def forward(self, stft_repr: Tensor) -> Tensor:
        """
        :param stft_repr: input spectrogram. shape (b, f*s, t, c)
        :return: estimated mask. shape (b, n, f*s, t, c)
        """
        batch, _, t_frames, _ = stft_repr.shape
        device = stft_repr.device
        if self.is_mel:
            batch_arange = torch.arange(batch, device=device)[..., None]
            x = stft_repr[batch_arange, cast(Tensor, self.freq_indices)]
            x = rearrange(x, "b f t c -> b t (f c)")
        else:
            x = rearrange(stft_repr, "b f t c -> b t (f c)")

        if self.debug and (torch.isnan(x).any() or torch.isinf(x).any()):
            raise RuntimeError(
                f"nan/inf in x after rearrange: {x.isnan().sum()} nans, {x.isinf().sum()} infs"
            )

        if self.use_torch_checkpoint:
            x = cast(Tensor, checkpoint(self.band_split, x, use_reentrant=False))
        else:
            x = cast(Tensor, self.band_split(x))

        if self.debug and (torch.isnan(x).any() or torch.isinf(x).any()):
            raise RuntimeError(
                f"nan/inf in x after band_split: {x.isnan().sum()} nans, {x.isinf().sum()} infs"
            )

        # axial / hierarchical attention

        store: list[Tensor | None] = [None] * len(self.layers)
        for i, transformer_block in enumerate(self.layers):
            block = cast(ModuleList, transformer_block)
            if len(block) == 3:
                linear_transformer, time_transformer, freq_transformer = block

                x, ft_ps = pack([x], "b * d")
                if self.use_torch_checkpoint:
                    x = checkpoint(linear_transformer, x, use_reentrant=False)
                else:
                    x = linear_transformer(x)
                (x,) = unpack(x, ft_ps, "b * d")
            else:
                time_transformer, freq_transformer = block

            if self.skip_connection:
                for j in range(i):
                    if store[j] is not None:
                        assert x is not None
                        x = x + cast(Tensor, store[j])

            x = rearrange(x, "b t f d -> b f t d")
            x, ps = pack([x], "* t d")

            if self.use_torch_checkpoint:
                x = checkpoint(time_transformer, x, use_reentrant=False)
            else:
                x = time_transformer(x)

            (x,) = unpack(x, ps, "* t d")
            x = rearrange(x, "b f t d -> b t f d")
            x, ps = pack([x], "* f d")

            if self.use_torch_checkpoint:
                x = checkpoint(freq_transformer, x, use_reentrant=False)
            else:
                x = freq_transformer(x)

            (x,) = unpack(x, ps, "* f d")

            if self.skip_connection:
                store[i] = x

        x = self.final_norm(x)

        if self.use_torch_checkpoint:
            mask = torch.stack(
                [
                    cast(Tensor, checkpoint(fn, x, use_reentrant=False))
                    for fn in self.mask_estimators
                ],
                dim=1,
            )
        else:
            mask = torch.stack([fn(x) for fn in self.mask_estimators], dim=1)
        mask = rearrange(mask, "b n t (f c) -> b n f t c", c=2)

        if not self.is_mel:
            return mask

        stft_repr = rearrange(stft_repr, "b f t c -> b 1 f t c")
        # stft_repr may be fp16 but complex32 support is experimental so we upcast it early
        stft_repr_complex = torch.view_as_complex(stft_repr.to(torch.float32))

        masks_per_band_complex = torch.view_as_complex(mask)
        masks_per_band_complex = masks_per_band_complex.type(stft_repr_complex.dtype)

        scatter_indices = repeat(
            cast(Tensor, self.freq_indices),
            "f -> b n f t",
            b=batch,
            n=self.num_stems,
            t=stft_repr_complex.shape[-1],
        )
        stft_repr_expanded_stems = repeat(stft_repr_complex, "b 1 ... -> b n ...", n=self.num_stems)

        masks_summed = torch.zeros_like(stft_repr_expanded_stems).scatter_add_(
            2, scatter_indices, masks_per_band_complex
        )

        denom = cast(Tensor, repeat(self.num_bands_per_freq, "f -> (f r) 1", r=self.audio_channels))
        masks_averaged = masks_summed / denom.clamp(min=1e-8)

        return torch.view_as_real(masks_averaged).to(stft_repr.dtype)
