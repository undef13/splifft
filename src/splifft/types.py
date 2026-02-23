"""Types for documentation and data validation (for use in pydantic).

They provide semantic meaning *only* and we additionally use `NewType` for strong semantic
distinction to avoid mixing up different kinds of tensors.

Note that **no code implementations shall be placed here**.
"""

from os import PathLike
from typing import Annotated, Callable, Literal, NewType, TypeAlias, TypeVar

import annotated_types as at
from torch import Tensor

StrPath: TypeAlias = str | PathLike[str]
BytesPath: TypeAlias = bytes | PathLike[bytes]

_T = TypeVar("_T")
Gt0: TypeAlias = Annotated[_T, at.Gt(0)]
Ge0: TypeAlias = Annotated[_T, at.Ge(0)]

ModelType: TypeAlias = str
"""The type of the model, e.g. `bs_roformer`, `demucs`"""

ModelInputChannels: TypeAlias = Literal["mono", "stereo"]
"""Required channel layout for model input audio.

- `mono`: model expects a single channel
- `stereo`: model expects two channels
"""

ModelInputType: TypeAlias = Literal["waveform", "spectrogram", "waveform_and_spectrogram"]
ModelOutputType: TypeAlias = Literal[
    "waveform",
    "spectrogram_mask",
    "spectrogram",
    "logits",
    "multi_stream",
]

ChunkSize: TypeAlias = Gt0[int]
"""The length of an audio segment, in samples, processed by the model at one time.

A full audio track is often too long to fit into GPU, instead we process it in fixed-size chunks.
A larger chunk size may allow the model to capture more temporal context at the cost of increased
memory usage.
"""
HopSize: TypeAlias = Gt0[int]
"""The step size, in samples, between the start of consecutive [chunks][splifft.types.ChunkSize].

To avoid artifacts at the edges of chunks, we process them with overlap. The hop size is the
distance we "slide" the chunking window forward. `ChunkSize < HopSize` implies overlap and the
overlap amount is `ChunkSize - HopSize`.
"""

Dropout: TypeAlias = Annotated[float, at.Ge(0.0), at.Le(1.0)]

ModelOutputStemName: TypeAlias = Annotated[str, at.MinLen(1)]
"""The output stem name, e.g. `vocals`, `drums`, `bass`, etc."""


#
# key time domain concepts
#

Samples: TypeAlias = Gt0[int]
"""Number of samples in the audio signal."""

SampleRate: TypeAlias = Gt0[int]
"""The number of samples of audio recorded per second (hertz).

See [concepts](../concepts.md#introduction) for more details.
"""

Channels: TypeAlias = Gt0[int]
"""Number of audio streams.

- 1: Mono audio
- 2: Stereo (left and right). Models are usually trained on stereo audio.
"""


FileFormat: TypeAlias = Literal["flac", "wav", "ogg", "npy"]
BitRate: TypeAlias = Literal[8, 16, 24, 32, 64]
"""Number of bits of information in each sample.

It determines the dynamic range of the audio signal: the difference between the quietest and loudest
possible sounds.

- 16-bit: Standard for CD audio: ~96 dB dynamic range.
- 24-bit: Common in professional audio, allowing for more headroom during mixing
- 32-bit float: Standard in digital audio workstations (DAWs) and deep learning models.
    The amplitude is represented by a floating-point number, which prevents clipping (distortion
    from exceeding the maximum value). This library primarily works with fp32 tensors.
"""

RawAudioTensor = NewType("RawAudioTensor", Tensor)
"""Time domain tensor of audio samples.
Shape ([channels][splifft.types.Channels], [samples][splifft.types.Samples])"""

NormalizedAudioTensor = NewType("NormalizedAudioTensor", Tensor)
"""A mixture tensor that has been normalized using [on-the-fly statistics][splifft.core.NormalizationStats].
Shape ([channels][splifft.types.Channels], [samples][splifft.types.Samples])"""

#
# key time-frequency domain concepts
#

ComplexSpectrogram = NewType("ComplexSpectrogram", Tensor)
r"""A complex-valued representation of audio's frequency content over time via the STFT.

Shape ([channels][splifft.types.Channels], [frequency bins][splifft.types.FftSize], [time frames][splifft.types.ChunkSize], 2)

See [concepts](../concepts.md#complex-spectrogram) for more details.
"""

LogMelSpectrogram = NewType("LogMelSpectrogram", Tensor)
"""A real-valued log-mel spectrogram.
Shape (1, [mels], [time])"""

HybridModelInput: TypeAlias = tuple[ComplexSpectrogram, RawAudioTensor | NormalizedAudioTensor]
"""Input for hybrid models that require both spectrogram and waveform."""


# NOTE: sharing both for stft and overlap-add stitching for now
WindowShape: TypeAlias = Literal["hann", "hamming", "linear_fade"]
"""The shape of the window function applied to each chunk before computing the STFT."""


FftSize: TypeAlias = Gt0[int]
"""The number of frequency bins in the STFT, controlling the [frequency resolution](../concepts.md#fft-size)."""

Bands: TypeAlias = Tensor
"""Groups of [adjacent frequency bins in the spectrogram](../concepts.md#bands)."""

#
# miscallaneous
#
BatchSize: TypeAlias = Gt0[int]
"""The number of chunks processed simultaneously by the GPU.

Increasing the batch size can improve GPU utilisation and speed up training, but it requires more
memory.
"""


# preprocessing

PaddingMode: TypeAlias = Literal["reflect", "constant", "replicate"]
"""The method used to pad the audio before chunking, crucial for handling the edges of the audio signal.

- `reflect`: Pads the signal by reflecting the audio at the boundary. This creates a smooth
  continuation and often yields the best results for music.
- `constant`: Pads with zeros. Simpler, but can introduce silence at the edges.
- `replicate`: Repeats the last sample at the edge.
"""
# TODO: we should intelligently decide whether to choose reflect or constant.
# for songs that start with silence, we should use constant padding.


ChunkDuration: TypeAlias = Gt0[float]
"""The length of an audio segment, in seconds, processed by the model at one time.

Equivalent to [chunk size][splifft.types.ChunkSize] divided by the [sample rate][splifft.types.SampleRate].
"""

InferenceArchetype: TypeAlias = Literal[
    "standard_end_to_end",
    "frequency_masking",
    "sequence_labeling",
]
"""Inference pipeline archetype used to route runtime execution.

- `standard_end_to_end`: waveform -> model -> waveform (e.g. demucs)
- `frequency_masking`: waveform -> STFT -> model -> iSTFT -> waveform (e.g. bs-roformer)
- `sequence_labeling`: waveform -> optional feature extraction -> model -> sequence outputs
  (e.g. beat tracking, pitch estimation)
"""

OverlapRatio: TypeAlias = Annotated[float, at.Ge(0), at.Lt(1)]
r"""The fraction of a chunk that overlaps with the next one.

The relationship with [hop size][splifft.types.HopSize] is:
$$
\text{hop\_size} = \text{chunk\_size} \cdot (1 - \text{overlap\_ratio})
$$

- A ratio of `0.0` means no overlap (hop_size = chunk_size).
- A ratio of `0.5` means 50% overlap (hop_size = chunk_size / 2).
- A higher overlap ratio increases computational cost as more chunks are processed, but it can lead
  to smoother results by averaging more predictions for each time frame.
"""

TrimMargin: TypeAlias = Annotated[int, at.Ge(0)]
"""Number of frames to trim from the edges of each chunk when aggregating logits.

Useful for models that produce artifacts at the boundaries of predictions.
"""

OverlapMode: TypeAlias = Literal["keep_first", "keep_last"]
"""How overlapping logits chunks are resolved during aggregation.

- `keep_first`: earlier chunks in time win on overlap
- `keep_last`: later chunks in time win on overlap
"""

Padding: TypeAlias = Gt0[int]
"""Samples to add to the beginning and end of each chunk.

- To ensure that the very beginning and end of a track can be centerd within a chunk, we often may
  add "reflection padding" or "zero padding" before chunking.
- To ensure that the last chunk is full-size, we may pad the audio so its length is a multiple of
  the hop size. 
"""

PaddedChunkedAudioTensor = NewType("PaddedChunkedAudioTensor", Tensor)
"""A batch of audio chunks from a padded source.
Shape ([batch size][splifft.types.BatchSize], [channels][splifft.types.Channels], [chunk size][splifft.types.ChunkSize])"""

NumModelStems: TypeAlias = Gt0[int]
"""The number of stems the model outputs. This should be the length of [splifft.models.ModelParamsLike.output_stem_names]."""

# post separation stitching
SeparatedSpectrogramTensor = NewType("SeparatedSpectrogramTensor", Tensor)
"""A batch of separated spectrograms.
Shape (b, n, f*s, t, c=2)"""

SeparatedChunkedTensor = NewType("SeparatedChunkedTensor", Tensor)
"""A batch of separated audio chunks from the model.
Shape ([batch size][splifft.types.BatchSize], [number of stems][splifft.types.NumModelStems], [channels][splifft.types.Channels], [chunk size][splifft.types.ChunkSize])"""

WindowTensor = NewType("WindowTensor", Tensor)
"""A 1D tensor representing a window function.
Shape ([chunk size][splifft.types.ChunkSize])"""

RawSeparatedTensor = NewType("RawSeparatedTensor", Tensor)
"""The final, stitched, raw-domain separated audio.
Shape ([number of stems][splifft.types.NumModelStems], [channels][splifft.types.Channels], [samples][splifft.types.Samples])"""

#
# logits
#

LogitsTensor = NewType("LogitsTensor", Tensor)
"""A time-series tensor representing activation probabilities or logits.
Shape ([number of stems][splifft.types.NumModelStems], [time]) or batched"""


#
# wave-to-wave wrapper
#

PreprocessFn: TypeAlias = Callable[[RawAudioTensor | NormalizedAudioTensor], tuple[Tensor, ...]]
PostprocessFn: TypeAlias = Callable[..., SeparatedChunkedTensor | LogitsTensor]

#
# registry
#

Identifier: TypeAlias = at.LowerCase[str]
"""`{{architecture}}-{{first_author}}-{{unique_name_short}}`, use underscore if it has spaces"""
# fmt: off
Instrument: TypeAlias = Literal[
    "instrum", "vocals", "drums", "bass", "other", "piano",
    "lead_vocals", "back_vocals",
    "guitar",
    "vocals1", "vocals2",
    "strings", "wind",
    "music", "sfx", "speech",
    "foreground", "background",
    "restored",
    "back", "lead",
    "back-instrum", "kick", "snare", "toms", "hh", "cymbals", "hh-cymbals",
    "male", "female",
    # we follow mvsep's leaderboard naming conventions up til this point, the following are extra
    "violin",
    "ride", "crash",
    "dry", "reverb",
    "clean", "crowd",
    "denoised", "noise",
    "similarity", "difference", "center", # see: https://github.com/ZFTurbo/Music-Source-Separation-Training/issues/1#issuecomment-2417116936
    "keyboards", "synthesizer", "percussion", "orchestral",
    # weird
    "no_drum-bass",  # viperx 1053
    "karaoke",
    # event detection
    "beat", "downbeat",
    # MIR sequence labeling / pitch
    "pitch", "confidence", "volume", "activations"
]
# fmt: on
Metric: TypeAlias = Literal[
    "sdr", "si_sdr", "l1_freq", "log_wmse", "aura_stft", "aura_mrstft", "bleedless", "fullness"
]

#
# evaluation metrics
# We use bold letters like $\mathbf{s}$ to denote the entire signal tensor.
# NOTE: once we implement these metrics, cut down on the docstrings.
#

Sdr: TypeAlias = float
r"""Signal-to-Distortion Ratio (decibels). Higher is better.

Measures the ratio of the power of clean reference signal to the power of all other error components
(interference, artifacts, and spatial distortion).

Definition:
$$
\text{SDR} = 10 \log_{10} \frac{|\mathbf{s}|^2}{|\mathbf{s} - \mathbf{\hat{s}}|^2},
$$
where:

- $\mathbf{s}$: ground truth source signal
- $\mathbf{\hat{s}}$: estimated source signal produced by the model
- $||\cdot||^2$: squared L2 norm (power) of the signal
"""

SiSdr: TypeAlias = float
r"""Scale-Invariant SDR (SI-SDR) is invariant to scaling errors (decibels). Higher is better.

It projects the estimate onto the reference to find the optimal scaling factor $\alpha$, creating a scaled reference that best matches the estimate's amplitude.

- Optimal scaling factor: $\alpha = \frac{\langle\mathbf{\hat{s}}, \mathbf{s}\rangle}{||\mathbf{s}||^2}$
- Scaled reference: $\mathbf{s}_\text{target} = \alpha \cdot \mathbf{s}$
- Error: $\mathbf{e} = \mathbf{\hat{s}} - \mathbf{s}_\text{target}$
- $\text{SI-SDR} = 10 \log_{10} \frac{||\mathbf{s}_\text{target}||^2}{||\mathbf{e}||^2}$
"""

L1Norm: TypeAlias = float
r"""L1 norm (mean absolute error) between two signals (dimensionless). Lower is better.

Measures the average absolute difference between the reference and estimated signals.

- Time domain: $\mathcal{L}_\text{L1} = \frac{1}{N}
\sum_{n=1}^{N} |\mathbf{s}[n] - \mathbf{\hat{s}}[n]|$,
- Frequency domain: $\mathcal{L}_\text{L1Freq} = \frac{1}{\text{MK}}\sum_{m=1}^{M}
\sum_{k=1}^{K} \left||S(m, k)| - |\hat{S}(m, k)|\right|$
"""  # NOTE: zfturbo scales by to 1-100

DbDifferenceMel: TypeAlias = float
r"""Difference in the dB-scaled mel spectrogram.
$$
\mathbf{D}(m, k) = \text{dB}(|\hat{S}_\text{mel}(m, k)|) - \text{dB}(|S_\text{mel}(m, k)|)
$$
"""

Bleedless: TypeAlias = float
r"""A metric to quantify the amount of "bleeding" from other sources. Higher is better.

Measures the average energy of the parts of the [mel spectrogram][splifft.types.DbDifferenceMel]
that are louder than the reference.
A high value indicates that the estimate contains unwanted energy (bleed) from other sources:
$$
\text{Bleed} = \text{mean}(\mathbf{D}(m, k)) \quad \forall \quad \mathbf{D}(m, k) > 0
$$
"""

Fullness: TypeAlias = float
r"""A metric to quantify how much of the original source is missing. Higher is better.

Complementary to [Bleedless][splifft.types.Bleedless].
Measures the average energy of the parts of the [mel spectrogram][splifft.types.DbDifferenceMel]
that are quieter than the reference.
A high value indicates that parts of the target loss were lost during the separation, indicating
that more of the original source's character is preserved.
$$
\text{Fullness} = \text{mean}(|\mathbf{D}(m, k)|) \quad \forall \quad \mathbf{D}(m, k) < 0
$$
"""
