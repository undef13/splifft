## Introduction

In the digital world, sound is captured as a discrete sequence of samples, a representation of the original continuous audio signal $x(t)$. We refer to this time-domain data as a [`RawAudioTensor`][splifft.types.RawAudioTensor]. This digital signal is defined by several key parameters: its [sample rate][splifft.types.SampleRate], number of [channels][splifft.types.Channels], [bit rate][splifft.types.BitRate] and [file format][splifft.types.FileFormat].

The full range of human hearing is approximately 20-20000 Hz, so according to the [Nyquist-Shannon sampling theorem](https://en.wikipedia.org/wiki/Nyquist%E2%80%93Shannon_sampling_theorem), the minimum [sample rate][splifft.types.SampleRate] to accurately represent this range is 40000 Hz. Common values are 44100 Hz (CD quality), 48000 Hz (professional audio), and 16000 Hz (voice).

## The Separation Pipeline

### Normalization

Neural networks perform best when their input data has a consistent statistical distribution. To prepare the audio for the model, we first [normalize it][splifft.core.normalize_audio]. This process transforms the [`RawAudioTensor`][splifft.types.RawAudioTensor] into a [`NormalizedAudioTensor`][splifft.types.NormalizedAudioTensor] with a mean of 0 and a standard deviation of 1. The original statistics ($\mu$ and $\sigma$) are stored in [`NormalizationStats`][splifft.core.NormalizationStats] to be used later for denormalization.

### Time-Frequency Transformation

Models usually operates not on raw time-domain samples, but in the time-frequency domain, which reveals the signal's frequency content over time. This is achieved via the [Short-Time Fourier Transform (STFT)](https://en.wikipedia.org/wiki/Short-time_Fourier_transform), which converts the 1D audio signal into a 2D [complex spectrogram][splifft.types.ComplexSpectrogram].

#### Complex Spectrogram

The [STFT coefficient][splifft.types.ComplexSpectrogram] $X[m, k]$ is a complex number that can be decomposed into:

- Magnitude $|X[m, k]|$: Tells us "how much" of a frequency is present (i.e., its loudness).
- Phase $\phi(m, k)$: Tells us "how it's aligned" in time. This is notoriously difficult to model, as it chaotically wraps around from $-\pi$ to $\pi$. Human hearing is highly sensitive to phase, which is crucial for sound localization and timbre perception.

Practically, the process involves:

1. Dividing the audio signal into short, overlapping segments in time (chunks), parameterized by the
   [hop size][splifft.types.HopSize] $H$
2. Applying a [window function][splifft.types.WindowShape] $w[n]$ (e.g.
   [Hann window][torch.hann_window]) to each chunk to reduce [spectral leakage](https://en.wikipedia.org/wiki/Spectral_leakage)
3. Computing the Fast Fourier Transform (FFT) on each windowed segment to get its complex frequency
   spectrum. The [FFT size][splifft.types.FftSize] $N_\text{fft}$ determines the number of frequency
   bins.
4. Stacking these spectra to form the 2D complex spectrogram.

This is commonly used as the input to models. The objective of source separation is to
approximate an ideal ratio mask or its complex equivalent:
$\hat{S}_\text{source} = M_\text{complex} \odot S_\text{mixture}$.

#### FFT Size

The choice of [`FftSize`][splifft.types.FftSize] presents a fundamental [trade-off](https://en.wikipedia.org/wiki/Uncertainty_principle#Signal_processing) between the uncertainty in time $t$ and frequency $f$: $\sigma_t \sigma_f \ge \frac{1}{4\pi}$

- a short window gives good time resolution, excellent for capturing sharp, percussive sounds (transients).
- a long window gives good frequency resolution, ideal for separating fine harmonics of tonal instruments.

To address this, some loss functions (e.g. `auraloss.MultiResolutionSTFTLoss`) calculate the error on spectrograms with multiple FFT sizes, forcing the model to optimize for both transient and tonal accuracy.

#### Bands

Instead of processing every frequency bin independently, we can group them into [`Bands`][splifft.types.Bands]. This reduces computational complexity and allows the model to learn relationships within frequency regions, which often correspond to musical harmonics. Some models use perceptually-motivated scales like the [Mel scale](https://en.wikipedia.org/wiki/Mel_scale), while others like [BS-Roformer][splifft.models.bs_roformer] use a linear frequency scale and learn their own relevant bandings.

### Chunking and Inference

Since a full audio track is too large for GPU memory, we process it in overlapping segments. The [`ChunkSize`][splifft.types.ChunkSize] defines the segment length, while the [`HopSize`][splifft.types.HopSize] dictates the step between them, controlled by the [`OverlapRatio`][splifft.types.OverlapRatio]. This process yields a stream of [`PaddedChunkedAudioTensor`][splifft.types.PaddedChunkedAudioTensor] batches, which are fed into the model. The model then outputs a corresponding stream of [`SeparatedChunkedTensor`][splifft.types.SeparatedChunkedTensor].

### Stitching and Post-processing

After the model processes each chunk, we must reconstruct the full-length audio. The [`stitch_chunks`][splifft.core.stitch_chunks] function implements the overlap-add method, which applies a [`WindowTensor`][splifft.types.WindowTensor] to each chunk to ensure an artifact-free reconstruction. The final result is a [`RawSeparatedTensor`][splifft.types.RawSeparatedTensor] for each separated stem.

With the separated audio back in the time domain, the final steps are to reverse the normalization using the original [`NormalizationStats`][splifft.core.NormalizationStats] and, optionally, to create new stems (e.g., an "instrumental" track) using rules defined in [`DerivedStemsConfig`][splifft.config.DerivedStemsConfig].
