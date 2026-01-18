# SpliFFT

[![image](https://img.shields.io/pypi/v/splifft)](https://pypi.python.org/pypi/splifft)
[![image](https://img.shields.io/pypi/l/splifft)](https://pypi.python.org/pypi/splifft)
[![image](https://img.shields.io/pypi/pyversions/splifft)](https://pypi.python.org/pypi/splifft)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![MkDocs](https://shields.io/badge/MkDocs-documentation-informational)](https://undef13.github.io/splifft/)

Lightweight utilities for music source separation.

This library is a ground-up rewrite of the [zfturbo's MSST repo](https://github.com/ZFTurbo/Music-Source-Separation-Training), with a strong focus on robustness, simplicity and extensibility. While it is a fantastic collection of models and training scripts, this rewrite adopts a different architecture to address common pain points in research code.

Key principles:

- **Configuration as code**: pydantic models are used instead of untyped dictionaries or `ConfigDict`. this provides static type safety, runtime data validation, IDE autocompletion, and a single, clear source of truth for all parameters.
- **Data-oriented and functional core**: complex class hierarchies and inheritance are avoided. the codebase is built on plain data structures (like `dataclasses`) and pure, stateless functions.
- **Semantic typing as documentation**: we leverage Python's type system to convey intent. types like `RawAudioTensor` vs. `NormalizedAudioTensor` make function signatures self-documenting, reducing the need for verbose comments and ensuring correctness.
- **Extensibility without modification**: new models can be integrated from external packages without altering the core library. the dynamic model loading system allows easy plug-and-play adhering to the open/closed principle.

⚠️ This is pre-alpha software, expect significant breaking changes.

## Features and Roadmap

**Short term (high priority)**

- [x] a robust, typed JSON configuration system powered by `pydantic`
- [x] inferencing:
    - [x] normalization and denormalization
    - [x] chunk generation: vectorized with `unfold`
    - [x] chunk stitching: vectorized overlap-add with `fold`
    - [x] flexible ruleset for stem deriving: add/subtract model outputs or any intermediate output (e.g., creating an `instrumental` track by subtracting `vocals` from the `mixture`).
- [x] web-based docs: generated with `mkdocs` with excellent crossrefs.
- [x] simple CLI for inferencing on a directory of audio files
- [ ] `BS-Roformer`: ensure bit-for-bit equivalence in pytorch and strive for max perf.
    - [x] initial fp16 support
    - [ ] support `coremltools` and `torch.compile`
        - [x] handroll complex multiplication implementation
        - [x] handroll stft in forward pass
- [x] port additional SOTA models from MSST (e.g. Mel Roformer, SCNet)
    - [x] directly support popular models (e.g. by [@unwa](https://huggingface.co/pcunwa), [gabox](https://huggingface.co/GaboxR67), by [@becruily](https://huggingface.co/becruily))
- [ ] model registry with simple file-based cache
- [ ] evals: SDR, bleedless, fullness, etc.
- [ ] proper benchmarking (MFU, memory...)
- [ ] datasets: MUSDB18-HQ, moises

**Medium term**

- [ ] simple web-based GUI with FastAPI and SolidJS.
- [ ] Jupyter notebook

**Long term (low priority)**

- [ ] data augmentation
- [ ] implement a complete, configurable training loop
- [ ] [`max` kernels](#mojo)

**Contributing**: PRs are very welcome!

## Installation & Usage

- [I just want to run it](#cli)
- [I want to add it as a library to my Python project](#library)
- [I want to hack around](#development)

Documentation on the config (amongst other details) can be found [here](https://undef13.github.io/splifft/config/)

### CLI

There are three steps. You do not need to have Python installed.

1. Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/) if you haven't already. It is an awesome Python package and library manager with pip comptability.

    ```sh
    # Linux / MacOS
    wget -qO- https://astral.sh/uv/install.sh | sh
    # Windows
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

2. Open a new terminal and install the latest stable PyPI release as a [tool](https://docs.astral.sh/uv/concepts/tools/). It will install the Python interpreter, all necessary packages and add the `splifft` executable to your `PATH`:

    ```sh
    uv tool install "splifft[config,inference,cli]"
    ```

    <details>
      <summary>I want the latest bleeding-edge version</summary>

    This directly pulls from the `main` branch, which may be unstable:

    ```sh
    uv tool install "git+https://github.com/undef13/splifft.git[config,inference,cli]"
    ```

    </details>

3. Go into a new directory and place the [model checkpoint](https://github.com/undef13/splifft/releases/download/v0.0.1/roformer-fp16.pt) and [configuration](https://raw.githubusercontent.com/undef13/splifft/refs/heads/main/data/config/bs_roformer.json) inside it. Assuming your current directory has this structure (doesn't have to be exactly this):

    <details>
      <summary>Minimal reproduction: with example audio from YouTube</summary>

    ```sh
    uv tool install yt-dlp
    yt-dlp -f bestaudio -o data/audio/input/3BFTio5296w.flac 3BFTio5296w
    wget -P data/models/ https://huggingface.co/undef13/splifft/resolve/main/roformer-fp16.pt?download=true
    wget -P data/config/ https://raw.githubusercontent.com/undef13/splifft/refs/heads/main/data/config/bs_roformer.json
    ```

    </details>

    ```txt
    .
    └── data
        ├── audio
        │   ├── input
        │   │   └── 3BFTio5296w.flac
        │   └── output
        ├── config
        │   └── bs_roformer.json
        └── models
            └── roformer-fp16.pt
    ```

    Run:

    ```sh
    splifft separate data/audio/input/3BFTio5296w.flac --config data/config/bs_roformer.json --checkpoint data/models/roformer-fp16.pt
    ```

    <details>
      <summary>Console output</summary>

    ```php
    [00:00:41] INFO     using device=device(type='cuda')                                                 __main__.py:111
               INFO     loading configuration from                                                       __main__.py:113
                        config_path=PosixPath('data/config/bs_roformer.json')                                           
               INFO     loading model metadata `BSRoformer` from module `splifft.models.bs_roformer`     __main__.py:126
    [00:00:42] INFO     loading weights from checkpoint_path=PosixPath('data/models/roformer-fp16.pt')   __main__.py:127
               INFO     processing audio file:                                                           __main__.py:135
                        mixture_path=PosixPath('data/audio/input/3BFTio5296w.flac')                                     
    ⠙ processing chunks... ━━━━━━━━━━╺━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  25% 0:00:10 (bs=4 • cuda • float16)
    [00:00:56] INFO     wrote stem `bass` to data/audio/output/3BFTio5296w/bass.flac                     __main__.py:158
               INFO     wrote stem `drums` to data/audio/output/3BFTio5296w/drums.flac                   __main__.py:158
               INFO     wrote stem `other` to data/audio/output/3BFTio5296w/other.flac                   __main__.py:158
    [00:00:57] INFO     wrote stem `vocals` to data/audio/output/3BFTio5296w/vocals.flac                 __main__.py:158
               INFO     wrote stem `guitar` to data/audio/output/3BFTio5296w/guitar.flac                 __main__.py:158
               INFO     wrote stem `piano` to data/audio/output/3BFTio5296w/piano.flac                   __main__.py:158
    [00:00:58] INFO     wrote stem `instrumental` to data/audio/output/3BFTio5296w/instrumental.flac     __main__.py:158
               INFO     wrote stem `drums_and_bass` to data/audio/output/3BFTio5296w/drums_and_bass.flac __main__.py:158
    ```

    </details>

    To update the tool:

    ```sh
    uv tool upgrade splifft --force-reinstall
    ```

### Library

Add `splifft` to your project:

```sh
# latest pypi version
uv add splifft
# latest bleeding edge
uv add git+https://github.com/undef13/splifft.git
```

This will install the absolutely minimal core dependencies used under the `src/splifft/models` directory. Higher level components, e.g. inference, training or CLI components **must** be installed via optional depedencies, as specified in the [`project.optional-dependencies` section of `pyproject.toml`](https://github.com/undef13/splifft/blob/main/pyproject.toml), for example:

```sh
# enable the built-in configuration, inference and CLI
uv add "splifft[config,inference,cli]"
```

This will install `splifft` in your venv.

### Development

If you'd like to make local changes, it is recommended to enable all optional and developer group dependencies:

```sh
git clone https://github.com/undef13/splifft.git
cd splifft
uv venv
uv sync --all-extras --all-groups
```

You may also want to use `--editable` with `sync`. Check your code:

```sh
# lint & format
just fmt
# build & host documentation
just docs
```

Format your code:

```sh
just fmt
```

This repo is no longer compatible with zfturbo's repo. The last version that does so is [`v0.0.1`](https://github.com/undef13/splifft/tree/v0.0.1). To pin a specific version in `uv`, change your `pyproject.toml`:

```toml
[tool.uv.sources]
splifft = { git = "https://github.com/undef13/splifft.git", rev = "287235e520f3bb927b58f9f53749fe3ccc248fac" }
```

## Mojo

While the primary goal is just to have minimalist PyTorch-based inference engine, I will be using this project as an opportunity to learn more about heterogenous computing, particularly with the [Mojo language](https://docs.modular.com/mojo/why-mojo/). The ultimate goal will be to understand to what extent can its compile-time metaprogramming and explicit memory layout control be used.

My approach will be incremental and bottom-up: I'll develop, test and benchmark small components against their PyTorch counterparts. The PyTorch implementation will **always** remain the "source of truth", the fully functional baseline and never be removed.

TODO:

- [ ] evaluate `pixi` in `pyproject.toml`.
- [ ] use `max.torch.CustomOpLibrary` to provide a callable from the pytorch side
- [ ] use [`DeviceContext`](https://github.com/modular/modular/blob/main/mojo/stdlib/stdlib/gpu/host/device_context.mojo) to interact with the GPU
- [ ] [attention](https://github.com/modular/modular/blob/main/examples/custom_ops/kernels/fused_attention.mojo)
    - [ ] use [`LayoutTensor`](https://github.com/modular/modular/blob/main/max/kernels/src/layout/layout_tensor.mojo) for QKV
- [ ] rotary embedding
- [ ] feedforward
- [ ] transformer
- [ ] `BandSplit` & `MaskEstimator`
- [ ] full graph compilation
