# SpliFFT

[![image](https://img.shields.io/pypi/v/splifft)](https://pypi.python.org/pypi/splifft)
[![image](https://img.shields.io/pypi/l/splifft)](https://pypi.python.org/pypi/splifft)
[![image](https://img.shields.io/pypi/pyversions/splifft)](https://pypi.python.org/pypi/splifft)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![MkDocs](https://shields.io/badge/MkDocs-documentation-informational)](https://undef13.github.io/splifft/)

Lightweight utilities for music source separation and transcription.

This library is a ground-up rewrite of the [zfturbo's MSST repo](https://github.com/ZFTurbo/Music-Source-Separation-Training), with a strong focus on robustness, simplicity and extensibility. We keep third-party dependencies to an absolute minimum to ease installation.

⚠️ This is pre-alpha software, expect significant breaking changes before v0.1.

## Supported Models

- [BS-Roformer](https://arxiv.org/abs/2309.02612) (including unwa's [HyperACE v1 and v2](https://huggingface.co/pcunwa/BS-Roformer-HyperACE), [Large Inst](https://huggingface.co/pcunwa/BS-Roformer-Large-Inst) modifications)
- [Mel-Roformer](https://arxiv.org/abs/2409.04702)
- [MDX23C TFC-TDF v3](https://arxiv.org/pdf/2306.09382)
- [beat this!](https://arxiv.org/abs/2407.21658) for beat tracking without DBN postprocessing
- [PESTO](https://arxiv.org/abs/2508.01488) for monophonic pitch estimation
- [basic pitch](https://arxiv.org/abs/2203.09893) for polyphonic pitch estimation (only frame-level onset, multipitch and posteriorgrams, no MIDI postprocessing)

Our default registry supports 110+ community-trained separation models.

## Installation & Usage

- [I just want to run it »](#cli)
- [I want to add it as a library to my Python project »](#library)
- [I want to contribute »](#development)

More information about models and config can be found on the [documentation](https://undef13.github.io/splifft/).

### CLI

There are three steps. You do not need to have Python installed.

1. Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/) if you haven't already. It is an awesome Python package and library manager with pip compatibility.

    ```sh
    # Linux / MacOS
    wget -qO- https://astral.sh/uv/install.sh | sh
    # Windows
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

2. Open a new terminal and install the latest stable PyPI release as a [tool](https://docs.astral.sh/uv/concepts/tools/). It will install the Python interpreter, all necessary packages and add the `splifft` executable to your `PATH`:

    ```sh
    uv tool install "splifft[config,inference,cli,web]"
    ```

    <details>
      <summary>Explanation of feature flags</summary>

      The core is kept as minimal as possible. Pick which ones you need:

      - The `config` extra is used to parse the model configuration from JSON and discover the registry's default cache dir.
      - The `inference` extra is used to decode audio formats.
      - The `cli` extra provides you with the `splifft` command line tool
      - The `web` extra is used to download models.
    </details>

    <details>
      <summary>I want the latest bleeding-edge version</summary>

    This directly pulls from the `main` branch, which may be unstable:

    ```sh
    uv tool install "git+https://github.com/undef13/splifft.git[config,inference,cli,web]"
    ```

    </details>

3. We recommend using our built-in registry to manage model config and weights:

    ```sh
    # list all available models, including those not yet available locally
    splifft ls -a

    # download model files and config to your user cache directory
    # ~/.cache/splifft on linux
    splifft pull bs_roformer-fruit-sw

    # view information about the configuration
    # modify the configuration, such as batch size according to your hardware
    splifft info bs_roformer-fruit-sw

    # run inference
    splifft run data/audio/input/3BFTio5296w.flac --model bs_roformer-fruit-sw
    ```

    Alternatively, for custom models, you can manage files manually. Go into a new directory and place the [model checkpoint](https://github.com/undef13/splifft/releases/download/v0.0.1/roformer-fp16.pt) and [configuration](https://raw.githubusercontent.com/undef13/splifft/refs/heads/main/data/config/bs_roformer.json) inside it. Assuming your current directory has this structure (doesn't have to be exactly this):

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
    splifft run data/audio/input/3BFTio5296w.flac --config data/config/bs_roformer.json --checkpoint data/models/roformer-fp16.pt
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
    [00:00:58] INFO     wrote stem `instrum` to data/audio/output/3BFTio5296w/instrum.flac               __main__.py:158
               INFO     wrote stem `drums_and_bass` to data/audio/output/3BFTio5296w/drums_and_bass.flac __main__.py:158
    ```

    </details>

    To update the tool:

    ```sh
    uv tool upgrade splifft --force-reinstall
    ```

## FAQ

> Where is my `config.json`, and which one is actually used?

Think of two locations:

1. **Built-in templates** bundled in the installed package ([`src/splifft/data/config/*.json`](https://github.com/undef13/splifft/tree/main/src/splifft/data/config) in this repo)
2. **Your editable copy** after `splifft pull {model_id}`

`splifft run --model {model_id}` uses your cached copy:

- Linux: `~/.cache/splifft/{model_id}/config.json`
- macOS: `~/Library/Caches/splifft/{model_id}/config.json`
- Windows: `%LOCALAPPDATA%\splifft\Cache\{model_id}\config.json`

> What is the difference between `--override-config` and editing `config.json`?

- `--override-config "inference.batch_size=2"` is **temporary** for that command only.
- editing `config.json` is **persistent** for all future runs.

Use overrides to experiment quickly, then copy stable values into your config.

> I hit `CUDA out of memory`.

Reduce memory pressure first:

```sh
splifft run --override-config "inference.batch_size=2"
```

Then, if you have a GPU and want to use fp16 mixed precision:

```sh
splifft run \
    --override-config "inference.batch_size=2" \
    --override-config 'inference.use_autocast_dtype="float16"'
```

> I only want some outputs (for example one stem).

Modify `inference.requested_stems` in the `config.json` or:

```sh
splifft run \
    --model bs_roformer-fruit-sw \
    --override-config 'inference.requested_stems=["piano"]'
```

> My config suddenly fails validation after an upgrade.

Your cached config may be from an older schema. If you want the latest preset config without redownloading checkpoint weights:

```sh
splifft pull --force-overwrite-config bs_roformer-fruit-sw
```

Note that this discards your previous changes!

> Where do I find the config contract?

- API docs: [`splifft.config.Config`](https://undef13.github.io/splifft/api/config/#splifft.config.Config)
- JSON schema: [`src/splifft/data/config.schema.json`](https://github.com/undef13/splifft/blob/main/src/splifft/data/config.schema.json)

For example, runtime batch size is `inference.batch_size`.

> How do I derive custom outputs (e.g. drumless)?

Use `derived_stems` in config (they will be executed in the order you define it), for example:

```jsonc
{
    // ...
    "derived_stems": {
        "drumless": {
            "operation": "subtract",
            "stem_name": "vocals",
            "by_stem_name": "mixture"
        },
        "drums_and_bass": {
            "operation": "add",
            "stem_names": ["drums", "bass"]
        }
    }
}
```

### Library

Add `splifft` to your project:

```sh
# latest pypi version
uv add splifft
# latest bleeding edge
uv add git+https://github.com/undef13/splifft.git
```

This will install the absolutely minimal core dependencies used under the `src/splifft/models` directory. Higher level components, e.g. inference, training or CLI components **must** be installed via optional dependencies, as specified in the [`project.optional-dependencies` section of `pyproject.toml`](https://github.com/undef13/splifft/blob/main/pyproject.toml), for example:

```sh
# enable the built-in configuration, inference and CLI
uv add "splifft[config,inference,cli,web]"
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

Code style:

- Use stdlib dataclasses or pydantic BaseModels instead of untyped dictionaries or `ConfigDict`. This provides static type safety, runtime data validation, IDE autocompletion, and a single, clear source of truth for all parameters.
- Avoid complex class hierarchies and inheritance. Use plain data structures and pure, stateless functions.
- Leverage Python's type system and our built-in types (e.g. `splifft.types.ChunkSize`) to convey intent. It reduces the needs of verbose docstrings.
- The core should remain agnostic and not contain any model-specific code other than high-level pre/postprocessing archetypes.

PRs are very welcome!

#### Registry

- Source of truth: `src/splifft/data/registry.json`
- Per-model runtime config: `src/splifft/data/config/{config_id}.json`
- JSON Schema are generated with `uv run scripts/gen_schema.py`.
- Validation gate: pydantic (`Registry.from_file`, `Config.from_file`)

If you would like to add a model to the `splifft` registry:

- upload checkpoint (ideally to huggingface), with optional MSST config
- add registry entry: `architecture`, `purpose`, `config_id`, `output`, `resources[]`
- write your own config JSON under `data/config`, or auto-convert your MSST yaml with `uv run scripts/community.py fix-registry-with-msst`
- optionally, run `uv run scripts/community.py fix-registry` to auto generate the `created_at` /`model_size`/`digest` fields using HF/GH metadata and sync outputs from configs.
- format registry JSON: `pnpm run fmt:json src/splifft/data/registry.json`

Right now, registry + configs are shipped in the package itself, with new model visibility inherently tied to package release/version bump. In the future, we may add a `splifft update` command.

## Roadmap

`splifft` is currently optimised for inferencing and does not yet support training.

Near term:

- `torch.jit.script`
- ONNX export
- `coremltools`
- support streaming with ring buffer
- simple web-based GUI with FastAPI and SolidJS.
- Jupyter notebook

Long term:

- evals: SDR, bleedless, fullness, etc.
- datasets: MUSDB18-HQ, moises
- implement a complete, configurable training loop
- data augmentation

<!-- holding off for now until Mojo reaches 1.0

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
-->