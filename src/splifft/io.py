"""Operations for reading and writing to disk, and network IO.

All side effects should go here."""

from __future__ import annotations

import difflib
import hashlib
import io
import logging
import os
import shutil
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Callable, NoReturn

import torch
from torchcodec.decoders import AudioDecoder

from . import DIR_CONFIG_DEFAULT  # TODO don't hardcode
from . import types as t
from .core import Audio

if TYPE_CHECKING:
    from .config import Registry
    from .models import ModelT

logger = logging.getLogger(__name__)


def _raise_missing_feature(*, extra: str, feature: str) -> NoReturn:  # perhaps move this to utils?
    raise ImportError(
        f"error: the '{feature}' feature requires the '{extra}' extra.\n"
        f"help: install with: 'splifft[{extra}]'\n"
    )


def read_audio(
    file: str | Path | io.RawIOBase | io.BufferedReader | bytes,
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
    checkpoint_file: torch.types.FileLike,
    device: torch.device | str,
    *,
    strict: bool = False,
    state_dict_transform: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]]
    | None = None,
) -> ModelT:
    """Load the weights from a checkpoint into the given model.

    Handles standard PyTorch checkpoints and PyTorch Lightning checkpoints (stripping `model.` prefix).
    """

    loaded_obj: object = torch.load(checkpoint_file, map_location=device, weights_only=True)
    if isinstance(loaded_obj, dict) and "state_dict" in loaded_obj:
        loaded_obj = loaded_obj["state_dict"]
    if not isinstance(loaded_obj, dict):
        raise TypeError(f"expected checkpoint dict, got {type(loaded_obj).__name__}")

    state_dict: dict[str, torch.Tensor] = {}
    for key, value in loaded_obj.items():
        if not isinstance(value, torch.Tensor):
            continue
        state_dict[key] = value

    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            new_state_dict[key[6:]] = value
        else:
            new_state_dict[key] = value
    state_dict = new_state_dict

    if state_dict_transform is not None:
        state_dict = state_dict_transform(state_dict)

    # TODO: DataParallel and `module.` prefix
    model.load_state_dict(state_dict, strict=strict)
    # NOTE: do not torch.compile here!

    return model.to(device)


#
# registry: caching and downloading
#


def get_model_cache_dir(model_id: str) -> Path:
    try:
        import platformdirs  # noqa: F401
        import pydantic  # noqa: F401
    except ImportError:
        _raise_missing_feature(extra="config", feature="caching")
    from platformdirs import user_cache_dir

    cache_dir = Path(user_cache_dir("splifft", appauthor=False)) / model_id
    return cache_dir


# assuming there is only one model file and not in separate parts
def is_model_cached(model_id: str) -> bool:
    """Checks if the model's config and checkpoint exist in the cache."""
    # NOTE: not validating the hash for speed
    try:
        cache_dir = get_model_cache_dir(model_id)
        return (cache_dir / "config.json").exists() and (cache_dir / "model.ckpt").exists()
    except ImportError:
        return False


class ModelSupportStatus(Enum):
    MISSING = "missing"
    UNTESTED = "untested"
    AVAILABLE = "available"


def is_model_supported(model_id: str, registry: Registry) -> ModelSupportStatus:
    """Return support status based on the presence of a default config in package data."""
    if model_id not in registry:
        return ModelSupportStatus.MISSING

    if not (config_id := registry[model_id].config_id):
        return ModelSupportStatus.MISSING

    if not (DIR_CONFIG_DEFAULT / f"{config_id}.json").exists():  # TODO dont hardcode
        return ModelSupportStatus.MISSING

    if config_id.startswith("."):
        return ModelSupportStatus.UNTESTED

    return ModelSupportStatus.AVAILABLE


def delete_model_from_cache(model_id: str) -> bool:
    try:
        if (cache_dir := get_model_cache_dir(model_id)).exists():
            shutil.rmtree(cache_dir)
            return True
        return False
    except ImportError:
        return False


@dataclass(frozen=True)
class LocalModelPaths:
    path_config: Path = field(kw_only=True)
    path_checkpoint: Path = field(kw_only=True)


def get_model_paths(
    model_id: str,
    *,
    fetch_if_missing: bool = False,
    force_overwrite_config: bool = False,
    force_overwrite_model: bool = False,
    registry: Registry,
) -> LocalModelPaths:
    if model_id not in registry:
        matches = difflib.get_close_matches(model_id, list(registry))
        suggestions = "\n".join(map(lambda m: f"- {m!r}", matches))
        suggestion = f" did you mean:\n{suggestions}\n" if matches else ""
        raise ValueError(
            f"model '{model_id}' not found in registry.{suggestion}\n"
            "help: use `splifft ls` to see downloaded models in the registry"
        )

    model_info = registry[model_id]
    cache_dir = get_model_cache_dir(model_id)
    cached_config = cache_dir / "config.json"
    cached_ckpt = cache_dir / "model.ckpt"

    is_config_present = cached_config.exists()
    is_ckpt_present = cached_ckpt.exists()

    if force_overwrite_config or not is_config_present:
        if not model_info.config_id:
            raise ValueError(
                f"model '{model_id}' does not specify a default configuration identifier.\n"
                "help: you must provide a config file manually."
            )

        if not (source_config := DIR_CONFIG_DEFAULT / f"{model_info.config_id}.json").exists():
            raise FileNotFoundError(
                f"default config '{model_info.config_id}.json' not found in package data ({DIR_CONFIG_DEFAULT}).\n"
                "help: the registry entry should point to a config that exists"
            )

        cache_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_config, cached_config)
        logger.info(f"wrote config for '{model_id}' at {cached_config}")

    if force_overwrite_model or (not is_ckpt_present and fetch_if_missing):
        # only download the first? or try the second if not?
        if not (
            ckpt_resource := next((r for r in model_info.resources if r.kind == "model_ckpt"), None)
        ):
            raise ValueError(f"model '{model_id}' has no `model_ckpt` resource URL in registry")

        logger.info(f"pulling weights for '{model_id}'")
        download_file(
            ckpt_resource.url,
            cached_ckpt,
            expected_digest=ckpt_resource.digest,
        )

    return LocalModelPaths(path_config=cached_config, path_checkpoint=cached_ckpt)


def download_file(url: str, dest: Path, expected_digest: str | None = None) -> None:
    try:
        import httpx  # noqa: F401
    except ImportError:
        _raise_missing_feature(extra="web", feature="download")

    try:
        from rich.progress import (
            BarColumn,
            DownloadColumn,
            Progress,
            TextColumn,
            TimeRemainingColumn,
            TransferSpeedColumn,
        )

        rich_progress = Progress(
            TextColumn("downloading [bold blue]{task.fields[filename]}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            transient=True,
        )
    except ImportError:
        rich_progress = None

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest_tmp = dest.with_suffix(".tmp")

    if expected_digest is None:
        logger.warning(f"no digest found for {url}, file integrity will not be verified.")
    else:
        if not expected_digest.startswith("sha256:"):
            logger.warning(f"unsupported digest format: {expected_digest}, skipping verification.")
            expected_digest = None

    hasher = hashlib.sha256() if expected_digest else None

    try:
        # TODO hoist httpx client up. and aiolimiter?
        with httpx.Client(http2=True, follow_redirects=True) as client:
            with client.stream("GET", url) as response:
                response.raise_for_status()
                content_length = response.headers.get("content-length")
                total = int(content_length) if content_length is not None else None

                def download(*, callback: Callable[[bytes], None]) -> None:
                    with open(dest_tmp, "wb") as f:
                        for chunk in response.iter_bytes():
                            f.write(chunk)
                            if hasher is not None:
                                hasher.update(chunk)
                            callback(chunk)

                if rich_progress is None:
                    logger.info(f"downloading {dest.name} ({total or 'unknown'} bytes)...")
                    download(callback=lambda *args: None)
                else:
                    with rich_progress as p:
                        task = p.add_task("download", filename=dest.name, total=total)
                        download(callback=lambda chunk: p.update(task, advance=len(chunk)))

        if expected_digest and hasher:
            if (actual_digest := f"sha256:{hasher.hexdigest()}") != expected_digest:
                raise RuntimeError(
                    f"digest mismatch for {url}:\n"
                    f"  expected: '{expected_digest}'\n"
                    f"  actual:   '{actual_digest}'"
                )
            else:
                logger.info(f"verified digest '{expected_digest}'")

        dest_tmp.replace(dest)  # atomic to ensure we dont have corrupted files if interrupted
    except Exception as e:
        if dest_tmp.exists():
            os.remove(dest_tmp)
        raise RuntimeError(f"failed to download {url}: {e}") from e
