"""Command line interface for `splifft`."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.logging import RichHandler

from . import PATH_REGISTRY_DEFAULT

logging.basicConfig(level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()])
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

app = typer.Typer(
    pretty_exceptions_show_locals=False,
    help="A CLI for source separation.",
    no_args_is_help=True,
)

ConfigPath = Annotated[
    Path | None,
    typer.Option(
        "--config",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to the model's JSON configuration file.",
    ),
]
CheckpointPath = Annotated[
    Path | None,
    typer.Option(
        "--checkpoint",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to the model's `.pt` or `.ckpt` checkpoint file.",
    ),
]
RegistryPath = Annotated[
    Path,
    typer.Option(
        "--registry",
        "-r",
        help="Path to the model registry file.",
    ),
]


@app.command()
def run(
    mixture_path: Annotated[
        Path,
        typer.Argument(
            ...,
            exists=True,
            file_okay=True,
            dir_okay=True,
            readable=True,
            help="Path to the audio file to be separated.",
        ),
    ],
    config_path: ConfigPath = None,
    checkpoint_path: CheckpointPath = None,
    model_id: Annotated[
        str | None,
        typer.Option(
            "--model",
            "-m",
            help="Model ID from the registry to use.",
        ),
    ] = None,
    module_name: Annotated[
        str | None,
        typer.Option(
            "--module",
            help="Python module containing the model and configuration class.",
        ),
    ] = None,
    class_name: Annotated[
        str | None,
        typer.Option(
            "--class",
            help="Name of the model class to load from the module.",
        ),
    ] = None,
    package_name: Annotated[
        str | None,
        typer.Option(
            "--package",
            help=(
                "The package to use as the anchor point from which to resolve the relative import "
                "to an absolute import. This is only required when performing a relative import."
            ),
        ),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            file_okay=False,
            dir_okay=True,
            writable=True,
            help="Directory to save the separated audio stems.",
        ),
    ] = None,
    cpu: Annotated[
        bool, typer.Option("--cpu", help="Force processing on CPU, even if CUDA is available.")
    ] = False,
) -> None:
    """Run inference on an audio file to get its constituent stems or logits."""
    import numpy as np
    import torch
    from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn
    from torchcodec.encoders import AudioEncoder

    from .inference import ChunkProcessed, InferenceEngine, InferenceOutput

    use_cuda = not cpu and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    logger.info(f"using torch device: {device}")

    if model_id is not None:
        if config_path is not None or checkpoint_path is not None:
            raise typer.BadParameter("cannot specify --config or --checkpoint when using --model")
        logger.info(f"loading model '{model_id}' from registry")
        engine = InferenceEngine.from_registry(model_id, device=device)
    elif config_path is not None and checkpoint_path is not None:
        logger.info(f"loading inference engine with custom {config_path=} and {checkpoint_path=}")
        engine = InferenceEngine.from_pretrained(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            device=device,
            module_name=module_name,
            class_name=class_name,
            package_name=package_name,
        )
    else:
        raise typer.BadParameter("must specify either --model OR both --config and --checkpoint")

    mixture_paths = mixture_path.glob("*") if mixture_path.is_dir() else [mixture_path]
    for mixture_path in mixture_paths:
        logger.info(f"processing audio file: {mixture_path=}")
        output_results = None
        sample_rate = None
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            transient=True,
        ) as progress:
            task_id = progress.add_task("inference", total=None)
            for event in engine.stream(mixture_path):
                if isinstance(event, ChunkProcessed):
                    progress.update(
                        task_id,
                        description="inference",
                        total=event.total_batches,
                        completed=event.batch_index,
                    )
                elif isinstance(event, InferenceOutput):
                    output_results = event.outputs
                    sample_rate = event.sample_rate
                else:
                    progress.update(task_id, description=event.stage)
        if output_results is None:
            raise RuntimeError("inference finished without outputs")
        curr_output_dir = output_dir or Path("./data/audio/output") / mixture_path.stem
        curr_output_dir.mkdir(parents=True, exist_ok=True)

        for key, data in output_results.items():
            if engine.model_params_concrete.output_type == "logits":
                output_file = (curr_output_dir / key).with_suffix(".npy")
                np.save(output_file, data.cpu().float().numpy())
                logger.info(f"wrote logits '{key}' to {output_file}")
            else:
                if (
                    engine.config.output.stem_names != "all"
                    and key not in engine.config.output.stem_names
                ):
                    continue
                output_file = (curr_output_dir / key).with_suffix(
                    f".{engine.config.output.file_format}"
                )
                assert sample_rate is not None
                encoder = AudioEncoder(samples=data.cpu(), sample_rate=sample_rate)
                encoder.to_file(str(output_file), bit_rate=engine.config.output.bit_rate)
                logger.info(f"wrote stem '{key}' to {output_file}")


@app.command()
def pull(
    model_id: Annotated[str, typer.Argument(help="Model ID from the registry.")],
    force_overwrite: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help="Force overwrite existing configuration or model checkpoint if already present",
        ),
    ] = False,
    registry_path: RegistryPath = PATH_REGISTRY_DEFAULT,
) -> None:
    """Downloads a model checkpoint and configuration from the registry."""
    from .config import Registry
    from .io import get_model_paths

    model_paths = get_model_paths(
        model_id,
        fetch_if_missing=True,
        force_overwrite=force_overwrite,
        registry=Registry.from_file(registry_path),
    )
    if (p_ckpt := model_paths.path_checkpoint).exists():
        logger.info(f"checkpoint @ {p_ckpt} ({p_ckpt.stat().st_size / 1_000_000:.1f} M)")
    else:
        logger.error(f"expected model checkpoint to exist at {p_ckpt} for '{model_id}'")
    if (p_cfg := model_paths.path_config).exists():
        logger.info(f"config @ {p_cfg}")
    else:
        logger.error(f"expected model config to exist at {p_cfg} for '{model_id}'")


@app.command()
def ls(
    show_all: Annotated[
        bool,
        typer.Option(
            "--all", "-a", help="Show all available models, including not downloaded ones."
        ),
    ] = False,
    registry_path: RegistryPath = PATH_REGISTRY_DEFAULT,
) -> None:
    """List available models in the registry."""
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text

    from .config import Registry
    from .io import ModelSupportStatus, is_model_cached, is_model_supported

    registry = Registry.from_file(registry_path)

    table = Table(
        show_lines=False,
        pad_edge=False,
        box=None,
    )
    table.add_column("id", no_wrap=True)
    table.add_column("size", no_wrap=True, justify="right")
    table.add_column("created_at", no_wrap=True, justify="right")
    table.add_column("purpose", no_wrap=True)
    table.add_column("outputs", overflow="fold")

    count_shown = 0
    count_total = len(registry)

    for identifier, model in registry.items():
        is_cached = is_model_cached(identifier)
        support_status = is_model_supported(identifier, registry)
        if not show_all and not is_cached:
            continue

        count_shown += 1
        ident = Text(f"{identifier}")
        if support_status is ModelSupportStatus.MISSING:
            ident.stylize("dim strike")
        elif support_status is ModelSupportStatus.UNTESTED:
            ident.stylize("dim")
        download_indicator = "  "
        if is_cached:
            ident.stylize("green")
        elif model.model_size:
            download_indicator = "⤓ "
        size = (
            Text(
                f"{download_indicator}{model.model_size / 1_000_000:>6.1f}M",
                style="green" if is_cached else "",
            )
            if model.model_size
            else Text("?", style="dim")
        )
        created_at_date = (
            Text(datetime.fromisoformat(model.created_at).strftime("%Y-%m-%d"))
            if model.created_at
            else Text("?", style="dim")
        )
        outputs = ",".join(model.output) if model.output else "-"
        table.add_row(
            ident,
            size,
            created_at_date,
            model.purpose,
            outputs,
        )

    Console().print(table)
    if not show_all and count_shown == 0:
        logger.info(
            f"no downloaded models found (total {count_total} in registry). use `splifft ls -a` to see all."
        )


@app.command()
def rm(
    model_id: Annotated[str, typer.Argument(help="Model ID from the registry.")],
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Force deletion without confirmation")
    ] = False,
) -> None:
    """Delete a model from the local cache."""
    from .io import delete_model_from_cache, is_model_cached

    if not is_model_cached(model_id):
        logger.error(f"model '{model_id}' is not in cache")
        raise typer.Exit(1)

    if not force and not typer.confirm(f"are you sure you want to delete '{model_id}'?"):
        logger.info("aborted")
        return

    if delete_model_from_cache(model_id):
        logger.info(f"deleted '{model_id}'")
    else:
        logger.error(f"failed to delete '{model_id}'")
        raise typer.Exit(1)


@app.command()
def info(
    model_id: Annotated[str, typer.Argument(help="Model ID from the registry.")],
    config_path: ConfigPath = None,
    checkpoint_path: CheckpointPath = None,
    registry_path: RegistryPath = PATH_REGISTRY_DEFAULT,
    all_details: Annotated[
        bool,
        typer.Option(
            "-a",
            "--all-details",
            help="Whether to show list of parameters, shapes, dtypes and total params",
        ),
    ] = False,
) -> None:
    """Show details about a model."""
    from rich.console import Console
    from rich.text import Text

    from . import DIR_DATA
    from .config import Registry
    from .io import get_model_cache_dir

    if model_id is not None:
        if config_path is not None or checkpoint_path is not None:
            raise typer.BadParameter("cannot specify --config or --checkpoint when using model_id")

        registry = Registry.from_file(registry_path)
        if model_id not in registry:
            logger.error(f"model '{model_id}' not found in registry")
            raise typer.Exit(1)

        model_info = registry[model_id]
        console = Console()
        console.print(f"[bold]registry info for '{model_id}' @ {DIR_DATA / 'registry.json'}[/bold]")
        console.print_json(data=model_info.model_dump(exclude_none=True))

        cache_dir = get_model_cache_dir(model_id)
        path_config: Path | None = cache_dir / "config.json"
        path_ckpt: Path | None = cache_dir / "model.ckpt"
    else:
        if config_path is None and checkpoint_path is None:
            raise typer.BadParameter("must specify either model_id OR --config/--checkpoint")
        path_config = config_path
        path_ckpt = checkpoint_path

    if path_config and path_config.exists():
        console = Console()
        console.print(f"\n[bold]configuration @ {path_config}[/bold]")
        console.print_json(path_config.read_text(), indent=2)

    if path_ckpt and path_ckpt.exists():
        console = Console()
        size_mb = path_ckpt.stat().st_size / (1024 * 1024)
        console.print(f"\n[bold]checkpoint @ {path_ckpt} ({size_mb:.2f} M)[/bold]")
        if not all_details:
            return
        try:
            import torch
            from rich.table import Table

            checkpoint = torch.load(path_ckpt, map_location="cpu", weights_only=True)
            state_dict: dict[str, Any] = (
                checkpoint.get("state_dict", checkpoint)
                if isinstance(checkpoint, dict)
                else checkpoint
            )

            table = Table(box=None, header_style="bold", padding=(0, 2))
            table.add_column("key", style="cyan")
            table.add_column("type", justify="left")

            total_params = 0
            for k, v in state_dict.items():
                if not isinstance(v, torch.Tensor):
                    continue
                table.add_row(
                    k,
                    Text(str(v.dtype).replace("torch.", "")).append_text(
                        Text(str(list(v.shape)) if v.shape else "", style="magenta")
                    ),
                )
                total_params += v.numel()

            console.print("\n[bold]state dict[/bold]")
            console.print(table)
            console.print(f"total parameters: {total_params:,}")
            console.print(f"total keys: {len(state_dict)}")
        except Exception as e:
            # possible that file is a zip, so in that case show a better error message
            console.print(f"[red]failed to load checkpoint info: {e}[/red]")


@app.command()
def debug() -> None:
    """Prints detailed information about the environment, dependencies, and hardware
    for debugging purposes."""
    import sys

    logger.info(f"{sys.version=}")
    logger.info(f"{sys.executable=}")
    logger.info(f"{sys.platform=}")
    import platform

    logger.info(f"{platform.system()=} ({platform.release()})")
    logger.info(f"{platform.machine()=}")
    import torch

    logger.info(f"{torch.__version__=}")
    logger.info(f"{torch.cuda.is_available()=}")
    if torch.cuda.is_available():
        logger.info(f"{torch.cuda.device_count()=}")
        logger.info(f"{torch.cuda.current_device()=}")
        device = torch.cuda.current_device()
        logger.info(f"{torch.cuda.get_device_name(device)=}")
        logger.info(f"{torch.cuda.get_device_capability(device)=}")
        logger.info(f"{torch.cuda.get_device_properties(device)=}")
    import torchaudio

    logger.info(f"{torchaudio.__version__=}")


def version_callback(value: bool) -> None:
    if value:
        import importlib.metadata

        version = importlib.metadata.version("splifft")
        typer.echo(f"splifft {version}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=version_callback,
            is_eager=True,
            help="Show version information and exit.",
        ),
    ] = False,
) -> None:
    pass


if __name__ == "__main__":
    app()
