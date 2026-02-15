"""Command line interface for `splifft`."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.logging import RichHandler

from splifft import PATH_DATA

logging.basicConfig(level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()])
logger = logging.getLogger(__name__)

app = typer.Typer(
    pretty_exceptions_show_locals=False,
    help="A CLI for source separation.",
    no_args_is_help=True,
)


@app.command()
def separate(
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
    config_path: Annotated[
        Path,
        typer.Option(
            "--config",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Path to the model's JSON configuration file.",
        ),
    ],
    checkpoint_path: Annotated[
        Path,
        typer.Option(
            "--checkpoint",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Path to the model's `.pt` or `.ckpt` checkpoint file.",
        ),
    ],
    module_name: Annotated[
        Optional[str],
        typer.Option(
            "--module",
            help="Python module containing the model and configuration class.",
        ),
    ] = None,
    class_name: Annotated[
        Optional[str],
        typer.Option(
            "--class",
            help="Name of the model class to load from the module.",
        ),
    ] = None,
    package_name: Annotated[
        Optional[str],
        typer.Option(
            "--package",
            help=(
                "The package to use as the anchor point from which to resolve the relative import "
                "to an absolute import. This is only required when performing a relative import."
            ),
        ),
    ] = None,
    output_dir: Annotated[
        Optional[Path],
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
    """Separates an audio file into its constituent stems."""
    import numpy as np
    import torch
    from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn
    from torchcodec.encoders import AudioEncoder

    from .inference import ChunkProcessed, InferenceEngine, InferenceOutput

    use_cuda = not cpu and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    logger.info(f"using {device}")

    logger.info(f"loading inference engine from {config_path=} and {checkpoint_path=}")
    engine = InferenceEngine.from_pretrained(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=device,
        module_name=module_name,
        class_name=class_name,
        package_name=package_name,
    )
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
                logger.info(f"wrote logits `{key}` to {output_file}")
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
                logger.info(f"wrote stem `{key}` to {output_file}")


@app.command()
def ls(
    registry_path: Annotated[
        Path,
        typer.Option(
            "--registry",
            "-r",
            help="Path to the model registry file.",
        ),
    ] = PATH_DATA / "registry.json",
) -> None:
    from rich.console import Console
    from rich.table import Table

    from .config import Registry

    registry = Registry.from_file(registry_path)
    table = Table(
        show_lines=False,
        pad_edge=False,
        box=None,
    )
    table.add_column("id", no_wrap=True)
    table.add_column("size", no_wrap=True)
    table.add_column("created_at", no_wrap=True)
    table.add_column("purpose", no_wrap=True)
    table.add_column("outputs", overflow="fold")

    for identifier, model in registry.items():
        size = f"{model.model_size / 1_000_000:.1f}M" if model.model_size else "?"
        created_at_date = (
            datetime.fromisoformat(model.created_at).strftime("%Y-%m-%d")
            if model.created_at
            else "?"
        )
        outputs = ",".join(model.output) if model.output else "-"
        table.add_row(
            identifier,
            size,
            created_at_date,
            model.purpose,
            outputs,
        )

    Console().print(table)


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


if __name__ == "__main__":
    app()
