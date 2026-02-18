from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from logging import basicConfig, getLogger
from pathlib import Path
from typing import Any, TypeAlias

import torch
import typer
from rich.logging import RichHandler
from torch.profiler import ProfilerActivity, profile
from torch.utils.flop_counter import FlopCounterMode

from splifft.inference import InferenceEngine

app = typer.Typer(pretty_exceptions_show_locals=False, no_args_is_help=True)

PATH_BASE = Path(__file__).parent.parent
PATH_TMP = PATH_BASE / "scripts" / "tmp"
PATH_INPUT = PATH_BASE / "data" / "audio" / "input"

logger = getLogger(__name__)
basicConfig(
    level="INFO",
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)

WallClockTime: TypeAlias = float


@dataclass(frozen=True)
class ProfileCase:
    name: str
    model_id: str
    mixture_path: Path


def run_once(engine: InferenceEngine, case: ProfileCase) -> WallClockTime:
    device = engine.model_device

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()

    for event in engine.stream(case.mixture_path):
        logger.info(event)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter() - t0


def run_profile_memory(engine: InferenceEngine, case: ProfileCase, *, out_dir: Path) -> None:
    device = engine.model_device
    if device.type != "cuda":
        return None

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    try:
        torch.cuda.memory._record_memory_history(enabled="all", context="all", stacks="all")
        wall_time_s = run_once(engine, case)

        snapshot = torch.cuda.memory._snapshot()
        path_pickle = out_dir / "memory_snapshot.pickle"
        with open(str(path_pickle), "wb") as f:
            pickle.dump(snapshot, f)

        memory_viz: Any = getattr(torch.cuda, "_memory_viz")
        (out_dir / "memory_snapshot.html").write_text(
            memory_viz.segment_plot(snapshot),
            encoding="utf-8",
        )

        summary = {
            "name": case.name,
            "wall_time_s": wall_time_s,
            "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        }
        (out_dir / "memory.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    finally:
        torch.cuda.memory._record_memory_history(enabled=None)


def run_profile_kernels(engine: InferenceEngine, case: ProfileCase, *, out_dir: Path) -> None:
    activities = [ProfilerActivity.CPU]
    if engine.model_device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    trace_path = out_dir / "profile_kernels_chrome_trace.json"
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        with_flops=True,
    ) as prof:
        _ = run_once(engine, case)
    prof.export_chrome_trace(str(trace_path))


def run_profile_mfu(
    engine: InferenceEngine,
    case: ProfileCase,
    *,
    out_dir: Path,
) -> None:
    with FlopCounterMode(display=False) as flop_counter:
        wall_time_s = run_once(engine, case)

    total_flops = int(flop_counter.get_total_flops())
    achieved_tflops = total_flops / max(wall_time_s, 1e-12) / 1e12

    payload = {
        "name": case.name,
        "wall_time_s": wall_time_s,
        "total_flops": total_flops,
        "achieved_tflops": achieved_tflops,
        "flop_counts": {
            str(module): {str(op): int(count) for op, count in ops.items()}
            for module, ops in flop_counter.get_flop_counts().items()
        },
    }
    (out_dir / "mfu_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


@app.command()
def run(
    out_root: Path = PATH_TMP / "profile",
    warmup: int = 1,
    force_cpu: bool = False,
    profile_kernels: bool = True,
    profile_mfu: bool = True,
    override_config: list[str] = [],
) -> None:
    """Run the default profile suite and emit machine-readable artifacts."""
    run_dir = out_root / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=True)
    for case in [
        ProfileCase(
            name="main_bs_roformer",
            model_id="bs_roformer-fruit-sw",
            mixture_path=PATH_INPUT / "main.flac",
        ),
        ProfileCase(
            name="main_drums_mdx23c",
            model_id="mdx23c-aufr33-drumsep_6stem",
            mixture_path=PATH_INPUT / "main_drums.flac",
        ),
    ]:
        out_dir = run_dir / case.name
        out_dir.mkdir(parents=True, exist_ok=True)
        assert case.mixture_path.exists()
        run_device = torch.device("cpu" if force_cpu or not torch.cuda.is_available() else "cuda")
        engine = InferenceEngine.from_registry(
            case.model_id,
            model_device=run_device,
            io_device=run_device,
            overrides=override_config,
        )
        for _ in range(warmup):
            _ = run_once(engine, case)

        run_profile_memory(engine, case, out_dir=out_dir)
        if profile_kernels:
            run_profile_kernels(engine, case, out_dir=out_dir)
        if profile_mfu:
            run_profile_mfu(engine, case, out_dir=out_dir)


if __name__ == "__main__":
    app()
