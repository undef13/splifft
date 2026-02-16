from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime
from pathlib import Path
from textwrap import wrap
from typing import Annotated, Any, Literal, TypeAlias, get_args

import httpx
import matplotlib.pyplot as plt
import orjson
import polars as pl
import typer
from aiolimiter import AsyncLimiter
from matplotlib.axes import Axes
from pydantic import BaseModel, BeforeValidator
from rich.logging import RichHandler
from rich.progress import Progress, TaskID

import splifft.types as t

logging.basicConfig(level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()])
logger = logging.getLogger(__name__)

app = typer.Typer(
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)

PATH_BASE = Path(__file__).parent.parent
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.3"
BASE_URL_API = "https://mvsep.com/api"
PATH_MVSEP = PATH_BASE / "docs" / "assets" / "mvsep"
PATH_MVSEP_QUALITY = PATH_MVSEP / "quality"


#
# networking
#


async def _fetch_and_save(
    session: httpx.AsyncClient,
    entry_id: int,
    progress: Progress,
    task_id: TaskID,
    *,
    base_url: str = f"{BASE_URL_API}/quality_checker/entry",
) -> None:
    """Fetches and saves a single MVSep entry."""
    output_file = PATH_MVSEP_QUALITY / f"{entry_id}.json"
    try:
        response = await session.get(base_url, params={"id": entry_id})
        if response.status_code != 400:
            response.raise_for_status()
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "wb") as f:
            f.write(orjson.dumps(response.json(), option=orjson.OPT_INDENT_2))
        logger.debug(f"wrote {output_file=}")
    except Exception as e:
        logger.error(f"request error for {entry_id=}: {e}")
    finally:
        progress.update(task_id, advance=1)


@app.command()
def fetch(
    start_id: int,
    end_id: int,
    proxy: Annotated[str | None, typer.Option(help="e.g., `socks5://127.0.0.1:10809`")] = None,
    max_concurrent: int = 1,
    rate_limit: float = 2.0,  # leaky bucket
) -> None:
    ids_to_fetch = [
        i for i in range(start_id, end_id + 1) if not (PATH_MVSEP_QUALITY / f"{i}.json").exists()
    ]

    if not ids_to_fetch:
        logger.info("all entries exist locally")
        return

    logger.info(
        f"fetching {len(ids_to_fetch)} new entries with {max_concurrent=} and {rate_limit=}"
    )

    async def _run() -> None:
        limiter = AsyncLimiter(rate_limit, 1)
        limits = httpx.Limits(max_connections=max_concurrent, max_keepalive_connections=5)

        async def _throttled_fetch(
            session: httpx.AsyncClient, entry_id: int, progress: Progress, task_id: TaskID
        ) -> None:
            async with limiter:
                await _fetch_and_save(session, entry_id, progress, task_id)

        async with httpx.AsyncClient(
            proxy=proxy, limits=limits, timeout=30.0, http2=True, headers={"User-Agent": USER_AGENT}
        ) as session:
            with Progress() as progress:
                task = progress.add_task("[cyan]fetching...", total=len(ids_to_fetch))
                await asyncio.gather(
                    *[_throttled_fetch(session, i, progress, task) for i in ids_to_fetch]
                )

    asyncio.run(_run())
    logger.info(f"fetch complete. data saved in `{PATH_MVSEP_QUALITY}`")


Sdr: TypeAlias = Annotated[
    t.Sdr | None, BeforeValidator(lambda v: None if v == 0 or v == -1000 else v)
]


METRIC_REGEX = re.compile(r"Metric\s+(\S+)\s+for\s+(\S+):\s+([-\d.]+)")


def parse_metrics(metrics_str: str) -> dict[str, float]:
    return {
        f"{match.group(1)}_{match.group(2)}": float(match.group(3))
        for match in METRIC_REGEX.finditer(metrics_str)
    }


class EntryData(BaseModel):
    id: int
    dataset_type: int
    date: datetime
    algo_name: str
    main_text: str
    proc: int
    proc_remote_date: str
    msg: str
    metrics: Annotated[
        dict[str, float] | None, BeforeValidator(lambda v: parse_metrics(v) if v else None)
    ]  # extra combos here
    is_ensemble: Annotated[
        bool, BeforeValidator(lambda v: False if v == 0 else True if v == 1 else v)
    ]
    # ... combos destructured here


#
# preprocessing
#

METRICS = get_args(t.Metric)
# fmt: off
# fmt: on
INSTRUMENTS = get_args(t.Instrument)
EXPECTED_METRIC_INSTRUMENTS = tuple(
    f"{metric}_{instrument}" for metric in METRICS for instrument in INSTRUMENTS
)
NON_METRICS = tuple(k for k in EntryData.model_fields if k != "metrics")
FP_QUALITY = PATH_MVSEP / "quality.parquet"
PATH_PLOTS = PATH_MVSEP / "plots"


class EntryNotFound(BaseModel):
    message: Literal["Job not found"]


class MetricsResponse(BaseModel):
    success: bool
    data: EntryData | EntryNotFound
    msg: str | None = None


def parse_into_flat_record(file_path: Path) -> dict[str, Any] | None:
    with open(file_path, "rb") as f:
        raw_data = orjson.loads(f.read())
    response = MetricsResponse.model_validate(raw_data)

    if not response.success or isinstance(response.data, EntryNotFound):
        logger.debug(f"invalid {response=}")
        return None
    if response.data.metrics is None:
        logger.debug(f"no metrics found in {file_path.name=}")
        return None

    flat_record = response.data.model_dump()
    flat_record.update(flat_record.pop("metrics"))
    for k, v in flat_record.items():
        if k in NON_METRICS:
            continue
        assert k in EXPECTED_METRIC_INSTRUMENTS, f"unexpected key {k} in {file_path.name=}"
        assert isinstance(v, float), f"expected float for {k} in {file_path.name=}, got {type(v)}"
        if v == 0 or v == -1000:  # fix broken SDR
            flat_record[k] = None
    return flat_record


@app.command()
def process() -> None:
    json_files = sorted(PATH_MVSEP_QUALITY.glob("*.json"), key=lambda p: int(p.stem))

    records = [
        record for path in json_files if (record := parse_into_flat_record(path)) is not None
    ]
    df = pl.DataFrame(records, infer_schema_length=100_000)
    df.write_parquet(FP_QUALITY)


#
# visualization
#
def _derive_bounds_from_highlights(
    cols_to_plot: list[str], highlights: dict[int, dict[str, Any]]
) -> dict[str, tuple[float, float]]:
    bounds: dict[str, tuple[float, float]] = {}
    for col in cols_to_plot:
        values = []
        for row in highlights.values():
            v = row.get(col)
            if isinstance(v, (int, float)):
                values.append(float(v))

        if not values:
            continue

        lo = min(values)
        hi = max(values)
        span = hi - lo
        margin = max(span * 0.20, 0.1) if span > 0 else 0.2
        bounds[col] = (lo - margin, hi + margin)

    return bounds


@app.command()
def correlations(
    instruments: list[str] = ["instrum", "vocals", "drums", "bass", "other", "piano"],
    metrics_to_plot: list[str] = [
        "sdr",
        "bleedless",
        "fullness",
        "l1_freq",
        # "aura_stft",
        # "aura_mrstft",
    ],
    ids: Annotated[
        list[int] | None,
        typer.Option("--id", "-i", help="one or more entry IDs to highlight on the plot."),
    ] = None,
) -> None:
    from matplotlib.lines import Line2D

    if not FP_QUALITY.exists():
        logger.error(f"quality data not found at `{FP_QUALITY}`. run `fetch` and `process` first.")
        raise typer.Exit(1)

    plt.style.use("dark_background")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Roboto Mono"],
            "figure.facecolor": "#1c1c1c",
            "axes.facecolor": "#1c1c1c",
            "axes.labelcolor": "#d0d0d0",
            "xtick.color": "#d0d0d0",
            "ytick.color": "#d0d0d0",
            "text.color": "#d0d0d0",
            "grid.color": "#444444",
            "svg.fonttype": "none",  # render as text, not paths
        }
    )

    df = pl.read_parquet(FP_QUALITY).filter(pl.col("dataset_type") == 1)

    highlights = {}
    if ids:
        highlight_df = df.filter(pl.col("id").is_in(ids))
        if highlight_df.height < len(ids):
            found_ids = highlight_df["id"].to_list()
            missing = set(ids) - set(found_ids)
            logger.warning(f"could not find data for ids: {missing}")
        highlights = {row["id"]: row for row in highlight_df.to_dicts()}

    for instrument in instruments:
        cols_to_plot = [f"{metric}_{instrument}" for metric in metrics_to_plot]

        if missing_cols := [col for col in cols_to_plot if col not in df.columns]:
            logger.error(f"instrument `{instrument}` missing data for metrics: {missing_cols}")
            continue

        col_bounds = _derive_bounds_from_highlights(cols_to_plot, highlights)

        plot_df = df.select(cols_to_plot).drop_nulls().to_numpy()
        num_metrics = len(cols_to_plot)

        fig, axes = plt.subplots(
            num_metrics, num_metrics, figsize=(num_metrics * 2, num_metrics * 2), layout="tight"
        )
        fig.suptitle(f"Metric Correlations for `{instrument}`", fontsize=4 * num_metrics)
        legend_handles, legend_labels, added_to_legend = [], [], set()

        for i in range(num_metrics):
            for j in range(num_metrics):
                ax: Axes = axes[i, j]

                if j < i:
                    ax.axis("off")
                    continue

                x_col = cols_to_plot[j]
                y_col = cols_to_plot[i]
                x_bounds = col_bounds.get(x_col)
                y_bounds = col_bounds.get(y_col)

                if i == j:
                    # ax.hist(plot_df[:, i], bins=30, color="#155473")
                    if x_bounds:
                        ax.set_xlim(*x_bounds)
                else:
                    ax.scatter(
                        plot_df[:, j],
                        plot_df[:, i],
                        alpha=0.15,
                        s=2,
                        color="#eeeeee",
                        edgecolors="none",
                    )
                    if x_bounds:
                        ax.set_xlim(*x_bounds)
                    if y_bounds:
                        ax.set_ylim(*y_bounds)

                for idx, (id, data) in enumerate(highlights.items()):
                    color = f"C{idx}"
                    if id not in added_to_legend:
                        label = f"{id}: {data['algo_name']}"
                        proxy_line = Line2D([0], [0], color=color, lw=1)
                        legend_handles.append(proxy_line)
                        legend_labels.append(label)
                        added_to_legend.add(id)

                    if i == j:
                        val = data.get(cols_to_plot[i])
                        if val is not None:
                            ax.axvline(val, color=color, lw=1)
                    elif j > i:
                        x_val = data.get(cols_to_plot[j])
                        y_val = data.get(cols_to_plot[i])
                        if x_val is not None and y_val is not None:
                            ax.scatter(
                                x_val,
                                y_val,
                                color=color,
                                s=20,
                                edgecolors="none",
                            )

                clean_label_j = cols_to_plot[j].replace(f"_{instrument}", "")
                clean_label_i = cols_to_plot[i].replace(f"_{instrument}", "")
                ax.tick_params(
                    axis="both",
                    labelleft=False,
                    labelbottom=False,
                    labelright=j == num_metrics - 1,
                    labeltop=i == 0,
                    color="#888",
                    labelsize=2 * num_metrics,
                    labelcolor="#888",
                    direction="in",
                )
                if i == 0:
                    ax.set_xlabel(clean_label_j, fontsize=3 * num_metrics)
                    ax.xaxis.set_label_position("top")
                    ax.xaxis.tick_top()
                if j == num_metrics - 1:
                    ax.set_ylabel(clean_label_i, fontsize=3 * num_metrics)
                    ax.yaxis.set_label_position("right")
                    ax.yaxis.tick_right()
                for spine in ax.spines.values():
                    spine.set_edgecolor("#888")

        if legend_handles:
            fig.legend(
                legend_handles,
                ["\n".join(wrap(label, width=40)) for label in legend_labels],
                loc="lower left",
                fontsize=3 * num_metrics,
            )

        PATH_PLOTS.mkdir(parents=True, exist_ok=True)
        output_path = PATH_PLOTS / f"correlations_{instrument}.svg"
        plt.savefig(output_path)
        plt.close(fig)
        logger.info(f"wrote correlation matrix `{output_path}`")


if __name__ == "__main__":
    app()
