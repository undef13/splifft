"""Utilities to collate community-contributed models and resources (jarredou's
colab notebook, deton25's guide, huggingface hub api: https://huggingface.co/.well-known/openapi.json)
into our `registry.json`.

- https://huggingface.co/{{namespace}}/{{repo}}/{resolve|blob|tree}/{{rev}}/{{path}}
- https://huggingface.co/{{namespace}}/{{repo}}
- https://huggingface.co/{{namespace}}/{{repo}}/raw/{{rev}}/{{path}}
"""

import io
import json
import logging
import re
import subprocess
from dataclasses import asdict, dataclass
from itertools import chain
from pathlib import Path, PurePosixPath
from typing import (
    IO,
    Annotated,
    Generator,
    Iterable,
    Iterator,
    Literal,
    NamedTuple,
    NotRequired,
    Self,
    Sequence,
    TypeAlias,
    TypedDict,
)
from urllib.parse import quote, unquote, urlparse

import httpx
import typer
from rich.logging import RichHandler

from splifft import PATH_BASE, PATH_DATA, PATH_SCRIPTS
from splifft.config import Registry

PATH_TMP = PATH_SCRIPTS / "tmp"
URL_JARREDOU_MSST_COLAB = "https://raw.githubusercontent.com/jarredou/Music-Source-Separation-Training-Colab-Inference/refs/heads/main/Music_Source_Separation_Training_(Colab_Inference).ipynb"
PATH_JARREDOU_MSST_COLAB = PATH_TMP / "jarredou_msst_colab.ipynb"
PATH_JARREDOU_MSST_JSON = PATH_TMP / "jarredou_msst_colab.json"

logging.basicConfig(level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()])
logger = logging.getLogger(__name__)
app = typer.Typer(
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)


@dataclass
class JarredouModel:
    name: str
    model_type: str
    config_path: str
    checkpoint_path: str
    config_url: str
    checkpoint_url: str


OutputPath = Annotated[Path, typer.Option("--output", "-o", help="use `-` for stdout")]


@app.command()
def parse_jarredou_colab(
    url: str = URL_JARREDOU_MSST_COLAB,
    path_ipynb: Path = PATH_JARREDOU_MSST_COLAB,
    output_path: OutputPath = PATH_JARREDOU_MSST_JSON,
) -> None:
    """Fetch and parse models from jarredou's colab notebook."""
    if not path_ipynb.exists():
        logger.info(f"downloading {url=}...")
        path_ipynb.parent.mkdir(exist_ok=True, parents=True)
        response = httpx.get(url)
        response.raise_for_status()
        path_ipynb.write_bytes(response.content)
        logger.info(f"wrote {path_ipynb=}")

    pattern = re.compile(
        r"(?:el)?if model == '(?P<name>[^']+)':\s*\n"
        r"\s*model_type = '(?P<model_type>[^']+)'.*\n"
        r"\s*config_path = '(?P<config_path>[^']+)'.*\n"
        r"\s*start_check_point = '(?P<checkpoint_path>[^']+)'.*\n"
        r"\s*download_file\('(?P<config_url>[^']+)'\).*\n"
        r"\s*download_file\('(?P<checkpoint_url>[^']+)'\)",
        re.MULTILINE,
    )
    models = []
    for block in _colab_blocks(path_ipynb):
        for match in pattern.finditer(block):
            models.append(asdict(JarredouModel(**match.groupdict())))

    output_json = json.dumps(models, indent=4)
    if str(output_path) == "-":
        print(output_json)
    else:
        output_path.write_text(output_json, encoding="utf-8")
        logger.info(f"wrote {len(models)} models to {output_path}")


def _colab_blocks(path_ipynb: Path) -> Generator[str, None, None]:
    with open(path_ipynb, "r", encoding="utf-8") as f:
        ipynb = json.load(f)
    for cell in ipynb.get("cells", []):
        source_lines = cell.get("source", [])
        if cell.get("cell_type") != "code":
            continue
        yield "\n".join(
            li
            for line in source_lines
            if (not (li := line.strip()).startswith("%")) and not li.startswith("!")
        )


#
# deton24's guide has hundreds of pages, we get rid of sections that are not model-related
#

PATH_GUIDE_DOCX = PATH_TMP / "guide.docx"
PATH_GUIDE_MD = PATH_TMP / "guide.md"
PATH_GUIDE_MODELS_MD = PATH_TMP / "guide_filtered.md"
# fmt: off
KEYWORDS_GUIDE_NON_MODELS = [
    "*plugins*", "mdx settings", "uvr5 gui", "sources of flacs",
    "arigato78", "karafan", "ensemble", "ripple", "training", "tips",
    "troubleshooting", "stems/multitracks", "gpu acceleration",
    "audioshake", "lalal", "gsep", "dango", "moises",
    "dolby atmos ripping", "ai-killing"
]
# fmt: on


@app.command()
def parse_guide(
    blacklist_keywords: list[str] = KEYWORDS_GUIDE_NON_MODELS,
    output_path: OutputPath = PATH_GUIDE_MODELS_MD,
) -> None:
    """Output a pruned version of the guide in markdown."""
    if not PATH_GUIDE_DOCX.exists():
        logger.error(
            f"{PATH_GUIDE_DOCX=} not found, "
            "navigate to https://docs.google.com/document/d/17fjNvJzj8ZGSer7c7OFe_CNfUKbAxEh_OBv94ZdRG5c/edit, "
            "then `File > Download > Microsoft Word (.docx)`"
        )
        return
    subprocess.run(["pandoc", str(PATH_GUIDE_DOCX), "-o", str(PATH_GUIDE_MD)])

    def write_pruned_guide(file_out: IO[str]) -> None:
        with open(PATH_GUIDE_MD, "r", encoding="utf-8") as file_in:
            for line in prune_sections(file_in, blacklist_keywords):
                file_out.write(line)

    if str(output_path) == "-":
        buf = io.StringIO()
        write_pruned_guide(buf)
        print(buf.getvalue())
        return
    with open(output_path, "w", encoding="utf-8") as fout:
        write_pruned_guide(fout)


def prune_sections(
    lines: Iterator[str], blacklist_keywords: Sequence[str]
) -> Generator[str, None, None]:
    exclude_level: int | None = None
    used_keywords = []
    for line in lines:
        if (line_ls := line.lstrip().lower()).startswith("#"):
            # print(line_ls)
            curr_level = line_ls.split(" ")[0].count("#")
            if exclude_level is not None and curr_level <= exclude_level:
                exclude_level = None
            for keyword in blacklist_keywords:
                if keyword not in line_ls:
                    continue
                exclude_level = curr_level
                used_keywords.append(keyword)
                logger.info(f"excluding `{line_ls}`")
        if exclude_level is None:
            yield line
    if unused := set(blacklist_keywords) - set(used_keywords):
        logger.warning(f"unused keywords: {unused}")


#
# verification: ensure MSST yaml and ckpt urls are reachable
# and add useful metadata such as commit hash, date and file size
#


class HfResourceNode(NamedTuple):
    owner: str
    repo_name: str
    rev: str
    base_path: str

    @property
    def slug(self) -> str:
        safe_path = self.base_path.replace("/", "_") if self.base_path else "_root"
        return f"{self.owner}__{self.repo_name}__{self.rev}__{safe_path}".replace("/", "_")

    @classmethod
    def try_from_url(cls, url: str) -> Self | None:
        if (parsed := urlparse(url)).netloc != "huggingface.co":
            return None
        parts = [part for part in parsed.path.split("/") if part]
        if not parts or parts[0] in {"spaces", "datasets", "api"}:
            return None
        if len(parts) < 2:
            return None
        return HfResourceNode(
            owner=parts[0],
            repo_name=parts[1],
            rev="main",
            base_path="",
        )  # type: ignore


class HfResourceUrl(NamedTuple):
    url: str

    @property
    def repo(self) -> HfResourceNode | None:
        return HfResourceNode.try_from_url(self.url)

    @property
    def is_ckpt_or_yaml(self) -> bool:
        path = urlparse(self.url).path.lower()
        return path.endswith(".ckpt") or path.endswith(".yaml") or path.endswith(".yml")


def _urls_from_registry(registry: Registry) -> Generator[HfResourceUrl, None, None]:
    for model in registry.values():
        for resource in model.resources:
            yield HfResourceUrl(resource.url)


def _urls_from_guide(path: Path) -> Generator[HfResourceUrl, None, None]:
    text = path.read_text(encoding="utf-8")
    matches = re.findall(r"https?://huggingface\.co/[^\s\)\]\}\"]+", text)
    for url in matches:
        clean = url.rstrip(".,;:")
        if urlparse(clean).path.lstrip("/").startswith("spaces/"):
            continue
        yield HfResourceUrl(clean)


def _urls_from_jarredou(path: Path) -> Generator[HfResourceUrl, None, None]:
    data = json.loads(path.read_text(encoding="utf-8"))
    for item in data:
        m = JarredouModel(**item)
        if uc := m.config_url:
            yield HfResourceUrl(uc)
        if ck := m.checkpoint_url:
            yield HfResourceUrl(ck)


def _warn_missing_in_registry(
    other_urls: Iterable[HfResourceUrl], registry_urls: Iterable[HfResourceUrl], *, source: str
) -> None:
    registry_set = {u.url for u in registry_urls if u.is_ckpt_or_yaml}
    seen = set()
    for u in other_urls:
        if u.is_ckpt_or_yaml and u.url not in registry_set and u.url not in seen:
            logger.warning(f"registry: missing url from {source}: {u.url}")
        seen.add(u.url)


EntryType: TypeAlias = Literal["file", "directory", "unknown"]


class lastCommitInfo(TypedDict):
    id: str
    title: str
    date: str  # ISO 8601 Date format


class LfsInfo(TypedDict):
    oid: str
    size: int
    pointerSize: int


class RepoTreeEntry(TypedDict):
    """Item returned by `/api/models/{namespace}/{repo}/tree/{rev}/{path}`
    or `/api/models/{namespace}/{repo}/paths-info/{rev}`
    """

    type: EntryType
    oid: str
    size: int
    """Actual file size (LFS resolved if applicable)"""
    path: NotRequired[str]  # included in paths-info, sometimes omitted in shallow tree
    lfs: NotRequired[LfsInfo]
    lastCommit: NotRequired[lastCommitInfo]  # present if '?expand=true' passed to api
    # securityFileStatus: NotRequired[SecurityFileStatus]
    xetHash: NotRequired[str]


_HF_FILE_EXTENSIONS = {".ckpt", ".yaml", ".yml"}


def _looks_like_file(path: str) -> bool:
    suffix = PurePosixPath(path).suffix.lower()
    return suffix in _HF_FILE_EXTENSIONS


def _cache_hf_repo_tree(
    repo: HfResourceNode,
    output_path: Path,
    *,
    client: httpx.Client | None = None,
) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    out_path = output_path / f"{repo.slug}.json"
    if out_path.exists():  # cache hit
        return
    assert client is not None
    base_path = repo.base_path
    if base_path and _looks_like_file(base_path):
        base_path = str(PurePosixPath(base_path).parent)
        if base_path == ".":
            base_path = ""
    path_for_api = unquote(base_path or ".")
    url = (
        f"https://huggingface.co/api/models/{repo.owner}/{repo.repo_name}"
        f"/tree/{repo.rev}/{quote(path_for_api, safe='/')}?expand=true"
    )
    logger.info(f"fetching {url}")
    response = client.get(url)
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        logger.error(f"Failed to fetch {url}: {e}, {locals()}")
        return
    data = response.json()
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.info(f"wrote {out_path}")


def _load_cached_hf_repo_tree(repo: HfResourceNode, output_path: Path) -> list[RepoTreeEntry]:
    out_path = output_path / f"{repo.slug}.json"
    return json.loads(out_path.read_text(encoding="utf-8"))  # type: ignore


def _build_entry_lookup(
    entries: list[RepoTreeEntry],
    base_path: str,
) -> dict[str, RepoTreeEntry]:
    lookup: dict[str, RepoTreeEntry] = {}
    for entry in entries:
        path = entry.get("path", "")
        if not path:
            continue
        lookup[path] = entry
        lookup[PurePosixPath(path).name] = entry
        if base_path and path.startswith(base_path + "/"):
            rel = path[len(base_path) + 1 :]
            if rel:
                lookup[rel] = entry
    return lookup


def _hf_node_and_file_from_url(url: str) -> tuple[HfResourceNode, str | None] | None:
    if (repo := HfResourceNode.try_from_url(url)) is None:
        return None
    parsed = urlparse(url)
    parts = [
        unquote(part) for part in parsed.path.split("/") if part
    ]  # unquote required when url has spaces
    if len(parts) < 2:
        return None

    rev = None
    file_path = None
    base_path = ""
    if len(parts) >= 4 and parts[2] in {"resolve", "blob", "raw", "tree"}:
        rev = parts[3]
        file_path = "/".join(parts[4:]) or None
        if parts[2] == "tree":
            base_path = file_path or ""
            if base_path and _looks_like_file(base_path):
                base_path = str(PurePosixPath(base_path).parent)
                if base_path == ".":
                    base_path = ""
                file_path = file_path
            else:
                file_path = None
        elif file_path:
            base_path = str(PurePosixPath(file_path).parent)
            if base_path == ".":
                base_path = ""
    elif len(parts) >= 3:
        file_path = "/".join(parts[2:]) or None
        if file_path:
            base_path = str(PurePosixPath(file_path).parent)
            if base_path == ".":
                base_path = ""

    resolved_rev = rev or "main"
    node = HfResourceNode(
        owner=repo.owner,
        repo_name=repo.repo_name,
        rev=resolved_rev,
        base_path=base_path,
    )
    return node, file_path


@app.command()
def fix_registry(
    registry_path: Path = PATH_DATA / "registry.json",
    guide_path: Path = PATH_GUIDE_MODELS_MD,
    jarredou_path: Path = PATH_JARREDOU_MSST_JSON,
    output_path: Path = PATH_TMP / "huggingface",
    warn_missing: bool = False,  # off by default since some are htdemucs / unsupported architecture
) -> None:
    """Checks that URLs in the guide and jarredou's colab are present in the registry,
    caches file metadata from hugging face and ensures dates are correct."""
    from pydantic import TypeAdapter

    registry = Registry.from_file(registry_path)

    registry_urls = list(_urls_from_registry(registry))
    guide_urls = list(_urls_from_guide(guide_path))
    jarredou_urls = list(_urls_from_jarredou(jarredou_path))
    if warn_missing:
        _warn_missing_in_registry(guide_urls, registry_urls, source="guide")
        _warn_missing_in_registry(jarredou_urls, registry_urls, source="jarredou colab")

    all_urls = list(chain(registry_urls, guide_urls, jarredou_urls))
    repos: set[HfResourceNode] = set()
    for u in all_urls:
        if (parsed := _hf_node_and_file_from_url(u.url)) is None:
            continue
        node, _ = parsed
        repos.add(node)
    with httpx.Client(http2=True) as client:
        for repo in sorted(repos):
            _cache_hf_repo_tree(repo, output_path, client=client)

    repo_trees: dict[HfResourceNode, dict[str, RepoTreeEntry]] = {}
    for repo in repos:
        tree_entries = _load_cached_hf_repo_tree(repo, output_path)
        repo_trees[repo] = _build_entry_lookup(tree_entries, repo.base_path)

    for model in registry.values():
        created_at = None
        model_size = None
        for resource in model.resources:
            url = resource.url
            if (parsed := _hf_node_and_file_from_url(url)) is None:
                continue
            repo, file_path = parsed
            if not file_path:
                continue
            rel_path = (
                file_path[len(repo.base_path) + 1 :]
                if repo.base_path and file_path.startswith(repo.base_path + "/")
                else file_path
            )
            entry = (
                repo_trees.get(repo, {}).get(file_path)
                or repo_trees.get(repo, {}).get(rel_path)
                or repo_trees.get(repo, {}).get(PurePosixPath(file_path).name)
            )
            if entry is None:
                continue
            if url.lower().endswith(".ckpt"):
                if (lc := entry.get("lastCommit")) is not None:
                    created_at = lc.get("date")
                model_size = entry.get("size")
        if created_at is not None:
            model.created_at = created_at
        if model_size is not None:
            model.model_size = model_size

    registry_path.write_bytes(
        TypeAdapter(Registry).dump_json(registry, indent=4, exclude_defaults=True)
    )
    subprocess.run(
        ["pnpm", "run", "fmt:json", "data/registry.json"],
        cwd=PATH_BASE,
        check=True,
    )


if __name__ == "__main__":
    app()
