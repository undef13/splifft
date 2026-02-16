"""Utilities to collate community-contributed models and resources (jarredou's
colab notebook, deton25's guide, huggingface hub api: https://huggingface.co/.well-known/openapi.json)
into our `registry.json`.
"""

from __future__ import annotations

import io
import json
import logging
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import chain
from pathlib import Path, PurePosixPath
from typing import (
    IO,
    TYPE_CHECKING,
    Annotated,
    Any,
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
    cast,
)
from urllib.parse import quote, unquote, urlparse

import httpx
import typer
from rich.logging import RichHandler

from splifft import DIR_DATA, PATH_REGISTRY_DEFAULT
from splifft.config import Registry

if TYPE_CHECKING:
    from splifft.config import Config

PATH_BASE = Path(__file__).parent.parent
PATH_TMP = PATH_BASE / "scripts" / "tmp"
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


class GitHubReleaseNode(NamedTuple):
    owner: str
    repo_name: str
    tag: str

    @property
    def slug(self) -> str:
        return f"{self.owner}__{self.repo_name}__{self.tag}".replace("/", "_")


class ResourceUrl(NamedTuple):
    url: str

    @property
    def is_ckpt_or_yaml(self) -> bool:
        path = urlparse(self.url).path.lower()
        return (
            path.endswith(".ckpt")
            or path.endswith(".pt")
            or path.endswith(".yaml")
            or path.endswith(".yml")
        )


def _urls_from_registry(registry: Registry) -> Generator[ResourceUrl, None, None]:
    for model in registry.values():
        for resource in model.resources:
            yield ResourceUrl(resource.url)


def _urls_from_guide(path: Path) -> Generator[ResourceUrl, None, None]:
    text = path.read_text(encoding="utf-8")
    matches = re.findall(
        r"https?://(?:huggingface\.co|github\.com)/[^\s\)\]\}\"]+",
        text,
    )
    for url in matches:
        clean = url.rstrip(".,;:")
        parsed = urlparse(clean)
        if parsed.netloc == "huggingface.co" and parsed.path.lstrip("/").startswith("spaces/"):
            continue
        if (
            _hf_node_and_file_from_url(clean) is None
            and _gh_release_node_and_asset_from_url(clean) is None
        ):
            continue
        yield ResourceUrl(clean)


def _urls_from_jarredou(path: Path) -> Generator[ResourceUrl, None, None]:
    data = json.loads(path.read_text(encoding="utf-8"))
    for item in data:
        m = JarredouModel(**item)
        if uc := m.config_url:
            yield ResourceUrl(uc)
        if ck := m.checkpoint_url:
            yield ResourceUrl(ck)


def _warn_missing_in_registry(
    other_urls: Iterable[ResourceUrl], registry_urls: Iterable[ResourceUrl], *, source: str
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
    oid: str  # sha256
    size: int
    pointerSize: int


class RepoTreeEntry(TypedDict):
    """Item returned by `/api/models/{namespace}/{repo}/tree/{rev}/{path}`
    or `/api/models/{namespace}/{repo}/paths-info/{rev}`
    """

    type: EntryType
    oid: str  # sha1 (git blob)?
    size: int
    """Actual file size (LFS resolved if applicable)"""
    path: NotRequired[str]  # included in paths-info, sometimes omitted in shallow tree
    lfs: NotRequired[LfsInfo]
    lastCommit: NotRequired[lastCommitInfo]  # present if '?expand=true' passed to api
    # securityFileStatus: NotRequired[SecurityFileStatus]
    xetHash: NotRequired[str]


class GitHubReleaseAsset(TypedDict, total=False):
    id: int
    name: str
    size: int
    digest: str | None  # e.g. "sha256:...", often null for files older than 2025
    created_at: str
    updated_at: str
    browser_download_url: str


class GitHubRelease(TypedDict):
    tag_name: str
    assets: list[GitHubReleaseAsset]


_HF_FILE_EXTENSIONS = {".ckpt", ".pt", ".yaml", ".yml"}


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
    response.raise_for_status()
    data = response.json()
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.info(f"wrote {out_path}")


def _load_cached_hf_repo_tree(
    repo: HfResourceNode, output_path: Path
) -> list[RepoTreeEntry] | None:
    if not (out_path := output_path / f"{repo.slug}.json").exists():
        return None
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
    """Parse a Hugging Face resource URL.

    Supported URL forms:
    - https://huggingface.co/{namespace}/{repo}/{resolve|blob|tree}/{rev}/{path}
    - https://huggingface.co/{namespace}/{repo}
    - https://huggingface.co/{namespace}/{repo}/raw/{rev}/{path}
    """
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


def _gh_release_node_and_asset_from_url(
    url: str,
) -> tuple[GitHubReleaseNode, str | None] | None:
    """Parse a GitHub release URL.

    Supported URL forms:
    - https://github.com/{owner}/{repo}/releases/tag/{tag}
    - https://github.com/{owner}/{repo}/releases/download/{tag}/{asset}
    """
    parsed = urlparse(url)
    if parsed.netloc not in {"github.com", "www.github.com"}:
        return None
    parts = [unquote(part) for part in parsed.path.split("/") if part]
    if len(parts) < 4:
        return None
    owner, repo = parts[0], parts[1]
    if parts[2] != "releases":
        return None
    if parts[3] == "download":
        if len(parts) < 6:
            return None
        tag = parts[4]
        asset = "/".join(parts[5:]) or None
    elif parts[3] == "tag":
        if len(parts) < 5:
            return None
        tag = parts[4]
        asset = None
    else:
        return None
    asset_name = PurePosixPath(asset).name if asset else None
    return GitHubReleaseNode(owner=owner, repo_name=repo, tag=tag), asset_name


def _cache_github_release_assets(
    release: GitHubReleaseNode,
    output_path: Path,
    *,
    client: httpx.Client | None = None,
) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    out_path = output_path / f"{release.slug}.json"
    if out_path.exists():  # cache hit
        return
    assert client is not None
    url = (
        f"https://api.github.com/repos/{release.owner}/{release.repo_name}"
        f"/releases/tags/{quote(release.tag, safe='')}"
    )
    logger.info(f"fetching {url}")
    response = client.get(url, headers={"Accept": "application/vnd.github+json"})
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        logger.error(f"failed to fetch {url}: {e}, {locals()}")
        return
    data = response.json()
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.info(f"wrote {out_path}")


def _load_cached_github_release(
    release: GitHubReleaseNode, output_path: Path
) -> GitHubRelease | None:
    if not (out_path := output_path / f"{release.slug}.json").exists():
        return None
    return json.loads(out_path.read_text(encoding="utf-8"))  # type: ignore


def _build_github_asset_lookup(release: GitHubRelease) -> dict[str, GitHubReleaseAsset]:
    lookup: dict[str, GitHubReleaseAsset] = {}
    for asset in release.get("assets", []):
        name = asset.get("name")
        if name:
            lookup[name] = asset
        url = asset.get("browser_download_url")
        if url:
            lookup[url] = asset
    return lookup


def _parse_iso_datetime(value: str | None) -> datetime:
    if not value:
        return datetime.min.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _write_registry_sorted(registry: Registry, registry_path: Path) -> None:
    from pydantic import TypeAdapter

    registry = Registry(
        dict(
            sorted(
                registry.items(),
                key=lambda item: _parse_iso_datetime(item[1].created_at),
                reverse=True,
            )
        )
    )

    registry_path.write_bytes(
        TypeAdapter(Registry).dump_json(registry, indent=4, exclude_defaults=True)
    )


@app.command()
def fix_registry(
    registry_path: Path = PATH_REGISTRY_DEFAULT,
    guide_path: Path = PATH_GUIDE_MODELS_MD,
    jarredou_path: Path = PATH_JARREDOU_MSST_JSON,
    output_path: Path = PATH_TMP / "cache",
    warn_missing: bool = False,  # off by default since some are htdemucs / unsupported architecture
    skip_overwrite_created_at: tuple[str] = ("bs_roformer-fruit-sw",),
) -> None:
    """Checks that URLs in the guide and jarredou's colab are present in the registry,
    caches file metadata from Hugging Face and GitHub releases and ensures dates are correct."""
    registry = Registry.from_file(registry_path)

    registry_urls = list(_urls_from_registry(registry))
    guide_urls = list(_urls_from_guide(guide_path))
    jarredou_urls = list(_urls_from_jarredou(jarredou_path))
    if warn_missing:
        _warn_missing_in_registry(guide_urls, registry_urls, source="guide")
        _warn_missing_in_registry(jarredou_urls, registry_urls, source="jarredou colab")

    all_urls = list(chain(registry_urls, guide_urls, jarredou_urls))
    hf_repos: set[HfResourceNode] = set()
    gh_releases: set[GitHubReleaseNode] = set()
    for u in all_urls:
        if (hf_parsed := _hf_node_and_file_from_url(u.url)) is not None:
            hf_node, _ = hf_parsed
            hf_repos.add(hf_node)
        if (gh_parsed := _gh_release_node_and_asset_from_url(u.url)) is not None:
            gh_node, _ = gh_parsed
            gh_releases.add(gh_node)

    output_hf = output_path / "huggingface"
    output_gh = output_path / "github"
    with httpx.Client(http2=True, headers={"User-Agent": "splifft-community/1.0"}) as client:
        for repo in sorted(hf_repos):
            _cache_hf_repo_tree(repo, output_hf, client=client)
        for release in sorted(gh_releases):
            _cache_github_release_assets(release, output_gh, client=client)

    repo_trees: dict[HfResourceNode, dict[str, RepoTreeEntry]] = {}
    for repo in hf_repos:
        if (tree_entries := _load_cached_hf_repo_tree(repo, output_hf)) is None:
            continue
        repo_trees[repo] = _build_entry_lookup(tree_entries, repo.base_path)

    gh_release_assets: dict[GitHubReleaseNode, dict[str, GitHubReleaseAsset]] = {}
    for release in gh_releases:
        if (release_data := _load_cached_github_release(release, output_gh)) is None:
            continue
        gh_release_assets[release] = _build_github_asset_lookup(release_data)

    for model_id, model in registry.items():
        created_at = None
        model_size = None
        digest = None
        for resource in model.resources:
            url = resource.url
            if (hf_parsed := _hf_node_and_file_from_url(url)) is not None:
                repo, file_path = hf_parsed
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
                if url.lower().endswith((".ckpt", ".pt")):
                    if (lc := entry.get("lastCommit")) is not None:
                        created_at = lc.get("date")
                    model_size = entry.get("size")
                    if (lfs := entry.get("lfs")) is not None:  # TODO what about Xet?
                        digest = f"sha256:{lfs['oid']}"
                continue

            if (gh_parsed := _gh_release_node_and_asset_from_url(url)) is not None:
                release, asset_name = gh_parsed
                lookup = gh_release_assets.get(release, {})
                asset = None
                if asset_name:
                    asset = lookup.get(asset_name)
                if asset is None:
                    asset = lookup.get(url)
                if asset is None:
                    continue
                if url.lower().endswith((".ckpt", ".pt")):
                    created_at = asset.get("created_at") or asset.get("updated_at")
                    model_size = asset.get("size")
                    if (d := asset.get("digest")) is not None and d.startswith("sha256:"):
                        digest = d
                continue
        if created_at is not None and model_id not in skip_overwrite_created_at:
            model.created_at = created_at
        if model_size is not None:
            model.model_size = model_size
        if digest is not None:
            # TODO handle digests for multiple resources, we assign to the first for now
            if ckpt := next((r for r in model.resources if r.kind == "model_ckpt"), None):
                ckpt.digest = digest

    _write_registry_sorted(registry, registry_path)
    subprocess.run(
        ["pnpm", "run", "fmt:json", str(PATH_REGISTRY_DEFAULT)],
        cwd=PATH_BASE,
        check=True,
    )


def slugify_url(url: str) -> str | None:  # quick and dirty
    for prefix in [
        "https://huggingface.co/",
        "https://raw.githubusercontent.com/",
        "https://github.com/",
    ]:
        if url.startswith(prefix):
            s1 = url.removeprefix(prefix).removesuffix(".yaml")
            for kw in [
                "resolve/",
                "blob/",
                "raw/",
                "main/",
                "master/",
                "releases/download/",
                "refs/heads/",
            ]:
                s1 = s1.replace(kw, "")
            s1.replace("Music-Source-Separation-Training", "msst")
            return s1.replace("/", "__").replace(".", "__")
    return None


Slug: TypeAlias = str


# multiple models can share the same config, we hardcode mappings of MSST slug -> ours model id for now
SHARED_CONFIG_ID_BY_SLUG: dict[Slug, str] = {
    "GaboxR67__MelBandRoformers__melbandroformers__instrumental__inst_gabox": "mel_roformer-gabox-inst",
    "GaboxR67__MelBandRoformers__melbandroformers__vocals__voc_gabox": "mel_roformer-gabox-vocals",
    "GaboxR67__MelBandRoformers__melbandroformers__karaoke__karaokegabox_1750911344": "mel_roformer-gabox-karaoke",
    "pcunwa__BS-Roformer-Revive__config": "bs_roformer-unwa-revive",
    "pcunwa__Mel-Band-Roformer-Inst__config_melbandroformer_inst": "mel_roformer-unwa-inst",
    "pcunwa__Kim-Mel-Band-Roformer-FT__config_kimmel_unwa_ft": "mel_roformer-unwa-kim_ft",
    "anvuew__dereverb_mel_band_roformer__dereverb_mel_band_roformer_anvuew": "mel_roformer-anvuew-dereverb",
    "pcunwa__Mel-Band-Roformer-InstVoc-Duality__config_melbandroformer_instvoc_duality": "mel_roformer-unwa-duality",
    "jarredou__aufr33_MelBand_Denoise__model_mel_band_roformer_denoise": "mel_roformer-aufr33-denoise",
    "Sucial__Chorus_Male_Female_BS_Roformer__config_chorus_male_female_bs_roformer": "bs_roformer-sucial-chorus-mf",
}


def _purge_unverified_configs(config_dir: Path) -> None:
    for path in config_dir.glob(".*"):
        if path.is_file():
            path.unlink()


def _ensure_msst_yaml(*, client: httpx.Client, dest: Path, url: str, model_id: str) -> None:
    if dest.exists():
        return
    logger.info(f"downloading config for '{model_id}' from {url}")
    try:
        response = client.get(url)
        response.raise_for_status()
        dest.write_bytes(response.content)
    except Exception as e:
        logger.error(f"failed to download {url}: {e}")


def _load_msst_yaml(path: Path) -> dict[str, Any] | None:
    # YAML is feature-gated to avoid requiring pyyaml for the core package.
    import yaml  # type: ignore

    class CustomYamlLoader(yaml.SafeLoader):  # type: ignore
        """Custom YAML loader that handles the `!!python/tuple` tag."""

    def _construct_python_tuple(loader: yaml.Loader, node: Any) -> tuple[Any, ...]:
        return tuple(loader.construct_sequence(node))

    CustomYamlLoader.add_constructor("!!python/tuple", _construct_python_tuple)
    CustomYamlLoader.add_constructor("tag:yaml.org,2002:python/tuple", _construct_python_tuple)

    with open(path, "r") as f:
        if not (loaded := yaml.load(f, Loader=CustomYamlLoader)):
            return None
        return cast(dict[str, Any], loaded)


def convert_msst_yaml_path(path: Path, *, model_id: str) -> Config | None:
    if not (msst := _load_msst_yaml(path)):
        return None
    return convert_msst_config_bs_roformer(msst, model_id=model_id)


@app.command()
def fix_registry_with_msst(
    registry_path: Path = PATH_REGISTRY_DEFAULT,
    msst_dir: Path = PATH_TMP / "cache" / "msst_configs",
) -> None:
    """Batch process MSST configs and mutate the registry.

    Converted configs are written to package data at data/config as dot-prefixed files
    (e.g. .{config_id}.json) to mark them as auto-imported and unverified. Any
    existing dot files are removed first so the conversion always produces fresh
    auto-imported configs. If a verified config already exists at {config_id}.json,
    no dot file is written and the registry is pointed at the verified config.
    For converted models, registry.config_id is updated to the verified config id
    or the dot-prefixed auto-imported id accordingly.
    """
    registry = Registry.from_file(registry_path)
    msst_dir.mkdir(parents=True, exist_ok=True)
    config_dir = DIR_DATA / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    _purge_unverified_configs(config_dir)

    slug_to_splifft_config: dict[Slug, Config] = {}
    slug_to_splifft_model_ids: dict[Slug, list[str]] = {}
    with httpx.Client(http2=True, follow_redirects=True) as client:
        for model_id, model in registry.items():
            for resource in model.resources:
                if resource.kind != "config_msst":
                    continue
                if (slug := slugify_url(resource.url)) is None:
                    logger.warning(f"skipping unsupported url for '{model_id}': {resource.url}")
                    continue
                dest = (msst_dir / slug).with_suffix(".yaml")
                _ensure_msst_yaml(client=client, dest=dest, url=resource.url, model_id=model_id)
                if model.architecture not in {"bs_roformer", "mel_roformer"}:
                    logger.warning(
                        f"skipping unsupported architecture '{model.architecture}' for '{model_id}'"
                    )
                    continue
                try:
                    if (result := convert_msst_yaml_path(dest, model_id=model_id)) is None:
                        continue
                except Exception as e:
                    logger.exception(
                        f"failed to convert MSST config for '{model_id}' ({dest} {resource.url} {slug}): {e}"
                    )
                    continue
                slug_to_splifft_config[slug] = result
                slug_to_splifft_model_ids.setdefault(slug, []).append(model_id)

    for slug, (config, model_ids) in zip(
        slug_to_splifft_config.keys(),
        zip(slug_to_splifft_config.values(), slug_to_splifft_model_ids.values()),
    ):
        if len(model_ids) > 1:
            config_id = SHARED_CONFIG_ID_BY_SLUG.get(slug)
            assert config_id is not None, f"{slug=} has {model_ids=} but no shared config mapping"
        else:
            config_id = model_ids[0]
        verified_path = (config_dir / config_id).with_suffix(".json")
        if verified_path.exists():
            registry_config_id = config_id
        else:
            registry_config_id = f".{config_id}"
            (config_dir / registry_config_id).with_suffix(".json").write_text(
                config.model_dump_json(indent=4, exclude_defaults=True)
            )
        for model_id in model_ids:
            if model_id in registry:
                registry[model_id].config_id = registry_config_id

    _write_registry_sorted(registry, registry_path)


def snake_case(name: str) -> str:
    return name.lower().replace(" ", "_")


def remove_missing(d: dict[str, Any]) -> dict[str, Any]:
    from pydantic.experimental.missing_sentinel import MISSING

    return {k: v for k, v in d.items() if v is not MISSING}


def convert_msst_config_bs_roformer(
    msst: dict[str, Any],
    model_id: str = "custom-model",
) -> Config | None:
    """Convert an MSST YAML configuration file to a splifft JSON configuration."""
    # right now we only support bs/mel roformer.
    from pydantic import TypeAdapter
    from pydantic.experimental.missing_sentinel import MISSING

    from splifft import types as t
    from splifft.config import Config, TorchDtype
    from splifft.models.bs_roformer import BSRoformerParams

    audio = msst.get("audio", {})
    model = msst.get("model", {})
    training = msst.get("training", {})
    inference = msst.get("inference", {})

    assert not isinstance(model, str), "possible htdemucs!"

    is_mel = "num_bands" in model
    band_config: dict[str, Any]
    if is_mel:
        band_config = remove_missing(
            {
                "kind": "mel",
                "num_bands": model.get("num_bands", MISSING),
                "sample_rate": model.get("sample_rate") or audio.get("sample_rate", MISSING),
                "stft_n_fft": model.get("stft_n_fft", MISSING),
            }
        )
    else:
        band_config = remove_missing(
            {
                "kind": "fixed",
                "freqs_per_bands": model.get("freqs_per_bands", MISSING),
            }
        )

    msst_depth = model.get("mask_estimator_depth", 2)
    splifft_mask_depth = msst_depth + 1 if is_mel else msst_depth

    # see utils/model_utils.py::prefer_target_instrument
    stems = list(map(snake_case, training.get("instruments", [])))
    target = training.get("target_instrument")
    output_stems = [snake_case(target)] if target else stems if stems else MISSING

    model_params = remove_missing(
        {
            "chunk_size": audio.get("chunk_size", MISSING),
            "output_stem_names": output_stems,
            "dim": model.get("dim", MISSING),
            "depth": model.get("depth", MISSING),
            "stft_hop_length": model.get("stft_hop_length") or audio.get("hop_length", MISSING),
            "stereo": model.get("stereo", MISSING),
            "time_transformer_depth": model.get("time_transformer_depth", MISSING),
            "freq_transformer_depth": model.get("freq_transformer_depth", MISSING),
            "linear_transformer_depth": model.get("linear_transformer_depth", MISSING),
            "band_config": band_config,
            "dim_head": model.get("dim_head", MISSING),
            "heads": model.get("heads", MISSING),
            "attn_dropout": model.get("attn_dropout", MISSING),
            "ff_dropout": model.get("ff_dropout", MISSING),
            "ff_mult": model.get(
                "mlp_expansion_factor", MISSING
            ),  # map mlp_expansion_factor to ff_mult?
            "flash_attn": model.get("flash_attn", MISSING),
            "norm_output": is_mel,  # msst inherits bug lucidrains
            "mask_estimator_depth": splifft_mask_depth,
            "mlp_expansion_factor": model.get("mlp_expansion_factor", MISSING),
            "use_torch_checkpoint": model.get("use_torch_checkpoint", MISSING),
            "skip_connection": model.get("skip_connection", MISSING),
        }
    )
    # this hack is required to flush away defaults in the concrete config, which is
    # then force casted back into the lazy config
    ta = TypeAdapter(BSRoformerParams)
    ta.rebuild(_types_namespace={"TorchDtype": TorchDtype, "t": t})
    model_params_lazy = ta.dump_python(ta.validate_python(model_params), exclude_defaults=True)

    stft_params = remove_missing(
        {
            "n_fft": model.get("nfft") or audio.get("n_fft", MISSING),
            "hop_length": model.get("hop_size") or audio.get("hop_length", MISSING),
            "win_length": model.get("win_size") or model.get("stft_n_fft", MISSING),
            "normalized": model.get("normalized", MISSING),
        }
    )

    audio_io_params = remove_missing(
        {
            "target_sample_rate": audio.get("sample_rate", MISSING),
            "force_channels": audio.get("num_channels", MISSING),
        }
    )

    inference_params = remove_missing(
        {
            "batch_size": inference.get("batch_size", MISSING),
        }
    )

    chunking_params = {}
    if (num_overlap := inference.get("num_overlap")) is not None:
        chunking_params["overlap_ratio"] = 1.0 - 1.0 / num_overlap

    # normalization is tricky. MSST has `inference.normalize` or `training.normalize`.
    normalize_enabled = inference.get("normalize", training.get("normalize", MISSING))
    normalization_params = {}
    if normalize_enabled is not MISSING:
        normalization_params["enabled"] = normalize_enabled

    return Config.model_validate(
        {
            "identifier": model_id,
            "model_type": "bs_roformer",
            "model": model_params_lazy,
            "stft": stft_params,
            "audio_io": audio_io_params,
            "inference": inference_params,
            "waveform_chunking": chunking_params,
            "normalization": normalization_params,
            "masking": {},  # empty defaults
            "output": {"stem_names": "all"},
        }
    )


if __name__ == "__main__":
    app()
