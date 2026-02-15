"""Lightweight utilities for music source separation."""

from pathlib import Path

# TODO verify default registry and config data are included by hatchling.
DIR_MODULE = Path(__file__).parent
DIR_DATA = DIR_MODULE / "data"
DIR_CONFIG_DEFAULT = DIR_DATA / "config"
PATH_REGISTRY_DEFAULT = DIR_DATA / "registry.json"

# NOTE: not re-exporting anything because 1) our structure is simple enough and
# 2) we have feature flags that otherwise might not be enabled by default
