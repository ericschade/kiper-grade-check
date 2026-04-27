"""Thin wrappers around nflreadpy that cache to parquet and return pandas DataFrames.

nflreadpy returns polars DataFrames; the rest of the codebase is pandas. We convert
at this boundary so callers never see polars.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import nflreadpy
import pandas as pd

DEFAULT_CACHE_DIR = Path("data/raw")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _to_pandas(obj) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj
    return obj.to_pandas()


def load_draft_picks(
    seasons: Iterable[int],
    cache_dir: Path = DEFAULT_CACHE_DIR,
    force: bool = False,
) -> pd.DataFrame:
    _ensure_dir(cache_dir)
    cache = cache_dir / "draft_picks.parquet"
    if cache.exists() and not force:
        return pd.read_parquet(cache)

    raw = nflreadpy.load_draft_picks(seasons=list(seasons))
    df = _to_pandas(raw)
    df.to_parquet(cache, index=False)
    return df


def load_contracts(cache_dir: Path = DEFAULT_CACHE_DIR, force: bool = False) -> pd.DataFrame:
    _ensure_dir(cache_dir)
    cache = cache_dir / "contracts.parquet"
    if cache.exists() and not force:
        return pd.read_parquet(cache)

    raw = nflreadpy.load_contracts()
    df = _to_pandas(raw)
    df.to_parquet(cache, index=False)
    return df


def load_schedules(
    seasons: Iterable[int],
    cache_dir: Path = DEFAULT_CACHE_DIR,
    force: bool = False,
) -> pd.DataFrame:
    _ensure_dir(cache_dir)
    cache = cache_dir / "schedules.parquet"
    if cache.exists() and not force:
        return pd.read_parquet(cache)

    raw = nflreadpy.load_schedules(seasons=list(seasons))
    df = _to_pandas(raw)
    df.to_parquet(cache, index=False)
    return df
