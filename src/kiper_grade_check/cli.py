"""CLI entrypoint: `kgc fetch`, `kgc build`."""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.logging import RichHandler

from kiper_grade_check import pipeline

app = typer.Typer(no_args_is_help=True, add_completion=False)


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="%H:%M:%S",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )


@app.command()
def fetch(
    year: list[int] = typer.Option(None, help="Specific draft year(s); defaults to 2011–2025."),
    raw_dir: Path = typer.Option(Path("data/raw"), help="Cache directory for raw data."),
    force: bool = typer.Option(False, help="Bypass caches and re-fetch."),
) -> None:
    """Stage 1 — pull raw nflverse + report-card data into raw_dir."""
    _setup_logging()
    years = year if year else list(range(2011, 2026))
    pipeline.fetch_all(years=years, raw_dir=raw_dir, force=force)


@app.command()
def build(
    year: list[int] = typer.Option(None, help="Specific draft year(s); defaults to 2011–2025."),
    raw_dir: Path = typer.Option(Path("data/raw"), help="Where raw data lives."),
    processed_dir: Path = typer.Option(Path("data/processed"), help="Where processed tables go."),
) -> None:
    """Stage 2 — transform raw → processed datasets."""
    _setup_logging()
    years = year if year else list(range(2011, 2026))
    pipeline.build_all(years=years, raw_dir=raw_dir, processed_dir=processed_dir)


if __name__ == "__main__":
    app()
