from pathlib import Path
from unittest.mock import patch

import pandas as pd
import polars as pl
import pytest

from kiper_grade_check.sources import nflverse


@pytest.fixture
def fake_polars_picks() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "season": [2014, 2014],
            "round": [1, 2],
            "pick": [32, 64],
            "team": ["NE", "NE"],
            "gsis_id": ["00-0031234", "00-0031235"],
            "pfr_player_name": ["Player One", "Player Two"],
            "position": ["WR", "DT"],
            "college": ["Florida", "FSU"],
            "car_av": [42, 18],
            "w_av": [40, 16],
            "dr_av": [42, 10],
            "games": [120, 80],
            "probowls": [2, 0],
            "allpro": [0, 0],
            "to": [2024, 2022],
        }
    )


def test_load_draft_picks_returns_pandas_with_expected_columns(
    tmp_path: Path, fake_polars_picks: pl.DataFrame
) -> None:
    with patch("kiper_grade_check.sources.nflverse.nflreadpy.load_draft_picks", return_value=fake_polars_picks):
        df = nflverse.load_draft_picks(seasons=range(2014, 2015), cache_dir=tmp_path)

    assert isinstance(df, pd.DataFrame)
    assert {"season", "round", "pick", "team", "gsis_id", "car_av", "w_av", "probowls"}.issubset(df.columns)
    assert (tmp_path / "draft_picks.parquet").exists()


def test_load_draft_picks_uses_cache_when_present(tmp_path: Path, fake_polars_picks: pl.DataFrame) -> None:
    cached = fake_polars_picks.to_pandas()
    cached.to_parquet(tmp_path / "draft_picks.parquet")

    with patch("kiper_grade_check.sources.nflverse.nflreadpy.load_draft_picks") as mock:
        df = nflverse.load_draft_picks(seasons=range(2014, 2015), cache_dir=tmp_path)
        mock.assert_not_called()

    pd.testing.assert_frame_equal(df.reset_index(drop=True), cached.reset_index(drop=True))


def test_load_draft_picks_force_refresh_bypasses_cache(
    tmp_path: Path, fake_polars_picks: pl.DataFrame
) -> None:
    cached = fake_polars_picks.to_pandas()
    cached.to_parquet(tmp_path / "draft_picks.parquet")

    with patch(
        "kiper_grade_check.sources.nflverse.nflreadpy.load_draft_picks",
        return_value=fake_polars_picks,
    ) as mock:
        nflverse.load_draft_picks(seasons=range(2014, 2015), cache_dir=tmp_path, force=True)
        mock.assert_called_once()
