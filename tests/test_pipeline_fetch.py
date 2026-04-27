from pathlib import Path
from unittest.mock import patch

import pandas as pd

from kiper_grade_check.pipeline import fetch_all


def test_fetch_all_creates_expected_artifacts(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"

    fake_picks = pd.DataFrame({"season": [2014], "round": [1], "pick": [29]})
    fake_contracts = pd.DataFrame({"gsis_id": ["P1"], "team": ["NE"], "year_signed": [2014]})
    fake_schedules = pd.DataFrame(
        {"season": [2014], "game_type": ["REG"], "home_team": ["NE"], "away_team": ["BUF"], "home_score": [30], "away_score": [10]}
    )

    def write_stub(year, cache_dir):
        cache_dir.mkdir(parents=True, exist_ok=True)
        p = cache_dir / f"{year}.html"
        p.write_text("<html></html>")
        return p

    with (
        patch("kiper_grade_check.pipeline.nflverse.load_draft_picks", return_value=fake_picks),
        patch("kiper_grade_check.pipeline.nflverse.load_contracts", return_value=fake_contracts),
        patch("kiper_grade_check.pipeline.nflverse.load_schedules", return_value=fake_schedules),
        patch("kiper_grade_check.pipeline.report_card.fetch_report_card_html", side_effect=write_stub),
    ):
        fetch_all(years=range(2014, 2016), raw_dir=raw_dir)

    assert (raw_dir / "report_card" / "2014.html").exists()
    assert (raw_dir / "report_card" / "2015.html").exists()
