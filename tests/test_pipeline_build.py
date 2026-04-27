from pathlib import Path

import pandas as pd
import pytest

from kiper_grade_check.pipeline import build_all


@pytest.fixture
def stub_raw(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    raw.mkdir()

    picks_raw = pd.DataFrame(
        {
            "season": [2014, 2014, 2014, 2014],
            "round": [1, 5, 1, 6],
            "pick": [10, 150, 20, 200],
            "team": ["NE", "NE", "DAL", "DAL"],
            "gsis_id": ["P1", "P2", "P3", "P4"],
            "pfr_player_name": ["A", "B", "C", "D"],
            "position": ["WR", "DT", "RB", "S"],
            "college": ["FL", "FSU", "OU", "GA"],
            "car_av": [40, 5, 30, 8],
            "w_av": [38, 4, 28, 7],
            "dr_av": [40, 5, 30, 8],
            "games": [120, 60, 100, 70],
            "probowls": [2, 0, 1, 0],
            "allpro": [1, 0, 0, 0],
            "to": [2024, 2018, 2022, 2020],
        }
    )
    picks_raw.to_parquet(raw / "draft_picks.parquet", index=False)

    contracts = pd.DataFrame(
        {
            "gsis_id": ["P1", "P1", "P2", "P2", "P3", "P3", "P4", "P4"],
            "team":    ["NE", "NE", "NE", "NE", "DAL","DAL","DAL","CHI"],
            "year_signed": [2014, 2018, 2014, 2018, 2014, 2018, 2014, 2018],
        }
    )
    contracts.to_parquet(raw / "contracts.parquet", index=False)

    sched_rows = []
    for season in range(2014, 2019):
        for _ in range(12):
            sched_rows.append({"season": season, "game_type": "REG", "home_team": "NE", "away_team": "BUF", "home_score": 30, "away_score": 10})
        for _ in range(4):
            sched_rows.append({"season": season, "game_type": "REG", "home_team": "NE", "away_team": "BUF", "home_score": 7, "away_score": 24})
        for _ in range(8):
            sched_rows.append({"season": season, "game_type": "REG", "home_team": "DAL", "away_team": "NYG", "home_score": 24, "away_score": 14})
        for _ in range(8):
            sched_rows.append({"season": season, "game_type": "REG", "home_team": "DAL", "away_team": "NYG", "home_score": 7, "away_score": 24})
    pd.DataFrame(sched_rows).to_parquet(raw / "schedules.parquet", index=False)

    rc = raw / "report_card"
    rc.mkdir()
    (rc / "2014.html").write_text(
        "<html><body><table><thead><tr><th>Team</th><th>Kiper</th></tr></thead>"
        "<tbody>"
        "<tr><td>NE</td><td>A</td></tr>"
        "<tr><td>DAL</td><td>C</td></tr>"
        "</tbody></table></body></html>"
    )
    return raw


def test_build_all_emits_four_tables(stub_raw: Path, tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    build_all(years=[2014], raw_dir=stub_raw, processed_dir=processed)

    for name in ["picks", "analyst_grades", "team_outcomes", "comparison"]:
        assert (processed / f"{name}.parquet").exists()
        assert (processed / f"{name}.csv").exists()

    picks = pd.read_parquet(processed / "picks.parquet")
    assert {"draft_year", "team", "career_av", "eval_completeness"}.issubset(picks.columns)
    assert len(picks) == 4

    outcomes = pd.read_parquet(processed / "team_outcomes.parquet")
    assert set(outcomes["team"]) == {"NE", "DAL"}
    z = outcomes.set_index("team")["actual_grade_z"]
    assert z["NE"] > z["DAL"]

    comparison = pd.read_parquet(processed / "comparison.parquet")
    assert {"analyst", "analyst_grade_z", "actual_grade_z", "residual"}.issubset(comparison.columns)
    assert (comparison["analyst"] == "Kiper").all()
