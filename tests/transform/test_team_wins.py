import pandas as pd

from kiper_grade_check.transform.team_wins import compute_team_wins_5yr, season_wins


def _schedule(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_season_wins_counts_only_regular_season() -> None:
    sched = _schedule(
        [
            {"season": 2017, "game_type": "REG", "home_team": "NE", "away_team": "BUF", "home_score": 23, "away_score": 3},
            {"season": 2017, "game_type": "REG", "home_team": "MIA", "away_team": "NE", "home_score": 14, "away_score": 27},
            {"season": 2017, "game_type": "WC",  "home_team": "NE", "away_team": "TEN", "home_score": 35, "away_score": 14},
        ]
    )
    wins = season_wins(sched, season=2017, team="NE")
    assert wins == 2


def test_team_wins_5yr_sums_consecutive_seasons() -> None:
    rows = []
    for season in range(2014, 2019):
        for _ in range(5):
            rows.append({"season": season, "game_type": "REG", "home_team": "NE", "away_team": "BUF", "home_score": 30, "away_score": 0})
    sched = _schedule(rows)
    df = compute_team_wins_5yr(sched, draft_years=[2014], teams=["NE"])
    assert df.set_index(["draft_year", "team"]).loc[(2014, "NE"), "team_wins_5yr"] == 25


def test_team_wins_5yr_handles_ties_as_half_or_zero() -> None:
    sched = _schedule(
        [
            {"season": 2014, "game_type": "REG", "home_team": "NE", "away_team": "BUF", "home_score": 17, "away_score": 17},
        ]
    )
    df = compute_team_wins_5yr(sched, draft_years=[2014], teams=["NE"])
    assert df.set_index(["draft_year", "team"]).loc[(2014, "NE"), "team_wins_5yr"] == 0
