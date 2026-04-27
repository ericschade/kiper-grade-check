"""Compute regular-season wins per team for a 5-year window starting at draft_year."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


def season_wins(schedules: pd.DataFrame, season: int, team: str) -> int:
    s = schedules[(schedules["season"] == season) & (schedules["game_type"] == "REG")]
    if s.empty:
        return 0

    home_wins = ((s["home_team"] == team) & (s["home_score"] > s["away_score"])).sum()
    away_wins = ((s["away_team"] == team) & (s["away_score"] > s["home_score"])).sum()
    return int(home_wins + away_wins)


def compute_team_wins_5yr(
    schedules: pd.DataFrame, draft_years: Iterable[int], teams: Iterable[str]
) -> pd.DataFrame:
    rows: list[dict] = []
    for year in draft_years:
        for team in teams:
            total = sum(season_wins(schedules, season=year + offset, team=team) for offset in range(5))
            rows.append({"draft_year": year, "team": team, "team_wins_5yr": total})
    return pd.DataFrame(rows)
