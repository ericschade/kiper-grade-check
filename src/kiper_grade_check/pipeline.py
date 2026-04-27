"""Pipeline orchestrators: fetch (Stage 1) and build (Stage 2).

Stage 3 (analysis) is a notebook, not part of this module.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from kiper_grade_check.grade import compute_actual_grade, compute_expected_av_by_slot, z_within_year
from kiper_grade_check.sources import nflverse, report_card
from kiper_grade_check.sources.report_card import parse_report_card_html
from kiper_grade_check.transform import careers, contracts, team_wins

log = logging.getLogger(__name__)

DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_PROCESSED_DIR = Path("data/processed")


def fetch_all(
    years: Iterable[int],
    raw_dir: Path = DEFAULT_RAW_DIR,
    force: bool = False,
) -> None:
    """Stage 1: pull all raw inputs into raw_dir/."""
    raw_dir.mkdir(parents=True, exist_ok=True)

    log.info("fetching nflverse draft picks")
    nflverse.load_draft_picks(seasons=range(2011, 2026), cache_dir=raw_dir, force=force)

    log.info("fetching nflverse contracts")
    nflverse.load_contracts(cache_dir=raw_dir, force=force)

    log.info("fetching nflverse schedules")
    nflverse.load_schedules(seasons=range(2011, 2030), cache_dir=raw_dir, force=force)

    rc_dir = raw_dir / "report_card"
    for year in years:
        log.info("fetching report card for %d", year)
        report_card.fetch_report_card_html(year, cache_dir=rc_dir)


def _emit(df: pd.DataFrame, processed_dir: Path, name: str) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(processed_dir / f"{name}.parquet", index=False)
    df.to_csv(processed_dir / f"{name}.csv", index=False)


_TEAM_NAME_MAP = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Oakland Raiders": "OAK",
    "Los Angeles Chargers": "LAC", "San Diego Chargers": "SD",
    "Los Angeles Rams": "LAR", "St. Louis Rams": "STL",
    "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN", "New England Patriots": "NE",
    "New Orleans Saints": "NO", "New York Giants": "NYG", "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT", "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS", "Washington Football Team": "WAS",
    "Washington Redskins": "WAS",
}


def _team_to_nflverse_code(name: str) -> str | None:
    if not isinstance(name, str):
        return None
    s = name.strip()
    if s in _TEAM_NAME_MAP:
        return _TEAM_NAME_MAP[s]
    if len(s) == 3 and s.isupper():
        return s
    return None


def build_all(
    years: Iterable[int],
    raw_dir: Path = DEFAULT_RAW_DIR,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
) -> None:
    """Stage 2: transform raw → processed/{picks,analyst_grades,team_outcomes,comparison}."""
    years_list = list(years)

    raw_picks = pd.read_parquet(raw_dir / "draft_picks.parquet")
    raw_contracts = pd.read_parquet(raw_dir / "contracts.parquet")
    raw_schedules = pd.read_parquet(raw_dir / "schedules.parquet")

    picks = careers.build_picks(raw_picks)
    target = picks[picks["draft_year"].isin(years_list)].copy()

    sc = contracts.derive_second_contract_same_team(target, raw_contracts)  # type: ignore[arg-type]
    target = target.merge(sc, on="player_id", how="left")
    _emit(target, processed_dir, "picks")

    rc_dir = raw_dir / "report_card"
    grade_frames: list[pd.DataFrame] = []
    for year in years_list:
        path = rc_dir / f"{year}.html"
        if not path.exists():
            log.warning("missing report card for %d, skipping", year)
            continue
        urls_dict = report_card.REPORT_CARD_URLS(year)
        df = parse_report_card_html(path.read_text(), draft_year=year, source_url=urls_dict["live"])
        if not df.empty:
            grade_frames.append(df)
    analyst_grades = (
        pd.concat(grade_frames, ignore_index=True) if grade_frames else pd.DataFrame()
    )

    if not analyst_grades.empty:
        analyst_grades["team"] = analyst_grades["team"].map(_team_to_nflverse_code).fillna(analyst_grades["team"])
    _emit(analyst_grades, processed_dir, "analyst_grades")

    historical = raw_picks.rename(columns={"pick": "pick_overall", "car_av": "career_av"})
    av_curve = compute_expected_av_by_slot(historical[["pick_overall", "career_av"]].dropna())  # type: ignore[arg-type]
    target = target.merge(av_curve, on="pick_overall", how="left")
    target["av_over_expected"] = target["career_av"] - target["expected_av"].fillna(0)

    grouped = (
        target.groupby(["draft_year", "team", "eval_completeness"], as_index=False)
        .agg(
            num_picks=("player_id", "count"),
            total_career_av=("career_av", "sum"),
            total_weighted_av=("weighted_av", "sum"),
            total_av_over_expected=("av_over_expected", "sum"),
            avg_pro_bowls_per_pick=("pro_bowls", "mean"),
            pct_second_contract=("second_contract_same_team", lambda s: s.dropna().mean() if s.notna().any() else None),
        )
    )
    teams_in_scope: list[str] = grouped["team"].drop_duplicates().tolist()  # type: ignore[assignment]
    wins = team_wins.compute_team_wins_5yr(raw_schedules, draft_years=years_list, teams=teams_in_scope)
    outcomes = grouped.merge(wins, on=["draft_year", "team"], how="left")
    outcomes = compute_actual_grade(outcomes)
    _emit(outcomes, processed_dir, "team_outcomes")

    if not analyst_grades.empty:
        ag = z_within_year(analyst_grades.rename(columns={"grade_numeric": "analyst_grade_numeric"}), "analyst_grade_numeric")
        ag = ag.rename(columns={"analyst_grade_numeric_z": "analyst_grade_z"})
        comparison = ag.merge(
            outcomes[["draft_year", "team", "actual_grade_z", "actual_grade_letter", "eval_completeness"]],
            on=["draft_year", "team"],
            how="inner",
        )
        comparison["residual"] = comparison["actual_grade_z"] - comparison["analyst_grade_z"]
    else:
        comparison = pd.DataFrame()
    _emit(comparison, processed_dir, "comparison")
