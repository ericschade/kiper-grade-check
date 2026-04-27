"""Build the per-pick dataframe from raw nflverse draft data."""

from __future__ import annotations

from typing import Literal

import pandas as pd

EvalCompleteness = Literal["full", "partial", "too_recent"]

TEAM_CODE_ALIASES: dict[tuple[int, str], str] = {
    **{(y, "OAK"): "LV" for y in range(2020, 2030)},
    **{(y, "SD"): "LAC" for y in range(2017, 2030)},
    **{(y, "STL"): "LAR" for y in range(2016, 2030)},
}


def add_eval_completeness(draft_year: int) -> EvalCompleteness:
    if draft_year <= 2021:
        return "full"
    if draft_year <= 2023:
        return "partial"
    return "too_recent"


def canonicalize_team_codes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    year_col = "draft_year" if "draft_year" in out.columns else "season"
    out["team"] = [
        TEAM_CODE_ALIASES.get((int(y), t), t)
        for y, t in zip(out[year_col], out["team"])
    ]
    return out


def build_picks(raw: pd.DataFrame) -> pd.DataFrame:
    """Reshape raw nflreadpy draft picks into our `picks.parquet` schema."""
    df = raw.rename(
        columns={
            "season": "draft_year",
            "pick": "pick_overall",
            "gsis_id": "player_id",
            "pfr_player_name": "player_name",
            "car_av": "career_av",
            "w_av": "weighted_av",
            "dr_av": "draft_team_av",
            "probowls": "pro_bowls",
            "allpro": "all_pros",
            "to": "final_season",
        }
    )

    df = canonicalize_team_codes(df)
    df["eval_completeness"] = df["draft_year"].map(add_eval_completeness)

    for col in ["career_av", "weighted_av", "draft_team_av", "games", "pro_bowls", "all_pros"]:
        df[col] = df[col].fillna(0).astype("int64")

    keep = [
        "draft_year",
        "round",
        "pick_overall",
        "team",
        "player_id",
        "player_name",
        "position",
        "college",
        "career_av",
        "weighted_av",
        "draft_team_av",
        "games",
        "pro_bowls",
        "all_pros",
        "final_season",
        "eval_completeness",
    ]
    return df[keep].reset_index(drop=True)
