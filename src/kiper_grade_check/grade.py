"""Compute the actual draft grade and helpers."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess


def compute_expected_av_by_slot(historical_picks: pd.DataFrame) -> pd.DataFrame:
    """Return a per-slot expected career_av curve, LOESS-smoothed.

    Input: a dataframe with at least columns [pick_overall, career_av] from picks
    across all available historical drafts (recommend 1980–present for a stable curve).
    Output: dataframe with columns [pick_overall, expected_av], one row per slot
    observed in input.
    """
    grouped = (
        historical_picks.groupby("pick_overall", as_index=False)["career_av"]
        .mean()
        .sort_values("pick_overall")
    )
    smoothed = lowess(
        endog=grouped["career_av"].to_numpy(),
        exog=grouped["pick_overall"].to_numpy(),
        frac=0.15,
        return_sorted=True,
    )
    return pd.DataFrame({"pick_overall": smoothed[:, 0].astype(int), "expected_av": smoothed[:, 1]})


DEFAULT_WEIGHTS: Mapping[str, float] = {
    "av_over_expected": 0.50,
    "pro_bowls_per_pick": 0.20,
    "pct_second_contract": 0.15,
    "team_wins_5yr": 0.15,
}

LETTER_THRESHOLDS = [
    (1.25, "A+"),
    (0.75, "A"),
    (0.25, "A-"),
    (0.0, "B+"),
    (-0.25, "B"),
    (-0.75, "B-"),
    (-1.25, "C+"),
    (-1.75, "C"),
    (-2.25, "C-"),
    (float("-inf"), "F"),
]


def bucket_z_to_letter(z: float) -> str:
    for threshold, letter in LETTER_THRESHOLDS:
        if z >= threshold:
            return letter
    return "F"


def z_within_year(df: pd.DataFrame, col: str, group_col: str = "draft_year") -> pd.DataFrame:
    """Return df with an added '{col}_z' column z-scored within each draft_year."""
    out = df.copy()

    def _z(s: pd.Series) -> pd.Series:
        std = s.std(ddof=0)
        if std == 0 or pd.isna(std):
            return pd.Series([0.0] * len(s), index=s.index)
        return (s - s.mean()) / std

    out[f"{col}_z"] = out.groupby(group_col)[col].transform(_z)
    return out


def compute_actual_grade(
    team_outcomes: pd.DataFrame,
    weights: Mapping[str, float] = DEFAULT_WEIGHTS,
) -> pd.DataFrame:
    """Compute actual_grade_z (composite) + actual_grade_letter for each team-year.

    `pct_second_contract` is treated as null when eval_completeness != "full"; its
    weight is redistributed pro-rata across the other three components.
    """
    df = team_outcomes.copy()

    df["pct_second_contract"] = df.apply(
        lambda r: r["pct_second_contract"] if r["eval_completeness"] == "full" else np.nan,
        axis=1,
    )

    df = z_within_year(df, "total_av_over_expected")
    df = z_within_year(df, "avg_pro_bowls_per_pick")
    df = z_within_year(df, "pct_second_contract")
    df = z_within_year(df, "team_wins_5yr")

    av_w = weights["av_over_expected"]
    pb_w = weights["pro_bowls_per_pick"]
    sc_w = weights["pct_second_contract"]
    tw_w = weights["team_wins_5yr"]

    def composite(row: pd.Series) -> float:
        if pd.isna(row["pct_second_contract"]):
            denom = av_w + pb_w + tw_w
            return (
                av_w * row["total_av_over_expected_z"]
                + pb_w * row["avg_pro_bowls_per_pick_z"]
                + tw_w * row["team_wins_5yr_z"]
            ) / denom
        return (
            av_w * row["total_av_over_expected_z"]
            + pb_w * row["avg_pro_bowls_per_pick_z"]
            + sc_w * row["pct_second_contract_z"]
            + tw_w * row["team_wins_5yr_z"]
        )

    df["actual_grade_z"] = df.apply(composite, axis=1)
    df["actual_grade_letter"] = df["actual_grade_z"].apply(bucket_z_to_letter)
    return df
