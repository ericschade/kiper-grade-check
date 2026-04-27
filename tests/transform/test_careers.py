import pandas as pd
import pytest

from kiper_grade_check.transform.careers import (
    EvalCompleteness,
    add_eval_completeness,
    build_picks,
    canonicalize_team_codes,
)


def test_eval_completeness_full() -> None:
    assert add_eval_completeness(2011) == "full"
    assert add_eval_completeness(2021) == "full"


def test_eval_completeness_partial() -> None:
    assert add_eval_completeness(2022) == "partial"
    assert add_eval_completeness(2023) == "partial"


def test_eval_completeness_too_recent() -> None:
    assert add_eval_completeness(2024) == "too_recent"
    assert add_eval_completeness(2025) == "too_recent"


def test_canonicalize_team_codes_oak_to_lv_in_2020() -> None:
    df = pd.DataFrame({"draft_year": [2019, 2020], "team": ["OAK", "OAK"]})
    canon = canonicalize_team_codes(df)
    assert canon.loc[0, "team"] == "OAK"
    assert canon.loc[1, "team"] == "LV"


def test_canonicalize_team_codes_sd_to_lac_in_2017() -> None:
    df = pd.DataFrame({"draft_year": [2016, 2017], "team": ["SD", "SD"]})
    canon = canonicalize_team_codes(df)
    assert canon.loc[0, "team"] == "SD"
    assert canon.loc[1, "team"] == "LAC"


def test_build_picks_minimal_columns() -> None:
    raw = pd.DataFrame(
        {
            "season": [2014, 2014],
            "round": [1, 7],
            "pick": [29, 247],
            "team": ["NE", "NE"],
            "gsis_id": ["00-0031234", "00-0031235"],
            "pfr_player_name": ["Dominique Easley", "Jemea Thomas"],
            "position": ["DT", "CB"],
            "college": ["Florida", "Georgia Tech"],
            "car_av": [3, 0],
            "w_av": [3, 0],
            "dr_av": [3, 0],
            "games": [22, 0],
            "probowls": [0, 0],
            "allpro": [0, 0],
            "to": [2017, 2014],
        }
    )
    picks = build_picks(raw)

    expected = {
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
    }
    assert expected.issubset(picks.columns)
    assert len(picks) == 2
    assert picks["pro_bowls"].dtype.kind in {"i", "u"}
