import numpy as np
import pandas as pd

from kiper_grade_check.grade import (
    DEFAULT_WEIGHTS,
    bucket_z_to_letter,
    compute_actual_grade,
    compute_expected_av_by_slot,
    z_within_year,
)


def test_expected_av_decreases_with_pick() -> None:
    rng = np.random.default_rng(42)
    rows = []
    for pick in range(1, 261):
        true_mean = max(0.5, 60 * np.exp(-pick / 60))
        for _ in range(20):
            rows.append({"pick_overall": pick, "career_av": rng.poisson(true_mean)})
    historical = pd.DataFrame(rows)

    curve = compute_expected_av_by_slot(historical)
    assert isinstance(curve, pd.DataFrame)
    assert {"pick_overall", "expected_av"}.issubset(curve.columns)
    assert curve["expected_av"].iloc[0] > curve["expected_av"].iloc[-1]
    assert curve["expected_av"].is_monotonic_decreasing or (curve["expected_av"].diff().mean() < 0)


def test_expected_av_smooths_thin_slots() -> None:
    rows = []
    for pick in range(1, 261):
        if pick % 2 == 0:
            rows.append({"pick_overall": pick, "career_av": 50})
        else:
            rows.append({"pick_overall": pick, "career_av": 0})
    historical = pd.DataFrame(rows)
    curve = compute_expected_av_by_slot(historical)
    diffs = curve["expected_av"].diff().abs().dropna()
    assert diffs.max() < 25


def test_bucket_z_to_letter_thresholds() -> None:
    assert bucket_z_to_letter(2.0) == "A+"
    assert bucket_z_to_letter(1.0) == "A"
    assert bucket_z_to_letter(0.5) == "A-"
    assert bucket_z_to_letter(0.1) == "B+"
    assert bucket_z_to_letter(-0.1) == "B"
    assert bucket_z_to_letter(-0.5) == "B-"
    assert bucket_z_to_letter(-1.0) == "C+"
    assert bucket_z_to_letter(-1.5) == "C"
    assert bucket_z_to_letter(-2.0) == "C-"
    assert bucket_z_to_letter(-3.0) == "F"


def test_z_within_year_normalizes_per_group() -> None:
    df = pd.DataFrame(
        {
            "draft_year": [2014, 2014, 2014, 2014, 2022, 2022, 2022, 2022],
            "value": [10, 20, 30, 40, 1, 2, 3, 4],
        }
    )
    out = z_within_year(df, "value")
    g = out.groupby("draft_year")["value_z"].mean()
    assert (g.abs() < 1e-9).all()  # type: ignore[union-attr]


def test_compute_actual_grade_full_eval_uses_all_four_components() -> None:
    df = pd.DataFrame(
        {
            "draft_year": [2014, 2014, 2014, 2014],
            "team": ["A", "B", "C", "D"],
            "total_av_over_expected": [10.0, 5.0, -5.0, -10.0],
            "avg_pro_bowls_per_pick": [1.0, 0.5, 0.2, 0.0],
            "pct_second_contract": [0.6, 0.4, 0.3, 0.1],
            "team_wins_5yr": [50, 45, 30, 20],
            "eval_completeness": ["full"] * 4,
        }
    )
    out = compute_actual_grade(df)
    assert {"actual_grade_z", "actual_grade_letter"}.issubset(out.columns)
    z = out.set_index("team")["actual_grade_z"]
    assert z["A"] > z["B"] > z["C"] > z["D"]


def test_compute_actual_grade_partial_eval_redistributes_weight() -> None:
    df = pd.DataFrame(
        {
            "draft_year": [2022, 2022, 2022, 2022],
            "team": ["A", "B", "C", "D"],
            "total_av_over_expected": [10.0, 5.0, -5.0, -10.0],
            "avg_pro_bowls_per_pick": [1.0, 0.5, 0.2, 0.0],
            "pct_second_contract": [None, None, None, None],
            "team_wins_5yr": [40, 35, 25, 20],
            "eval_completeness": ["partial"] * 4,
        }
    )
    out = compute_actual_grade(df)
    z = out.set_index("team")["actual_grade_z"]
    assert z["A"] > z["B"] > z["C"] > z["D"]


def test_default_weights_sum_to_one() -> None:
    assert abs(sum(DEFAULT_WEIGHTS.values()) - 1.0) < 1e-9
