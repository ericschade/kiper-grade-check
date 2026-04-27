import numpy as np
import pandas as pd

from kiper_grade_check.grade import compute_expected_av_by_slot


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
    rng = np.random.default_rng(0)
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
