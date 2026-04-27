import pandas as pd

from kiper_grade_check.transform.contracts import derive_second_contract_same_team


def _picks() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": ["P1", "P2", "P3", "P4", "P5"],
            "draft_year": [2014, 2014, 2014, 2014, 2022],
            "team": ["NE", "NE", "NE", "NE", "NE"],
            "eval_completeness": ["full", "full", "full", "full", "partial"],
        }
    )


def _contracts(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["gsis_id", "team", "year_signed"])


def test_player_resigned_with_drafting_team_after_rookie_deal() -> None:
    contracts = _contracts(
        [
            {"gsis_id": "P1", "team": "NE", "year_signed": 2014},
            {"gsis_id": "P1", "team": "NE", "year_signed": 2018},
        ]
    )
    out = derive_second_contract_same_team(_picks(), contracts)
    assert out.set_index("player_id").loc["P1", "second_contract_same_team"] is True


def test_restructured_in_year_2_does_not_count() -> None:
    contracts = _contracts(
        [
            {"gsis_id": "P2", "team": "NE", "year_signed": 2014},
            {"gsis_id": "P2", "team": "NE", "year_signed": 2015},
        ]
    )
    out = derive_second_contract_same_team(_picks(), contracts)
    assert out.set_index("player_id").loc["P2", "second_contract_same_team"] is False


def test_signed_with_different_team() -> None:
    contracts = _contracts(
        [
            {"gsis_id": "P3", "team": "NE", "year_signed": 2014},
            {"gsis_id": "P3", "team": "DAL", "year_signed": 2018},
        ]
    )
    out = derive_second_contract_same_team(_picks(), contracts)
    assert out.set_index("player_id").loc["P3", "second_contract_same_team"] is False


def test_never_got_second_contract() -> None:
    contracts = _contracts([{"gsis_id": "P4", "team": "NE", "year_signed": 2014}])
    out = derive_second_contract_same_team(_picks(), contracts)
    assert out.set_index("player_id").loc["P4", "second_contract_same_team"] is False


def test_partial_eval_returns_null() -> None:
    contracts = _contracts([{"gsis_id": "P5", "team": "NE", "year_signed": 2022}])
    out = derive_second_contract_same_team(_picks(), contracts)
    val = out.set_index("player_id").loc["P5", "second_contract_same_team"]
    assert val is None or pd.isna(val)
