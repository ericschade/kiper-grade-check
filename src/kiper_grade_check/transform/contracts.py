"""Derive the boolean 'second contract with drafting team' per pick.

`nflreadpy.load_contracts()` doesn't label rookie vs extension. We approximate:
- Sort a player's contracts by year_signed
- The first row is treated as the rookie deal
- The second row qualifies as a 'second contract' if team == draft_team AND
  year_signed >= draft_year + 3 (suppresses restructures of the rookie deal,
  which usually happen in years 1–2).
- Players with eval_completeness != "full" return None (not yet decidable).
"""

from __future__ import annotations

import pandas as pd


def derive_second_contract_same_team(
    picks: pd.DataFrame, contracts: pd.DataFrame
) -> pd.DataFrame:
    """Return a DataFrame with columns [player_id, second_contract_same_team]."""
    out = picks[["player_id", "draft_year", "team", "eval_completeness"]].copy()
    contracts_sorted = contracts.sort_values(["gsis_id", "year_signed"])

    rank = contracts_sorted.groupby("gsis_id").cumcount()
    second = contracts_sorted[rank == 1].set_index("gsis_id")

    def evaluate(row: pd.Series) -> object:
        if row["eval_completeness"] != "full":
            return None
        pid = row["player_id"]
        if pid not in second.index:
            return False
        s = second.loc[pid]
        return bool(s["team"] == row["team"] and s["year_signed"] >= row["draft_year"] + 3)

    out["second_contract_same_team"] = out.apply(evaluate, axis=1)
    return out[["player_id", "second_contract_same_team"]]
