# kiper-grade-check

Quantitatively evaluate the historical accuracy of Mel Kiper Jr.'s post-draft team grades for the 2011–2025 NFL drafts.

For each draft, we compare the analyst's team letter grade against an "actual" grade computed from player career outcomes:

- **AV-over-expected** — career Approximate Value relative to what an average pick at that draft slot produces
- **Pro Bowls per pick**
- **Pct of picks signing a 2nd contract with the drafting team**
- **5-year team wins post-draft**

Football Outsiders / FTN aggregate ~6–10 analysts per year, so we capture Kiper's peers (McShay, Prisco, Brooks, Pompei, Cole, …) for free and rank them all.

## Status

Project under construction. See [`docs/superpowers/specs/2026-04-27-kiper-grade-evaluation-design.md`](docs/superpowers/specs/2026-04-27-kiper-grade-evaluation-design.md) for the design.

## Quick start (once implemented)

```sh
uv sync
uv run kgc fetch        # pull raw data into data/raw/
uv run kgc build        # transform to data/processed/{picks,analyst_grades,team_outcomes,comparison}.{parquet,csv}
jupyter lab notebooks/01-kiper-vs-reality.ipynb
```

## Data sources

- **Player / draft / contract / schedule data** — [`nflreadpy`](https://github.com/nflverse/nflreadpy) (community-maintained nflverse data)
- **Analyst grades** — Football Outsiders / FTN "Draft Report Card Report" (with Wayback Machine fallback)
