# kiper-grade-check

Quantitatively evaluate the historical accuracy of Mel Kiper Jr.'s post-draft team grades for the 2011–2025 NFL drafts.

For each draft, we compare the analyst's team letter grade against an "actual" grade computed from player career outcomes:

- **AV-over-expected** — career Approximate Value relative to what an average pick at that draft slot produces
- **Pro Bowls per pick**
- **Pct of picks signing a 2nd contract with the drafting team**
- **5-year team wins post-draft**

Football Outsiders / FTN aggregate ~6–10 analysts per year, so we capture Kiper's peers (McShay, Prisco, Brooks, Pompei, Cole, …) for free and rank them all.

## Status

Implemented and tested. See [`docs/superpowers/specs/2026-04-27-kiper-grade-evaluation-design.md`](docs/superpowers/specs/2026-04-27-kiper-grade-evaluation-design.md) for the design.

## Quick start

```sh
uv sync
uv run kgc fetch        # pull raw data into data/raw/
uv run kgc build        # transform to data/processed/{picks,analyst_grades,team_outcomes,comparison}.{parquet,csv}
jupyter lab notebooks/01-kiper-vs-reality.ipynb
```

## Data sources

- **Player / draft / contract / schedule data** — [`nflreadpy`](https://github.com/nflverse/nflreadpy) (community-maintained nflverse data). Stable, comprehensive, and the headline metrics (career AV, Pro Bowls, schedule wins) come straight from this single source.
- **Analyst grades** — Football Outsiders / FTN "Draft Report Card Report" (with Wayback Machine fallback for missing years).

### Important caveat about analyst grade coverage

The Football Outsiders / FTN Report Card pages publish each team's *highest* and *lowest* analyst grade (with attribution) plus a consensus GPA — they do **not** publish a full per-team × per-analyst grid. As a result, each named analyst (including Kiper) appears in only ~3-5 rows per year — only the teams where they were the high or low.

The pipeline still computes the comparison correctly on whatever data exists, but the Kiper-specific sample is much sparser than would be ideal (60-80 grades across 15 years rather than 480). The headline correlation is therefore statistically thinner than this project originally hoped for.

If you want fuller per-analyst coverage, the natural extension is to scrape Kiper's actual ESPN Insider articles directly — feasible but ESPN's URLs change yearly and most years are paywalled. That extension is left as future work.
