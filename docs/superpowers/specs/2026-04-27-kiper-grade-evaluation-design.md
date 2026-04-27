# Kiper Grade Check — Design

**Date:** 2026-04-27
**Working title:** `kiper-grade-check`
**Goal:** Quantitatively evaluate the historical accuracy of Mel Kiper Jr.'s post-draft team grades for the 2011–2025 NFL drafts by comparing them against an "actual" team draft grade computed from player career outcomes. Output is a reproducible dataset plus a Jupyter notebook with the headline analysis.

## Scope decisions (from brainstorm)

- **Subject:** Kiper's *post-draft team grades* (the letter grades published in the days after each draft), not mock drafts and not pre-draft prospect rankings.
- **Years:** Drafts 2011–2025 (15 most recent).
- **Eval window:** Each row carries an `eval_completeness` flag — `full` (≤2021), `partial` (2022–2023), `too_recent` (≥2024). Headline numbers default to `full` only; partials are visible-but-flagged in the notebook.
- **Data acquisition:** Public sources only, fully automated. Manual top-up paths exist for years where scrapes fail.
- **Output:** Primarily a dataset (CSV + Parquet). Secondary deliverable is a single analysis notebook.
- **Stretch:** Football Outsiders / FTN's Report Card aggregates ~6–10 analysts per year, so we capture all of them — Kiper plus peers — and rank analysts against each other for free.

## Tech stack

- **Python 3.13**, managed with [`uv`](https://github.com/astral-sh/uv).
- **`nflreadpy`** (active replacement for the deprecated `nfl_data_py`) for drafts, careers, contracts, schedules.
- **`httpx` + `selectolax`** for the Football Outsiders / FTN scrape; **`wayback`** package as fallback.
- **`pandas` + `pyarrow`** for transforms and storage. (`nflreadpy` returns polars; we convert at the source-module boundary.)
- **`jupyter`** for the analysis notebook.
- **`pytest`** for tests; **`ruff`** for lint; **`pyright`** for type-checking. CI runs all three on push (no pre-commit hooks — keep it minimal).

## Repo layout

```
kiper-grade-check/
├── README.md
├── pyproject.toml
├── .python-version
├── .gitignore
├── src/kiper_grade_check/
│   ├── sources/                # one module per data source
│   │   ├── nflverse.py             # wraps nflreadpy calls
│   │   └── report_card.py          # scrapes FO + FTN + Wayback
│   ├── transform/              # raw → clean dataframes
│   │   ├── grades.py               # parse "A-", "B+" etc. to numeric 0–12
│   │   ├── careers.py              # career outcomes per pick
│   │   ├── contracts.py            # second-contract derivation
│   │   └── team_wins.py            # 5-yr team wins from schedules
│   ├── grade.py                # the actual-grade formula
│   ├── pipeline.py             # orchestrates fetch → transform → emit
│   └── cli.py                  # `kgc fetch`, `kgc build`, `kgc analyze`
├── data/
│   ├── raw/                    # cached scrapes/parquet pulls; gitignored except small fixtures
│   ├── processed/              # final CSV/Parquet, committed
│   └── manual/                 # hand-entered top-ups (e.g. report_card_2011.csv)
├── notebooks/
│   └── 01-kiper-vs-reality.ipynb
├── tests/
│   └── ...
└── docs/superpowers/specs/
    └── 2026-04-27-kiper-grade-evaluation-design.md   # this file
```

The `sources/` → `transform/` → `grade.py` separation ensures each layer is independently testable and re-runnable. If FTN changes their HTML, only `sources/report_card.py` breaks; the rest of the pipeline runs from cached raw.

## Pipeline stages

Three stages, each with a persisted intermediate so any stage can be re-run in isolation. CLI entrypoints under `kgc`.

```
fetch  →  data/raw/   →  build  →  data/processed/   →  analyze (notebook)
```

### Stage 1 — `kgc fetch [--year YYYY]`

- `nflreadpy.load_draft_picks(seasons=range(2011, 2026))` → `data/raw/draft_picks.parquet`
- `nflreadpy.load_contracts()` → `data/raw/contracts.parquet`
- `nflreadpy.load_schedules(seasons=range(2011, 2030))` → `data/raw/schedules.parquet`
  - We pull through 2029 to cover the 5-year window for the 2025 draft class (`draft_year..draft_year+4` = 2025..2029). Future seasons return empty until played.
- For each year 2011–2025: fetch the **Report Card** page, write raw HTML to `data/raw/report_card/{year}.html`.
  - URL patterns: `footballoutsiders.com/nfl-draft/{year}/{year}-nfl-draft-report-card-report` (2018–2023); `ftnfantasy.com/nfl/{year}-nfl-draft-report-card-report` (2024+); Wayback fallback for everything else and for live-URL failures.

All four are idempotent — re-running just refreshes caches. Each year's HTML is independent so a single failed year doesn't kill the run. Throttled to 1 request/sec; exponential backoff on 5xx.

### Stage 2 — `kgc build`

Produces these processed tables (written as both `.parquet` and `.csv`):

- `picks.parquet` — one row per pick.
- `analyst_grades.parquet` — one row per (year × team × analyst).
- `team_outcomes.parquet` — one row per (year × team), with computed actual grade.
- `comparison.parquet` — join of analyst_grades and team_outcomes; one row per (year × team × analyst).

### Stage 3 — `notebooks/01-kiper-vs-reality.ipynb`

Read-only consumer of `comparison.parquet`. Produces:

- Kiper's correlation (Pearson r, Spearman ρ) with the actual grade, overall and broken out by year/round.
- Leaderboard of all analysts in the Report Card by accuracy.
- Per-team residual table — biggest "Kiper said A, reality said F" misses.

## Data schemas

### `picks.parquet`

| Column | Type | Notes |
|---|---|---|
| `draft_year` | int | 2011–2025 |
| `round` | int | 1–7 |
| `pick_overall` | int | 1 .. last pick of round 7 (varies by year w/ compensatory picks) |
| `team` | str | nflverse 3-letter code |
| `player_id` | str | gsis_id from nflverse |
| `player_name` | str | |
| `position` | str | nflverse position code |
| `college` | str | |
| `career_av` | int | `car_av` from `load_draft_picks` |
| `weighted_av` | int | `w_av` (peak years weighted higher) |
| `draft_team_av` | int | `dr_av` (AV with drafting team only) |
| `games` | int | career games played |
| `pro_bowls` | int | career Pro Bowl selections |
| `all_pros` | int | AP First-Team All-Pros |
| `final_season` | int | last NFL season; null if active |
| `second_contract_same_team` | bool\|null | derived; null when not yet evaluable |
| `eval_completeness` | str | `full` / `partial` / `too_recent` |

### `analyst_grades.parquet`

| Column | Type | Notes |
|---|---|---|
| `draft_year` | int | |
| `team` | str | canonical nflverse code |
| `analyst` | str | "Kiper", "McShay", "Prisco", … |
| `grade_letter` | str | "A+", "A", "A-", … "F" |
| `grade_numeric` | float | 13-point scale: A+=12, A=11, A-=10, …, F=0 |
| `source_url` | str | for traceability |

### `team_outcomes.parquet`

| Column | Type | Notes |
|---|---|---|
| `draft_year` | int | |
| `team` | str | |
| `num_picks` | int | |
| `total_career_av` | int | |
| `total_weighted_av` | int | |
| `total_av_over_expected` | float | **headline**: Σ (actual_av − expected_av_for_slot) |
| `avg_pro_bowls_per_pick` | float | |
| `pct_second_contract` | float | nullable when `eval_completeness != "full"` |
| `team_wins_5yr` | int | regular-season wins, `draft_year..draft_year+4` |
| `actual_grade_z` | float | weighted z-score composite |
| `actual_grade_letter` | str | bucketed for readability |
| `eval_completeness` | str | |

### `comparison.parquet`

One row per (year × team × analyst); the join of `analyst_grades` with all of `team_outcomes`. Adds:

| Column | Type | Notes |
|---|---|---|
| `analyst_grade_numeric` | float | |
| `analyst_grade_z` | float | z-scored within draft year |
| `actual_grade_z` | float | from `team_outcomes` |
| `residual` | float | `actual_grade_z − analyst_grade_z` |

## "Actual grade" formula

### Step 1 — Per-pick AV-over-expected

For each draft slot (pick_overall = 1, 2, 3 …), expected AV is the LOESS-smoothed mean of `career_av` across all picks at that slot from 1980 to present. (Using 1980+ — not just 2011+ — gives a stable curve, especially for late rounds.) Smoothing handles thin counts at the tail.

Per pick: `av_over_expected = career_av − expected_av(pick_overall)`.

### Step 2 — Per-team-per-year, four components

| Component | Formula | Captures |
|---|---|---|
| AV-over-expected | `Σ av_over_expected over team's picks` | did picks produce vs. their slot? |
| Pro Bowls per pick | `mean(pro_bowls)` over picks | peak-talent yield |
| Pct second contract | `mean(second_contract_same_team)` (full eval only) | did team retain its picks? |
| 5-yr team wins | `Σ regular_season_wins, draft_year..draft_year+4` | did the picks contribute to winning? |

### Step 3 — Z-score each component within draft year

Each component is z-scored across the 32 teams in that draft year. Neutralizes era effects; "above average" means the same yardstick across years.

### Step 4 — Weighted composite

```python
actual_grade_z = (
    0.50 * z_av_over_expected
  + 0.20 * z_pro_bowls_per_pick
  + 0.15 * z_pct_second_contract   # if eval_completeness == "full"
  + 0.15 * z_team_wins_5yr
)
```

When `pct_second_contract` is `null`, its 0.15 weight redistributes pro-rata across the other three. Weights are exposed as constants in `grade.py`; the notebook can re-run the formula with alternate weights for sensitivity analysis.

### Step 5 — Letter grade

Fixed thresholds calibrated once to roughly match Kiper's empirical letter-grade distribution across the dataset:

```
z > +1.25 → A+    +0.75..+1.25 → A     +0.25..+0.75 → A-
+0.00..+0.25 → B+   −0.25..0.00 → B   −0.75..−0.25 → B-
−1.25..−0.75 → C+   −1.75..−1.25 → C   −2.25..−1.75 → C-
< −2.25 → D / F
```

Letter grade is **for human readability only**. The headline statistical comparison uses the numeric/z-score scale (letters discard information).

### Step 6 — Comparison

Per-analyst:

- Pearson r and Spearman ρ between `analyst_grade_z` and `actual_grade_z` across all team-years where `eval_completeness == "full"`.
- Same broken out by year, round-1-only, etc.
- "Biggest misses" table sorted by `|residual|`.

## Edge cases & error handling

### Second-contract derivation

Operationally:

```python
contracts_for_player = contracts[contracts.gsis_id == p].sort_values("year_signed")
second = contracts_for_player.iloc[1] if len(contracts_for_player) >= 2 else None
second_contract_same_team = (
    second is not None
    and second.team == draft_team
    and second.year_signed >= draft_year + 3   # suppress restructure noise
)
```

Known noise sources documented inline. Pilot validation: spot-check 50 random rows against OverTheCap web pages before committing the dataset; if error rate > 10%, drop the metric and redistribute its 0.15 weight.

### Team relocations / renames

Maintain a `TEAM_CODE_ALIASES` dict in `transform/careers.py` (small enough — ~3 entries — to keep inline rather than as a CSV file). Known: `OAK→LV` (2020+), `SD→LAC` (2017+), `STL→LAR` (2016+). nflverse keeps `WAS` constant across the rebrand, so no entry needed there.

### Football Outsiders / FTN scrape failures

- Live URL 404 → Wayback fallback.
- Wayback empty → write `data/raw/report_card/{year}.MISSING` stub, build skips with a warning.
- `data/manual/report_card_{year}.csv` (if present) overrides scrape for that year — manual top-up path for the 2011–2013 thin window.
- Throttled to 1 req/sec; exponential backoff on 5xx.

### eval_completeness flag

Deterministic from `draft_year`:

```
draft_year ≤ 2021                 → "full"
draft_year ∈ {2022, 2023}         → "partial"
draft_year ≥ 2024                 → "too_recent"
```

Headline plots default to `full` only.

### nflreadpy / pandas

`nflreadpy` returns polars DataFrames. `sources/nflverse.py` calls `.to_pandas()` at the boundary so the rest of the codebase is uniformly pandas. Documented in that module.

## Testing approach

- **Unit tests** for each `transform/` module:
  - `grades.py`: round-trip parse/format for every grade A+..F; reject malformed.
  - `careers.py`: schema/type assertions on the picks output; spot-checks for known-quantity drafts (e.g. 2014 Hawks; 2017 Saints).
  - `contracts.py`: synthetic fixtures for the four canonical cases (stayed via 2nd contract; restructured; traded then resigned with new team; never got a 2nd).
  - `team_wins.py`: known-totals check (e.g. 2017 Patriots regular-season wins).
  - `grade.py`: z-score and bucketing properties; weight redistribution when `pct_second_contract` is null.
- **Integration test** for the full pipeline against a small fixture year (one draft year with ~5 picks, hand-curated): fetch (mocked) → build → assert the four output tables shape and a few known values.
- **Fixtures** in `tests/fixtures/` — small Parquet/CSV/HTML samples committed to repo. No live network calls in tests.
- **CI**: GitHub Actions workflow running `ruff check`, `pyright`, and `pytest` on push. No live data in CI; tests use fixtures only.

## Out of scope (YAGNI)

- Per-season AV (PFR scraping required; bundled career AV is sufficient).
- Position-specific metrics beyond AV+Pro Bowls (user flagged "I don't know how to benchmark in a vacuum"; AV already absorbs most of this).
- Web dashboard (notebook is enough for v1).
- Pre-2011 drafts (research showed FO archive thins out before then).
- Automated grading of analysts' written commentary (text NLP is a separate project).

## Open risks (carried into implementation plan)

1. **2011–2013 Kiper grade availability** — may require manual top-up via `data/manual/`. Build pipeline supports this; effort is hand-entry for ~96 team-year cells if Wayback fails.
2. **Second-contract metric noise** — pilot validation step gated; we drop the metric if too noisy.
3. **Football Outsiders site instability** — research saw `ECONNREFUSED` on some live URLs. Wayback fallback handles this. Worst case, the Wayback snapshots are good enough for all years.
