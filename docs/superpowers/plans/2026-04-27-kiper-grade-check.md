# Kiper Grade Check Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python pipeline that produces four parquet/CSV tables comparing Mel Kiper Jr.'s post-draft team grades (and his peers') against an "actual" grade computed from player career outcomes, plus a Jupyter notebook that reports the headline correlations.

**Architecture:** Three-stage pipeline (`fetch` → `build` → `analyze`). Each stage persists its outputs so any stage can be re-run from cached intermediates. Code is split: `sources/` does I/O, `transform/` does dataframe shaping, `grade.py` does the math, `pipeline.py` orchestrates, `cli.py` is the entrypoint. Pandas-only inside the codebase; polars→pandas conversion happens at the `sources/nflverse.py` boundary.

**Tech Stack:** Python 3.13, `uv`, `nflreadpy`, `httpx`, `selectolax`, `wayback`, `pandas`, `pyarrow`, `numpy`, `statsmodels` (for LOESS), `scipy` (for z-scoring), `typer` (CLI), `jupyter`, `pytest`, `ruff`, `pyright`.

**Working directory:** `/Users/ericschade/Documents/GitHub/kiper-grade-check`. Repo already initialized; initial scaffold (README, .gitignore, package skeleton, design spec) committed at `05de422`. Remote: `https://github.com/ericschade/kiper-grade-check`.

---

## Phase 0 — Project bootstrap

### Task 1: `pyproject.toml` + uv sync

**Files:**
- Create: `pyproject.toml`
- Create: `.python-version`

- [ ] **Step 1: Write `.python-version`**

```
3.13
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "kiper-grade-check"
version = "0.1.0"
description = "Evaluate Mel Kiper Jr.'s historical NFL draft grades against actual player outcomes."
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "nflreadpy>=0.1.0",
    "pandas>=2.2",
    "pyarrow>=15",
    "numpy>=1.26",
    "scipy>=1.13",
    "statsmodels>=0.14",
    "httpx>=0.27",
    "selectolax>=0.3.21",
    "wayback>=0.4",
    "typer>=0.12",
    "rich>=13.7",
]

[project.scripts]
kgc = "kiper_grade_check.cli:app"

[dependency-groups]
dev = [
    "pytest>=8",
    "pytest-cov>=5",
    "ruff>=0.6",
    "pyright>=1.1.380",
    "jupyter>=1",
    "ipykernel>=6",
    "matplotlib>=3.9",
    "seaborn>=0.13",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/kiper_grade_check"]

[tool.ruff]
line-length = 100
target-version = "py313"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP", "B", "SIM", "RET"]
ignore = ["E501"]  # line-too-long handled by formatter

[tool.pyright]
include = ["src", "tests"]
pythonVersion = "3.13"
typeCheckingMode = "basic"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra --strict-markers"
```

- [ ] **Step 3: Run `uv sync`**

Run: `uv sync`
Expected: creates `.venv/`, resolves dependencies, no errors. If `nflreadpy>=0.1.0` is unavailable, drop the lower bound to whatever PyPI returns (`uv pip index versions nflreadpy`) and pin to that.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml .python-version uv.lock
git commit -m "chore: pyproject + uv lockfile"
```

---

### Task 2: Pre-commit, ruff, pyright sanity check

**Files:**
- Create: `src/kiper_grade_check/__init__.py` (already exists empty; populate with version)

- [ ] **Step 1: Populate `src/kiper_grade_check/__init__.py`**

```python
"""Kiper Grade Check — evaluate NFL draft analyst grades against player outcomes."""

__version__ = "0.1.0"
```

- [ ] **Step 2: Run lint and type-check**

Run: `uv run ruff check . && uv run pyright`
Expected: ruff clean, pyright reports 0 errors (may report 0 informational notes — fine).

- [ ] **Step 3: Commit**

```bash
git add src/kiper_grade_check/__init__.py
git commit -m "chore: package version + lint/type clean"
```

---

### Task 3: pytest scaffolding

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/test_smoke.py`
- Create: `tests/fixtures/.gitkeep`

- [ ] **Step 1: Write `tests/__init__.py`**

Empty file.

- [ ] **Step 2: Write `tests/conftest.py`**

```python
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES
```

- [ ] **Step 3: Write `tests/test_smoke.py`**

```python
from kiper_grade_check import __version__


def test_version() -> None:
    assert __version__ == "0.1.0"
```

- [ ] **Step 4: Run pytest**

Run: `uv run pytest -v`
Expected: 1 test passes.

- [ ] **Step 5: Commit**

```bash
mkdir -p tests/fixtures
touch tests/fixtures/.gitkeep
git add tests/
git commit -m "test: pytest scaffolding + smoke test"
```

---

## Phase 1 — Sources layer

### Task 4: `transform/grades.py` (letter ↔ numeric, parse helpers)

We do this **before** the sources layer because the report-card scraper will use the parser.

**Files:**
- Create: `src/kiper_grade_check/transform/grades.py`
- Create: `tests/transform/__init__.py`
- Create: `tests/transform/test_grades.py`

- [ ] **Step 1: Write the failing tests**

`tests/transform/test_grades.py`:

```python
import pytest

from kiper_grade_check.transform.grades import (
    GRADE_TO_NUMERIC,
    grade_to_numeric,
    is_valid_grade,
    numeric_to_grade,
    parse_grade,
)


@pytest.mark.parametrize(
    "letter,expected",
    [
        ("A+", 12.0),
        ("A", 11.0),
        ("A-", 10.0),
        ("B+", 9.0),
        ("B", 8.0),
        ("B-", 7.0),
        ("C+", 6.0),
        ("C", 5.0),
        ("C-", 4.0),
        ("D+", 3.0),
        ("D", 2.0),
        ("D-", 1.0),
        ("F", 0.0),
    ],
)
def test_grade_to_numeric_canonical(letter: str, expected: float) -> None:
    assert grade_to_numeric(letter) == expected


def test_numeric_to_grade_round_trip() -> None:
    for letter, value in GRADE_TO_NUMERIC.items():
        assert numeric_to_grade(value) == letter


def test_parse_grade_strips_whitespace_and_normalizes_case() -> None:
    assert parse_grade(" a- ") == "A-"
    assert parse_grade("b+") == "B+"


def test_parse_grade_handles_unicode_minus() -> None:
    # Some scraped sources use the Unicode minus sign U+2212
    assert parse_grade("A−") == "A-"


def test_parse_grade_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        parse_grade("Z+")


def test_is_valid_grade() -> None:
    assert is_valid_grade("A+")
    assert is_valid_grade("F")
    assert not is_valid_grade("A++")
    assert not is_valid_grade("")
    assert not is_valid_grade("X")
```

- [ ] **Step 2: Verify the test fails**

Run: `uv run pytest tests/transform/test_grades.py -v`
Expected: ImportError (module doesn't exist yet).

- [ ] **Step 3: Implement `src/kiper_grade_check/transform/grades.py`**

```python
"""Letter ↔ numeric grade conversion on a 13-point scale (A+ = 12, F = 0)."""

from __future__ import annotations

GRADE_TO_NUMERIC: dict[str, float] = {
    "A+": 12.0,
    "A": 11.0,
    "A-": 10.0,
    "B+": 9.0,
    "B": 8.0,
    "B-": 7.0,
    "C+": 6.0,
    "C": 5.0,
    "C-": 4.0,
    "D+": 3.0,
    "D": 2.0,
    "D-": 1.0,
    "F": 0.0,
}

NUMERIC_TO_GRADE: dict[float, str] = {v: k for k, v in GRADE_TO_NUMERIC.items()}


def parse_grade(raw: str) -> str:
    """Normalize a raw grade string to canonical form (e.g. 'a-' → 'A-').

    Handles common scraped quirks: whitespace, lowercase, Unicode minus.
    Raises ValueError if the result is not a recognized letter grade.
    """
    if raw is None:
        raise ValueError("grade cannot be None")
    cleaned = raw.strip().upper().replace("−", "-")
    if cleaned not in GRADE_TO_NUMERIC:
        raise ValueError(f"unknown grade: {raw!r}")
    return cleaned


def grade_to_numeric(letter: str) -> float:
    return GRADE_TO_NUMERIC[parse_grade(letter)]


def numeric_to_grade(value: float) -> str:
    if value not in NUMERIC_TO_GRADE:
        raise ValueError(f"no canonical letter for value {value!r}")
    return NUMERIC_TO_GRADE[value]


def is_valid_grade(raw: str) -> bool:
    try:
        parse_grade(raw)
    except (ValueError, AttributeError):
        return False
    return True
```

- [ ] **Step 4: Verify tests pass**

Run: `uv run pytest tests/transform/test_grades.py -v`
Expected: 17 tests pass (parametrized 13 + 4 others).

- [ ] **Step 5: Commit**

```bash
git add src/kiper_grade_check/transform/grades.py tests/transform/
git commit -m "feat(grades): letter↔numeric grade conversion"
```

---

### Task 5: `sources/nflverse.py` — wrap nflreadpy and convert polars→pandas

**Files:**
- Create: `src/kiper_grade_check/sources/nflverse.py`
- Create: `tests/sources/__init__.py`
- Create: `tests/sources/test_nflverse.py`

- [ ] **Step 1: Write the failing tests**

`tests/sources/test_nflverse.py`:

```python
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import polars as pl
import pytest

from kiper_grade_check.sources import nflverse


@pytest.fixture
def fake_polars_picks() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "season": [2014, 2014],
            "round": [1, 2],
            "pick": [32, 64],
            "team": ["NE", "NE"],
            "gsis_id": ["00-0031234", "00-0031235"],
            "pfr_player_name": ["Player One", "Player Two"],
            "position": ["WR", "DT"],
            "college": ["Florida", "FSU"],
            "car_av": [42, 18],
            "w_av": [40, 16],
            "dr_av": [42, 10],
            "games": [120, 80],
            "probowls": [2, 0],
            "allpro": [0, 0],
            "to": [2024, 2022],
        }
    )


def test_load_draft_picks_returns_pandas_with_expected_columns(
    tmp_path: Path, fake_polars_picks: pl.DataFrame
) -> None:
    with patch("kiper_grade_check.sources.nflverse.nflreadpy.load_draft_picks", return_value=fake_polars_picks):
        df = nflverse.load_draft_picks(seasons=range(2014, 2015), cache_dir=tmp_path)

    assert isinstance(df, pd.DataFrame)
    assert {"season", "round", "pick", "team", "gsis_id", "car_av", "w_av", "probowls"}.issubset(df.columns)
    assert (tmp_path / "draft_picks.parquet").exists()


def test_load_draft_picks_uses_cache_when_present(tmp_path: Path, fake_polars_picks: pl.DataFrame) -> None:
    cached = fake_polars_picks.to_pandas()
    cached.to_parquet(tmp_path / "draft_picks.parquet")

    # nflreadpy should NOT be called if cache hits
    with patch("kiper_grade_check.sources.nflverse.nflreadpy.load_draft_picks") as mock:
        df = nflverse.load_draft_picks(seasons=range(2014, 2015), cache_dir=tmp_path)
        mock.assert_not_called()

    pd.testing.assert_frame_equal(df.reset_index(drop=True), cached.reset_index(drop=True))


def test_load_draft_picks_force_refresh_bypasses_cache(
    tmp_path: Path, fake_polars_picks: pl.DataFrame
) -> None:
    cached = fake_polars_picks.to_pandas()
    cached.to_parquet(tmp_path / "draft_picks.parquet")

    with patch(
        "kiper_grade_check.sources.nflverse.nflreadpy.load_draft_picks",
        return_value=fake_polars_picks,
    ) as mock:
        nflverse.load_draft_picks(seasons=range(2014, 2015), cache_dir=tmp_path, force=True)
        mock.assert_called_once()
```

- [ ] **Step 2: Verify it fails**

Run: `uv run pytest tests/sources/test_nflverse.py -v`
Expected: ImportError (module doesn't exist).

- [ ] **Step 3: Implement `src/kiper_grade_check/sources/nflverse.py`**

```python
"""Thin wrappers around nflreadpy that cache to parquet and return pandas DataFrames.

nflreadpy returns polars DataFrames; the rest of the codebase is pandas. We convert
at this boundary so callers never see polars.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import nflreadpy
import pandas as pd

DEFAULT_CACHE_DIR = Path("data/raw")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _to_pandas(obj) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj
    return obj.to_pandas()


def load_draft_picks(
    seasons: Iterable[int],
    cache_dir: Path = DEFAULT_CACHE_DIR,
    force: bool = False,
) -> pd.DataFrame:
    _ensure_dir(cache_dir)
    cache = cache_dir / "draft_picks.parquet"
    if cache.exists() and not force:
        return pd.read_parquet(cache)

    raw = nflreadpy.load_draft_picks(seasons=list(seasons))
    df = _to_pandas(raw)
    df.to_parquet(cache, index=False)
    return df


def load_contracts(cache_dir: Path = DEFAULT_CACHE_DIR, force: bool = False) -> pd.DataFrame:
    _ensure_dir(cache_dir)
    cache = cache_dir / "contracts.parquet"
    if cache.exists() and not force:
        return pd.read_parquet(cache)

    raw = nflreadpy.load_contracts()
    df = _to_pandas(raw)
    df.to_parquet(cache, index=False)
    return df


def load_schedules(
    seasons: Iterable[int],
    cache_dir: Path = DEFAULT_CACHE_DIR,
    force: bool = False,
) -> pd.DataFrame:
    _ensure_dir(cache_dir)
    cache = cache_dir / "schedules.parquet"
    if cache.exists() and not force:
        return pd.read_parquet(cache)

    raw = nflreadpy.load_schedules(seasons=list(seasons))
    df = _to_pandas(raw)
    df.to_parquet(cache, index=False)
    return df
```

- [ ] **Step 4: Verify tests pass**

Run: `uv run pytest tests/sources/test_nflverse.py -v`
Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/kiper_grade_check/sources/nflverse.py tests/sources/
git commit -m "feat(sources): nflreadpy wrapper with parquet caching"
```

---

### Task 6: `sources/report_card.py` — fetch FO/FTN HTML with Wayback fallback

**Files:**
- Create: `src/kiper_grade_check/sources/report_card.py`
- Create: `tests/sources/test_report_card.py`
- Create: `tests/fixtures/report_card_2020_sample.html`

This task scrapes the Football Outsiders / FTN "Draft Report Card Report" pages and extracts a long-format table of `(year, team, analyst, grade)`. We'll do **fetch** + **parse** in this task; both functions are independently testable.

- [ ] **Step 1: Capture a real fixture**

Run from a shell once (NOT in tests):

```bash
uv run python - <<'PY'
import httpx
url = "https://www.footballoutsiders.com/nfl-draft/2020/2020-nfl-draft-report-card-report"
r = httpx.get(url, follow_redirects=True, timeout=30, headers={"User-Agent": "kiper-grade-check/0.1 research"})
r.raise_for_status()
open("tests/fixtures/report_card_2020_sample.html", "w").write(r.text)
print("saved", len(r.text), "bytes")
PY
```

If the URL is dead/ECONNREFUSED, fall back to Wayback:

```bash
uv run python - <<'PY'
import httpx
url = "https://web.archive.org/web/2020/https://www.footballoutsiders.com/nfl-draft/2020/2020-nfl-draft-report-card-report"
r = httpx.get(url, follow_redirects=True, timeout=60, headers={"User-Agent": "kiper-grade-check/0.1 research"})
r.raise_for_status()
open("tests/fixtures/report_card_2020_sample.html", "w").write(r.text)
PY
```

If both fail entirely, manually save one Report Card year from a browser to that path. The fixture only needs to be one real page that contains the analyst-grade table for testing.

- [ ] **Step 2: Inspect the fixture and document the table structure**

Open the saved HTML in a browser or text editor. Identify the CSS selector(s) that locate the analyst-grade table. Football Outsiders typically uses an HTML `<table>` with team names as rows and analyst names as columns. Note the selector — you'll need it in `_parse_report_card_html` below.

If after inspection it turns out the table structure differs from what the design assumed (e.g. it's images, or a JS-rendered table), update this task's parsing logic. Document the actual structure inline in the parser.

- [ ] **Step 3: Write the failing tests**

`tests/sources/test_report_card.py`:

```python
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from kiper_grade_check.sources.report_card import (
    REPORT_CARD_URLS,
    fetch_report_card_html,
    parse_report_card_html,
)


def test_report_card_url_pattern_2020() -> None:
    assert "footballoutsiders.com" in REPORT_CARD_URLS(2020)["live"]
    assert "web.archive.org" in REPORT_CARD_URLS(2020)["wayback"]


def test_report_card_url_pattern_2024_uses_ftn() -> None:
    assert "ftnfantasy.com" in REPORT_CARD_URLS(2024)["live"]


def test_parse_report_card_html_extracts_long_format(fixtures_dir: Path) -> None:
    html = (fixtures_dir / "report_card_2020_sample.html").read_text()
    df = parse_report_card_html(html, draft_year=2020, source_url="https://example.test")

    # Long-format: one row per (team, analyst)
    assert {"draft_year", "team", "analyst", "grade_letter", "grade_numeric", "source_url"}.issubset(df.columns)
    assert (df["draft_year"] == 2020).all()
    assert df["analyst"].nunique() >= 3   # at least Kiper + 2 others
    assert df["team"].nunique() >= 28      # at least 28 of 32 teams (allow for parsing slack)
    assert df["grade_letter"].isin(
        {"A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D+", "D", "D-", "F"}
    ).all()
    assert (df["grade_numeric"] >= 0).all()


def test_fetch_report_card_html_falls_back_to_wayback_on_404(tmp_path: Path) -> None:
    cache_dir = tmp_path / "report_card"

    live_response = MagicMock()
    live_response.status_code = 404
    live_response.text = ""
    wayback_response = MagicMock()
    wayback_response.status_code = 200
    wayback_response.text = "<html>WAYBACK CONTENT</html>"
    wayback_response.raise_for_status = MagicMock()

    def fake_get(url: str, **kwargs):
        if "web.archive.org" in url:
            return wayback_response
        return live_response

    with patch("kiper_grade_check.sources.report_card.httpx.Client") as MockClient:
        MockClient.return_value.__enter__.return_value.get.side_effect = fake_get
        path = fetch_report_card_html(2020, cache_dir=cache_dir)

    assert path.read_text() == "<html>WAYBACK CONTENT</html>"


def test_fetch_report_card_writes_missing_stub_when_both_sources_fail(tmp_path: Path) -> None:
    cache_dir = tmp_path / "report_card"
    fail = MagicMock()
    fail.status_code = 404
    fail.text = ""

    with patch("kiper_grade_check.sources.report_card.httpx.Client") as MockClient:
        MockClient.return_value.__enter__.return_value.get.return_value = fail
        path = fetch_report_card_html(2011, cache_dir=cache_dir)

    assert path.suffix == ".MISSING"
    assert path.exists()


def test_fetch_report_card_uses_cache_when_present(tmp_path: Path) -> None:
    cache_dir = tmp_path / "report_card"
    cache_dir.mkdir(parents=True)
    cached = cache_dir / "2020.html"
    cached.write_text("<html>cached</html>")

    with patch("kiper_grade_check.sources.report_card.httpx.Client") as MockClient:
        path = fetch_report_card_html(2020, cache_dir=cache_dir)
        MockClient.assert_not_called()

    assert path == cached
```

- [ ] **Step 4: Verify tests fail**

Run: `uv run pytest tests/sources/test_report_card.py -v`
Expected: ImportError.

- [ ] **Step 5: Implement `src/kiper_grade_check/sources/report_card.py`**

```python
"""Fetch and parse the Football Outsiders / FTN 'Draft Report Card Report' pages.

Each page aggregates ~6–10 analysts' team draft grades for one draft year. We extract
a long-format dataframe with one row per (team, analyst).

URL patterns:
    - 2018–2023: footballoutsiders.com/nfl-draft/{year}/{year}-nfl-draft-report-card-report
    - 2024+:     ftnfantasy.com/nfl/{year}-nfl-draft-report-card-report
    - <2018 and any failure: Wayback Machine

If both live and Wayback fail, write a `{year}.MISSING` stub so the build stage
can skip that year with a logged warning. A `data/manual/report_card_{year}.csv`
override is supported by the build stage (not this module).
"""

from __future__ import annotations

import time
from pathlib import Path

import httpx
import pandas as pd
from selectolax.parser import HTMLParser

from kiper_grade_check.transform.grades import grade_to_numeric, parse_grade

USER_AGENT = "kiper-grade-check/0.1 (research; +https://github.com/ericschade/kiper-grade-check)"
REQUEST_INTERVAL_SEC = 1.0


def REPORT_CARD_URLS(year: int) -> dict[str, str]:
    if year >= 2024:
        live = f"https://ftnfantasy.com/nfl/{year}-nfl-draft-report-card-report"
    else:
        live = f"https://www.footballoutsiders.com/nfl-draft/{year}/{year}-nfl-draft-report-card-report"
    wayback = f"https://web.archive.org/web/{year}/{live}"
    return {"live": live, "wayback": wayback}


def _try_get(client: httpx.Client, url: str) -> str | None:
    try:
        r = client.get(url, follow_redirects=True, timeout=30)
    except httpx.HTTPError:
        return None
    if r.status_code != 200 or not r.text.strip():
        return None
    return r.text


def fetch_report_card_html(
    year: int,
    cache_dir: Path = Path("data/raw/report_card"),
) -> Path:
    """Fetch the Report Card HTML for a year. Returns path to cached file or .MISSING stub."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"{year}.html"
    missing = cache_dir / f"{year}.MISSING"
    if cached.exists():
        return cached
    if missing.exists():
        return missing

    urls = REPORT_CARD_URLS(year)
    headers = {"User-Agent": USER_AGENT}

    with httpx.Client(headers=headers) as client:
        time.sleep(REQUEST_INTERVAL_SEC)
        body = _try_get(client, urls["live"])
        if body is None:
            time.sleep(REQUEST_INTERVAL_SEC)
            body = _try_get(client, urls["wayback"])

    if body is None:
        missing.write_text(f"# fetch failed for {year}\n# urls tried: {urls}\n")
        return missing

    cached.write_text(body)
    return cached


def parse_report_card_html(html: str, draft_year: int, source_url: str) -> pd.DataFrame:
    """Extract a long-format (team, analyst, grade) dataframe from one Report Card page.

    Football Outsiders / FTN render the grades as an HTML <table> with team rows and
    analyst columns. The first column is the team name; subsequent columns are analyst
    grades. We pivot from wide to long.

    If the page structure differs (e.g. multiple tables, or an inline rich-text grid),
    update the table-locator below. The fixture in tests/fixtures/ documents the structure
    we expect.
    """
    tree = HTMLParser(html)

    # Locate the grade table. Heuristic: the first <table> whose header row
    # contains "Team" and at least one column whose name matches a known analyst.
    KNOWN_ANALYSTS = {
        "Kiper", "McShay", "Prisco", "Brooks", "Pompei", "Cole", "Pauline",
        "Schein", "Florio", "Brandt", "Reuter", "PFF", "Trapasso",
    }

    target_table = None
    for table in tree.css("table"):
        headers = [
            (cell.text(strip=True) or "").strip()
            for cell in table.css("thead th, tr:nth-child(1) th, tr:nth-child(1) td")
        ]
        # Match header that looks like ["Team", <analyst>, <analyst>, ...]
        if not headers:
            continue
        if headers[0].lower().startswith("team") and any(h in KNOWN_ANALYSTS for h in headers[1:]):
            target_table = table
            analyst_headers = headers[1:]
            break

    if target_table is None:
        return pd.DataFrame(
            columns=["draft_year", "team", "analyst", "grade_letter", "grade_numeric", "source_url"]
        )

    rows: list[dict] = []
    body_rows = target_table.css("tbody tr") or target_table.css("tr")[1:]
    for tr in body_rows:
        cells = [(cell.text(strip=True) or "").strip() for cell in tr.css("th, td")]
        if not cells or len(cells) < 2:
            continue
        team = cells[0]
        for analyst, raw_grade in zip(analyst_headers, cells[1:]):
            if analyst not in KNOWN_ANALYSTS:
                continue
            if not raw_grade:
                continue
            try:
                letter = parse_grade(raw_grade)
            except ValueError:
                continue
            rows.append(
                {
                    "draft_year": draft_year,
                    "team": team,
                    "analyst": analyst,
                    "grade_letter": letter,
                    "grade_numeric": grade_to_numeric(letter),
                    "source_url": source_url,
                }
            )

    return pd.DataFrame(rows)
```

- [ ] **Step 6: Verify tests pass**

Run: `uv run pytest tests/sources/test_report_card.py -v`
Expected: 5 tests pass. The `test_parse_report_card_html_extracts_long_format` test will only pass if the fixture HTML's table structure matches the parser. If not, debug the parser using the fixture before continuing.

- [ ] **Step 7: Commit**

```bash
git add src/kiper_grade_check/sources/report_card.py tests/sources/test_report_card.py tests/fixtures/report_card_2020_sample.html
git commit -m "feat(sources): fetch + parse FO/FTN draft report cards"
```

---

## Phase 2 — Transform layer

### Task 7: `transform/careers.py` — picks dataframe with eval_completeness

**Files:**
- Create: `src/kiper_grade_check/transform/careers.py`
- Create: `tests/transform/test_careers.py`

- [ ] **Step 1: Write the failing tests**

```python
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
    assert canon.loc[0, "team"] == "OAK"   # 2019 stays Oakland
    assert canon.loc[1, "team"] == "LV"    # 2020 becomes Las Vegas


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
```

- [ ] **Step 2: Verify it fails**

Run: `uv run pytest tests/transform/test_careers.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `src/kiper_grade_check/transform/careers.py`**

```python
"""Build the per-pick dataframe from raw nflverse draft data."""

from __future__ import annotations

from typing import Literal

import pandas as pd

EvalCompleteness = Literal["full", "partial", "too_recent"]

# (year, raw_code) → canonical_code. nflverse codes are mostly canonical, but historical
# rows for relocated franchises sometimes carry the old code.
TEAM_CODE_ALIASES: dict[tuple[int, str], str] = {
    # Raiders moved Oakland → Las Vegas in 2020
    **{(y, "OAK"): "LV" for y in range(2020, 2030)},
    # Chargers moved San Diego → Los Angeles in 2017
    **{(y, "SD"): "LAC" for y in range(2017, 2030)},
    # Rams moved St. Louis → Los Angeles in 2016
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
    out["team"] = [
        TEAM_CODE_ALIASES.get((int(y), t), t)
        for y, t in zip(out["draft_year"] if "draft_year" in out.columns else out["season"], out["team"])
    ]
    return out


def build_picks(raw: pd.DataFrame) -> pd.DataFrame:
    """Reshape raw nflreadpy draft picks into our `picks.parquet` schema.

    Input columns expected (from nflreadpy.load_draft_picks):
        season, round, pick, team, gsis_id, pfr_player_name, position, college,
        car_av, w_av, dr_av, games, probowls, allpro, to
    """
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

    # Coerce nullable int columns to plain int (nflreadpy may return Int64)
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
```

- [ ] **Step 4: Verify tests pass**

Run: `uv run pytest tests/transform/test_careers.py -v`
Expected: 6 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/kiper_grade_check/transform/careers.py tests/transform/test_careers.py
git commit -m "feat(transform): build per-pick dataframe with eval_completeness"
```

---

### Task 8: `transform/contracts.py` — second-contract derivation

**Files:**
- Create: `src/kiper_grade_check/transform/contracts.py`
- Create: `tests/transform/test_contracts.py`

- [ ] **Step 1: Write the failing tests**

```python
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
            {"gsis_id": "P1", "team": "NE", "year_signed": 2014},   # rookie
            {"gsis_id": "P1", "team": "NE", "year_signed": 2018},   # extension w/ NE
        ]
    )
    out = derive_second_contract_same_team(_picks(), contracts)
    assert out.set_index("player_id").loc["P1", "second_contract_same_team"] is True


def test_restructured_in_year_2_does_not_count() -> None:
    contracts = _contracts(
        [
            {"gsis_id": "P2", "team": "NE", "year_signed": 2014},
            {"gsis_id": "P2", "team": "NE", "year_signed": 2015},   # restructure (year < draft+3)
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
```

- [ ] **Step 2: Verify it fails**

Run: `uv run pytest tests/transform/test_contracts.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `src/kiper_grade_check/transform/contracts.py`**

```python
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

    # group by player to get the second contract row, if any
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
```

- [ ] **Step 4: Verify tests pass**

Run: `uv run pytest tests/transform/test_contracts.py -v`
Expected: 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/kiper_grade_check/transform/contracts.py tests/transform/test_contracts.py
git commit -m "feat(transform): derive second-contract-with-drafting-team flag"
```

---

### Task 9: `transform/team_wins.py` — 5-year win sum from schedules

**Files:**
- Create: `src/kiper_grade_check/transform/team_wins.py`
- Create: `tests/transform/test_team_wins.py`

- [ ] **Step 1: Write the failing tests**

```python
import pandas as pd

from kiper_grade_check.transform.team_wins import compute_team_wins_5yr, season_wins


def _schedule(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_season_wins_counts_only_regular_season() -> None:
    sched = _schedule(
        [
            {"season": 2017, "game_type": "REG", "home_team": "NE", "away_team": "BUF", "home_score": 23, "away_score": 3},
            {"season": 2017, "game_type": "REG", "home_team": "MIA", "away_team": "NE", "home_score": 14, "away_score": 27},
            {"season": 2017, "game_type": "WC",  "home_team": "NE", "away_team": "TEN", "home_score": 35, "away_score": 14},   # playoff — excluded
        ]
    )
    wins = season_wins(sched, season=2017, team="NE")
    assert wins == 2


def test_team_wins_5yr_sums_consecutive_seasons() -> None:
    rows = []
    # 5 wins/season for 5 years
    for season in range(2014, 2019):
        for _ in range(5):
            rows.append({"season": season, "game_type": "REG", "home_team": "NE", "away_team": "BUF", "home_score": 30, "away_score": 0})
    sched = _schedule(rows)
    df = compute_team_wins_5yr(sched, draft_years=[2014], teams=["NE"])
    assert df.set_index(["draft_year", "team"]).loc[(2014, "NE"), "team_wins_5yr"] == 25


def test_team_wins_5yr_handles_ties_as_half_or_zero() -> None:
    # Real NFL ties are rare but they exist. We count them as 0 (i.e., not a win).
    sched = _schedule(
        [
            {"season": 2014, "game_type": "REG", "home_team": "NE", "away_team": "BUF", "home_score": 17, "away_score": 17},
        ]
    )
    df = compute_team_wins_5yr(sched, draft_years=[2014], teams=["NE"])
    assert df.set_index(["draft_year", "team"]).loc[(2014, "NE"), "team_wins_5yr"] == 0
```

- [ ] **Step 2: Verify it fails**

Run: `uv run pytest tests/transform/test_team_wins.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `src/kiper_grade_check/transform/team_wins.py`**

```python
"""Compute regular-season wins per team for a 5-year window starting at draft_year."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


def season_wins(schedules: pd.DataFrame, season: int, team: str) -> int:
    s = schedules[(schedules["season"] == season) & (schedules["game_type"] == "REG")]
    if s.empty:
        return 0

    home_wins = ((s["home_team"] == team) & (s["home_score"] > s["away_score"])).sum()
    away_wins = ((s["away_team"] == team) & (s["away_score"] > s["home_score"])).sum()
    return int(home_wins + away_wins)


def compute_team_wins_5yr(
    schedules: pd.DataFrame, draft_years: Iterable[int], teams: Iterable[str]
) -> pd.DataFrame:
    rows: list[dict] = []
    for year in draft_years:
        for team in teams:
            total = sum(season_wins(schedules, season=year + offset, team=team) for offset in range(5))
            rows.append({"draft_year": year, "team": team, "team_wins_5yr": total})
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Verify tests pass**

Run: `uv run pytest tests/transform/test_team_wins.py -v`
Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/kiper_grade_check/transform/team_wins.py tests/transform/test_team_wins.py
git commit -m "feat(transform): 5-year regular-season win sums per team"
```

---

## Phase 3 — Grade computation

### Task 10: `grade.py` — expected AV by slot (LOESS curve)

**Files:**
- Create: `src/kiper_grade_check/grade.py`
- Create: `tests/test_grade.py`

- [ ] **Step 1: Write the failing tests for the AV curve**

```python
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
    assert curve["expected_av"].iloc[0] > curve["expected_av"].iloc[-1]   # pick 1 > pick 260
    assert curve["expected_av"].is_monotonic_decreasing or (curve["expected_av"].diff().mean() < 0)


def test_expected_av_smooths_thin_slots() -> None:
    # Synthetic: alternating sparse high/low at adjacent picks. After smoothing,
    # the values should be close (no whipsaw).
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
    # Smoothed curve shouldn't have 50-point swings between adjacent slots
    assert diffs.max() < 25
```

- [ ] **Step 2: Verify it fails**

Run: `uv run pytest tests/test_grade.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `compute_expected_av_by_slot` in `src/kiper_grade_check/grade.py`**

```python
"""Compute the actual draft grade and helpers."""

from __future__ import annotations

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
```

- [ ] **Step 4: Verify tests pass**

Run: `uv run pytest tests/test_grade.py -v`
Expected: 2 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/kiper_grade_check/grade.py tests/test_grade.py
git commit -m "feat(grade): expected-AV-by-slot LOESS curve"
```

---

### Task 11: `grade.py` — composite grade + letter bucket

**Files:**
- Modify: `src/kiper_grade_check/grade.py`
- Modify: `tests/test_grade.py` (append more tests)

- [ ] **Step 1: Append failing tests to `tests/test_grade.py`**

```python
from kiper_grade_check.grade import (
    DEFAULT_WEIGHTS,
    bucket_z_to_letter,
    compute_actual_grade,
    z_within_year,
)


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
    # Both years should sum to ~0 by construction (mean-centered)
    g = out.groupby("draft_year")["value_z"].mean()
    assert (g.abs() < 1e-9).all()


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
    # Best team (A) has highest z; worst team (D) has lowest
    z = out.set_index("team")["actual_grade_z"]
    assert z["A"] > z["B"] > z["C"] > z["D"]


def test_compute_actual_grade_partial_eval_redistributes_weight() -> None:
    df = pd.DataFrame(
        {
            "draft_year": [2022, 2022, 2022, 2022],
            "team": ["A", "B", "C", "D"],
            "total_av_over_expected": [10.0, 5.0, -5.0, -10.0],
            "avg_pro_bowls_per_pick": [1.0, 0.5, 0.2, 0.0],
            "pct_second_contract": [None, None, None, None],   # all null in partial
            "team_wins_5yr": [40, 35, 25, 20],
            "eval_completeness": ["partial"] * 4,
        }
    )
    out = compute_actual_grade(df)
    z = out.set_index("team")["actual_grade_z"]
    # Order should still be preserved; weights redistributed pro-rata
    assert z["A"] > z["B"] > z["C"] > z["D"]


def test_default_weights_sum_to_one() -> None:
    assert abs(sum(DEFAULT_WEIGHTS.values()) - 1.0) < 1e-9
```

- [ ] **Step 2: Verify tests fail**

Run: `uv run pytest tests/test_grade.py -v`
Expected: 4 of the 6 fail (only the AV curve tests pass; the rest reference unwritten functions).

- [ ] **Step 3: Append the implementations to `src/kiper_grade_check/grade.py`**

```python
from typing import Mapping

# Component weights — exposed as a constant so the notebook can run sensitivity analysis
DEFAULT_WEIGHTS: Mapping[str, float] = {
    "av_over_expected": 0.50,
    "pro_bowls_per_pick": 0.20,
    "pct_second_contract": 0.15,
    "team_wins_5yr": 0.15,
}

# Letter-bucket thresholds — left edge inclusive, right edge exclusive (except top bucket)
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

    Input columns required:
        draft_year, team, total_av_over_expected, avg_pro_bowls_per_pick,
        pct_second_contract (nullable), team_wins_5yr, eval_completeness

    `pct_second_contract` is treated as null when eval_completeness != "full"; its
    weight is redistributed pro-rata across the other three components.
    """
    df = team_outcomes.copy()

    # mask out pct_second_contract for partial/too_recent rows
    df["pct_second_contract"] = df.apply(
        lambda r: r["pct_second_contract"] if r["eval_completeness"] == "full" else np.nan,
        axis=1,
    )

    # z-score each component within draft year
    df = z_within_year(df, "total_av_over_expected")
    df = z_within_year(df, "avg_pro_bowls_per_pick")
    df = z_within_year(df, "pct_second_contract")
    df = z_within_year(df, "team_wins_5yr")

    # weighted sum, redistributing pct_second_contract weight if null
    av_w = weights["av_over_expected"]
    pb_w = weights["pro_bowls_per_pick"]
    sc_w = weights["pct_second_contract"]
    tw_w = weights["team_wins_5yr"]

    def composite(row: pd.Series) -> float:
        if pd.isna(row["pct_second_contract_z"]):
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
```

- [ ] **Step 4: Add the imports at the top of `grade.py`**

Make sure `from typing import Mapping` and `import numpy as np` are imported (numpy already imported from Task 10).

- [ ] **Step 5: Verify tests pass**

Run: `uv run pytest tests/test_grade.py -v`
Expected: 6 tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/kiper_grade_check/grade.py tests/test_grade.py
git commit -m "feat(grade): z-score composite grade with letter bucketing"
```

---

## Phase 4 — Pipeline + CLI

### Task 12: `pipeline.py` — fetch orchestrator

**Files:**
- Create: `src/kiper_grade_check/pipeline.py`
- Create: `tests/test_pipeline_fetch.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from kiper_grade_check.pipeline import fetch_all


def test_fetch_all_creates_expected_artifacts(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"

    fake_picks = pd.DataFrame({"season": [2014], "round": [1], "pick": [29]})
    fake_contracts = pd.DataFrame({"gsis_id": ["P1"], "team": ["NE"], "year_signed": [2014]})
    fake_schedules = pd.DataFrame(
        {"season": [2014], "game_type": ["REG"], "home_team": ["NE"], "away_team": ["BUF"], "home_score": [30], "away_score": [10]}
    )

    with (
        patch("kiper_grade_check.pipeline.nflverse.load_draft_picks", return_value=fake_picks),
        patch("kiper_grade_check.pipeline.nflverse.load_contracts", return_value=fake_contracts),
        patch("kiper_grade_check.pipeline.nflverse.load_schedules", return_value=fake_schedules),
        patch("kiper_grade_check.pipeline.report_card.fetch_report_card_html") as mock_rc,
    ):
        mock_rc.side_effect = lambda year, cache_dir: cache_dir / f"{year}.html"
        # no-op: write empty files so caller sees "fetched"
        def write_stub(year, cache_dir):
            cache_dir.mkdir(parents=True, exist_ok=True)
            p = cache_dir / f"{year}.html"
            p.write_text("<html></html>")
            return p
        mock_rc.side_effect = write_stub

        fetch_all(years=range(2014, 2016), raw_dir=raw_dir)

    # nflverse caches are written by the underlying functions; we only check they were called.
    # report_card files should exist for each year
    assert (raw_dir / "report_card" / "2014.html").exists()
    assert (raw_dir / "report_card" / "2015.html").exists()
```

- [ ] **Step 2: Verify it fails**

Run: `uv run pytest tests/test_pipeline_fetch.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `src/kiper_grade_check/pipeline.py` (fetch portion)**

```python
"""Pipeline orchestrators: fetch (Stage 1) and build (Stage 2).

Stage 3 (analysis) is a notebook, not part of this module.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

from kiper_grade_check.sources import nflverse, report_card

log = logging.getLogger(__name__)

DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_PROCESSED_DIR = Path("data/processed")


def fetch_all(
    years: Iterable[int],
    raw_dir: Path = DEFAULT_RAW_DIR,
    force: bool = False,
) -> None:
    """Stage 1: pull all raw inputs into raw_dir/."""
    raw_dir.mkdir(parents=True, exist_ok=True)

    log.info("fetching nflverse draft picks")
    nflverse.load_draft_picks(seasons=range(2011, 2026), cache_dir=raw_dir, force=force)

    log.info("fetching nflverse contracts")
    nflverse.load_contracts(cache_dir=raw_dir, force=force)

    log.info("fetching nflverse schedules")
    nflverse.load_schedules(seasons=range(2011, 2030), cache_dir=raw_dir, force=force)

    rc_dir = raw_dir / "report_card"
    for year in years:
        log.info("fetching report card for %d", year)
        report_card.fetch_report_card_html(year, cache_dir=rc_dir)
```

- [ ] **Step 4: Verify the test passes**

Run: `uv run pytest tests/test_pipeline_fetch.py -v`
Expected: 1 test passes.

- [ ] **Step 5: Commit**

```bash
git add src/kiper_grade_check/pipeline.py tests/test_pipeline_fetch.py
git commit -m "feat(pipeline): fetch-stage orchestrator"
```

---

### Task 13: `pipeline.py` — build orchestrator (4 output tables)

**Files:**
- Modify: `src/kiper_grade_check/pipeline.py` (append `build_all` and helpers)
- Create: `tests/test_pipeline_build.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

import pandas as pd
import pytest

from kiper_grade_check.pipeline import build_all


@pytest.fixture
def stub_raw(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    raw.mkdir()

    # Two teams, two picks each, one draft year
    picks_raw = pd.DataFrame(
        {
            "season": [2014, 2014, 2014, 2014],
            "round": [1, 5, 1, 6],
            "pick": [10, 150, 20, 200],
            "team": ["NE", "NE", "DAL", "DAL"],
            "gsis_id": ["P1", "P2", "P3", "P4"],
            "pfr_player_name": ["A", "B", "C", "D"],
            "position": ["WR", "DT", "RB", "S"],
            "college": ["FL", "FSU", "OU", "GA"],
            "car_av": [40, 5, 30, 8],
            "w_av": [38, 4, 28, 7],
            "dr_av": [40, 5, 30, 8],
            "games": [120, 60, 100, 70],
            "probowls": [2, 0, 1, 0],
            "allpro": [1, 0, 0, 0],
            "to": [2024, 2018, 2022, 2020],
        }
    )
    picks_raw.to_parquet(raw / "draft_picks.parquet", index=False)

    # Contracts: NE picks both stayed; DAL P3 stayed; DAL P4 left
    contracts = pd.DataFrame(
        {
            "gsis_id": ["P1", "P1", "P2", "P2", "P3", "P3", "P4", "P4"],
            "team":    ["NE", "NE", "NE", "NE", "DAL","DAL","DAL","CHI"],
            "year_signed": [2014, 2018, 2014, 2018, 2014, 2018, 2014, 2018],
        }
    )
    contracts.to_parquet(raw / "contracts.parquet", index=False)

    # Schedules: 5 seasons of made-up wins
    sched_rows = []
    for season in range(2014, 2019):
        # NE wins 12, loses 4 (16 reg-season games per year for 2014–2020)
        for i in range(12):
            sched_rows.append({"season": season, "game_type": "REG", "home_team": "NE", "away_team": "BUF", "home_score": 30, "away_score": 10})
        for i in range(4):
            sched_rows.append({"season": season, "game_type": "REG", "home_team": "NE", "away_team": "BUF", "home_score": 7, "away_score": 24})
        # DAL wins 8, loses 8
        for i in range(8):
            sched_rows.append({"season": season, "game_type": "REG", "home_team": "DAL", "away_team": "NYG", "home_score": 24, "away_score": 14})
        for i in range(8):
            sched_rows.append({"season": season, "game_type": "REG", "home_team": "DAL", "away_team": "NYG", "home_score": 7, "away_score": 24})
    pd.DataFrame(sched_rows).to_parquet(raw / "schedules.parquet", index=False)

    # Report card: simple wide table parsed elsewhere; here we just write a minimal HTML
    rc = raw / "report_card"
    rc.mkdir()
    (rc / "2014.html").write_text(
        "<html><body><table><thead><tr><th>Team</th><th>Kiper</th></tr></thead>"
        "<tbody>"
        "<tr><td>NE</td><td>A</td></tr>"
        "<tr><td>DAL</td><td>C</td></tr>"
        "</tbody></table></body></html>"
    )
    return raw


def test_build_all_emits_four_tables(stub_raw: Path, tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    build_all(years=[2014], raw_dir=stub_raw, processed_dir=processed)

    for name in ["picks", "analyst_grades", "team_outcomes", "comparison"]:
        assert (processed / f"{name}.parquet").exists()
        assert (processed / f"{name}.csv").exists()

    picks = pd.read_parquet(processed / "picks.parquet")
    assert {"draft_year", "team", "career_av", "eval_completeness"}.issubset(picks.columns)
    assert len(picks) == 4

    outcomes = pd.read_parquet(processed / "team_outcomes.parquet")
    assert set(outcomes["team"]) == {"NE", "DAL"}
    # NE should have higher actual_grade_z than DAL given fixture (more AV, more wins)
    z = outcomes.set_index("team")["actual_grade_z"]
    assert z["NE"] > z["DAL"]

    comparison = pd.read_parquet(processed / "comparison.parquet")
    assert {"analyst", "analyst_grade_z", "actual_grade_z", "residual"}.issubset(comparison.columns)
    assert (comparison["analyst"] == "Kiper").all()
```

- [ ] **Step 2: Verify it fails**

Run: `uv run pytest tests/test_pipeline_build.py -v`
Expected: AttributeError on `build_all`.

- [ ] **Step 3: Append `build_all` to `src/kiper_grade_check/pipeline.py`**

```python
import pandas as pd

from kiper_grade_check.grade import compute_actual_grade, compute_expected_av_by_slot, z_within_year
from kiper_grade_check.sources.report_card import parse_report_card_html
from kiper_grade_check.transform import careers, contracts, team_wins


def _emit(df: pd.DataFrame, processed_dir: Path, name: str) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(processed_dir / f"{name}.parquet", index=False)
    df.to_csv(processed_dir / f"{name}.csv", index=False)


def build_all(
    years: Iterable[int],
    raw_dir: Path = DEFAULT_RAW_DIR,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
) -> None:
    """Stage 2: transform raw → processed/{picks,analyst_grades,team_outcomes,comparison}."""
    years = list(years)

    # --- picks ---
    raw_picks = pd.read_parquet(raw_dir / "draft_picks.parquet")
    raw_contracts = pd.read_parquet(raw_dir / "contracts.parquet")
    raw_schedules = pd.read_parquet(raw_dir / "schedules.parquet")

    picks = careers.build_picks(raw_picks)
    target = picks[picks["draft_year"].isin(years)].copy()

    # second contract
    sc = contracts.derive_second_contract_same_team(target, raw_contracts)
    target = target.merge(sc, on="player_id", how="left")
    _emit(target, processed_dir, "picks")

    # --- analyst grades ---
    rc_dir = raw_dir / "report_card"
    grade_frames: list[pd.DataFrame] = []
    for year in years:
        path = rc_dir / f"{year}.html"
        if not path.exists():
            log.warning("missing report card for %d, skipping", year)
            continue
        urls_dict = report_card.REPORT_CARD_URLS(year)
        df = parse_report_card_html(path.read_text(), draft_year=year, source_url=urls_dict["live"])
        if not df.empty:
            grade_frames.append(df)
    analyst_grades = (
        pd.concat(grade_frames, ignore_index=True) if grade_frames else pd.DataFrame()
    )

    # canonicalize team codes from the report-card source to nflverse codes.
    if not analyst_grades.empty:
        analyst_grades["team"] = analyst_grades["team"].map(_team_to_nflverse_code).fillna(analyst_grades["team"])
    _emit(analyst_grades, processed_dir, "analyst_grades")

    # --- team_outcomes ---
    # av-over-expected curve from full historical picks (1980+)
    historical = raw_picks.rename(columns={"pick": "pick_overall", "car_av": "career_av"})
    av_curve = compute_expected_av_by_slot(historical[["pick_overall", "career_av"]].dropna())
    target = target.merge(av_curve, on="pick_overall", how="left")
    target["av_over_expected"] = target["career_av"] - target["expected_av"].fillna(0)

    # roll up to team-year
    grouped = (
        target.groupby(["draft_year", "team", "eval_completeness"], as_index=False)
        .agg(
            num_picks=("player_id", "count"),
            total_career_av=("career_av", "sum"),
            total_weighted_av=("weighted_av", "sum"),
            total_av_over_expected=("av_over_expected", "sum"),
            avg_pro_bowls_per_pick=("pro_bowls", "mean"),
            pct_second_contract=("second_contract_same_team", lambda s: s.dropna().mean() if s.notna().any() else None),
        )
    )
    # team wins
    teams_in_scope = grouped["team"].unique().tolist()
    wins = team_wins.compute_team_wins_5yr(raw_schedules, draft_years=years, teams=teams_in_scope)
    outcomes = grouped.merge(wins, on=["draft_year", "team"], how="left")
    outcomes = compute_actual_grade(outcomes)
    _emit(outcomes, processed_dir, "team_outcomes")

    # --- comparison ---
    if not analyst_grades.empty:
        # z-score the analyst's grade within draft year for direct comparison
        ag = z_within_year(analyst_grades.rename(columns={"grade_numeric": "analyst_grade_numeric"}), "analyst_grade_numeric")
        ag = ag.rename(columns={"analyst_grade_numeric_z": "analyst_grade_z"})
        comparison = ag.merge(
            outcomes[["draft_year", "team", "actual_grade_z", "actual_grade_letter", "eval_completeness"]],
            on=["draft_year", "team"],
            how="inner",
        )
        comparison["residual"] = comparison["actual_grade_z"] - comparison["analyst_grade_z"]
    else:
        comparison = pd.DataFrame()
    _emit(comparison, processed_dir, "comparison")


# Football Outsiders / FTN typically write team names in long form ("New England Patriots").
# nflverse uses 3-letter codes. Map source-team → canonical code.
_TEAM_NAME_MAP = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Oakland Raiders": "OAK",
    "Los Angeles Chargers": "LAC", "San Diego Chargers": "SD",
    "Los Angeles Rams": "LAR", "St. Louis Rams": "STL",
    "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN", "New England Patriots": "NE",
    "New Orleans Saints": "NO", "New York Giants": "NYG", "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT", "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS", "Washington Football Team": "WAS",
    "Washington Redskins": "WAS",
}


def _team_to_nflverse_code(name: str) -> str | None:
    if not isinstance(name, str):
        return None
    if name.strip() in _TEAM_NAME_MAP:
        return _TEAM_NAME_MAP[name.strip()]
    if len(name.strip()) == 3 and name.strip().isupper():
        return name.strip()
    return None
```

- [ ] **Step 4: Verify the test passes**

Run: `uv run pytest tests/test_pipeline_build.py -v`
Expected: 1 test passes.

- [ ] **Step 5: Verify the full test suite still passes**

Run: `uv run pytest -v`
Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/kiper_grade_check/pipeline.py tests/test_pipeline_build.py
git commit -m "feat(pipeline): build-stage orchestrator producing 4 output tables"
```

---

### Task 14: CLI

**Files:**
- Create: `src/kiper_grade_check/cli.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

```python
from typer.testing import CliRunner

from kiper_grade_check.cli import app


def test_cli_help() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "fetch" in result.output
    assert "build" in result.output


def test_cli_build_smoke(monkeypatch, tmp_path) -> None:
    called: dict = {}

    def fake_build(years, raw_dir, processed_dir):
        called["args"] = (list(years), raw_dir, processed_dir)

    monkeypatch.setattr("kiper_grade_check.cli.pipeline.build_all", fake_build)

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["build", "--year", "2014", "--year", "2015",
         "--raw-dir", str(tmp_path / "raw"),
         "--processed-dir", str(tmp_path / "processed")],
    )
    assert result.exit_code == 0, result.output
    assert called["args"][0] == [2014, 2015]
```

- [ ] **Step 2: Verify it fails**

Run: `uv run pytest tests/test_cli.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `src/kiper_grade_check/cli.py`**

```python
"""CLI entrypoint: `kgc fetch`, `kgc build`."""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.logging import RichHandler

from kiper_grade_check import pipeline

app = typer.Typer(no_args_is_help=True, add_completion=False)


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="%H:%M:%S",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )


@app.command()
def fetch(
    year: list[int] = typer.Option(None, help="Specific draft year(s); defaults to 2011–2025."),
    raw_dir: Path = typer.Option(Path("data/raw"), help="Cache directory for raw data."),
    force: bool = typer.Option(False, help="Bypass caches and re-fetch."),
) -> None:
    """Stage 1 — pull raw nflverse + report-card data into raw_dir."""
    _setup_logging()
    years = year if year else list(range(2011, 2026))
    pipeline.fetch_all(years=years, raw_dir=raw_dir, force=force)


@app.command()
def build(
    year: list[int] = typer.Option(None, help="Specific draft year(s); defaults to 2011–2025."),
    raw_dir: Path = typer.Option(Path("data/raw"), help="Where raw data lives."),
    processed_dir: Path = typer.Option(Path("data/processed"), help="Where processed tables go."),
) -> None:
    """Stage 2 — transform raw → processed datasets."""
    _setup_logging()
    years = year if year else list(range(2011, 2026))
    pipeline.build_all(years=years, raw_dir=raw_dir, processed_dir=processed_dir)


if __name__ == "__main__":
    app()
```

- [ ] **Step 4: Verify tests pass**

Run: `uv run pytest tests/test_cli.py -v`
Expected: 2 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/kiper_grade_check/cli.py tests/test_cli.py
git commit -m "feat(cli): typer-based fetch/build commands"
```

---

## Phase 5 — End-to-end run + notebook

### Task 15: Real fetch + build, commit processed datasets

This task **actually runs the pipeline** against live data. Expect failures around the report-card scraper for thin years; iterate until at least 2018–2025 work, then accept any 2011–2017 gaps as `MISSING` stubs (or fall back to manual top-up if too many).

**Files:**
- Modify: nothing in code; outputs go to `data/processed/` and `data/raw/`

- [ ] **Step 1: Run fetch**

Run: `uv run kgc fetch`
Expected: completes within 5–10 minutes. Watch logs for which years fall through to Wayback or write `.MISSING` stubs. Report which years failed.

- [ ] **Step 2: Inspect report-card cache**

Run: `ls data/raw/report_card/`
Expected: ~15 files, mostly `.html`. Note any `.MISSING` files.

For each `.MISSING` file, attempt one manual recovery: search the web for the year's grade table, save HTML to `data/raw/report_card/{year}.html`, delete the `.MISSING` stub. If after 2 attempts a year still won't yield, leave the stub and continue.

- [ ] **Step 3: Run build**

Run: `uv run kgc build`
Expected: 4 files appear under `data/processed/` (parquet + csv each). Logs report row counts.

- [ ] **Step 4: Sanity-check the processed datasets**

```bash
uv run python - <<'PY'
import pandas as pd
for name in ["picks", "analyst_grades", "team_outcomes", "comparison"]:
    df = pd.read_parquet(f"data/processed/{name}.parquet")
    print(f"{name}: {len(df)} rows, columns: {list(df.columns)}")
    print(df.head(3))
    print()
PY
```

Expected counts (approximate):
- picks: 3,500–4,000 rows (15 years × ~250 picks)
- analyst_grades: 2,000–3,500 rows depending on how many years' Report Cards loaded and how many analysts each contains
- team_outcomes: ~480 rows (15 × 32, minus a couple for missing report cards)
- comparison: 2,000–3,500 rows

If any number is wildly off, debug before committing.

- [ ] **Step 5: Commit processed datasets**

```bash
git add data/processed/ data/raw/report_card/*.html data/raw/report_card/*.MISSING 2>/dev/null
git commit -m "data: initial fetch + build of processed tables (2011–2025)"
```

We commit the report_card HTML (it's small, ~MB total) so the build is reproducible without re-scraping. We do NOT commit the nflverse parquet caches — they're regenerable and large.

---

### Task 16: Analysis notebook

**Files:**
- Create: `notebooks/01-kiper-vs-reality.ipynb`

- [ ] **Step 1: Author the notebook**

Create the notebook with these cells (use `jupyter nbconvert --to notebook --execute` if scripting; otherwise hand-author in Jupyter):

```python
# Cell 1 — setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_theme(style="whitegrid")

comparison = pd.read_parquet("../data/processed/comparison.parquet")
team_outcomes = pd.read_parquet("../data/processed/team_outcomes.parquet")
analyst_grades = pd.read_parquet("../data/processed/analyst_grades.parquet")
print(f"comparison: {len(comparison):,} rows; analysts: {sorted(comparison.analyst.unique())}")
```

```python
# Cell 2 — headline: how well does each analyst correlate with reality?
full = comparison.query("eval_completeness == 'full'")
results = []
for analyst in sorted(full["analyst"].unique()):
    sub = full[full["analyst"] == analyst]
    pearson = stats.pearsonr(sub["analyst_grade_z"], sub["actual_grade_z"])
    spearman = stats.spearmanr(sub["analyst_grade_z"], sub["actual_grade_z"])
    results.append({
        "analyst": analyst,
        "n": len(sub),
        "pearson_r": pearson.statistic,
        "pearson_p": pearson.pvalue,
        "spearman_rho": spearman.statistic,
        "spearman_p": spearman.pvalue,
    })
leaderboard = pd.DataFrame(results).sort_values("pearson_r", ascending=False)
leaderboard
```

```python
# Cell 3 — Kiper specifically: scatter plot
kiper = full[full["analyst"] == "Kiper"]
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(kiper["analyst_grade_z"], kiper["actual_grade_z"], alpha=0.5)
m, b = np.polyfit(kiper["analyst_grade_z"], kiper["actual_grade_z"], 1)
xs = np.linspace(kiper["analyst_grade_z"].min(), kiper["analyst_grade_z"].max(), 50)
ax.plot(xs, m * xs + b, color="red")
ax.set_xlabel("Kiper grade (z within year)")
ax.set_ylabel("Actual grade (z within year)")
ax.set_title(f"Kiper vs. reality (n={len(kiper)}, Pearson r={leaderboard.set_index('analyst').loc['Kiper', 'pearson_r']:.3f})")
plt.tight_layout()
```

```python
# Cell 4 — Kiper, by year
by_year = (
    full[full["analyst"] == "Kiper"]
    .groupby("draft_year")
    .apply(lambda g: stats.pearsonr(g["analyst_grade_z"], g["actual_grade_z"]).statistic if len(g) > 2 else None)
    .reset_index(name="pearson_r")
)
by_year
```

```python
# Cell 5 — Biggest Kiper misses
misses = full[full["analyst"] == "Kiper"].copy()
misses["abs_residual"] = misses["residual"].abs()
misses = misses.sort_values("abs_residual", ascending=False).head(20)
misses[["draft_year", "team", "analyst_grade_z", "actual_grade_z", "actual_grade_letter", "residual"]]
```

- [ ] **Step 2: Execute the notebook**

Run: `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01-kiper-vs-reality.ipynb`
Expected: all cells execute without error; the headline leaderboard and scatter plot produce sensible numbers.

- [ ] **Step 3: Commit**

```bash
git add notebooks/01-kiper-vs-reality.ipynb
git commit -m "feat(notebook): kiper-vs-reality headline analysis"
```

---

## Phase 6 — CI

### Task 17: GitHub Actions workflow

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Write the workflow**

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
        with:
          python-version: "3.13"
      - run: uv sync
      - run: uv run ruff check .
      - run: uv run pyright
      - run: uv run pytest -v
```

- [ ] **Step 2: Commit and push**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: ruff + pyright + pytest on push/PR"
git push
```

- [ ] **Step 3: Verify CI is green**

Run: `gh run watch`
Expected: workflow completes successfully. If it fails, fix and push again.

---

## Wrap-up

- [ ] **Final check: README links work**

Run: `uv run python -c "from kiper_grade_check.cli import app; print('ok')"`
Expected: `ok`

- [ ] **Final check: full test suite green**

Run: `uv run pytest -v`
Expected: all tests pass.

- [ ] **Final commit**

```bash
git push
```

Project is now in a state where:
- `data/processed/` contains the final dataset
- `notebooks/01-kiper-vs-reality.ipynb` is the headline analysis
- CI runs on push
- The pipeline is re-runnable end-to-end via `uv run kgc fetch && uv run kgc build`
