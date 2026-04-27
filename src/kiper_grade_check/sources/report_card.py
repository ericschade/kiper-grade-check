"""Fetch and parse the Football Outsiders / FTN 'Draft Report Card Report' pages.

Each page aggregates analyst team draft grades for one draft year. We extract
a long-format dataframe with one row per (team, analyst) where grade data is
available.

URL patterns:
    - 2018–2023: footballoutsiders.com/nfl-draft/{year}/{year}-nfl-draft-report-card-report
    - 2024+:     ftnfantasy.com/nfl/{year}-nfl-draft-report-card-report
    - <2018 and any failure: Wayback Machine

Fixture observations (2020 FO Report Card, Wayback Machine, 275252 bytes):
    The page has 2 HTML tables:
      Table 0 ("2020 NFL Draft Grades"): aggregate per-team grades.
        Headers: Team | High | Low | GPA | Rk | SD | Rk
        32 data rows (one per team, team names as short codes: "DAL", "NE", etc.)
        High/Low cells contain grade + analyst info: "A+ (8 tied)", "B- (Kadar)",
        "A+ (PFF, Farrar)", etc. When specific analyst names appear (not "N tied"),
        we can extract per-analyst grade data.
      Table 1 ("2020 NFL Draft Graders"): aggregate per-analyst stats.
        Headers: Grader | High | Low | GPA | SD
        18 analyst rows.

    IMPORTANT: The page does NOT contain a full per-analyst × per-team grade matrix.
    Individual analyst grades per team are only visible when that analyst gave the
    highest or lowest grade for that team. We extract those sparse (team, analyst,
    grade) triples — typically 70+ rows covering all 32 teams and 15–18 analysts.

    Analyst names on the 2020 page: Reuter, Iyer, PFF, Benoit, Easterling, Farrar,
    Slater, Kadar, Maske, Prisco, Dunleavy, Kiper, Rill, Edholm, Davis, Winks,
    Silva, Tagliere.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import httpx
import pandas as pd
from selectolax.parser import HTMLParser

from kiper_grade_check.transform.grades import grade_to_numeric, parse_grade

USER_AGENT = "kiper-grade-check/0.1 (research; +https://github.com/ericschade/kiper-grade-check)"
REQUEST_INTERVAL_SEC = 1.0

# All analyst surnames that may appear in FO/FTN report card pages across years.
# Used to distinguish "Kiper, Maske" from "4 tied" in grade cells.
KNOWN_ANALYSTS: frozenset[str] = frozenset(
    {
        # Classic ESPN/NFL Network analysts
        "Kiper", "McShay", "Prisco", "Brooks", "Pompei", "Cole", "Pauline",
        "Schein", "Florio", "Brandt", "Reuter", "PFF", "Trapasso",
        # FO/Sporting News panel (varies by year)
        "Iyer", "Maske", "Davis", "Edholm", "Easterling", "Farrar",
        "Kadar", "Silva", "Tagliere", "Winks", "Rill", "Dunleavy",
        "Slater", "Benoit", "Jeremiah",
    }
)


def REPORT_CARD_URLS(year: int) -> dict[str, str]:  # noqa: N802
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


def _parse_grade_cell(cell_text: str) -> tuple[str | None, list[str]]:
    """Parse a High/Low grade cell like 'A+ (Iyer, Rill)' or 'B- (4 tied)'.

    Returns (grade_letter, [analyst_names]).
    When the cell is anonymous ("N tied" or "N total"), returns (grade, []).
    """
    m = re.match(r"^([A-F][+-]?)\s*\((.+)\)$", cell_text.strip())
    if not m:
        return None, []
    grade_raw = m.group(1)
    note = m.group(2).strip()

    # Anonymous: "4 tied", "8 total", etc.
    if re.search(r"\d+\s+(?:tied|total)", note):
        return grade_raw, []

    analysts = [a.strip() for a in note.split(",")]
    return grade_raw, analysts


def parse_report_card_html(html: str, draft_year: int, source_url: str) -> pd.DataFrame:
    """Extract a long-format (team, analyst, grade) dataframe from one Report Card page.

    Football Outsiders pages present grades as an aggregate table where each team row
    shows the highest and lowest analyst grade, with analyst name(s) in parentheses.
    We extract (team, analyst, grade) triples from those High/Low cells for all rows
    where a specific analyst name appears (not "N tied").

    For pages that contain a full analyst × team grid (team rows, analyst columns),
    we also attempt to parse that structure.
    """
    tree = HTMLParser(html)

    rows: list[dict] = []

    for table in tree.css("table"):
        table_rows = table.css("tr")
        if len(table_rows) < 3:
            continue

        # Collect all header cells from first two rows
        header_cells: list[str] = []
        for header_row in table_rows[:2]:
            header_cells += [
                (c.text(strip=True) or "").strip()
                for c in header_row.css("th, td")
            ]

        first_data_cells = [
            (c.text(strip=True) or "").strip()
            for c in table_rows[2].css("th, td")
        ]

        # ── Full analyst × team grid ──────────────────────────────────────────
        # Header row: Team | Analyst1 | Analyst2 | ...
        # Data rows: team_name | grade | grade | ...
        analyst_cols = [h for h in header_cells if h in KNOWN_ANALYSTS]
        if header_cells and header_cells[0].lower() == "team" and len(analyst_cols) >= 1:
            headers = header_cells
            for tr in table_rows[1:]:
                cells = [(c.text(strip=True) or "").strip() for c in tr.css("th, td")]
                if not cells or len(cells) < 2:
                    continue
                team = cells[0]
                for col_idx, header in enumerate(headers[1:], start=1):
                    if header not in KNOWN_ANALYSTS or col_idx >= len(cells):
                        continue
                    raw_grade = cells[col_idx]
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
                            "analyst": header,
                            "grade_letter": letter,
                            "grade_numeric": grade_to_numeric(letter),
                            "source_url": source_url,
                        }
                    )
            if rows:
                return pd.DataFrame(rows)

        # ── FO aggregate table: Team | High | Low | GPA | ... ────────────────
        # Identify by: first non-title data cell is a short team code, High/Low
        # columns contain grade cells like "A+ (Iyer, Rill)".
        is_team_row = bool(
            first_data_cells
            and 1 <= len(first_data_cells[0]) <= 4  # short team code
            and len(first_data_cells) >= 3
            and re.match(r"^[A-F][+-]?\s*\(", first_data_cells[1] or "")
        )
        if not is_team_row:
            continue

        # High is column index 1, Low is column index 2
        for tr in table_rows[2:]:
            cells = [(c.text(strip=True) or "").strip() for c in tr.css("th, td")]
            if len(cells) < 3:
                continue
            team = cells[0]
            for grade_cell in (cells[1], cells[2]):
                letter, analysts = _parse_grade_cell(grade_cell)
                if letter is None:
                    continue
                try:
                    letter = parse_grade(letter)
                except ValueError:
                    continue
                for analyst in analysts:
                    if analyst not in KNOWN_ANALYSTS:
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

    return pd.DataFrame(
        rows,
        columns=["draft_year", "team", "analyst", "grade_letter", "grade_numeric", "source_url"]
        if not rows
        else None,
    )
