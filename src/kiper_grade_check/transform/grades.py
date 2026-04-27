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
