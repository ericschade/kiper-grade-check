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
