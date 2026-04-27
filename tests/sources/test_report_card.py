from pathlib import Path
from unittest.mock import MagicMock, patch

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

    assert {"draft_year", "team", "analyst", "grade_letter", "grade_numeric", "source_url"}.issubset(df.columns)
    assert (df["draft_year"] == 2020).all()
    assert df["analyst"].nunique() >= 3
    assert df["team"].nunique() >= 28
    assert df["grade_letter"].isin(  # type: ignore[arg-type]
        ["A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D+", "D", "D-", "F"]
    ).all()  # type: ignore[union-attr]
    assert (df["grade_numeric"] >= 0).all()  # type: ignore[union-attr]


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

    with patch("kiper_grade_check.sources.report_card.httpx.Client") as mock_client:
        mock_client.return_value.__enter__.return_value.get.side_effect = fake_get
        path = fetch_report_card_html(2020, cache_dir=cache_dir)

    assert path.read_text() == "<html>WAYBACK CONTENT</html>"


def test_fetch_report_card_writes_missing_stub_when_both_sources_fail(tmp_path: Path) -> None:
    cache_dir = tmp_path / "report_card"
    fail = MagicMock()
    fail.status_code = 404
    fail.text = ""

    with patch("kiper_grade_check.sources.report_card.httpx.Client") as mock_client:
        mock_client.return_value.__enter__.return_value.get.return_value = fail
        path = fetch_report_card_html(2011, cache_dir=cache_dir)

    assert path.suffix == ".MISSING"
    assert path.exists()


def test_fetch_report_card_uses_cache_when_present(tmp_path: Path) -> None:
    cache_dir = tmp_path / "report_card"
    cache_dir.mkdir(parents=True)
    cached = cache_dir / "2020.html"
    cached.write_text("<html>cached</html>")

    with patch("kiper_grade_check.sources.report_card.httpx.Client") as mock_client:
        path = fetch_report_card_html(2020, cache_dir=cache_dir)
        mock_client.assert_not_called()

    assert path == cached
