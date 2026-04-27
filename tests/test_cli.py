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
