"""Tests for the Click CLI entry points."""

from __future__ import annotations

from click.testing import CliRunner

from stat_arb.cli import cli


class TestCLI:
    def test_help(self) -> None:
        """Top-level --help works."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "run-backtest" in result.output
        assert "run-live" in result.output

    def test_run_backtest_help(self) -> None:
        """run-backtest --help shows options."""
        runner = CliRunner()
        result = runner.invoke(cli, ["run-backtest", "--help"])
        assert result.exit_code == 0
        assert "--start" in result.output
        assert "--end" in result.output
        assert "--config" in result.output
        assert "--persist" in result.output

    def test_run_live_help(self) -> None:
        """run-live --help shows options."""
        runner = CliRunner()
        result = runner.invoke(cli, ["run-live", "--help"])
        assert result.exit_code == 0
        assert "--loop" in result.output
        assert "--broker-mode" in result.output
        assert "--config" in result.output
        assert "paper" in result.output

    def test_run_backtest_missing_start(self) -> None:
        """run-backtest without --start fails."""
        runner = CliRunner()
        result = runner.invoke(cli, ["run-backtest", "--end", "2025-01-01"])
        assert result.exit_code != 0
        assert "Missing" in result.output or "required" in result.output.lower()

    def test_version(self) -> None:
        """--version shows the package version."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "stat-arb" in result.output.lower() or "version" in result.output.lower()
