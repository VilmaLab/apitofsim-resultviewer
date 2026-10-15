"""Wiring and asset declarations for the shared desktop helpers."""

from argparse import Namespace

import pytest

from apitofresview import cli
from apitofresview.build import ASSETS, _installed_assets
from apitofresview.desktop import APP
from workingtitle.desktop.freezing import platform_options


def test_asset_plan_watches_sources_without_watching_outputs():
    for path in (
        "src/js/index.ts",
        "src/css/index.css",
        "src/apitofresview/templates/base.html",
        "package-lock.json",
    ):
        assert ASSETS.is_input(ASSETS.root / path)
    for path in (
        "src/apitofresview/static/index.js",
        "src/apitofresview/static/index.css",
        "node_modules/htmx.org/package.json",
    ):
        assert not ASSETS.is_input(ASSETS.root / path)


def test_installed_assets_are_validation_only():
    plan = _installed_assets()
    assert plan.source is False
    assert plan.steps[0].outputs == (
        "apitofresview/static/index.js",
        "apitofresview/static/index.css",
    )


def test_pywebview_platform_policy_matches_runtime():
    hidden, excluded = platform_options(APP, "linux")
    assert "webview" in excluded
    assert "webview" not in hidden
    hidden, excluded = platform_options(APP, "darwin")
    assert "webview.platforms.cocoa" in hidden
    assert "webview" not in excluded
    hidden, _ = platform_options(APP, "win32")
    assert "webview.platforms.edgechromium" in hidden


def test_prepare_passes_database_to_reload_factory(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE", raising=False)
    args = Namespace(
        database=tmp_path / "experiments.duckdb",
        no_window=False,
        no_browser=False,
        mode="auto",
    )
    assert cli.prepare(args) == {"DATABASE": str(args.database)}
    assert args.mode == "auto"


def test_prepare_requires_database(monkeypatch):
    monkeypatch.delenv("DATABASE", raising=False)
    args = Namespace(database=None, no_window=False, no_browser=False, mode="auto")
    with pytest.raises(ValueError, match="no database"):
        cli.prepare(args)


def test_main_wires_shared_runner(monkeypatch):
    captured = {}

    def fake_run_cli(spec, argv, **kwargs):
        captured.update(spec=spec, argv=argv, **kwargs)
        return 0

    monkeypatch.setattr(cli, "run_cli", fake_run_cli)
    assert cli.main(["--database", "sample.duckdb"]) == 0
    assert captured["spec"] is cli.APP
    assert captured["assets"] is ASSETS
    assert captured["session"] is cli.SESSION
    assert captured["smoke_test"] is cli.smoke_test
