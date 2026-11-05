import re
import shutil

import pytest
from apitofsim.workflow.db import ExperimentDatabase
from playwright.sync_api import Page, expect

from tests.test_browser import assert_clean_browser, watch_browser_errors


@pytest.fixture(scope="session", params=[True, False], ids=["shared", "missing"])
def browser_database(browser_database, tmp_path_factory, request):
    path = tmp_path_factory.mktemp("cluster-navigation") / "test.duckdb"
    shutil.copy2(browser_database, path)
    if not request.param:
        db = ExperimentDatabase(path)
        try:
            db.db.execute("insert into pathway values (2, 2, 1, 3)")
            db.db.execute(
                "update single_pathway_experiment_result set pathway_id = 2 where experiment_run_id = 2"
            )
        finally:
            db.close()
    return path


@pytest.mark.parametrize(
    "route", ["cluster", "report", "realizations", "explorer", "spectrogram"]
)
def test_switch_experiment_with_selected_cluster(page: Page, live_server, route):
    errors = watch_browser_errors(page)
    page.set_default_navigation_timeout(60_000)
    path = "/report" if route == "report" else "/experiment/cluster"
    if route not in {"cluster", "report"}:
        path += f"/{route}"
    page.goto(f"{live_server}{path}?experiment=1&cluster=1")
    expect(page.get_by_label("Cluster:")).to_have_value("1")

    page.get_by_label("Experiment:").select_option("2")

    expect(page.get_by_label("Experiment:")).to_have_value("2", timeout=60_000)
    choices = (
        page.get_by_label("Cluster:")
        .locator("option")
        .evaluate_all("options => options.map(option => option.value)")
    )
    if "1" in choices:
        expect(page).to_have_url(re.compile(re.escape(path) + r"\?"))
        expect(page.get_by_label("Cluster:")).to_have_value("1")
    else:
        expect(page).to_have_url(f"{live_server}/experiment/cluster?experiment=2")
        expect(page.get_by_label("Cluster:")).to_have_value("")

    assert_clean_browser(*errors)
