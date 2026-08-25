import csv
import io
import re

from playwright.sync_api import Page, expect


def watch_browser_errors(page):
    page.set_default_timeout(5_000)
    page_errors = []
    console_errors = []
    failed_responses = []
    page.on("pageerror", lambda error: page_errors.append(error))
    page.on(
        "console",
        lambda message: console_errors.append((message.text, message.location))
        if message.type == "error"
        else None,
    )
    page.on(
        "response",
        lambda response: failed_responses.append((response.status, response.url))
        if response.status >= 400
        else None,
    )
    return page_errors, console_errors, failed_responses


def assert_clean_browser(page_errors, console_errors, failed_responses):
    assert page_errors == []
    assert console_errors == [], failed_responses
    assert failed_responses == []


def test_navigation_and_report_workflow(page: Page, live_server):
    errors = watch_browser_errors(page)

    page.goto(live_server)
    expect(page).to_have_title("Overview")
    expect(page.get_by_role("navigation").first).to_have_css(
        "background-color", "rgb(0, 0, 0)"
    )

    page.get_by_label("Report:").select_option("cluster-report")
    expect(page).to_have_url(re.compile(r"/report\?report=cluster-report$"))
    expect(page.get_by_role("link", name="Download CSV")).to_be_visible()
    expect(page.locator(".tabulator-row")).to_have_count(3)
    expect(page.locator(".tabulator-row").first).to_contain_text("Parent")

    page.get_by_role("link", name="Single experiment").click()
    page.get_by_label("Experiment:").select_option("1")
    expect(page.get_by_text("Overview", exact=True).last).to_be_visible()
    expect(page.get_by_text("Report", exact=True)).to_be_visible()
    expect(page.get_by_text("Survivals", exact=True)).to_be_visible()
    expect(page.get_by_text("Cluster", exact=True)).to_be_visible()

    assert_clean_browser(*errors)


def test_selected_experiment_opens_overview_with_summary_and_config(
    page: Page, live_server
):
    errors = watch_browser_errors(page)

    page.goto(f"{live_server}/experiment?experiment=1")

    experiment_tabs = page.get_by_role("navigation").nth(1).get_by_role("link")
    expect(experiment_tabs.first).to_have_text("Overview")
    expect(experiment_tabs.first.locator("..")).to_have_class(
        re.compile(r"\bbg-slate-300\b")
    )

    overview = page.get_by_role("main")
    expect(overview.get_by_role("heading", name="Overview")).to_be_visible()
    table = overview.get_by_role("table")
    for label, value in [
        ("Experiment run ID", "1"),
        ("Configuration name", "Browser test"),
        ("Start time", "2 Jan 2026, 03:04:05"),
        ("Successes", "1"),
        ("Failures", "0"),
        ("Pathway mode", "Single pathway"),
    ]:
        row = table.get_by_role("row").filter(has_text=label)
        expect(row.get_by_role("cell")).to_have_text(value)

    viewer = overview.locator("json-viewer")
    expect(viewer).to_be_visible()
    expect(viewer).to_have_css("background-color", "rgb(245, 245, 245)")
    expect(viewer).to_have_css("color", "rgb(51, 51, 51)")
    assert viewer.evaluate("element => element.data") == {
        "experiment_config": {"temperature": 300, "label": "experiment"},
        "run_config": {"simulation_mode": "SINGLE_CLUSTER", "label": "run"},
    }

    page.set_viewport_size({"width": 320, "height": 720})
    expect(table).to_be_visible()
    expect(viewer).to_be_visible()

    assert_clean_browser(*errors)


def test_unknown_experiment_has_a_clear_empty_state(page: Page, live_server):
    errors = watch_browser_errors(page)

    page.goto(f"{live_server}/experiment?experiment=999")

    expect(page.get_by_text("Experiment 999 was not found.", exact=True)).to_be_visible()
    expect(page.locator("json-viewer")).to_have_count(0)

    assert_clean_browser(*errors)


def test_report_can_be_sorted_and_downloaded(page: Page, live_server):
    errors = watch_browser_errors(page)
    page.goto(f"{live_server}/report?report=cluster-report")
    expect(page.locator(".tabulator-row")).to_have_count(3)

    page.locator(
        '.tabulator-col[tabulator-field="cluster_atomic_mass"]'
    ).click()
    expect(page.locator(".tabulator-row").first).to_contain_text("Product A")

    expect(page.get_by_role("link", name="Download CSV")).to_have_attribute(
        "href", re.compile(r"/report/download\?report=cluster-report$")
    )

    assert_clean_browser(*errors)


def test_experiment_report_only_shows_the_selected_run(page: Page, live_server):
    errors = watch_browser_errors(page)
    page.goto(f"{live_server}/experiment")
    page.get_by_label("Experiment:").select_option("1")
    page.get_by_text("Report", exact=True).click()
    page.get_by_label("Report:").select_option("experiment-cluster-report")

    expect(page.locator(".tabulator-row")).to_have_count(1)
    expect(page.locator(".tabulator-row").first).to_contain_text("Browser test")
    expect(page.locator(".tabulator-row").first).not_to_contain_text("Other run")

    download_url = page.get_by_role("link", name="Download CSV").get_attribute("href")
    assert "experiment=1" in download_url
    response = page.request.get(download_url, timeout=5_000)
    assert response.ok
    rows = list(csv.DictReader(io.StringIO(response.text())))
    assert len(rows) == 1
    assert rows[0]["experiment_run_id"] == "1"

    assert_clean_browser(*errors)
