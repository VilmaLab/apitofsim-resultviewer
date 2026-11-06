"""Compatibility with apitofsim's cumulative and regional outcome plots."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from shutil import copyfile

import duckdb

import pytest
from apitofsim.api import Q_
from apitofsim.plotting import events
from apitofsim.workflow.db import ExperimentDatabase
from bokeh.document import Document
from mplbed.asgi import MplbedMiddleware
from starlette.exceptions import HTTPException
from starlette.requests import Request

from tests.arrow_fixtures import event, publish_events

from apitofresview import webapp
from apitofresview.plotting.explorer import build_document
from apitofresview.plotting.explorer import data as explorer_data


@pytest.fixture
def outcome_app(tmp_path, monkeypatch):
    path = tmp_path / "outcomes.duckdb"
    db = ExperimentDatabase(path)
    db.create_tables()
    db.db.execute("""
        insert into cluster (id,common_name) values (1,'Parent'),(2,'A'),(3,'B');
        insert into pathway values (7,1,2,3);
        insert into experiment_config (id,name,config) values (1,'Outcomes','{}');
        insert into experiment_run (id,experiment_config_id,simulation_mode,start_time)
        values (1,1,'SINGLE_CLUSTER','2026-01-01'),
               (2,1,'PATHWAY_AT_A_TIME','2026-01-01'),
               (3,1,'CLUSTER_TREE','2026-01-01');
        insert into multi_pathway_experiment_result
            (id,experiment_run_id,cluster_id,n_escaped_total) values (10,1,1,1);
        insert into single_pathway_experiment_result
            (id,experiment_run_id,pathway_id) values (11,2,7);
        insert into cluster_tree_experiment_result
            (id,experiment_run_id,cluster_id) values (12,3,1);
    """)
    db.begin_execution(1, 1, None, [7], result_id=10, recorded=True)
    db.begin_execution(2, 1, 7, [7], result_id=11, recorded=True)
    db.begin_execution(3, 1, None, [], result_id=12, recorded=True)
    db.begin_execution(1, 2, None, [], result_id=13)
    publish_events(
        db,
        [
            event(10, 0, 0, "fragmentation", 1, 0.002, transition_index=0),
            event(10, 1, 0, "escape", 2, 0.005),
            event(10, 2, 0, "init", 0, 0),
            event(11, 0, 0, "init", 0, 0),
            event(12, 0, 0, "init", 0, 0),
        ],
        statuses={10: "cancelled"},
    )
    db.close()
    monkeypatch.setattr(events, "get_geometry", lambda *args: [Q_(1, "mm")] * 5)
    app = webapp.create_app(path)
    try:
        yield app
    finally:
        app.state.db.close()


def response(app, query):
    result = None

    async def endpoint(scope, receive, send):
        nonlocal result
        result = await webapp.realizations(Request(scope))

    middleware = MplbedMiddleware(endpoint, prefix="/mplbed")
    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(
            asyncio.run,
            middleware(
                {
                    "type": "http",
                    "method": "GET",
                    "path": "/experiment/cluster/realizations",
                    "headers": [],
                    "scheme": "http",
                    "server": ("localhost", 80),
                    "query_string": query.encode(),
                    "app": app,
                    "router": app.router,
                },
                None,
                None,
            ),
        ).result()
    return result


@pytest.mark.parametrize("panel", ["all", "cumulative", "bars"])
@pytest.mark.parametrize("rescale", ["none", "equal", "schematic"])
def test_outcome_panels_render_with_current_api(outcome_app, panel, rescale):
    result = response(
        outcome_app, f"experiment=1&cluster=1&panel={panel}&rescale={rescale}"
    )
    assert result.status_code == 200
    assert result.context["notice"] is None
    assert result.context["fig"]
    assert f'<option value="{panel}" selected>' in result.body.decode()


def test_old_plot_links_use_new_default(outcome_app):
    result = response(outcome_app, "experiment=1&cluster=1&plottype=beeswarm")
    assert result.context["panel"] == "all"
    assert result.context["rescale"] == "schematic"


@pytest.mark.parametrize("query", ["panel=beeswarm", "rescale=invalid"])
def test_invalid_outcome_options_are_rejected(outcome_app, query):
    with pytest.raises(HTTPException) as error:
        response(outcome_app, f"experiment=1&cluster=1&{query}")
    assert error.value.status_code == 400


@pytest.mark.parametrize("run,mode", [(2, "pathway-at-a-time"), (3, "cluster-tree")])
def test_unsupported_outcomes_explain_mode(outcome_app, run, mode):
    result = response(outcome_app, f"experiment={run}&cluster=1")
    assert result.status_code == 200
    assert mode in result.context["notice"]
    assert not result.context["fig"]
    assert 'id="panel-select"' not in result.body.decode()


def test_tree_explorer_explains_missing_execution_tree(outcome_app, monkeypatch):
    monkeypatch.setattr(explorer_data, "get_geometry", events.get_geometry)
    doc = Document()
    build_document(outcome_app.state.db, doc, 3, 1)
    assert len(doc.roots) == 1
    assert (
        doc.roots[0].text == "No recorded cluster-tree executions for this selection."
    )


@pytest.mark.parametrize("run,cluster", [(1, 2), (2, 2), (3, 2), (99, 1)])
def test_unrecorded_selection(outcome_app, run, cluster):
    result = response(outcome_app, f"experiment={run}&cluster={cluster}")
    assert (
        result.context["notice"]
        == "No events were recorded for this experiment run and cluster."
    )
    assert not result.context["fig"]
    doc = Document()
    build_document(outcome_app.state.db, doc, run, cluster)
    assert doc.roots[0].text == result.context["notice"]


@pytest.mark.parametrize("missing", [True, False])
def test_required_sidecar_has_a_distinct_open_error(outcome_app, tmp_path, missing):
    source = outcome_app.state.db.events_path
    path = tmp_path / "copy.duckdb"
    copyfile(source.with_name("outcomes.duckdb"), path)
    if missing:
        with pytest.raises(FileNotFoundError, match="matching .events.arrow sidecar"):
            webapp.create_app(path)
    else:
        path.with_suffix(".events.arrow").write_bytes(b"wrong archive")
        with pytest.raises(ValueError, match="sidecar identity/size"):
            webapp.create_app(path)


def test_legacy_event_database_requires_migration(tmp_path):
    path = tmp_path / "legacy.duckdb"
    with duckdb.connect(str(path)) as conn:
        conn.execute("create table event_info (id integer)")
    with pytest.raises(ValueError, match="Legacy relational-event database"):
        webapp.create_app(path)


def test_recorded_cohort_with_hidden_events(outcome_app, monkeypatch):
    monkeypatch.setattr(explorer_data, "get_geometry", events.get_geometry)
    doc = Document()
    build_document(outcome_app.state.db, doc, 1, 1)
    for kind in explorer_data.EVENT_TYPES:
        doc.select_one({"name": f"event:{kind}"}).active = []
    plots = doc.select_one({"name": "explorer_plots"})
    assert any(
        getattr(child, "text", "") == "<p>No events match the current filters.</p>"
        for child in plots.children
    )
