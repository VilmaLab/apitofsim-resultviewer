"""Compatibility with apitofsim's cumulative and regional outcome plots."""

import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest
from apitofsim.api import Q_
from apitofsim.plotting import events
from apitofsim.workflow.db import RealizationDatabase
from bokeh.document import Document
from mplbed.asgi import MplbedMiddleware
from starlette.exceptions import HTTPException
from starlette.requests import Request

from apitofresview import webapp
from apitofresview.plotting.explorer import build_document


@pytest.fixture
def outcome_app(tmp_path, monkeypatch):
    path = tmp_path / "outcomes.duckdb"
    db = RealizationDatabase(path)
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
        insert into realization values (100,10),(101,10),(102,10);
        insert into event_info
            (id,realization_id,event_type,postime,velocity,omega,
             rot_energy,vibrational_energy,particle_index) values
            (1,100,'fragmentation',{'x':0,'y':0,'z':0.002,'t':1},
             {'x':0,'y':0,'z':1},{'x':0,'y':0,'z':0},0,0,0),
            (2,101,'escape',{'x':0,'y':0,'z':0.005,'t':2},
             {'x':0,'y':0,'z':1},{'x':0,'y':0,'z':0},0,0,0);
        insert into fragmentation_event (id,pathway_id) values (1,7);
    """)
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


def test_tree_explorer_does_not_classify_serial_splits_as_ambiguous(outcome_app):
    doc = Document()
    build_document(outcome_app.state.db, doc, 3, 1)
    assert len(doc.roots) == 1
    assert "does not yet support cluster-tree histories" in doc.roots[0].text
