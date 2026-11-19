"""Sparse overview loading preserves outcomes and complete detail histories."""

import json

import polars as pl
import pytest
from apitofsim.workflow.db import ExperimentDatabase
from bokeh.document import Document
from bokeh.models import Div
from polars.testing import assert_frame_equal

from tests.arrow_fixtures import event as recorded_event, publish_events

from apitofresview.plotting.explorer import data, session
from apitofresview.plotting.explorer import _overview, _parquet

from . import test_tree
from .helpers import collision_cohort

tree_database = test_tree.tree_database


def test_overview_matches_tree_histories(tree_database):
    members, events, _, _ = data.load_data(tree_database, 1, 1)
    overview, sparse, _, regions = data.load_overview(tree_database, 1, 1)
    assert_frame_equal(overview.select(members.columns), members)
    assert (
        overview["collisions"].sum()
        == events.filter(pl.col("type") == "collision").height
    )
    assert sparse.height == events.height  # This fixture has no collisions.
    assert "velocity" not in sparse.columns
    assert_frame_equal(
        data.load_histories(
            tree_database, 1, 1, regions, members["id"].head(1), tree=True
        ),
        events.filter(pl.col("realization_id") == members["id"][0]),
    )


@pytest.mark.parametrize("ordered", [True, False])
def test_parquet_tree_overview_matches_sql(tmp_path, monkeypatch, ordered):
    path = tmp_path / "tree.duckdb"
    with ExperimentDatabase(path) as db:
        db.create_tables()
        db.db.execute("""
            insert into cluster (id,common_name,charge) values (1,'Parent',-1),(2,'Product',-1);
            insert into pathway values (7,1,2,2);
            insert into experiment_config (id) values (1);
            insert into experiment_run (id,experiment_config_id,simulation_mode) values (1,1,'CLUSTER_TREE');
            insert into cluster_tree_experiment_result (id,experiment_run_id,cluster_id)
                values (10,1,1),(20,1,1);
            insert into cluster_tree_node values (10,0,1),(10,1,2),(20,0,1);
            insert into cluster_tree_transition values (10,0,0,7,1);
        """)
        db.begin_execution(1, 1, None, [7], result_id=10, recorded=True)
        db.begin_execution(1, 1, None, [], result_id=20, recorded=True)
        rows = [
            recorded_event(10, 0, 0, "init", 0, 0),
            recorded_event(10, 0, 1, "collision", 4, 4, x=3, y=4, accepted=1),
            recorded_event(10, 0, 2, "postcollision", 4, 4),
            recorded_event(10, 0, 3, "collision", 3, -1, accepted=0),
            recorded_event(
                10,
                0,
                4,
                "fragmentation",
                3,
                1,
                transition_index=0,
                next_particle_index=1,
            ),
            recorded_event(10, 0, 5, "postfragmentation", 3, 1, particle_index=1),
            recorded_event(10, 0, 6, "escape", 3, 2, particle_index=1),
            recorded_event(10, 1, 0, "init", 0, 0),
            recorded_event(10, 1, 1, "collision", 1, 1, particle_index=3, accepted=1),
            recorded_event(10, 1, 2, "postcollision", 1, 1),
            recorded_event(10, 2, 0, "init", 0, 0),
            recorded_event(
                10,
                2,
                1,
                "fragmentation",
                1,
                1,
                transition_index=0,
                next_particle_index=0,
            ),
            recorded_event(10, 2, 2, "escape", 2, 2),
            recorded_event(20, 0, 0, "init", 0, 0),
            recorded_event(20, 0, 1, "escape", 3, 3),
        ]
        if not ordered:
            rows[1], rows[2] = rows[2], rows[1]
        publish_events(db, rows, format="parquet", row_group_size=3)

    with ExperimentDatabase(path, readonly=True) as db:
        monkeypatch.setattr(data, "get_geometry", lambda *_: None)
        monkeypatch.setattr(
            data, "lengths_to_cumulative_lengths", lambda *_: (0, 1, 2, 3, 4, 5)
        )
        expected = data.load_overview(db, 1, 1)
        monkeypatch.setattr(_overview, "MIN_STREAM_EVENTS", 1)
        monkeypatch.setattr(_overview, "BATCH_SIZE", 2)
        assert (_overview.stream_tree_overview(db, 1, 1) is not None) == ordered
        for fast, sql in zip(data.load_overview(db, 1, 1), expected, strict=True):
            assert_frame_equal(fast, sql, check_exact=True)

        # Keep realizations scoped to their execution, including in the chooser.
        keys = ("10:1", "20:0")
        ids = ("10:1:2", "20:0:1")
        baseline = data._event_frame(db, 1, 1, True).filter(
            pl.col("realization_id").is_in(keys) & pl.col("id").is_in(ids)
        )
        indexed = _parquet.parquet_history_queries(db, data._history_keys(keys))
        assert indexed is not None
        result = data._event_frame(db, 1, 1, True, members=keys, ids=ids)
        assert result["continuity_valid"].to_list() == [False, True]
        assert_frame_equal(result, baseline, check_exact=True)

        # Missing statistics must fall back to the exact numeric SQL lookup.
        monkeypatch.setattr(_parquet, "_row_group_index", lambda *_: None)
        result = data._event_frame(db, 1, 1, True, members=keys, ids=ids)
        assert_frame_equal(result, baseline, check_exact=True)


def test_sparse_session_loads_complete_details_once(monkeypatch):
    members, events, pathways, regions = collision_cohort()
    overview = members.with_columns(
        pl.lit(3, dtype=pl.UInt64).alias("collisions"),
        pl.lit(1, dtype=pl.UInt64).alias("rejected_collisions"),
        pl.lit(0).alias("clamped_events"),
        pl.lit(5.0).alias("max_radial"),
    )
    sparse = events.filter(~pl.col("type").is_in(["collision", "postcollision"]))
    monkeypatch.setattr(session, "MAX_COLLISIONS", 1)
    monkeypatch.setattr(
        session, "load_data", lambda *args: (overview, sparse, pathways, regions)
    )
    loads = []

    def load(*args, **kwargs):
        loads.append(args[4])
        return events

    monkeypatch.setattr(session, "load_histories", load)
    doc = Document()
    session.build_document(None, doc, 1, 1)
    assert not loads
    assert doc.select_one({"name": "event:collision"}).disabled
    source = next(
        r.data_source
        for r in doc.select_one({"name": "events"}).renderers
        if hasattr(r, "data_source") and "id" in r.data_source.data
    )
    source.selected.indices = [0]
    assert loads == [(1,)]
    details = next(
        d for d in doc.select({"type": Div}) if "Realization #1</h3>" in d.text
    )
    assert 'aria-label="Inspect event #3"' in details.text
    assert "Axial velocity" in details.text
    doc.roots[0].children[1].text = json.dumps(["inspect", "3"])
    assert loads == [(1,)]
    assert 'style="background:#fde68a"' in details.text


def test_sparse_pager_loads_collisions_within_budget(monkeypatch):
    from bokeh.models import CheckboxGroup, RangeSlider

    from .helpers import cohort, event

    members, events, pathways, regions = cohort(
        [1, 2, 3],
        event(1, "collision", 1, 0, 1),
        event(2, "collision", 1, 1, 2),
        event(3, "escape", 1, 2, 5),
        event(4, "collision", 2, 0, 1),
        event(5, "collision", 2, 1, 2),
        event(6, "escape", 2, 2, 5),
        event(7, "collision", 3, 0, 1),
        event(8, "collision", 3, 1, 2),
        event(9, "escape", 3, 2, 5),
    )
    members = members.with_columns(
        pl.lit(2).alias("collisions"),
        pl.lit(0).alias("rejected_collisions"),
        pl.lit(0).alias("clamped_events"),
        pl.lit(5.0).alias("max_radial"),
    )
    sparse = events.filter(pl.col("type") == "escape")
    loads = []

    def load(db, run, cluster, regions, members, **kwargs):
        loads.append(members)
        return events.filter(pl.col("realization_id").is_in(members))

    monkeypatch.setattr(
        session, "load_data", lambda *args: (members, sparse, pathways, regions)
    )
    monkeypatch.setattr(session, "load_histories", load)
    monkeypatch.setattr(session, "MAX_COLLISIONS", 4)
    doc = Document()
    session.build_document(None, doc, 1, 1)
    assert not loads
    collision = doc.select_one({"name": "event:collision"})
    assert collision.disabled
    pager = next(
        w
        for w in doc.select({"type": CheckboxGroup})
        if w.labels == ["Realizations pager"]
    )
    pager.active = [0]
    doc.select_one({"type": RangeSlider}).value = (1, 2)
    assert loads == [(1, 2)]
    assert not collision.disabled
    source = next(
        r.data_source
        for r in doc.select_one({"name": "events"}).renderers
        if hasattr(r, "data_source") and "id" in r.data_source.data
    )
    assert source.data["type"].count("collision") == 4
    source.selected.indices = [source.data["id"].index(3)]
    assert loads == [(1, 2)]

    slider = doc.select_one({"type": RangeSlider})
    slider.value = (2, 3)
    assert loads == [(1, 2), (3,)]
    slider.value = (3, 3)
    assert loads == [(1, 2), (3,)]
    slider.value = (1, 1)
    assert loads == [(1, 2), (3,), (1,)]
    pager.active = []
    assert collision.disabled


def test_cohort_history_cache_preserves_order_and_numbering():
    from .helpers import cohort, event

    _, events, _, _ = cohort(
        [1, 2, 10],
        event(1, "escape", 1, 1, 5),
        event(2, "escape", 2, 1, 5),
        event(3, "escape", 10, 1, 5),
    )
    events = events.with_columns(pl.col("realization_id").cast(pl.String))

    def load(ids):
        return (
            events.filter(pl.col("realization_id").is_in(ids))
            .drop("event_number")
            .with_row_index("event_number", offset=1)
        )

    histories = data.CohortHistories(load)
    histories(("2", "10"))
    assert_frame_equal(histories(("1", "2", "10")), load(("1", "2", "10")))
    assert_frame_equal(histories(("2",)), load(("2",)))

    uncached = data.CohortHistories(load, max_bytes=0)
    uncached(("1",))
    assert uncached.events is None
