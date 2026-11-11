"""Sparse overview loading preserves outcomes and complete detail histories."""

import json

import polars as pl
from bokeh.document import Document
from bokeh.models import Div
from polars.testing import assert_frame_equal

from apitofresview.plotting.explorer import data, session

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
        [1, 2],
        event(1, "collision", 1, 0, 1),
        event(2, "collision", 1, 1, 2),
        event(3, "escape", 1, 2, 5),
        event(4, "collision", 2, 0, 1),
        event(5, "collision", 2, 1, 2),
        event(6, "escape", 2, 2, 5),
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
    monkeypatch.setattr(session, "MAX_COLLISIONS", 2)
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
    doc.select_one({"type": RangeSlider}).value = (1, 1)
    assert loads == [(1,)]
    assert not collision.disabled
    source = next(
        r.data_source
        for r in doc.select_one({"name": "events"}).renderers
        if hasattr(r, "data_source") and "id" in r.data_source.data
    )
    assert source.data["type"].count("collision") == 2
    pager.active = []
    assert loads == [(1,)]
    assert collision.disabled
