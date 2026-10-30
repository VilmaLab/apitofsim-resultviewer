"""Explorer data behavior."""

import duckdb
import polars as pl
import pytest
from apitofsim.workflow.db import RealizationDatabase

from apitofresview.plotting.explorer import data

from .helpers import cohort, event, outcome_cohort


@pytest.mark.parametrize(
    "history,fate,terminal",
    [
        ([], "Incomplete", None),
        ([event(1, "collision", 1, 1, 1)], "Incomplete", None),
        ([event(1, "fragmentation", 1, 1, 1)], "Incomplete", None),
        ([event(1, "fragmentation", 1, 1, 6, 7)], "Incomplete", None),
        ([event(1, "escape", 1, 1, 6)], "Parent → Parent", 1),
        (
            [event(1, "escape", 1, 1, 5), event(2, "collision", 1, 2, 4)],
            "Incomplete",
            None,
        ),
        (
            [
                event(1, "fragmentation", 1, 1, 2, 7),
                event(2, "fragmentation", 1, 1, 2, 7),
            ],
            "Ambiguous",
            None,
        ),
        (
            [event(1, "fragmentation", 1, 1, 2, 7), event(2, "escape", 1, 2, 5)],
            "Ambiguous",
            None,
        ),
        (
            [event(2, "fragmentation", 1, 1, 5, 7), event(1, "collision", 1, 1, 5)],
            "Parent → A + B",
            2,
        ),
        (
            [event(1, "fragmentation", 1, 1, 2, 7), event(2, "collision", 1, 1, 2)],
            "Incomplete",
            None,
        ),
    ],
)
def test_terminal_histories(history, fate, terminal):
    realizations, events, _, _ = cohort([1], *history)
    assert realizations["fate"][0] == fate
    assert realizations["terminal_event_id"][0] == terminal
    assert events["id"].to_list() == sorted(e[0] for e in history)


@pytest.mark.parametrize("negative_z", [False, True])
@pytest.mark.parametrize("attached", [False, True])
def test_adapter_resolves_results_and_preserves_ids(
    monkeypatch, negative_z, attached, tmp_path
):
    database_path = str(tmp_path / "explorer.duckdb")
    db = RealizationDatabase(database_path)
    db.create_tables()
    conn = db.db
    for column, value in {
        "velocity": "{'x':1,'y':2,'z':3}",
        "omega": "{'x':4,'y':5,'z':6}",
        "rot_energy": "1e-21",
        "vibrational_energy": "2e-20",
        "particle_index": "0",
    }.items():
        conn.execute(f"alter table event_info alter {column} set default {value}")
    conn.execute("""
        insert into cluster (id, common_name) values
        (1,'5A_5SA_negative'),(2,'4A_5SA_negative'),(3,'1A_neutral');
        insert into pathway values (7,1,2,3);
        insert into experiment_config (id) values (1);
        insert into experiment_run (id, experiment_config_id, simulation_mode)
        values (1,1,'SINGLE_CLUSTER'),(2,1,'SINGLE_CLUSTER');
        insert into multi_pathway_experiment_result (id,experiment_run_id,cluster_id)
        values (10,1,1),(11,2,1);
        insert into single_pathway_experiment_result (id,experiment_run_id,pathway_id)
        values (12,1,7);
        insert into realization values (100,10),(101,11),(102,12);
    """)
    conn.execute(
        "insert into event_info (id,realization_id,event_type,postime) values (501,100,'collision', {'x':1,'y':2,'z':0,'t':0.5})"
    )
    conn.execute(
        "insert into event_info (id,realization_id,event_type,postime) values (502,100,'fragmentation', {'x':1,'y':2,'z':2,'t':1.5})"
    )
    conn.execute(
        "insert into event_info (id,realization_id,event_type,postime) values (503,102,'escape', {'x':1,'y':2,'z':5,'t':2})"
    )
    conn.execute("insert into collision_event values (501,0.75,-114.5,false)")
    conn.execute("insert into fragmentation_event values (502,7)")
    conn.execute(
        "insert into event_info (id,realization_id,event_type,postime) values (500,100,'init', {'x':0,'y':0,'z':0,'t':0})"
    )
    if negative_z:
        conn.execute(
            "update event_info set postime = struct_update(postime, z := -0.000001) where id = 501"
        )
        conn.execute(
            "update event_info set postime = struct_update(postime, z := -0.000002) where id = 502"
        )
    monkeypatch.setattr(data, "get_geometry", lambda *args: None)
    monkeypatch.setattr(
        data, "lengths_to_cumulative_lengths", lambda *args: (0, 1, 2, 3, 4, 5)
    )
    snapshots = {
        table: conn.execute(f"select * from {table}").fetchall()
        for table in (
            "realization",
            "collision_event",
            "fragmentation_event",
            "event_info",
        )
    }
    db.close()
    if attached:
        cluster_path = str(tmp_path / "clusters.duckdb")
        clusters = duckdb.connect(cluster_path)
        clusters.execute(
            f"attach '{database_path.replace(chr(39), chr(39) * 2)}' as results (read_only)"
        )
        for table in ("cluster", "pathway"):
            clusters.execute(f"create table {table} as select * from results.{table}")
        clusters.close()
        conn = duckdb.connect(":memory:")
        conn.execute(
            f"attach '{database_path.replace(chr(39), chr(39) * 2)}' as results (read_only)"
        )
        conn.execute(
            f"attach '{cluster_path.replace(chr(39), chr(39) * 2)}' as clusters (read_only)"
        )
        for (table,) in conn.execute(
            "select table_name from information_schema.tables "
            "where table_catalog = 'results' and table_type = 'BASE TABLE'"
        ).fetchall():
            source = "clusters" if table in ("cluster", "pathway") else "results"
            conn.execute(f"create view {table} as select * from {source}.{table}")
        db = RealizationDatabase((conn, None))
        db.refresh_views()
    else:
        db = RealizationDatabase(database_path, readonly=True)
    conn = db.db
    realizations, events, pathways, regions = data.load_data(db, 1, 1)
    assert realizations["id"].to_list() == [100, 102]
    assert realizations["fate"].to_list() == [
        "5A_5SA_negative → 4A_5SA_negative + 1A_neutral",
        "5A_5SA_negative → 5A_5SA_negative",
    ]
    assert events["id"].to_list() == [500, 501, 502, 503]
    collision = events.filter(pl.col("id") == 501).row(0, named=True)
    assert collision["theta"] == 0.75 and collision["u_norm"] == -114.5
    assert collision["accepted"] is False
    assert collision["velocity"] == {"x": 1, "y": 2, "z": 3}
    assert collision["rot_energy"] == 1e-21
    assert collision["vibrational_energy"] == 2e-20
    assert collision["particle_index"] == 0
    assert events.filter(pl.col("id") == 502)["pathway_id"][0] == 7
    assert pathways.to_dict(as_series=False) == {
        "pathway_id": [7],
        "pathway": ["5A_5SA_negative → 4A_5SA_negative + 1A_neutral"],
    }
    assert events["z_clamped"].sum() == (2 if negative_z else 0)
    if negative_z:
        assert events.filter(pl.col("z_clamped"))["z"].to_list() == [0, 0]
        assert events.filter(pl.col("id") == 502)["region"][0] == 0
    for table, snapshot in snapshots.items():
        assert conn.execute(f"select * from {table}").fetchall() == snapshot
    db.close()
    db = RealizationDatabase(database_path)
    conn = db.db
    conn.execute(
        "insert into multi_pathway_experiment_result (id,experiment_run_id,cluster_id) values (13,1,2)"
    )
    conn.execute("insert into realization values (103,13),(104,10),(105,12)")
    conn.execute(
        "insert into event_info (id,realization_id,event_type,postime) "
        "values (507,105,'fragmentation', {'x':0,'y':0,'z':2,'t':1})"
    )
    conn.execute(
        "insert into event_info (id,realization_id,event_type,postime) values (504,101,'collision', {'x':1,'y':2,'z':0,'t':1}), (505,103,'collision', {'x':1,'y':2,'z':0,'t':1}), (506,100,'collision', {'x':1,'y':2,'z':0,'t':0.5})"
    )
    db.close()
    db = RealizationDatabase(database_path, readonly=True)
    conn = db.db
    realizations, events, _, _ = data.load_data(db, 1, 1)
    assert realizations["id"].to_list() == [100, 102, 104, 105]
    assert realizations["fate"].to_list()[-2:] == ["Incomplete", "Incomplete"]
    assert events["id"].to_list() == [500, 501, 506, 502, 503, 507]
    assert events.filter(pl.col("id") == 506)["theta"][0] is None
    assert events.filter(pl.col("id") == 507)["pathway_id"][0] is None
    empty = data.load_data(db, 99, 1)
    assert empty[0].is_empty() and empty[1].is_empty()
    db.close()


def test_fates_and_preprocessing():
    realizations, events, _, _ = outcome_cohort()
    assert realizations["fate"].to_list() == [
        "Parent → A + B",
        "Parent → A + B",
        "Parent → Parent",
        "Incomplete",
        "Ambiguous",
    ]
    assert realizations["terminal_event_id"].to_list() == [2, 3, 4, None, None]
    assert events["radial"].to_list() == [5] * events.height


def test_initial_event_fate():
    members, _, _, _ = cohort(
        [1], event(1, "init", 1, 0, 0), event(2, "escape", 1, 1, 5)
    )
    assert members["terminal_event_id"].to_list() == [2]
    assert members["fate"].to_list() == ["Parent → Parent"]


@pytest.mark.parametrize(
    "accepted,next_kind,next_rid,next_time,next_z,expected",
    [
        (True, "postcollision", 1, 1, 1, 1),
        (False, "postcollision", 1, 1, 1, None),
        (None, "postcollision", 1, 1, 1, None),
        (True, "collision", 1, 1, 1, None),
        (True, "postcollision", 2, 1, 1, None),
        (True, "postcollision", 1, 2, 1, None),
        (True, "postcollision", 1, 1, 2, None),
    ],
)
def test_collision_pairing(accepted, next_kind, next_rid, next_time, next_z, expected):
    _, events, _, _ = cohort(
        [1, 2],
        event(1, "collision", 1, 1, 1),
        event(2, next_kind, next_rid, next_time, next_z),
    )
    events = data.pair_collisions(
        events.with_columns(pl.Series("accepted", [accepted, None], dtype=pl.Boolean))
    )
    assert events["collision_id"].to_list() == [None, expected]


def test_pairing_does_not_skip_records_or_reuse_a_collision():
    _, events, _, _ = cohort(
        [1],
        event(1, "collision", 1, 1, 1),
        event(2, "postcollision", 1, 1, 1),
        event(3, "postcollision", 1, 1, 1),
        event(4, "collision", 1, 1, 1),
        event(5, "init", 1, 1, 1),
        event(6, "postcollision", 1, 1, 1),
        event(7, "collision", 1, 1, 1),
        event(8, "postcollision", 1, 1, 1),
    )
    events = data.pair_collisions(events.with_columns(pl.lit(True).alias("accepted")))
    assert events["collision_id"].to_list() == [
        None,
        1,
        None,
        None,
        None,
        None,
        None,
        7,
    ]


def test_pathway_cohorts_keep_simulated_pathway_for_escaped_and_empty_results(
    monkeypatch, tmp_path
):
    path = tmp_path / "independent.duckdb"
    db = RealizationDatabase(path)
    db.create_tables()
    db.db.execute("""
        insert into cluster (id,common_name) values (1,'Parent'),(2,'A'),(3,'B');
        insert into pathway values (7,1,2,3),(8,1,3,2),(9,1,2,2),(10,1,3,3);
        insert into experiment_config (id) values (1);
        insert into experiment_run (id,experiment_config_id,simulation_mode)
        values (1,1,'PATHWAY_AT_A_TIME');
        insert into single_pathway_experiment_result (id,experiment_run_id,pathway_id)
        values (10,1,7),(11,1,8),(12,1,9);
        insert into realization values (100,10),(200,11);
    """)
    for column, value in {
        "velocity": "{'x':1,'y':2,'z':3}",
        "omega": "{'x':4,'y':5,'z':6}",
        "rot_energy": "1e-21",
        "vibrational_energy": "2e-20",
        "particle_index": "0",
    }.items():
        db.db.execute(f"alter table event_info alter {column} set default {value}")
    db.db.execute("""
        insert into event_info (id,realization_id,event_type,postime) values
        (1,100,'fragmentation',{'x':0,'y':0,'z':2,'t':1}),
        (2,200,'escape',{'x':0,'y':0,'z':5,'t':2});
        insert into fragmentation_event values (1,7);
    """)
    db.close()
    monkeypatch.setattr(data, "get_geometry", lambda *args: None)
    monkeypatch.setattr(
        data, "lengths_to_cumulative_lengths", lambda *args: (0, 1, 2, 3, 4, 5)
    )
    db = RealizationDatabase(path, readonly=True)
    try:
        members, events, pathways, _ = data.load_data(db, 1, 1)
        assert members["simulated_pathway_id"].to_list() == [7, 8]
        assert members["simulated_pathway"].to_list() == [
            "Parent → A + B",
            "Parent → B + A",
        ]
        assert members["fate"].to_list() == ["Parent → A + B", "Parent → Parent"]
        assert events.filter(pl.col("type") == "escape")["pathway_id"][0] is None
        assert pathways.filter(pl.col("simulated"))["pathway_id"].to_list() == [7, 8, 9]
        assert (
            pathways.height == 4
        )  # Other pathways remain available for rate comparisons.
    finally:
        db.close()
