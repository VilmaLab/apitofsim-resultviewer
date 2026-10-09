"""Explorer data behavior."""

from types import SimpleNamespace

import duckdb
import polars as pl
import pytest

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
def test_adapter_resolves_results_and_preserves_ids(monkeypatch, negative_z, tmp_path):
    database_path = str(tmp_path / "explorer.duckdb")
    conn = duckdb.connect(database_path)
    conn.execute(
        "create table multi_pathway_experiment_result(id int, experiment_run_id int, cluster_id int)"
    )
    conn.execute(
        "create table single_pathway_experiment_result(id int, experiment_run_id int, pathway_id int)"
    )
    conn.execute(
        "create table pathway(id int, cluster_id int, product1_id int, product2_id int)"
    )
    conn.execute("create table cluster(id int, common_name varchar)")
    conn.execute("create table realization(id int, experiment_result_id int)")
    conn.execute("""
        create table event_info(
            id int, realization_id int, event_type varchar,
            postime struct(x double,y double,z double,t double),
            velocity struct(x double,y double,z double) default {'x':1,'y':2,'z':3},
            omega struct(x double,y double,z double) default {'x':4,'y':5,'z':6},
            rot_energy double default 1e-21,
            vibrational_energy double default 2e-20,
            particle_index int default 0
        )
    """)
    conn.execute(
        "create table collision_event(id int, theta double, u_norm double, accepted bool)"
    )
    conn.execute("create table fragmentation_event(id int, pathway_id int)")
    conn.execute(
        "insert into cluster values (1,'5A_5SA_negative'),(2,'4A_5SA_negative'),(3,'1A_neutral')"
    )
    conn.execute("insert into pathway values (7,1,2,3)")
    conn.execute("insert into multi_pathway_experiment_result values (10,1,1),(11,2,1)")
    conn.execute("insert into single_pathway_experiment_result values (12,1,7)")
    conn.execute("insert into realization values (100,10),(101,11),(102,12)")
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
    conn.close()
    conn = duckdb.connect(database_path, read_only=True)
    db = SimpleNamespace(db=conn)
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
    conn.close()
    conn = duckdb.connect(database_path)
    # UNION of result IDs prevents duplicate histories across result tables.
    conn.execute("insert into single_pathway_experiment_result values (10,1,7)")
    conn.execute("insert into multi_pathway_experiment_result values (13,1,9)")
    conn.execute("insert into realization values (103,13)")
    conn.execute(
        "insert into event_info (id,realization_id,event_type,postime) values (504,101,'collision', {'x':1,'y':2,'z':0,'t':1}), (505,103,'collision', {'x':1,'y':2,'z':0,'t':1}), (506,100,'collision', {'x':1,'y':2,'z':0,'t':0.5})"
    )
    conn.close()
    conn = duckdb.connect(database_path, read_only=True)
    db = SimpleNamespace(db=conn)
    realizations, events, _, _ = data.load_data(db, 1, 1)
    assert realizations["id"].to_list() == [100, 102]
    assert events["id"].to_list() == [500, 501, 506, 502, 503]
    empty = data.load_data(db, 99, 1)
    assert empty[0].is_empty() and empty[1].is_empty()
    conn.close()


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
