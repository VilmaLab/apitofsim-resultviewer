"""Explorer data behavior."""

import duckdb
import polars as pl
import pytest
from apitofsim.plotting.events import NoRecordedEventsError
from apitofsim.recording import events_filename
from apitofsim.workflow.db import ExperimentDatabase

from tests.arrow_fixtures import event as recorded_event, publish_events

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
    db = ExperimentDatabase(database_path)
    db.create_tables()
    conn = db.db
    conn.execute("""
        insert into cluster (id, common_name) values
        (1,'5A_5SA_negative'),(2,'4A_5SA_negative'),(3,'1A_neutral');
        insert into pathway values (7,1,2,3);
        insert into experiment_config (id) values (1);
        insert into experiment_run (id, experiment_config_id, simulation_mode)
        values (1,1,'SINGLE_CLUSTER'),(2,1,'SINGLE_CLUSTER');
        insert into multi_pathway_experiment_result (id,experiment_run_id,cluster_id)
        values (10,1,1),(11,2,1),(9,1,1),(13,1,2);
        insert into single_pathway_experiment_result (id,experiment_run_id,pathway_id)
        values (12,1,7);
    """)
    conn.execute("update cluster set charge = case when id = 3 then 0 else -1 end")
    db.begin_execution(1, 1, None, [7], result_id=10, recorded=True)
    db.begin_execution(2, 1, None, [], result_id=11)
    db.begin_execution(1, 1, 7, [7], result_id=12, recorded=True)
    db.begin_execution(1, 1, None, [], result_id=9, recorded=True)
    db.begin_execution(1, 2, None, [], result_id=13, recorded=True)
    publish_events(
        db,
        [
            recorded_event(
                10,
                0,
                10,
                "fragmentation",
                0.5,
                -2e-6 if negative_z else 2,
                transition_index=0,
            ),
            recorded_event(
                10,
                0,
                2,
                "collision",
                0.5,
                -1e-6 if negative_z else 0,
                vx=1,
                vy=2,
                vz=3,
                rot_energy=1e-21,
                vibrational_energy=2e-20,
                theta=0.75,
                u_norm=-114.5,
            ),
            recorded_event(10, 0, 0, "init", 0, 0),
            recorded_event(12, 0, 0, "escape", 2, 5),
            recorded_event(9, 11, 0, "init", 0, 0),
            recorded_event(9, 2, 0, "init", 0, 0),
            recorded_event(13, 0, 0, "init", 0, 0),
        ],
        statuses={9: "failed", 13: "cancelled"},
    )
    monkeypatch.setattr(data, "get_geometry", lambda *args: None)
    monkeypatch.setattr(
        data, "lengths_to_cumulative_lengths", lambda *args: (0, 1, 2, 3, 4, 5)
    )
    snapshot = conn.sql(
        "select * from event_details order by result_id, realization_index, event_index"
    ).fetchall()
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
        db = ExperimentDatabase(
            (conn, None), readonly=True, events_filename=events_filename(database_path)
        )
        db.refresh_views()
    else:
        db = ExperimentDatabase(database_path, readonly=True)
    conn = db.db
    realizations, events, pathways, regions = data.load_data(db, 1, 1)
    assert realizations["id"].to_list() == ["9:2", "9:11", "10:0", "12:0"]
    assert realizations["fate"].to_list() == [
        "Incomplete",
        "Incomplete",
        "5A_5SA_negative → 4A_5SA_negative + 1A_neutral",
        "5A_5SA_negative → 5A_5SA_negative",
    ]
    assert events["id"].to_list() == [
        "9:2:0",
        "9:11:0",
        "10:0:0",
        "10:0:2",
        "10:0:10",
        "12:0:0",
    ]
    collision = events.filter(pl.col("id") == "10:0:2").row(0, named=True)
    assert collision["theta"] == 0.75 and collision["u_norm"] == -114.5
    assert collision["accepted"] is False
    assert collision["velocity"] == {"x": 1, "y": 2, "z": 3}
    assert collision["rot_energy"] == 1e-21
    assert collision["vibrational_energy"] == 2e-20
    assert collision["particle_index"] == 0
    assert events.filter(pl.col("id") == "10:0:10")["pathway_id"][0] == 7
    assert pathways.to_dict(as_series=False) == {
        "pathway_id": [7],
        "daughter": ["4A_5SA_negative"],
        "charged_products": [[{"id": 2, "name": "4A_5SA_negative"}]],
        "pathway": ["5A_5SA_negative → 4A_5SA_negative + 1A_neutral"],
    }
    assert events["z_clamped"].sum() == (2 if negative_z else 0)
    if negative_z:
        assert events.filter(pl.col("z_clamped"))["z"].to_list() == [0, 0]
        assert events.filter(pl.col("id") == "10:0:10")["region"][0] == 0
    assert (
        conn.sql(
            "select * from event_details order by result_id, realization_index, event_index"
        ).fetchall()
        == snapshot
    )
    for run, cluster in [(2, 1), (1, 3), (99, 1)]:
        with pytest.raises(NoRecordedEventsError):
            data.load_data(db, run, cluster)
    assert data.load_data(db, 1, 2)[0]["id"].to_list() == ["13:0"]
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
    db = ExperimentDatabase(path)
    db.create_tables()
    db.db.execute("""
        insert into cluster (id,common_name) values (1,'Parent'),(2,'A'),(3,'B');
        insert into pathway values (7,1,2,3),(8,1,3,2),(9,1,2,2),(10,1,3,3);
        insert into experiment_config (id) values (1);
        insert into experiment_run (id,experiment_config_id,simulation_mode)
        values (1,1,'PATHWAY_AT_A_TIME');
        insert into single_pathway_experiment_result (id,experiment_run_id,pathway_id)
        values (10,1,7),(11,1,8),(12,1,9);
    """)
    db.begin_execution(1, 1, 7, [7], result_id=10, recorded=True)
    db.begin_execution(1, 1, 8, [8], result_id=11, recorded=True)
    db.begin_execution(1, 1, 9, [9], result_id=12)
    publish_events(
        db,
        [
            recorded_event(10, 0, 0, "fragmentation", 1, 2, transition_index=0),
            recorded_event(11, 0, 0, "escape", 2, 5),
        ],
    )
    db.close()
    monkeypatch.setattr(data, "get_geometry", lambda *args: None)
    monkeypatch.setattr(
        data, "lengths_to_cumulative_lengths", lambda *args: (0, 1, 2, 3, 4, 5)
    )
    db = ExperimentDatabase(path, readonly=True)
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
