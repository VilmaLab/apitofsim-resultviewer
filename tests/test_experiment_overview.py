from apitofsim.workflow.db import ExperimentDatabase

from apitofresview.webapp import (
    get_cluster_choices,
    get_experiment_choices,
    get_experiment_overview,
    get_is_single_pathway,
)


def overview_database():
    db = ExperimentDatabase(":memory:")
    connection = db.db
    connection.execute(
        """
        create table experiment_config (id integer, name varchar, config json);
        create table experiment_run (
            id integer,
            experiment_config_id integer,
            run_config json,
            start_time timestamp
        );
        create table experiment_summary (
            experiment_run_id integer,
            config_name varchar,
            start_time timestamp,
            successes bigint,
            failures bigint,
            is_single_pathway boolean
        );

        insert into experiment_config values
            (1, 'Overview test', '{"temperature": 300}');
        insert into experiment_run values
            (7, 1, '{"simulation_mode": "SINGLE_CLUSTER"}',
             '2026-01-02 03:04:05');
        insert into experiment_summary values
            (7, 'Overview test', '2026-01-02 03:04:05', 12, 3, true);
        """
    )
    return db


def test_get_experiment_overview_returns_formatted_summary_and_both_configs():
    db = overview_database()

    overview = get_experiment_overview(db, 7)

    assert overview == {
        "experiment_run_id": 7,
        "config_name": "Overview test",
        "start_time": "2 Jan 2026, 03:04:05",
        "successes": 12,
        "failures": 3,
        "pathway_mode": "Single pathway",
        "configuration": {
            "experiment_config": {"temperature": 300},
            "run_config": {"simulation_mode": "SINGLE_CLUSTER"},
        },
    }


def test_get_experiment_overview_returns_none_for_an_unknown_run():
    db = overview_database()

    assert get_experiment_overview(db, 999) is None


def test_metadata_choices_and_modes_are_scoped_and_readonly(browser_database):
    db = ExperimentDatabase(browser_database, readonly=True)
    try:
        assert get_experiment_choices(db) == [
            (
                "#1 Browser test run at 2026-01-02 03:04:05 (single pathway, success rate: 1/1)",
                1,
            ),
            (
                "#2 Other run run at 2026-01-03 03:04:05 (single pathway, success rate: 1/1)",
                2,
            ),
        ]
        assert get_cluster_choices(db, 1) == [("Parent (100.0 amu)", 1)]
        assert get_cluster_choices(db, 2) == [("Parent (100.0 amu)", 1)]
        assert get_is_single_pathway(db, 1, 1) is True
        for run in (None, 999):
            assert get_cluster_choices(db, run) == []
            assert get_is_single_pathway(db, run, 1) is None
        for cluster in (None, 999):
            assert get_is_single_pathway(db, 1, cluster) is None
    finally:
        db.close()


def test_multi_pathway_metadata_keeps_display_sorting_local():
    db = ExperimentDatabase(":memory:")
    db.create_tables()
    try:
        db.db.execute("""
            insert into cluster (id, common_name, atomic_mass) values
            (1,'Heavy',100),(2,'Light Z',40),(3,'Light A',40);
            insert into pathway values (1,1,2,3),(2,2,1,3),(3,3,1,2);
            insert into experiment_config (id, name, config) values (1,'Multi','{}');
            insert into experiment_run
            (id, experiment_config_id, simulation_mode, run_config, start_time) values
            (1,1,'SINGLE_CLUSTER','{}','2026-01-02 03:04:05');
            insert into multi_pathway_experiment_result (id,experiment_run_id,cluster_id)
            values (1,1,1),(2,1,2),(3,1,3);
            insert into pathway_fragmentation values (1,1,1,2),(2,2,2,2),(3,3,3,2);
        """)
        assert get_cluster_choices(db, 1) == [
            ("Light A (40.0 amu)", 3),
            ("Light Z (40.0 amu)", 2),
            ("Heavy (100.0 amu)", 1),
        ]
        assert get_is_single_pathway(db, 1, 1) is False
        assert get_experiment_overview(db, 1)["pathway_mode"] == "Multi-pathway"
    finally:
        db.close()
