from types import SimpleNamespace

import duckdb

from apitofresview.webapp import get_experiment_overview


def overview_database():
    connection = duckdb.connect(":memory:")
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
    return SimpleNamespace(db=connection)


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
