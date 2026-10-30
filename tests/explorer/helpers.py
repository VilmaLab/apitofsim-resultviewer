"""Small cohort builders shared by Explorer tests."""

import polars as pl

from apitofresview.plotting.explorer import data

EVENT_SCHEMA = {
    "id": pl.Int64,
    "type": pl.String,
    "realization_id": pl.Int64,
    "t": pl.Float64,
    "x": pl.Float64,
    "y": pl.Float64,
    "z": pl.Float64,
    "pathway_id": pl.Int64,
    "z_clamped": pl.Boolean,
}
BOUNDARIES = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)


def event(id, kind, rid, t, z, pathway=None, clamped=False):
    return (id, kind, rid, t, 3.0, 4.0, z, pathway, clamped)


def cohort(ids, *events, boundaries=BOUNDARIES):
    regions = data.make_regions(boundaries)
    events = data.preprocess_events(
        pl.DataFrame(events, schema=EVENT_SCHEMA, orient="row"), regions
    )
    realizations = pl.DataFrame(
        {"id": ids, "experiment_result_id": [10] * len(ids)},
        schema={"id": pl.Int64, "experiment_result_id": pl.Int64},
    )
    pathways = pl.DataFrame({"pathway_id": [7], "pathway": ["Parent → A + B"]})
    return (
        data.classify(realizations, events, pathways, "Parent"),
        events,
        pathways,
        regions,
    )


def outcome_cohort():
    return cohort(
        [1, 2, 3, 4, 5],
        event(1, "collision", 1, 0.1, 0),
        event(2, "fragmentation", 1, 2, 2, 7),
        event(3, "fragmentation", 2, 3, 2, 7),
        event(4, "escape", 3, 4, 5),
        event(5, "collision", 4, 1, 1),
        event(6, "fragmentation", 5, 2, 2, 7),
        event(7, "escape", 5, 3, 5),
    )


def collision_cohort():
    members, events, pathways, regions = cohort(
        [1],
        event(1, "collision", 1, 1, 1),
        event(2, "postcollision", 1, 1, 1),
        event(3, "collision", 1, 2, 2),
        event(4, "collision", 1, 3, 3),
        event(5, "postcollision", 1, 4, 4),  # Unmatched.
        event(6, "escape", 1, 5, 5),
    )
    events = data.pair_collisions(
        events.with_columns(
            pl.Series(
                "accepted", [True, None, False, None, None, None], dtype=pl.Boolean
            ),
            pl.Series("theta", [0.5, None, 0.7, None, None, None]),
            pl.Series("u_norm", [-100.0, None, -200.0, None, None, None]),
            pl.Series("rot_energy", [1e-20, 2e-20, None, None, None, None]),
            pl.Series("vibrational_energy", [1e-20, 1e-20, None, None, None, None]),
            pl.Series("velocity", [dict(x=3.0, y=4.0, z=v) for v in range(1, 7)]),
            pl.Series("omega", [dict(x=0.0, y=0.0, z=v) for v in range(1, 7)]),
        )
    )
    return members, events, pathways, regions


def pathway_cohort():
    """Independent cohorts: mixed outcomes, all escaped, and no recordings."""
    members, events, _, regions = cohort(
        [100, 200, 300, 400, 500],
        event(1, "fragmentation", 100, 1, 2, 7),
        event(2, "escape", 200, 2, 5),
        event(3, "collision", 300, 1, 1),
        event(4, "fragmentation", 400, 1, 2, 7),
        event(5, "escape", 400, 2, 5),
        event(6, "escape", 500, 2, 5),
    )
    pathways = pl.DataFrame(
        {
            "pathway_id": [7, 8, 9, 10],
            "pathway": [
                "Parent → A + B",
                "Parent → C + D",
                "Parent → E + F",
                "Parent → G + H",
            ],
            "simulated": [True, True, True, False],
        }
    )
    members = members.with_columns(
        pl.Series("simulated_pathway_id", [7, 7, 7, 7, 8]),
        pl.Series("experiment_result_id", [10, 10, 10, 10, 11]),
    ).join(
        pathways.select("pathway_id", "pathway").rename(
            {"pathway_id": "simulated_pathway_id", "pathway": "simulated_pathway"}
        ),
        on="simulated_pathway_id",
        how="left",
        maintain_order="left",
    )
    return members, events, pathways, regions
