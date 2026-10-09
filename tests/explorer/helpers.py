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
