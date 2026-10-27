"""Read-only cohort loading and numerical preprocessing for the Explorer.

Time coordinates use the recorder's raw ``postime.t``; there is no separate
realization start timestamp in the database.
"""

import duckdb
import polars as pl
from apitofsim.plotting.events import get_geometry, lengths_to_cumulative_lengths

EVENT_TYPES = ("init", "collision", "fragmentation", "escape")
REGIONS = (
    "First chamber",
    "Skimmer",
    "Second chamber (before quadrupole)",
    "Quadrupole",
    "Second chamber (after quadrupole)",
)


def make_regions(boundaries):
    return pl.DataFrame(
        {
            "region": range(len(boundaries) - 1),
            "name": REGIONS,
            "physical_left": boundaries[:-1],
            "physical_right": boundaries[1:],
        },
        schema_overrides={"physical_left": pl.Float64, "physical_right": pl.Float64},
    )


def load_data(db, experiment: int, cluster: int):
    """Return realization, event, pathway and region frames for one cohort."""
    realizations = (
        db.realizations_query(run_id=experiment, parent=cluster)
        .select("realization_id as id, experiment_result_id")
        .order("id")
        .pl()
    )
    events = (
        db.events_query(run_id=experiment, parent=cluster, sort=True)
        .pl()
        .rename({"event_id": "id", "event_type": "type"})
        .with_columns(
            pl.col("type").cast(pl.String),
            pl.col("t", "x", "y", "z").cast(pl.Float64),
            (pl.col("z") < 0).alias("z_clamped"),
        )
        .with_columns(pl.col("z").clip(lower_bound=0))
    )
    pathways = (
        db.db.table("pathway_report")
        .filter(
            duckdb.ColumnExpression("cluster_id") == duckdb.ConstantExpression(cluster)
        )
        .pl()
        .select(
            "pathway_id",
            pl.concat_str(
                "cluster_common_name",
                pl.lit(" → "),
                "product1_common_name",
                pl.lit(" + "),
                "product2_common_name",
            ).alias("pathway"),
        )
    )
    regions = make_regions(
        tuple(
            float(v)
            for v in lengths_to_cumulative_lengths(get_geometry(db, experiment))
        )
    )
    events = preprocess_events(events, regions)
    cluster_name = (
        db.clusters_query()
        .filter(duckdb.ColumnExpression("id") == duckdb.ConstantExpression(cluster))
        .select("common_name")
        .fetchone()[0]
    )
    return (
        classify(realizations, events, pathways, cluster_name),
        events,
        pathways,
        regions,
    )


def pair_collisions(events):
    """Link only adjacent accepted before/after states at the same spacetime point."""
    events = events.sort("realization_id", "t", "id")
    accepted = pl.col("accepted") if "accepted" in events.columns else pl.lit(False)
    paired = (
        (pl.col("type") == "postcollision")
        & (pl.col("type").shift() == "collision")
        & accepted.shift().fill_null(False)
    )
    for key in ("realization_id", "t", "x", "y", "z"):
        paired &= pl.col(key) == pl.col(key).shift()
    return events.with_columns(
        pl.when(paired).then(pl.col("id").shift()).alias("collision_id")
    )


def preprocess_events(events, regions):
    """Assign half-open regions, including the machine's final endpoint."""
    region = pl.lit(None, dtype=pl.Int64)
    last = regions.height - 1
    for index, _, lo, hi in reversed(list(regions.iter_rows())):
        inside = (pl.col("z") >= lo) & (
            (pl.col("z") < hi) | ((index == last) & (pl.col("z") == hi))
        )
        region = pl.when(inside).then(pl.lit(index)).otherwise(region)
    return (
        pair_collisions(events)
        .with_columns(
            region.alias("region"),
            (pl.col("x").pow(2) + pl.col("y").pow(2)).sqrt().alias("radial"),
        )
        .sort("realization_id", "t", "id")
    )


def classify(realizations, events, pathways, cluster_name):
    """Accept exactly one terminal event, which must also be the last event."""
    terminal = pl.col("type").is_in(["fragmentation", "escape"])
    histories = events.group_by("realization_id").agg(
        terminal.sum().alias("terminal_count"),
        pl.col("id").last().alias("last_id"),
        pl.col("type").last().alias("last_type"),
        pl.col("pathway_id").last().alias("last_pathway"),
        pl.col("region").last().alias("last_region"),
    )
    valid = (pl.col("terminal_count") == 1) & (
        (pl.col("last_type") == "escape")
        | (
            (pl.col("last_type") == "fragmentation")
            & pl.col("last_pathway").is_not_null()
            & pl.col("last_region").is_not_null()
        )
    )
    return (
        realizations.join(
            histories, left_on="id", right_on="realization_id", how="left"
        )
        .join(pathways, left_on="last_pathway", right_on="pathway_id", how="left")
        .with_columns(
            pl.when(pl.col("terminal_count") > 1)
            .then(pl.lit("Ambiguous"))
            .when(valid & (pl.col("last_type") == "escape"))
            .then(pl.lit(f"{cluster_name} → {cluster_name}"))
            .when(valid)
            .then(pl.col("pathway").fill_null("Unknown pathway"))
            .otherwise(pl.lit("Incomplete"))
            .alias("fate"),
            pl.when(valid).then(pl.col("last_id")).alias("terminal_event_id"),
        )
        .select("id", "experiment_result_id", "fate", "terminal_event_id")
        .sort("id")
    )


def select_realizations(realizations, bounds, fates):
    lo, hi = (int(v) for v in bounds)
    return realizations.slice(lo - 1, hi - lo + 1).filter(pl.col("fate").is_in(fates))


def events_for(events, realizations):
    return events.join(
        realizations.select(pl.col("id").alias("realization_id")),
        on="realization_id",
        how="semi",
        maintain_order="left",
    )
