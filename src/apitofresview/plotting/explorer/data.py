"""Read-only cohort loading and numerical preprocessing for the Explorer.

Time coordinates use the recorder's raw ``postime.t``; there is no separate
realization start timestamp in the database.
"""

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
    connection = db.db
    realizations = connection.execute(
        """
        with results as (
            select id from multi_pathway_experiment_result
            where experiment_run_id = ? and cluster_id = ?
            union
            select s.id from single_pathway_experiment_result s
            join pathway p on p.id = s.pathway_id
            where s.experiment_run_id = ? and p.cluster_id = ?
        )
        select r.id, r.experiment_result_id from realization r
        join results on results.id = r.experiment_result_id
        order by r.id
        """,
        (experiment, cluster, experiment, cluster),
    ).pl()
    # DuckDB replacement scans reference the local Polars frame directly.
    events = connection.execute(
        """
        select e.id, e.realization_id, e.event_type::varchar as type,
               e.postime.t::double as t, e.postime.x::double as x,
               e.postime.y::double as y,
               greatest(e.postime.z::double, 0.0) as z,
               f.pathway_id, e.postime.z < 0 as z_clamped,
               e.velocity, e.omega, e.rot_energy, e.vibrational_energy,
               e.particle_index, c.theta, c.u_norm, c.accepted
        from event_info e join realizations r on r.id = e.realization_id
        left join collision_event c on c.id = e.id
        left join fragmentation_event f on f.id = e.id
        order by e.realization_id, e.postime.t, e.id
        """
    ).pl()
    pathways = connection.execute(
        """
        select p.id as pathway_id, c.common_name || ' → ' ||
               a.common_name || ' + ' || b.common_name as pathway
        from pathway p
        join cluster c on c.id = p.cluster_id
        join cluster a on a.id = p.product1_id
        join cluster b on b.id = p.product2_id
        where p.cluster_id = ?
        """,
        (cluster,),
    ).pl()
    regions = make_regions(
        tuple(
            float(v)
            for v in lengths_to_cumulative_lengths(get_geometry(db, experiment))
        )
    )
    events = preprocess_events(events, regions)
    cluster_name = connection.execute(
        "select common_name from cluster where id = ?", (cluster,)
    ).fetchone()[0]
    return (
        classify(realizations, events, pathways, cluster_name),
        events,
        pathways,
        regions,
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
    return events.with_columns(
        region.alias("region"),
        (pl.col("x").pow(2) + pl.col("y").pow(2)).sqrt().alias("radial"),
    ).sort("realization_id", "t", "id")


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
