"""Read-only cohort loading and numerical preprocessing for the Explorer.

Time coordinates use the recorder's raw ``postime.t``; there is no separate
realization start timestamp in the database.
"""

import duckdb
import polars as pl
from apitofsim.plotting.events import (
    NoRecordedEventsError,
    get_geometry,
    lengths_to_cumulative_lengths,
)

from apitofresview.modes import simulation_mode
from apitofresview.recordings import NO_RECORDED_EVENTS

from .tree import ExecutionTree

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


SUMMARY_COLUMNS = ("collisions", "rejected_collisions", "clamped_events", "max_radial")


def _event_frame(
    db, experiment, cluster, tree_mode, *, summary=False, members=None, ids=None
):
    query = (
        db.tree_events_query(run_id=experiment, parent=cluster)
        if tree_mode
        else db.events_query(run_id=experiment, parent=cluster)
    )
    if members is not None:
        query = query.filter(
            duckdb.ColumnExpression("realization_id").isin(
                *(duckdb.ConstantExpression(rid) for rid in members)
            )
        )
    if summary:
        partition = "partition by realization_id"
        history = f"{partition} order by event_index rows unbounded preceding"
        # Prefix maxima preserve population timing when hidden collisions move back.
        # Keep the final event so collision-only/interrupted histories remain visible.
        validity = (
            f", bool_and(provenance = 'persisted' and coalesce(continuity_valid, false)) over ({partition}) as history_valid"
            if tree_mode
            else ""
        )
        query = query.select(f"""
            * exclude (postime, velocity, omega, rot_energy, vibrational_energy, theta, u_norm),
            count(*) filter (where event_type = 'collision') over ({partition}) as collisions,
            count(*) filter (where event_type = 'collision' and accepted = false) over ({partition}) as rejected_collisions,
            count(*) filter (where z < 0) over ({partition}) as clamped_events,
            max(sqrt(x*x + y*y)) over ({partition}) as max_radial,
            max(event_index) over ({partition}) as last_index,
            max(z) over ({history}) as population_z,
            max(t) over ({history}) as population_t
            {validity}
        """).filter(
            "event_type not in ('collision', 'postcollision') or event_index = last_index"
        )
    if ids is not None:
        query = query.filter(
            duckdb.ColumnExpression("event_id").isin(
                *(duckdb.ConstantExpression(id) for id in ids)
            )
        )
    # One sort, after reducing to the requested rows and columns. Tree queries
    # already guarantee numeric event order; filtering does not change it.
    events = (
        query.pl()
        .rename({"event_id": "id", "event_type": "type"})
        .with_columns(
            pl.col("type").cast(pl.String),
            pl.col("t", "x", "y", "z").cast(pl.Float64),
            (pl.col("z") < 0).alias("z_clamped"),
        )
        .with_columns(pl.col("z").clip(lower_bound=0))
    )
    if tree_mode:
        if summary:
            events = events.with_columns(
                pl.col("history_valid").alias("continuity_valid")
            )
        names = dict(db.db.execute("select id, common_name from cluster").fetchall())
        events = events.with_columns(
            pl.col("current_cluster_id")
            .replace_strict(names, default=None, return_dtype=pl.String)
            .alias("current_cluster_name")
        )
    return events


def load_overview(db, experiment, cluster):
    """Load sparse cohort events and per-realization totals, without motion fields."""
    return load_data(db, experiment, cluster, summary=True)


def load_histories(db, experiment, cluster, regions, members, *, tree=False, ids=None):
    """Materialize only requested histories (or individual candidate event rows)."""
    return preprocess_events(
        _event_frame(db, experiment, cluster, tree, members=members, ids=ids),
        regions,
        sorted_events=tree,
    )


def load_data(db, experiment: int, cluster: int, *, summary=False):
    """Return realization, event, pathway and region frames for one cohort."""
    mode = simulation_mode(db, experiment)
    pathway_mode = mode == "PATHWAY_AT_A_TIME"
    tree_mode = mode == "CLUSTER_TREE"
    events = _event_frame(db, experiment, cluster, tree_mode, summary=summary)
    if events.is_empty():
        raise NoRecordedEventsError(NO_RECORDED_EVENTS)
    realizations = (
        events.select(
            pl.col("realization_id").alias("id"),
            "experiment_result_id",
            "realization_index",
        )
        .unique(maintain_order=True)
        .sort("experiment_result_id", "realization_index")
        .drop("realization_index")
    )
    if pathway_mode:
        realizations = realizations.join(
            db.results_query(run_id=experiment, parent=cluster)
            .select("experiment_result_id, pathway_id as simulated_pathway_id")
            .pl(),
            on="experiment_result_id",
            how="left",
            maintain_order="left",
        )
    pathway_query = (
        db.db.sql(
            """select p.* from pathway_report p where p.cluster_id in (
            select n.cluster_id from cluster_tree_node n
            join cluster_tree_experiment_result r on r.id = n.experiment_result_id
            where r.experiment_run_id = ? and r.cluster_id = ?
        )""",
            params=[experiment, cluster],
        )
        if tree_mode
        else db.db.table("pathway_report").filter(
            duckdb.ColumnExpression("cluster_id") == duckdb.ConstantExpression(cluster)
        )
    )
    pathways = pathway_query.pl().select(
        "pathway_id",
        pl.concat_list(
            [
                pl.when(pl.col(f"product{i}_charge") != 0).then(
                    pl.struct(
                        pl.col(f"product{i}_id").alias("id"),
                        pl.col(f"product{i}_common_name").alias("name"),
                    )
                )
                for i in (1, 2)
            ]
        )
        .list.drop_nulls()
        .alias("charged_products"),
        pl.concat_str(
            pl.when(pl.col("product1_charge") != 0).then(
                pl.col("product1_common_name")
            ),
            pl.when(pl.col("product2_charge") != 0).then(
                pl.col("product2_common_name")
            ),
            separator=" + ",
            ignore_nulls=True,
        )
        .replace("", "Unknown charged daughter")
        .alias("daughter"),
        pl.concat_str(
            "cluster_common_name",
            pl.lit(" → "),
            "product1_common_name",
            pl.lit(" + "),
            "product2_common_name",
        ).alias("pathway"),
    )
    if pathway_mode:
        recorded = db.db.execute(
            "select distinct pathway_id from experiment_result "
            "where experiment_run_id = ? and cluster_id = ?",
            [experiment, cluster],
        ).pl()
        pathways = (
            pathways.join(
                recorded.with_columns(pl.lit(True).alias("simulated")),
                on="pathway_id",
                how="left",
            )
            .with_columns(pl.col("simulated").fill_null(False))
            .sort("pathway_id")
        )
    regions = make_regions(
        tuple(
            float(v)
            for v in lengths_to_cumulative_lengths(get_geometry(db, experiment))
        )
    )
    events = preprocess_events(events, regions, sorted_events=tree_mode and not summary)
    cluster_name, cluster_charge = (
        db.clusters_query()
        .filter(duckdb.ColumnExpression("id") == duckdb.ConstantExpression(cluster))
        .select("common_name, charge")
        .fetchone()
    )
    if tree_mode:
        tree = ExecutionTree.load(db, experiment, cluster, pathways)
        classified = tree.classify(realizations, events)
    else:
        classified = classify(realizations, events, pathways, cluster_name)
    if pathway_mode:
        classified = classified.join(
            realizations.select("id", "simulated_pathway_id"),
            on="id",
            how="left",
            maintain_order="left",
        ).join(
            pathways.select("pathway_id", "pathway").rename(
                {
                    "pathway_id": "simulated_pathway_id",
                    "pathway": "simulated_pathway",
                }
            ),
            on="simulated_pathway_id",
            how="left",
            maintain_order="left",
        )
    if summary:
        classified = classified.join(
            events.group_by("realization_id").agg(pl.col(*SUMMARY_COLUMNS).first()),
            left_on="id",
            right_on="realization_id",
            how="left",
            maintain_order="left",
        )
    return (
        classified.with_columns(
            pl.lit(cluster_name).alias("initial_species"),
            pl.lit(cluster_charge not in (0, None)).alias("initial_charged"),
        ),
        events,
        pathways,
        regions,
    )


def _pair_states(events, kind, valid):
    paired = (
        (pl.col("type") == f"post{kind}") & (pl.col("type").shift() == kind) & valid
    )
    for key in ("realization_id", "t", "x", "y", "z"):
        paired &= pl.col(key) == pl.col(key).shift()
    return events.with_columns(
        pl.when(paired).then(pl.col("id").shift()).alias(f"{kind}_id")
    )


def pair_collisions(events):
    """Link only adjacent accepted before/after states at the same spacetime point."""
    accepted = pl.col("accepted") if "accepted" in events.columns else pl.lit(False)
    return _pair_states(events, "collision", accepted.shift().fill_null(False))


def pair_fragmentations(events):
    """Pair adjacent continuing splits using persisted destination and spacetime."""
    valid = pl.lit(False)
    if "provenance" in events.columns:
        valid = (
            (pl.col("next_particle_index").shift() == pl.col("particle_index"))
            & (pl.col("particle_index") >= 0)
            & (pl.col("provenance") == "persisted")
            & (pl.col("provenance").shift() == "persisted")
            & pl.col("continuity_valid").fill_null(False)
        )
    return _pair_states(events, "fragmentation", valid)


def preprocess_events(events, regions, *, sorted_events=False):
    """Assign half-open regions, including the machine's final endpoint."""
    region = pl.lit(None, dtype=pl.Int64)
    last = regions.height - 1
    for index, _, lo, hi in reversed(list(regions.iter_rows())):
        inside = (pl.col("z") >= lo) & (
            (pl.col("z") < hi) | ((index == last) & (pl.col("z") == hi))
        )
        region = pl.when(inside).then(pl.lit(index)).otherwise(region)
    if not sorted_events:
        events = events.sort("result_id", "realization_index", "event_index")
    events = events.with_row_index("event_number", offset=1)
    return pair_fragmentations(pair_collisions(events)).with_columns(
        region.alias("region"),
        (pl.col("x").pow(2) + pl.col("y").pow(2)).sqrt().alias("radial"),
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
            histories,
            left_on="id",
            right_on="realization_id",
            how="left",
            maintain_order="left",
        )
        .join(
            pathways,
            left_on="last_pathway",
            right_on="pathway_id",
            how="left",
            maintain_order="left",
        )
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
