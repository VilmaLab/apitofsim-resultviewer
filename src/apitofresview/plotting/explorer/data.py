"""Read-only cohort loading and numerical preprocessing for the Explorer.

Time coordinates use the recorder's raw ``postime.t``; there is no separate
realization start timestamp in the database.
"""

import polars as pl

from apitofsim.plotting.events import get_geometry, lengths_to_cumulative_lengths


EVENT_TYPES = ("collision", "fragmentation", "escape")
REGIONS = (
    "First chamber",
    "Skimmer",
    "Skimmer to quadrupole",
    "Quadrupole",
    "Second chamber",
)
SCHEMATIC_WEIGHTS = (2, 1, 1, 3, 2)


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
        with events as (
            select id, realization_id, 'collision' as type, postime,
                   null::integer as pathway_id from collision_event
            union all
            select id, realization_id, 'fragmentation', postime, pathway_id
            from fragmentation_event
            union all
            select id, realization_id, 'escape', postime, null::integer
            from escape_event
        )
        select e.id, e.realization_id, e.type,
               e.postime.t::double as t, e.postime.x::double as x,
               e.postime.y::double as y,
               greatest(e.postime.z::double, 0.0) as z,
               e.pathway_id, e.postime.z < 0 as z_clamped
        from events e join realizations r on r.id = e.realization_id
        order by e.realization_id, e.postime.t, e.id
        """
    ).pl()
    pathways = connection.execute(
        """
        select p.id as pathway_id, a.common_name || ' + ' || b.common_name as pathway
        from pathway p
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
    return classify(realizations, events), events, pathways, regions


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


def classify(realizations, events):
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
        .with_columns(
            pl.when(pl.col("terminal_count") > 1)
            .then(pl.lit("Ambiguous"))
            .when(valid & (pl.col("last_type") == "escape"))
            .then(pl.lit("Escaped"))
            .when(valid)
            .then(pl.concat_str(pl.lit("Pathway "), pl.col("last_pathway")))
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


def coordinate_regions(regions, mode):
    if mode == "physical":
        return regions.with_columns(
            pl.col("physical_left").alias("left"),
            pl.col("physical_right").alias("right"),
        )
    widths = SCHEMATIC_WEIGHTS if mode == "schematic" else (1,) * regions.height
    return (
        regions.with_columns(pl.Series("width", widths, dtype=pl.Float64))
        .with_columns(
            (pl.col("width").cum_sum() - pl.col("width")).alias("left"),
            pl.col("width").cum_sum().alias("right"),
        )
        .drop("width")
    )


def escape_slot(regions):
    end = regions["right"][-1]
    return end, end + max((end - regions["left"][-1]) * 0.65, 0.5)


def position_events(events, regions, mode):
    if mode in ("time", "physical"):
        position = pl.col("t" if mode == "time" else "z")
    else:
        # Mapping uses the first inclusive upper boundary, with extrapolation
        # past the final region, matching the original spatial display.
        position = pl.lit(0.0)
        for index, _, lo, hi, left, right in reversed(list(regions.iter_rows())):
            mapped = (
                pl.lit(left)
                if hi == lo
                else left + (pl.col("z") - lo) * (right - left) / (hi - lo)
            )
            position = (
                pl.when((pl.col("z") <= hi) | (index == regions.height - 1))
                .then(mapped)
                .otherwise(position)
            )
        position = (
            pl.when(pl.col("type") == "escape")
            .then(pl.lit(sum(escape_slot(regions)) / 2))
            .otherwise(position)
        )
    return events.with_columns(position.alias("position"))


def layout_events(events, mode):
    if mode == "radial":
        return events.with_columns(pl.col("radial").alias("plot_y"))
    if mode == "strip":
        return events.with_columns(
            (0.25 * (pl.col("id") * 12.9898).sin()).alias("plot_y")
        )
    # Stable greedy packing is sequential; keep its result aligned with the frame.
    span = (events["position"].max() or 0) - (events["position"].min() or 0)
    radius = max(span / 160, 1e-9)
    packed = {}
    result = [0.0] * events.height
    ordered = events.with_row_index().sort("position", "id")
    for index, event_id, x in ordered.select("index", "id", "position").iter_rows():
        nearby = [(px, py) for px, py in packed.values() if abs(px - x) < 2 * radius]
        level = next(
            (
                v
                for v in (0, 1, -1, 2, -2, 3, -3)
                if all(
                    (x - px) ** 2 / radius**2 + (v - py) ** 2 >= 1 for px, py in nearby
                )
            ),
            len(nearby) + 1,
        )
        result[index] = float(level) * 0.18
        packed[event_id] = (x, level)
    return events.with_columns(pl.Series("plot_y", result, dtype=pl.Float64))


def violin(events):
    continuous = events.filter(pl.col("type") != "escape").select("position")
    lo, hi = continuous["position"].min(), continuous["position"].max()
    if continuous.height < 2 or lo == hi:
        return pl.DataFrame(schema={"position": pl.Float64, "plot_y": pl.Float64})
    bandwidth = max((hi - lo) / 18, 1e-12)
    grid = pl.DataFrame({"position": [lo + (hi - lo) * i / 79 for i in range(80)]})
    density = (
        grid.join(continuous.rename({"position": "sample"}), how="cross")
        .group_by("position")
        .agg(
            (-0.5 * ((pl.col("position") - pl.col("sample")) / bandwidth).pow(2))
            .exp()
            .sum()
            .alias("density")
        )
        .sort("position")
        .with_columns(
            (0.4 * pl.col("density") / pl.col("density").max()).alias("plot_y")
        )
        .select("position", "plot_y")
    )
    return pl.concat([density, density.reverse().with_columns(-pl.col("plot_y"))])


def aggregate(realizations, events, regions):
    """CDF and region fractions use every selected realization as denominator."""
    n = realizations.height
    terminals = realizations.join(
        events, left_on="terminal_event_id", right_on="id", how="inner"
    )
    fragmented = terminals.filter(pl.col("type") == "fragmentation")
    escaped = realizations.filter(pl.col("fate") == "Escaped").height
    unresolved = n - fragmented.height - escaped
    counts = (
        fragmented.group_by("position")
        .len()
        .sort("position")
        .with_columns(pl.col("len").cum_sum().alias("cumulative"))
    )
    # Two points per coordinate preserve vertical jumps, including tied events.
    cdf = (
        counts.with_columns(
            pl.concat_list(
                (pl.col("cumulative") - pl.col("len")) / max(n, 1),
                pl.col("cumulative") / max(n, 1),
            ).alias("fraction")
        )
        .select("position", "fraction")
        .explode("fraction")
    )
    regional = fragmented.group_by("region").len()
    bars = (
        regions.select("region", "name", "left", "right")
        .join(regional, on="region", how="left", maintain_order="left")
        .with_columns((pl.col("len").fill_null(0) / max(n, 1)).alias("fraction"))
        .drop("len")
    )
    lo, hi = escape_slot(regions)
    bars = pl.concat(
        [
            bars,
            pl.DataFrame(
                {
                    "region": [regions.height],
                    "name": ["Escaped"],
                    "left": [lo],
                    "right": [hi],
                    "fraction": [escaped / max(n, 1)],
                }
            ),
        ],
        how="vertical_relaxed",
    )
    return cdf, bars, escaped, unresolved
