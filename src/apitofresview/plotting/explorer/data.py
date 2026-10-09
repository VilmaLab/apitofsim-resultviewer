"""Read-only cohort loading and numerical preprocessing for the Explorer.

Time coordinates use the recorder's raw ``postime.t``; there is no separate
realization start timestamp in the database.
"""

from collections import deque
from math import ceil, floor, sin, sqrt

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
SCHEMATIC_WEIGHTS = (3, 3, 1, 1, 1)


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


def coordinate_regions(regions, mode):
    if mode == "physical":
        return regions.with_columns(
            (pl.col("physical_left") * 1000).alias("left"),
            (pl.col("physical_right") * 1000).alias("right"),
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


def initial_slot(regions):
    start = regions["left"][0]
    lo, hi = escape_slot(regions)
    return start - (hi - lo), start


def position_events(events, regions, mode, own_zones=True):
    if mode in ("time", "physical"):
        position = pl.col("t") if mode == "time" else pl.col("z") * 1000
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
            pl.when(pl.col("type") == "init")
            .then(
                pl.lit(
                    sum(initial_slot(regions)) / 2 if own_zones else regions["left"][0]
                )
            )
            .when(pl.col("type") == "escape")
            .then(
                pl.lit(
                    sum(escape_slot(regions)) / 2 if own_zones else regions["right"][-1]
                )
            )
            .otherwise(position)
        )
    return events.with_columns(position.alias("position"))


def layout_events(events, mode):
    if mode == "realization":
        return events.with_columns(pl.col("realization_number").alias("plot_y"))
    if mode == "radial":
        return events.with_columns((pl.col("radial") * 1000).alias("plot_y"))
    if mode == "strip":
        return events.with_columns(
            (0.25 * (pl.col("id") * 12.9898).sin()).alias("plot_y")
        )
    return events.with_columns(pl.lit(0.0).alias("plot_y"))


def spread_terminal_x(
    events, regions, mode, start, end, width, height, y_bounds, diameter=6
):
    """Spread display coordinates only, using a fixed modulo period per plot."""
    scale = width / max(end - start, 1e-12)
    row_spacing = (
        height / max(abs(y_bounds[1] - y_bounds[0]), 1)
        if mode == "realization"
        else diameter
    )
    positions = events["position"].to_list()
    for kind, slot in (
        ("init", initial_slot(regions)),
        ("escape", escape_slot(regions)),
    ):
        lo, hi = slot
        padding = min(diameter / 2 / scale, (hi - lo) / 2)
        lo, hi = lo + padding, hi - padding
        capacity = max(1, floor((hi - lo) * scale / diameter) + 1)
        count = events.filter(pl.col("type") == kind).height
        period = min(capacity, max(count, 1), max(2, ceil(diameter / row_spacing)))
        for index, event in enumerate(events.iter_rows(named=True)):
            if event["type"] != kind:
                continue
            fraction = (
                (event["realization_number"] % period) / (period - 1)
                if mode == "realization" and period > 1
                else 0.5
                if mode == "realization"
                else (sin(event["id"] * 78.233) * 43758.5453) % 1
            )
            positions[index] = lo + fraction * (hi - lo)
    return events.with_columns(pl.Series("position", positions, dtype=pl.Float64))


def _stack_swarm(points, width, diameter):
    nearby = deque()
    result = {}
    for index, x in points:
        if x < -diameter or x > width + diameter:
            result[index] = 0.0
            continue
        while nearby and x - nearby[0][0] >= diameter:
            nearby.popleft()
        intervals = sorted(
            (
                y - sqrt(max(0, diameter**2 - (x - px) ** 2)),
                y + sqrt(max(0, diameter**2 - (x - px) ** 2)),
            )
            for px, y in nearby
        )
        # Find the nearest tangent endpoint of the forbidden interval at zero.
        merged = []
        for lo, hi in intervals:
            if merged and lo < merged[-1][1] - 1e-9:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        y = 0.0
        for lo, hi in merged:
            if lo + 1e-9 < 0 < hi - 1e-9:
                y = hi if hi <= -lo else lo
                break
        result[index] = y
        nearby.append((x, y))
    return result


def _terminal_swarm(events, points, regions, start, scale, width, diameter):
    terminal = dict(
        events.with_row_index()
        .filter(pl.col("type").is_in(["init", "escape"]))
        .select("index", "type")
        .iter_rows()
    )
    ordinary = [(i, x) for i, x in points if i not in terminal]
    ys = _stack_swarm(ordinary, width, diameter)
    xs = dict(points)
    ids = events["id"].to_list()
    extent = max((abs(y) for y in ys.values()), default=0)
    buckets = {}

    def add(x, y):
        buckets.setdefault((floor(x / diameter), floor(y / diameter)), []).append(
            (x, y)
        )

    def free(x, y):
        bx, by = floor(x / diameter), floor(y / diameter)
        return all(
            (x - px) ** 2 + (y - py) ** 2 >= diameter**2 * (1 - 1e-9)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for px, py in buckets.get((bx + dx, by + dy), ())
        )

    for i, x in ordinary:
        if -diameter <= x <= width + diameter:
            add(x, ys[i])
    for kind, slot in (
        ("init", initial_slot(regions)),
        ("escape", escape_slot(regions)),
    ):
        indices = sorted(
            (i for i in terminal if terminal[i] == kind), key=lambda i: ids[i]
        )
        lo, hi = ((x - start) * scale for x in slot)
        center = (lo + hi) / 2
        if hi < -diameter or lo > width + diameter:
            ys.update(dict.fromkeys(indices, 0.0))
            continue
        radius = min(diameter / 2, (hi - lo) / 2)
        columns = [center]
        for step in range(
            1, min(len(indices), floor(((hi - lo) / 2 - radius) / diameter)) + 1
        ):
            columns.extend((center - step * diameter, center + step * diameter))
        levels = [0.0]
        for step in range(1, floor(extent / diameter) + 1):
            levels.extend((step * diameter, -step * diameter))

        def candidates():
            for x in columns:
                for y in levels:
                    yield x, y
            step = len(levels) // 2 + 1
            while True:
                for sign in (1, -1):
                    for x in columns:
                        yield x, sign * step * diameter
                step += 1

        slots = candidates()
        for i in indices:
            x, y = next(slots)
            while not free(x, y):
                x, y = next(slots)
            xs[i], ys[i] = x, y
            add(x, y)
    return xs, ys


def pack_beeswarm(
    events, start, end, width, height, diameter=10, terminal_regions=None
):
    """Pack tangent circles in screen coordinates, shrinking to fit vertically."""
    scale = width / max(end - start, 1e-12)
    if (
        terminal_regions is not None
        and not events.filter(pl.col("type").is_in(["init", "escape"])).is_empty()
    ):
        slot_width = (
            initial_slot(terminal_regions)[1] - initial_slot(terminal_regions)[0]
        )
        diameter = min(diameter, slot_width * scale)
    ordered = events.with_row_index().sort("position", "id")
    points = [
        (index, (x - start) * scale)
        for index, x in ordered.select("index", "position").iter_rows()
    ]
    lower, upper = 0.0, diameter
    best = None
    for _ in range(16):
        if terminal_regions is None:
            xs, ys = dict(points), _stack_swarm(points, width, diameter)
        else:
            xs, ys = _terminal_swarm(
                events, points, terminal_regions, start, scale, width, diameter
            )
        result = [ys[i] for i in range(events.height)]
        extent = max((abs(y) for y in result), default=0) * 2 + diameter
        if extent <= height:
            best = (result, xs, diameter)
            lower = diameter
        else:
            upper = diameter
        if upper - lower <= max(upper * 0.01, 1e-6):
            break
        diameter = (lower + upper) / 2 if lower else diameter * height / extent * 0.95
    result, xs, diameter = best
    return events.with_columns(
        pl.Series("plot_y", result, dtype=pl.Float64),
        *(
            [
                pl.Series(
                    "position",
                    [xs[i] / scale + start for i in range(events.height)],
                    dtype=pl.Float64,
                )
            ]
            if terminal_regions is not None
            else []
        ),
    ), diameter


def beeswarm_envelope(events, x_padding, y_padding):
    """Outline the packed circles using their radii in each coordinate."""
    if events.is_empty():
        return pl.DataFrame(schema={"position": pl.Float64, "plot_y": pl.Float64})
    points = sorted(events.select("position", "plot_y").iter_rows())
    samples = sorted(
        {
            x + offset * x_padding
            for x, _ in points
            for offset in (-1, -sqrt(3) / 2, -0.5, 0, 0.5, sqrt(3) / 2, 1)
        }
    )
    nearby = deque()
    cursor = 0
    upper, lower = [], []
    for x in samples:
        while cursor < len(points) and points[cursor][0] <= x + x_padding:
            nearby.append(points[cursor])
            cursor += 1
        while nearby and nearby[0][0] < x - x_padding:
            nearby.popleft()
        edges = [
            (y, y_padding * sqrt(max(0, 1 - ((x - px) / x_padding) ** 2)))
            for px, y in nearby
        ]
        upper.append(max((y + radius for y, radius in edges), default=0.0))
        lower.append(min((y - radius for y, radius in edges), default=0.0))
    return pl.DataFrame(
        {"position": samples + samples[::-1], "plot_y": upper + lower[::-1]}
    )


def aggregate(realizations, events, regions):
    """CDF and region fractions use every selected realization as denominator."""
    n = realizations.height
    terminals = realizations.join(
        events, left_on="terminal_event_id", right_on="id", how="inner"
    )
    fragmented = terminals.filter(pl.col("type") == "fragmentation")
    escaped = terminals.filter(pl.col("type") == "escape").height
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
