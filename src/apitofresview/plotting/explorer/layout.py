"""Display coordinates and screen-space packing for the Explorer."""

from collections import deque
from math import ceil, floor, sin, sqrt

import polars as pl

SCHEMATIC_WEIGHTS = (3, 3, 1, 1, 1)


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
