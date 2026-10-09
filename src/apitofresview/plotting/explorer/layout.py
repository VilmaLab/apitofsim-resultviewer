"""Display coordinates and screen-space packing for the Explorer."""

from dataclasses import dataclass
from math import ceil, floor, sin, sqrt

import numpy as np
import polars as pl

from ._beeswarm import envelope_edges, stack_swarm, terminal_swarm

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


@dataclass(frozen=True)
class PreparedBeeswarm:
    """Cohort coordinates and ordering, independent of zoom and screen size."""

    positions: np.ndarray
    order: np.ndarray
    ordinary: np.ndarray
    init: np.ndarray
    escape: np.ndarray
    slots: np.ndarray | None


def prepare_beeswarm(events, terminal_regions=None):
    positions = events["position"].to_numpy().astype(np.float64, copy=False)
    ids = events["id"].to_numpy()
    order = np.lexsort((ids, positions))
    empty = np.empty(0, dtype=np.int64)
    if terminal_regions is None:
        return PreparedBeeswarm(positions, order, order, empty, empty, None)
    initial = (events["type"] == "init").to_numpy()
    escaped = (events["type"] == "escape").to_numpy()
    ordinary = order[~(initial | escaped)[order]]
    terminals = []
    for mask in (initial, escaped):
        indices = np.flatnonzero(mask)
        terminals.append(indices[np.argsort(ids[indices], kind="stable")])
    slots = np.array([initial_slot(terminal_regions), escape_slot(terminal_regions)])
    return PreparedBeeswarm(positions, order, ordinary, *terminals, slots)


def pack_beeswarm(
    events,
    start,
    end,
    width,
    height,
    diameter=10,
    terminal_regions=None,
    *,
    prepared=None,
):
    """Pack tangent circles, reusing preparation for the same events and regions."""
    if prepared is None:
        prepared = prepare_beeswarm(events, terminal_regions)
    scale = float(width) / max(end - start, 1e-12)
    width, diameter = float(width), float(diameter)
    xs = (prepared.positions - start) * scale
    slots = None if prepared.slots is None else (prepared.slots - start) * scale
    if slots is not None and len(prepared.init) + len(prepared.escape):
        diameter = min(diameter, slots[0, 1] - slots[0, 0])
    ordinary_xs = xs[prepared.ordinary]
    lower, upper = 0.0, diameter
    best = None
    for _ in range(16):
        if slots is None:
            ys = np.empty(events.height)
            ys[prepared.order] = stack_swarm(ordinary_xs, width, diameter)
            packed_xs = xs
        else:
            packed_xs, ys = terminal_swarm(
                xs,
                ordinary_xs,
                prepared.ordinary,
                prepared.init,
                prepared.escape,
                slots,
                width,
                diameter,
            )
        extent = np.max(np.abs(ys), initial=0) * 2 + diameter
        if extent <= height:
            best = ys, packed_xs, diameter
            lower = diameter
        else:
            upper = diameter
        if upper - lower <= max(upper * 0.01, 1e-6):
            break
        diameter = (lower + upper) / 2 if lower else diameter * height / extent * 0.95
    ys, packed_xs, diameter = best
    columns = [pl.Series("plot_y", ys)]
    if slots is not None:
        columns.append(pl.Series("position", packed_xs / scale + start))
    return events.with_columns(*columns), diameter


def beeswarm_envelope(events, x_padding, y_padding):
    """Outline the packed circles using their radii in each coordinate."""
    if events.is_empty():
        return pl.DataFrame(schema={"position": pl.Float64, "plot_y": pl.Float64})
    xs = events["position"].to_numpy().astype(np.float64, copy=False)
    ys = events["plot_y"].to_numpy().astype(np.float64, copy=False)
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    offsets = np.array([-1, -sqrt(3) / 2, -0.5, 0, 0.5, sqrt(3) / 2, 1])
    samples = np.unique((xs[:, None] + offsets * x_padding).ravel())
    upper, lower = envelope_edges(xs, ys, samples, float(x_padding), float(y_padding))
    return pl.DataFrame(
        {
            "position": np.concatenate((samples, samples[::-1])),
            "plot_y": np.concatenate((upper, lower[::-1])),
        }
    )
