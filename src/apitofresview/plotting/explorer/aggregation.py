"""Numerical data for regional bars and cumulative pathway areas."""

import polars as pl

from .layout import escape_slot


def regional_fractions(realizations, events, regions):
    """Region fractions use every selected realization as denominator."""
    n = realizations.height
    terminals = realizations.join(
        events, left_on="terminal_event_id", right_on="id", how="inner"
    )
    fragmented = terminals.filter(pl.col("type") == "fragmentation")
    escaped = terminals.filter(pl.col("type") == "escape").height
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
    return bars


def cumulative_areas(members, events, left_edge, right_edge, fates):
    """Stack pathways on a shared step grid, including tied terminal events."""
    terminals = members.join(
        events, left_on="terminal_event_id", right_on="id", how="inner"
    )
    counts = terminals.group_by("position", "fate").len().sort("position", "fate")
    positions = counts["position"].unique().sort().to_list()
    x = [left_edge, *[p for p in positions for _ in range(2)], right_edge]
    bottom = [0.0] * len(x)
    areas = {}
    for name in fates:
        pathway_counts = dict(
            counts.filter(pl.col("fate") == name).select("position", "len").iter_rows()
        )
        if not pathway_counts:
            continue
        cumulative = 0
        fractions = [0.0]
        for position in positions:
            fractions.append(cumulative / members.height)
            cumulative += pathway_counts.get(position, 0)
            fractions.append(cumulative / members.height)
        fractions.append(cumulative / members.height)
        top = [lo + value for lo, value in zip(bottom, fractions, strict=True)]
        areas[name] = pl.DataFrame(
            dict(position=x, bottom=bottom, top=top, fraction=fractions),
            schema_overrides={"position": pl.Float64},
        )
        bottom = top
    return areas
