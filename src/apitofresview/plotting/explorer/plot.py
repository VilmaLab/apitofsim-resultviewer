"""Bokeh presentation and interaction for the realization Explorer."""

from textwrap import fill

import polars as pl
from bokeh.core.property.descriptors import UnsetValueError
from bokeh.models import (
    ColumnDataSource,
    FixedTicker,
    HoverTool,
    Label,
    Range1d,
    Span,
    TapTool,
)
from bokeh.palettes import Category10, Category20
from bokeh.plotting import figure

from .data import REGIONS
from .layout import (
    beeswarm_envelope,
    escape_slot,
    initial_slot,
    pack_beeswarm,
    spread_terminal_x,
)

COLORS = {
    "init": "#a16207",
    "collision": Category10[10][0],
    "fragmentation": Category10[10][3],
    "escape": Category10[10][2],
}
PATHWAY_COLORS = (
    COLORS["fragmentation"],
    *(color for color in Category10[10] if color not in COLORS.values()),
    # Extend with lighter categorical colours, reserving both blues and greens.
    *(color for i, color in enumerate(Category20[20][1::2]) if i not in (0, 2)),
)


def _pathway_colors(names, escaped):
    unresolved = {"Ambiguous", "Incomplete"}
    fragmented = sorted(
        name for name in names if name != escaped and name not in unresolved
    )
    return {
        **{
            name: PATHWAY_COLORS[i % len(PATHWAY_COLORS)]
            for i, name in enumerate(fragmented)
        },
        **{name: Category10[10][7] for name in names if name in unresolved},
        escaped: COLORS["escape"],
    }


def _source(frame):
    return ColumnDataSource(frame.to_dict(as_series=False))


def _schematic_plot(regions, shared_x, available_width, initial=True, escaped=True):
    diagram = figure(
        height=56,
        min_height=56,
        x_range=shared_x,
        toolbar_location=None,
        output_backend="webgl",
        sizing_mode="stretch_width",
    )
    diagram.toolbar.logo = None
    diagram.y_range = Range1d(0, 1)
    diagram.yaxis.visible = False
    diagram.xaxis.visible = False
    diagram.grid.visible = False
    source = _source(
        regions.select("left", "right", "name").with_columns(
            pl.Series("color", ["#dbeafe", "#e2e8f0", "#dbeafe", "#e2e8f0", "#dbeafe"])
        )
    )
    glyph = diagram.quad(
        left="left",
        right="right",
        bottom=0.12,
        top=0.88,
        source=source,
        fill_color="color",
        line_color="#64748b",
    )
    diagram.add_tools(HoverTool(renderers=[glyph], tooltips=[("Zone", "@name")]))
    labels = []
    for index, name, lo, hi in regions.select(
        "region", "name", "left", "right"
    ).iter_rows():
        short = ("C1", "Sk", "C2 (before Quad)", "Quad", "C2 (after Quad)")[index]
        full = fill(name, width=16)
        display_name = short if available_width < 600 else full
        label = Label(
            x=(lo + hi) / 2,
            y=0.5,
            text=display_name,
            text_align="center",
            text_baseline="middle",
            text_font_size="9px",
        )
        diagram.add_layout(label)
        labels.append((label, short, full))
    for visible, slot, label, short in (
        (initial, initial_slot(regions), "Initial", "Init"),
        (escaped, escape_slot(regions), "Escaped", "Esc"),
    ):
        if not visible:
            continue
        annotation = Label(
            x=sum(slot) / 2,
            y=0.5,
            text=short if available_width < 600 else label,
            text_align="center",
            text_baseline="middle",
            text_font_size="9px",
        )
        diagram.add_layout(annotation)
        labels.append((annotation, short, label))

    def resize_labels(attr, old, new):
        for annotation, short, full in labels:
            annotation.text = short if new < 600 else full

    diagram.on_change("inner_width", resize_labels)
    return diagram


def _physical_guides(plot, regions):
    boundaries = [regions["left"][0], *regions["right"]]
    for boundary in boundaries:
        plot.add_layout(
            Span(
                location=boundary,
                dimension="height",
                line_color="#94a3b8",
                line_dash="dotted",
            )
        )
    # Invisible hover targets use their own range so they don't affect the events.
    plot.extra_y_ranges["guides"] = Range1d(0, 1)
    targets = plot.scatter(
        x="boundary",
        y=0.5,
        y_range_name="guides",
        source=ColumnDataSource(
            dict(
                boundary=boundaries,
                name=["Start", *[name + " end" for name in REGIONS]],
            )
        ),
        fill_alpha=0,
        line_alpha=0,
    )
    plot.add_tools(
        HoverTool(
            renderers=[targets],
            mode="vline",
            point_policy="follow_mouse",
            tooltips=[("Boundary", "@name")],
        )
    )


def _highlight(plot, get_events, selection):
    """Keep selection overlays alive and update them from displayed coordinates."""
    source = ColumnDataSource(dict(position=[], plot_y=[]))
    clicked_source = ColumnDataSource(dict(position=[], plot_y=[]))
    plot.line("position", "plot_y", source=source, color="#111827", line_width=2.5)
    plot.scatter(
        "position",
        "plot_y",
        source=source,
        marker="circle",
        size=14,
        fill_alpha=0,
        line_color="#111827",
        line_width=2,
    )
    plot.scatter(
        "position",
        "plot_y",
        source=clicked_source,
        marker="circle",
        size=20,
        fill_alpha=0,
        line_color="#f59e0b",
        line_width=3,
    )

    def update():
        events = get_events()
        visible = (
            events.head(0)
            if selection["selected"] is None
            else events.filter(pl.col("realization_id") == selection["selected"]).sort(
                "t", "id"
            )
        )
        clicked = (
            visible.head(0)
            if selection["event"] is None
            else visible.filter(pl.col("id") == selection["event"])
        )
        source.data = visible.select("position", "plot_y").to_dict(as_series=False)
        clicked_source.data = clicked.select("position", "plot_y").to_dict(
            as_series=False
        )

    update()
    return update


def _event_plot(
    events,
    shared_x,
    height,
    layout,
    use_x_markers,
    regions,
    physical_guides,
    show_envelope,
    selection,
    on_selected,
    event_colors,
    realization_bounds=(1, 1),
    spread_terminals=False,
    selection_updates=None,
    doc=None,
):
    plot = figure(
        name="events",
        title=None,
        height=height,
        min_height=360,
        min_width=210,
        x_range=shared_x,
        sizing_mode="stretch_width",
        output_backend="webgl",
        tools="pan,wheel_zoom,box_zoom,reset,save,tap",
    )
    plot.yaxis.axis_label = {
        "radial": "Radial distance (mm)",
        "realization": "Realization #",
    }.get(layout)
    plot.yaxis.visible = layout in ("radial", "realization")
    plot.ygrid.visible = layout == "radial"
    if layout == "realization":
        first, last = realization_bounds
        plot.y_range = Range1d(first - 0.5, last + 0.5)
        plot.yaxis.ticker = FixedTicker(ticks=sorted({first, last}))
        plot.yaxis.major_tick_line_color = None
        plot.yaxis.minor_tick_line_color = None
    if physical_guides:
        _physical_guides(plot, regions)
    swarm_glyphs = []
    envelope_source = None
    if layout == "beeswarm":
        plot.y_range = Range1d(-height / 2, height / 2)
        for tool in plot.tools:
            if hasattr(tool, "dimensions"):
                tool.dimensions = "width"
        if show_envelope:
            envelope_source = ColumnDataSource(dict(position=[], plot_y=[]))
            plot.patch(
                "position",
                "plot_y",
                source=envelope_source,
                fill_alpha=0.13,
                line_alpha=0.25,
                color="#64748b",
            )
    for kind, color in event_colors.items():
        subset = events.filter(pl.col("event_group") == kind)
        if subset.is_empty():
            continue
        source = _source(
            subset.select(
                pl.col("position").alias("x"),
                pl.col("plot_y").alias("y"),
                "id",
                pl.col("realization_id").alias("realization"),
                pl.col("t").alias("time"),
                (pl.col("z") * 1000).alias("axial_distance"),
            )
        )
        glyph = plot.scatter(
            "x",
            "y",
            source=source,
            marker="circle"
            if layout == "beeswarm"
            else "x"
            if use_x_markers
            else "dot",
            size=10 if layout == "beeswarm" else 6,
            color=color,
            line_width=0 if layout == "beeswarm" else 1,
        )
        swarm_glyphs.append(glyph)
        plot.add_tools(
            HoverTool(
                renderers=[glyph],
                tooltips=[
                    ("Event", "@id"),
                    ("Realization", "@realization"),
                    ("t (s)", "@time"),
                    ("Axial distance (mm)", "@axial_distance"),
                ],
            )
        )
        source.selected.on_change("indices", on_selected(source))
    plot.select_one(TapTool).renderers = swarm_glyphs
    packed = events
    update_highlight = _highlight(plot, lambda: packed, selection)
    if selection_updates is not None:
        selection_updates.append(update_highlight)
    if layout == "beeswarm" or spread_terminals:
        last_dimensions = None
        pending = False
        live = doc is not None and doc.session_context is not None

        def repack():
            nonlocal last_dimensions, packed, pending
            pending = False
            if live and plot.document is not doc:
                return
            try:
                width = plot.inner_width
            except UnsetValueError:
                width = None
            try:
                inner_height = plot.inner_height
            except UnsetValueError:
                inner_height = None
            if live and (not width or not inner_height):
                return
            width = width or 600
            inner_height = inner_height or height - 60
            dimensions = (
                shared_x.start,
                shared_x.end,
                width,
                inner_height,
                plot.y_range.start,
                plot.y_range.end,
            )
            if dimensions == last_dimensions:
                return
            if layout == "beeswarm":
                packed, diameter = pack_beeswarm(
                    events,
                    *dimensions[:4],
                    terminal_regions=regions if spread_terminals else None,
                )
                plot.y_range.start, plot.y_range.end = (
                    -inner_height / 2,
                    inner_height / 2,
                )
                plot.y_range.reset_start = plot.y_range.start
                plot.y_range.reset_end = plot.y_range.end
            else:
                diameter = 6
                packed = spread_terminal_x(
                    events, regions, layout, *dimensions[:4], dimensions[4:], diameter
                )
            # Beeswarm sets its own Y range; cache the resulting bounds.
            last_dimensions = (*dimensions[:4], plot.y_range.start, plot.y_range.end)
            positions = {
                i: (x, y)
                for i, x, y in packed.select("id", "position", "plot_y").iter_rows()
            }
            for glyph in swarm_glyphs:
                glyph.data_source.data = {
                    **glyph.data_source.data,
                    "x": [positions[i][0] for i in glyph.data_source.data["id"]],
                    "y": [positions[i][1] for i in glyph.data_source.data["id"]],
                }
                glyph.glyph.size = diameter
            if envelope_source is not None:
                outline = beeswarm_envelope(
                    packed,
                    diameter / 2 * (shared_x.end - shared_x.start) / width,
                    diameter / 2,
                )
                envelope_source.data = outline.to_dict(as_series=False)
            update_highlight()

        def schedule_repack(attr, old, new):
            nonlocal pending
            if live:
                if not pending:
                    pending = True
                    doc.add_next_tick_callback(repack)
            else:
                repack()

        for property_name in ("inner_width", "inner_height"):
            plot.on_change(property_name, schedule_repack)
        for property_name in ("start", "end"):
            shared_x.on_change(property_name, schedule_repack)
            if spread_terminals and layout == "realization":
                plot.y_range.on_change(property_name, schedule_repack)
        if not live:
            repack()
    return plot


def _cdf_plot(areas, shared_x, fate_colors):
    plot = figure(
        name="cdf",
        title=None,
        height=175,
        min_height=150,
        x_range=shared_x,
        y_range=Range1d(0, 1),
        output_backend="webgl",
        sizing_mode="stretch_width",
        tools="pan,wheel_zoom,reset,save",
    )
    plot.yaxis.axis_label = "Fraction"
    for name, frame in areas.items():
        area = plot.varea(
            x="position",
            y1="bottom",
            y2="top",
            source=_source(frame),
            fill_color=fate_colors[name],
            fill_alpha=0.85,
        )
        plot.add_tools(
            HoverTool(
                renderers=[area],
                mode="vline",
                tooltips=[("Pathway", name), ("Fraction", "@fraction{0.0%}")],
            )
        )
    return plot


def _bar_plot(bars, shared_x):
    top = bars["fraction"].max() or 0
    plot = figure(
        name="bars",
        title=None,
        height=155,
        min_height=130,
        x_range=shared_x,
        y_range=Range1d(0, top * 1.08 if top else 1),
        output_backend="webgl",
        sizing_mode="stretch_width",
        tools="pan,wheel_zoom,reset,save",
    )
    plot.yaxis.axis_label = "Fraction"
    for name, lo, hi, value in bars.select(
        "name", "left", "right", "fraction"
    ).iter_rows():
        plot.quad(
            left=lo,
            right=hi,
            bottom=0,
            top=value,
            fill_color=COLORS["escape"]
            if name == "Escaped"
            else COLORS["fragmentation"],
            fill_alpha=0.65,
        )
        plot.add_layout(
            Label(
                x=(lo + hi) / 2,
                y=value,
                text=name,
                text_align="center",
                text_font_size="8px",
            )
        )
    return plot
