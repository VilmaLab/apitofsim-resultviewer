"""Bokeh presentation and interaction for the realization Explorer."""

from textwrap import fill

import polars as pl
from bokeh.events import MouseMove, Pan, PanEnd, PanStart
from bokeh.core.property.descriptors import UnsetValueError
from bokeh.models import (
    ColumnDataSource,
    CustomJS,
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
from .details import format_value
from .layout import (
    beeswarm_envelope,
    escape_slot,
    initial_slot,
    pack_beeswarm,
    prepare_beeswarm,
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

EVENT_TOOLTIP_CSS = """
.bk-tooltip-content > div > div:not(:first-child) .explorer-hover-head { display: none; }
.explorer-hover-head, .explorer-hover-row {
    display: grid; grid-template-columns: 74px 84px 70px 96px;
    gap: 4px; padding: 4px;
}
.explorer-hover-head { font-weight: bold; border-bottom: 1px solid #cbd5e1; }
.explorer-hover-row { border-bottom: 1px solid #e2e8f0; }
.explorer-hover-row span:nth-child(n+3) { text-align: right; }
"""

EVENT_TOOLTIP = """
<div class="explorer-hover-head"><span>Realization</span><span>Type</span><span>Time (ns)</span><span>Axial dist. (mm)</span></div>
<div class="explorer-hover-row"><span>@realization</span><span>@type</span><span>@time_text</span><span>@axial_text</span></div>
"""


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


def _schematic_plot(
    regions, shared_x, available_width, drag_bridge, initial=True, escaped=True
):
    diagram = figure(
        height=56,
        min_height=56,
        x_range=shared_x,
        tools="",
        toolbar_location=None,
        output_backend="webgl",
        sizing_mode="stretch_width",
    )
    diagram.name = "schematic"
    diagram.toolbar.logo = None
    diagram.y_range = Range1d(0, 1)
    diagram.yaxis.visible = False
    diagram.xaxis.visible = False
    diagram.grid.visible = False
    diagram.outline_line_color = None
    slots = [
        initial_slot(regions),
        *regions.select("left", "right").iter_rows(),
        escape_slot(regions),
    ]
    names = ["Initial", *regions["name"].to_list(), "Escaped"]
    short = ["Init", "C1", "Sk", "C2 pre", "Quad", "C2 post", "Esc"]
    visible = [initial, True, True, True, True, True, escaped]
    source = _source(
        pl.DataFrame(
            {
                "left": [slot[0] for slot in slots],
                "right": [slot[1] for slot in slots],
                "name": names,
                "color": [
                    "#f8fafc",
                    "#dbeafe",
                    "#e2e8f0",
                    "#dbeafe",
                    "#bfdbfe",
                    "#dbeafe",
                    "#f8fafc",
                ],
                "alpha": [int(visible[0]), 1, 1, 0, 0, 0, int(visible[6])],
                "bottom": [0.12, 0.12, 0.12, 0.12, 0.196, 0.12, 0.12],
                "top": [0.88, 0.88, 0.88, 0.88, 0.804, 0.88, 0.88],
            }
        )
    )
    glyph = diagram.quad(
        left="left",
        right="right",
        bottom="bottom",
        top="top",
        source=source,
        fill_color="color",
        fill_alpha="alpha",
        line_color="#64748b",
        line_alpha="alpha",
    )
    chamber_source = ColumnDataSource(
        dict(left=[slots[3][0]], right=[slots[5][1]], name=["Second chamber"])
    )
    diagram.quad(
        left="left",
        right="right",
        bottom=0.12,
        top=0.88,
        source=chamber_source,
        fill_color="#dbeafe",
        line_color="#64748b",
    )
    quadrupole_source = ColumnDataSource(
        dict(left=[slots[4][0]], right=[slots[4][1]], name=["Quadrupole"])
    )
    diagram.quad(
        left="left",
        right="right",
        bottom=0.196,
        top=0.804,
        source=quadrupole_source,
        fill_color="#bfdbfe",
        line_color="#64748b",
    )
    diagram.add_tools(HoverTool(renderers=[glyph], tooltips=[("Zone", "@name")]))
    labels = []
    full = [fill(name, width=16) for name in names]
    for index, (lo, hi) in enumerate(slots):
        label = Label(
            x=(lo + hi) / 2,
            y=0.5,
            text=short[index]
            if visible[index] and available_width < 600
            else full[index]
            if visible[index]
            else "",
            text_align="center",
            text_baseline="middle",
            text_font_size="9px",
        )
        diagram.add_layout(label)
        labels.append(label)

    args = dict(
        source=source,
        chamber=chamber_source,
        quadrupole=quadrupole_source,
        labels=labels,
        full=full,
        short=short,
        visible=visible,
        plot=diagram,
        bridge=drag_bridge,
    )
    common = """
        const data = source.data;
        const bounds = data.right.slice(0, 6);
        const span = plot.x_range.end - plot.x_range.start;
        const units = span / Math.max(plot.inner_width, 1);
        const min = units * 5;
        const instrument = [1, 2, 3, 4, 5];
        const secondChamber = [3, 4, 5];
        // Each boundary exchanges width within its containing section.
        const groups = [
            [[0], instrument],
            [[1], [2, ...secondChamber]],
            [[1, 2], secondChamber],
            [[3], [4]], [[4], [5]],
            [instrument, [6]],
        ];
        function hit(x) {
            let best = -1, distance = units * 10;
            for (let i = 0; i < bounds.length; i++) {
                if ((i === 0 && !visible[0]) || (i === 5 && !visible[6])) continue;
                const d = Math.abs(bounds[i] - x);
                if (d < distance) { best = i; distance = d; }
            }
            return best;
        }
        function labelsFit() {
            const canvas = document.createElement('canvas');
            const context = canvas.getContext('2d');
            context.font = '9px sans-serif';
            for (let i = 0; i < labels.length; i++) {
                labels[i].x = (source.data.left[i] + source.data.right[i]) / 2;
                if (!visible[i]) { labels[i].text = ''; continue; }
                const room = (source.data.right[i] - source.data.left[i]) / units - 4;
                const fits = text => Math.max(...text.split('\\n').map(line => context.measureText(line).width)) <= room;
                labels[i].text = fits(full[i]) ? full[i] : fits(short[i]) ? short[i] : '';
            }
        }
        function resizeGroup(widths, indices, change) {
            if (Math.abs(change) < 1e-12) return;
            if (change > 0) {
                const total = indices.reduce((sum, i) => sum + widths[i], 0);
                indices.forEach(i => { widths[i] *= 1 + change / total; });
                return;
            }
            let remaining = -change;
            let active = indices.slice();
            while (remaining > 1e-12 && active.length) {
                const total = active.reduce((sum, i) => sum + widths[i], 0);
                const scale = Math.max(0, 1 - remaining / total);
                const limited = active.filter(i => widths[i] * scale < min);
                if (!limited.length) {
                    active.forEach(i => { widths[i] *= scale; });
                    break;
                }
                for (const i of limited) {
                    remaining -= widths[i] - min;
                    widths[i] = min;
                }
                active = active.filter(i => !limited.includes(i));
            }
        }
        function preview(x) {
            const drag = source._explorerDrag;
            if (drag == null) return;
            const widths = drag.widths.slice();
            const [left, right] = groups[drag.index];
            const slack = indices => indices.reduce((sum, i) => sum + widths[i] - min, 0);
            const delta = Math.max(-slack(left), Math.min(slack(right), x - drag.x));
            resizeGroup(widths, left, delta);
            resizeGroup(widths, right, -delta);
            let edge = drag.start;
            const lefts = [], rights = [];
            for (const width of widths) { lefts.push(edge); edge += width; rights.push(edge); }
            source.data = {...source.data, left: lefts, right: rights};
            chamber.data = {...chamber.data, left: [lefts[3]], right: [rights[5]]};
            quadrupole.data = {...quadrupole.data, left: [lefts[4]], right: [rights[4]]};
            drag.current = widths;
            labelsFit();
        }
    """
    diagram.js_on_change(
        "inner_width", CustomJS(args=args, code=common + "labelsFit();")
    )
    diagram.js_on_event(
        MouseMove,
        CustomJS(
            args=args,
            code=common
            + """
        const el = Bokeh.index.find_one(plot)?.canvas_view?.events_el;
        if (el != null) el.style.cursor = hit(cb_obj.x) >= 0 ? 'col-resize' : '';
    """,
        ),
    )
    diagram.js_on_event(
        PanStart,
        CustomJS(
            args=args,
            code=common
            + """
        const index = hit(cb_obj.x);
        if (index >= 0) source._explorerDrag = {
            index, x: cb_obj.x, sx: cb_obj.sx, start: data.left[0],
            widths: data.right.map((right, i) => right - data.left[i]),
        };
    """,
        ),
    )
    diagram.js_on_event(
        Pan,
        CustomJS(
            args=args,
            code=common
            + "if (source._explorerDrag) preview(source._explorerDrag.x + (cb_obj.sx - source._explorerDrag.sx) * units);",
        ),
    )
    diagram.js_on_event(
        PanEnd,
        CustomJS(
            args=args,
            code=common
            + """
        if (source._explorerDrag) preview(source._explorerDrag.x + (cb_obj.sx - source._explorerDrag.sx) * units);
        if (source._explorerDrag != null) {
            bridge.text = JSON.stringify(source._explorerDrag.current);
            source._explorerDrag = null;
        }
    """,
        ),
    )
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
    if not events.is_empty():
        source = _source(
            events.select(
                pl.col("position").alias("x"),
                pl.col("plot_y").alias("y"),
                "id",
                pl.col("realization_id").alias("realization"),
                "type",
                (pl.col("t") * 1e9).alias("time_ns"),
                (pl.col("z") * 1000).alias("axial_distance"),
                pl.Series("color", [event_colors[k] for k in events["event_group"]]),
                pl.Series("time_text", [format_value(t * 1e9) for t in events["t"]]),
                pl.Series("axial_text", [format_value(z * 1000) for z in events["z"]]),
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
            color="color",
            line_width=0 if layout == "beeswarm" else 1,
        )
        swarm_glyphs.append(glyph)
        plot.add_tools(
            HoverTool(
                renderers=[glyph],
                tooltips=EVENT_TOOLTIP,
                sort_by=["@realization", "@time_ns", "@id"],
                callback=CustomJS(
                    args=dict(css=EVENT_TOOLTIP_CSS),
                    code="""
                    const view = Bokeh.index.find_one(cb_obj);
                    for (const tooltip of view?._ttviews?.values() ?? []) {
                        const root = tooltip.shadow_el;
                        if (root && !root.querySelector('.explorer-tooltip-style')) {
                            const style = document.createElement('style');
                            style.className = 'explorer-tooltip-style';
                            style.textContent = css;
                            root.appendChild(style);
                        }
                    }
                    """,
                ),
            )
        )
        source.selected.on_change("indices", on_selected(source))
    plot.select_one(TapTool).renderers = swarm_glyphs
    packed = events
    update_highlight = _highlight(plot, lambda: packed, selection)
    if selection_updates is not None:
        selection_updates.append(update_highlight)
    if layout == "beeswarm" or spread_terminals:
        prepared = (
            prepare_beeswarm(events, regions if spread_terminals else None)
            if layout == "beeswarm"
            else None
        )
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
                    prepared=prepared,
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
