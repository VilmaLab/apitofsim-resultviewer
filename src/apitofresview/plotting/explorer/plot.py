"""Bokeh presentation and interaction for the realization Explorer."""

from collections.abc import Callable
from dataclasses import dataclass
from textwrap import fill

import polars as pl
from bokeh.events import MouseMove, Pan, PanEnd, PanStart
from bokeh.core.property.descriptors import UnsetValueError
from bokeh.models import (
    ColumnDataSource,
    CustomJS,
    FixedTicker,
    GlyphRenderer,
    HoverTool,
    Label,
    Quad,
    Range1d,
    Span,
    TapTool,
    VArea,
)
from bokeh.palettes import Category10, Category20
from bokeh.plotting import figure

from .data import REGIONS
from .browser import ON_FRAME
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
    "rejected_collision": Category20[20][1],
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


def _schematic_plot(shared_x, drag_bridge, *, live):
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
    names = ["Initial", *REGIONS, "Escaped"]
    short = ["Init", "C1", "Sk", "C2 pre", "Quad", "C2 post", "Esc"]
    source = ColumnDataSource(syncable=False)
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
    chamber_source = ColumnDataSource(syncable=False)
    diagram.quad(
        left="left",
        right="right",
        bottom=0.12,
        top=0.88,
        source=chamber_source,
        fill_color="#dbeafe",
        line_color="#64748b",
    )
    quadrupole_source = ColumnDataSource(syncable=False)
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
    for _ in names:
        label = Label(
            x=0,
            y=0.5,
            text="",
            text_align="center",
            text_baseline="middle",
            text_font_size="9px",
            syncable=False,
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
        plot=diagram,
        bridge=drag_bridge,
    )
    common = (
        ON_FRAME
        + """
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
                if ((i === 0 && !data.alpha[0]) || (i === 5 && !data.alpha[6])) continue;
                const d = Math.abs(bounds[i] - x);
                if (d < distance) { best = i; distance = d; }
            }
            return best;
        }
        function labelsFit() {
            if (!(plot.inner_width > 0)) return;
            if (!source.explorerLabelWidths) {
                const context = document.createElement('canvas').getContext('2d');
                context.font = '9px sans-serif';
                const measure = text => Math.max(...text.split('\\n').map(line => context.measureText(line).width));
                source.explorerLabelWidths = [full.map(measure), short.map(measure)];
            }
            const [fullWidths, shortWidths] = source.explorerLabelWidths;
            for (let i = 0; i < labels.length; i++) {
                labels[i].x = (source.data.left[i] + source.data.right[i]) / 2;
                if ((i === 0 || i === 6) && !source.data.alpha[i]) {
                    labels[i].text = '';
                    continue;
                }
                const room = (source.data.right[i] - source.data.left[i]) / units - 4;
                labels[i].text = fullWidths[i] <= room ? full[i] : shortWidths[i] <= room ? short[i] : '';
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
            if (drag.current?.every((value, i) => value === widths[i])) return;
            let edge = drag.start;
            const lefts = [], rights = [];
            for (const width of widths) { lefts.push(edge); edge += width; rights.push(edge); }
            source.data = {...source.data, left: lefts, right: rights};
            chamber.data = {...chamber.data, left: [lefts[3]], right: [rights[5]]};
            quadrupole.data = {...quadrupole.data, left: [lefts[4]], right: [rights[4]]};
            drag.current = widths;
        }
    """
    )
    fit_labels = CustomJS(args=args, code=common + "labelsFit();")
    diagram.js_on_change("inner_width", fit_labels)
    source.js_on_change("data", fit_labels)
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
            + """
        const drag = source._explorerDrag;
        if (drag) {
            drag.sxLatest = cb_obj.sx;
            onFrame(source, 'explorerDragFrame', () => {
                if (source._explorerDrag === drag)
                    preview(drag.x + (drag.sxLatest - drag.sx) * units);
            });
        }
        """,
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
            const {current, widths} = source._explorerDrag;
            if (current.some((value, i) => Math.abs(value - widths[i]) > 1e-10))
                bridge.text = JSON.stringify(current);
            source._explorerDrag = null;
        }
    """,
        ),
    )

    def update(regions, initial, escaped):
        slots = [
            initial_slot(regions),
            *regions.select("left", "right").iter_rows(),
            escape_slot(regions),
        ]
        data = dict(
            left=[slot[0] for slot in slots],
            right=[slot[1] for slot in slots],
            name=regions["name"].to_list(),
            color=[
                "#f8fafc",
                "#dbeafe",
                "#e2e8f0",
                "#dbeafe",
                "#bfdbfe",
                "#dbeafe",
                "#f8fafc",
            ],
            alpha=[int(initial), 1, 1, 0, 0, 0, int(escaped)],
            bottom=[0.12, 0.12, 0.12, 0.12, 0.196, 0.12, 0.12],
            top=[0.88, 0.88, 0.88, 0.88, 0.804, 0.88, 0.88],
        )
        data["name"] = ["Initial", *data["name"], "Escaped"]
        if not live:
            for label, (left, right), text in zip(labels, slots, full, strict=True):
                label.update(x=(left + right) / 2, text=text)
            labels[0].text = full[0] if initial else ""
            labels[-1].text = full[-1] if escaped else ""
        if source.data != data:
            source.data = data
            chamber_source.data = dict(
                left=[slots[3][0]], right=[slots[5][1]], name=["Second chamber"]
            )
            quadrupole_source.data = dict(
                left=[slots[4][0]], right=[slots[4][1]], name=["Quadrupole"]
            )

    return diagram, update


def _physical_guides(plot, regions):
    boundaries = [regions["left"][0], *regions["right"]]
    guides = []
    for boundary in boundaries:
        guide = Span(
            location=boundary,
            dimension="height",
            line_color="#94a3b8",
            line_dash="dotted",
        )
        plot.add_layout(guide)
        guides.append(guide)
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

    return [*guides, targets]


def _highlight(plot, get_events, get_members, selection, local_spread, glyph):
    """Keep selection overlays alive and update them from displayed coordinates."""
    columns = ["position", "plot_y"]
    initial = dict(position=[], plot_y=[])
    if local_spread:
        initial.update(event_id=[], base_x=[])
    source = ColumnDataSource(initial, syncable=not local_spread)
    clicked_source = ColumnDataSource(initial, syncable=not local_spread)
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
    hidden_source = ColumnDataSource(
        dict(x=[], y=[], color=[]), name="selected_hidden_collision"
    )
    hidden_glyph = plot.scatter(
        "x",
        "y",
        source=hidden_source,
        marker=glyph.glyph.marker,
        size=glyph.glyph.size,
        color="color",
        line_width=glyph.glyph.line_width,
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
        if events is None:
            return
        visible = (
            events.head(0)
            if selection["selected"] is None
            else events.filter(pl.col("realization_id") == selection["selected"])
        )
        clicked = (
            visible.head(0)
            if selection.get("plot_event", selection["event"]) is None
            else visible.filter(
                pl.col("id") == selection.get("plot_event", selection["event"])
            )
        )
        hidden = selection["hidden_collision"]
        reveal = (
            clicked.is_empty()
            and hidden is not None
            and hidden["realization_id"] in get_members()
        )
        hidden_data = (
            dict(
                x=[hidden["position"]],
                y=[hidden["plot_y"]],
                color=[COLORS[hidden["event_group"]]],
            )
            if reveal
            else dict(x=[], y=[], color=[])
        )
        if hidden_source.data != hidden_data:
            hidden_source.data = hidden_data
        for target, frame in ((source, visible), (clicked_source, clicked)):
            data = frame.select(columns).to_dict(as_series=False)
            if local_spread:
                data.update(
                    event_id=[str(id) for id in frame["id"]],
                    base_x=data["position"].copy(),
                )
            if target is clicked_source and reveal:
                data.update(position=hidden_data["x"], plot_y=hidden_data["y"])
                if local_spread:
                    data.update(event_id=[str(hidden["id"])], base_x=hidden_data["x"])
            if target.data != data:
                target.data = data

    return update, [source, clicked_source], hidden_glyph


class PackingQueue:
    """Coalesce geometry from every facet before running server-side packing."""

    def __init__(self, doc):
        self.doc = doc
        self.live = doc.session_context is not None
        self.pending = {}
        self.timeout = None

    def schedule(self, plot, pack):
        if not self.live:
            pack()
            return
        self.pending[plot] = pack
        if self.timeout is not None:
            self.doc.remove_timeout_callback(self.timeout)
        self.timeout = self.doc.add_timeout_callback(self.flush, 50)

    def flush(self):
        self.timeout = None
        pending, self.pending = self.pending, {}
        for plot, pack in pending.items():
            if plot.document is self.doc:
                pack()


def update_range(bounds, start, end):
    """Update reset bounds and keep an explicitly zoomed view."""
    if bounds.reset_start is None or (bounds.start, bounds.end) == (
        bounds.reset_start,
        bounds.reset_end,
    ):
        bounds.start, bounds.end = start, end
    bounds.reset_start, bounds.reset_end = start, end


def _terminal_spread(plot, glyph, highlight_sources):
    """Spread terminal markers and their highlights in the same browser update."""
    source = glyph.data_source
    geometry = ColumnDataSource(data=dict(slots=[]), syncable=False)
    update = CustomJS(
        args=dict(
            plot=plot, glyph=glyph, geometry=geometry, highlights=highlight_sources
        ),
        code=ON_FRAME
        + r"""
        const x = plot.x_range, y = plot.y_range;
        const byRow = y.type === 'Range1d';
        if (!geometry.data.slots.length) {
            geometry.disconnectSpread?.();
            return;
        }
        const update = () => onFrame(geometry, 'explorerSpreadFrame', () => {
            if (!geometry.data.slots.length || !plot.document) return;
            const source = glyph.data_source;
            const {inner_width: width, inner_height: height} = plot;
            if (!(width > 0 && height > 0)) return;
            const {x: positions, type, id, realization, event_number: numbers} = source.data;
            const [slots] = geometry.data.slots;
            const scale = width / Math.max(x.end - x.start, 1e-12);
            const diameter = 6;
            const spacing = byRow ? height / Math.max(Math.abs(y.end - y.start), 1) : diameter;
            if (geometry.terminalData !== source.data) {
                geometry.terminalData = source.data;
                geometry.terminals = {init: [], escape: []};
                for (let i = 0; i < id.length; i++) {
                    if (type[i] === 'init' || type[i] === 'escape') geometry.terminals[type[i]].push([i, positions[i]]);
                }
            }
            const terminals = geometry.terminals;
            const moved = new Map();
            const patches = [];
            for (const [index, kind] of ['init', 'escape'].entries()) {
                let [lo, hi] = slots[index];
                const padding = Math.min(diameter / 2 / scale, (hi - lo) / 2);
                lo += padding;
                hi -= padding;
                const capacity = Math.max(1, Math.floor((hi - lo) * scale / diameter) + 1);
                const period = Math.min(capacity, Math.max(terminals[kind].length, 1),
                    Math.max(2, Math.ceil(diameter / spacing)));
                for (const [i, base] of terminals[kind]) {
                    let next = base;
                    if (slots[2]) {
                        const jitter = Math.sin(numbers[i] * 78.233) * 43758.5453;
                        const fraction = byRow ?
                            (period > 1 ? (realization[i] % period) / (period - 1) : 0.5) :
                            jitter - Math.floor(jitter);
                        next = lo + fraction * (hi - lo);
                    }
                    moved.set(String(id[i]), next);
                    if (positions[i] !== next) patches.push([i, next]);
                }
            }
            if (patches.length) source.patch({x: patches}, {sync: false});
            for (const highlight of highlights) {
                const changes = [];
                const {event_id, position, base_x} = highlight.data;
                for (let i = 0; i < event_id.length; i++) {
                    const next = moved.get(event_id[i]) ?? base_x[i];
                    if (position[i] !== next) changes.push([i, next]);
                }
                if (changes.length) highlight.patch({position: changes}, {sync: false});
            }
            glyph.visible = true;
        });
        // Runtime connections keep one facet's models out of the shared range.
        if (!geometry.disconnectSpread) {
            const ranges = byRow ? [x, y] : [x];
            const signals = ranges.flatMap(r => [r.properties.start.change, r.properties.end.change]);
            for (const signal of signals) signal.connect(update);
            geometry.disconnectSpread = () => {
                for (const signal of signals) signal.disconnect(update);
                geometry.disconnectSpread = null;
            };
        }
        update();
        """,
    )
    for name in ("inner_width", "inner_height"):
        plot.js_on_change(name, update)
    for model in (source, geometry, *highlight_sources):
        model.js_on_change("data", update)

    def set_slots(regions, enabled):
        bounds = [initial_slot(regions), escape_slot(regions)]
        data = dict(slots=[[*bounds, enabled]])
        if geometry.data != data:
            geometry.data = data

    def disconnect():
        geometry.data = dict(slots=[])

    return set_slots, disconnect


@dataclass
class EventView:
    plot: figure
    update_events: Callable
    update_display: Callable
    highlight: Callable
    set_spread: Callable
    dispose: Callable


def _event_plot(
    shared_x,
    height,
    layout,
    selection,
    on_selected,
    packing,
    *,
    shared_y=None,
    physical_regions=None,
    tree=False,
    terminal_spread=False,
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
    swarm = layout == "beeswarm"
    local_spread = terminal_spread and not swarm and packing.live
    if shared_y is not None:
        plot.y_range = shared_y
    plot.yaxis.axis_label = {
        "radial": "Radial distance (mm)",
        "realization": "Realization #",
    }.get(layout)
    plot.yaxis.visible = layout in ("radial", "realization")
    plot.ygrid.visible = layout == "radial"
    if layout == "realization":
        if shared_y is None:
            plot.y_range = Range1d(0.5, 1.5)
        plot.yaxis.ticker = FixedTicker(ticks=[1])
        plot.yaxis.major_tick_line_color = None
        plot.yaxis.minor_tick_line_color = None
    guides = (
        _physical_guides(plot, physical_regions) if physical_regions is not None else []
    )
    envelope = None
    if swarm:
        plot.y_range = Range1d(-height / 2, height / 2)
        for tool in plot.tools:
            if hasattr(tool, "dimensions"):
                tool.dimensions = "width"
        envelope = plot.patch(
            "position",
            "plot_y",
            source=ColumnDataSource(dict(position=[], plot_y=[])),
            fill_alpha=0.13,
            line_alpha=0.25,
            color="#64748b",
            visible=False,
        )
    source = ColumnDataSource(syncable=not local_spread)
    glyph = plot.scatter(
        "x",
        "y",
        source=source,
        marker="circle" if swarm else "dot",
        size=10 if swarm else 6,
        color="color",
        line_width=0 if swarm else 1,
        visible=not (packing.live and (swarm or local_spread)),
        syncable=not local_spread,
    )
    plot.add_tools(
        HoverTool(
            renderers=[glyph],
            tooltips=(
                EVENT_TOOLTIP.replace(
                    "<span>Type</span>", "<span>Species / type</span>"
                ).replace(
                    "<span>@type</span>",
                    "<span>@current_cluster_name<br>@type</span>",
                )
                if tree
                else EVENT_TOOLTIP
            ),
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
    next(tool for tool in plot.tools if isinstance(tool, TapTool)).renderers = [glyph]
    events = packed = prepared = regions = None
    spread = False
    last_dimensions = None
    realization_ids = ()
    update_highlight, highlight_sources, hidden_glyph = _highlight(
        plot, lambda: packed, lambda: realization_ids, selection, local_spread, glyph
    )
    set_slots, disconnect_spread = (
        _terminal_spread(plot, glyph, highlight_sources)
        if local_spread
        else (None, None)
    )

    def update_envelope():
        if envelope is not None and envelope.visible and last_dimensions is not None:
            start, end, width, *_ = last_dimensions
            radius = glyph.glyph.size / 2
            outline = beeswarm_envelope(packed, radius * (end - start) / width, radius)
            envelope.data_source.data = outline.to_dict(as_series=False)

    def repack():
        nonlocal last_dimensions, packed
        if events is None:
            return
        try:
            width, inner_height = plot.inner_width, plot.inner_height
        except UnsetValueError:
            width = inner_height = None
        if packing.live and (not width or not inner_height):
            return
        width = width or 600
        inner_height = inner_height or plot.height - 60
        dimensions = (shared_x.start, shared_x.end, width, inner_height)
        if not swarm and layout == "realization":
            dimensions += (plot.y_range.start, plot.y_range.end)
        if dimensions == last_dimensions:
            return
        last_dimensions = dimensions
        if swarm:
            if events.is_empty():
                packed, diameter = events, 10
            else:
                packed, diameter = pack_beeswarm(
                    events,
                    *dimensions,
                    terminal_regions=regions if spread else None,
                    prepared=prepared,
                )
            plot.y_range.start = plot.y_range.reset_start = -inner_height / 2
            plot.y_range.end = plot.y_range.reset_end = inner_height / 2
        else:
            diameter = 6
            packed = (
                spread_terminal_x(
                    events, regions, layout, *dimensions[:4], dimensions[4:], diameter
                )
                if spread
                else events
            )
        patch = {}
        for axis, column in (("x", "position"), ("y", "plot_y")):
            values = packed[column].to_list()
            if values != source.data[axis]:
                patch[axis] = [(slice(0, len(values)), values)]
        if patch:
            source.patch(patch)
        glyph.glyph.size = hidden_glyph.glyph.size = diameter
        update_envelope()
        update_highlight()
        glyph.visible = True

    def geometry_changed(attr, old, new):
        schedule_repack()

    def schedule_repack():
        if swarm or (terminal_spread and not local_spread):
            packing.schedule(plot, repack)

    if swarm or (terminal_spread and not local_spread):
        for name in ("inner_width", "inner_height"):
            plot.on_change(name, geometry_changed)
        for name in ("start", "end"):
            shared_x.on_change(name, geometry_changed)
            if not swarm and layout == "realization":
                plot.y_range.on_change(name, geometry_changed)

    def update_display(use_x_markers, physical_guides, show_envelope):
        if not swarm:
            glyph.glyph.marker = hidden_glyph.glyph.marker = (
                "x" if use_x_markers else "dot"
            )
        for guide in guides:
            guide.visible = physical_guides
        if envelope is not None and envelope.visible != show_envelope:
            envelope.visible = show_envelope
            update_envelope()

    def set_spread(enabled, force=False):
        nonlocal spread, prepared, last_dimensions
        if not force and spread == enabled:
            return
        spread = enabled
        last_dimensions = None
        if swarm:
            prepared = prepare_beeswarm(events, regions if spread else None)
        if local_spread:
            set_slots(regions, spread)
        else:
            schedule_repack()

    def update_events(
        frame,
        mapped_regions,
        event_colors,
        realization_bounds,
        spread_terminals,
        members,
    ):
        nonlocal events, packed, regions, realization_ids
        realization_ids = members
        events = None  # The source and bounds are being updated together.
        if layout == "realization":
            first, last = realization_bounds
            update_range(plot.y_range, first - 0.5, last + 0.5)
            plot.yaxis.ticker.ticks = sorted({first, last})
        events = packed = frame
        regions = mapped_regions
        if swarm and packing.live:
            packed = events.head(0)
            glyph.visible = False
            envelope.data_source.data = dict(position=[], plot_y=[])
        source.data = events.select(
            pl.col("position").alias("x"),
            pl.col("plot_y").alias("y"),
            "id",
            "realization_id",
            pl.col("realization_number").alias("realization"),
            "type",
            "event_number",
            *(["current_cluster_name"] if tree else []),
            (pl.col("t") * 1e9).alias("time_ns"),
            (pl.col("z") * 1000).alias("axial_distance"),
            pl.Series("color", [event_colors[k] for k in events["event_group"]]),
            pl.Series("time_text", [format_value(t * 1e9) for t in events["t"]]),
            pl.Series("axial_text", [format_value(z * 1000) for z in events["z"]]),
        ).to_dict(as_series=False)
        set_spread(spread_terminals, force=True)

    def dispose():
        if disconnect_spread is not None:
            disconnect_spread()
        if swarm or (terminal_spread and not local_spread):
            for name in ("start", "end"):
                shared_x.remove_on_change(name, geometry_changed)
                if not swarm and layout == "realization":
                    plot.y_range.remove_on_change(name, geometry_changed)

    return EventView(
        plot, update_events, update_display, update_highlight, set_spread, dispose
    )


def _set_series(plot, series):
    plot.renderers = [renderer for renderer, _ in series.values()]
    plot.toolbar.tools = [
        *(tool for tool in plot.tools if not isinstance(tool, HoverTool)),
        *(hover for _, hover in series.values() if hover is not None),
    ]


def _fraction_plot(shared_x, *, population=False):
    plot = figure(
        name="population" if population else "cdf",
        title=None,
        height=175,
        min_height=150,
        x_range=shared_x,
        y_range=Range1d(0, 1),
        output_backend="webgl",
        sizing_mode="stretch_width",
        tools="pan,wheel_zoom,reset,save",
    )
    plot.yaxis.axis_label = "Ions / realization" if population else "Fraction"
    series = {}

    def update(areas, colors, *, labels=None):
        nonlocal series
        next_series = {}
        for name, frame in areas.items():
            label = (labels or {}).get(name, name)
            if name in series:
                renderer, hover = series[name]
                renderer.data_source.data = frame.to_dict(as_series=False)
                renderer.glyph.fill_color = colors[name]
            else:
                renderer = GlyphRenderer(
                    glyph=VArea(
                        x="position",
                        y1="bottom",
                        y2="top",
                        fill_color=colors[name],
                        fill_alpha=0.85,
                    ),
                    data_source=_source(frame),
                    name=label if population else None,
                )
                hover = HoverTool(
                    renderers=[renderer],
                    mode="mouse" if population else "vline",
                    tooltips=[
                        ("Species", label) if population else ("Pathway", label),
                        ("Ions / realization", "@fraction{0.000}")
                        if population
                        else ("Fraction", "@fraction{0.0%}"),
                    ],
                )
            next_series[name] = (renderer, hover)
        series = next_series
        _set_series(plot, series)
        return max(1, max((frame["top"].max() for frame in areas.values()), default=0))

    return plot, update


def _bar_plot(shared_x, *, tree=False, pathway_mode=False):
    plot = figure(
        name="bars",
        title=None,
        height=155,
        min_height=130,
        x_range=shared_x,
        y_range=Range1d(0, 1),
        output_backend="webgl",
        sizing_mode="stretch_width",
        tools="pan,wheel_zoom,reset,save",
    )
    plot.yaxis.axis_label = "Events / realization" if tree else "Fraction"
    series, labels = {}, {}

    def update(bars, escapes, escape_colors, escape_labels):
        nonlocal series, labels
        next_series, next_labels = {}, {}
        top = bars["fraction"].max() or 0
        update_range(plot.y_range, 0, top * 1.08 if top and not pathway_mode else 1)
        for name, lo, hi, value in bars.select(
            "name", "left", "right", "fraction"
        ).iter_rows():
            segments = escapes.items() if name == "Escaped" else [(None, value)]
            bottom = 0
            for key, fraction in segments:
                identity = (name, key)
                color = (
                    escape_colors[key] if key is not None else COLORS["fragmentation"]
                )
                if identity in series:
                    renderer, hover = series[identity]
                else:
                    renderer = GlyphRenderer(
                        glyph=Quad(fill_alpha=0.65), data_source=ColumnDataSource()
                    )
                    hover = HoverTool(renderers=[renderer]) if key is not None else None
                renderer.glyph.update(
                    left=lo,
                    right=hi,
                    bottom=bottom,
                    top=bottom + fraction,
                    fill_color=color,
                )
                if hover is not None:
                    hover.tooltips = [
                        ("Pathway", escape_labels[key]),
                        ("Fraction", f"{fraction:.1%}"),
                    ]
                next_series[identity] = (renderer, hover)
                bottom += fraction
            label = labels.get(name)
            if label is None:
                label = Label(text=name, text_align="center", text_font_size="8px")
            label.update(x=(lo + hi) / 2, y=value)
            next_labels[name] = label
        _set_series(plot, next_series)
        plot.center = [
            *(item for item in plot.center if item not in labels.values()),
            *next_labels.values(),
        ]
        series, labels = next_series, next_labels

    return plot, update
