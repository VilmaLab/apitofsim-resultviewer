"""Bokeh presentation and interaction for the realization Explorer."""

from html import escape
from textwrap import fill

import polars as pl
from bokeh.events import DocumentReady
from bokeh.core.property.descriptors import UnsetValueError
from bokeh.layouts import column, row
from bokeh.models import (
    CheckboxGroup,
    ColumnDataSource,
    CustomAction,
    CustomJS,
    Div,
    FixedTicker,
    GroupBox,
    HoverTool,
    InlineStyleSheet,
    Label,
    Range1d,
    RangeSlider,
    Select,
    Span,
    Toggle,
)
from bokeh.palettes import Category10, Category20
from bokeh.plotting import figure

from .data import (
    EVENT_TYPES,
    REGIONS,
    aggregate,
    coordinate_regions,
    escape_slot,
    events_for,
    layout_events,
    load_data,
    position_events,
    pack_beeswarm,
    select_realizations,
    beeswarm_envelope,
)


COLORS = {
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


def _details(realization, events, selected_event, pathways):
    if realization.is_empty():
        return "<p>Click an event to inspect its realization.</p>"
    member = realization.row(0, named=True)
    rows = []
    history = events.join(pathways, on="pathway_id", how="left", maintain_order="left")
    for event in history.iter_rows(named=True):
        fields = (
            f"Event #{event['id']} · {event['type']} · t={event['t']:.6g} s · "
            f"x={event['x'] * 1000:.6g} mm · y={event['y'] * 1000:.6g} mm · "
            f"Axial distance={event['z'] * 1000:.6g} mm · "
            f"Radial distance={event['radial'] * 1000:.6g} mm"
        )
        if event["pathway_id"] is not None:
            fields += f" · {event['pathway'] or 'Unknown pathway'}"
        rows.append(
            f'<button class="explorer-event" data-event="{event["id"]}" id="event-{event["id"]}" '
            f'style="display:block;width:100%;text-align:left;padding:8px;'
            f'background:{"#fde68a" if event["id"] == selected_event else "white"};border-bottom:1px solid #ddd">'
            f"{escape(fields)}</button>"
        )
    return (
        f"<h3>Realization #{member['id']}</h3><p>Result #{member['experiment_result_id']}<br>"
        f"Pathway: {escape(member['fate'])}</p><button id='clear-selection'>Clear selection</button>"
        + "".join(rows)
    )


def _schematic_plot(regions, shared_x, available_width):
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
    for index, name, lo, hi in regions.select(
        "region", "name", "left", "right"
    ).iter_rows():
        display_name = (
            ("C1", "Sk", "C2 (before Quad)", "Quad", "C2 (after Quad)")[index]
            if available_width < 600
            else fill(name, width=16)
        )
        diagram.add_layout(
            Label(
                x=(lo + hi) / 2,
                y=0.5,
                text=display_name,
                text_align="center",
                text_baseline="middle",
                text_font_size="9px",
            )
        )
    diagram.add_layout(
        Label(
            x=sum(escape_slot(regions)) / 2,
            y=0.5,
            text="Esc" if available_width < 600 else "Escaped",
            text_align="center",
            text_baseline="middle",
            text_font_size="9px",
        )
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


def _highlight(plot, events, selected_id, event_id):
    visible = events.filter(pl.col("realization_id") == selected_id).sort("t", "id")
    if visible.is_empty():
        return
    source = _source(visible.select("position", "plot_y"))
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
    clicked = (
        visible.head(0)
        if event_id is None
        else visible.filter(pl.col("id") == event_id)
    )
    if not clicked.is_empty():
        plot.scatter(
            "position",
            "plot_y",
            source=_source(clicked.select("position", "plot_y")),
            marker="circle",
            size=20,
            fill_alpha=0,
            line_color="#f59e0b",
            line_width=3,
        )


def _event_plot(
    events,
    shared_x,
    height,
    layout,
    use_x_markers,
    regions,
    physical_guides,
    show_envelope,
    selected_id,
    event_id,
    on_selected,
    event_colors,
    realization_bounds=(1, 1),
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
    if layout == "beeswarm":
        last_dimensions = None
        highlights = []

        def repack(attr, old, new):
            nonlocal last_dimensions
            try:
                width = plot.inner_width or 600
            except UnsetValueError:
                width = 600
            try:
                inner_height = plot.inner_height or height - 60
            except UnsetValueError:
                inner_height = height - 60
            dimensions = (shared_x.start, shared_x.end, width, inner_height)
            if dimensions == last_dimensions:
                return
            last_dimensions = dimensions
            packed, diameter = pack_beeswarm(events, *dimensions)
            plot.y_range.start, plot.y_range.end = -inner_height / 2, inner_height / 2
            plot.y_range.reset_start = plot.y_range.start
            plot.y_range.reset_end = plot.y_range.end
            positions = dict(packed.select("id", "plot_y").iter_rows())
            for glyph in swarm_glyphs:
                glyph.data_source.data = {
                    **glyph.data_source.data,
                    "y": [positions[i] for i in glyph.data_source.data["id"]],
                }
                glyph.glyph.size = diameter
            if envelope_source is not None:
                outline = beeswarm_envelope(
                    packed,
                    diameter / 2 * (shared_x.end - shared_x.start) / width,
                    diameter / 2,
                )
                envelope_source.data = outline.to_dict(as_series=False)
            for renderer in highlights:
                plot.renderers.remove(renderer)
            highlights.clear()
            if selected_id is not None:
                previous = len(plot.renderers)
                _highlight(plot, packed, selected_id, event_id)
                highlights.extend(plot.renderers[previous:])

        for property_name in ("inner_width", "inner_height"):
            plot.on_change(property_name, repack)
        for property_name in ("start", "end"):
            shared_x.on_change(property_name, repack)
        repack(None, None, None)
    elif selected_id is not None:
        _highlight(plot, events, selected_id, event_id)
    return plot


def _cdf_plot(members, events, shared_x, left_edge, right_edge, fate_colors):
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
    # All pathways share a step grid so their areas meet even at tied events.
    terminals = members.join(
        events, left_on="terminal_event_id", right_on="id", how="inner"
    )
    counts = terminals.group_by("position", "fate").len().sort("position", "fate")
    positions = counts["position"].unique().sort().to_list()
    x = [left_edge, *[p for p in positions for _ in range(2)], right_edge]
    bottom = [0.0] * len(x)
    for name, color in fate_colors.items():
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
        area = plot.varea(
            x="position",
            y1="bottom",
            y2="top",
            source=ColumnDataSource(
                dict(position=x, bottom=bottom, top=top, fraction=fractions)
            ),
            fill_color=color,
            fill_alpha=0.85,
        )
        plot.add_tools(
            HoverTool(
                renderers=[area],
                mode="vline",
                tooltips=[("Pathway", name), ("Fraction", "@fraction{0.0%}")],
            )
        )
        bottom = top
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


def build_document(db, doc, experiment: int, cluster: int):
    """Build one Bokeh session; every control uses the same cohort and X mapping."""
    try:
        realizations, events, pathways, regions = load_data(db, experiment, cluster)
    except Exception as exc:
        # Non-realization databases and older configs have no Explorer data.
        doc.add_root(Div(text=f"Explorer data unavailable: {escape(str(exc))}"))
        return

    fate_names = realizations["fate"].unique().sort().to_list()
    fate_counts = dict(realizations.group_by("fate").len().iter_rows())
    escaped_fates = (
        realizations.join(
            events.filter(pl.col("type") == "escape"),
            left_on="terminal_event_id",
            right_on="id",
            how="inner",
        )["fate"]
        .unique()
        .to_list()
    )
    escape_name = escaped_fates[0] if escaped_fates else "Escape"
    events = (
        events.join(pathways, on="pathway_id", how="left", maintain_order="left")
        .rename({"pathway": "event_pathway"})
        .with_columns(pl.col("event_pathway").fill_null("Unknown fragmentation"))
    )
    fragmentation_names = (
        events.filter(pl.col("type") == "fragmentation")["event_pathway"]
        .unique()
        .sort()
        .to_list()
    )
    fate_colors = _pathway_colors(
        sorted(set(fate_names) | set(fragmentation_names)), escape_name
    )
    pathway_legend = "\n".join(
        f'label:nth-child({i + 1})::before {{ content: ""; background: {fate_colors[name]}; '
        "width: 12px; height: 12px; flex-shrink: 0; margin-right: 4px; }"
        for i, name in enumerate(fate_names)
    )
    count_styles = "\n".join(
        f'label:nth-child({i + 1}) span::after {{ content: " ({fate_counts[name]})"; font-style: italic; }}'
        for i, name in enumerate(fate_names)
    )
    state = {
        "selected": None,
        "event": None,
        "updating": False,
        "grouped": True,
        "auto_cdf": False,
        "syncing_sidebar": False,
        "sidebar": {False: [True, False], True: [False, False]},
    }

    counts = Div(styles={"white-space": "nowrap"})
    quality = Div()
    fate = CheckboxGroup(
        labels=fate_names,
        active=list(range(len(fate_names))),
        sizing_mode="stretch_width",
        stylesheets=[
            InlineStyleSheet(
                css="""
                    .bk-input-group { white-space: normal; }
                    label { display: flex; align-items: center; width: 100%; }
                    input { flex-shrink: 0; }
                    span { min-width: 0; overflow-wrap: anywhere; }
                """
                + count_styles
                + pathway_legend
            )
        ],
    )

    def checkbox(label, checked=False):
        return CheckboxGroup(labels=[label], active=[0] if checked else [])

    def enabled(widget):
        return bool(widget.active)

    fate_facets = checkbox("Facet pathways")
    group_fragmentations = checkbox("Group fragmentations", True)
    event_options = {kind: checkbox(kind.title(), True) for kind in EVENT_TYPES} | {
        name: checkbox(name, True) for name in fragmentation_names
    }
    for kind, widget in event_options.items():
        widget.name = f"event:{kind}"
        widget.sizing_mode = "stretch_width"
        widget.stylesheets = [InlineStyleSheet()]
    event_controls = column(spacing=4, width_policy="max")
    mode = Select(
        title="X coordinate",
        value="schematic",
        options=[
            ("schematic", "Equal chambers"),
            ("equal", "Equal zones"),
            ("physical", "Axial distance (mm)"),
            ("time", "Elapsed time (s)"),
        ],
    )
    layout = Select(
        title="Y coordinate",
        value="realization",
        options=[
            ("realization", "Realization #"),
            ("radial", "Radial distance (mm)"),
            ("strip", "Jittered strip"),
            ("beeswarm", "Beeswarm"),
        ],
    )
    envelope = checkbox("Envelope")
    schematic = checkbox("Schematic", True)
    guides = checkbox("Guides", True)
    show_events = checkbox("Realizations", True)
    x_markers = checkbox("Use x markers", True)
    show_cdf = checkbox("Cumulative")
    show_bars = checkbox("Bar chart")
    show_pager = checkbox("Realizations pager")
    total = realizations.height
    selector = RangeSlider(
        title="Realization IDs (inclusive index range)",
        start=1,
        end=max(total, 1),
        step=1,
        value=(1, max(total, 1)),
        disabled=total == 0,
        sizing_mode="stretch_width",
        height=50,
        visible=False,
    )
    details = Div(
        text=_details(realizations.head(0), events.head(0), None, pathways),
        sizing_mode="stretch_width",
    )
    detail_bridge = Div(text="", visible=False)
    fullscreen_state = Div(text="normal", visible=False)
    viewport_width = Div(text="1440,900", visible=False)

    def group(title, child):
        child.width_policy = "max"
        child.height_policy = "min"
        return GroupBox(
            title=title,
            child=child,
            width_policy="max",
            height_policy="min",
            margin=(5, 0),
            stylesheets=[InlineStyleSheet(css="fieldset { min-inline-size: 0; }")],
        )

    left_controls = column(
        group("Pathways", column(fate, fate_facets, spacing=4)),
        group("X-axis", column(mode, guides, spacing=4)),
        group(
            "Elements",
            column(
                schematic,
                group("Views", column(show_events, show_cdf, show_bars, spacing=4)),
                show_pager,
                spacing=4,
            ),
        ),
        group(
            "Realizations",
            column(
                x_markers,
                group_fragmentations,
                group("Events", event_controls),
                group("Y-axis", column(layout, envelope, spacing=4)),
                spacing=4,
            ),
        ),
        spacing=4,
        styles={"max-height": "calc(100vh - 295px)", "overflow-y": "auto"},
    )
    right_controls = column(
        group("Selected realization", details),
        styles={"max-height": "calc(100vh - 295px)", "overflow-y": "auto"},
    )
    left_toggle = Toggle(label="« Controls", active=True, width=100, height=32)
    right_toggle = Toggle(
        label="☷",
        active=False,
        width=36,
        height=32,
        html_attributes={"title": "Show selected realization"},
    )
    left = column(left_toggle, left_controls, width=205, spacing=4)
    right = column(right_toggle, right_controls, width=38, spacing=4)
    right_controls.visible = False
    center_plots = column(sizing_mode="stretch_width", spacing=0)
    center = column(
        counts,
        quality,
        center_plots,
        selector,
        sizing_mode="stretch_width",
        spacing=2,
        css_classes=["explorer-center"],
    )
    frame = column(
        row(left, center, right, sizing_mode="stretch_width", spacing=6),
        detail_bridge,
        fullscreen_state,
        viewport_width,
        sizing_mode="stretch_width",
        spacing=4,
        css_classes=["explorer-bokeh-frame"],
        stylesheets=[
            InlineStyleSheet(
                css="""
                       :host { background: white; }
                       :host(:fullscreen) {
                           width: 100vw !important; height: 100vh !important;
                           overflow: auto; padding: 6px;
                       }
                   """
            )
        ],
    )
    doc.add_root(frame)
    doc.js_on_event(
        DocumentReady,
        CustomJS(
            args=dict(width=viewport_width),
            code="""
        const update = () => { width.text = `${window.innerWidth},${window.innerHeight}`; };
        update();
        window.addEventListener('resize', update);
    """,
        ),
    )

    details.js_on_change(
        "text",
        CustomJS(
            args=dict(bridge=detail_bridge),
            code="""
        setTimeout(() => {
            const view = Bokeh.index.find_one(cb_obj);
            const root = view?.shadow_el ?? view?.el?.shadowRoot ?? view?.el;
            if (!root) return;
            root.querySelectorAll('.explorer-event').forEach(button => {
                button.onclick = () => { bridge.text = button.dataset.event; };
            });
            const clear = root.querySelector('#clear-selection');
            if (clear) clear.onclick = () => { bridge.text = 'clear'; };
            const active = root.querySelector('.explorer-event[style*="fde68a"]');
            if (active) active.scrollIntoView({block: 'nearest'});
        }, 0);
    """,
        ),
    )

    def apply_sidebars(full):
        left_open, right_open = state["sidebar"][full]
        narrow = int(viewport_width.text.split(",")[0]) < 900
        if narrow and left_open and right_open:
            left_open = False
            state["sidebar"][full][0] = False
        state["syncing_sidebar"] = True
        try:
            left_toggle.active, right_toggle.active = left_open, right_open
        finally:
            state["syncing_sidebar"] = False
        left_controls.visible, right_controls.visible = left_open, right_open
        left.width = 205 if left_open else 38
        right.width = (250 if narrow else 300) if right_open else 38
        left_toggle.width, right_toggle.width = (
            (100 if left_open else 32),
            (110 if right_open else 32),
        )
        left_toggle.label = "« Controls" if left_open else "»"
        right_toggle.label = "Hide details »" if right_open else "☷"
        render()

    def sidebar_changed(index, active):
        if state["syncing_sidebar"]:
            return
        state["sidebar"][fullscreen_state.text == "full"][index] = active
        apply_sidebars(fullscreen_state.text == "full")

    left_toggle.on_change("active", lambda attr, old, new: sidebar_changed(0, new))
    right_toggle.on_change("active", lambda attr, old, new: sidebar_changed(1, new))

    def full_changed(attr, old, new):
        sidebar_height = (
            "calc(100vh - 46px)" if new == "full" else "calc(100vh - 295px)"
        )
        left_controls.styles = {**left_controls.styles, "max-height": sidebar_height}
        right_controls.styles = {**right_controls.styles, "max-height": sidebar_height}
        apply_sidebars(new == "full")

    fullscreen_state.on_change("text", full_changed)
    viewport_width.on_change(
        "text", lambda attr, old, new: apply_sidebars(fullscreen_state.text == "full")
    )

    def selected_cohort():
        lo, hi = (int(v) for v in selector.value)
        chosen_fates = {fate_names[i] for i in fate.active}
        return select_realizations(realizations, (lo, hi), chosen_fates)

    def set_selection(rid, event_id):
        state["selected"], state["event"] = rid, event_id
        if rid is not None:
            state["sidebar"][fullscreen_state.text == "full"][1] = True
            apply_sidebars(fullscreen_state.text == "full")
        render()

    def sidebar_event(attr, old, new):
        if new == "clear":
            set_selection(None, None)
        elif new.isdigit() and state["selected"] is not None:
            history = events.filter(pl.col("realization_id") == state["selected"])
            if int(new) in history["id"]:
                set_selection(state["selected"], int(new))

    detail_bridge.on_change("text", sidebar_event)

    def prepare_toolbar(plot):
        """Use Bokeh's fullscreen toolbar icon for the entire explorer frame."""
        plot.toolbar.logo = None
        plot.add_tools(
            CustomAction(
                description="Fullscreen explorer",
                icon="fullscreen",
                callback=CustomJS(
                    args=dict(frame=frame, fullscreen_state=fullscreen_state),
                    code="""
                const view = Bokeh.index.find_one(frame);
                const element = view?.el;
                if (element == null) return;
                if (element._explorerFullscreenListener == null) {
                    element._explorerFullscreenListener = () => {
                        fullscreen_state.text = document.fullscreenElement === element ? 'full' : 'normal';
                    };
                    document.addEventListener('fullscreenchange', element._explorerFullscreenListener);
                }
                if (document.fullscreenElement === element) void document.exitFullscreen();
                else if (document.fullscreenElement == null) void element.requestFullscreen();
            """,
                ),
            )
        )
        plot.js_on_change(
            "inner_width",
            CustomJS(
                args=dict(toolbar=plot.toolbar),
                code="""
            const view = Bokeh.index.find_one(toolbar);
            if (view == null || view._explorerSized) return;
            view._explorerSized = true;
            for (const button of view.tool_button_views) {
                const style = button.el.style;
                style.setProperty('--button-width', '40px', 'important');
                style.setProperty('--button-height', '40px', 'important');
                style.setProperty('width', '40px', 'important');
                style.setProperty('height', '40px', 'important');
            }
        """,
            ),
        )

    def render():
        if state["updating"]:
            return
        state["updating"] = True
        try:
            chosen = selected_cohort().with_row_index("realization_number", offset=1)
            chosen_ids = chosen["id"]
            if state["selected"] not in chosen_ids:
                state["selected"] = state["event"] = None
            selected = (
                chosen.head(0)
                if state["selected"] is None
                else chosen.filter(pl.col("id") == state["selected"])
            )
            chosen_events = events_for(events, chosen)
            chosen_events = chosen_events.join(
                chosen.select(
                    pl.col("id").alias("realization_id"), "realization_number"
                ),
                on="realization_id",
                how="left",
                maintain_order="left",
            )
            selected_events = events_for(chosen_events, selected)
            details.text = _details(selected, selected_events, state["event"], pathways)
            counts.text = (
                f"<b>{chosen.height} selected / {total} total realizations</b>"
            )
            incomplete = chosen.filter(
                pl.col("fate").is_in(["Incomplete", "Ambiguous"])
            ).height
            clamped = chosen_events["z_clamped"].sum()
            quality.text = "<br>".join(
                message
                for message in (
                    f"{clamped} events with negative z position have been clamped to 0"
                    if clamped
                    else "",
                    f"<span style='color:#a21caf'>{incomplete} incomplete or ambiguous histories; "
                    "unresolved outcomes are not counted as escaped.</span>"
                    if incomplete
                    else "",
                )
                if message
            )
            group_fragmentations.disabled = enabled(fate_facets)
            if group_fragmentations.disabled:
                group_fragmentations.active = [0]
            grouped = enabled(group_fragmentations)
            if grouped != state["grouped"]:
                if grouped:
                    event_options["fragmentation"].active = (
                        [0]
                        if any(
                            enabled(event_options[name]) for name in fragmentation_names
                        )
                        else []
                    )
                else:
                    for name in fragmentation_names:
                        event_options[name].active = (
                            [0] if enabled(event_options["fragmentation"]) else []
                        )
                state["grouped"] = grouped
            selected_fates = {fate_names[i] for i in fate.active}
            available_events = events_for(
                events, realizations.filter(pl.col("fate").is_in(selected_fates))
            )
            kinds = (
                ["collision", "fragmentation", "escape"]
                if grouped
                else ["collision", *fragmentation_names, "escape"]
            )
            event_colors = {
                kind: COLORS[kind] if kind in EVENT_TYPES else fate_colors[kind]
                for kind in kinds
            }
            for kind, widget in event_options.items():
                if kind == "escape":
                    available = escape_name in selected_fates
                elif kind in EVENT_TYPES:
                    available = not available_events.filter(
                        pl.col("type") == kind
                    ).is_empty()
                else:
                    available = (
                        kind not in fate_names or kind in selected_fates
                    ) and kind in available_events["event_pathway"]
                widget.disabled = not available
            event_options["escape"].labels = ["Escape" if grouped else escape_name]
            event_controls.children = [event_options[kind] for kind in kinds]
            active_groups = {
                kind
                for kind in kinds
                if enabled(event_options[kind]) and not event_options[kind].disabled
            }
            chosen_events = chosen_events.with_columns(
                pl.when(pl.col("type") == "fragmentation")
                .then(pl.lit("fragmentation") if grouped else pl.col("event_pathway"))
                .otherwise(pl.col("type"))
                .alias("event_group")
            )
            selected_events = events_for(chosen_events, selected)
            mapped_regions = coordinate_regions(regions, mode.value)
            positioned = position_events(chosen_events, mapped_regions, mode.value)
            slot = escape_slot(mapped_regions)
            spatial = mode.value != "time"
            regional = mode.value in ("schematic", "equal")
            show_bars.disabled = not regional
            schematic.disabled = not regional
            guides.visible = mode.value == "physical"
            envelope.visible = layout.value == "beeswarm"
            x_markers.disabled = layout.value == "beeswarm"
            use_x_markers = enabled(x_markers) and not x_markers.disabled
            marker = "×" if use_x_markers else "●"
            for kind, color in event_colors.items():
                event_options[kind].stylesheets[0].css = (
                    f'label::before {{ content: "{marker}"; color: {color}; '
                    "display: inline-block; width: 12px; margin-right: 4px; text-align: center; font-size: 16px; }"
                    ".bk-input-group { white-space: normal; }"
                    "label { display: flex; align-items: center; width: 100%; }"
                    "input { flex-shrink: 0; } span { overflow-wrap: anywhere; min-width: 0; }"
                )
            if regional and state["auto_cdf"] and enabled(show_bars):
                show_cdf.active = []
                state["auto_cdf"] = False
            if not any(
                (
                    enabled(show_events),
                    enabled(show_cdf),
                    enabled(show_bars) and regional,
                )
            ):
                state["auto_cdf"] = enabled(show_bars) and not regional
                show_cdf.active = [0]
            rows = (
                [
                    (chosen.filter(pl.col("fate") == name), name)
                    for name in fate_names
                    if name in {fate_names[i] for i in fate.active}
                ]
                if enabled(fate_facets)
                else [(chosen, "All selected pathways")]
            )
            max_position = positioned["position"].max()
            if spatial:
                left_edge = mapped_regions["left"][0]
                right_edge = (
                    slot[1]
                    if regional
                    else max(
                        mapped_regions["right"][-1],
                        max_position
                        if max_position is not None
                        else mapped_regions["right"][-1],
                    )
                )
            else:
                left_edge, right_edge = (
                    0,
                    max_position if max_position is not None else 1,
                )
            if right_edge <= left_edge:
                right_edge = left_edge + 1
            screen_width, screen_height = (
                int(v) for v in viewport_width.text.split(",")
            )
            available_width = screen_width - left.width - right.width
            plot_height = (
                max(
                    440,
                    min(
                        760,
                        screen_height
                        - 175
                        - (175 if enabled(show_cdf) else 0)
                        - (155 if enabled(show_bars) and regional else 0),
                    ),
                )
                if fullscreen_state.text == "full"
                else 440
            )
            margin = (right_edge - left_edge) * 0.015
            shared_x = Range1d(
                left_edge - margin if layout.value == "beeswarm" else left_edge,
                right_edge + margin,
            )
            panels = []
            if enabled(schematic) and spatial and regional and not chosen.is_empty():
                panels.append(
                    _schematic_plot(mapped_regions, shared_x, available_width)
                )

            def selection_callback(source):
                def on_selected(attr, old, new):
                    if new and not state["updating"]:
                        i = new[0]
                        set_selection(
                            source.data["realization"][i], source.data["id"][i]
                        )

                return on_selected

            for members, row_name in rows:
                if members.is_empty() and chosen.is_empty():
                    continue
                member_events = events_for(positioned, members)
                plots = []
                if enabled(show_events):
                    visible = layout_events(
                        member_events.filter(
                            pl.col("event_group").is_in(active_groups)
                        ),
                        layout.value,
                    )
                    plots.append(
                        _event_plot(
                            visible,
                            shared_x,
                            plot_height,
                            layout.value,
                            use_x_markers,
                            mapped_regions,
                            enabled(guides) and mode.value == "physical",
                            enabled(envelope) and layout.value == "beeswarm",
                            state["selected"],
                            state["event"],
                            selection_callback,
                            event_colors,
                            (
                                members["realization_number"].min() or 1,
                                members["realization_number"].max() or 1,
                            ),
                        )
                    )
                if enabled(show_cdf) or (enabled(show_bars) and regional):
                    if enabled(show_cdf):
                        # Cumulative escapes occur at the machine endpoint, not
                        # in the padded escape slot used by the event view.
                        cumulative_events = (
                            member_events.with_columns(
                                pl.when(pl.col("type") == "escape")
                                .then(pl.lit(mapped_regions["right"][-1]))
                                .otherwise(pl.col("position"))
                                .alias("position")
                            )
                            if spatial
                            else member_events
                        )
                        plots.append(
                            _cdf_plot(
                                members,
                                cumulative_events,
                                shared_x,
                                left_edge,
                                right_edge,
                                {
                                    name: fate_colors[name]
                                    for name in fate_names
                                    if name in members["fate"]
                                },
                            )
                        )
                    if enabled(show_bars) and regional:
                        _, bars, _, _ = aggregate(
                            members, member_events, mapped_regions
                        )
                        plots.append(_bar_plot(bars, shared_x))
                if enabled(fate_facets) and plots:
                    plots[0].title = f"{row_name} (N={members.height})"
                for plot in plots:
                    plot.min_border_top = 0
                    plot.min_border_bottom = 0
                    if regional:
                        centers = [
                            (lo + hi) / 2
                            for lo, hi in mapped_regions.select(
                                "left", "right"
                            ).iter_rows()
                        ]
                        plot.xaxis.ticker = FixedTicker(ticks=centers)
                        plot.xaxis.major_label_overrides = {
                            center: str(index)
                            for index, center in enumerate(centers, start=1)
                        }
                        plot.xaxis.major_tick_line_color = None
                        plot.xaxis.minor_tick_line_color = None
                        plot.xgrid.ticker = FixedTicker(
                            ticks=mapped_regions["right"].to_list()
                        )
                        plot.xgrid.grid_line_color = "#94a3b8"
                        plot.xgrid.grid_line_alpha = 0.3
                    prepare_toolbar(plot)
                if plots:
                    for upper in plots[:-1]:
                        upper.xaxis.visible = False
                    plots[-1].xaxis.axis_label = (
                        "Elapsed time (s)"
                        if not spatial
                        else "Axial distance (mm)"
                        if mode.value == "physical"
                        else "Zone"
                    )
                panels.extend(plots)
            if chosen.is_empty():
                panels.insert(
                    0,
                    Div(
                        text="<p>No realizations in this cohort. Adjust the range or pathway filters.</p>"
                    ),
                )
            hidden = (
                selected_events.head(0)
                if state["event"] is None
                else selected_events.filter(
                    (pl.col("id") == state["event"])
                    & ~pl.col("event_group").is_in(active_groups)
                )
            )
            if not hidden.is_empty():
                highlighted = hidden.row(0, named=True)
                panels.insert(
                    0,
                    Div(
                        text=(
                            f"<b>Selected hidden event #{highlighted['id']}</b> · "
                            f"{highlighted['type']} · t={highlighted['t']:.6g} s · "
                            f"Axial distance={highlighted['z'] * 1000:.6g} mm"
                        )
                    ),
                )
            center_plots.children = panels
        finally:
            state["updating"] = False

    def control_changed(attr, old, new):
        render()

    def cdf_changed(attr, old, new):
        if not state["updating"]:
            state["auto_cdf"] = False
        render()

    for widget in event_options.values():
        widget.on_change("active", control_changed)
    for widget, property_name in [
        (fate, "active"),
        (fate_facets, "active"),
        (group_fragmentations, "active"),
        (mode, "value"),
        (layout, "value"),
        (envelope, "active"),
        (schematic, "active"),
        (guides, "active"),
        (show_events, "active"),
        (x_markers, "active"),
        (show_bars, "active"),
        (selector, "value"),
    ]:
        widget.on_change(property_name, control_changed)
    show_cdf.on_change("active", cdf_changed)
    show_pager.on_change(
        "active",
        lambda attr, old, new: setattr(selector, "visible", enabled(show_pager)),
    )
    render()
