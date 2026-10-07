"""Bokeh presentation and interaction for the realization Explorer."""

from html import escape

import polars as pl
from bokeh.events import DocumentReady
from bokeh.layouts import column, row
from bokeh.models import (
    CheckboxGroup,
    ColumnDataSource,
    CustomAction,
    CustomJS,
    Div,
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
    select_realizations,
    violin,
)


COLORS = {"collision": "#2563eb", "fragmentation": "#e11d48", "escape": "#059669"}


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
            f"x={event['x']:.6g} m · y={event['y']:.6g} m · z={event['z']:.6g} m · "
            f"r={event['radial']:.6g} m"
        )
        if event["pathway_id"] is not None:
            fields += (
                f" · pathway #{event['pathway_id']} ({event['pathway'] or 'unknown'})"
            )
        rows.append(
            f'<button class="explorer-event" data-event="{event["id"]}" id="event-{event["id"]}" '
            f'style="display:block;width:100%;text-align:left;padding:8px;'
            f'background:{"#fde68a" if event["id"] == selected_event else "white"};border-bottom:1px solid #ddd">'
            f"{escape(fields)}</button>"
        )
    return (
        f"<h3>Realization #{member['id']}</h3><p>Result #{member['experiment_result_id']}<br>"
        f"Fate: {escape(member['fate'])}</p><button id='clear-selection'>Clear selection</button>"
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
    diagram.add_tools(HoverTool(renderers=[glyph], tooltips=[("Region", "@name")]))
    for index, name, lo, hi in regions.select(
        "region", "name", "left", "right"
    ).iter_rows():
        display_name = (
            ("C1", "Sk", "Gap", "Quad", "C2")[index] if available_width < 600 else name
        )
        diagram.add_layout(
            Label(
                x=(lo + hi) / 2,
                y=0.5,
                text=display_name,
                text_align="center",
                text_font_size="9px",
            )
        )
    diagram.add_layout(
        Label(
            x=sum(escape_slot(regions)) / 2,
            y=0.5,
            text="Esc" if available_width < 600 else "Escaped",
            text_align="center",
        )
    )
    return diagram


def _physical_guides(plot, regions):
    boundaries = [regions["left"][0], *regions["right"]]
    for index, boundary in enumerate(boundaries):
        plot.add_layout(
            Span(
                location=boundary,
                dimension="height",
                line_color="#94a3b8",
                line_dash="dotted",
            )
        )
        plot.add_layout(
            Label(
                x=boundary,
                y=0,
                y_units="screen",
                text="Start" if index == 0 else REGIONS[index - 1] + " end",
                text_font_size="8px",
                text_color="#475569",
                angle=1.57,
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
    title,
    shared_x,
    height,
    layout,
    regions,
    physical_guides,
    show_violin,
    selected_id,
    event_id,
    on_selected,
):
    plot = figure(
        title=title,
        height=height,
        min_height=360,
        min_width=210,
        x_range=shared_x,
        sizing_mode="stretch_width",
        output_backend="webgl",
        tools="pan,wheel_zoom,box_zoom,reset,save,tap",
    )
    plot.yaxis.axis_label = (
        "Radial distance (m)" if layout == "radial" else "Event layout"
    )
    if physical_guides:
        _physical_guides(plot, regions)
    if show_violin:
        envelope = violin(events)
        if not envelope.is_empty():
            plot.patch(
                "position",
                "plot_y",
                source=_source(envelope),
                fill_alpha=0.13,
                line_alpha=0.25,
                color="#64748b",
            )
    for kind in EVENT_TYPES:
        subset = events.filter(pl.col("type") == kind)
        if subset.is_empty():
            continue
        source = _source(
            subset.select(
                pl.col("position").alias("x"),
                pl.col("plot_y").alias("y"),
                "id",
                pl.col("realization_id").alias("realization"),
                pl.col("t").alias("time"),
                "z",
            )
        )
        glyph = plot.scatter(
            "x",
            "y",
            source=source,
            marker="x",
            size=10,
            color=COLORS[kind],
            legend_label=kind.title(),
            line_width=1.5,
        )
        plot.add_tools(
            HoverTool(
                renderers=[glyph],
                tooltips=[
                    ("Event", "@id"),
                    ("Realization", "@realization"),
                    ("t (s)", "@time"),
                    ("z (m)", "@z"),
                ],
            )
        )
        source.selected.on_change("indices", on_selected(source))
    if selected_id is not None:
        _highlight(plot, events, selected_id, event_id)
    if plot.legend:
        plot.legend.click_policy = "hide"
        plot.legend.location = "top_left"
    return plot


def _cdf_plot(
    cdf,
    title,
    n,
    shared_x,
    left_edge,
    right_edge,
    escaped,
    unresolved,
    spatial,
    regional,
    regions,
):
    plot = figure(
        title=f"Fragmentation CDF · {title} (N={n})",
        height=175,
        min_height=150,
        x_range=shared_x,
        y_range=Range1d(0, 1),
        output_backend="webgl",
        sizing_mode="stretch_width",
        tools="pan,wheel_zoom,reset,save",
    )
    plot.yaxis.axis_label = "Fraction of all realizations"
    if n:
        plot.step(
            [left_edge, *cdf["position"], right_edge],
            [0, *cdf["fraction"], cdf["fraction"][-1] if not cdf.is_empty() else 0],
            mode="after",
            line_width=2,
            color="#be123c",
        )
        if spatial and escaped:
            bracket_x = sum(escape_slot(regions)) / 2 if regional else right_edge
            start = (n - escaped - unresolved) / n
            end = 1 if not unresolved else start + escaped / n
            plot.line(
                [bracket_x, bracket_x], [start, end], color="#059669", line_width=3
            )
            plot.add_layout(
                Label(
                    x=bracket_x,
                    y=(start + end) / 2,
                    text=f"Escaped {escaped / n:.1%}",
                    text_color="#047857",
                )
            )
    return plot


def _bar_plot(bars, title, n, shared_x):
    top = bars["fraction"].max() or 0
    plot = figure(
        title=f"Regional outcomes · {title} (N={n})",
        height=155,
        min_height=130,
        x_range=shared_x,
        y_range=Range1d(0, top * 1.08 if top else 1),
        output_backend="webgl",
        sizing_mode="stretch_width",
        tools="pan,wheel_zoom,reset,save",
    )
    plot.yaxis.axis_label = "Fraction of all realizations"
    for name, lo, hi, value in bars.select(
        "name", "left", "right", "fraction"
    ).iter_rows():
        plot.quad(
            left=lo,
            right=hi,
            bottom=0,
            top=value,
            fill_color="#059669" if name == "Escaped" else "#6366f1",
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
    if not fate_names:
        fate_names = ["Escaped"]
    state = {
        "selected": None,
        "event": None,
        "updating": False,
        "auto_cdf": False,
        "syncing_sidebar": False,
        "sidebar": {False: [True, False], True: [False, False]},
    }

    counts = Div(styles={"white-space": "nowrap"})
    quality = Div()
    fate = CheckboxGroup(labels=fate_names, active=list(range(len(fate_names))))

    def checkbox(label, checked=False):
        return CheckboxGroup(labels=[label], active=[0] if checked else [])

    def enabled(widget):
        return bool(widget.active)

    fate_facets = checkbox("Facet fates")
    event_types = CheckboxGroup(
        labels=[v.title() for v in EVENT_TYPES], active=[0, 1, 2]
    )
    mode = Select(
        title="X coordinate",
        value="schematic",
        options=[
            ("schematic", "Schematic distance"),
            ("equal", "Equal regions"),
            ("physical", "Physical distance (m)"),
            ("time", "Elapsed time (s)"),
        ],
    )
    layout = Select(
        title="Event layout",
        value="radial",
        options=[
            ("radial", "Radial distance"),
            ("strip", "Jittered strip"),
            ("beeswarm", "Beeswarm"),
        ],
    )
    violin = checkbox("Violin envelope")
    schematic = checkbox("Schematic / guides", True)
    restrictions = Div()
    show_events = checkbox("Realizations", True)
    show_cdf = checkbox("CDF")
    show_bars = checkbox("Regional bars")
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
        group("Events", column(event_types, spacing=4)),
        group("X axis", column(mode, schematic, restrictions, spacing=4)),
        group("Layout", column(layout, violin, spacing=4)),
        group("Views", column(row(show_events, show_cdf), show_bars, spacing=4)),
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
    center_plots = column(sizing_mode="stretch_width", spacing=4)
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
            chosen = selected_cohort()
            chosen_ids = chosen["id"]
            if state["selected"] not in chosen_ids:
                state["selected"] = state["event"] = None
            selected = (
                chosen.head(0)
                if state["selected"] is None
                else chosen.filter(pl.col("id") == state["selected"])
            )
            chosen_events = events_for(events, chosen)
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
            active_types = {EVENT_TYPES[i] for i in event_types.active}
            mapped_regions = coordinate_regions(regions, mode.value)
            positioned = position_events(chosen_events, mapped_regions, mode.value)
            slot = escape_slot(mapped_regions)
            spatial = mode.value != "time"
            regional = mode.value in ("schematic", "equal")
            if spatial:
                restrictions.text = (
                    ""
                    if regional
                    else "Regional bars require schematic or equal regions. Physical mode shows boundary guides."
                )
            else:
                restrictions.text = (
                    "Schematic and regional bars are unavailable in elapsed time."
                )
            show_bars.disabled = not regional
            schematic.disabled = not spatial
            violin.disabled = layout.value == "radial"
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
                else [(chosen, "All selected fates")]
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
            shared_x = Range1d(left_edge, right_edge + (right_edge - left_edge) * 0.015)
            panels = []
            axis_figures = []
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
                        member_events.filter(pl.col("type").is_in(active_types)),
                        layout.value,
                    )
                    plots.append(
                        _event_plot(
                            visible,
                            row_name if enabled(fate_facets) else "Realizations",
                            shared_x,
                            plot_height,
                            layout.value,
                            mapped_regions,
                            enabled(schematic) and mode.value == "physical",
                            enabled(violin) and layout.value != "radial",
                            state["selected"],
                            state["event"],
                            selection_callback,
                        )
                    )
                if enabled(show_cdf) or (enabled(show_bars) and regional):
                    cdf, bars, escaped, unresolved = aggregate(
                        members, member_events, mapped_regions
                    )
                    if enabled(show_cdf):
                        plots.append(
                            _cdf_plot(
                                cdf,
                                row_name,
                                members.height,
                                shared_x,
                                left_edge,
                                right_edge,
                                escaped,
                                unresolved,
                                spatial,
                                regional,
                                mapped_regions,
                            )
                        )
                    if enabled(show_bars) and regional:
                        plots.append(
                            _bar_plot(bars, row_name, members.height, shared_x)
                        )
                for plot in plots:
                    prepare_toolbar(plot)
                panels.extend(plots)
                axis_figures.extend(plots)
            if axis_figures:
                for upper in axis_figures[:-1]:
                    upper.xaxis.visible = False
                axis_figures[-1].xaxis.axis_label = (
                    "Elapsed time (s)"
                    if not spatial
                    else "Axial distance (m)"
                    if mode.value == "physical"
                    else "Machine position"
                )
            if chosen.is_empty():
                panels.insert(
                    0,
                    Div(
                        text="<p>No realizations in this cohort. Adjust the range or fate filters.</p>"
                    ),
                )
            hidden = (
                selected_events.head(0)
                if state["event"] is None
                else selected_events.filter(
                    (pl.col("id") == state["event"])
                    & ~pl.col("type").is_in(active_types)
                )
            )
            if not hidden.is_empty():
                highlighted = hidden.row(0, named=True)
                panels.insert(
                    0,
                    Div(
                        text=(
                            f"<b>Selected hidden event #{highlighted['id']}</b> · "
                            f"{highlighted['type']} · t={highlighted['t']:.6g} s · z={highlighted['z']:.6g} m"
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

    for widget, property_name in [
        (fate, "active"),
        (fate_facets, "active"),
        (event_types, "active"),
        (mode, "value"),
        (layout, "value"),
        (violin, "active"),
        (schematic, "active"),
        (show_events, "active"),
        (show_bars, "active"),
        (selector, "value"),
    ]:
        widget.on_change(property_name, control_changed)
    show_cdf.on_change("active", cdf_changed)
    render()
