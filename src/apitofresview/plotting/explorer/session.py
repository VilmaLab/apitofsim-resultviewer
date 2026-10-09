"""Cohort selection, controls, and rendering for one Explorer session."""

from html import escape

import polars as pl
from bokeh.models import Div, FixedTicker, Range1d

from .aggregation import cumulative_areas, regional_fractions
from .controls import CHECKBOX_CSS, create_controls, enabled, prepare_toolbar
from .data import EVENT_TYPES, events_for, load_data, select_realizations
from .details import render_details
from .layout import (
    coordinate_regions,
    escape_slot,
    initial_slot,
    layout_events,
    position_events,
)
from .plot import (
    COLORS,
    _bar_plot,
    _cdf_plot,
    _event_plot,
    _pathway_colors,
    _schematic_plot,
)


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
    state = {
        "selected": None,
        "event": None,
        "updating": False,
        "grouped": True,
        "auto_cdf": False,
        "syncing_sidebar": False,
        "sidebar": {False: [True, False], True: [False, False]},
    }

    total = realizations.height
    ui = create_controls(
        doc,
        fate_names,
        fate_counts,
        fragmentation_names,
        fate_colors,
        total,
        render_details(realizations.head(0), events.head(0), None),
    )

    selection_updates = []
    chosen = realizations.head(0)
    selection_events = events.head(0)
    active_groups = set()
    selection_notice = Div(visible=False)

    def event_plot_height():
        screen_height = int(ui.viewport_width.text.split(",")[1])
        regional = ui.mode.value in ("schematic", "equal")
        return (
            max(
                440,
                min(
                    760,
                    screen_height
                    - 175
                    - (175 if enabled(ui.show_cdf) else 0)
                    - (155 if enabled(ui.show_bars) and regional else 0),
                ),
            )
            if ui.fullscreen_state.text == "full"
            else 440
        )

    def resize_plots():
        for panel in ui.center_plots.children:
            if panel.name == "events":
                panel.height = event_plot_height()

    def apply_sidebars(full):
        left_open, right_open = state["sidebar"][full]
        narrow = int(ui.viewport_width.text.split(",")[0]) < 900
        if narrow and left_open and right_open:
            left_open = False
            state["sidebar"][full][0] = False
        state["syncing_sidebar"] = True
        try:
            ui.left_toggle.active, ui.right_toggle.active = left_open, right_open
        finally:
            state["syncing_sidebar"] = False
        ui.left_controls.visible, ui.right_controls.visible = left_open, right_open
        ui.left.width = 205 if left_open else 38
        ui.right.width = (250 if narrow else 300) if right_open else 38
        ui.left_toggle.width, ui.right_toggle.width = (
            (100 if left_open else 32),
            (110 if right_open else 32),
        )
        ui.left_toggle.label = "« Controls" if left_open else "»"
        ui.right_toggle.label = "Hide details »" if right_open else "☷"
        resize_plots()

    def sidebar_changed(index, active):
        if state["syncing_sidebar"]:
            return
        state["sidebar"][ui.fullscreen_state.text == "full"][index] = active
        apply_sidebars(ui.fullscreen_state.text == "full")

    ui.left_toggle.on_change("active", lambda attr, old, new: sidebar_changed(0, new))
    ui.right_toggle.on_change("active", lambda attr, old, new: sidebar_changed(1, new))

    def full_changed(attr, old, new):
        sidebar_height = (
            "calc(100vh - 46px)" if new == "full" else "calc(100vh - 295px)"
        )
        ui.left_controls.styles = {
            **ui.left_controls.styles,
            "max-height": sidebar_height,
        }
        ui.right_controls.styles = {
            **ui.right_controls.styles,
            "max-height": sidebar_height,
        }
        apply_sidebars(new == "full")

    ui.fullscreen_state.on_change("text", full_changed)
    ui.viewport_width.on_change(
        "text",
        lambda attr, old, new: apply_sidebars(ui.fullscreen_state.text == "full"),
    )

    def selected_cohort():
        lo, hi = (
            (int(v) for v in ui.selector.value)
            if enabled(ui.show_pager)
            else (1, max(total, 1))
        )
        chosen_fates = {fate_names[i] for i in ui.fate.active}
        return select_realizations(realizations, (lo, hi), chosen_fates)

    def set_selection(rid, event_id):
        state["selected"], state["event"] = rid, event_id
        full = ui.fullscreen_state.text == "full"
        if rid is not None and not state["sidebar"][full][1]:
            state["sidebar"][full][1] = True
            apply_sidebars(full)
        update_selection()

    def update_selection():
        selected = (
            chosen.head(0)
            if state["selected"] is None
            else chosen.filter(pl.col("id") == state["selected"])
        )
        selected_events = events_for(selection_events, selected)
        ui.details.text = render_details(
            selected, selected_events, state["event"], enabled(ui.summarise_motion)
        )
        ui.summarise_motion.visible = not selected.is_empty()
        hidden = (
            selected_events.head(0)
            if state["event"] is None
            else selected_events.filter(
                (pl.col("id") == state["event"])
                & ~pl.col("event_group").is_in(active_groups)
            )
        )
        selection_notice.visible = not hidden.is_empty()
        notice = ""
        if not hidden.is_empty():
            highlighted = hidden.row(0, named=True)
            notice = (
                f"<b>Selected hidden event #{highlighted['id']}</b> · "
                f"{highlighted['type']} · t={highlighted['t']:.6g} s · "
                f"Axial distance={highlighted['z'] * 1000:.6g} mm"
            )
        selection_notice.text = notice
        for update in selection_updates:
            update()

    def sidebar_event(attr, old, new):
        if new == "clear":
            set_selection(None, None)
        elif new.isdigit() and state["selected"] is not None:
            history = events.filter(pl.col("realization_id") == state["selected"])
            if int(new) in history["id"]:
                set_selection(state["selected"], int(new))
        if new:
            ui.detail_bridge.text = ""

    ui.detail_bridge.on_change("text", sidebar_event)

    def prepare_cohort():
        """Filter and number the cohort, then update its details and summary."""
        chosen = selected_cohort().with_row_index("realization_number", offset=1)
        chosen_ids = chosen["id"]
        if state["selected"] not in chosen_ids:
            state["selected"] = state["event"] = None
        chosen_events = events_for(events, chosen)
        chosen_events = chosen_events.join(
            chosen.select(pl.col("id").alias("realization_id"), "realization_number"),
            on="realization_id",
            how="left",
            maintain_order="left",
        )
        ui.counts.text = f"<b>{chosen.height} selected / {total} total realizations</b>"
        ui.counts.visible = ui.selector.visible = enabled(ui.show_pager)
        incomplete = chosen.filter(
            pl.col("fate").is_in(["Incomplete", "Ambiguous"])
        ).height
        clamped = chosen_events["z_clamped"].sum()
        ui.quality.text = "<br>".join(
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
        return chosen, chosen_events

    def sync_event_controls():
        """Keep grouping, event availability, and colors in sync with pathways."""
        ui.group_fragmentations.disabled = enabled(ui.fate_facets)
        if ui.group_fragmentations.disabled:
            ui.group_fragmentations.active = [0]
        grouped = enabled(ui.group_fragmentations)
        if grouped != state["grouped"]:
            if grouped:
                ui.event_options["fragmentation"].active = (
                    [0]
                    if any(
                        enabled(ui.event_options[name]) for name in fragmentation_names
                    )
                    else []
                )
            else:
                for name in fragmentation_names:
                    ui.event_options[name].active = (
                        [0] if enabled(ui.event_options["fragmentation"]) else []
                    )
            state["grouped"] = grouped
        selected_fates = {fate_names[i] for i in ui.fate.active}
        available_events = events_for(
            events, realizations.filter(pl.col("fate").is_in(selected_fates))
        )
        kinds = (
            list(EVENT_TYPES)
            if grouped
            else ["init", "collision", *fragmentation_names, "escape"]
        )
        event_colors = {
            kind: COLORS[kind] if kind in EVENT_TYPES else fate_colors[kind]
            for kind in kinds
        }
        for kind, widget in ui.event_options.items():
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
        ui.event_options["escape"].labels = ["Escape" if grouped else escape_name]
        ui.event_controls.children = [ui.event_options[kind] for kind in kinds]
        active_groups = {
            kind
            for kind in kinds
            if enabled(ui.event_options[kind]) and not ui.event_options[kind].disabled
        }
        return grouped, event_colors, active_groups

    def sync_display_controls(event_colors):
        """Apply coordinate restrictions and keep at least one view enabled."""
        regional = ui.mode.value in ("schematic", "equal")
        ui.own_zones.disabled = not regional
        ui.spread_x.disabled = (
            not regional or not enabled(ui.own_zones) or ui.layout.value == "radial"
        )
        ui.show_bars.disabled = not regional
        ui.schematic.disabled = not regional
        ui.guides.visible = ui.mode.value == "physical"
        ui.envelope.visible = ui.layout.value == "beeswarm"
        ui.x_markers.disabled = ui.layout.value == "beeswarm"
        marker = "×" if enabled(ui.x_markers) and not ui.x_markers.disabled else "●"
        for kind, color in event_colors.items():
            ui.event_options[kind].stylesheets[0].css = (
                f'label::before {{ content: "{marker}"; color: {color}; '
                "display: inline-block; width: 12px; margin-right: 4px; text-align: center; font-size: 16px; }"
                + CHECKBOX_CSS
            )
        if regional and state["auto_cdf"] and enabled(ui.show_bars):
            ui.show_cdf.active = []
            state["auto_cdf"] = False
        if not any(
            (
                enabled(ui.show_events),
                enabled(ui.show_cdf),
                enabled(ui.show_bars) and regional,
            )
        ):
            state["auto_cdf"] = enabled(ui.show_bars) and not regional
            ui.show_cdf.active = [0]

    def assemble_panels(chosen, chosen_events, grouped, event_colors, active_groups):
        """Position the cohort and assemble shared-axis plots and selection notices."""
        nonlocal selection_events
        chosen_events = chosen_events.with_columns(
            pl.when(pl.col("type") == "fragmentation")
            .then(pl.lit("fragmentation") if grouped else pl.col("event_pathway"))
            .otherwise(pl.col("type"))
            .alias("event_group")
        )
        selection_events = chosen_events
        mapped_regions = coordinate_regions(regions, ui.mode.value)
        spatial = ui.mode.value != "time"
        regional = ui.mode.value in ("schematic", "equal")
        dedicated_zones = regional and enabled(ui.own_zones)
        spread_terminals = enabled(ui.spread_x) and not ui.spread_x.disabled
        positioned = position_events(
            chosen_events, mapped_regions, ui.mode.value, dedicated_zones
        )
        slot = escape_slot(mapped_regions)
        use_x_markers = enabled(ui.x_markers) and not ui.x_markers.disabled
        rows = (
            [
                (chosen.filter(pl.col("fate") == name), name)
                for name in fate_names
                if name in {fate_names[i] for i in ui.fate.active}
            ]
            if enabled(ui.fate_facets)
            else [(chosen, "All selected pathways")]
        )
        max_position = positioned["position"].max()
        if spatial:
            left_edge = (
                initial_slot(mapped_regions)[0]
                if dedicated_zones
                else mapped_regions["left"][0]
            )
            right_edge = (
                slot[1]
                if dedicated_zones or (regional and enabled(ui.show_bars))
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
        screen_width = int(ui.viewport_width.text.split(",")[0])
        available_width = screen_width - ui.left.width - ui.right.width
        plot_height = event_plot_height()
        margin = (right_edge - left_edge) * 0.015
        shared_x = Range1d(
            left_edge - margin if ui.layout.value == "beeswarm" else left_edge,
            right_edge + margin,
        )
        panels = []
        if enabled(ui.schematic) and spatial and regional and not chosen.is_empty():
            panels.append(
                _schematic_plot(
                    mapped_regions,
                    shared_x,
                    available_width,
                    initial=dedicated_zones,
                    escaped=dedicated_zones or enabled(ui.show_bars),
                )
            )

        def selection_callback(source):
            def on_selected(attr, old, new):
                if new and not state["updating"]:
                    i = new[0]
                    set_selection(source.data["realization"][i], source.data["id"][i])
                    source.selected.indices = []

            return on_selected

        for members, row_name in rows:
            if members.is_empty() and chosen.is_empty():
                continue
            member_events = events_for(positioned, members)
            plots = []
            if enabled(ui.show_events):
                visible = layout_events(
                    member_events.filter(pl.col("event_group").is_in(active_groups)),
                    ui.layout.value,
                )
                plots.append(
                    _event_plot(
                        visible,
                        shared_x,
                        plot_height,
                        ui.layout.value,
                        use_x_markers,
                        mapped_regions,
                        enabled(ui.guides) and ui.mode.value == "physical",
                        enabled(ui.envelope) and ui.layout.value == "beeswarm",
                        state,
                        selection_callback,
                        event_colors,
                        (
                            members["realization_number"].min() or 1,
                            members["realization_number"].max() or 1,
                        ),
                        spread_terminals=spread_terminals,
                        selection_updates=selection_updates,
                        doc=doc,
                    )
                )
            if enabled(ui.show_cdf) or (enabled(ui.show_bars) and regional):
                if enabled(ui.show_cdf):
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
                    row_colors = {
                        name: fate_colors[name]
                        for name in fate_names
                        if name in members["fate"]
                    }
                    plots.append(
                        _cdf_plot(
                            cumulative_areas(
                                members,
                                cumulative_events,
                                left_edge,
                                right_edge,
                                row_colors,
                            ),
                            shared_x,
                            row_colors,
                        )
                    )
                if enabled(ui.show_bars) and regional:
                    bars = regional_fractions(members, member_events, mapped_regions)
                    plots.append(_bar_plot(bars, shared_x))
            if enabled(ui.fate_facets) and plots:
                plots[0].title = f"{row_name} (N={members.height})"
            for plot in plots:
                plot.min_border_top = 0
                plot.min_border_bottom = 0
                if regional:
                    centers = [
                        (lo + hi) / 2
                        for lo, hi in mapped_regions.select("left", "right").iter_rows()
                    ]
                    plot.xaxis.ticker = FixedTicker(ticks=centers)
                    plot.xaxis.major_label_overrides = {
                        center: str(index)
                        for index, center in enumerate(centers, start=1)
                    }
                    plot.xaxis.major_tick_line_color = None
                    plot.xaxis.minor_tick_line_color = None
                    plot.xgrid.ticker = FixedTicker(
                        ticks=[mapped_regions["left"][0], *mapped_regions["right"]]
                    )
                    plot.xgrid.grid_line_color = "#94a3b8"
                    plot.xgrid.grid_line_alpha = 0.3
                prepare_toolbar(plot, ui)
            if plots:
                for upper in plots[:-1]:
                    upper.xaxis.visible = False
                plots[-1].xaxis.axis_label = (
                    "Elapsed time (s)"
                    if not spatial
                    else "Axial distance (mm)"
                    if ui.mode.value == "physical"
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
        ui.center_plots.children = [selection_notice, *panels]

    def render():
        nonlocal chosen, active_groups
        if state["updating"]:
            return
        state["updating"] = True
        try:
            selection_updates.clear()
            chosen, chosen_events = prepare_cohort()
            grouped, event_colors, active_groups = sync_event_controls()
            sync_display_controls(event_colors)
            assemble_panels(chosen, chosen_events, grouped, event_colors, active_groups)
            update_selection()
        finally:
            state["updating"] = False

    def control_changed(attr, old, new):
        render()

    def cdf_changed(attr, old, new):
        if not state["updating"]:
            state["auto_cdf"] = False
        render()

    for widget in ui.event_options.values():
        widget.on_change("active", control_changed)
    for widget, property_name in [
        (ui.fate, "active"),
        (ui.fate_facets, "active"),
        (ui.group_fragmentations, "active"),
        (ui.own_zones, "active"),
        (ui.spread_x, "active"),
        (ui.mode, "value"),
        (ui.layout, "value"),
        (ui.envelope, "active"),
        (ui.schematic, "active"),
        (ui.guides, "active"),
        (ui.show_events, "active"),
        (ui.x_markers, "active"),
        (ui.show_bars, "active"),
        (ui.show_pager, "active"),
        (ui.selector, "value"),
    ]:
        widget.on_change(property_name, control_changed)
    ui.summarise_motion.on_change("active", lambda attr, old, new: update_selection())
    ui.show_cdf.on_change("active", cdf_changed)
    render()
