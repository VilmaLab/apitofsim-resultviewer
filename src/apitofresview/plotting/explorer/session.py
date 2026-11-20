"""Cohort selection, controls, and rendering for one Explorer session."""

from functools import lru_cache
from html import escape
import json
from math import isfinite

import polars as pl
from apitofsim.plotting.events import NoRecordedEventsError
from bokeh.events import DocumentReady
from bokeh.layouts import column
from bokeh.models import CustomJS, Div, FixedTicker, Range1d

from .aggregation import cumulative_areas, regional_fractions
from .browser import ON_FRAME
from .controls import (
    CHECKBOX_CSS,
    SIDEBAR_BREAKPOINT,
    apply_layout,
    create_controls,
    enabled,
    label_facets,
    group,
    prepare_toolbar,
    realization_range_bridge,
)
from .data import (
    EVENT_TYPES,
    CohortHistories,
    events_for,
    load_histories,
    select_realizations,
)
from .data import load_overview as load_data
from .details import history_page, render_choices, render_details
from .escapes import EscapeGroups
from .layout import (
    coordinate_regions,
    escape_slot,
    initial_slot,
    layout_events,
    position_events,
    schematic_widths,
)
from .plot import (
    COLORS,
    PackingQueue,
    _bar_plot,
    _event_plot,
    _fraction_plot,
    _pathway_colors,
    _schematic_plot,
    update_range,
)
from .population import population_areas, species_catalog
from .rates import RateViews
from .tree import ExecutionTree, install_tree_controls


MAX_COLLISIONS = 100_000


def build_document(db, doc, experiment: int, cluster: int, *, layout="viewport"):
    """Build one Bokeh session; every control uses the same cohort and X mapping."""
    if layout not in ("viewport", "page", "contained"):
        raise ValueError(f"Unknown Explorer layout: {layout!r}")
    if experiment is None or cluster is None:
        doc.add_root(Div(text="Select an experiment run and cluster."))
        return
    try:
        realizations, events, pathways, regions = load_data(db, experiment, cluster)
    except NoRecordedEventsError as exc:
        doc.add_root(Div(text=escape(str(exc))))
        return
    except Exception as exc:
        # Non-realization databases and older configs have no Explorer data.
        doc.add_root(Div(text=f"Explorer data unavailable: {escape(str(exc))}"))
        return

    pathway_mode = "simulated_pathway_id" in realizations.columns
    tree = (
        ExecutionTree.load(db, experiment, cluster, pathways)
        if "current_cluster_id" in events.columns
        else None
    )
    if tree and not tree.nodes:
        doc.add_root(
            Div(text="No recorded cluster-tree executions for this selection.")
        )
        return
    if tree:
        tree.incomplete = set(realizations["fate"])
        realizations = realizations.with_columns(
            pl.col("fate")
            .replace_strict(tree.leaf_labels(), default="Incomplete")
            .alias("fate_label")
        )
    # Number the full population before applying pathway and range filters.
    realizations = (
        realizations.with_columns(
            pl.col("id")
            .cum_count()
            .over("simulated_pathway_id")
            .alias("realization_number")
        )
        if pathway_mode
        else realizations.with_row_index("realization_number", offset=1)
    )
    sparse = "collisions" in realizations.columns
    if not sparse:
        # Count once per session; subsequent budget checks only touch realizations.
        realizations = realizations.join(
            events.group_by("realization_id").agg(
                (pl.col("type") == "collision")
                .cast(pl.UInt64)
                .sum()
                .alias("collisions")
            ),
            left_on="id",
            right_on="realization_id",
            how="left",
            maintain_order="left",
        ).with_columns(pl.col("collisions").fill_null(0))
    collision_realizations = int(
        (
            realizations["collisions"].sort(descending=True).cum_sum() <= MAX_COLLISIONS
        ).sum()
    )
    fate_names = realizations["fate"].unique().sort().to_list()
    if tree:
        fate_names = list(tree.leaf_labels())
        if "Incomplete" not in realizations["fate"]:
            fate_names.remove("Incomplete")
    fate_counts = dict(realizations.group_by("fate").len().iter_rows())
    fate_counts = {name: fate_counts.get(name, 0) for name in fate_names}
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
    escapes = EscapeGroups(
        events, cluster, tree, root_name=escape_name.split(" → ")[-1]
    )

    def annotate_events(frame):
        return (
            escapes.annotate(frame)
            .join(
                pathways.drop("charged_products", strict=False),
                on="pathway_id",
                how="left",
                maintain_order="left",
            )
            .rename({"pathway": "event_pathway"})
            .with_columns(pl.col("event_pathway").fill_null("Unknown fragmentation"))
        )

    def load_annotated_histories(realization_ids, event_ids=None):
        return annotate_events(
            load_histories(
                db,
                experiment,
                cluster,
                regions,
                realization_ids,
                tree=bool(tree),
                ids=event_ids,
            )
        )

    histories = lru_cache(maxsize=2)(load_annotated_histories)
    cohort_histories = CohortHistories(load_annotated_histories)
    events = annotate_events(events)
    fragmentation_names = (
        events.filter(pl.col("type") == "fragmentation")["event_pathway"]
        .unique()
        .sort()
        .to_list()
    )
    fate_colors = _pathway_colors(
        sorted(
            set(fragmentation_names) | ({"Incomplete"} if tree else set(fate_names))
        ),
        "Escape" if tree else escape_name,
    )
    if tree:
        for key, count in fate_counts.items():
            if not count:
                continue
            if ":stop:" in key:
                result, _, transition = key.split(":")
                pathway = tree.transitions[int(result), int(transition)][1]
                fate_colors[key] = fate_colors[tree.pathways[pathway]]
            else:
                fate_colors[key] = (
                    fate_colors["Incomplete"]
                    if "incomplete" in key.lower()
                    else COLORS["escape"]
                )
    fate_colors.update(escapes.route_colors())
    facet_names = (
        dict(pathways.select("pathway", "daughter").iter_rows())
        if "daughter" in pathways.columns
        else {}
    )
    facet_names = {name: f"→ {daughter}" for name, daughter in facet_names.items()}
    for name in fate_names:
        if name not in facet_names and " → " in name and " + " not in name:
            facet_names[name] = f"→ {name.split(' → ')[-1]}"

    simulated_pathways = (
        pathways.filter(pl.col("simulated")) if pathway_mode else pathways
    )
    pathway_ids = simulated_pathways["pathway_id"].to_list() if pathway_mode else []
    selector_names = (
        simulated_pathways["pathway"].to_list() if pathway_mode else fate_names
    )
    selector_counts = (
        {
            name: realizations.filter(pl.col("simulated_pathway_id") == pid).height
            for pid, name in zip(pathway_ids, selector_names, strict=True)
        }
        if pathway_mode
        else fate_counts
    )
    if pathway_mode:
        fate_colors.update(
            _pathway_colors(
                sorted(set(selector_names) | set(fragmentation_names)), escape_name
            )
        )
    state = {
        "selected": None,
        "event": None,
        "candidates": {},
        "detail_page": None,
        "updating": False,
        "grouped": True,
        "grouped_escapes": True,
        "escape_keys": escapes.keys(),
        "separated": False,
        "plot_event": None,
        "hidden_collision": None,
        "auto_cdf": False,
        "syncing_sidebar": False,
        "sidebar": {False: [True, False], True: [False, False]},
        "details_layout": {False: "sidebar", True: "sidebar"},
        "widths": schematic_widths("chambers"),
        "setting_sizing": False,
    }

    species_labels, charged_products, species_colors = species_catalog(
        cluster, realizations, pathways, events, fate_colors
    )
    total = realizations.height
    ui = create_controls(
        doc,
        selector_names,
        selector_counts,
        fragmentation_names,
        fate_colors,
        max(realizations["realization_number"].max() or 1, 1)
        if pathway_mode
        else total,
        render_details(realizations.head(0), events.head(0), None),
        layout=layout,
        pathway_mode=pathway_mode,
        tree_mode=bool(tree),
        escape_options=escapes.labels,
    )

    ui.distinguish_escapees.visible = bool(tree and escapes.ambiguous)
    scope, tree_labels, full_tree, show_full_tree, recolor_tree = (
        install_tree_controls(doc, ui, tree, fate_names, fate_counts, fate_colors)
        if tree
        else (None, {}, None, None, None)
    )

    event_groups = {}
    if tree:
        for key in fragmentation_names:
            ui.event_options[key].labels = [facet_names.get(key, key)]
            ui.event_options[key].html_attributes = {"title": key}
        for key in escapes.labels:
            ui.event_options[key].labels = [escapes.short_labels[key]]
            ui.event_options[key].html_attributes = {"title": escapes.labels[key]}
        event_groups = {
            "fragmentation": group("Fragmentation", column()),
            "escape": group("Escapes", column()),
        }

    if not tree:
        label_facets(doc, ui.fate, facet_names)
    else:
        facet_names = tree.short_labels()

    if pathway_mode:
        ui.selector.title = "Realization numbers per pathway (inclusive)"
        ui.selector.disabled = realizations.is_empty()
    applied_range = tuple(ui.selector.value)
    selection_updates = []
    event_panels = {}
    display_colors = {}
    summary_panels = {}
    packing = PackingQueue(doc)
    shared_x = None
    shared_y = None
    x_mode = None
    y_mode = None
    schematic = None
    cohort_key = None
    chosen_events = events.head(0)
    positioned_key = population_key = None
    positioned = population_events = None
    chosen = realizations.head(0)
    selection_events = events.head(0)
    selection_has_histories = not sparse
    details_key = None
    active_groups = set()
    rates = RateViews(
        db,
        experiment,
        cluster,
        pathways,
        fate_names,
        fate_colors,
        ui,
        pathway_mode=pathway_mode,
        tree=tree,
    )
    rate_selection = (chosen, None)
    collision_coordinates = None
    collision_panels = []
    sidebar_layout = None
    align_notes = CustomJS(
        args=dict(notes=[ui.counts, ui.quality]),
        code=ON_FRAME
        + """
        const parent = Bokeh.index.find_one(cb_obj)?.parent?.model;
        const panels = cb_obj.name === 'explorer_plots' ? cb_obj :
            parent?.name === 'explorer_plots' ? parent :
            parent?.children?.find(model => model.name === 'explorer_plots');
        if (!panels) return;
        onFrame(panels, 'explorerNotesFrame', () => {
            const plot = panels.children.find(p => p.inner_width > 0);
            const view = plot && Bokeh.index.find_one(plot);
            if (!view?.canvas_view) return;
            const axis = view.canvas_view.events_el.getBoundingClientRect().left + view.frame.bbox.left;
            const insets = notes.flatMap(note => {
                const el = Bokeh.index.find_one(note)?.el;
                if (!el?.checkVisibility()) return [];
                return [[el, `${Math.max(0, axis - el.getBoundingClientRect().left)}px`]];
            });
            for (const [el, inset] of insets)
                if (el.style.paddingLeft !== inset) el.style.paddingLeft = inset;
        });
        """,
    )
    ui.center_plots.js_on_change("children", align_notes)
    for note in (ui.counts, ui.quality):
        note.js_on_change("visible", align_notes)
    drag_bridge = Div(text="", visible=False, name="schematic_drag_bridge")
    ui.frame.children.append(drag_bridge)

    def event_plot_height():
        screen_height = int(ui.viewport_width.text.split(",")[1])
        regional = ui.mode.value == "schematic"
        return (
            max(
                440,
                min(
                    760,
                    screen_height
                    - 175
                    - (175 if enabled(ui.show_cdf) else 0)
                    - (175 if enabled(ui.show_population) else 0)
                    - (155 if enabled(ui.show_bars) and regional else 0),
                ),
            )
            if ui.fullscreen_state.text == "full"
            else 440
        )

    def resize_plots():
        height = event_plot_height()
        for panel in ui.center_plots.children:
            if panel.name == "events":
                panel.height = height

    def apply_sidebars(full):
        nonlocal sidebar_layout
        left_open, right_open = state["sidebar"][full]
        mode = state["details_layout"][full]
        if mode == "sidebar":
            ui.show_rates.active = []
        bottom = right_open and mode == "bottom"
        narrow = int(ui.viewport_width.text.split(",")[0]) < SIDEBAR_BREAKPOINT
        if narrow and left_open and right_open and not bottom:
            left_open = False
            state["sidebar"][full][0] = False
        geometry = (full, left_open, right_open, mode)
        if geometry == sidebar_layout:
            # Clicking the selected placement toggles its button off locally.
            if not ui.detail_layouts[mode].active:
                state["syncing_sidebar"] = True
                try:
                    ui.detail_layouts[mode].active = True
                finally:
                    state["syncing_sidebar"] = False
            resize_plots()
            return
        sidebar_layout = geometry
        state["syncing_sidebar"] = True
        try:
            ui.left_toggle.active, ui.right_toggle.active = left_open, right_open
            for placement, button in ui.detail_layouts.items():
                button.active = placement == mode
                button.visible = right_open
        finally:
            state["syncing_sidebar"] = False
        ui.left_controls.visible, ui.right_controls.visible = left_open, right_open
        ui.left.width = 205 if left_open else 38
        split = right_open and mode != "sidebar"
        ui.right.width = None if split else 300 if right_open else 38
        ui.right.sizing_mode = "stretch_width" if split else "fixed"
        apply_layout(ui, layout, full=full, bottom=bottom)
        ui.left_toggle.width, ui.right_toggle.width = (
            (100 if left_open else 38),
            (110 if right_open else 38),
        )
        ui.left_toggle.label = "« Controls" if left_open else "»"
        ui.right_toggle.label = (
            ("Hide details ↓" if bottom else "Hide details »") if right_open else "☷"
        )
        resize_plots()

    def sidebar_changed(index, active):
        if state["syncing_sidebar"]:
            return
        full = ui.fullscreen_state.text == "full"
        state["sidebar"][full][index] = active
        if (
            index == 0 and active and state["details_layout"][full] != "bottom"
        ) or index == 1:
            state["details_layout"][full] = "sidebar"
        # On narrow screens, opening controls should keep them visible.
        if (
            index == 0
            and active
            and state["details_layout"][full] != "bottom"
            and int(ui.viewport_width.text.split(",")[0]) < SIDEBAR_BREAKPOINT
        ):
            state["sidebar"][full][1] = False
        apply_sidebars(full)

    ui.left_toggle.on_change("active", lambda attr, old, new: sidebar_changed(0, new))
    ui.right_toggle.on_change("active", lambda attr, old, new: sidebar_changed(1, new))

    def layout_changed(mode):
        if state["syncing_sidebar"]:
            return
        full = ui.fullscreen_state.text == "full"
        state["details_layout"][full] = mode
        if mode == "right":
            state["sidebar"][full][0] = False
        apply_sidebars(full)

    for mode, button in ui.detail_layouts.items():
        button.on_change(
            "active", lambda attr, old, new, mode=mode: layout_changed(mode)
        )

    def full_changed(attr, old, new):
        apply_sidebars(new == "full")

    ui.fullscreen_state.on_change("text", full_changed)
    ui.viewport_width.on_change(
        "text",
        lambda attr, old, new: apply_sidebars(ui.fullscreen_state.text == "full"),
    )

    def realization_bounds():
        return applied_range if enabled(ui.show_pager) else (1, int(ui.selector.end))

    def selected_cohort():
        lo, hi = realization_bounds()
        if pathway_mode:
            return realizations.filter(
                pl.col("simulated_pathway_id").is_in(
                    [pathway_ids[i] for i in ui.fate.active]
                )
                & pl.col("realization_number").is_between(lo, hi)
            )
        chosen_fates = {fate_names[i] for i in ui.fate.active}
        return select_realizations(realizations, (lo, hi), chosen_fates)

    def set_selection(rid, event_id):
        state["selected"], state["event"] = rid, event_id
        state["candidates"] = {}
        state["detail_page"] = None
        update_selection(refit_collision=True)
        if rid is not None:
            open_details()

    def set_candidates(candidates):
        state["selected"] = state["event"] = None
        state["candidates"] = candidates
        update_selection()
        open_details()

    def open_details():
        full = ui.fullscreen_state.text == "full"
        if not state["sidebar"][full][1]:
            state["sidebar"][full][1] = True
            with doc.models.freeze():
                apply_sidebars(full)

    def selected_history():
        selected = (
            chosen.head(0)
            if state["selected"] is None
            else chosen.filter(pl.col("id") == state["selected"])
        )
        if selection_has_histories or selected.is_empty():
            selected_events = events_for(selection_events, selected)
        else:
            selected_events = histories(tuple(selected["id"])).join(
                selected.select(
                    pl.col("id").alias("realization_id"), "realization_number"
                ),
                on="realization_id",
                how="left",
                maintain_order="left",
            )
            selected_events = group_events(
                selected_events, state["grouped"] and not pathway_mode
            )
        return selected, selected_events

    def show_history_page(selected, selected_events):
        nonlocal details_key
        event = state["event"]
        window = history_page(selected_events, event, state["detail_page"])
        radial = enabled(ui.summarise_motion)
        energy = enabled(ui.summarise_energy)
        key = (state["selected"], window[0], radial, energy)
        if details_key != key or doc.session_context is None:
            ui.details.text = render_details(
                selected, selected_events, event, radial, energy, window=window
            )
            details_key = key
        ui.detail_selection.data = (
            dict(event=[], pair=[])
            if event is None
            else dict(event=[str(event)], pair=[str(state["plot_event"])])
        )

    def update_details(selected, selected_events):
        nonlocal details_key
        if state["candidates"]:
            candidates = selection_events.filter(
                pl.col("id").is_in(state["candidates"])
            )
            if not selection_has_histories:
                candidates = (
                    histories(
                        tuple(candidates["realization_id"].unique()),
                        tuple(state["candidates"]),
                    )
                    .filter(pl.col("id").is_in(state["candidates"]))
                    .join(
                        chosen.select(
                            pl.col("id").alias("realization_id"), "realization_number"
                        ),
                        on="realization_id",
                        how="left",
                        maintain_order="left",
                    )
                )
            ui.details.text = render_choices(
                candidates, enabled(ui.summarise_motion), enabled(ui.summarise_energy)
            )
            details_key = None
            ui.detail_selection.data = dict(event=[], pair=[])
        else:
            show_history_page(selected, selected_events)

    def update_rates(*, refit_collision=False):
        ui.rate_controls.visible = ui.show_rates.visible and enabled(ui.show_rates)
        ui.rate_separator.visible = ui.rate_controls.visible
        ui.detail_log_rates.disabled = not ui.rate_controls.visible
        ui.detail_show_readoffs.disabled = not ui.rate_controls.visible
        rates.update(*rate_selection, refit_collision=refit_collision)
        if not state["updating"] and current_collision_panels() != collision_panels:
            sync_collision_panels()

    def current_collision_panels():
        return list(rates.collision.children) if rates.collision.visible else []

    def update_selection(*, refit_collision=False):
        nonlocal rate_selection
        selected, selected_events = selected_history()
        selected_event = (
            selected_events.filter(pl.col("id") == state["event"])
            if state["event"] is not None
            else selected_events.head(0)
        )
        state["plot_event"] = (
            selected_event["collision_id"][0]
            or (
                selected_event["fragmentation_id"][0]
                if "fragmentation_id" in selected_event.columns
                else None
            )
            or state["event"]
            if not selected_event.is_empty()
            else None
        )
        update_details(selected, selected_events)
        ui.summarise_motion.visible = not selected.is_empty() or bool(
            state["candidates"]
        )
        ui.summarise_energy.visible = ui.summarise_motion.visible
        ui.show_rates.visible = not selected.is_empty()
        rate_selection = (
            selected,
            selected_event.row(0, named=True)
            if not selected_event.is_empty()
            else None,
        )
        update_rates(refit_collision=refit_collision)
        state["hidden_collision"] = None
        if state["plot_event"] is not None:
            hidden = selected_events.filter(
                (pl.col("id") == state["plot_event"])
                & (pl.col("type") == "collision")
                & ~pl.col("event_group").is_in(active_groups)
            )
            if not hidden.is_empty():
                state["hidden_collision"] = layout_events(
                    position_events(hidden, collision_coordinates[0], ui.mode.value),
                    ui.layout.value,
                ).row(0, named=True)
        for update in selection_updates:
            update()

    def sidebar_event(attr, old, new):
        if not new:
            return
        ui.detail_bridge.text = ""
        try:
            action, event = json.loads(new)
        except ValueError, TypeError:
            return
        if action == "clear":
            set_selection(None, None)
        elif action == "choose":
            for event_id, rid in state["candidates"].items():
                if str(event_id) == event:
                    set_selection(rid, event_id)
                    break
        elif state["selected"] is not None:
            if action == "page" and type(event) is int:
                state["detail_page"] = event
                show_history_page(*selected_history())
            elif action == "inspect":
                _, history = selected_history()
                for event_id in history["id"]:
                    if str(event_id) == event:
                        set_selection(state["selected"], event_id)
                        break

    ui.detail_bridge.on_change("text", sidebar_event)

    def prepare_cohort():
        """Filter the cohort, then update its details and summary."""
        nonlocal selection_has_histories
        chosen = selected_cohort()
        chosen_ids = chosen["id"]
        if state["selected"] not in chosen_ids:
            state["selected"] = state["event"] = None
            state["detail_page"] = None
        selection_has_histories = not sparse
        if (
            sparse
            and not chosen.is_empty()
            and chosen["collisions"].sum() <= MAX_COLLISIONS
            and (
                enabled(ui.event_options["collision"])
                or enabled(ui.event_options["rejected_collision"])
            )
        ):
            chosen_events = cohort_histories(tuple(chosen["id"]))
            selection_has_histories = True
        else:
            chosen_events = events_for(events, chosen)
        chosen_events = chosen_events.join(
            chosen.select(
                pl.col("id").alias("realization_id"),
                "realization_number",
                *(["simulated_pathway"] if pathway_mode else []),
            ),
            on="realization_id",
            how="left",
            maintain_order="left",
        )
        ui.counts.text = f"<b>{chosen.height} selected / {total} total realizations</b>"
        ui.counts.visible = not pathway_mode and enabled(ui.show_pager)
        ui.selector.visible = enabled(ui.show_pager)
        incomplete = chosen.filter(
            pl.col("terminal_event_id").is_null()
            if tree
            else pl.col("fate").is_in(["Incomplete", "Ambiguous"])
        ).height
        clamped = (
            chosen["clamped_events"].sum()
            if sparse
            else chosen_events["z_clamped"].sum()
        )
        collision_count = chosen["collisions"].sum()
        collision_notice = ""
        if collision_count > MAX_COLLISIONS:
            collision_notice = (
                f"Collisions hidden: {collision_count:,} exceeds the {MAX_COLLISIONS:,} "
                "collision limit. "
            )
            collision_notice += (
                "Use the realizations pager to view no more than "
                f"{collision_realizations:,} realization{'s' if collision_realizations != 1 else ''} "
                "in total to enable the collision checkbox."
                if collision_realizations
                else "Use the realizations pager to select histories with fewer collisions; "
                "a single realization can exceed the limit."
            )
        ui.quality.text = "<br>".join(
            message
            for message in (
                collision_notice,
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
        ui.group_fragmentations.disabled = enabled(ui.fate_facets) and not tree
        if ui.group_fragmentations.disabled:
            ui.group_fragmentations.active = [0]
        grouped = enabled(ui.group_fragmentations) and not pathway_mode
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
        escape_keys = escapes.keys(enabled(ui.distinguish_escapees))
        grouped_escapes = enabled(ui.group_escapes)
        if escape_keys != state["escape_keys"]:
            old_keys = state["escape_keys"]
            for key in escape_keys:
                related = [
                    old
                    for old in old_keys
                    if escapes.route_daughters.get(old, old)
                    == escapes.route_daughters.get(key, key)
                ]
                ui.event_options[key].active = (
                    [0]
                    if any(enabled(ui.event_options[old]) for old in related)
                    else []
                )
            state["escape_keys"] = escape_keys
        if grouped_escapes != state["grouped_escapes"]:
            if grouped_escapes:
                ui.event_options["escape"].active = (
                    [0]
                    if any(enabled(ui.event_options[key]) for key in escape_keys)
                    else []
                )
            else:
                for key in escape_keys:
                    ui.event_options[key].active = list(
                        ui.event_options["escape"].active
                    )
            state["grouped_escapes"] = grouped_escapes
        separated = enabled(ui.separate_rejected)
        if separated and not state["separated"]:
            ui.event_options["rejected_collision"].active = list(
                ui.event_options["collision"].active
            )
        state["separated"] = separated
        ui.event_options["collision"].labels = [
            "Accepted collisions" if separated else "Collision"
        ]
        selected_fates = (
            set(chosen["fate"])
            if pathway_mode
            else {fate_names[i] for i in ui.fate.active}
        )
        available_members = (
            chosen
            if pathway_mode
            else realizations.filter(pl.col("fate").is_in(selected_fates))
        )
        available_events = events_for(events, available_members)
        kinds = [
            "init",
            "collision",
            *(["fragmentation"] if grouped else fragmentation_names),
            *(["escape"] if grouped_escapes else escape_keys),
        ]
        if separated:
            kinds.insert(kinds.index("collision") + 1, "rejected_collision")
        event_colors = {
            kind: COLORS[kind]
            if kind in COLORS
            else escapes.colors[kind]
            if kind in escapes.colors
            else fate_colors[kind]
            for kind in kinds
        }
        collisions_limited = chosen["collisions"].sum() > MAX_COLLISIONS
        for kind, widget in ui.event_options.items():
            if kind == "escape":
                available = (
                    "escape" in available_events["type"]
                    if pathway_mode or tree
                    else escape_name in selected_fates
                )
            elif kind in escapes.labels:
                column_name = (
                    "escape_route"
                    if enabled(ui.distinguish_escapees)
                    else "escape_daughter"
                )
                available = not available_events.filter(
                    pl.col(column_name) == kind
                ).is_empty()
            elif kind in ("collision", "rejected_collision"):
                rejected = (
                    pl.col("accepted").eq(False).fill_null(False)
                    if "accepted" in available_events.columns
                    else pl.lit(False)
                )
                matches = pl.col("type") == "collision"
                if kind == "rejected_collision":
                    matches &= rejected
                elif separated:
                    matches &= ~rejected
                if sparse:
                    rejected_count = available_members["rejected_collisions"].sum()
                    count = available_members["collisions"].sum()
                    if kind == "rejected_collision":
                        count = rejected_count
                    elif separated:
                        count -= rejected_count
                    available = count > 0
                else:
                    available = not available_events.filter(matches).is_empty()
            elif kind in EVENT_TYPES:
                available = not available_events.filter(
                    pl.col("type") == kind
                ).is_empty()
            else:
                available = (
                    pathway_mode
                    or tree
                    or kind not in fate_names
                    or kind in selected_fates
                ) and kind in available_events["event_pathway"]
            widget.disabled = not available or (
                kind in ("collision", "rejected_collision") and collisions_limited
            )
        ui.event_options["escape"].labels = ["Escape"]
        children = []
        for kind in kinds:
            family = (
                "fragmentation"
                if kind in fragmentation_names
                else "escape"
                if kind in escape_keys
                else None
            )
            if (
                tree
                and family
                and not (grouped if family == "fragmentation" else grouped_escapes)
            ):
                container = event_groups[family]
                if container not in children:
                    keys = (
                        fragmentation_names
                        if family == "fragmentation"
                        else escape_keys
                    )
                    container.child.children = [ui.event_options[key] for key in keys]
                    children.append(container)
            else:
                children.append(ui.event_options[kind])
        ui.event_controls.children = children
        active_groups = {
            kind
            for kind in kinds
            if enabled(ui.event_options[kind]) and not ui.event_options[kind].disabled
        }
        return grouped, event_colors, active_groups

    def sync_display_controls(event_colors):
        """Apply coordinate restrictions and keep at least one view enabled."""
        regional = ui.mode.value == "schematic"
        ui.sizing.visible = regional
        ui.own_zones.disabled = not regional
        ui.spread_x.disabled = (
            not regional or not enabled(ui.own_zones) or ui.layout.value == "radial"
        )
        ui.show_bars.disabled = not regional
        ui.schematic.disabled = not regional
        ui.guides.visible = ui.mode.value == "physical"
        ui.envelope.visible = ui.layout.value == "beeswarm"
        ui.x_markers.disabled = ui.layout.value == "beeswarm"
        sync_event_markers(event_colors)
        if regional and state["auto_cdf"] and enabled(ui.show_bars):
            ui.show_cdf.active = []
            state["auto_cdf"] = False
        if not any(
            (
                enabled(ui.show_events),
                enabled(ui.show_cdf),
                enabled(ui.show_population),
                enabled(ui.show_bars) and regional,
            )
        ):
            state["auto_cdf"] = enabled(ui.show_bars) and not regional
            ui.show_cdf.active = [0]

    def sync_event_markers(event_colors):
        nonlocal display_colors
        display_colors = event_colors
        marker = "×" if enabled(ui.x_markers) and not ui.x_markers.disabled else "●"
        for kind, color in event_colors.items():
            ui.event_options[kind].stylesheets[0].css = (
                f'label::before {{ content: "{marker}"; color: {color}; '
                "display: inline-block; width: 12px; margin-right: 4px; text-align: center; font-size: 16px; }"
                + CHECKBOX_CSS
            )

    def group_events(frame, grouped):
        rejected = (
            pl.col("accepted").eq(False).fill_null(False)
            if "accepted" in frame.columns
            else pl.lit(False)
        )
        return frame.with_columns(
            pl.when(
                pl.lit(enabled(ui.separate_rejected))
                & (pl.col("type") == "collision")
                & rejected
            )
            .then(pl.lit("rejected_collision"))
            .when(pl.col("type") == "fragmentation")
            .then(pl.lit("fragmentation") if grouped else pl.col("event_pathway"))
            .when(pl.col("type") == "escape")
            .then(
                pl.lit("escape")
                if enabled(ui.group_escapes)
                else pl.col(
                    "escape_route"
                    if enabled(ui.distinguish_escapees)
                    else "escape_daughter"
                )
            )
            .otherwise(pl.col("type"))
            .alias("event_group"),
            pl.col(
                "escape_route"
                if enabled(ui.distinguish_escapees)
                else "escape_daughter"
            ).alias("event_escape"),
        )

    def assemble_panels(chosen, chosen_events, grouped, event_colors, active_groups):
        """Position the cohort and assemble shared-axis plots."""
        nonlocal selection_events, collision_coordinates, collision_panels
        nonlocal shared_x, shared_y, x_mode, y_mode, schematic
        nonlocal event_panels, summary_panels
        nonlocal positioned_key, population_key, positioned, population_events
        next_events, next_summaries = {}, {}

        def summary(key, signature, build):
            cached = summary_panels.get(key)
            if cached is None or cached[0] != signature:
                cached = (signature, build(cached[1] if cached is not None else None))
            next_summaries[key] = cached
            return cached[1]

        def panel_notice(key, text, **attributes):
            def update(previous):
                if previous is None:
                    return Div(text=text, **attributes)
                previous.text = text
                return previous

            return summary(("notice", key), text, update)

        mode, y_layout = ui.mode.value, ui.layout.value
        grouped_key = (
            cohort_key,
            grouped,
            state["grouped_escapes"],
            state["separated"],
            enabled(ui.distinguish_escapees),
        )
        mapped_regions = coordinate_regions(
            regions,
            ui.mode.value,
            state["widths"] if ui.mode.value == "schematic" else None,
        )
        spatial = ui.mode.value != "time"
        regional = ui.mode.value == "schematic"
        dedicated_zones = regional and enabled(ui.own_zones)
        spread_terminals = enabled(ui.spread_x) and not ui.spread_x.disabled
        mapping_key = (mode, state["widths"] if regional else (), dedicated_zones)
        position_key = (grouped_key, mapping_key)
        if position_key != positioned_key:
            selection_events = group_events(chosen_events, grouped)
            positioned = position_events(
                selection_events.filter(
                    ~pl.col("type").is_in(["postcollision", "postfragmentation"])
                ),
                mapped_regions,
                mode,
                dedicated_zones,
            )
            positioned_key = position_key
        if enabled(ui.show_population):
            key = (cohort_key, mapping_key)
            if population_key != key:
                population_events = position_events(
                    chosen_events.with_columns(
                        pl.col("population_z").clip(lower_bound=0).alias("z"),
                        pl.col("population_t").alias("t"),
                    )
                    if "population_z" in chosen_events.columns
                    else chosen_events,
                    mapped_regions,
                    mode,
                    terminal_slots=False,
                )
                population_key = key
        else:
            population_events = population_key = None
        slot = escape_slot(mapped_regions)
        use_x_markers = enabled(ui.x_markers) and not ui.x_markers.disabled
        rows = (
            [
                (chosen.filter(pl.col("fate") == name), name, None)
                for name in fate_names
                if name in {fate_names[i] for i in ui.fate.active}
                and (not tree or enabled(full_tree) or fate_counts[name])
            ]
            if enabled(ui.fate_facets) and not pathway_mode
            else [(chosen, "All selected pathways", None)]
        )
        if pathway_mode:
            rows = [
                (
                    chosen.filter(pl.col("simulated_pathway_id") == pathway_ids[i]),
                    selector_names[i],
                    pathway_ids[i],
                )
                for i in ui.fate.active
            ]
        if y_mode != y_layout:
            shared_y = None
            y_mode = y_layout
        lo, hi = realization_bounds()
        if pathway_mode and y_layout == "radial":
            maximum = (
                chosen["max_radial"].max()
                if sparse
                else events_for(events, chosen)["radial"].max()
            ) or 0
            if shared_y is None:
                shared_y = Range1d(0, 1)
            update_range(shared_y, 0, maximum * 1000 * 1.05 if maximum else 1)
        elif pathway_mode and y_layout == "realization" and shared_y is None:
            shared_y = Range1d(lo - 0.5, hi + 0.5)
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
        if population_events is not None:
            positions = population_events["position"]
            population_max = positions.filter(positions.is_finite()).max()
            if population_max is not None:
                right_edge = max(right_edge, population_max)
        if right_edge <= left_edge:
            right_edge = left_edge + 1
        plot_height = event_plot_height()
        margin = (right_edge - left_edge) * 0.015
        x_start = left_edge - margin if y_layout == "beeswarm" else left_edge
        x_end = right_edge if regional else right_edge + margin
        if x_mode != mode:
            shared_x = Range1d(x_start, x_end)
            x_mode = mode
        update_range(shared_x, x_start, x_end)

        event_key = (grouped_key, mapping_key, frozenset(active_groups))
        panels = []
        population_plots = []
        shown_species = set()
        if enabled(ui.schematic) and regional and not chosen.is_empty():
            if schematic is None:
                schematic = _schematic_plot(shared_x, drag_bridge, live=packing.live)
            diagram, update_schematic = schematic
            diagram.x_range = shared_x
            update_schematic(
                mapped_regions,
                dedicated_zones,
                dedicated_zones or enabled(ui.show_bars),
            )
            panels.append(diagram)

        collision_coordinates = (mapped_regions, shared_x, spatial)
        rates.configure_collision(*collision_coordinates)
        # Insert the collision chart after its sources have been updated.

        axis_label = (
            "Zone"
            if regional
            else "Axial distance (mm)"
            if spatial
            else "Elapsed time (s)"
        )
        if regional:
            centers = ((mapped_regions["left"] + mapped_regions["right"]) / 2).to_list()
            grid_ticks = [mapped_regions["left"][0], *mapped_regions["right"]]
            zone_labels = {
                center: str(index) for index, center in enumerate(centers, start=1)
            }

        def selection_callback(source):
            def on_selected(attr, old, new):
                if new and not state["updating"]:
                    if len(new) > 1 and ui.layout.value != "beeswarm":
                        set_candidates(
                            {
                                source.data["id"][i]: source.data["realization_id"][i]
                                for i in new
                            }
                        )
                    else:
                        i = new[0]
                        set_selection(
                            source.data["realization_id"][i], source.data["id"][i]
                        )
                    source.selected.indices = []

            return on_selected

        for members, row_name, pid in rows:
            row_key = (
                ("pathway", pid)
                if pathway_mode
                else ("fate", row_name)
                if enabled(ui.fate_facets)
                else ("all", None)
            )
            if members.is_empty() and chosen.is_empty() and not pathway_mode:
                continue
            if tree and members.is_empty():
                panels.append(
                    panel_notice(
                        (row_key, "empty"),
                        f'<b title="{escape(tree_labels[row_name], quote=True)}">{escape(facet_names[row_name])}</b>: No recorded histories match this leaf in the selected range.',
                        sizing_mode="stretch_width",
                        styles={"overflow-wrap": "anywhere"},
                    )
                )
                continue
            denominator = None
            if pathway_mode:
                # Rows follow the pathway selector, including empty cohorts.
                denominator = chosen.filter(
                    pl.col("simulated_pathway_id") == pid
                ).height
                if members.is_empty():
                    panels.append(
                        panel_notice(
                            (row_key, "empty"),
                            f"<b>{escape(row_name)}</b>: No realizations match the selected range.",
                        )
                    )
            member_events = events_for(positioned, members)
            summary_events = member_events.filter(
                pl.col("event_group").is_in(active_groups)
            )
            plots = []
            if enabled(ui.show_events):
                key = (row_key, mode, y_layout)
                cached = event_panels.get(key)
                bounds = (
                    (lo, hi)
                    if pathway_mode and y_layout == "realization"
                    else (
                        members["realization_number"].min() or 1,
                        members["realization_number"].max() or 1,
                    )
                )
                signature = (event_key, bounds)
                if cached is None:
                    view = _event_plot(
                        shared_x,
                        plot_height,
                        y_layout,
                        state,
                        selection_callback,
                        packing,
                        shared_y=shared_y,
                        physical_regions=mapped_regions if mode == "physical" else None,
                        tree="current_cluster_name" in chosen_events.columns,
                        terminal_spread=regional and y_layout != "radial",
                    )
                else:
                    _, view = cached
                if cached is None or cached[0] != signature:
                    view.update_events(
                        layout_events(summary_events, y_layout),
                        mapped_regions,
                        event_colors,
                        bounds,
                        spread_terminals,
                        members["id"],
                    )
                view.plot.height = plot_height
                view.update_display(
                    use_x_markers, enabled(ui.guides), enabled(ui.envelope)
                )
                selection_updates.append(view.highlight)
                next_events[key] = (signature, view)
                plots.append(view.plot)
            if population_events is not None and not members.is_empty():

                def build_population(previous):
                    areas = population_areas(
                        members,
                        events_for(population_events, members),
                        x_start,
                        x_end,
                        cluster,
                        species_labels,
                        charged_products,
                        tree=bool(tree),
                    )
                    chart, update = (
                        previous[:2]
                        if previous is not None
                        else _fraction_plot(shared_x, population=True)
                    )
                    top = update(areas, species_colors, labels=species_labels)
                    return chart, update, set(areas), top

                chart, _, species, top = summary(
                    (row_key, mode, "population"),
                    (cohort_key, mapping_key, x_start, x_end),
                    build_population,
                )
                shown_species.update(species)
                population_plots.append((chart, top))
                plots.append(chart)
            if enabled(ui.show_cdf) or (enabled(ui.show_bars) and regional):
                if enabled(ui.show_cdf):

                    def build_cdf(previous):
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
                        cumulative_members = (
                            tree.cumulative_members(
                                members, cumulative_events, scope.value
                            )
                            if tree
                            else members
                        )
                        row_colors = {
                            name: fate_colors[name]
                            for name in fate_names
                            if name in members["fate"]
                        }
                        if tree:
                            available = set(cumulative_members["fate"].drop_nulls())
                            names = sorted(available - escapes.colors.keys()) + [
                                key
                                for key in escapes.keys(
                                    enabled(ui.distinguish_escapees)
                                )
                                if key in available
                            ]
                            row_colors = {
                                name: escapes.colors.get(
                                    name, fate_colors.get(name, COLORS["escape"])
                                )
                                for name in names
                            }
                        chart, update = (
                            previous
                            if previous is not None
                            else _fraction_plot(shared_x)
                        )
                        areas = cumulative_areas(
                            cumulative_members,
                            cumulative_events.filter(
                                pl.col("event_group").is_in(active_groups)
                            ),
                            left_edge,
                            right_edge,
                            row_colors,
                            denominator=denominator,
                        )
                        update(areas, row_colors, labels=escapes.labels)
                        return chart, update

                    chart, _ = summary(
                        (row_key, mode, "cdf"),
                        (
                            event_key,
                            left_edge,
                            right_edge,
                            scope.value if tree else None,
                        ),
                        build_cdf,
                    )
                    plots.append(chart)
                if enabled(ui.show_bars) and regional:

                    def build_bar(previous):
                        bars = regional_fractions(
                            members,
                            summary_events,
                            mapped_regions,
                            denominator=denominator,
                        )
                        escape_counts = (
                            members.join(
                                summary_events,
                                left_on="terminal_event_id",
                                right_on="id",
                                how="inner",
                            )
                            .filter(pl.col("type") == "escape")
                            .group_by("event_escape")
                            .len()
                        )
                        counts = dict(escape_counts.iter_rows())
                        escape_fractions = {
                            key: counts[key]
                            / max(
                                members.height if denominator is None else denominator,
                                1,
                            )
                            for key in escapes.keys(enabled(ui.distinguish_escapees))
                            if key in counts
                        }
                        chart, update = (
                            previous
                            if previous is not None
                            else _bar_plot(
                                shared_x, tree=bool(tree), pathway_mode=pathway_mode
                            )
                        )
                        update(bars, escape_fractions, escapes.colors, escapes.labels)
                        return chart, update

                    chart, _ = summary((row_key, mode, "bars"), event_key, build_bar)
                    plots.append(chart)
            faceted = enabled(ui.fate_facets)
            label = facet_names.get(row_name, row_name)
            if tree and faceted and plots:
                panels.append(
                    panel_notice(
                        (row_key, "facet"),
                        f'<b title="{escape(tree_labels.get(row_name, row_name), quote=True)}">{escape(label)} (N={members.height})</b>',
                        sizing_mode="stretch_width",
                        styles={"overflow-wrap": "anywhere"},
                    )
                )
            for index, chart in enumerate(plots):
                title = ""
                attributes = {}
                if tree and chart.name == "cdf":
                    title = f"Cumulative from {tree.label(scope.value)} (N={members.height} selected)"
                elif not tree and faceted and index == 0:
                    title = (
                        f"{label} ({members.height} shown / {denominator} in range)"
                        if pathway_mode
                        else f"{label} (N={members.height})"
                    )
                    attributes = {"title": row_name}
                if chart.title is not None:
                    chart.title.text = title
                elif title:
                    chart.title = title
                chart.html_attributes = attributes
                chart.min_border_top = 0
                chart.min_border_bottom = 0
                if regional:
                    if not isinstance(chart.xaxis[0].ticker, FixedTicker):
                        chart.xaxis.ticker = FixedTicker()
                        chart.xgrid.ticker = FixedTicker()
                    chart.xaxis.ticker.ticks = centers
                    chart.xgrid.ticker.ticks = grid_ticks
                    chart.xaxis.major_label_overrides = zone_labels
                    chart.xaxis.major_tick_line_color = None
                    chart.xaxis.minor_tick_line_color = None
                    chart.xgrid.grid_line_color = "#94a3b8"
                    chart.xgrid.grid_line_alpha = 0.3
                if chart.toolbar.logo is not None:
                    prepare_toolbar(chart)
                chart.xaxis.visible = index == len(plots) - 1
                if chart.xaxis[0].visible:
                    chart.xaxis.axis_label = axis_label
            panels.extend(plots)
        if (
            enabled(ui.show_events)
            and not chosen.is_empty()
            and positioned.filter(pl.col("event_group").is_in(active_groups)).is_empty()
        ):
            panels.insert(
                0,
                panel_notice(
                    "no-events", "<p>No events match the current filters.</p>"
                ),
            )
        if chosen.is_empty() and (not pathway_mode or not rows):
            panels.insert(
                0,
                panel_notice(
                    "no-cohort",
                    "<p>Select at least one pathway.</p>"
                    if pathway_mode
                    else "<p>No realizations in this cohort. Adjust the range or pathway filters.</p>",
                ),
            )
        if population_plots:
            top = max(value for _, value in population_plots)
            for chart, _ in population_plots:
                update_range(chart.y_range, 0, top)
            entries = [
                f'<span><span style="color:{species_colors[species]}" aria-hidden="true">■</span> '
                f"{escape(species_labels[species])}</span>"
                for species in sorted(
                    shown_species, key=lambda s: (s != cluster, species_labels[s], s)
                )
            ]
            panels.append(
                panel_notice(
                    "species-legend",
                    '<div style="display:flex;flex-wrap:wrap;gap:4px 12px">'
                    + "<strong>Charged species</strong> "
                    + " ".join(entries)
                    + "</div>"
                    if entries
                    else "No charged species can be resolved for this selection.",
                    name="species_legend",
                    sizing_mode="stretch_width",
                )
            )
        for key in event_panels.keys() - next_events.keys():
            event_panels[key][1].dispose()
        event_panels, summary_panels = next_events, next_summaries
        update_selection()
        collision_panels = current_collision_panels()
        insert_collision_panels(panels)
        sync_collision_spacing(panels)
        for panel in panels:
            watch_plot_width(panel)
        ui.center_plots.children = panels

    def watch_plot_width(panel):
        if (
            "inner_width" in panel.properties()
            and align_notes
            not in panel.js_property_callbacks.get("change:inner_width", ())
        ):
            panel.js_on_change("inner_width", align_notes)

    def render():
        nonlocal chosen, chosen_events, active_groups, details_key, cohort_key
        if state["updating"]:
            return
        state["updating"] = True
        try:
            # Reconcile the model graph once after replacing controls/plots,
            # rather than walking it after each intermediate layout change.
            with doc.models.freeze():
                selection_updates.clear()
                state["candidates"] = {}
                key = (
                    tuple(ui.fate.active),
                    realization_bounds(),
                    sparse
                    and (
                        enabled(ui.event_options["collision"])
                        or enabled(ui.event_options["rejected_collision"])
                    ),
                )
                if key != cohort_key:
                    cohort_key = key
                    details_key = None
                    chosen, chosen_events = prepare_cohort()
                ui.counts.visible = not pathway_mode and enabled(ui.show_pager)
                ui.selector.visible = enabled(ui.show_pager)
                grouped, event_colors, active_groups = sync_event_controls()
                sync_display_controls(event_colors)
                assemble_panels(
                    chosen, chosen_events, grouped, event_colors, active_groups
                )
        finally:
            state["updating"] = False

    def control_changed(attr, old, new):
        render()

    def cdf_changed(attr, old, new):
        if not state["updating"]:
            state["auto_cdf"] = False
        render()

    def sizing_changed(attr, old, new):
        if state["setting_sizing"]:
            return
        if new != "custom":
            state["widths"] = schematic_widths(new)
        render()

    def dragged(attr, old, new):
        try:
            widths = tuple(float(value) for value in json.loads(new))
        except TypeError, ValueError, json.JSONDecodeError:
            return
        if (
            len(widths) != 7
            or not all(isfinite(width) and width > 0 for width in widths)
            or abs(sum(widths) - sum(state["widths"])) > 1e-6
        ):
            return
        if all(
            abs(old - new) < 1e-10
            for old, new in zip(state["widths"], widths, strict=True)
        ):
            return
        state["widths"] = widths
        state["setting_sizing"] = True
        try:
            ui.sizing.value = "custom"
        finally:
            state["setting_sizing"] = False
        render()

    ui.sizing.on_change("value", sizing_changed)
    drag_bridge.on_change("text", dragged)

    def apply_range(bounds):
        nonlocal applied_range
        if bounds != applied_range:
            applied_range = bounds
            if enabled(ui.show_pager):
                render()

    def selector_changed(attr, old, new):
        apply_range(tuple(int(value) for value in new))

    ui.selector.on_change("value", selector_changed)
    if doc.session_context is not None:
        range_bridge = realization_range_bridge(doc, ui.selector)
        ui.frame.children.append(range_bridge)

        def range_requested(attr, old, new):
            try:
                sequence, lo, hi = json.loads(new)
            except TypeError, ValueError:
                return
            if (
                type(sequence) is int
                and type(lo) is int
                and type(hi) is int
                and ui.selector.start <= lo <= hi <= ui.selector.end
            ):
                try:
                    apply_range((lo, hi))
                finally:
                    range_bridge.tags = [sequence]

        range_bridge.on_change("text", range_requested)

    for widget in ui.event_options.values():
        widget.on_change("active", control_changed)
    for widget, property_name in [
        (ui.fate, "active"),
        (ui.fate_facets, "active"),
        (ui.group_fragmentations, "active"),
        (ui.group_escapes, "active"),
        (ui.separate_rejected, "active"),
        (ui.own_zones, "active"),
        (ui.mode, "value"),
        (ui.layout, "value"),
        (ui.schematic, "active"),
        (ui.show_events, "active"),
        (ui.show_population, "active"),
        (ui.show_bars, "active"),
        (ui.show_pager, "active"),
    ]:
        widget.on_change(property_name, control_changed)

    def display_changed(attr, old, new):
        use_x = enabled(ui.x_markers) and not ui.x_markers.disabled
        for _, view in event_panels.values():
            view.update_display(use_x, enabled(ui.guides), enabled(ui.envelope))
        sync_event_markers(display_colors)

    def spread_changed(attr, old, new):
        for _, view in event_panels.values():
            view.set_spread(enabled(ui.spread_x) and not ui.spread_x.disabled)

    for widget in (ui.x_markers, ui.guides, ui.envelope):
        widget.on_change("active", display_changed)
    ui.spread_x.on_change("active", spread_changed)

    def detail_format_changed(attr, old, new):
        update_details(*selected_history())

    ui.summarise_motion.on_change("active", detail_format_changed)
    ui.summarise_energy.on_change("active", detail_format_changed)

    def show_rates_changed(attr, old, new):
        full = ui.fullscreen_state.text == "full"
        if new and state["details_layout"][full] == "sidebar":
            layout_changed("right")
        update_rates()

    def insert_collision_panels(panels):
        # Direct siblings let Bokeh align all frames to the widest axis labels.
        index = next(
            (i + 1 for i, panel in enumerate(panels) if panel.name == "schematic"), 0
        )
        panels[index:index] = collision_panels

    def sync_collision_panels():
        nonlocal collision_panels
        panels = [
            panel for panel in ui.center_plots.children if panel not in collision_panels
        ]
        collision_panels = current_collision_panels()
        insert_collision_panels(panels)
        sync_collision_spacing(panels)
        for panel in collision_panels:
            watch_plot_width(panel)
        ui.center_plots.children = panels

    def sync_collision_spacing(panels):
        tight = rates.collision_plot in panels and not enabled(ui.fate_facets)
        first = next(
            (
                panel
                for panel in panels
                if panel.name in ("events", "population", "cdf", "bars")
            ),
            None,
        )
        if first is not None:
            first.margin = 0 if tight else None
        if rates.collision_plot is not None:
            rates.collision_plot.margin = 0 if tight else None
            rates.collision_plot.min_border_top = (
                rates.collision_plot.min_border_bottom
            ) = 0 if tight else 5

    def collision_changed(attr, old, new):
        if collision_coordinates is not None:
            rates.configure_collision(*collision_coordinates)
        update_rates()

    def distinguish_changed(attr, old, new):
        colors = escapes.route_colors(bool(new))
        fate_colors.update(colors)
        if recolor_tree is not None:
            recolor_tree(colors)
        render()

    ui.distinguish_escapees.on_change("active", distinguish_changed)
    ui.show_rates.on_change("active", show_rates_changed)
    if full_tree is not None:

        def tree_changed(attr, old, new):
            previous_scope = scope.value
            show_full_tree(bool(new))
            if enabled(ui.fate_facets) or (
                enabled(ui.show_cdf) and scope.value != previous_scope
            ):
                render()

        full_tree.on_change("active", tree_changed)
    if scope is not None:

        def scope_changed(attr, old, new):
            if old not in dict(scope.options):
                return  # The tree toggle will render with the normalized scope.
            if enabled(ui.show_cdf):
                render()
            else:
                ui.show_cdf.active = [0]

        scope.on_change("value", scope_changed)

    ui.collision_rates.on_change("active", collision_changed)
    ui.log_rates.on_change("active", collision_changed)

    def readouts_changed(attr, old, new):
        rates.update_collision()
        rates.update_rate_annotations()

    ui.show_readoffs.on_change("active", readouts_changed)
    for sidebar, detail in (
        (ui.log_rates, ui.detail_log_rates),
        (ui.show_readoffs, ui.detail_show_readoffs),
    ):
        sidebar.on_change(
            "active",
            lambda attr, old, new, detail=detail: setattr(detail, "active", list(new)),
        )
        detail.on_change(
            "active",
            lambda attr, old, new, sidebar=sidebar: setattr(
                sidebar, "active", list(new)
            ),
        )
    for widget in (ui.share_rates, ui.all_pathways):
        widget.on_change("active", lambda attr, old, new: update_rates())
    ui.show_cdf.on_change("active", cdf_changed)
    if doc.session_context is not None:
        ui.center_plots.children = [
            Div(
                text="Loading plots…",
                height=event_plot_height(),
                sizing_mode="stretch_width",
            )
        ]
        started = False

        def ready(event):
            nonlocal started
            # Wait until the controls have reached the browser before building
            # plots. DocumentReady can fire again after a layout change.
            if not started:
                started = True
                render()

        doc.on_event(DocumentReady, ready)
    else:
        render()
    doc.add_root(ui.frame)
