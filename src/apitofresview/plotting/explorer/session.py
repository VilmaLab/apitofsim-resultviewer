"""Cohort selection, controls, and rendering for one Explorer session."""

from functools import lru_cache
from html import escape
import json
from math import isfinite

import polars as pl
from apitofsim.plotting.events import NoRecordedEventsError
from bokeh.events import DocumentReady
from bokeh.layouts import column, row
from bokeh.models import Column, CustomJS, Div, FixedTicker, Range1d

from .aggregation import cumulative_areas, regional_fractions
from .controls import (
    CHECKBOX_CSS,
    apply_layout,
    create_controls,
    enabled,
    label_facets,
    group,
    prepare_toolbar,
)
from .data import EVENT_TYPES, events_for, load_histories, select_realizations
from .data import load_overview as load_data
from .details import render_choices, render_details
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
    _bar_plot,
    _event_plot,
    _fraction_plot,
    _pathway_colors,
    _schematic_plot,
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

    @lru_cache(maxsize=2)
    def histories(realization_ids, event_ids=None):
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
        "updating": False,
        "grouped": True,
        "grouped_escapes": True,
        "escape_keys": escapes.keys(),
        "separated": False,
        "plot_event": None,
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
    scope, tree_labels, full_tree, refresh_tree = (
        install_tree_controls(doc, ui, tree, fate_names, fate_counts, fate_colors)
        if tree
        else (None, {}, None, lambda: None)
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
    selection_updates = []
    chosen = realizations.head(0)
    selection_events = events.head(0)
    active_groups = set()
    selection_notice = Div(visible=False)
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
    collision_coordinates = None
    collision_panels = []
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
        for panel in ui.center_plots.children:
            if panel.name == "events":
                panel.height = event_plot_height()

    def apply_sidebars(full):
        left_open, right_open = state["sidebar"][full]
        mode = state["details_layout"][full]
        if mode == "sidebar":
            ui.show_rates.active = []
        bottom = right_open and mode == "bottom"
        narrow = int(ui.viewport_width.text.split(",")[0]) < 900
        if narrow and left_open and right_open and not bottom:
            left_open = False
            state["sidebar"][full][0] = False
        state["syncing_sidebar"] = True
        try:
            ui.left_toggle.active, ui.right_toggle.active = left_open, right_open
            for placement, button in ui.detail_layouts.items():
                button.active = placement == mode
                button.visible = right_open
        finally:
            state["syncing_sidebar"] = False
        apply_layout(ui, layout, full=full, bottom=bottom)
        ui.left_controls.visible, ui.right_controls.visible = left_open, right_open
        ui.left.width = 205 if left_open else 38
        ui.right.width = (250 if narrow else 300) if right_open else 38
        ui.right.sizing_mode = (
            "stretch_width" if right_open and mode != "sidebar" else "fixed"
        )
        if right_open and mode != "sidebar":
            ui.right.width = None
        if bottom != isinstance(ui.workspace, Column):
            container = column if bottom else row
            ui.workspace = container(
                ui.center,
                ui.right,
                sizing_mode="stretch_width",
                spacing=6,
                styles=ui.workspace.styles,
            )
            ui.body.children = [ui.left, ui.workspace]
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
            and int(ui.viewport_width.text.split(",")[0]) < 900
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

    def selected_cohort():
        lo, hi = (
            (int(v) for v in ui.selector.value)
            if enabled(ui.show_pager)
            else (1, int(ui.selector.end))
        )
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
        full = ui.fullscreen_state.text == "full"
        if rid is not None and not state["sidebar"][full][1]:
            state["sidebar"][full][1] = True
            apply_sidebars(full)
        update_selection()

    def set_candidates(candidates):
        state["selected"] = state["event"] = None
        state["candidates"] = candidates
        full = ui.fullscreen_state.text == "full"
        if not state["sidebar"][full][1]:
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
        if sparse and not selected.is_empty():
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
        if state["candidates"]:
            candidates = selection_events.filter(
                pl.col("id").is_in(state["candidates"])
            )
            if sparse:
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
        else:
            ui.details.text = render_details(
                selected,
                selected_events,
                state["event"],
                enabled(ui.summarise_motion),
                enabled(ui.summarise_energy),
            )
        ui.summarise_motion.visible = not selected.is_empty() or bool(
            state["candidates"]
        )
        ui.summarise_energy.visible = ui.summarise_motion.visible
        ui.show_rates.visible = not selected.is_empty()
        ui.rate_controls.visible = ui.show_rates.visible and enabled(ui.show_rates)
        ui.rate_separator.visible = ui.rate_controls.visible
        ui.detail_log_rates.disabled = not ui.rate_controls.visible
        ui.detail_show_readoffs.disabled = not ui.rate_controls.visible
        selected_event = (
            selected_events.filter(pl.col("id") == state["event"])
            if state["event"] is not None
            else selected_events.head(0)
        )
        old_collision = rates.collision_plot
        rates.update(
            selected,
            selected_event.row(0, named=True)
            if not selected_event.is_empty()
            else None,
        )
        if rates.collision_plot is not old_collision:
            sync_collision_panels()
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
        hidden = (
            selected_events.head(0)
            if state["event"] is None
            else selected_events.filter(
                (pl.col("id") == state["plot_event"])
                & ~pl.col("event_group").is_in(active_groups)
            )
        )
        selection_notice.visible = not hidden.is_empty()
        notice = ""
        if not hidden.is_empty():
            highlighted = hidden.row(0, named=True)
            notice = (
                f"<b>Selected hidden event #{escape(str(highlighted['id']))}</b> · "
                f"{highlighted['type']} · t={highlighted['t']:.6g} s · "
                f"Axial distance={highlighted['z'] * 1000:.6g} mm"
            )
        selection_notice.text = notice
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
        elif action == "inspect" and state["selected"] is not None:
            history = (
                histories((state["selected"],))
                if sparse
                else events.filter(pl.col("realization_id") == state["selected"])
            )
            for event_id in history["id"]:
                if str(event_id) == event:
                    set_selection(state["selected"], event_id)
                    break

    ui.detail_bridge.on_change("text", sidebar_event)

    def prepare_cohort():
        """Filter the cohort, then update its details and summary."""
        chosen = selected_cohort()
        chosen_ids = chosen["id"]
        if state["selected"] not in chosen_ids:
            state["selected"] = state["event"] = None
        chosen_events = events_for(events, chosen)
        if (
            sparse
            and not chosen.is_empty()
            and chosen["collisions"].sum() <= MAX_COLLISIONS
            and (
                enabled(ui.event_options["collision"])
                or enabled(ui.event_options["rejected_collision"])
            )
        ):
            chosen_events = histories(tuple(chosen["id"]))
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
            set(selected_cohort()["fate"])
            if pathway_mode
            else {fate_names[i] for i in ui.fate.active}
        )
        available_members = (
            selected_cohort()
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
                enabled(ui.show_population),
                enabled(ui.show_bars) and regional,
            )
        ):
            state["auto_cdf"] = enabled(ui.show_bars) and not regional
            ui.show_cdf.active = [0]

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
        """Position the cohort and assemble shared-axis plots and selection notices."""
        nonlocal selection_events, collision_coordinates, collision_panels
        chosen_events = group_events(chosen_events, grouped)
        selection_events = chosen_events
        mapped_regions = coordinate_regions(
            regions,
            ui.mode.value,
            state["widths"] if ui.mode.value == "schematic" else None,
        )
        spatial = ui.mode.value != "time"
        regional = ui.mode.value == "schematic"
        dedicated_zones = regional and enabled(ui.own_zones)
        spread_terminals = enabled(ui.spread_x) and not ui.spread_x.disabled
        positioned = position_events(
            chosen_events.filter(
                ~pl.col("type").is_in(["postcollision", "postfragmentation"])
            ),
            mapped_regions,
            ui.mode.value,
            dedicated_zones,
        )
        population_events = (
            position_events(
                chosen_events.with_columns(
                    pl.col("population_z").clip(lower_bound=0).alias("z"),
                    pl.col("population_t").alias("t"),
                )
                if "population_z" in chosen_events.columns
                else chosen_events,
                mapped_regions,
                ui.mode.value,
                terminal_slots=False,
            )
            if enabled(ui.show_population)
            else None
        )
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
        population = selected_cohort() if pathway_mode else chosen
        if pathway_mode:
            rows = [
                (
                    chosen.filter(pl.col("simulated_pathway_id") == pathway_ids[i]),
                    selector_names[i],
                    pathway_ids[i],
                )
                for i in ui.fate.active
            ]
        shared_y = None
        if pathway_mode and ui.layout.value in ("realization", "radial"):
            if ui.layout.value == "realization":
                lo = int(ui.selector.value[0]) if enabled(ui.show_pager) else 1
                hi = (
                    int(ui.selector.value[1])
                    if enabled(ui.show_pager)
                    else int(ui.selector.end)
                )
                shared_y = Range1d(lo - 0.5, hi + 0.5)
            else:
                maximum = (
                    population["max_radial"].max()
                    if sparse
                    else events_for(events, population)["radial"].max()
                ) or 0
                shared_y = Range1d(0, maximum * 1000 * 1.05 if maximum else 1)
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
        screen_width = int(ui.viewport_width.text.split(",")[0])
        available_width = screen_width - ui.left.width
        if state["sidebar"][ui.fullscreen_state.text == "full"][1]:
            mode = state["details_layout"][ui.fullscreen_state.text == "full"]
            if mode == "right":
                available_width /= 2
            elif mode == "sidebar":
                available_width -= ui.right.width
        else:
            available_width -= ui.right.width
        plot_height = event_plot_height()
        margin = (right_edge - left_edge) * 0.015
        shared_x = Range1d(
            left_edge - margin if ui.layout.value == "beeswarm" else left_edge,
            right_edge if regional else right_edge + margin,
        )
        panels = []
        population_plots = []
        shown_species = set()
        if enabled(ui.schematic) and spatial and regional and not chosen.is_empty():
            panels.append(
                _schematic_plot(
                    mapped_regions,
                    shared_x,
                    available_width,
                    drag_bridge,
                    initial=dedicated_zones,
                    escaped=dedicated_zones or enabled(ui.show_bars),
                )
            )

        collision_coordinates = (mapped_regions, shared_x, spatial)
        rates.configure_collision(*collision_coordinates)
        # Direct siblings let Bokeh align all frames to the widest axis labels.
        collision_panels = (
            list(rates.collision.children) if rates.collision.visible else []
        )
        panels.extend(collision_panels)

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
            if members.is_empty() and chosen.is_empty() and not pathway_mode:
                continue
            if tree and members.is_empty():
                panels.append(
                    Div(
                        text=f'<b title="{escape(tree_labels[row_name], quote=True)}">{escape(facet_names[row_name])}</b>: No recorded histories match this leaf in the selected range.',
                        sizing_mode="stretch_width",
                        styles={"overflow-wrap": "anywhere"},
                    )
                )
                continue
            denominator = None
            if pathway_mode:
                # Rows follow the pathway selector, including empty cohorts.
                denominator = population.filter(
                    pl.col("simulated_pathway_id") == pid
                ).height
                if members.is_empty():
                    panels.append(
                        Div(
                            text=f"<b>{escape(row_name)}</b>: No realizations match the selected range."
                        )
                    )
            member_events = events_for(positioned, members)
            summary_events = member_events.filter(
                pl.col("event_group").is_in(active_groups)
            )
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
                        (lo, hi)
                        if pathway_mode and ui.layout.value == "realization"
                        else (
                            members["realization_number"].min() or 1,
                            members["realization_number"].max() or 1,
                        ),
                        y_range=shared_y,
                        spread_terminals=spread_terminals,
                        selection_updates=selection_updates,
                        doc=doc,
                    )
                )
            if population_events is not None and not members.is_empty():
                areas = population_areas(
                    members,
                    events_for(population_events, members),
                    shared_x.start,
                    shared_x.end,
                    cluster,
                    species_labels,
                    charged_products,
                    tree=bool(tree),
                )
                shown_species.update(areas)
                chart = _fraction_plot(
                    areas,
                    shared_x,
                    species_colors,
                    labels=species_labels,
                    population=True,
                )
                population_plots.append(chart)
                plots.append(chart)
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
                    cumulative_members = (
                        tree.cumulative_members(members, cumulative_events, scope.value)
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
                            for key in escapes.keys(enabled(ui.distinguish_escapees))
                            if key in available
                        ]
                        row_colors = {
                            name: escapes.colors.get(
                                name, fate_colors.get(name, COLORS["escape"])
                            )
                            for name in names
                        }
                    plots.append(
                        _fraction_plot(
                            cumulative_areas(
                                cumulative_members,
                                cumulative_events.filter(
                                    pl.col("event_group").is_in(active_groups)
                                ),
                                left_edge,
                                right_edge,
                                row_colors,
                                denominator=denominator,
                            ),
                            shared_x,
                            row_colors,
                            labels=escapes.labels,
                        )
                    )
                if enabled(ui.show_bars) and regional:
                    bars = regional_fractions(
                        members, summary_events, mapped_regions, denominator=denominator
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
                        / max(members.height if denominator is None else denominator, 1)
                        for key in escapes.keys(enabled(ui.distinguish_escapees))
                        if key in counts
                    }
                    bar = _bar_plot(
                        bars,
                        shared_x,
                        escapes=escape_fractions,
                        escape_colors=escapes.colors,
                        escape_labels=escapes.labels,
                    )
                    if tree:
                        bar.yaxis.axis_label = "Events / realization"
                    if pathway_mode:
                        bar.y_range = Range1d(0, 1)
                    plots.append(bar)
            if tree and enabled(ui.fate_facets) and plots:
                panels.append(
                    Div(
                        text=f'<b title="{escape(tree_labels.get(row_name, row_name), quote=True)}">{escape(facet_names.get(row_name, row_name))} (N={members.height})</b>',
                        sizing_mode="stretch_width",
                        styles={"overflow-wrap": "anywhere"},
                    )
                )
            elif enabled(ui.fate_facets) and plots:
                plots[0].title = (
                    f"{facet_names.get(row_name, row_name)} ({members.height} shown / {denominator} in range)"
                    if pathway_mode
                    else f"{facet_names.get(row_name, row_name)} (N={members.height})"
                )
            if not tree and enabled(ui.fate_facets) and plots:
                plots[0].html_attributes = {"title": row_name}
            if tree and enabled(ui.show_cdf):
                for chart in plots:
                    if chart.name == "cdf":
                        chart.title = f"Cumulative from {tree.label(scope.value)} (N={members.height} selected)"
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
                prepare_toolbar(plot)
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
        if (
            enabled(ui.show_events)
            and not chosen.is_empty()
            and positioned.filter(pl.col("event_group").is_in(active_groups)).is_empty()
        ):
            panels.insert(0, Div(text="<p>No events match the current filters.</p>"))
        if chosen.is_empty() and (not pathway_mode or not rows):
            panels.insert(
                0,
                Div(
                    text="<p>Select at least one pathway.</p>"
                    if pathway_mode
                    else "<p>No realizations in this cohort. Adjust the range or pathway filters.</p>"
                ),
            )
        if population_plots:
            top = max(plot.y_range.end for plot in population_plots)
            for plot in population_plots:
                plot.y_range.end = top
            entries = [
                f'<span><span style="color:{species_colors[species]}" aria-hidden="true">■</span> '
                f"{escape(species_labels[species])}</span>"
                for species in sorted(
                    shown_species, key=lambda s: (s != cluster, species_labels[s], s)
                )
            ]
            panels.append(
                Div(
                    text='<div style="display:flex;flex-wrap:wrap;gap:4px 12px">'
                    + "<strong>Charged species</strong> "
                    + " ".join(entries)
                    + "</div>"
                    if entries
                    else "No charged species can be resolved for this selection.",
                    name="species_legend",
                    sizing_mode="stretch_width",
                )
            )
        sync_collision_spacing(panels)
        ui.center_plots.children = [selection_notice, *panels]
        axis_plot = next((p for p in panels if getattr(p, "yaxis", None)), None)
        if axis_plot is not None:
            axis_plot.js_on_change(
                "inner_width",
                CustomJS(
                    args=dict(plot=axis_plot, notes=[ui.counts, ui.quality]),
                    code="""
                    const view = Bokeh.index.find_one(plot);
                    if (!view) return;
                    const axis = view.canvas_view.events_el.getBoundingClientRect().left + view.frame.bbox.left;
                    for (const note of notes) {
                        const el = Bokeh.index.find_one(note)?.el;
                        if (el?.checkVisibility())
                            note.styles = {...note.styles, "box-sizing": "border-box",
                                "padding-left": `${Math.max(0, axis - el.getBoundingClientRect().left)}px`};
                    }
                    """,
                ),
            )

    def render():
        nonlocal chosen, active_groups
        if state["updating"]:
            return
        state["updating"] = True
        try:
            # Reconcile the model graph once after replacing controls/plots,
            # rather than walking it after each intermediate layout change.
            with doc.models.freeze():
                state["candidates"] = {}
                selection_updates.clear()
                chosen, chosen_events = prepare_cohort()
                grouped, event_colors, active_groups = sync_event_controls()
                sync_display_controls(event_colors)
                assemble_panels(
                    chosen, chosen_events, grouped, event_colors, active_groups
                )
                update_selection()
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
        state["widths"] = widths
        state["setting_sizing"] = True
        try:
            ui.sizing.value = "custom"
        finally:
            state["setting_sizing"] = False
        render()

    ui.sizing.on_change("value", sizing_changed)
    drag_bridge.on_change("text", dragged)

    for widget in ui.event_options.values():
        widget.on_change("active", control_changed)
    for widget, property_name in [
        (ui.fate, "active"),
        (ui.fate_facets, "active"),
        (ui.group_fragmentations, "active"),
        (ui.group_escapes, "active"),
        (ui.separate_rejected, "active"),
        (ui.own_zones, "active"),
        (ui.spread_x, "active"),
        (ui.mode, "value"),
        (ui.layout, "value"),
        (ui.envelope, "active"),
        (ui.schematic, "active"),
        (ui.guides, "active"),
        (ui.show_events, "active"),
        (ui.show_population, "active"),
        (ui.x_markers, "active"),
        (ui.show_bars, "active"),
        (ui.show_pager, "active"),
        (ui.selector, "value"),
    ]:
        widget.on_change(property_name, control_changed)
    ui.summarise_motion.on_change("active", lambda attr, old, new: update_selection())
    ui.summarise_energy.on_change("active", lambda attr, old, new: update_selection())

    def show_rates_changed(attr, old, new):
        full = ui.fullscreen_state.text == "full"
        if new and state["details_layout"][full] == "sidebar":
            layout_changed("right")
        update_selection()

    def sync_collision_panels():
        nonlocal collision_panels
        panels = [
            panel for panel in ui.center_plots.children if panel not in collision_panels
        ]
        index = next(
            (i + 1 for i, panel in enumerate(panels) if panel.name == "schematic"), 1
        )
        collision_panels = (
            list(rates.collision.children) if rates.collision.visible else []
        )
        panels = [*panels[:index], *collision_panels, *panels[index:]]
        sync_collision_spacing(panels)
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
            sync_collision_panels()
        update_selection()

    def rate_scale_changed(attr, old, new):
        rates.change_scale()
        sync_collision_panels()
        update_selection()

    def distinguish_changed(attr, old, new):
        fate_colors.update(escapes.route_colors(bool(new)))
        refresh_tree()
        render()

    ui.distinguish_escapees.on_change("active", distinguish_changed)
    ui.show_rates.on_change("active", show_rates_changed)
    if full_tree is not None:
        full_tree.on_change("active", control_changed)
    if scope is not None:

        def scope_changed(attr, old, new):
            if enabled(ui.show_cdf):
                render()
            else:
                ui.show_cdf.active = [0]

        scope.on_change("value", scope_changed)

    ui.collision_rates.on_change("active", collision_changed)
    ui.log_rates.on_change("active", rate_scale_changed)
    ui.show_readoffs.on_change("active", lambda attr, old, new: update_selection())
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
        widget.on_change("active", lambda attr, old, new: update_selection())
    ui.show_cdf.on_change("active", cdf_changed)
    if doc.session_context is not None:
        ui.center_plots.children = [Div(text="Loading plots…")]
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
