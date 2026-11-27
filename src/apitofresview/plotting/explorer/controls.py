"""Explorer widgets, frame, styles, and browser bridges."""

from dataclasses import dataclass

from bokeh.events import DocumentReady
from bokeh.layouts import column, row
from bokeh.models import (
    Button,
    CheckboxGroup,
    Column,
    ColumnDataSource,
    CustomJS,
    Div,
    InlineStyleSheet,
    RangeSlider,
    Row,
    Select,
    Toggle,
)

from workingtitle.bokeh.browser import ON_FRAME
from workingtitle.bokeh.panels import PlacementSelector, ResponsivePanels
from workingtitle.bokeh.workspace import (
    FrameFullscreen,
    observe_panel_width,
    Sidebar,
    apply_layout as apply_workspace_layout,
)
from workingtitle.bokeh.widgets import CHECKBOX_CSS, checkbox, control_group
from .data import EVENT_TYPES

# Bokeh caps non-fixed layouts at 100% of their parent. Scroll contents must
# retain their intrinsic height, including when taller than the whole panel.
CONTENT_CSS = ":host > * { flex-shrink: 0; max-height: none; }"


@dataclass
class Controls:
    """Widgets and containers shared with the session callbacks."""

    counts: Div
    quality: Div
    fate: CheckboxGroup
    fate_facets: CheckboxGroup
    separate_rejected: CheckboxGroup
    group_fragmentations: CheckboxGroup
    group_escapes: CheckboxGroup
    distinguish_escapees: CheckboxGroup
    own_zones: CheckboxGroup
    spread_x: CheckboxGroup
    event_options: dict[str, CheckboxGroup]
    event_controls: Column
    mode: Select
    sizing: Select
    layout: Select
    envelope: CheckboxGroup
    schematic: CheckboxGroup
    guides: CheckboxGroup
    show_events: CheckboxGroup
    x_markers: CheckboxGroup
    show_population: CheckboxGroup
    show_cdf: CheckboxGroup
    show_bars: CheckboxGroup
    show_pager: CheckboxGroup
    selector: RangeSlider
    details: Div
    detail_selection: ColumnDataSource
    summarise_motion: CheckboxGroup
    summarise_energy: CheckboxGroup
    show_rates: CheckboxGroup
    share_rates: CheckboxGroup
    all_pathways: CheckboxGroup
    log_rates: CheckboxGroup
    show_readoffs: CheckboxGroup
    detail_log_rates: CheckboxGroup
    detail_show_readoffs: CheckboxGroup
    rate_separator: Div
    collision_rates: CheckboxGroup
    rate_controls: Column
    rate_plots: Column
    detail_bridge: Div
    fullscreen_state: Div
    panels: ResponsivePanels
    fullscreen_button: Button
    left_controls: Column
    right_controls: Column
    left_toggle: Toggle
    right_toggle: Toggle
    placement_selector: PlacementSelector
    left_sidebar: Sidebar
    left: Column
    right: Column
    center: Column
    workspace: Row | Column
    body: Row
    center_plots: Column
    frame: Column


def group(title, child, *, name=None):
    return control_group(title, child, name=name, scrolling=True)


def create_controls(
    doc,
    fate_names,
    fate_counts,
    fragmentation_names,
    fate_colors,
    total,
    initial_details,
    *,
    layout="viewport",
    pathway_mode=False,
    tree_mode=False,
    escape_options=None,
):
    """Build the frame and attach browser bridges for one document."""
    layout_policy = layout
    pathway_legend = "\n".join(
        f'label:nth-child({i + 1})::before {{ content: ""; background: {fate_colors[name]}; '
        "width: 12px; height: 12px; flex-shrink: 0; margin-right: 4px; }"
        for i, name in enumerate(fate_names)
        if name in fate_colors
    )
    count_styles = (
        ""
        if tree_mode
        else "\n".join(
            f'label:nth-child({i + 1}) span::after {{ content: " ({fate_counts[name]})"; font-style: italic; }}'
            for i, name in enumerate(fate_names)
        )
    )
    counts = Div(
        sizing_mode="stretch_width",
        styles={"box-sizing": "border-box", "overflow-wrap": "anywhere"},
    )
    quality = Div(sizing_mode="stretch_width", styles={"box-sizing": "border-box"})
    fate = CheckboxGroup(
        labels=fate_names,
        active=list(range(len(fate_names))),
        sizing_mode="stretch_width",
        stylesheets=[
            InlineStyleSheet(css=CHECKBOX_CSS + count_styles + pathway_legend)
        ],
    )

    fate_facets = checkbox("Facet pathways", pathway_mode)
    fate_facets.visible = not pathway_mode
    separate_rejected = checkbox("Separate rejected collisions")
    group_fragmentations = checkbox("Group fragmentations", True)
    group_fragmentations.visible = not pathway_mode
    group_escapes = checkbox("Group escapes", True)
    group_escapes.name = "group_escapes"
    group_escapes.visible = tree_mode
    distinguish_escapees = checkbox("Distinguish identical escapees by pathway")
    distinguish_escapees.visible = False
    distinguish_escapees.name = "distinguish_escapees"
    own_zones = checkbox("Place initial/escape in their own zones", True)
    spread_x = checkbox("Spread initial/escape X", True)
    event_options = (
        {
            kind: checkbox("Initial" if kind == "init" else kind.title(), True)
            for kind in EVENT_TYPES
        }
        | {"rejected_collision": checkbox("Rejected collisions", True)}
        | {name: checkbox(name, True) for name in fragmentation_names}
        | {key: checkbox(label, True) for key, label in (escape_options or {}).items()}
    )
    for kind, widget in event_options.items():
        widget.name = f"event:{kind}"
        widget.sizing_mode = "stretch_width"
    event_controls = column(spacing=4, width_policy="max")
    mode = Select(
        title="X coordinate",
        value="schematic",
        options=[
            ("schematic", "Schematic"),
            ("physical", "Axial distance (mm)"),
            ("time", "Elapsed time (s)"),
        ],
    )
    sizing = Select(
        title="Sizing",
        value="chambers",
        options=[
            ("chambers", "Equal chambers"),
            ("zones", "Equal zones"),
            ("custom", "Custom"),
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
    show_population = checkbox("Species populations")
    show_cdf = checkbox("Cumulative")
    show_bars = checkbox("Bar chart")
    show_pager = checkbox("Realizations pager")
    selector = RangeSlider(
        title="Realization IDs (inclusive index range)",
        stylesheets=[
            InlineStyleSheet(
                css="""
            .bk-input-group { box-sizing: border-box; padding-inline: 8px; }
            .bk-slider-title { white-space: normal; }
        """
            )
        ],
        start=1,
        end=max(total, 1),
        step=1,
        value=(1, max(total, 1)),
        disabled=total == 0,
        sizing_mode="stretch_width",
        height=50,
        margin=(5, 10),
        visible=False,
    )
    details = Div(
        text=initial_details,
        sizing_mode="stretch_width",
        stylesheets=[
            InlineStyleSheet(
                css="""
                :host { color: #1f2937; }
                .bk-clearfix { width: 100%; }
                table { width: 100%; border-collapse: collapse; }
                th, td { padding: 6px 4px; border-bottom: 1px solid #ddd; }
                th { text-align: left; }
                td:first-child { white-space: nowrap; }
                td:not(:first-child) { text-align: right; white-space: nowrap; }
                .explorer-after td { text-align: right; white-space: nowrap; }
                .explorer-choice td:nth-child(2) { text-align: left; }
                .explorer-event, .explorer-choice { cursor: pointer; }
                .explorer-event button, .explorer-choice button {
                    background: transparent; border: 0; padding: 0;
                    color: inherit; font: inherit; text-align: left; cursor: pointer;
                }
            """
            )
        ],
    )
    detail_selection = ColumnDataSource(data=dict(event=[], pair=[]))
    summarise_motion = checkbox("Summarise non-axial motion", True)
    summarise_motion.visible = False
    summarise_energy = checkbox("Summarise energy", True)
    summarise_energy.visible = False
    show_rates = checkbox("Show rate constants")
    show_rates.visible = False
    share_rates = checkbox("Share plot", True)
    all_pathways = checkbox("Show all pathways")
    log_rates = checkbox("Logarithmic rates", True)
    log_rates.name = "log_rates"
    show_readoffs = checkbox("Show readoff values", True)
    show_readoffs.name = "show_readoffs"
    detail_log_rates = checkbox("Logarithmic rates", True)
    detail_log_rates.name = "detail_log_rates"
    detail_show_readoffs = checkbox("Show readoff values", True)
    detail_show_readoffs.name = "detail_show_readoffs"
    rate_separator = Div(
        height=1,
        sizing_mode="stretch_width",
        visible=False,
        margin=(8, 0),
        styles={"border-top": "1px solid #cbd5e1"},
        html_attributes={"role": "separator"},
        name="rate_separator",
    )
    collision_rates = checkbox("Gas collision rate")
    rate_plots = column(Div(), sizing_mode="stretch_width", name="rate_plots")
    rate_controls = column(
        rate_separator,
        row(
            group(
                "Rates",
                column(detail_log_rates, detail_show_readoffs, spacing=4),
                name="detail_rates",
            ),
            group(
                "Pathways",
                column(share_rates, all_pathways, spacing=4),
                name="detail_pathways",
            ),
            sizing_mode="stretch_width",
            spacing=8,
            name="detail_rate_groups",
        ),
        rate_plots,
        sizing_mode="stretch_width",
        visible=False,
    )
    detail_bridge = Div(text="", visible=False)
    fullscreen_state = Div(text="normal", visible=False)
    frame = column(
        sizing_mode="stretch_width", spacing=4, css_classes=["explorer-bokeh-frame"]
    )
    panels = ResponsivePanels(
        doc, frame, breakpoint=900, name="explorer_viewport", host_selector="#content"
    )
    fullscreen = FrameFullscreen(
        frame, document=doc, state=fullscreen_state, overflow="hidden", padding=6
    )
    fullscreen_button = fullscreen.button(sizing_mode="stretch_width")

    left_controls = column(
        fullscreen_button,
        group(
            "Pathways",
            column(
                fate,
                fate_facets,
                spacing=4,
            ),
            name="sidebar_pathways",
        ),
        group("X-axis", column(mode, sizing, guides, spacing=4)),
        group(
            "Elements",
            column(
                schematic,
                group(
                    "Views",
                    column(
                        collision_rates,
                        show_events,
                        show_population,
                        show_cdf,
                        show_bars,
                        spacing=4,
                    ),
                ),
                show_pager,
                spacing=4,
            ),
        ),
        group(
            "Realizations",
            column(
                x_markers,
                group_fragmentations,
                group_escapes,
                separate_rejected,
                own_zones,
                spread_x,
                group("Events", event_controls),
                group("Y-axis", column(layout, envelope, spacing=4)),
                spacing=4,
            ),
        ),
        group(
            "Rates", column(log_rates, show_readoffs, spacing=4), name="sidebar_rates"
        ),
        spacing=4,
        css_classes=["explorer-controls"],
        stylesheets=[InlineStyleSheet(css=CONTENT_CSS)],
    )
    right_controls = column(
        details,
        summarise_motion,
        summarise_energy,
        show_rates,
        rate_controls,
        css_classes=["explorer-details"],
        visible=False,
        stylesheets=[InlineStyleSheet(css=CONTENT_CSS)],
    )
    left_sidebar = Sidebar.create(
        left_controls,
        widths=(205, 38),
        labels=("« Controls", "»"),
        button_widths=(100, 38),
        toggle_options=dict(height=32, margin=0),
        shell_options=dict(spacing=4),
    )
    left, left_toggle = left_sidebar.shell, left_sidebar.toggle
    right_toggle = Toggle(
        label="☷",
        margin=0,
        active=False,
        width=36,
        height=32,
        html_attributes={"title": "Show selected realization"},
    )
    placement = PlacementSelector(name="details_layout", panel=right_controls)
    details_header = row(right_toggle, placement.widget, spacing=2, margin=0)
    right = column(
        details_header,
        right_controls,
        width=38,
        spacing=4,
        stylesheets=[
            InlineStyleSheet(
                css=f"""
                :host {{ --sidebar-width: 300px; }}
                @container explorer (width < {panels.breakpoint}px) {{
                    :host {{ --sidebar-width: 250px; }}
                }}
                """
            )
        ],
    )
    center_plots = column(
        sizing_mode="stretch_width",
        spacing=0,
        name="explorer_plots",
        css_classes=["explorer-plots"],
    )
    center = column(
        counts,
        quality,
        center_plots,
        selector,
        sizing_mode="stretch_width",
        spacing=2,
        css_classes=["explorer-center"],
        stylesheets=[InlineStyleSheet(css=CONTENT_CSS)],
    )
    observe_panel_width(doc, center)
    workspace = row(
        center,
        right,
        sizing_mode="stretch_width",
        spacing=6,
    )
    body = row(
        left,
        workspace,
        sizing_mode="stretch_width",
        spacing=6,
    )
    frame.children = [body, detail_bridge, fullscreen_state, panels.viewport]
    doc.js_on_event(
        DocumentReady,
        CustomJS(
            args=dict(frame=frame, bridge=detail_bridge),
            code="""
        const view = Bokeh.index.find_one(frame);
        if (!view) return;
        const jump = input => {
            if (input?.validity.valid && Number.isInteger(input.valueAsNumber))
                bridge.text = JSON.stringify(['page', input.valueAsNumber - 1]);
        };
        // Details markup changes when navigating to a different history page.
        view.el.addEventListener('click', event => {
            const path = event.composedPath().filter(node => node instanceof Element);
            if (path.some(node => node.matches('#clear-selection'))) {
                bridge.text = JSON.stringify(['clear', null]);
                return;
            }
            const page = path.find(node => node.matches('button[data-history-page]'));
            if (page && !page.disabled) {
                bridge.text = JSON.stringify(['page', Number(page.dataset.historyPage)]);
                return;
            }
            const go = path.find(node => node.matches('[data-history-page-jump]'));
            if (go) {
                jump(go.closest('nav').querySelector('[data-history-page-input]'));
                return;
            }
            const row = path.find(node => node.matches('.explorer-event, .explorer-choice'));
            if (row) {
                bridge.text = JSON.stringify([
                    row.dataset.realization ? 'choose' : 'inspect', row.dataset.event,
                ]);
            }
        });
        view.el.addEventListener('keydown', event => {
            if (event.key !== 'Enter') return;
            const input = event.composedPath().find(
                node => node instanceof Element && node.matches('[data-history-page-input]'));
            if (input) {
                event.preventDefault();
                jump(input);
            }
        });
    """,
        ),
    )

    scroll_to_event = CustomJS(
        args=dict(details=details, selection=detail_selection, panel=right_controls),
        code=ON_FRAME
        + """
        if (cb_obj === panel && panel.visible) details.explorerScrollOnOpen = true;
        onFrame(details, 'explorerScrollFrame', () => {
            if (!panel.visible) return;
            const view = Bokeh.index.find_one(details);
            const root = view?.shadow_el ?? view?.el?.shadowRoot ?? view?.el;
            if (!root) return;
            const [event] = selection.data.event;
            const [pair] = selection.data.pair;
            const findRow = id => id == null ? null :
                root.querySelector(`.explorer-event[data-event="${CSS.escape(id)}"]`);
            const active = findRow(event);
            const shared = findRow(pair);
            const old = root.explorerActiveRow?.isConnected ? root.explorerActiveRow :
                root.querySelector('.explorer-event[style*="fde68a"]');
            if (old !== active) {
                old?.setAttribute('style', 'background:white');
                active?.setAttribute('style', 'background:#fde68a');
            }
            root.explorerActiveRow = active;
            const oldPair = root.explorerPairRow;
            if (oldPair !== shared || !old?.isConnected) {
                const cells = oldPair?.isConnected ? oldPair.querySelectorAll('td[rowspan][style]') :
                    root.querySelectorAll('.explorer-event td[rowspan][style]');
                for (const cell of cells) cell.removeAttribute('style');
                for (const cell of shared?.querySelectorAll('td[rowspan]') ?? [])
                    cell.setAttribute('style', 'background:#fde68a');
                root.explorerPairRow = shared;
            }
            const panelView = Bokeh.index.find_one(panel);
            const scroller = panelView?.el;
            // Visibility and pane placement can arrive in separate patches.
            if (active && scroller && getComputedStyle(panelView.parent.el).position === 'relative' &&
                (root.explorerScrolledRow !== active || details.explorerScrollOnOpen)) {
                const row = active.getBoundingClientRect();
                const area = scroller.getBoundingClientRect();
                if (row.top < Math.max(0, area.top) || row.bottom > Math.min(innerHeight, area.bottom))
                    active.scrollIntoView({block: 'nearest', inline: 'nearest'});
                root.explorerScrolledRow = active;
                details.explorerScrollOnOpen = false;
            }
        });
        """,
    )
    details.js_on_change("text", scroll_to_event)
    detail_selection.js_on_change("data", scroll_to_event)
    right_controls.js_on_change("visible", scroll_to_event)
    body.js_on_change("styles", scroll_to_event)
    ui = Controls(
        counts=counts,
        quality=quality,
        fate=fate,
        fate_facets=fate_facets,
        group_fragmentations=group_fragmentations,
        group_escapes=group_escapes,
        distinguish_escapees=distinguish_escapees,
        separate_rejected=separate_rejected,
        own_zones=own_zones,
        spread_x=spread_x,
        event_options=event_options,
        event_controls=event_controls,
        mode=mode,
        sizing=sizing,
        layout=layout,
        envelope=envelope,
        schematic=schematic,
        guides=guides,
        show_events=show_events,
        x_markers=x_markers,
        show_population=show_population,
        show_cdf=show_cdf,
        show_bars=show_bars,
        show_pager=show_pager,
        selector=selector,
        details=details,
        detail_selection=detail_selection,
        summarise_motion=summarise_motion,
        summarise_energy=summarise_energy,
        show_rates=show_rates,
        share_rates=share_rates,
        all_pathways=all_pathways,
        log_rates=log_rates,
        show_readoffs=show_readoffs,
        detail_log_rates=detail_log_rates,
        detail_show_readoffs=detail_show_readoffs,
        rate_separator=rate_separator,
        collision_rates=collision_rates,
        rate_controls=rate_controls,
        rate_plots=rate_plots,
        detail_bridge=detail_bridge,
        fullscreen_state=fullscreen_state,
        panels=panels,
        fullscreen_button=fullscreen_button,
        left_controls=left_controls,
        right_controls=right_controls,
        left_toggle=left_toggle,
        right_toggle=right_toggle,
        placement_selector=placement,
        left_sidebar=left_sidebar,
        left=left,
        right=right,
        center=center,
        workspace=workspace,
        body=body,
        center_plots=center_plots,
        frame=frame,
    )

    apply_layout(ui, layout_policy)
    return ui


def apply_layout(ui, layout, *, full=False, bottom=False):
    apply_workspace_layout(
        ui,
        layout,
        full=full,
        bottom=bottom,
        container="explorer",
        viewport_height="var(--explorer-height, calc(100dvh - var(--workspace-top, 0px) - 6px))",
    )


def label_facets(doc, widget, labels):
    """Keep pathway identities while presenting charged daughters with hover context."""
    callback = CustomJS(
        args=dict(widget=widget, labels=labels),
        code="""
        const view = Bokeh.index.find_one(widget);
        const root = view?.shadow_el ?? view?.el?.shadowRoot ?? view?.el;
        if (!root) return;
        root.querySelectorAll('label').forEach((label, i) => {
            const name = widget.labels[i];
            const span = label.querySelector('span');
            if (span) span.textContent = labels[name] ?? name;
            label.title = name;
        });
    """,
    )
    doc.js_on_event(DocumentReady, callback)
    widget.js_on_change("labels", callback)
