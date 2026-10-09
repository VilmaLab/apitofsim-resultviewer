"""Explorer widgets, frame, styles, and browser bridges."""

from dataclasses import dataclass

from bokeh.events import DocumentReady
from bokeh.layouts import column, row
from bokeh.models import (
    CheckboxGroup,
    Column,
    CustomAction,
    CustomJS,
    Div,
    GroupBox,
    InlineStyleSheet,
    RangeSlider,
    Select,
    Toggle,
)

from .data import EVENT_TYPES

CHECKBOX_CSS = """
    .bk-input-group { white-space: normal; }
    label { display: flex; align-items: center; width: 100%; }
    input { flex-shrink: 0; }
    span { min-width: 0; overflow-wrap: anywhere; }
"""


@dataclass
class Controls:
    """Widgets and containers shared with the session callbacks."""

    counts: Div
    quality: Div
    fate: CheckboxGroup
    fate_facets: CheckboxGroup
    group_fragmentations: CheckboxGroup
    own_zones: CheckboxGroup
    spread_x: CheckboxGroup
    event_options: dict[str, CheckboxGroup]
    event_controls: Column
    mode: Select
    layout: Select
    envelope: CheckboxGroup
    schematic: CheckboxGroup
    guides: CheckboxGroup
    show_events: CheckboxGroup
    x_markers: CheckboxGroup
    show_cdf: CheckboxGroup
    show_bars: CheckboxGroup
    show_pager: CheckboxGroup
    selector: RangeSlider
    details: Div
    summarise_motion: CheckboxGroup
    detail_bridge: Div
    fullscreen_state: Div
    viewport_width: Div
    left_controls: Column
    right_controls: Column
    left_toggle: Toggle
    right_toggle: Toggle
    left: Column
    right: Column
    center_plots: Column
    frame: Column


def checkbox(label, checked=False):
    return CheckboxGroup(
        labels=[label],
        active=[0] if checked else [],
        stylesheets=[InlineStyleSheet(css=CHECKBOX_CSS)],
    )


def enabled(widget):
    return bool(widget.active)


def group(title, child):
    child.width_policy = "max"
    child.height_policy = "min"
    return GroupBox(
        title=title,
        child=child,
        width_policy="max",
        height_policy="min",
        margin=(5, 0),
        # The scroll container must not shrink groups below their contents.
        styles={"flex-shrink": "0"},
        stylesheets=[InlineStyleSheet(css="fieldset { min-inline-size: 0; }")],
    )


def create_controls(
    doc,
    fate_names,
    fate_counts,
    fragmentation_names,
    fate_colors,
    total,
    initial_details,
):
    """Build the frame and attach browser bridges for one document."""
    pathway_legend = "\n".join(
        f'label:nth-child({i + 1})::before {{ content: ""; background: {fate_colors[name]}; '
        "width: 12px; height: 12px; flex-shrink: 0; margin-right: 4px; }"
        for i, name in enumerate(fate_names)
    )
    count_styles = "\n".join(
        f'label:nth-child({i + 1}) span::after {{ content: " ({fate_counts[name]})"; font-style: italic; }}'
        for i, name in enumerate(fate_names)
    )
    counts = Div(styles={"white-space": "nowrap"})
    quality = Div()
    fate = CheckboxGroup(
        labels=fate_names,
        active=list(range(len(fate_names))),
        sizing_mode="stretch_width",
        stylesheets=[
            InlineStyleSheet(css=CHECKBOX_CSS + count_styles + pathway_legend)
        ],
    )

    fate_facets = checkbox("Facet pathways")
    group_fragmentations = checkbox("Group fragmentations", True)
    own_zones = checkbox("Place initial/escape in their own zones", True)
    spread_x = checkbox("Spread initial/escape X", True)
    event_options = {
        kind: checkbox("Initial" if kind == "init" else kind.title(), True)
        for kind in EVENT_TYPES
    } | {name: checkbox(name, True) for name in fragmentation_names}
    for kind, widget in event_options.items():
        widget.name = f"event:{kind}"
        widget.sizing_mode = "stretch_width"
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
        text=initial_details,
        sizing_mode="stretch_width",
        stylesheets=[
            InlineStyleSheet(
                css="""
                :host { color: #1f2937; }
                table { width: 100%; border-collapse: collapse; }
                th, td { padding: 6px 4px; border-bottom: 1px solid #ddd; }
                th { text-align: left; }
                td:not(:first-child) { text-align: right; white-space: nowrap; }
                .explorer-event { cursor: pointer; }
                .explorer-event button {
                    background: transparent; border: 0; padding: 0;
                    color: inherit; font: inherit; text-align: left; cursor: pointer;
                }
            """
            )
        ],
    )
    summarise_motion = checkbox("Summarise non-axial motion", True)
    summarise_motion.visible = False
    detail_bridge = Div(text="", visible=False)
    fullscreen_state = Div(text="normal", visible=False)
    viewport_width = Div(text="1440,900", visible=False)

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
                own_zones,
                spread_x,
                group("Events", event_controls),
                group("Y-axis", column(layout, envelope, spacing=4)),
                spacing=4,
            ),
        ),
        spacing=4,
        styles={"max-height": "calc(100vh - 295px)", "overflow-y": "auto"},
    )
    right_controls = column(
        details,
        summarise_motion,
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
    return Controls(
        counts=counts,
        quality=quality,
        fate=fate,
        fate_facets=fate_facets,
        group_fragmentations=group_fragmentations,
        own_zones=own_zones,
        spread_x=spread_x,
        event_options=event_options,
        event_controls=event_controls,
        mode=mode,
        layout=layout,
        envelope=envelope,
        schematic=schematic,
        guides=guides,
        show_events=show_events,
        x_markers=x_markers,
        show_cdf=show_cdf,
        show_bars=show_bars,
        show_pager=show_pager,
        selector=selector,
        details=details,
        summarise_motion=summarise_motion,
        detail_bridge=detail_bridge,
        fullscreen_state=fullscreen_state,
        viewport_width=viewport_width,
        left_controls=left_controls,
        right_controls=right_controls,
        left_toggle=left_toggle,
        right_toggle=right_toggle,
        left=left,
        right=right,
        center_plots=center_plots,
        frame=frame,
    )


def prepare_toolbar(plot, controls):
    """Use Bokeh's fullscreen toolbar icon for the entire explorer frame."""
    plot.toolbar.logo = None
    plot.add_tools(
        CustomAction(
            description="Fullscreen explorer",
            icon="fullscreen",
            callback=CustomJS(
                args=dict(
                    frame=controls.frame, fullscreen_state=controls.fullscreen_state
                ),
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
