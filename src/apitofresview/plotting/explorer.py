"""Scoped realization explorer and its numerical data model.

The event recorder persists simulator ``postime.t`` directly. The database has
no separate realization start timestamp, so the time view uses that raw value.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from html import escape
from math import exp, sin, sqrt

from bokeh.layouts import column, row
from bokeh.models import (
    CheckboxGroup, ColumnDataSource, CustomAction, CustomJS, Div, GroupBox,
    HoverTool, InlineStyleSheet, Range1d, RangeSlider, Select, Toggle,
)
from bokeh.events import DocumentReady
from bokeh.plotting import figure


EVENT_TYPES = ("collision", "fragmentation", "escape")
COLORS = {"collision": "#2563eb", "fragmentation": "#e11d48", "escape": "#059669"}
SCHEMATIC_WEIGHTS = (2, 1, 1, 3, 2)
REGIONS = (
    "First chamber", "Skimmer", "Skimmer to quadrupole",
    "Quadrupole", "Second chamber",
)


@dataclass(frozen=True)
class Event:
    id: int
    type: str
    realization_id: int
    t: float
    x: float
    y: float
    z: float
    pathway_id: int | None = None

    @property
    def radial(self):
        return sqrt(self.x * self.x + self.y * self.y)


@dataclass
class Realization:
    id: int
    experiment_result_id: int
    events: list[Event] = field(default_factory=list)
    fate: str = "Incomplete"
    terminal: Event | None = None


@dataclass
class Cohort:
    realizations: list[Realization]
    boundaries: tuple[float, ...]
    pathways: dict[int, str]


def classify(realization: Realization):
    """Only one terminal event establishes a fate; all other cases are visible."""
    realization.events.sort(key=lambda e: (e.t, e.id))
    terminals = [e for e in realization.events if e.type in ("fragmentation", "escape")]
    if len(terminals) != 1 or realization.events[-1] != terminals[0]:
        realization.fate = "Ambiguous" if len(terminals) > 1 else "Incomplete"
        return
    terminal = terminals[0]
    if terminal.type == "fragmentation" and terminal.pathway_id is None:
        realization.fate = "Incomplete"
        return
    realization.terminal = terminal
    realization.fate = "Escaped" if terminal.type == "escape" else f"Pathway {terminal.pathway_id}"


def load_cohort(db, experiment: int, cluster: int) -> Cohort:
    """Resolve run -> result -> realization, then fetch every event by stable ID."""
    connection = db.db
    results = connection.execute("""
        select id, cluster_id from multi_pathway_experiment_result
        where experiment_run_id = ? and cluster_id = ?
        union all
        select s.id, p.cluster_id from single_pathway_experiment_result s
        join pathway p on p.id = s.pathway_id
        where s.experiment_run_id = ? and p.cluster_id = ?
    """, (experiment, cluster, experiment, cluster)).fetchall()
    result_ids = {int(row[0]) for row in results}
    placeholders = ",".join("?" for _ in result_ids)
    realization_rows = (connection.execute(
        f"select id, experiment_result_id from realization where experiment_result_id in ({placeholders}) order by id",
        tuple(result_ids)).fetchall() if result_ids else [])
    realizations = [Realization(int(rid), int(result_id)) for rid, result_id in realization_rows]
    by_id = {r.id: r for r in realizations}
    if by_id:
        # The event tables share one ID sequence. UNION ALL preserves duplicate
        # positions and still gives each event its own stable identity.
        events = connection.execute(f"""
            select * from (
            select id, realization_id, 'collision' as kind, postime, null::integer as pathway_id from collision_event
            union all select id, realization_id, 'fragmentation', postime, pathway_id from fragmentation_event
            union all select id, realization_id, 'escape', postime, null::integer from escape_event
            ) event where realization_id in
            (select id from realization where experiment_result_id in ({placeholders}))
        """, tuple(result_ids)).fetchall()
        for event_id, rid, kind, pos, pathway_id in events:
            if rid in by_id:
                by_id[rid].events.append(Event(int(event_id), kind, int(rid),
                    float(pos["t"]), float(pos["x"]), float(pos["y"]),
                    float(pos["z"]), int(pathway_id) if pathway_id is not None else None))
    for realization in realizations:
        classify(realization)
    pathway_rows = connection.execute("""
        select p.id, a.common_name, b.common_name from pathway p
        join cluster a on a.id = p.product1_id
        join cluster b on b.id = p.product2_id
        where p.cluster_id = ?
    """, (cluster,)).fetchall()
    pathways = {int(pid): f"{first} + {second}" for pid, first, second in pathway_rows}
    # ConfigFile's length order is L0, L1, L2, L3, Lsk. This order follows
    # the machine: first chamber, skimmer, gap, quadrupole, second chamber.
    from apitofsim.plotting.events import get_geometery, lengths_to_cumulative_lengths
    boundaries = tuple(float(v) for v in lengths_to_cumulative_lengths(get_geometery(db, experiment)))
    for realization in realizations:
        if (realization.terminal and realization.terminal.type == "fragmentation"
                and region_index(realization.terminal.z, boundaries) is None):
            realization.fate = "Incomplete"
            realization.terminal = None
    return Cohort(realizations, boundaries, pathways)


class Coordinates:
    def __init__(self, boundaries: tuple[float, ...], mode: str):
        self.physical = boundaries
        self.mode = mode
        widths = SCHEMATIC_WEIGHTS if mode == "schematic" else (1,) * (len(boundaries) - 1)
        mapped = [0.0]
        for width in widths:
            mapped.append(mapped[-1] + width)
        self.boundaries = boundaries if mode == "physical" else tuple(mapped)
        last_width = self.boundaries[-1] - self.boundaries[-2]
        self.slot = (self.boundaries[-1], self.boundaries[-1] + max(last_width * .65, .5))

    def position(self, event: Event):
        if self.mode == "time":
            return event.t
        if self.mode == "physical":
            return event.z
        if event.type == "escape":
            return sum(self.slot) / 2
        z = event.z
        for i, (lo, hi) in enumerate(zip(self.physical, self.physical[1:])):
            if z <= hi or i == len(self.physical) - 2:
                if hi == lo:
                    return self.boundaries[i]
                return self.boundaries[i] + (z - lo) * (self.boundaries[i + 1] - self.boundaries[i]) / (hi - lo)
        return self.boundaries[0]


def region_index(z: float, boundaries: tuple[float, ...]):
    """Half-open intervals except for the final included endpoint."""
    for index in range(len(boundaries) - 1):
        if boundaries[index] <= z < boundaries[index + 1] or (index == len(boundaries) - 2 and z == boundaries[-1]):
            return index
    return None


def aggregate(realizations: list[Realization], coordinates: Coordinates):
    n = len(realizations)
    if not n:
        return [], [], [], 0, 0
    fragmented = sorted(coordinates.position(r.terminal) for r in realizations
                        if r.terminal and r.terminal.type == "fragmentation")
    escaped = sum(r.fate == "Escaped" for r in realizations)
    incomplete = n - len(fragmented) - escaped
    # A vertical step at each unique coordinate, with tied events combined.
    xs, ys = [], []
    count = 0
    for x, multiplicity in sorted(Counter(fragmented).items()):
        xs.extend((x, x))
        ys.extend((count / n, (count + multiplicity) / n))
        count += multiplicity
    bars = [0] * (len(coordinates.physical) - 1)
    for r in realizations:
        if r.terminal and r.terminal.type == "fragmentation":
            index = region_index(r.terminal.z, coordinates.physical)
            if index is not None:
                bars[index] += 1
    return xs, ys, [v / n for v in bars] + [escaped / n], escaped, incomplete


def _layout_y(events: list[Event], positions: list[float], mode: str):
    if mode == "radial":
        return [e.radial for e in events]
    if mode == "strip":
        return [.25 * sin(e.id * 12.9898) for e in events]
    # Stable greedy packing in X order; the strip width follows the visible span.
    span = max(positions, default=1) - min(positions, default=0)
    radius = max(span / 160, 1e-9)
    packed = {}
    result = [0.0] * len(events)
    for i in sorted(range(len(events)), key=lambda j: (positions[j], events[j].id)):
        x = positions[i]
        nearby = [(px, py) for px, py in packed.values() if abs(px - x) < 2 * radius]
        levels = [0, 1, -1, 2, -2, 3, -3]
        level = next((v for v in levels if all((x - px) ** 2 / radius ** 2 + (v - py) ** 2 >= 1 for px, py in nearby)), len(nearby) + 1)
        result[i] = float(level) * .18
        packed[events[i].id] = (x, level)
    return result


def _violin(events: list[Event], positions: list[float]):
    continuous = [x for e, x in zip(events, positions) if e.type != "escape"]
    if len(continuous) < 2 or min(continuous) == max(continuous):
        return [], []
    lo, hi = min(continuous), max(continuous)
    bandwidth = max((hi - lo) / 18, 1e-12)
    grid = [lo + (hi - lo) * i / 79 for i in range(80)]
    density = [sum(exp(-.5 * ((x - v) / bandwidth) ** 2) for v in continuous) for x in grid]
    peak = max(density)
    upper = [.4 * value / peak for value in density]
    return grid + grid[::-1], upper + [-v for v in upper[::-1]]


def _details(realization: Realization | None, selected_event: int | None, pathways: dict[int, str]):
    if realization is None:
        return "<p>Click an event to inspect its realization.</p>"
    rows = []
    for event in realization.events:
        fields = (f"Event #{event.id} · {event.type} · t={event.t:.6g} s · "
                  f"x={event.x:.6g} m · y={event.y:.6g} m · z={event.z:.6g} m · "
                  f"r={event.radial:.6g} m")
        if event.pathway_id is not None:
            fields += f" · pathway #{event.pathway_id} ({pathways.get(event.pathway_id, 'unknown')})"
        rows.append(f'<button class="explorer-event" data-event="{event.id}" id="event-{event.id}" '
                    f'style="display:block;width:100%;text-align:left;padding:8px;'
                    f'background:{"#fde68a" if event.id == selected_event else "white"};border-bottom:1px solid #ddd">'
                    f'{escape(fields)}</button>')
    return (f"<h3>Realization #{realization.id}</h3><p>Result #{realization.experiment_result_id}<br>"
            f"Fate: {escape(realization.fate)}</p><button id='clear-selection'>Clear selection</button>"
            + "".join(rows))


def build_document(db, doc, experiment: int, cluster: int):
    """Build one Bokeh session; every control uses the same cohort and X mapping."""
    from bokeh.models import Label, Span

    try:
        cohort = load_cohort(db, experiment, cluster)
    except Exception as exc:
        # Non-realization databases and older configs have no Explorer data.
        doc.add_root(Div(text=f"Explorer data unavailable: {escape(str(exc))}"))
        return

    fate_names = sorted({r.fate for r in cohort.realizations})
    if not fate_names:
        fate_names = ["Escaped"]
    state = {"selected": None, "event": None, "updating": False, "auto_cdf": False,
             "syncing_sidebar": False,
             "sidebar": {False: [True, False], True: [False, False]}}

    counts = Div(styles={"white-space": "nowrap"})
    quality = Div()
    fate = CheckboxGroup(labels=fate_names, active=list(range(len(fate_names))))
    def checkbox(label, checked=False):
        return CheckboxGroup(labels=[label], active=[0] if checked else [])

    def enabled(widget):
        return bool(widget.active)

    fate_facets = checkbox("Facet fates")
    event_types = CheckboxGroup(labels=[v.title() for v in EVENT_TYPES], active=[0, 1, 2])
    quick = Select(title="Event preset", value="All", options=["All", "Collision only", "Fragmentation only", "Collision + fragmentation", "Escape only"])
    mode = Select(title="X coordinate", value="schematic", options=[
        ("schematic", "Schematic distance"), ("equal", "Equal regions"),
        ("physical", "Physical distance (m)"), ("time", "Elapsed time (s)")])
    layout = Select(title="Event layout", value="radial", options=[
        ("radial", "Radial distance"), ("strip", "Jittered strip"), ("beeswarm", "Beeswarm")])
    violin = checkbox("Violin envelope")
    schematic = checkbox("Schematic / guides", True)
    restrictions = Div()
    show_events = checkbox("Realizations", True)
    show_cdf = checkbox("CDF")
    show_bars = checkbox("Regional bars")
    total = len(cohort.realizations)
    selector = RangeSlider(title="Realization IDs (inclusive index range)",
                           start=1, end=max(total, 1), step=1, value=(1, max(total, 1)),
                           disabled=total == 0, sizing_mode="stretch_width", height=50)
    details = Div(text=_details(None, None, cohort.pathways), sizing_mode="stretch_width")
    detail_bridge = Div(text="", visible=False)
    fullscreen_state = Div(text="normal", visible=False)
    viewport_width = Div(text="1440,900", visible=False)
    def group(title, child):
        child.width_policy = "max"
        child.height_policy = "min"
        return GroupBox(title=title, child=child, width_policy="max", height_policy="min",
                        margin=(5, 0),
                        stylesheets=[InlineStyleSheet(css="fieldset { min-inline-size: 0; }")])

    left_controls = column(
        group("Fates", column(fate, fate_facets, spacing=4)),
        group("Events", column(event_types, quick, spacing=4)),
        group("X axis", column(mode, schematic, restrictions, spacing=4)),
        group("Layout", column(layout, violin, spacing=4)),
        group("Views", column(row(show_events, show_cdf), show_bars, spacing=4)),
        spacing=4, styles={"max-height": "calc(100vh - 295px)", "overflow-y": "auto"},
    )
    right_controls = column(group("Selected realization", details),
                            styles={"max-height": "calc(100vh - 295px)", "overflow-y": "auto"})
    left_toggle = Toggle(label="« Controls", active=True, width=100, height=32)
    right_toggle = Toggle(label="☷", active=False, width=36, height=32,
                          html_attributes={"title": "Show selected realization"})
    left = column(left_toggle, left_controls, width=205, spacing=4)
    right = column(right_toggle, right_controls, width=38, spacing=4)
    right_controls.visible = False
    center_plots = column(sizing_mode="stretch_width", spacing=4)
    center = column(counts, quality, center_plots, selector,
                    sizing_mode="stretch_width", spacing=2,
                    css_classes=["explorer-center"])
    frame = column(row(left, center, right, sizing_mode="stretch_width", spacing=6),
                   detail_bridge, fullscreen_state, viewport_width,
                   sizing_mode="stretch_width", spacing=4,
                   css_classes=["explorer-bokeh-frame"],
                   stylesheets=[InlineStyleSheet(css="""
                       :host { background: white; }
                       :host(:fullscreen) {
                           width: 100vw !important; height: 100vh !important;
                           overflow: auto; padding: 6px;
                       }
                   """)])
    doc.add_root(frame)
    doc.js_on_event(DocumentReady, CustomJS(args=dict(width=viewport_width), code="""
        const update = () => { width.text = `${window.innerWidth},${window.innerHeight}`; };
        update();
        window.addEventListener('resize', update);
    """))

    details.js_on_change("text", CustomJS(args=dict(bridge=detail_bridge), code="""
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
    """))

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
        left_toggle.width, right_toggle.width = (100 if left_open else 32), (110 if right_open else 32)
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
        sidebar_height = "calc(100vh - 46px)" if new == "full" else "calc(100vh - 295px)"
        left_controls.styles = {**left_controls.styles, "max-height": sidebar_height}
        right_controls.styles = {**right_controls.styles, "max-height": sidebar_height}
        apply_sidebars(new == "full")
    fullscreen_state.on_change("text", full_changed)
    viewport_width.on_change("text", lambda attr, old, new: apply_sidebars(fullscreen_state.text == "full"))

    def selected_cohort():
        lo, hi = (int(v) for v in selector.value)
        chosen_fates = {fate_names[i] for i in fate.active}
        return [r for r in cohort.realizations[lo - 1:hi] if r.fate in chosen_fates]

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
            selected = next((r for r in cohort.realizations if r.id == state["selected"]), None)
            if selected and any(e.id == int(new) for e in selected.events):
                set_selection(selected.id, int(new))
    detail_bridge.on_change("text", sidebar_event)

    def prepare_toolbar(plot):
        """Use Bokeh's fullscreen toolbar icon for the entire explorer frame."""
        plot.toolbar.logo = None
        plot.add_tools(CustomAction(description="Fullscreen explorer", icon="fullscreen",
            callback=CustomJS(args=dict(frame=frame, fullscreen_state=fullscreen_state), code="""
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
            """)))
        plot.js_on_change("inner_width", CustomJS(args=dict(toolbar=plot.toolbar), code="""
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
        """))

    def render():
        if state["updating"]:
            return
        state["updating"] = True
        try:
            chosen = selected_cohort()
            chosen_ids = {r.id for r in chosen}
            if state["selected"] not in chosen_ids:
                state["selected"] = state["event"] = None
            selected = next((r for r in chosen if r.id == state["selected"]), None)
            details.text = _details(selected, state["event"], cohort.pathways)
            counts.text = f"<b>{len(chosen)} selected / {total} total realizations</b>"
            incomplete = sum(r.fate in ("Incomplete", "Ambiguous") for r in chosen)
            quality.text = (f"<span style='color:#a21caf'>{incomplete} incomplete or ambiguous histories; "
                            "unresolved outcomes are not counted as escaped.</span>" if incomplete else "")
            active_types = {EVENT_TYPES[i] for i in event_types.active}
            coordinates = Coordinates(cohort.boundaries, mode.value)
            spatial = mode.value != "time"
            regional = mode.value in ("schematic", "equal")
            if spatial:
                restrictions.text = "" if regional else "Regional bars require schematic or equal regions. Physical mode shows boundary guides."
            else:
                restrictions.text = "Schematic and regional bars are unavailable in elapsed time."
            show_bars.disabled = not regional
            schematic.disabled = not spatial
            violin.disabled = layout.value == "radial"
            if regional and state["auto_cdf"] and enabled(show_bars):
                show_cdf.active = []
                state["auto_cdf"] = False
            if not any((enabled(show_events), enabled(show_cdf), enabled(show_bars) and regional)):
                state["auto_cdf"] = enabled(show_bars) and not regional
                show_cdf.active = [0]
            rows = ([([r for r in chosen if r.fate == name], name) for name in fate_names if name in {fate_names[i] for i in fate.active}]
                    if enabled(fate_facets) else [(chosen, "All selected fates")])
            all_positions = [coordinates.position(e) for r in chosen for e in r.events]
            if spatial:
                left_edge = coordinates.boundaries[0]
                right_edge = coordinates.slot[1] if regional else max(coordinates.boundaries[-1], max(all_positions, default=coordinates.boundaries[-1]))
            else:
                left_edge, right_edge = 0, max(all_positions, default=1)
            if right_edge <= left_edge:
                right_edge = left_edge + 1
            screen_width, screen_height = (int(v) for v in viewport_width.text.split(","))
            available_width = screen_width - left.width - right.width
            plot_height = (max(440, min(760, screen_height - 175 -
                (175 if enabled(show_cdf) else 0) - (155 if enabled(show_bars) and regional else 0)))
                if fullscreen_state.text == "full" else 440)
            shared_x = Range1d(left_edge, right_edge + (right_edge - left_edge) * .015)
            panels = []
            axis_figures = []
            if enabled(schematic) and spatial and regional and chosen:
                diagram = figure(height=56, min_height=56, x_range=shared_x, toolbar_location=None,
                                 output_backend="webgl", sizing_mode="stretch_width")
                diagram.toolbar.logo = None
                diagram.y_range = Range1d(0, 1)
                diagram.yaxis.visible = False
                diagram.xaxis.visible = False
                region_source = ColumnDataSource(dict(
                    left=list(coordinates.boundaries[:-1]),
                    right=list(coordinates.boundaries[1:]),
                    name=list(REGIONS),
                    color=["#dbeafe", "#e2e8f0", "#dbeafe", "#e2e8f0", "#dbeafe"],
                ))
                region_glyph = diagram.quad(left="left", right="right", bottom=.12, top=.88,
                    source=region_source, fill_color="color", line_color="#64748b")
                diagram.add_tools(HoverTool(renderers=[region_glyph], tooltips=[("Region", "@name")]))
                for i, name in enumerate(REGIONS):
                    lo, hi = coordinates.boundaries[i:i+2]
                    display_name = (("C1", "Sk", "Gap", "Quad", "C2")[i]
                                    if available_width < 600 else name)
                    diagram.add_layout(Label(x=(lo + hi) / 2, y=.5, text=display_name,
                                             text_align="center", text_font_size="9px"))
                diagram.add_layout(Label(x=sum(coordinates.slot)/2, y=.5,
                    text="Esc" if available_width < 600 else "Escaped", text_align="center"))
                panels.append(diagram)
            for members, row_name in rows:
                if not members and not chosen:
                    continue
                if enabled(show_events):
                    events = [e for r in members for e in r.events if e.type in active_types]
                    positions = [coordinates.position(e) for e in events]
                    yvalues = _layout_y(events, positions, layout.value)
                    plot_title = "Realizations" if not enabled(fate_facets) else row_name
                    plot = figure(title=plot_title,
                                  height=plot_height, min_height=360, min_width=210,
                                  x_range=shared_x, sizing_mode="stretch_width",
                                  output_backend="webgl", tools="pan,wheel_zoom,box_zoom,reset,save,tap")
                    prepare_toolbar(plot)
                    plot.yaxis.axis_label = "Radial distance (m)" if layout.value == "radial" else "Event layout"
                    if enabled(schematic) and mode.value == "physical":
                        for i, boundary in enumerate(coordinates.boundaries):
                            plot.add_layout(Span(location=boundary, dimension="height", line_color="#94a3b8", line_dash="dotted"))
                            plot.add_layout(Label(x=boundary, y=0, y_units="screen",
                                text=("Start" if i == 0 else REGIONS[i-1] + " end"),
                                text_font_size="8px", text_color="#475569", angle=1.57))
                    if enabled(violin) and layout.value != "radial":
                        vx, vy = _violin(events, positions)
                        if vx:
                            plot.patch(vx, vy, fill_alpha=.13, line_alpha=.25, color="#64748b")
                    for kind in EVENT_TYPES:
                        indices = [i for i, e in enumerate(events) if e.type == kind]
                        if not indices:
                            continue
                        source = ColumnDataSource(dict(x=[positions[i] for i in indices], y=[yvalues[i] for i in indices],
                            id=[events[i].id for i in indices], realization=[events[i].realization_id for i in indices],
                            time=[events[i].t for i in indices], z=[events[i].z for i in indices]))
                        glyph = plot.scatter("x", "y", source=source, marker="x", size=10,
                                             color=COLORS[kind], legend_label=kind.title(), line_width=1.5)
                        plot.add_tools(HoverTool(renderers=[glyph], tooltips=[("Event", "@id"),
                            ("Realization", "@realization"), ("t (s)", "@time"), ("z (m)", "@z")]))
                        def on_selected(attr, old, new, source=source):
                            if new and not state["updating"]:
                                i = new[0]
                                set_selection(source.data["realization"][i], source.data["id"][i])
                        source.selected.on_change("indices", on_selected)
                    if selected:
                        visible = [e for e in selected.events if e in events]
                        if visible:
                            coordinates_by_id = {e.id: (positions[i], yvalues[i]) for i, e in enumerate(events)}
                            visible.sort(key=lambda e: (e.t, e.id))
                            plot.line([coordinates_by_id[e.id][0] for e in visible],
                                      [coordinates_by_id[e.id][1] for e in visible],
                                      color="#111827", line_width=2.5)
                            plot.scatter([coordinates_by_id[e.id][0] for e in visible],
                                         [coordinates_by_id[e.id][1] for e in visible],
                                         marker="circle", size=14, fill_alpha=0, line_color="#111827", line_width=2)
                            clicked = coordinates_by_id.get(state["event"])
                            if clicked:
                                plot.scatter([clicked[0]], [clicked[1]], marker="circle", size=20,
                                             fill_alpha=0, line_color="#f59e0b", line_width=3)
                    if plot.legend:
                        plot.legend.click_policy = "hide"
                        plot.legend.location = "top_left"
                    plot.toolbar.logo = None
                    panels.append(plot)
                    axis_figures.append(plot)
                xs, ys, bars, escaped, unresolved = aggregate(members, coordinates)
                if enabled(show_cdf):
                    cdf = figure(title=f"Fragmentation CDF · {row_name} (N={len(members)})", height=175, min_height=150,
                                 x_range=shared_x, y_range=Range1d(0, 1), output_backend="webgl",
                                 sizing_mode="stretch_width", tools="pan,wheel_zoom,reset,save")
                    prepare_toolbar(cdf)
                    cdf.yaxis.axis_label = "Fraction of all realizations"
                    if members:
                        cdf.step([left_edge, *xs, right_edge], [0, *ys, ys[-1] if ys else 0],
                                 mode="after", line_width=2, color="#be123c")
                        if spatial and escaped:
                            bracket_x = sum(coordinates.slot) / 2 if regional else right_edge
                            start = (len(members) - escaped - unresolved) / len(members)
                            end = 1 if not unresolved else start + escaped / len(members)
                            cdf.line([bracket_x, bracket_x], [start, end], color="#059669", line_width=3)
                            cdf.add_layout(Label(x=bracket_x, y=(start + end)/2, text=f"Escaped {escaped/len(members):.1%}",
                                                 text_color="#047857"))
                    panels.append(cdf)
                    axis_figures.append(cdf)
                if enabled(show_bars) and regional:
                    top = max(bars, default=0)
                    bar = figure(title=f"Regional outcomes · {row_name} (N={len(members)})", height=155, min_height=130,
                                 x_range=shared_x, y_range=Range1d(0, top * 1.08 if top else 1),
                                 output_backend="webgl", sizing_mode="stretch_width",
                                 tools="pan,wheel_zoom,reset,save")
                    prepare_toolbar(bar)
                    bar.yaxis.axis_label = "Fraction of all realizations"
                    for i, value in enumerate(bars):
                        lo, hi = (coordinates.slot if i == len(bars)-1 else coordinates.boundaries[i:i+2])
                        bar.quad(left=lo, right=hi, bottom=0, top=value,
                                 fill_color="#059669" if i == len(bars)-1 else "#6366f1", fill_alpha=.65)
                        bar.add_layout(Label(x=(lo + hi) / 2, y=value,
                            text="Escaped" if i == len(bars)-1 else REGIONS[i],
                            text_align="center", text_font_size="8px"))
                    panels.append(bar)
                    axis_figures.append(bar)
            if axis_figures:
                for upper in axis_figures[:-1]:
                    upper.xaxis.visible = False
                axis_figures[-1].xaxis.axis_label = (
                    "Elapsed time (s)" if not spatial else
                    "Axial distance (m)" if mode.value == "physical" else "Machine position")
            if not chosen:
                panels.insert(0, Div(text="<p>No realizations in this cohort. Adjust the range or fate filters.</p>"))
            if selected and state["event"] is not None:
                highlighted = next((e for e in selected.events if e.id == state["event"]), None)
                if highlighted and highlighted.type not in active_types:
                    panels.insert(0, Div(text=(f"<b>Selected hidden event #{highlighted.id}</b> · "
                        f"{highlighted.type} · t={highlighted.t:.6g} s · z={highlighted.z:.6g} m")))
            center_plots.children = panels
        finally:
            state["updating"] = False

    def control_changed(attr, old, new):
        render()

    def cdf_changed(attr, old, new):
        if not state["updating"]:
            state["auto_cdf"] = False
        render()

    def preset_changed(attr, old, new):
        presets = {"All": [0, 1, 2], "Collision only": [0], "Fragmentation only": [1],
                   "Collision + fragmentation": [0, 1], "Escape only": [2]}
        event_types.active = presets[new]

    quick.on_change("value", preset_changed)
    for widget, property_name in [(fate, "active"), (fate_facets, "active"),
        (event_types, "active"), (mode, "value"),
        (layout, "value"), (violin, "active"), (schematic, "active"),
        (show_events, "active"), (show_bars, "active"),
        (selector, "value")]:
        widget.on_change(property_name, control_changed)
    show_cdf.on_change("active", cdf_changed)
    render()
