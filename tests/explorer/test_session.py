"""Explorer session behavior."""

import json

import polars as pl
import pytest
from bokeh.document import Document
from bokeh.models import (
    Button,
    CheckboxGroup,
    ColumnDataSource,
    Div,
    FullscreenTool,
    GroupBox,
    HoverTool,
    Label,
    PanTool,
    RangeSlider,
    Select,
)
from bokeh.palettes import Category10
from bokeh.plotting import figure

from apitofresview.plotting.explorer import build_document, data, plot, session
from apitofresview.plotting.explorer import layout as event_layout

from .helpers import cohort, event


def test_realization_coordinate(monkeypatch):
    frames = cohort(
        [100, 102, 105, 110],
        event(1, "collision", 100, 0, 0),
        event(2, "collision", 102, 0, 0),
        event(3, "collision", 102, 1, 1),
        event(4, "escape", 105, 1, 5),
    )
    monkeypatch.setattr(session, "load_data", lambda *args: frames)
    doc = Document()
    build_document(None, doc, 1, 1)
    mode = next(s for s in doc.select({"type": Select}) if s.title == "Y coordinate")
    assert mode.value == "realization"
    assert mode.options[0] == ("realization", "Realization #")

    def check(expected, endpoints):
        events = next(p for p in doc.select({"type": figure}) if p.name == "events")
        points = {
            id: y
            for renderer in events.renderers
            if hasattr(renderer, "data_source") and "id" in renderer.data_source.data
            for id, y in zip(
                renderer.data_source.data["id"], renderer.data_source.data["y"]
            )
        }
        assert points == expected
        labels = {
            id: number
            for renderer in events.renderers
            if hasattr(renderer, "data_source") and "id" in renderer.data_source.data
            for id, number in zip(
                renderer.data_source.data["id"],
                renderer.data_source.data["realization"],
            )
        }
        assert labels == expected
        axis = events.yaxis[0]
        assert axis.axis_label == "Realization #"
        assert axis.ticker.ticks == endpoints
        assert axis.major_tick_line_color is None
        assert axis.minor_tick_line_color is None
        assert not events.ygrid[0].visible

    check({1: 1, 2: 2, 3: 2, 4: 3}, [1, 4])
    pager = next(
        w
        for w in doc.select({"type": CheckboxGroup})
        if w.labels == ["Realizations pager"]
    )
    source = next(
        r.data_source
        for r in doc.select_one({"name": "events"}).renderers
        if "id" in r.data_source.data
    )
    source.selected.indices = [source.data["id"].index(2)]
    details = next(
        d for d in doc.select({"type": Div}) if "Realization #2</h3>" in d.text
    )
    pager.active = [0]
    next(iter(doc.select({"type": RangeSlider}))).value = (2, 4)
    check({2: 2, 3: 2, 4: 3}, [2, 4])
    assert "Realization #2</h3>" in details.text
    pager.active = []
    check({1: 1, 2: 2, 3: 2, 4: 3}, [1, 4])
    fate = next(
        w for w in doc.select({"type": CheckboxGroup}) if "Incomplete" in w.labels
    )
    fate.active = [fate.labels.index("Parent → Parent")]
    check({4: 3}, [3])


def test_overlapping_event_choice_and_stale_candidates(monkeypatch):
    frames = cohort(
        [100, 101],
        event(1, "collision", 100, 2e-9, 0.003),
        event(2, "collision", 101, 2e-9, 0.003),
        event(3, "escape", 101, 2e-9, 0.003),
    )
    monkeypatch.setattr(session, "load_data", lambda *args: frames)
    doc = Document()
    build_document(None, doc, 1, 1)
    chart = doc.select_one({"name": "events"})
    source = next(r.data_source for r in chart.renderers if "id" in r.data_source.data)
    assert len([r for r in chart.renderers if "id" in r.data_source.data]) == 1
    hover = next(
        tool
        for tool in chart.tools
        if isinstance(tool, HoverTool) and tool.renderers[0].data_source is source
    )
    assert "<span>@realization</span>" in hover.tooltips
    assert source.data["realization"] == [1, 2, 2]
    assert source.data["realization_id"] == [100, 101, 101]
    assert source.data["time_text"] == ["2"] * 3
    assert source.data["axial_text"] == ["3"] * 3

    source.selected.indices = [0, 1, 2]
    details = next(d for d in doc.select({"type": Div}) if "Choose an event" in d.text)
    assert details.text.count('class="explorer-choice"') == 3
    assert details.text.index('data-event="1"') < details.text.index('data-event="2"')
    assert "<th>#</th><th>Type</th>" in details.text
    bridge = doc.roots[0].children[1]
    bridge.text = "choose:101:3"
    assert "Realization #2</h3>" in details.text
    assert 'id="event-3" style="background:#fde68a' in details.text

    source.selected.indices = [0, 1]
    assert "Choose an event" in details.text
    next(
        w for w in doc.select({"type": Select}) if w.title == "Y coordinate"
    ).value = "beeswarm"
    bridge.text = "choose:100:1"
    assert "Choose an event" not in details.text
    assert "Click an event" in details.text

    chart = doc.select_one({"name": "events"})
    source = next(r.data_source for r in chart.renderers if "id" in r.data_source.data)
    source.selected.indices = [source.data["id"].index(2)]
    assert "Realization #2</h3>" in details.text


def test_bokeh_document_modes_and_views(monkeypatch):
    frames = cohort(
        [1], event(1, "collision", 1, 0, 0, clamped=True), event(2, "escape", 1, 1, 5)
    )
    monkeypatch.setattr(session, "load_data", lambda *args: frames)
    doc = Document()
    build_document(None, doc, 1, 1)
    assert any(
        d.text == "1 events with negative z position have been clamped to 0"
        for d in doc.select({"type": Div})
    )
    x_mode = next(s for s in doc.select({"type": Select}) if s.title == "X coordinate")
    assert x_mode.options[0] == ("schematic", "Schematic")
    sizing = next(s for s in doc.select({"type": Select}) if s.title == "Sizing")
    assert sizing.options == [
        ("chambers", "Equal chambers"),
        ("zones", "Equal zones"),
        ("custom", "Custom"),
    ]

    for value, preset in (
        ("schematic", "zones"),
        ("physical", None),
        ("time", None),
        ("schematic", "chambers"),
    ):
        x_mode.value = value
        assert sizing.visible == (value == "schematic")
        assert not sizing.disabled
        if preset:
            sizing.value = preset
            regions = event_layout.coordinate_regions(
                frames[3], value, event_layout.schematic_widths(preset)
            )
            widths = regions["right"] - regions["left"]
            if preset == "chambers":
                assert widths[0] == widths[1] == widths[2:].sum()
            centers = ((regions["left"] + regions["right"]) / 2).to_list()
            for p in doc.select({"type": type(figure())}):
                if p.xaxis[0].visible:
                    assert p.xaxis[0].axis_label == "Zone"
                    assert p.xaxis[0].ticker.ticks == centers
                    assert p.xaxis[0].major_label_overrides == dict(
                        zip(centers, ["1", "2", "3", "4", "5"], strict=True)
                    )
                    assert p.xgrid[0].ticker.ticks == [
                        regions["left"][0],
                        *regions["right"],
                    ]
                    assert p.xgrid[0].grid_line_alpha == 0.3
    for toggle in doc.select({"type": CheckboxGroup}):
        if toggle.labels and toggle.labels[0] in (
            "Cumulative",
            "Bar chart",
            "Realizations",
        ):
            toggle.active = [0]
    frame = doc.roots[0]
    left, workspace = frame.children[0].children
    sidebars = [left, *workspace.children]
    fullscreen = next(d for d in doc.select({"type": Div}) if d.text == "normal")
    fullscreen.text = "full"
    assert not sidebars[0].children[1].visible and not sidebars[2].children[1].visible
    assert sidebars[0].styles["height"] == "100%"
    fullscreen.text = "normal"
    assert sidebars[0].children[1].visible and not sidebars[2].children[1].visible
    viewport = next(d for d in doc.select({"type": Div}) if d.text == "1440,900")
    viewport.text = "533,900"
    sidebars[2].children[0].children[0].active = True
    assert not sidebars[0].children[1].visible
    assert sidebars[2].width == 250

    plots = list(doc.select({"type": type(figure())}))
    event_plots = [p for p in plots if p.name == "events"]
    assert len(event_plots) == 1
    assert not any(p.title and p.title.text for p in plots)
    assert sum(bool(p.xaxis[0].visible) for p in plots) == 1
    assert {g.title for g in doc.select({"type": GroupBox})} >= {
        "Pathways",
        "Events",
        "X-axis",
        "Y-axis",
        "Views",
        "Elements",
        "Realizations",
    }
    assert all(p.toolbar.logo is None for p in plots)
    assert all(
        any(isinstance(tool, FullscreenTool) for tool in p.tools)
        for p in plots
        if p.toolbar_location is not None
    )
    assert len(doc.roots) == 1
    controls = sidebars[0].children[1].children
    assert isinstance(controls[0], Button)
    assert controls[0].label == "Fullscreen"
    assert controls[0].icon.icon_name == "fullscreen"
    assert [g.title for g in controls[1:]] == [
        "Pathways",
        "X-axis",
        "Elements",
        "Realizations",
        "Rates",
    ]
    assert controls[2].child.children[1] is sizing
    assert controls[2].child.children[2].labels == ["Guides"]
    assert controls[3].child.children[0].labels == ["Schematic"]
    assert controls[3].child.children[1].title == "Views"
    assert controls[4].child.children[0].labels == ["Use x markers"]
    assert controls[4].child.children[1].labels == ["Group fragmentations"]
    assert controls[4].child.children[2].labels == ["Separate rejected collisions"]
    assert controls[4].child.children[3].labels == [
        "Place initial/escape in their own zones"
    ]
    assert controls[4].child.children[4].labels == ["Spread initial/escape X"]
    assert [g.title for g in controls[4].child.children[5:]] == ["Events", "Y-axis"]
    assert not event_plots[0].legend
    assert doc.to_json() is not None


def test_schematic_custom_widths_persist_and_align_plots(monkeypatch):
    frames = cohort(
        [1],
        event(1, "init", 1, 0, 0),
        event(2, "collision", 1, 1, 3.5),
        event(3, "escape", 1, 2, 5),
    )
    monkeypatch.setattr(session, "load_data", lambda *args: frames)
    doc = Document()
    build_document(None, doc, 1, 1)
    sizing = next(s for s in doc.select({"type": Select}) if s.title == "Sizing")
    coordinate = next(
        s for s in doc.select({"type": Select}) if s.title == "X coordinate"
    )
    custom = (0.75, 2.5, 2.5, 1.25, 1.0, 1.75, 0.55)
    assert sum(custom) == sum(event_layout.schematic_widths("chambers"))
    doc.select_one({"name": "schematic_drag_bridge"}).text = json.dumps(custom)
    assert sizing.value == "custom"

    def check():
        schematic = doc.select_one({"name": "schematic"})
        source = schematic.renderers[0].data_source.data
        assert [
            right - left
            for left, right in zip(source["left"], source["right"], strict=True)
        ] == pytest.approx(custom)
        assert schematic.outline_line_color is None
        assert not any(isinstance(tool, PanTool) for tool in schematic.tools)
        assert source["alpha"][3:6] == [0, 0, 0]
        chamber = schematic.renderers[1].data_source.data
        quadrupole = schematic.renderers[2].data_source.data
        assert chamber["left"] == [source["left"][3]]
        assert chamber["right"] == [source["right"][5]]
        assert quadrupole["left"] == [source["left"][4]]
        assert quadrupole["right"] == [source["right"][4]]
        assert schematic.select_one({"type": HoverTool}).renderers == [
            schematic.renderers[0]
        ]
        assert source["name"][3:6] == list(data.REGIONS[2:5])
        assert len([item for item in schematic.center if isinstance(item, Label)]) == 7
        assert source["top"][4] - source["bottom"][4] == pytest.approx(
            0.8 * (source["top"][3] - source["bottom"][3])
        )
        events = doc.select_one({"name": "events"})
        assert events.xgrid[0].ticker.ticks == pytest.approx(
            [source["left"][1], *source["right"][1:6]]
        )
        return next(
            r.data_source.data["x"][r.data_source.data["id"].index(2)]
            for r in events.renderers
            if 2 in r.data_source.data.get("id", [])
        )

    custom_position = check()
    coordinate.value = "physical"
    assert not sizing.visible
    coordinate.value = "schematic"
    assert sizing.visible and sizing.value == "custom"
    assert check() == pytest.approx(custom_position)
    sizing.value = "zones"
    assert sizing.value == "zones"
    equal = doc.select_one({"name": "schematic"}).renderers[0].data_source.data
    assert [
        right - left for left, right in zip(equal["left"], equal["right"], strict=True)
    ] == pytest.approx(event_layout.schematic_widths("zones"))


def test_bokeh_filtering_selection_facets_and_restrictions(monkeypatch):
    frames = cohort(
        [1, 2, 3],
        event(1, "collision", 1, 0, 0),
        event(2, "fragmentation", 1, 1, 2, 7),
        event(3, "escape", 2, 2, 5),
        event(4, "collision", 3, 1, 1),
    )
    frames = (
        *frames[:2],
        pl.DataFrame({"pathway_id": [7], "pathway": ["Parent → A < B + C"]}),
        frames[3],
    )
    loads = []

    def load(*args):
        loads.append(args)
        return frames

    monkeypatch.setattr(session, "load_data", load)
    doc = Document()
    build_document(None, doc, 1, 1)

    def checkbox(label):
        return next(
            w
            for w in doc.select({"type": CheckboxGroup})
            if w.labels and w.labels[0] == label
        )

    def text_contains(text):
        return any(text in d.text for d in doc.select({"type": Div}))

    def plots():
        return list(doc.select({"type": figure}))

    fates = next(
        g.child.children[0]
        for g in doc.select({"type": GroupBox})
        if g.name == "sidebar_pathways"
    )
    assert set(fates.labels) == {"Parent → A + B", "Parent → Parent", "Incomplete"}
    source = next(
        s for s in doc.select({"type": ColumnDataSource}) if 2 in s.data.get("id", [])
    )
    source.selected.indices = [source.data["id"].index(2)]
    assert text_contains("Realization #1")
    assert text_contains("<em>Parent → A + B</em>")
    assert not text_contains("Pathway:")
    assert not text_contains("Parent → A &lt; B + C")
    assert not text_contains("pathway #7")
    details = next(d for d in doc.select({"type": Div}) if "Realization #1" in d.text)
    assert details.text.startswith("<button id='clear-selection'>")
    assert "background:#fde68a" in details.text
    assert "<th>Time (ns)</th>" in details.text
    assert "<th>Axial dist. (mm)</th>" in details.text
    assert "<th>Radial dist. (mm)</th>" in details.text
    radial = checkbox("Summarise non-axial motion")
    assert radial.visible and radial.active == [0]
    radial.active = []
    assert all(f"<th>{axis} (mm)</th>" in details.text for axis in ("x", "y", "z"))
    assert "Axial dist." not in details.text and "Radial dist." not in details.text
    radial.active = [0]
    assert any(
        r.glyph.line_color == "#f59e0b"
        for p in plots()
        for r in p.renderers
        if hasattr(r, "glyph")
    )
    bridge = doc.roots[0].children[1]
    bridge.text = "1"
    assert 'id="event-1" style="background:#fde68a' in details.text
    bridge.text = "999"
    assert 'id="event-1" style="background:#fde68a' in details.text
    checkbox("Collision").active = []
    assert text_contains("Selected hidden event #1")
    checkbox("Collision").active = [0]
    assert not text_contains("Selected hidden event #1")
    checkbox("Cumulative").active = [0]
    checkbox("Bar chart").active = [0]
    checkbox("Facet pathways").active = [0]
    assert {p.title.text for p in plots() if p.title and p.title.text} == {
        f"{name} (N=1)" for name in fates.labels
    }
    views = [p for p in plots() if p.name in {"events", "cdf", "bars"}]
    assert len(views) == 3 * len(fates.labels)
    assert len({p.x_range for p in views}) == 1
    assert sum(bool(p.xaxis[0].visible) for p in views) == len(fates.labels)
    assert all(p.min_border_top == p.min_border_bottom == 0 for p in views)
    for name in ("events", "cdf", "bars"):
        assert all(
            bool(p.xaxis[0].visible) == (name == "bars")
            for p in views
            if p.name == name
        )
    for label in ("Realizations", "Cumulative"):
        checkbox(label).active = []
        assert {p.title.text for p in plots() if p.title and p.title.text} == {
            f"{name} (N=1)" for name in fates.labels
        }
        assert sum(bool(p.xaxis[0].visible) for p in plots()) == len(fates.labels)
    checkbox("Realizations").active = [0]
    checkbox("Cumulative").active = [0]
    fates.active = [fates.labels.index("Parent → Parent")]
    assert text_contains("1 selected / 3 total realizations")
    assert not text_contains("Realization #1")
    assert text_contains("<em>Click an event to inspect its realization.</em>")
    assert not radial.visible
    fates.active = list(range(len(fates.labels)))
    slider = next(iter(doc.select({"type": RangeSlider})))
    pager = checkbox("Realizations pager")
    counts = next(
        d for d in doc.select({"type": Div}) if "total realizations" in d.text
    )
    assert not pager.active and not slider.visible and not counts.visible
    pager.active = [0]
    assert slider.visible and counts.visible
    slider.value = (2, 3)
    assert text_contains("2 selected / 3 total realizations")
    pager.active = []
    assert not slider.visible and slider.value == (2, 3)
    assert not counts.visible
    assert text_contains("3 selected / 3 total realizations")
    assert {p.title.text for p in plots() if p.title and p.title.text} == {
        f"{name} (N=1)" for name in fates.labels
    }
    pager.active = [0]
    assert slider.visible and counts.visible
    assert text_contains("2 selected / 3 total realizations")
    fates.active = [fates.labels.index("Parent → A + B")]
    assert text_contains("No realizations in this cohort")
    slider.value = (1, 3)
    fates.active = list(range(len(fates.labels)))
    checkbox("Facet pathways").active = []
    mode = next(w for w in doc.select({"type": Select}) if w.title == "X coordinate")
    checkbox("Realizations").active = []
    checkbox("Cumulative").active = []
    schematic = checkbox("Schematic")
    guides = checkbox("Guides")
    mode.value = "time"
    assert checkbox("Bar chart").disabled and schematic.visible and schematic.disabled
    assert not guides.visible
    assert checkbox("Cumulative").active == [0]
    assert not text_contains("unavailable in elapsed time")
    assert any(p.name == "cdf" for p in plots())
    assert not any(p.title and p.title.text for p in plots())
    mode.value = "schematic"
    assert schematic.visible and not schematic.disabled
    assert not guides.visible
    assert checkbox("Cumulative").active == []
    assert not checkbox("Bar chart").disabled
    mode.value = "physical"
    assert checkbox("Bar chart").disabled and checkbox("Guides").visible
    assert schematic.visible and schematic.disabled
    schematic.active = []
    assert guides.active == [0]
    assert not text_contains("Physical mode shows boundary guides")
    checkbox("Realizations").active = [0]
    y_mode = next(w for w in doc.select({"type": Select}) if w.title == "Y coordinate")
    checkbox("Collision").active = [0]
    for value in ("strip", "beeswarm", "radial"):
        y_mode.value = value
        assert checkbox("Envelope").visible == (value == "beeswarm")
        checkbox("Envelope").active = [0]
        event_plot = next(p for p in plots() if p.name == "events")
        guide_hover = next(
            tool
            for tool in event_plot.tools
            if isinstance(tool, HoverTool) and tool.tooltips == [("Boundary", "@name")]
        )
        assert guide_hover.mode == "vline"
        assert guide_hover.renderers[0].data_source.data["name"] == [
            "Start",
            *[name + " end" for name in data.REGIONS],
        ]
        assert not any(
            label.text in guide_hover.renderers[0].data_source.data["name"]
            for label in event_plot.select({"type": Label})
        )
        assert event_plot.yaxis[0].visible == (value == "radial")
        assert event_plot.yaxis[0].axis_label == (
            "Radial distance (mm)" if value == "radial" else None
        )
        assert (
            next(p for p in plots() if p.xaxis[0].visible).xaxis[0].axis_label
            == "Axial distance (mm)"
        )
        markers = [r.glyph for r in event_plot.renderers if "id" in r.data_source.data]
        assert all(
            g.marker == ("circle" if value == "beeswarm" else "x") for g in markers
        )
        if value == "beeswarm":
            glyph = next(r for r in event_plot.renderers if "id" in r.data_source.data)
            event_plot.set_from_json("inner_width", 320)
            event_plot.set_from_json("inner_height", 240)
            assert event_plot.y_range.end - event_plot.y_range.start == 240
            assert glyph.glyph.size <= 10
            assert event_plot.y_range.reset_start == event_plot.y_range.start
            assert event_plot.y_range.reset_end == event_plot.y_range.end
        patches = [
            r for r in event_plot.renderers if r.glyph.__class__.__name__ == "Patch"
        ]
        assert bool(patches) == (value == "beeswarm")
        if value == "radial":
            source = next(
                r.data_source
                for r in event_plot.renderers
                if "id" in r.data_source.data
            )
            assert source.data["y"] == [5000] * len(source.data["id"])
            assert dict(
                zip(source.data["id"], source.data["axial_distance"], strict=True)
            ) == {
                1: 0,
                2: 2000,
                3: 5000,
                4: 1000,
            }
    bridge.text = "clear"
    assert text_contains("Click an event")
    assert len(loads) == 1
    assert doc.to_json() is not None


@pytest.mark.parametrize("empty", [False, True])
def test_empty_and_eventless_document(monkeypatch, empty):
    monkeypatch.setattr(
        session, "load_data", lambda *args: cohort([] if empty else [1])
    )
    doc = Document()
    build_document(None, doc, 1, 1)
    slider = next(iter(doc.select({"type": RangeSlider})))
    assert slider.disabled == empty
    assert any(
        ("No realizations" if empty else "incomplete or ambiguous histories") in d.text
        for d in doc.select({"type": Div})
    )
    assert doc.to_json() is not None


def test_unavailable_document(monkeypatch):
    def unavailable(*args):
        raise RuntimeError("Missing <events>")

    monkeypatch.setattr(session, "load_data", unavailable)
    doc = Document()
    build_document(None, doc, 1, 1)
    assert doc.roots[0].text == "Explorer data unavailable: Missing &lt;events&gt;"


@pytest.mark.parametrize(
    "mode,sizing",
    [
        ("schematic", "chambers"),
        ("schematic", "zones"),
        ("physical", None),
        ("time", None),
    ],
)
def test_cumulative_pathway_areas(monkeypatch, mode, sizing):
    frames = cohort(
        [1, 2, 3, 4],
        event(1, "fragmentation", 1, 1, 2, 7),
        event(2, "fragmentation", 2, 1, 2, 7),
        event(3, "escape", 3, 2, 6),
        event(4, "collision", 4, 1, 1),
    )
    monkeypatch.setattr(session, "load_data", lambda *args: frames)
    doc = Document()
    build_document(None, doc, 1, 1)
    controls = list(doc.select({"type": CheckboxGroup}))
    next(w for w in controls if w.labels == ["Cumulative"]).active = [0]
    next(
        w for w in doc.select({"type": Select}) if w.title == "X coordinate"
    ).value = mode
    if sizing:
        next(
            w for w in doc.select({"type": Select}) if w.title == "Sizing"
        ).value = sizing
    fates = next(
        g.child.children[0]
        for g in doc.select({"type": GroupBox})
        if g.name == "sidebar_pathways"
    )
    colors = plot._pathway_colors(fates.labels, "Parent → Parent")
    assert colors["Parent → Parent"] == plot.COLORS["escape"]
    assert plot.COLORS["fragmentation"] == Category10[10][3]
    for color in colors.values():
        assert f"background: {color}" in fates.stylesheets[0].css

    cumulative = next(p for p in doc.select({"type": figure}) if p.name == "cdf")
    assert not cumulative.legend
    assert len(cumulative.renderers) == 2
    fragmented, escaped = cumulative.renderers
    assert (
        fragmented.glyph.__class__.__name__
        == escaped.glyph.__class__.__name__
        == "VArea"
    )
    first, second = fragmented.data_source.data, escaped.data_source.data
    assert first["position"] == second["position"]
    mapped = event_layout.coordinate_regions(
        frames[3], mode, event_layout.schematic_widths(sizing) if sizing else None
    )
    endpoint = mapped["right"][-1]
    escape_position = 2 if mode == "time" else endpoint
    assert second["position"][3:5] == [escape_position] * 2
    event_plot = next(p for p in doc.select({"type": figure}) if p.name == "events")
    escape_source = next(
        r.data_source
        for r in event_plot.renderers
        if 3 in r.data_source.data.get("id", [])
    )
    assert escape_source.data["x"][escape_source.data["id"].index(3)] == (
        2
        if mode == "time"
        else 6000
        if mode == "physical"
        else sum(event_layout.escape_slot(mapped)) / 2
    )
    # Tied fragmentations form one vertical jump; unresolved histories stay out.
    assert first["fraction"] == [0, 0, 0.5, 0.5, 0.5, 0.5]
    assert second["fraction"] == [0, 0, 0, 0, 0.25, 0.25]
    assert first["bottom"] == [0] * 6
    assert second["bottom"] == first["top"]
    assert second["top"][-1] == 0.75
    assert fragmented.glyph.fill_color == colors["Parent → A + B"]
    assert escaped.glyph.fill_color == colors["Parent → Parent"]

    next(w for w in controls if w.labels == ["Realizations"]).active = []
    next(w for w in controls if w.labels == ["Facet pathways"]).active = [0]
    facets = [p for p in doc.select({"type": figure}) if p.name == "cdf"]
    assert len(facets) == 3
    for facet in facets:
        if "Incomplete" in facet.title.text:
            assert not facet.renderers
        else:
            assert len(facet.renderers) == 1
            area = facet.renderers[0]
            assert area.data_source.data["top"][-1] == 1
            assert not any(area.data_source.data["bottom"])
            name = facet.title.text.split(" (N=")[0]
            assert area.glyph.fill_color == colors[name]
            if name == "Parent → Parent":
                assert area.data_source.data["position"][1:3] == [escape_position] * 2

    next(w for w in controls if w.labels == ["Facet pathways"]).active = []
    fates.active = [fates.labels.index("Parent → Parent")]
    cumulative = next(p for p in doc.select({"type": figure}) if p.name == "cdf")
    assert cumulative.renderers[0].glyph.fill_color == colors["Parent → Parent"]
    assert cumulative.renderers[0].data_source.data["top"][-1] == 1
    fates.active = []
    assert not any(p.name == "cdf" for p in doc.select({"type": figure}))
    assert doc.to_json() is not None


def test_fragmentation_grouping_and_pathway_overrides(monkeypatch):
    members, events, _, regions = cohort(
        [1, 2, 3],
        event(1, "collision", 1, 0, 0),
        event(2, "fragmentation", 1, 1, 2, 7),
        event(3, "collision", 2, 0, 0),
        event(4, "fragmentation", 2, 1, 3, 8),
        event(5, "collision", 3, 0, 0),
        event(6, "escape", 3, 2, 5),
    )
    pathways = pl.DataFrame(
        {"pathway_id": [7, 8], "pathway": ["Parent → A + B", "Parent → C + D"]}
    )
    members = data.classify(
        members.select("id", "experiment_result_id"), events, pathways, "Parent"
    )
    monkeypatch.setattr(
        session, "load_data", lambda *args: (members, events, pathways, regions)
    )
    doc = Document()
    build_document(None, doc, 1, 1)

    def checkbox(label):
        return next(
            w for w in doc.select({"type": CheckboxGroup}) if w.labels == [label]
        )

    pathway_group = next(
        g for g in doc.select({"type": GroupBox}) if g.name == "sidebar_pathways"
    )
    fates = pathway_group.child.children[0]
    options = next(
        g for g in doc.select({"type": GroupBox}) if g.title == "Events"
    ).child
    grouping = checkbox("Group fragmentations")
    facets = checkbox("Facet pathways")
    colors = plot._pathway_colors(fates.labels, "Parent → Parent")

    def event_points():
        chart = next(p for p in doc.select({"type": figure}) if p.name == "events")
        return {
            event_id: color
            for r in chart.renderers
            if "id" in r.data_source.data
            for event_id, color in zip(
                r.data_source.data["id"], r.data_source.data["color"], strict=True
            )
        }

    def event_glyph():
        chart = next(p for p in doc.select({"type": figure}) if p.name == "events")
        return next(r.glyph for r in chart.renderers if "id" in r.data_source.data)

    assert grouping.active == [0] and not grouping.disabled
    assert [w.labels[0] for w in options.children] == [
        "Initial",
        "Collision",
        "Fragmentation",
        "Escape",
    ]
    assert event_points()[2] == event_points()[4] == Category10[10][3]
    assert event_points()[6] == plot.COLORS["escape"]
    checkbox("Cumulative").active = [0]
    cumulative = next(p for p in doc.select({"type": figure}) if p.name == "cdf")
    assert [r.glyph.fill_color for r in cumulative.renderers] == [
        colors[name] for name in fates.labels
    ]

    grouping.active = []
    assert [w.labels[0] for w in options.children] == [
        "Initial",
        "Collision",
        "Parent → A + B",
        "Parent → C + D",
        "Parent → Parent",
    ]
    for event_id, name in [
        (2, "Parent → A + B"),
        (4, "Parent → C + D"),
        (6, "Parent → Parent"),
    ]:
        assert event_points()[event_id] == colors[name]
        assert f"color: {colors[name]}" in checkbox(name).stylesheets[0].css
    checkbox("Parent → A + B").active = []
    assert 2 not in event_points() and 4 in event_points()
    cumulative = next(p for p in doc.select({"type": figure}) if p.name == "cdf")
    assert len(cumulative.renderers) == 3

    fates.active = [fates.labels.index("Parent → C + D")]
    assert checkbox("Parent → A + B").disabled and not checkbox("Parent → A + B").active
    assert checkbox("Parent → Parent").disabled and checkbox("Parent → Parent").active
    assert not checkbox("Parent → C + D").disabled and checkbox("Parent → C + D").active
    assert set(event_points()) == {3, 4}
    fates.active = list(range(len(fates.labels)))
    assert not checkbox("Parent → A + B").disabled
    assert not checkbox("Parent → A + B").active
    assert (
        checkbox("Parent → Parent").active and not checkbox("Parent → Parent").disabled
    )
    assert 2 not in event_points() and 6 in event_points()
    fates.active = [fates.labels.index("Parent → C + D")]
    facets.active = [0]
    assert grouping.active == [0] and grouping.disabled
    assert [w.labels[0] for w in options.children] == [
        "Initial",
        "Collision",
        "Fragmentation",
        "Escape",
    ]
    assert checkbox("Escape").disabled and checkbox("Escape").active
    assert checkbox("Fragmentation").active
    facets.active = []
    assert grouping.active == [0] and not grouping.disabled
    fates.active = [fates.labels.index("Parent → Parent")]
    assert checkbox("Fragmentation").disabled and checkbox("Fragmentation").active
    assert not checkbox("Escape").disabled and checkbox("Escape").active
    checkbox("Escape").active = [0]
    assert 6 in event_points()
    fates.active = [fates.labels.index("Parent → C + D")]
    assert checkbox("Escape").disabled and checkbox("Escape").active

    # Pathway colours also survive a marker-layout change and regrouping.
    fates.active = list(range(len(fates.labels)))
    checkbox("Fragmentation").active = [0]
    grouping.active = []
    mode = next(s for s in doc.select({"type": Select}) if s.title == "Y coordinate")
    mode.value = "beeswarm"
    assert event_glyph().marker == "circle"
    assert event_points()[2] == colors["Parent → A + B"]
    assert 'content: "●"' in checkbox("Parent → A + B").stylesheets[0].css
    fates.active = []
    assert all(w.disabled and w.active for w in options.children)
    assert doc.to_json() is not None


@pytest.mark.parametrize("sizing", ["chambers", "zones"])
def test_terminal_zone_controls_and_bar_only_slot(monkeypatch, sizing):
    frames = cohort(
        [1, 2],
        event(1, "init", 1, 0, 0),
        event(2, "escape", 1, 1, 5),
        event(3, "init", 2, 0, 0),
        event(4, "escape", 2, 1, 5),
    )
    monkeypatch.setattr(session, "load_data", lambda *args: frames)
    doc = Document()
    build_document(None, doc, 1, 1)
    controls = {
        w.labels[0]: w
        for w in doc.select({"type": CheckboxGroup})
        if len(w.labels) == 1
    }
    own = controls["Place initial/escape in their own zones"]
    spread = controls["Spread initial/escape X"]
    x_mode = next(w for w in doc.select({"type": Select}) if w.title == "X coordinate")
    y_mode = next(w for w in doc.select({"type": Select}) if w.title == "Y coordinate")
    x_mode.value = "schematic"
    next(w for w in doc.select({"type": Select}) if w.title == "Sizing").value = sizing
    mapped = event_layout.coordinate_regions(
        frames[3], "schematic", event_layout.schematic_widths(sizing)
    )

    def coordinates():
        chart = doc.select_one({"name": "events"})
        return {
            i: x
            for r in chart.renderers
            if "id" in r.data_source.data
            for i, x in zip(
                r.data_source.data["id"], r.data_source.data["x"], strict=True
            )
        }

    def labels():
        return {label.text for label in doc.select({"type": Label})}

    assert (
        own.active == spread.active == [0] and not own.disabled and not spread.disabled
    )
    assert coordinates()[1] != coordinates()[3]
    spread.active = []
    assert (
        coordinates()[1]
        == coordinates()[3]
        == sum(event_layout.initial_slot(mapped)) / 2
    )
    spread.active = [0]
    own.active = []
    assert spread.disabled and spread.active == [0]
    assert coordinates() == {1: 0, 2: mapped["right"][-1], 3: 0, 4: mapped["right"][-1]}
    assert "Initial" not in labels() and "Escaped" not in labels()
    assert doc.select_one({"name": "events"}).x_range.start == 0
    assert (
        doc.select_one({"name": "events"}).x_range.end
        < event_layout.escape_slot(mapped)[1]
    )
    controls["Bar chart"].active = [0]
    assert "Escaped" in labels() and "Initial" not in labels()
    assert (
        doc.select_one({"name": "events"}).x_range.end
        == event_layout.escape_slot(mapped)[1]
    )
    assert coordinates()[2] == mapped["right"][-1]
    bars = doc.select_one({"name": "bars"})
    assert bars.renderers[-1].glyph.left == event_layout.escape_slot(mapped)[0]
    assert bars.renderers[-1].glyph.top == 1
    controls["Bar chart"].active = []
    assert "Escaped" not in labels()
    own.active = [0]
    y_mode.value = "radial"
    assert spread.disabled and spread.active == [0]
    assert coordinates()[1] == sum(event_layout.initial_slot(mapped)) / 2
    y_mode.value = "strip"
    assert not spread.disabled and coordinates()[1] != coordinates()[3]
    for value in ("physical", "time"):
        x_mode.value = value
        assert own.disabled and spread.disabled
        assert own.active == spread.active == [0]
        assert coordinates()[1] == 0
        assert coordinates()[2] == (5000 if value == "physical" else 1)
    x_mode.value = "schematic"
    y_mode.value = "realization"
    assert not own.disabled and not spread.disabled
    assert coordinates()[1] != coordinates()[3]


@pytest.mark.parametrize("layout", ["realization", "strip", "beeswarm"])
def test_spread_plot_resize_selection_filtering_and_facets(monkeypatch, layout):
    frames = cohort(
        list(range(1, 14)),
        *[event(i * 3, "init", i, 0, 0) for i in range(1, 14)],
        *[event(i * 3 + 1, "collision", i, 1, 2) for i in range(1, 4)],
        *[event(i * 3 + 2, "escape", i, 2, 5) for i in range(1, 13)],
        event(41, "fragmentation", 13, 2, 2, 7),
    )
    monkeypatch.setattr(session, "load_data", lambda *args: frames)
    doc = Document()
    build_document(None, doc, 1, 1)
    controls = {
        w.labels[0]: w
        for w in doc.select({"type": CheckboxGroup})
        if len(w.labels) == 1
    }
    controls["Cumulative"].active = [0]
    y_mode = next(w for w in doc.select({"type": Select}) if w.title == "Y coordinate")
    y_mode.value = layout
    if layout == "beeswarm":
        controls["Envelope"].active = [0]

    def chart():
        return doc.select_one({"name": "events"})

    def points(p):
        return {
            i: (x, y)
            for r in p.renderers
            if "id" in r.data_source.data
            for i, x, y in zip(
                r.data_source.data["id"],
                r.data_source.data["x"],
                r.data_source.data["y"],
                strict=True,
            )
        }

    def check_highlights(p):
        visible = points(p)
        for renderer in p.renderers:
            values = renderer.data_source.data
            if (
                "position" in values
                and "id" not in values
                and renderer.glyph.__class__.__name__ != "Patch"
            ):
                expected = (
                    [visible[3]]
                    if getattr(renderer.glyph, "size", None) == 20
                    else [visible[i] for i in (3, 4, 5)]
                )
                assert (
                    list(zip(values["position"], values["plot_y"], strict=True))
                    == expected
                )
        if layout == "beeswarm":
            patch = next(
                r for r in p.renderers if r.glyph.__class__.__name__ == "Patch"
            )
            assert min(patch.data_source.data["position"]) <= min(
                x for x, _ in visible.values()
            )
            assert max(patch.data_source.data["position"]) >= max(
                x for x, _ in visible.values()
            )

    source = next(
        r.data_source
        for r in chart().renderers
        if 3 in r.data_source.data.get("id", [])
    )
    source.selected.indices = [source.data["id"].index(3)]
    p = chart()
    before = points(p)
    # Highlights are updated from the same displayed positions after each repack.
    for width, height in ((300, 240), (900, 600)):
        p.set_from_json("inner_width", width)
        p.set_from_json("inner_height", height)
        check_highlights(p)
        if layout != "beeswarm":
            assert all(
                r.glyph.size == 6 for r in p.renderers if "id" in r.data_source.data
            )
    assert points(p) != before
    p.x_range.start -= 0.3
    check_highlights(p)
    if layout == "realization":
        p.y_range.end += 100
        check_highlights(p)
    cumulative = doc.select_one({"name": "cdf"})
    escaped = next(
        r for r in cumulative.renderers if r.glyph.fill_color == plot.COLORS["escape"]
    )
    assert escaped.data_source.data["position"][-3:-1] == [9, 9]
    controls["Initial"].active = []
    assert 3 not in points(chart())
    controls["Initial"].active = [0]
    controls["Facet pathways"].active = [0]
    facets = list(doc.select({"name": "events"}))
    assert len(facets) == 2
    assert sum(len(points(p)) for p in facets) == frames[1].height
    for p in facets:
        p.set_from_json("inner_width", 320)
        assert points(p)


def test_initial_event_controls_and_schematic(monkeypatch):
    frames = cohort([1], event(1, "init", 1, 0, 0), event(2, "escape", 1, 1, 5))
    monkeypatch.setattr(session, "load_data", lambda *args: frames)
    doc = Document()
    build_document(None, doc, 1, 1)
    control = doc.select_one({"name": "event:init"})
    assert isinstance(control, CheckboxGroup)
    assert control.labels == ["Initial"] and not control.disabled
    coordinate = next(
        s for s in doc.select({"type": Select}) if s.title == "X coordinate"
    )
    sizing = next(w for w in doc.select({"type": Select}) if w.title == "Sizing")
    for preset in ("chambers", "zones"):
        coordinate.value = "schematic"
        sizing.value = preset
        chart = doc.select_one({"name": "events"})
        renderer = next(
            r for r in chart.renderers if 1 in r.data_source.data.get("id", [])
        )
        assert (
            chart.x_range.start
            < renderer.data_source.data["x"][renderer.data_source.data["id"].index(1)]
            < 0
        )
        mapped = event_layout.coordinate_regions(
            frames[3], "schematic", event_layout.schematic_widths(preset)
        )
        assert chart.xgrid[0].ticker.ticks == [mapped["left"][0], *mapped["right"]]
        assert any(
            label.text in ("Initial", "Init") for label in doc.select({"type": Label})
        )
    control.active = []
    chart = doc.select_one({"name": "events"})
    assert all(1 not in r.data_source.data.get("id", []) for r in chart.renderers)
    control.active = [0]
    assert doc.to_json() is not None


@pytest.mark.parametrize("layout", ["realization", "strip", "beeswarm", "radial"])
def test_inspection_and_frame_changes_preserve_plots(monkeypatch, layout):
    frames = cohort(
        [1, 2],
        event(1, "init", 1, 0, 0),
        event(2, "escape", 1, 1, 5),
        event(3, "init", 2, 0, 0),
        event(4, "escape", 2, 1, 5),
    )
    monkeypatch.setattr(session, "load_data", lambda *args: frames)
    doc = Document()
    build_document(None, doc, 1, 1)
    next(
        w for w in doc.select({"type": Select}) if w.title == "Y coordinate"
    ).value = layout
    p = doc.select_one({"name": "events"})
    p.set_from_json("inner_width", 900)
    p.set_from_json("inner_height", 380)
    p.x_range.start, p.x_range.end = 1, 8
    renderers = list(p.renderers)
    ranges = p.x_range, p.y_range
    source = next(
        r.data_source for r in p.renderers if 1 in r.data_source.data.get("id", [])
    )

    # Inspection must not re-enter the cohort/layout pipeline.
    monkeypatch.setattr(
        session, "layout_events", lambda *args: pytest.fail("Plot rebuilt")
    )
    monkeypatch.setattr(
        plot, "pack_beeswarm", lambda *args, **kwargs: pytest.fail("Selection repacked")
    )
    monkeypatch.setattr(
        plot, "spread_terminal_x", lambda *args: pytest.fail("Selection repacked")
    )
    source.selected.indices = [source.data["id"].index(1)]
    bridge = doc.roots[0].children[1]
    bridge.text = "2"
    clicked = next(r for r in p.renderers if getattr(r.glyph, "size", None) == 20)
    assert clicked.data_source.data["position"]
    bridge.text = "clear"
    assert not clicked.data_source.data["position"]
    source.selected.indices = [source.data["id"].index(1)]
    bridge.text = "2"
    assert clicked.data_source.data["position"]
    next(
        w
        for w in doc.select({"type": CheckboxGroup})
        if w.labels == ["Summarise non-axial motion"]
    ).active = []

    viewport = next(d for d in doc.select({"type": Div}) if d.text == "1440,900")
    viewport.text = "1280,901"
    viewport.text = "640,901"
    fullscreen = next(d for d in doc.select({"type": Div}) if d.text == "normal")
    fullscreen.text = "full"
    assert p.height > 440
    fullscreen.text = "normal"
    assert p.height == 440
    assert doc.select_one({"name": "events"}) is p
    assert p.renderers == renderers
    assert (p.x_range, p.y_range) == ranges
    assert (p.x_range.start, p.x_range.end) == (1, 8)


@pytest.mark.parametrize("layout", ["realization", "beeswarm"])
def test_live_packing_waits_for_dimensions_and_coalesces_updates(monkeypatch, layout):
    frames = cohort([1], event(1, "init", 1, 0, 0), event(2, "escape", 1, 1, 5))
    monkeypatch.setattr(session, "load_data", lambda *args: frames)
    doc = Document()
    # Exercise server scheduling without starting a network server.
    doc._session_context = lambda: object()
    packing_name = "pack_beeswarm" if layout == "beeswarm" else "spread_terminal_x"
    original = getattr(plot, packing_name)
    calls = []

    def pack(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(plot, packing_name, pack)
    build_document(None, doc, 1, 1)
    next(
        w for w in doc.select({"type": Select}) if w.title == "Y coordinate"
    ).value = layout
    p = doc.select_one({"name": "events"})
    assert not calls
    p.set_from_json("inner_width", 300)
    assert len(doc.session_callbacks) == 1
    doc.session_callbacks[0].callback()
    assert not calls  # Height is still unknown.
    p.set_from_json("inner_width", 900)
    p.set_from_json("inner_height", 380)
    p.x_range.start, p.x_range.end = 0, 8
    assert len(doc.session_callbacks) == 1
    doc.session_callbacks[0].callback()
    assert len(calls) == 1
    assert not doc.session_callbacks
    # Returning to the same dimensions before the next tick needs no work.
    p.x_range.start = 1
    p.x_range.start = 0
    doc.session_callbacks[0].callback()
    assert len(calls) == 1
    # A queued update for a plot removed by a control change must be discarded.
    p.set_from_json("inner_width", 800)
    next(
        w for w in doc.select({"type": Select}) if w.title == "X coordinate"
    ).value = "time"
    doc.session_callbacks[0].callback()
    assert len(calls) == 1


def test_beeswarm_reuses_preparation_until_cohort_or_coordinates_change(monkeypatch):
    frames = cohort(
        [1],
        event(1, "init", 1, 0, 0),
        event(2, "collision", 1, 1, 2),
        event(3, "escape", 1, 2, 5),
    )
    monkeypatch.setattr(session, "load_data", lambda *args: frames)
    preparations, packs = [], []
    prepare, pack = plot.prepare_beeswarm, plot.pack_beeswarm

    def preparing(*args):
        result = prepare(*args)
        preparations.append(result)
        return result

    def packing(*args, **kwargs):
        packs.append(kwargs["prepared"])
        return pack(*args, **kwargs)

    monkeypatch.setattr(plot, "prepare_beeswarm", preparing)
    monkeypatch.setattr(plot, "pack_beeswarm", packing)
    doc = Document()
    build_document(None, doc, 1, 1)
    next(
        w for w in doc.select({"type": Select}) if w.title == "Y coordinate"
    ).value = "beeswarm"
    chart = doc.select_one({"name": "events"})
    chart.set_from_json("inner_width", 900)
    chart.set_from_json("inner_height", 380)
    chart.x_range.start, chart.x_range.end = 1, 4
    chart.set_from_json("inner_width", 300)
    assert len(preparations) == 1
    assert len(packs) > 1
    assert all(item is preparations[0] for item in packs)

    next(
        w for w in doc.select({"type": CheckboxGroup}) if w.name == "event:collision"
    ).active = []
    assert len(preparations) == 2
    assert len(preparations[-1].positions) == 2
    assert packs[-1] is preparations[-1]
    next(
        w for w in doc.select({"type": Select}) if w.title == "X coordinate"
    ).value = "time"
    assert len(preparations) == 3
    assert preparations[-1].slots is None
    assert packs[-1] is preparations[-1]


@pytest.mark.parametrize("mode", ["bottom", "right"])
def test_details_layout_transitions(monkeypatch, mode):
    frames = cohort([1], event(1, "escape", 1, 1, 5))
    monkeypatch.setattr(session, "load_data", lambda *args: frames)
    doc = Document()
    build_document(None, doc, 1, 1)
    body = doc.roots[0].children[0]
    left, workspace = body.children
    center, right = workspace.children
    toggle = right.children[0].children[0]
    layouts = {
        m: doc.select_one({"name": f"details_layout_{m}"})
        for m in ("sidebar", "bottom", "right")
    }
    fullscreen = next(d for d in doc.select({"type": Div}) if d.text == "normal")
    chart = doc.select_one({"name": "events"})
    chart.x_range.start, chart.x_range.end = 1, 2

    toggle.active = True
    assert layouts["sidebar"].active
    layouts[mode].active = True
    assert not left.children[1].visible
    assert ("border-top" in right.styles) == (mode == "bottom")
    assert right.sizing_mode == "stretch_width" and right.width is None
    assert type(body.children[1]).__name__ == ("Column" if mode == "bottom" else "Row")
    if mode == "bottom":
        assert right.styles["max-height"] == "50%"
    else:
        assert right.styles["height"] == "100%"
    # Clicking the selected layout again keeps it selected.
    layouts[mode].active = False
    assert layouts[mode].active
    fullscreen.text = "full"
    assert not toggle.active and layouts["sidebar"].active
    toggle.active = True
    layouts["right" if mode == "bottom" else "bottom"].active = True
    fullscreen.text = "normal"
    assert toggle.active and layouts[mode].active
    assert doc.select_one({"name": "events"}) is chart
    assert (chart.x_range.start, chart.x_range.end) == (1, 2)

    toggle.active = False
    assert layouts["sidebar"].active
    assert "border-top" not in right.styles
    assert all(not button.visible for button in layouts.values())
    toggle.active = True
    assert right.width == 300 and layouts["sidebar"].active
    layouts[mode].active = True
    left.children[0].active = True
    assert left.children[1].visible and layouts["sidebar"].active
    assert right.width == 300
    assert body.children[1].children == [center, right]


def test_separate_rejected_collisions_and_visibility_transitions(monkeypatch):
    from .helpers import collision_cohort

    monkeypatch.setattr(session, "load_data", lambda *args: collision_cohort())
    doc = Document()
    build_document(None, doc, 1, 1)
    split = next(
        w
        for w in doc.select({"type": CheckboxGroup})
        if w.labels == ["Separate rejected collisions"]
    )
    accepted = doc.select_one({"name": "event:collision"})

    def points():
        return {
            id: color
            for renderer in doc.select_one({"name": "events"}).renderers
            if "id" in renderer.data_source.data
            for id, color in zip(
                renderer.data_source.data["id"], renderer.data_source.data["color"]
            )
        }

    assert not split.active
    assert points() == {1: "#1f77b4", 3: "#1f77b4", 4: "#1f77b4", 6: "#2ca02c"}
    split.active = [0]
    rejected = doc.select_one({"name": "event:rejected_collision"})
    assert accepted.labels == ["Accepted collisions"]
    assert rejected.labels == ["Rejected collisions"]
    assert points() == {1: "#1f77b4", 3: "#aec7e8", 4: "#1f77b4", 6: "#2ca02c"}
    rejected.active = []
    assert set(points()) == {1, 4, 6}
    split.active = []
    assert accepted.labels == ["Collision"]
    assert set(points()) == {1, 3, 4, 6}
    split.active = [0]
    assert rejected.active == [0]
    accepted.active = []
    assert set(points()) == {3, 6}
    split.active = []
    assert set(points()) == {6}  # Merging follows accepted visibility.
    split.active = [0]
    assert not rejected.active
    rejected.active = [0]
    # Detail history still includes hidden collisions and both postcollision records.
    source = next(
        r.data_source
        for r in doc.select_one({"name": "events"}).renderers
        if 3 in r.data_source.data.get("id", [])
    )
    source.selected.indices = [source.data["id"].index(3)]
    details = next(
        d for d in doc.select({"type": Div}) if "Realization #1</h3>" in d.text
    )
    assert all(f'id="event-{id}"' in details.text for id in range(1, 7))
    doc.roots[0].children[1].text = "2"
    assert any("Selected hidden event #1" in d.text for d in doc.select({"type": Div}))


@pytest.mark.parametrize("layout", ["realization", "radial", "strip", "beeswarm"])
def test_postcollision_never_enters_explorer_sources(monkeypatch, layout):
    from .helpers import collision_cohort

    monkeypatch.setattr(session, "load_data", lambda *args: collision_cohort())
    doc = Document()
    build_document(None, doc, 1, 1)
    next(
        w for w in doc.select({"type": Select}) if w.title == "Y coordinate"
    ).value = layout
    for source in doc.select({"type": ColumnDataSource}):
        if "id" in source.data:
            assert set(source.data["id"]).isdisjoint({2, 5})
