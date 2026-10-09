"""Numerical, storage and interaction behavior of the realization Explorer."""

from math import sin
from types import SimpleNamespace

import duckdb
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from apitofresview.plotting.explorer import build_document
from apitofresview.plotting.explorer import data, plot


EVENT_SCHEMA = {
    "id": pl.Int64,
    "type": pl.String,
    "realization_id": pl.Int64,
    "t": pl.Float64,
    "x": pl.Float64,
    "y": pl.Float64,
    "z": pl.Float64,
    "pathway_id": pl.Int64,
    "z_clamped": pl.Boolean,
}
BOUNDARIES = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)


def event(id, kind, rid, t, z, pathway=None, clamped=False):
    return (id, kind, rid, t, 3.0, 4.0, z, pathway, clamped)


def cohort(ids, *events, boundaries=BOUNDARIES):
    regions = data.make_regions(boundaries)
    events = data.preprocess_events(
        pl.DataFrame(events, schema=EVENT_SCHEMA, orient="row"), regions
    )
    realizations = pl.DataFrame(
        {"id": ids, "experiment_result_id": [10] * len(ids)},
        schema={"id": pl.Int64, "experiment_result_id": pl.Int64},
    )
    pathways = pl.DataFrame({"pathway_id": [7], "pathway": ["Parent → A + B"]})
    return (
        data.classify(realizations, events, pathways, "Parent"),
        events,
        pathways,
        regions,
    )


def test_fates_coordinates_and_aggregate():
    realizations, events, _, regions = cohort(
        [1, 2, 3, 4, 5],
        event(1, "collision", 1, 0.1, 0),
        event(2, "fragmentation", 1, 2, 2, 7),
        event(3, "fragmentation", 2, 3, 2, 7),
        event(4, "escape", 3, 4, 5),
        event(5, "collision", 4, 1, 1),
        event(6, "fragmentation", 5, 2, 2, 7),
        event(7, "escape", 5, 3, 5),
    )
    assert realizations["fate"].to_list() == [
        "Parent → A + B",
        "Parent → A + B",
        "Parent → Parent",
        "Incomplete",
        "Ambiguous",
    ]
    assert realizations["terminal_event_id"].to_list() == [2, 3, 4, None, None]
    mapping = data.coordinate_regions(regions, "schematic")
    assert mapping["left"].to_list() == [0, 3, 6, 7, 8]
    assert mapping["right"].to_list() == [3, 6, 7, 8, 9]
    positioned = data.position_events(events, mapping, "schematic")
    assert positioned.filter(pl.col("id") == 2)["position"][0] == 6
    assert (
        positioned.filter(pl.col("id") == 4)["position"][0]
        == sum(data.escape_slot(mapping)) / 2
    )
    cdf, bars, escaped, unresolved = data.aggregate(realizations, positioned, mapping)
    assert cdf.to_dict(as_series=False) == {"position": [6, 6], "fraction": [0, 0.4]}
    assert bars["fraction"].to_list() == [0, 0, 0.4, 0, 0, 0.2]
    assert (escaped, unresolved) == (1, 2)
    cdf, bars, escaped, unresolved = data.aggregate(
        realizations.head(0), positioned.head(0), mapping
    )
    assert cdf.is_empty() and bars["fraction"].sum() == 0
    assert (escaped, unresolved) == (0, 0)
    for rid, column in [(1, "cdf"), (3, "bars")]:
        members = realizations.filter(pl.col("id") == rid)
        cdf, bars, _, _ = data.aggregate(
            members, data.events_for(positioned, members), mapping
        )
        assert (cdf if column == "cdf" else bars)["fraction"][-1] == 1
    assert (
        data.position_events(events, data.coordinate_regions(regions, "time"), "time")[
            "position"
        ].to_list()
        == events["t"].to_list()
    )
    assert events["radial"].to_list() == [5] * events.height


@pytest.mark.parametrize(
    "history,fate,terminal",
    [
        ([], "Incomplete", None),
        ([event(1, "collision", 1, 1, 1)], "Incomplete", None),
        ([event(1, "fragmentation", 1, 1, 1)], "Incomplete", None),
        ([event(1, "fragmentation", 1, 1, 6, 7)], "Incomplete", None),
        ([event(1, "escape", 1, 1, 6)], "Parent → Parent", 1),
        (
            [event(1, "escape", 1, 1, 5), event(2, "collision", 1, 2, 4)],
            "Incomplete",
            None,
        ),
        (
            [
                event(1, "fragmentation", 1, 1, 2, 7),
                event(2, "fragmentation", 1, 1, 2, 7),
            ],
            "Ambiguous",
            None,
        ),
        (
            [event(1, "fragmentation", 1, 1, 2, 7), event(2, "escape", 1, 2, 5)],
            "Ambiguous",
            None,
        ),
        (
            [event(2, "fragmentation", 1, 1, 5, 7), event(1, "collision", 1, 1, 5)],
            "Parent → A + B",
            2,
        ),
        (
            [event(1, "fragmentation", 1, 1, 2, 7), event(2, "collision", 1, 1, 2)],
            "Incomplete",
            None,
        ),
    ],
)
def test_terminal_histories(history, fate, terminal):
    realizations, events, _, _ = cohort([1], *history)
    assert realizations["fate"][0] == fate
    assert realizations["terminal_event_id"][0] == terminal
    assert events["id"].to_list() == sorted(e[0] for e in history)


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("physical", [0, 1000, 2000, 5000, 6000, 5000]),
        ("equal", [0, 1, 2, 5, 6, 5.325]),
        ("schematic", [0, 3, 6, 9, 10, 9.325]),
        ("time", [0, 1, 2, 3, 4, 5]),
    ],
)
def test_coordinate_modes_and_boundaries(mode, expected):
    _, events, _, regions = cohort(
        [1],
        *[
            event(i + 1, "escape" if i == 5 else "collision", 1, i, z)
            for i, z in enumerate([0, 1, 2, 5, 6, 5])
        ],
    )
    assert events["region"].to_list() == [0, 1, 2, 4, None, 4]
    positioned = data.position_events(
        events, data.coordinate_regions(regions, mode), mode
    )
    assert positioned["position"].to_list() == pytest.approx(expected)


def test_zero_width_regions():
    _, events, _, regions = cohort(
        [1],
        event(1, "fragmentation", 1, 1, 0, 7),
        boundaries=(0.0, 0.0, 1.0, 2.0, 3.0, 4.0),
    )
    assert events["region"][0] == 1
    positioned = data.position_events(
        events, data.coordinate_regions(regions, "schematic"), "schematic"
    )
    assert positioned["position"][0] == 0


def test_realization_coordinate(monkeypatch):
    from bokeh.document import Document
    from bokeh.models import RangeSlider, Select

    frames = cohort(
        [100, 102, 105, 110],
        event(1, "collision", 100, 0, 0),
        event(2, "collision", 102, 0, 0),
        event(3, "collision", 102, 1, 1),
        event(4, "escape", 105, 1, 5),
    )
    monkeypatch.setattr(plot, "load_data", lambda *args: frames)
    doc = Document()
    build_document(None, doc, 1, 1)
    mode = next(s for s in doc.select({"type": Select}) if s.title == "Y coordinate")
    assert mode.value == "realization"
    assert mode.options[0] == ("realization", "Realization #")

    def check(expected, endpoints):
        events = next(
            p for p in doc.select({"type": plot.figure}) if p.name == "events"
        )
        points = {
            id: y
            for renderer in events.renderers
            if hasattr(renderer, "data_source") and "id" in renderer.data_source.data
            for id, y in zip(
                renderer.data_source.data["id"], renderer.data_source.data["y"]
            )
        }
        assert points == expected
        axis = events.yaxis[0]
        assert axis.axis_label == "Realization #"
        assert axis.ticker.ticks == endpoints
        assert axis.major_tick_line_color is None
        assert axis.minor_tick_line_color is None
        assert not events.ygrid[0].visible

    check({1: 1, 2: 2, 3: 2, 4: 3}, [1, 4])
    pager = next(
        w
        for w in doc.select({"type": plot.CheckboxGroup})
        if w.labels == ["Realizations pager"]
    )
    pager.active = [0]
    next(iter(doc.select({"type": RangeSlider}))).value = (2, 4)
    check({2: 1, 3: 1, 4: 2}, [1, 3])
    pager.active = []
    check({1: 1, 2: 2, 3: 2, 4: 3}, [1, 4])
    fate = next(
        w for w in doc.select({"type": plot.CheckboxGroup}) if "Incomplete" in w.labels
    )
    fate.active = [fate.labels.index("Parent → Parent")]
    check({4: 1}, [1])


def test_layouts_and_envelope():
    _, events, _, regions = cohort(
        [1],
        event(3, "collision", 1, 0, 0),
        event(1, "collision", 1, 1, 0),
        event(2, "collision", 1, 2, 0),
        event(4, "collision", 1, 3, 2),
        event(5, "escape", 1, 4, 5),
    )
    events = data.position_events(
        events, data.coordinate_regions(regions, "physical"), "physical"
    )
    assert data.layout_events(events, "radial")["plot_y"].to_list() == [5000] * 5
    assert data.layout_events(events, "strip")["plot_y"].to_list() == pytest.approx(
        [0.25 * sin(i * 12.9898) for i in events["id"]]
    )

    def pack(frame):
        return data.pack_beeswarm(frame, 0, 5000, 600, 380)[0]

    packed = pack(events)
    assert packed["plot_y"].to_list() == pytest.approx([-10, 0, 10, 0, 0])
    assert_frame_equal(
        packed.select("id", "plot_y").sort("id"),
        pack(events.reverse()).select("id", "plot_y").sort("id"),
    )
    envelope = data.beeswarm_envelope(packed, 5 * 5000 / 600, 5)
    assert envelope["plot_y"].max() == 15
    assert envelope["plot_y"].min() == -15
    assert envelope["position"].min() < 0
    assert envelope["position"].max() > 5000
    assert_frame_equal(
        envelope, data.beeswarm_envelope(packed.reverse(), 5 * 5000 / 600, 5)
    )
    for subset in (events.head(0), events.head(1), events.head(3)):
        packed = pack(subset)
        outline = data.beeswarm_envelope(packed, 1, 5)
        assert outline.is_empty() == subset.is_empty()
        assert packed.height == subset.height
        if not subset.is_empty():
            assert outline["plot_y"].max() > packed["plot_y"].max()
            assert outline["plot_y"].min() < packed["plot_y"].min()


@pytest.mark.parametrize("width,height", [(600, 380), (300, 200), (900, 600)])
def test_beeswarm_circles_touch_without_overlaps(width, height):
    from math import hypot

    # A dense stack exercises more than the old seven levels; nearby columns
    # exercise diagonal tangencies and resizing.
    points = pl.DataFrame(
        {"id": range(80), "position": [i // 20 * 0.005 for i in range(80)]}
    )
    packed, diameter = data.pack_beeswarm(points, 0, 1, width, height)
    centers = list(zip(packed["position"] * width, packed["plot_y"], strict=True))
    for index, (x, y) in enumerate(centers):
        distances = [hypot(x - px, y - py) for px, py in centers[:index]]
        if distances:
            assert min(distances) >= diameter - 1e-7
            if abs(y) > 1e-7:
                assert min(distances) == pytest.approx(diameter)
    assert max(abs(y) for _, y in centers) + diameter / 2 <= height / 2
    identical = points.with_columns(pl.lit(0.0).alias("position"))
    stacked, size = data.pack_beeswarm(identical, 0, 1, width, height)
    assert [
        b - a for a, b in zip(sorted(stacked["plot_y"]), sorted(stacked["plot_y"])[1:])
    ] == pytest.approx([size] * 79)


@pytest.mark.parametrize("negative_z", [False, True])
def test_adapter_resolves_results_and_preserves_ids(monkeypatch, negative_z, tmp_path):
    database_path = str(tmp_path / "explorer.duckdb")
    conn = duckdb.connect(database_path)
    conn.execute(
        "create table multi_pathway_experiment_result(id int, experiment_run_id int, cluster_id int)"
    )
    conn.execute(
        "create table single_pathway_experiment_result(id int, experiment_run_id int, pathway_id int)"
    )
    conn.execute(
        "create table pathway(id int, cluster_id int, product1_id int, product2_id int)"
    )
    conn.execute("create table cluster(id int, common_name varchar)")
    conn.execute("create table realization(id int, experiment_result_id int)")
    conn.execute("""
        create table event_info(
            id int, realization_id int, event_type varchar,
            postime struct(x double,y double,z double,t double),
            velocity struct(x double,y double,z double) default {'x':1,'y':2,'z':3},
            omega struct(x double,y double,z double) default {'x':4,'y':5,'z':6},
            rot_energy double default 1e-21,
            vibrational_energy double default 2e-20,
            particle_index int default 0
        )
    """)
    conn.execute(
        "create table collision_event(id int, theta double, u_norm double, accepted bool)"
    )
    conn.execute("create table fragmentation_event(id int, pathway_id int)")
    conn.execute(
        "insert into cluster values (1,'5A_5SA_negative'),(2,'4A_5SA_negative'),(3,'1A_neutral')"
    )
    conn.execute("insert into pathway values (7,1,2,3)")
    conn.execute("insert into multi_pathway_experiment_result values (10,1,1),(11,2,1)")
    conn.execute("insert into single_pathway_experiment_result values (12,1,7)")
    conn.execute("insert into realization values (100,10),(101,11),(102,12)")
    conn.execute(
        "insert into event_info (id,realization_id,event_type,postime) values (501,100,'collision', {'x':1,'y':2,'z':0,'t':0.5})"
    )
    conn.execute(
        "insert into event_info (id,realization_id,event_type,postime) values (502,100,'fragmentation', {'x':1,'y':2,'z':2,'t':1.5})"
    )
    conn.execute(
        "insert into event_info (id,realization_id,event_type,postime) values (503,102,'escape', {'x':1,'y':2,'z':5,'t':2})"
    )
    conn.execute("insert into collision_event values (501,0.75,-114.5,false)")
    conn.execute("insert into fragmentation_event values (502,7)")
    conn.execute(
        "insert into event_info (id,realization_id,event_type,postime) values (500,100,'init', {'x':0,'y':0,'z':0,'t':0})"
    )
    if negative_z:
        conn.execute(
            "update event_info set postime = struct_update(postime, z := -0.000001) where id = 501"
        )
        conn.execute(
            "update event_info set postime = struct_update(postime, z := -0.000002) where id = 502"
        )
    geometry = data

    monkeypatch.setattr(geometry, "get_geometry", lambda *args: None)
    monkeypatch.setattr(
        geometry, "lengths_to_cumulative_lengths", lambda *args: (0, 1, 2, 3, 4, 5)
    )
    snapshots = {
        table: conn.execute(f"select * from {table}").fetchall()
        for table in (
            "realization",
            "collision_event",
            "fragmentation_event",
            "event_info",
        )
    }
    conn.close()
    conn = duckdb.connect(database_path, read_only=True)
    db = SimpleNamespace(db=conn)
    realizations, events, pathways, regions = data.load_data(db, 1, 1)
    assert realizations["id"].to_list() == [100, 102]
    assert realizations["fate"].to_list() == [
        "5A_5SA_negative → 4A_5SA_negative + 1A_neutral",
        "5A_5SA_negative → 5A_5SA_negative",
    ]
    assert events["id"].to_list() == [500, 501, 502, 503]
    collision = events.filter(pl.col("id") == 501).row(0, named=True)
    assert collision["theta"] == 0.75 and collision["u_norm"] == -114.5
    assert collision["accepted"] is False
    assert collision["velocity"] == {"x": 1, "y": 2, "z": 3}
    assert collision["rot_energy"] == 1e-21
    assert collision["vibrational_energy"] == 2e-20
    assert collision["particle_index"] == 0
    details = plot._details(
        realizations.head(1).with_columns(pl.lit(1).alias("realization_number")),
        events,
        500,
    )
    assert "Angular speed (rad/s)" in details
    assert "<td>No</td>" in details and "<td>—</td>" in details
    assert events.filter(pl.col("id") == 502)["pathway_id"][0] == 7
    assert pathways.to_dict(as_series=False) == {
        "pathway_id": [7],
        "pathway": ["5A_5SA_negative → 4A_5SA_negative + 1A_neutral"],
    }
    assert events["z_clamped"].sum() == (2 if negative_z else 0)
    if negative_z:
        assert events.filter(pl.col("z_clamped"))["z"].to_list() == [0, 0]
        regions = data.coordinate_regions(regions, "schematic")
        events = data.position_events(events, regions, "schematic")
        assert data.aggregate(realizations, events, regions)[1]["fraction"][0] == 0.5
    for table, snapshot in snapshots.items():
        assert conn.execute(f"select * from {table}").fetchall() == snapshot
    conn.close()
    conn = duckdb.connect(database_path)
    # UNION of result IDs prevents duplicate histories across result tables.
    conn.execute("insert into single_pathway_experiment_result values (10,1,7)")
    conn.execute("insert into multi_pathway_experiment_result values (13,1,9)")
    conn.execute("insert into realization values (103,13)")
    conn.execute(
        "insert into event_info (id,realization_id,event_type,postime) values (504,101,'collision', {'x':1,'y':2,'z':0,'t':1}), (505,103,'collision', {'x':1,'y':2,'z':0,'t':1}), (506,100,'collision', {'x':1,'y':2,'z':0,'t':0.5})"
    )
    conn.close()
    conn = duckdb.connect(database_path, read_only=True)
    db = SimpleNamespace(db=conn)
    realizations, events, _, _ = data.load_data(db, 1, 1)
    assert realizations["id"].to_list() == [100, 102]
    assert events["id"].to_list() == [500, 501, 506, 502, 503]
    empty = data.load_data(db, 99, 1)
    assert empty[0].is_empty() and empty[1].is_empty()
    conn.close()


def test_bokeh_document_modes_and_views(monkeypatch):
    from bokeh.document import Document
    from bokeh.models import Select, CheckboxGroup, GroupBox, CustomAction

    frames = cohort(
        [1], event(1, "collision", 1, 0, 0, clamped=True), event(2, "escape", 1, 1, 5)
    )
    monkeypatch.setattr(plot, "load_data", lambda *args: frames)
    doc = Document()
    build_document(None, doc, 1, 1)
    assert any(
        d.text == "1 events with negative z position have been clamped to 0"
        for d in doc.select({"type": plot.Div})
    )
    x_mode = next(s for s in doc.select({"type": Select}) if s.title == "X coordinate")
    assert x_mode.options[:2] == [
        ("schematic", "Equal chambers"),
        ("equal", "Equal zones"),
    ]
    from bokeh.plotting import figure

    for value in ("equal", "physical", "time", "schematic"):
        x_mode.value = value
        if value in ("equal", "schematic"):
            regions = data.coordinate_regions(frames[3], value)
            widths = regions["right"] - regions["left"]
            if value == "schematic":
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
    sidebars = frame.children[0].children
    fullscreen = next(d for d in doc.select({"type": plot.Div}) if d.text == "normal")
    fullscreen.text = "full"
    assert not sidebars[0].children[1].visible and not sidebars[2].children[1].visible
    assert "46px" in sidebars[0].children[1].styles["max-height"]
    fullscreen.text = "normal"
    assert sidebars[0].children[1].visible and not sidebars[2].children[1].visible
    viewport = next(d for d in doc.select({"type": plot.Div}) if d.text == "1440,900")
    viewport.text = "533,900"
    sidebars[2].children[0].active = True
    assert not sidebars[0].children[1].visible
    assert sidebars[2].width == 250
    from bokeh.plotting import figure

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
    assert any(a.icon == "fullscreen" for a in doc.select({"type": CustomAction}))
    assert len(doc.roots) == 1
    controls = sidebars[0].children[1].children
    assert [g.title for g in controls] == [
        "Pathways",
        "X-axis",
        "Elements",
        "Realizations",
    ]
    assert controls[1].child.children[1].labels == ["Guides"]
    assert controls[2].child.children[0].labels == ["Schematic"]
    assert controls[2].child.children[1].title == "Views"
    assert controls[3].child.children[0].labels == ["Use x markers"]
    assert controls[3].child.children[1].labels == ["Group fragmentations"]
    assert [g.title for g in controls[3].child.children[2:]] == ["Events", "Y-axis"]
    assert not event_plots[0].legend
    assert doc.to_json() is not None


def test_bokeh_filtering_selection_facets_and_restrictions(monkeypatch):
    from bokeh.document import Document
    from bokeh.models import ColumnDataSource, Div, GroupBox, RangeSlider, Select

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

    monkeypatch.setattr(plot, "load_data", load)
    doc = Document()
    build_document(None, doc, 1, 1)

    def checkbox(label):
        return next(
            w
            for w in doc.select({"type": plot.CheckboxGroup})
            if w.labels and w.labels[0] == label
        )

    def text_contains(text):
        return any(text in d.text for d in doc.select({"type": Div}))

    def plots():
        return list(doc.select({"type": plot.figure}))

    fates = next(
        g.child.children[0]
        for g in doc.select({"type": GroupBox})
        if g.title == "Pathways"
    )
    assert set(fates.labels) == {"Parent → A + B", "Parent → Parent", "Incomplete"}
    source = next(
        s for s in doc.select({"type": ColumnDataSource}) if s.data.get("id") == [2]
    )
    source.selected.indices = [0]
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
    layout = next(w for w in doc.select({"type": Select}) if w.title == "Y coordinate")
    checkbox("Collision").active = [0]
    for value in ("strip", "beeswarm", "radial"):
        layout.value = value
        assert checkbox("Envelope").visible == (value == "beeswarm")
        checkbox("Envelope").active = [0]
        event_plot = next(p for p in plots() if p.name == "events")
        guide_hover = next(
            tool
            for tool in event_plot.tools
            if isinstance(tool, plot.HoverTool)
            and tool.tooltips == [("Boundary", "@name")]
        )
        assert guide_hover.mode == "vline"
        assert guide_hover.renderers[0].data_source.data["name"] == [
            "Start",
            *[name + " end" for name in data.REGIONS],
        ]
        assert not any(
            label.text in guide_hover.renderers[0].data_source.data["name"]
            for label in event_plot.select({"type": plot.Label})
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
            assert source.data["y"] == [5000, 5000]
            assert source.data["axial_distance"] == [0, 1000]
    bridge.text = "clear"
    assert text_contains("Click an event")
    assert len(loads) == 1
    assert doc.to_json() is not None


@pytest.mark.parametrize("empty", [False, True])
def test_empty_and_eventless_document(monkeypatch, empty):
    from bokeh.document import Document
    from bokeh.models import RangeSlider

    monkeypatch.setattr(plot, "load_data", lambda *args: cohort([] if empty else [1]))
    doc = Document()
    build_document(None, doc, 1, 1)
    slider = next(iter(doc.select({"type": RangeSlider})))
    assert slider.disabled == empty
    assert any(
        ("No realizations" if empty else "incomplete or ambiguous histories") in d.text
        for d in doc.select({"type": plot.Div})
    )
    assert doc.to_json() is not None


def test_unavailable_document(monkeypatch):
    from bokeh.document import Document

    def unavailable(*args):
        raise RuntimeError("Missing <events>")

    monkeypatch.setattr(plot, "load_data", unavailable)
    doc = Document()
    build_document(None, doc, 1, 1)
    assert doc.roots[0].text == "Explorer data unavailable: Missing &lt;events&gt;"


@pytest.mark.parametrize("width,height", [(920, 995), (640, 600)])
def test_control_groups_do_not_overlap(monkeypatch, page, tmp_path, width, height):
    from bokeh.document import Document
    from bokeh.embed import file_html
    from bokeh.resources import INLINE
    from playwright.sync_api import expect

    frames = cohort([1], event(1, "escape", 1, 1, 5))
    monkeypatch.setattr(plot, "load_data", lambda *args: frames)
    doc = Document()
    build_document(None, doc, 1, 1)
    html = tmp_path / "explorer.html"
    html.write_text(file_html(doc, INLINE, "Explorer"))
    page.set_viewport_size({"width": width, "height": height})
    page.goto(html.as_uri())
    groups = page.locator(".bk-GroupBox")
    expect(groups).to_have_count(7)
    # A shrinking flex item can be shorter than its fieldset, which then
    # paints over the next group even though all controls remain in the DOM.
    page.wait_for_function("""() => {
        function check(root) {
            for (const element of root.querySelectorAll('*')) {
                if (element.matches('.bk-GroupBox')) {
                    const fieldset = element.shadowRoot.querySelector('fieldset');
                    const outer = element.getBoundingClientRect();
                    const inner = fieldset.getBoundingClientRect();
                    if (!outer.height || inner.bottom > outer.bottom + 1)
                        return false;
                }
                if (element.shadowRoot && !check(element.shadowRoot)) return false;
            }
            return true;
        }
        return check(document);
    }""")


@pytest.mark.parametrize("mode", ["schematic", "equal", "physical", "time"])
def test_cumulative_pathway_areas(monkeypatch, mode):
    from bokeh.document import Document
    from bokeh.models import CheckboxGroup, GroupBox, Select

    frames = cohort(
        [1, 2, 3, 4],
        event(1, "fragmentation", 1, 1, 2, 7),
        event(2, "fragmentation", 2, 1, 2, 7),
        event(3, "escape", 3, 2, 6),
        event(4, "collision", 4, 1, 1),
    )
    monkeypatch.setattr(plot, "load_data", lambda *args: frames)
    doc = Document()
    build_document(None, doc, 1, 1)
    controls = list(doc.select({"type": CheckboxGroup}))
    next(w for w in controls if w.labels == ["Cumulative"]).active = [0]
    next(
        w for w in doc.select({"type": Select}) if w.title == "X coordinate"
    ).value = mode
    fates = next(
        g.child.children[0]
        for g in doc.select({"type": GroupBox})
        if g.title == "Pathways"
    )
    colors = plot._pathway_colors(fates.labels, "Parent → Parent")
    assert colors["Parent → Parent"] == plot.COLORS["escape"]
    assert plot.COLORS["fragmentation"] == plot.Category10[10][3]
    for color in colors.values():
        assert f"background: {color}" in fates.stylesheets[0].css

    cumulative = next(p for p in doc.select({"type": plot.figure}) if p.name == "cdf")
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
    endpoint = data.coordinate_regions(frames[3], mode)["right"][-1]
    escape_position = 2 if mode == "time" else endpoint
    assert second["position"][3:5] == [escape_position] * 2
    event_plot = next(
        p for p in doc.select({"type": plot.figure}) if p.name == "events"
    )
    escape_source = next(
        r.data_source
        for r in event_plot.renderers
        if r.data_source.data.get("id") == [3]
    )
    assert escape_source.data["x"] == [
        2
        if mode == "time"
        else 6000
        if mode == "physical"
        else sum(data.escape_slot(data.coordinate_regions(frames[3], mode))) / 2
    ]
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
    facets = [p for p in doc.select({"type": plot.figure}) if p.name == "cdf"]
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
    cumulative = next(p for p in doc.select({"type": plot.figure}) if p.name == "cdf")
    assert cumulative.renderers[0].glyph.fill_color == colors["Parent → Parent"]
    assert cumulative.renderers[0].data_source.data["top"][-1] == 1
    fates.active = []
    assert not any(p.name == "cdf" for p in doc.select({"type": plot.figure}))
    assert doc.to_json() is not None


def test_fragmentation_grouping_and_pathway_overrides(monkeypatch):
    from bokeh.document import Document
    from bokeh.models import CheckboxGroup, GroupBox, Select

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
        plot, "load_data", lambda *args: (members, events, pathways, regions)
    )
    doc = Document()
    build_document(None, doc, 1, 1)

    def checkbox(label):
        return next(
            w for w in doc.select({"type": CheckboxGroup}) if w.labels == [label]
        )

    pathway_group = next(
        g for g in doc.select({"type": GroupBox}) if g.title == "Pathways"
    )
    fates = pathway_group.child.children[0]
    options = next(
        g for g in doc.select({"type": GroupBox}) if g.title == "Events"
    ).child
    grouping = checkbox("Group fragmentations")
    facets = checkbox("Facet pathways")
    colors = plot._pathway_colors(fates.labels, "Parent → Parent")

    def event_renderers():
        chart = next(p for p in doc.select({"type": plot.figure}) if p.name == "events")
        return {
            tuple(r.data_source.data["id"]): r
            for r in chart.renderers
            if "id" in r.data_source.data
        }

    assert grouping.active == [0] and not grouping.disabled
    assert [w.labels[0] for w in options.children] == [
        "Initial",
        "Collision",
        "Fragmentation",
        "Escape",
    ]
    assert event_renderers()[(2, 4)].glyph.fill_color == plot.Category10[10][3]
    assert event_renderers()[(6,)].glyph.fill_color == plot.COLORS["escape"]
    checkbox("Cumulative").active = [0]
    cumulative = next(p for p in doc.select({"type": plot.figure}) if p.name == "cdf")
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
    for ids, name in [
        ((2,), "Parent → A + B"),
        ((4,), "Parent → C + D"),
        ((6,), "Parent → Parent"),
    ]:
        assert event_renderers()[ids].glyph.fill_color == colors[name]
        assert f"color: {colors[name]}" in checkbox(name).stylesheets[0].css
    checkbox("Parent → A + B").active = []
    assert (2,) not in event_renderers() and (4,) in event_renderers()
    cumulative = next(p for p in doc.select({"type": plot.figure}) if p.name == "cdf")
    assert len(cumulative.renderers) == 3

    fates.active = [fates.labels.index("Parent → C + D")]
    assert checkbox("Parent → A + B").disabled and not checkbox("Parent → A + B").active
    assert checkbox("Parent → Parent").disabled and checkbox("Parent → Parent").active
    assert not checkbox("Parent → C + D").disabled and checkbox("Parent → C + D").active
    assert set(event_renderers()) == {(3,), (4,)}
    fates.active = list(range(len(fates.labels)))
    assert not checkbox("Parent → A + B").disabled
    assert not checkbox("Parent → A + B").active
    assert (
        checkbox("Parent → Parent").active and not checkbox("Parent → Parent").disabled
    )
    assert (2,) not in event_renderers() and (6,) in event_renderers()
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
    assert (6,) in event_renderers()
    fates.active = [fates.labels.index("Parent → C + D")]
    assert checkbox("Escape").disabled and checkbox("Escape").active

    # Pathway colours also survive a marker-layout change and regrouping.
    fates.active = list(range(len(fates.labels)))
    checkbox("Fragmentation").active = [0]
    grouping.active = []
    mode = next(s for s in doc.select({"type": Select}) if s.title == "Y coordinate")
    mode.value = "beeswarm"
    assert event_renderers()[(2,)].glyph.marker == "circle"
    assert event_renderers()[(2,)].glyph.fill_color == colors["Parent → A + B"]
    assert 'content: "●"' in checkbox("Parent → A + B").stylesheets[0].css
    fates.active = []
    assert all(w.disabled and w.active for w in options.children)
    assert doc.to_json() is not None


def test_categorical_pathway_palette():
    names = [f"Pathway {i:02}" for i in range(16)]
    colors = plot._pathway_colors(
        ["Incomplete", *reversed(names), "Survival"], "Survival"
    )
    assert colors[names[0]] == plot.COLORS["fragmentation"] == "#d62728"
    assert colors["Survival"] == plot.COLORS["escape"] == "#2ca02c"
    assert plot.COLORS["collision"] == "#1f77b4"
    assert [colors[name] for name in names[:8]] == [
        "#d62728",
        "#ff7f0e",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    pathway_colors = {colors[name] for name in names}
    assert len(pathway_colors) == len(names)
    assert pathway_colors <= set(plot.Category20[20])
    assert pathway_colors.isdisjoint({"#1f77b4", "#aec7e8", "#2ca02c", "#98df8a"})
    assert plot._pathway_colors(names[:2], "Survival")[names[1]] == colors[names[1]]


def test_details_use_cohort_number_and_convert_units():
    realizations, events, _, _ = cohort(
        [16052], event(42, "collision", 16052, 2e-9, 0.003)
    )
    events = events.with_columns(
        pl.Series("velocity", [{"x": -3.0, "y": 4.0, "z": -12.0}]),
        pl.Series("omega", [{"x": 2.0, "y": -3.0, "z": 6.0}]),
        pl.lit(7).alias("particle_index"),
    )
    selected = realizations.with_row_index("realization_number", offset=1)
    details = plot._details(selected, events, 42)
    assert "Realization #1</h3>" in details
    assert "Realization #16052" not in details
    assert "Result #" not in details and "disabled" not in details
    assert (
        "<td>2</td><td>3</td><td>5000</td><td>-12</td><td>5</td><td>7</td>" in details
    )
    assert all(
        f"<th>{heading}</th>" in details
        for heading in (
            "Axial velocity (m/s)",
            "Radial speed (m/s)",
            "Angular speed (rad/s)",
        )
    )
    assert "Particle index" not in details and "Pathway ID" not in details
    cartesian = plot._details(selected, events, 42, radial=False)
    assert (
        "<td>2</td><td>3000</td><td>4000</td><td>3</td><td>-3</td><td>4</td><td>-12</td><td>2</td><td>-3</td><td>6</td>"
        in cartesian
    )
    assert all(
        f"<th>{label} {axis} ({unit})</th>" in cartesian
        for label, unit in (("Velocity", "m/s"), ("Angular velocity", "rad/s"))
        for axis in "xyz"
    )
    assert "Radial speed" not in cartesian and "Angular speed" not in cartesian
    assert "Particle index" not in cartesian and "Pathway ID" not in cartesian


@pytest.mark.parametrize("mode", ["schematic", "equal", "physical", "time"])
def test_initial_events_position_and_fate(mode):
    members, events, _, regions = cohort(
        [1], event(1, "init", 1, 0, 0), event(2, "escape", 1, 1, 5)
    )
    assert members["terminal_event_id"].to_list() == [2]
    mapped = data.coordinate_regions(regions, mode)
    positioned = data.position_events(events, mapped, mode)
    if mode in ("schematic", "equal"):
        lo, hi = data.initial_slot(mapped)
        escape_lo, escape_hi = data.escape_slot(mapped)
        assert hi - lo == pytest.approx(escape_hi - escape_lo)
        assert positioned["position"][0] == (lo + hi) / 2 < mapped["left"][0]
    else:
        assert positioned["position"][0] == 0
    assert data.aggregate(members, positioned, mapped)[2:] == (1, 0)


def test_initial_event_controls_and_schematic(monkeypatch):
    from bokeh.document import Document
    from bokeh.models import CheckboxGroup, Label, Select

    frames = cohort([1], event(1, "init", 1, 0, 0), event(2, "escape", 1, 1, 5))
    monkeypatch.setattr(plot, "load_data", lambda *args: frames)
    doc = Document()
    build_document(None, doc, 1, 1)
    control = doc.select_one({"name": "event:init"})
    assert isinstance(control, CheckboxGroup)
    assert control.labels == ["Initial"] and not control.disabled
    coordinate = next(
        s for s in doc.select({"type": Select}) if s.title == "X coordinate"
    )
    for mode in ("schematic", "equal"):
        coordinate.value = mode
        chart = doc.select_one({"name": "events"})
        renderer = next(
            r for r in chart.renderers if r.data_source.data.get("id") == [1]
        )
        assert chart.x_range.start < renderer.data_source.data["x"][0] < 0
        mapped = data.coordinate_regions(frames[3], mode)
        assert chart.xgrid[0].ticker.ticks == [mapped["left"][0], *mapped["right"]]
        assert any(
            label.text in ("Initial", "Init") for label in doc.select({"type": Label})
        )
    control.active = []
    chart = doc.select_one({"name": "events"})
    assert all(1 not in r.data_source.data.get("id", []) for r in chart.renderers)
    control.active = [0]
    assert doc.to_json() is not None
