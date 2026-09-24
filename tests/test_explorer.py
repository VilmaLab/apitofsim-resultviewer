"""Numerical and storage contracts of the realization explorer."""

import duckdb

from apitofresview.plotting.explorer import (
    Cohort, Coordinates, Event, Realization, aggregate, build_document, classify, load_cohort,
    region_index,
)


def event(id, kind, rid, t, z, pathway=None):
    return Event(id, kind, rid, t, 3, 4, z, pathway)


def realization(rid, *events):
    r = Realization(rid, 10, list(events))
    classify(r)
    return r


def test_fates_coordinates_and_aggregate():
    a = realization(1, event(1, "collision", 1, .1, 0), event(2, "fragmentation", 1, 2, 2, 7))
    b = realization(2, event(3, "fragmentation", 2, 3, 2, 7))
    c = realization(3, event(4, "escape", 3, 4, 5))
    d = realization(4, event(5, "collision", 4, 1, 1))
    e = realization(5, event(6, "fragmentation", 5, 2, 2, 7), event(7, "escape", 5, 3, 5))
    assert [r.fate for r in (a, b, c, d, e)] == ["Pathway 7", "Pathway 7", "Escaped", "Incomplete", "Ambiguous"]
    physical = (0, 1, 2, 3, 4, 5)
    assert [region_index(v, physical) for v in (0, 1, 2, 5, -1)] == [0, 1, 2, 4, None]
    mapping = Coordinates(physical, "schematic")
    assert mapping.boundaries == (0, 2, 3, 4, 7, 9)
    assert mapping.position(a.terminal) == 3
    assert mapping.position(c.terminal) == sum(mapping.slot) / 2
    xs, ys, bars, escaped, unresolved = aggregate([a, b, c, d, e], mapping)
    assert xs == [3, 3]
    assert ys == [0, .4]
    assert bars == [0, 0, .4, 0, 0, .2]
    assert (escaped, unresolved) == (1, 2)
    assert aggregate([], mapping) == ([], [], [], 0, 0)
    assert aggregate([c], mapping)[2][-1] == 1
    assert aggregate([a], mapping)[1][-1] == 1
    assert Coordinates(physical, "time").position(a.terminal) == 2
    assert a.events[0].radial == 5


def test_adapter_resolves_results_and_preserves_ids(monkeypatch):
    conn = duckdb.connect(":memory:")
    conn.execute("create table multi_pathway_experiment_result(id int, experiment_run_id int, cluster_id int)")
    conn.execute("create table single_pathway_experiment_result(id int, experiment_run_id int, pathway_id int)")
    conn.execute("create table pathway(id int, cluster_id int, product1_id int, product2_id int)")
    conn.execute("create table cluster(id int, common_name varchar)")
    conn.execute("create table realization(id int, experiment_result_id int)")
    for kind in ("collision", "fragmentation", "escape"):
        suffix = ", pathway_id int" if kind == "fragmentation" else ""
        conn.execute(f"create table {kind}_event(id int, realization_id int, postime struct(x float,y float,z float,t float){suffix})")
    conn.execute("insert into cluster values (2,'A'),(3,'B')")
    conn.execute("insert into pathway values (7,1,2,3)")
    conn.execute("insert into multi_pathway_experiment_result values (10,1,1),(11,2,1)")
    conn.execute("insert into single_pathway_experiment_result values (12,1,7)")
    conn.execute("insert into realization values (100,10),(101,11),(102,12)")
    conn.execute("insert into collision_event values (501,100, {'x':1,'y':2,'z':0,'t':0.5})")
    conn.execute("insert into fragmentation_event values (502,100, {'x':1,'y':2,'z':2,'t':1.5},7)")
    conn.execute("insert into escape_event values (503,102, {'x':1,'y':2,'z':5,'t':2})")
    import apitofsim.plotting.events as geometry
    monkeypatch.setattr(geometry, "get_geometery", lambda *args: None)
    monkeypatch.setattr(geometry, "lengths_to_cumulative_lengths", lambda *args: (0, 1, 2, 3, 4, 5))
    db = type("DB", (), {"db": conn})()
    cohort = load_cohort(db, 1, 1)
    assert [r.id for r in cohort.realizations] == [100, 102]
    assert [r.fate for r in cohort.realizations] == ["Pathway 7", "Escaped"]
    assert [e.id for e in cohort.realizations[0].events] == [501, 502]
    assert cohort.realizations[0].events[1].pathway_id == 7
    assert cohort.pathways == {7: "A + B"}


def test_bokeh_document_modes_and_views(monkeypatch):
    from bokeh.document import Document
    from bokeh.models import Select, CheckboxGroup, GroupBox, CustomAction
    import apitofresview.plotting.explorer as explorer

    member = realization(1, event(1, "collision", 1, 0, 0), event(2, "escape", 1, 1, 5))
    monkeypatch.setattr(explorer, "load_cohort", lambda *args: Cohort([member], (0, 1, 2, 3, 4, 5), {}))
    doc = Document()
    build_document(None, doc, 1, 1)
    x_mode = next(s for s in doc.select({"type": Select}) if s.title == "X coordinate")
    for value in ("equal", "physical", "time", "schematic"):
        x_mode.value = value
    for toggle in doc.select({"type": CheckboxGroup}):
        if toggle.labels and toggle.labels[0] in ("CDF", "Regional bars", "Realizations"):
            toggle.active = [0]
    frame = doc.roots[0]
    sidebars = frame.children[0].children
    fullscreen = next(d for d in doc.select({"type": explorer.Div}) if d.text == "normal")
    fullscreen.text = "full"
    assert not sidebars[0].children[1].visible and not sidebars[2].children[1].visible
    assert "46px" in sidebars[0].children[1].styles["max-height"]
    fullscreen.text = "normal"
    assert sidebars[0].children[1].visible and not sidebars[2].children[1].visible
    viewport = next(d for d in doc.select({"type": explorer.Div}) if d.text == "1440,900")
    viewport.text = "533,900"
    sidebars[2].children[0].active = True
    assert not sidebars[0].children[1].visible
    assert sidebars[2].width == 250
    from bokeh.plotting import figure
    plots = list(doc.select({"type": type(figure())}))
    event_plots = [p for p in plots if p.title and p.title.text == "Realizations"]
    assert len(event_plots) == 1
    assert sum(bool(p.xaxis[0].visible) for p in plots) == 1
    assert {g.title for g in doc.select({"type": GroupBox})} >= {"Fates", "Events", "X axis", "Layout", "Views", "Selected realization"}
    assert all(p.toolbar.logo is None for p in plots)
    assert any(a.icon == "fullscreen" for a in doc.select({"type": CustomAction}))
    assert len(doc.roots) == 1
    assert doc.to_json() is not None
