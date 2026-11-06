"""Serial histories use persisted routes and count splits independently of outcomes."""

from xml.etree import ElementTree

import polars as pl
import pytest
from apitofsim.workflow.db import RealizationDatabase
from bokeh.document import Document
from bokeh.models import CheckboxGroup, Select

from apitofresview.plotting.explorer import aggregation, data, session
from apitofresview.plotting.explorer.details import render_details
from apitofresview.plotting.explorer.layout import coordinate_regions, position_events
from apitofresview.plotting.explorer.tree import ExecutionTree


@pytest.fixture
def tree_database(tmp_path, monkeypatch):
    path = tmp_path / "tree.duckdb"
    db = RealizationDatabase(path)
    db.create_tables()
    db.db.execute("""
        insert into cluster (id,common_name,charge) values (1,'Parent',-1),(2,'A',-1),(3,'Neutral',0),(4,'Leaf',-1);
        insert into pathway values (7,1,2,3),(8,1,3,2),(9,2,4,3),(10,4,4,3),(11,3,1,2),(12,2,3,4);
        insert into experiment_config (id) values (1);
        insert into experiment_run (id,experiment_config_id,simulation_mode) values (1,1,'CLUSTER_TREE'),(2,1,'CLUSTER_TREE');
        insert into cluster_tree_experiment_result (id,experiment_run_id,cluster_id,total_us) values (10,1,1,100);
        insert into cluster_tree_node values (10,0,1),(10,1,2),(10,2,2),(10,3,4),(10,4,4);
        insert into cluster_tree_transition values (10,0,0,7,1),(10,1,0,8,2),(10,2,1,9,3),(10,3,2,9,4),(10,4,3,10,NULL),(10,5,2,12,NULL);
        insert into realization values (100,10),(101,10),(102,10),(103,10),(104,10);
    """)
    next_id = 0

    def event(rid, kind, node, t, transition=None, dest=None, pathway=None):
        nonlocal next_id
        next_id += 1
        db.db.execute(
            """
            insert into event_info values (?, ?, ?, {'x':0,'y':0,'z':?,'t':?},
            {'x':0,'y':0,'z':?}, {'x':0,'y':0,'z':0}, ?, ?, ?)
        """,
            [
                next_id,
                rid,
                kind,
                t,
                t,
                node + 1,
                (node + 1) * 1e-20,
                (node + 1) * 2e-20,
                node,
            ],
        )
        if kind == "fragmentation":
            db.db.execute(
                "insert into fragmentation_event values (?,?,?,?)",
                [next_id, transition, dest, pathway],
            )
        return next_id

    for rid in range(100, 105):
        event(rid, "init", 0, 0)
        if rid == 101:
            event(rid, "escape", 0, 5)
            continue
        node, transition, pathway = (2, 1, 8) if rid == 102 else (1, 0, 7)
        event(rid, "fragmentation", 0, 1, transition, node, pathway)
        if rid == 103:
            continue  # Recording interrupted at a continuing split.
        event(rid, "postfragmentation", node, 1)
        if rid == 102:
            event(rid, "escape", node, 5)
            continue
        event(rid, "fragmentation", node, 2, 2, 3, 9)
        event(rid, "postfragmentation", 3, 2)
        if rid == 104:
            event(rid, "fragmentation", 3, 3, 4, -1, 10)
        else:
            event(rid, "escape", 3, 5)
    db.close()
    monkeypatch.setattr(data, "get_geometry", lambda *args: None)
    monkeypatch.setattr(
        data, "lengths_to_cumulative_lengths", lambda *args: (0, 1, 2, 3, 4, 5)
    )
    db = RealizationDatabase(path, readonly=True)
    try:
        yield db
    finally:
        db.close()


def test_tree_identity_classification_and_after_states(tree_database):
    members, events, pathways, _ = data.load_data(tree_database, 1, 1)
    assert members["fate"].to_list() == [
        "10:escape:3",
        "10:escape:0",
        "10:escape:2",
        "10:incomplete:1",
        "10:stop:4",
    ]
    assert members["terminal_event_id"].null_count() == 1
    assert (
        events.filter(pl.col("type") == "postfragmentation")[
            "fragmentation_id"
        ].null_count()
        == 0
    )
    assert set(pathways["pathway_id"]) == {7, 8, 9, 10, 12}
    tree = ExecutionTree.load(tree_database, 1, 1, pathways)
    labels = tree.leaf_labels()
    assert "10:escape:4" in labels  # Enabled, unobserved route.
    assert (
        labels["10:escape:1"] != labels["10:escape:2"]
    )  # Same species, separate routes.
    assert "A → Leaf + Neutral" in labels["10:escape:3"]
    assert labels["10:escape:3"].endswith("Leaf → Leaf")
    assert tree.short_labels()["10:escape:3"] == "Parent → A → Leaf → Leaf"
    assert tree.short_label("10:stop:5", "") == "→ Leaf"  # Charged product is second.


def test_all_splits_in_bars_and_immediate_splits_in_cumulative(tree_database):
    members, events, pathways, regions = data.load_data(tree_database, 1, 1)
    tree = ExecutionTree.load(tree_database, 1, 1, pathways)
    events = events.join(pathways, on="pathway_id", how="left").rename(
        {"pathway": "event_pathway"}
    )
    mapped = coordinate_regions(regions, "time")
    events = position_events(events, mapped, "time")
    bars = aggregation.regional_fractions(members, events, mapped)
    assert bars["fraction"][:-1].sum() == pytest.approx(7 / 5)
    assert bars["fraction"][-1] == pytest.approx(3 / 5)
    root = tree.cumulative_members(members, events, "root")
    areas = aggregation.cumulative_areas(
        root, events, 0, 5, root["fate"].drop_nulls().unique().sort()
    )
    assert sum(frame["fraction"][-1] for frame in areas.values()) == pytest.approx(1)
    assert root.filter(pl.col("fate") == "Parent → A + Neutral").height == 3
    subtree = tree.cumulative_members(members, events, "10:1")
    assert subtree["terminal_event_id"].drop_nulls().len() == 2
    areas = aggregation.cumulative_areas(subtree, events, 0, 5, ["A → Leaf + Neutral"])
    assert areas["A → Leaf + Neutral"]["fraction"][-1] == pytest.approx(2 / 5)


def test_fragmentation_details_share_spacetime_but_keep_both_states(tree_database):
    members, events, pathways, _ = data.load_data(tree_database, 1, 1)
    members = members.filter(pl.col("id") == 100).with_row_index(
        "realization_number", offset=1
    )
    events = (
        events.filter(pl.col("realization_id") == 100)
        .join(pathways, on="pathway_id", how="left")
        .rename({"pathway": "event_pathway"})
    )
    after = events.filter(pl.col("type") == "postfragmentation").row(0, named=True)
    html = render_details(members, events, after["id"])
    parsed = ElementTree.fromstring("<div>" + html + "</div>")
    before_row = parsed.find(f".//tr[@id='event-{after['fragmentation_id']}']")
    after_row = parsed.find(f".//tr[@id='event-{after['id']}']")
    assert len(before_row.findall('td[@rowspan="2"]')) == 7
    assert "→ A" in "".join(before_row.itertext())
    assert before_row.find(".//span").get("title") == "Parent → A + Neutral"
    assert "Neutral" not in "".join(before_row.itertext())
    assert "Species / state" not in html
    assert (
        after_row.find(".//button").get("aria-label") == f"Inspect event #{after['id']}"
    )
    assert len(before_row.findall("td")) == len(parsed.findall(".//th"))
    assert len(after_row.findall("td")) == len(parsed.findall(".//th")) - 7
    assert "3e-20" in "".join(before_row.itertext())
    assert "6e-20" in "".join(after_row.itertext())
    assert "Parent → A + Neutral" in html
    assert after_row.get("style") == "background:#fde68a"


def check(doc, label):
    return next(w for w in doc.select({"type": CheckboxGroup}) if w.labels == [label])


def test_session_scope_selection_and_complete_history_facets(tree_database):
    doc = Document()
    session.build_document(tree_database, doc, 1, 1)
    fate = next(
        w for w in doc.select({"type": CheckboxGroup}) if "10:escape:3" in w.labels
    )
    assert not fate.visible
    assert 'data-scope="10:1"' in doc.select_one({"name": "fragmentation_tree"}).text
    check(doc, "Group fragmentations").active = []
    assert doc.select_one({"name": "event:escape"}).labels == ["Escape"]
    scope = doc.select_one({"name": "cumulative_scope"})
    scope.value = "10:1"
    assert check(doc, "Cumulative").active == [0]
    cdf = doc.select_one({"name": "cdf"})
    assert "Cumulative from A" in cdf.title.text
    assert sum(
        r.data_source.data["fraction"][-1] for r in cdf.renderers
    ) == pytest.approx(2 / 5)
    fate.active = [fate.labels.index("10:escape:3")]
    check(doc, "Facet histories").active = [0]
    chart = doc.select_one({"name": "events"})
    ids = [id for r in chart.renderers for id in r.data_source.data.get("id", [])]
    members, events, _, _ = data.load_data(tree_database, 1, 1)
    history = events.filter(
        (pl.col("realization_id") == 100) & (pl.col("type") != "postfragmentation")
    )
    assert set(ids) == set(history["id"])
    source = next(r.data_source for r in chart.renderers if "id" in r.data_source.data)
    source.selected.indices = [0]
    details = next(d for d in doc.select({"name": "explorer_plots"}))
    assert members.height == 5 and details is not None
    assert doc.to_json() is not None
    fate.active = list(range(len(fate.labels)))
    assert len(list(doc.select({"name": "events"}))) == 5


@pytest.mark.parametrize("width", [640, 1280])
def test_browser_tree_selection_folding_and_scope(
    tree_database, monkeypatch, page, tmp_path, width
):
    from playwright.sync_api import expect

    from .test_browser import serve_explorer

    frames = data.load_data(tree_database, 1, 1)
    tree = ExecutionTree.load(tree_database, 1, 1, frames[2])
    tree.incomplete = set(frames[0]["fate"])
    names = list(tree.leaf_labels())
    names.remove("Incomplete")
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    with serve_explorer(monkeypatch, frames, tree_database) as url:
        page.set_viewport_size({"width": width, "height": 900})
        page.goto(url)
        root = page.locator("details").first
        root_input = root.locator("summary").first.locator("input")
        expect(root_input).to_be_checked()
        full_tree = page.get_by_label("Show empty pathways", exact=True)
        expect(full_tree).not_to_be_checked()
        unobserved = page.locator('.tree-leaf[data-observed="false"]').first
        expect(unobserved).not_to_be_visible()
        full_tree.check()
        expect(unobserved).to_be_visible()
        expect(unobserved.locator("span[aria-hidden]")).to_have_count(0)
        full_tree.uncheck()
        expect(unobserved).not_to_be_visible()
        expect(
            root.locator("summary").first.locator("svg path").first
        ).to_have_attribute("d", "M16 128C88 128 88 20 160 20")
        expect(
            page.locator('label[title="Parent → A + Neutral"]').first
        ).to_contain_text("→ A")
        assert (
            "Neutral"
            not in page.locator(
                'label[title="Parent → A + Neutral"]'
            ).first.inner_text()
        )
        root_input.uncheck()
        expect(
            page.get_by_text("No realizations in this cohort.", exact=False)
        ).to_be_visible()
        leaf = page.locator(
            f'.tree-leaf input[data-leaves="{names.index("10:escape:3")}"]'
        )
        leaf.check()
        expect(root_input).not_to_be_checked()
        assert root_input.evaluate("el => el.indeterminate")
        # Non-leaf checkboxes are shortcuts, so choosing another subtree retains this leaf.
        branch = (
            page.locator("details")
            .filter(has=page.locator('summary > button[data-scope="10:2"]'))
            .last
        )
        branch.locator("summary").first.locator("input").check()
        expect(leaf).to_be_checked()
        assert root_input.evaluate("el => el.indeterminate")
        button = page.locator('button[data-scope="10:1"]')
        button.click()
        expect(button).to_have_attribute("aria-pressed", "true")
        page.wait_for_function(
            "() => Bokeh.documents[0].get_model_by_name('cdf')?.title.text.includes('Cumulative from A')"
        )
        branch.locator("summary").first.click(position={"x": 3, "y": 3})
        expect(branch).not_to_have_attribute("open", "")
        # Lazy regeneration preserves folded branches and the independent cumulative scope.
        full_tree.check()
        expect(branch).not_to_have_attribute("open", "")
        full_tree.uncheck()
        expect(branch).not_to_have_attribute("open", "")
        leaf.uncheck()
        expect(branch).not_to_have_attribute("open", "")
        expect(button).to_have_attribute("aria-pressed", "true")
        page.screenshot(path=str(tmp_path / f"tree-{width}.png"))
        assert not errors


def test_rate_panels_follow_before_and_after_species(tree_database, monkeypatch):
    from types import SimpleNamespace

    import numpy as np
    from apitofsim.api import Histogram, Q_
    from bokeh.models import Span

    from apitofresview.plotting.explorer import rates

    from .test_rates import CachedCurve, select

    curve_reads, profile_reads = [], []

    def curves(run, parent, **kwargs):
        curve_reads.append(parent)
        return [
            CachedCurve(
                pathway_id=pathway,
                participating=True,
                bonding_energy=Q_(0, "J"),
                histogram=Histogram(
                    Q_(np.array([1, 2, 3]) * 1e-20, "J"), np.array([1.0, 10.0, 100.0])
                ),
            )
            for pathway in ([7, 8] if parent == 1 else [9, 12])
        ]

    def profile_inputs(**kwargs):
        profile_reads.append(kwargs["cluster_id"])
        return {}, kwargs["cluster_id"]

    monkeypatch.setattr(tree_database, "pathway_rate_curves", curves)
    monkeypatch.setattr(tree_database, "collision_profile_inputs", profile_inputs)
    monkeypatch.setattr(rates, "skimmer", lambda config: None)
    monkeypatch.setattr(
        rates,
        "collision_profile",
        lambda config, cluster, **kwargs: SimpleNamespace(
            frequency=lambda z, velocity, **kwargs: Q_(
                np.full(len(z), cluster * sum(velocity)), "1/s"
            )
        ),
    )
    doc = Document()
    session.build_document(tree_database, doc, 1, 1)
    check(doc, "Gas collision rate").active = [0]
    select(doc, 2)  # First fragmentation, before: Parent.
    check(doc, "Show rate constants").active = [0]
    assert curve_reads == [1]
    assert len(doc.select_one({"name": "pathway_rates"}).renderers) == 2
    main = doc.select_one({"name": "events"})
    before_highlight = dict(
        next(
            r.data_source.data
            for r in main.renderers
            if getattr(r.glyph, "line_color", None) == "#f59e0b"
        )
    )
    doc.roots[0].children[1].text = "3"  # Same split, after: A.
    assert curve_reads == [1, 2]
    assert profile_reads[-1] == 2
    assert len(doc.select_one({"name": "pathway_rates"}).renderers) == 1
    assert doc.select_one({"name": "pathway_rates"}).select_one(
        {"type": Span}
    ).location == pytest.approx(6e-20, abs=1e-32)
    source = doc.select_one({"name": "collision_rates_source"})
    np.testing.assert_allclose(source.data["rate"], 4)
    main = doc.select_one({"name": "events"})
    highlight = next(
        r.data_source
        for r in main.renderers
        if getattr(r.glyph, "line_color", None) == "#f59e0b"
    )
    assert highlight.data == before_highlight
    doc.roots[0].children[1].text = "2"
    assert curve_reads == [1, 2]  # Cached separately for each species.
    assert profile_reads[-1] == 1
    assert len(doc.select_one({"name": "pathway_rates"}).renderers) == 2


def test_empty_tree_run(tree_database):
    doc = Document()
    session.build_document(tree_database, doc, 2, 1)
    assert (
        doc.roots[0].text == "No recorded cluster-tree executions for this selection."
    )


def test_inconsistent_recording_stays_incomplete_and_does_not_pair(tree_database):
    members, events, pathways, _ = data.load_data(tree_database, 1, 1)
    events = events.with_columns(
        pl.when(pl.col("id") == 3)
        .then(pl.lit(False))
        .otherwise(pl.col("continuity_valid"))
        .alias("continuity_valid")
    )
    tree = ExecutionTree.load(tree_database, 1, 1, pathways)
    classified = tree.classify(members.select("id", "experiment_result_id"), events)
    assert classified.filter(pl.col("id") == 100)["terminal_event_id"][0] is None
    assert (
        data.pair_fragmentations(events).filter(pl.col("id") == 3)["fragmentation_id"][
            0
        ]
        is None
    )
    member = classified.filter(pl.col("id") == 100).with_row_index(
        "realization_number", offset=1
    )
    assert "Recording diagnostics: unresolved continuity" in render_details(
        member, events.filter(pl.col("realization_id") == 100), 3
    )


@pytest.mark.parametrize("pathway_mode", [False, True])
def test_browser_daughter_labels_in_other_modes(monkeypatch, page, pathway_mode):
    from playwright.sync_api import expect

    from .helpers import cohort, event, pathway_cohort
    from .test_browser import serve_explorer

    frames = (
        pathway_cohort()
        if pathway_mode
        else cohort(
            [1, 2], event(1, "fragmentation", 1, 1, 2, 7), event(2, "escape", 2, 2, 5)
        )
    )
    pathways = frames[2].with_columns(
        pl.Series("daughter", ["A", "C", "E", "G"] if pathway_mode else ["A"])
    )
    frames = (frames[0], frames[1], pathways, frames[3])
    with serve_explorer(monkeypatch, frames) as url:
        page.goto(url)
        label = page.locator('label[title="Parent → A + B"]')
        expect(label).to_contain_text("→ A")
        assert "+ B" not in label.inner_text()
        expect(page.get_by_label("Show empty pathways", exact=True)).to_have_count(0)
        expect(page.get_by_label("Group escapes", exact=True)).not_to_be_visible()
        if not pathway_mode:
            expect(page.locator('label[title="Parent → Parent"]')).to_contain_text(
                "→ Parent"
            )
            page.get_by_text("Facet pathways", exact=True).click()
        page.wait_for_function("""() => {
            const plot = Bokeh.documents[0].get_model_by_name('explorer_plots').children.find(p => p.name === 'events');
            return plot?.title?.text.startsWith('→ A');
        }""")
        assert (
            page.evaluate(
                "Bokeh.documents[0].get_model_by_name('explorer_plots').children.find(p => p.name === 'events').html_attributes.title"
            )
            == "Parent → A + B"
        )


def test_tree_colors_use_observed_pathways_without_changing_palette(tree_database):
    from apitofresview.plotting.explorer.plot import _pathway_colors

    doc = Document()
    session.build_document(tree_database, doc, 1, 1)
    html = doc.select_one({"name": "fragmentation_tree"}).text
    expected = _pathway_colors(
        [
            "Parent → A + Neutral",
            "Parent → Neutral + A",
            "A → Leaf + Neutral",
            "Leaf → Leaf + Neutral",
        ],
        "Escape",
    )
    # Only observed pathways participate in palette allocation; adding a zero-event branch cannot shift them.
    for pathway, color in expected.items():
        if pathway == "Escape":
            continue
        assert f"color:{color}" in html
    assert "A → Neutral + Leaf · Stop" not in html
    check(doc, "Show empty pathways").active = [0]
    html = doc.select_one({"name": "fragmentation_tree"}).text
    assert "A → Neutral + Leaf · Stop" in html
    assert (
        "color:"
        not in html.split('title="Parent → Neutral + A / A → Neutral + Leaf · Stop"')[
            -1
        ].split("</label>")[0]
    )


def test_large_unobserved_tree_is_built_only_on_request(tree_database, monkeypatch):
    frames = data.load_data(tree_database, 1, 1)
    tree = ExecutionTree.load(tree_database, 1, 1, frames[2])
    for node in range(100, 4100):
        tree.nodes[10, node] = 4
        tree.transitions[10, node] = (0, 9, node)
    monkeypatch.setattr(session, "load_data", lambda *args: frames)
    monkeypatch.setattr(ExecutionTree, "load", lambda *args: tree)
    doc = Document()
    session.build_document(tree_database, doc, 1, 1)
    view = doc.select_one({"name": "fragmentation_tree"})
    assert 'data-scope="10:4099"' not in view.text
    assert len(view.text) < 100_000
    fate = next(
        w for w in doc.select({"type": CheckboxGroup}) if "10:escape:4099" in w.labels
    )
    assert len(fate.stylesheets[0].css) < 2_000
    check(doc, "Show empty pathways").active = [0]
    assert 'data-scope="10:4099"' in view.text
    fate.active = [fate.labels.index("10:escape:3")]
    check(doc, "Show empty pathways").active = []
    assert 'data-scope="10:4099"' not in view.text
    assert fate.labels[fate.active[0]] == "10:escape:3"


def test_species_population_tracks_serial_history_and_ignores_event_filters(
    tree_database,
):
    doc = Document()
    session.build_document(tree_database, doc, 1, 1)
    mode = next(
        widget
        for widget in doc.select({"type": Select})
        if widget.title == "X coordinate"
    )
    mode.value = "time"
    check(doc, "Species populations").active = [0]
    check(doc, "Realizations").active = []
    assert not check(doc, "Cumulative").active

    def fractions_at(t):
        plot = doc.select_one({"name": "population"})
        fractions = {}
        for renderer in plot.renderers:
            frame = renderer.data_source.data
            index = max(i for i, x in enumerate(frame["position"]) if x <= t)
            fractions[renderer.name] = frame["fraction"][index]
        return {key: value for key, value in fractions.items() if value}

    assert fractions_at(0) == {"Parent": 1}
    assert fractions_at(1) == {"Parent": 0.2, "A": 0.6}
    assert fractions_at(3) == {"Parent": 0.2, "A": 0.2, "Leaf": 0.4}
    assert fractions_at(5) == fractions_at(3)
    chart = doc.select_one({"name": "population"})
    assert all(
        3 not in r.data_source.data["position"]
        and 5 not in r.data_source.data["position"]
        for r in chart.renderers
    )
    colors = {r.name: r.glyph.fill_color for r in chart.renderers}
    legend = doc.select_one({"name": "species_legend"})
    assert all(f"color:{color}" in legend.text for color in colors.values())

    check(doc, "Distinguish identical escapees by pathway").active = [0]
    check(doc, "Group fragmentations").active = []
    pathway = doc.select_one({"name": "event:Parent → A + Neutral"})
    assert f"color: {colors['A']};" in pathway.stylesheets[0].css
    pathway.active = []
    assert fractions_at(1)["A"] == 0.6
    assert {
        r.name: r.glyph.fill_color
        for r in doc.select_one({"name": "population"}).renderers
    } == colors

    mode.value = "schematic"
    chart = doc.select_one({"name": "population"})
    assert chart.x_range.start < 0
    assert fractions_at(chart.x_range.start) == fractions_at(0) == {"Parent": 1}

    check(doc, "Facet histories").active = [0]
    assert len(list(doc.select({"name": "population"}))) == 5


def test_browser_population_band_hover(tree_database, monkeypatch, page):
    from playwright.sync_api import expect

    from .test_browser import serve_explorer

    frames = data.load_data(tree_database, 1, 1)
    with serve_explorer(monkeypatch, frames, tree_database) as url:
        page.set_viewport_size({"width": 1280, "height": 900})
        page.goto(url)
        page.get_by_label("X coordinate").select_option(label="Elapsed time (s)")
        page.get_by_label("Species populations", exact=True).check()
        page.get_by_label("Realizations", exact=True).uncheck()
        point = page.wait_for_function("""() => {
            const doc = Bokeh.documents[0];
            const plots = doc.get_model_by_name('explorer_plots').children;
            const plot = plots.find(p => p.name === 'population');
            const view = plot && Bokeh.index.find_one(plot);
            if (plots.some(p => p.name === 'events') || !plot?.inner_width ||
                !plot.below.some(axis => axis.axis_label === 'Elapsed time (s)') ||
                !view?.el?.isConnected)
                return false;
            const bounds = view.canvas_view.events_el.getBoundingClientRect();
            return {x: bounds.left + view.frame.x_scale.compute(1.5),
                    y: bounds.top + view.frame.y_scale.compute(0.5)};
        }""").json_value()
        page.mouse.move(point["x"], point["y"])
        expect(page.get_by_text("Charged species", exact=True)).to_be_visible()
        expect(page.get_by_text("0.600", exact=True)).to_be_visible()
