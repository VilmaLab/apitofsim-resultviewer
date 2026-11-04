"""Escape colors and event filters stay consistent across Explorer views."""

from types import SimpleNamespace

import polars as pl
import pytest
from bokeh.document import Document
from bokeh.models import CheckboxGroup, GroupBox

from apitofresview.plotting.explorer import data, session
from apitofresview.plotting.explorer.escapes import EscapeGroups
from apitofresview.plotting.explorer.plot import COLORS

from .helpers import cohort, event
from . import test_tree

tree_database = test_tree.tree_database


def check(doc, label):
    return next(w for w in doc.select({"type": CheckboxGroup}) if w.labels == [label])


def event_colors(doc):
    return {
        id: color
        for chart in doc.select({"name": "events"})
        for renderer in chart.renderers
        if "id" in renderer.data_source.data
        for id, color in zip(
            renderer.data_source.data["id"], renderer.data_source.data["color"]
        )
    }


def cumulative_total(doc):
    return sum(
        renderer.data_source.data["fraction"][-1]
        for chart in doc.select({"name": "cdf"})
        for renderer in chart.renderers
    )


@pytest.fixture
def ambiguous_frames(tree_database):
    members, events, pathways, regions = data.load_data(tree_database, 1, 1)
    terminal = events.filter(
        (pl.col("realization_id") == 100) & (pl.col("type") == "escape")
    )
    added = terminal.with_columns(
        pl.lit(1000).cast(events.schema["id"]).alias("id"),
        pl.lit(105).cast(events.schema["realization_id"]).alias("realization_id"),
        pl.lit(4).cast(events.schema["particle_index"]).alias("particle_index"),
    )
    member = members.filter(pl.col("id") == 100).with_columns(
        pl.lit(105).cast(members.schema["id"]).alias("id"),
        pl.lit(1000)
        .cast(members.schema["terminal_event_id"])
        .alias("terminal_event_id"),
        pl.lit("10:escape:4").alias("fate"),
    )
    return pl.concat([members, member]), pl.concat([events, added]), pathways, regions


def test_escape_colors_preserve_daughters_when_distinguishing_routes():
    events = pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "type": ["escape"] * 4,
            "experiment_result_id": [10] * 4,
            "particle_index": [0, 1, 2, 3],
            "current_cluster_id": [1, 2, 2, 3],
            "current_cluster_name": ["Parent", "A", "A", "B"],
        }
    )
    tree = SimpleNamespace(
        leaf_labels=lambda: {f"10:escape:{i}": f"Route {i}" for i in range(4)}
    )
    escapes = EscapeGroups(events, 1, tree)
    grouped, distinct = escapes.route_colors(), escapes.route_colors(True)
    assert escapes.ambiguous
    assert grouped["10:escape:0"] == COLORS["escape"]
    assert grouped["10:escape:1"] == grouped["10:escape:2"]
    assert len(set(grouped.values())) == 3
    assert len(set(distinct.values())) == 4
    assert all(
        distinct[key] == grouped[key]
        for key in ("10:escape:0", "10:escape:1", "10:escape:3")
    )
    assert escapes.route_colors() == grouped
    assert escapes.route_colors(True) == distinct


def test_escape_palette_does_not_repeat_after_exhaustion():
    n = 40
    events = pl.DataFrame(
        {
            "id": range(n),
            "type": ["escape"] * n,
            "experiment_result_id": [10] * n,
            "particle_index": range(n),
            "current_cluster_id": range(n),
            "current_cluster_name": [f"Species {i}" for i in range(n)],
        }
    )
    escapes = EscapeGroups(events, 0, SimpleNamespace(leaf_labels=lambda: {}))
    assert len(set(escapes.route_colors().values())) == n
    assert escapes.colors["escape:daughter:0"] == COLORS["escape"]


def test_event_checkboxes_filter_summaries_without_changing_denominator(monkeypatch):
    members, events, _, regions = cohort(
        [1, 2, 3, 4],
        event(1, "fragmentation", 1, 1, 1, 7),
        event(2, "fragmentation", 2, 2, 2, 8),
        event(3, "escape", 3, 3, 5),
        event(4, "collision", 4, 1, 1),
    )
    pathways = pl.DataFrame(
        {
            "pathway_id": [7, 8],
            "pathway": ["Parent → A + B", "Parent → C + D"],
            "daughter": ["A", "C"],
        }
    )
    members = data.classify(
        members.select("id", "experiment_result_id"), events, pathways, "Parent"
    )
    monkeypatch.setattr(
        session, "load_data", lambda *args: (members, events, pathways, regions)
    )
    doc = Document()
    session.build_document(None, doc, 1, 1)
    check(doc, "Cumulative").active = [0]
    check(doc, "Bar chart").active = [0]
    assert cumulative_total(doc) == pytest.approx(3 / 4)
    check(doc, "Group fragmentations").active = []
    doc.select_one({"name": "event:Parent → A + B"}).active = []
    assert cumulative_total(doc) == pytest.approx(2 / 4)
    check(doc, "Group escapes").active = []
    doc.select_one({"name": "event:escape:daughter:1"}).active = []
    assert cumulative_total(doc) == pytest.approx(1 / 4)
    assert 1 not in event_colors(doc) and 3 not in event_colors(doc)
    assert 2 in event_colors(doc)
    bars = doc.select_one({"name": "bars"})
    assert sum(
        renderer.glyph.top - renderer.glyph.bottom for renderer in bars.renderers
    ) == pytest.approx(1 / 4)


def test_tree_ambiguity_control_and_colors(
    tree_database, ambiguous_frames, monkeypatch
):
    monkeypatch.setattr(session, "load_data", lambda *args: ambiguous_frames)
    doc = Document()
    session.build_document(tree_database, doc, 1, 1)
    distinguish = check(doc, "Distinguish identical escapees by pathway")
    assert distinguish.visible and distinguish.active == []
    assert check(doc, "Group escapes").active == [0]
    assert set(event_colors(doc)[id] for id in (6, 8, 12, 1000)) == {COLORS["escape"]}
    check(doc, "Group escapes").active = []
    before = event_colors(doc)
    check(doc, "Cumulative").active = [0]
    check(doc, "Bar chart").active = [0]
    doc.select_one({"name": "cumulative_scope"}).value = "10:4"
    assert cumulative_total(doc) == pytest.approx(1 / 6)
    assert doc.select_one({"name": "cdf"}).renderers[0].glyph.fill_color == before[1000]
    assert (
        len(
            [r for r in doc.select_one({"name": "bars"}).renderers if r.glyph.left == 9]
        )
        == 3
    )
    assert before[6] == before[1000] and before[6] != before[8]
    assert before[8] == COLORS["escape"]
    assert any(group.title == "Escapes" for group in doc.select({"type": GroupBox}))
    assert doc.select_one({"name": "event:escape:daughter:4"}).labels == ["→ Leaf"]
    distinguish.active = [0]
    after = event_colors(doc)
    assert after[6] != after[1000]
    assert doc.select_one({"name": "cdf"}).renderers[0].glyph.fill_color == after[1000]
    assert (
        len(
            [r for r in doc.select_one({"name": "bars"}).renderers if r.glyph.left == 9]
        )
        == 4
    )
    assert all(after[id] == before[id] for id in (6, 8, 12))
    assert doc.select_one({"name": "event:10:escape:4"}).labels == ["→ Leaf"]
    distinguish.active = []
    assert event_colors(doc) == before
    check(doc, "Group fragmentations").active = []
    assert any(
        group.title == "Fragmentation" for group in doc.select({"type": GroupBox})
    )
    assert doc.select_one({"name": "event:Parent → A + Neutral"}).labels == ["→ A"]
    doc.select_one({"name": "cumulative_scope"}).value = "root"
    doc.select_one({"name": "event:Parent → A + Neutral"}).active = []
    assert cumulative_total(doc) == pytest.approx(2 / 6)
    assert 2 not in event_colors(doc) and 6 in event_colors(doc)


def test_tree_without_recorded_ambiguity_hides_distinction(tree_database):
    doc = Document()
    session.build_document(tree_database, doc, 1, 1)
    assert not check(doc, "Distinguish identical escapees by pathway").visible


@pytest.mark.parametrize("scope", ["root", "10:1"])
def test_cumulative_icon_reopens_selected_scope(
    tree_database, monkeypatch, page, scope
):
    from playwright.sync_api import expect
    from .test_browser import serve_explorer

    frames = data.load_data(tree_database, 1, 1)
    with serve_explorer(monkeypatch, frames, tree_database) as url:
        page.goto(url)
        cumulative = page.get_by_label("Cumulative", exact=True)
        expect(cumulative).not_to_be_checked()
        button = page.locator(f'button[data-scope="{scope}"]')
        button.click()
        expect(cumulative).to_be_checked()
        page.wait_for_function(
            "() => Bokeh.documents[0].get_model_by_name('explorer_plots').children.some(p => p.name === 'cdf')"
        )
        cumulative.uncheck()
        page.wait_for_function(
            "() => !Bokeh.documents[0].get_model_by_name('explorer_plots').children.some(p => p.name === 'cdf')"
        )
        button.click()
        expect(cumulative).to_be_checked()
        page.wait_for_function(
            "() => Bokeh.documents[0].get_model_by_name('explorer_plots').children.some(p => p.name === 'cdf')"
        )
        assert (
            page.evaluate(
                "Bokeh.documents[0].get_model_by_name('cumulative_scope').value"
            )
            == scope
        )


def test_browser_escape_groups_and_ambiguity(
    tree_database, ambiguous_frames, monkeypatch, page, tmp_path
):
    from playwright.sync_api import expect
    from .test_browser import serve_explorer

    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    with serve_explorer(monkeypatch, ambiguous_frames, tree_database) as url:
        page.set_viewport_size({"width": 920, "height": 900})
        page.goto(url)
        distinguish = page.get_by_label(
            "Distinguish identical escapees by pathway", exact=True
        )
        expect(distinguish).to_be_visible()
        expect(distinguish).not_to_be_checked()
        page.get_by_label("Group escapes", exact=True).uncheck()
        escapes = page.get_by_role("group", name="Escapes", exact=True)
        expect(escapes.get_by_label("→ Leaf", exact=True)).to_have_count(1)
        page.get_by_label("Group fragmentations", exact=True).uncheck()
        expect(
            page.get_by_role("group", name="Fragmentation", exact=True)
        ).to_be_visible()
        distinguish.check()
        expect(escapes.get_by_label("→ Leaf", exact=True)).to_have_count(2)
        escapes.get_by_label("→ Leaf", exact=True).first.uncheck()
        page.wait_for_function("""() => {
            const chart = Bokeh.documents[0].get_model_by_name('explorer_plots').children.find(p => p.name === 'events');
            const ids = chart.renderers.flatMap(r => r.data_source?.data.id ?? []);
            return !ids.includes(6) && ids.includes(1000);
        }""")
        page.screenshot(path=str(tmp_path / "escape-groups.png"))
        assert not errors
