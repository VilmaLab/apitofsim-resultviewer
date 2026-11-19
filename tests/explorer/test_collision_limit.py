"""Collision budgets follow the selected cohort without truncating histories."""

import polars as pl
import pytest
from bokeh.document import Document
from bokeh.models import CheckboxGroup, Div, RangeSlider

from apitofresview.plotting.explorer import session

from .helpers import cohort, event


@pytest.mark.parametrize("limit, safe", [(6, 1), (7, 2), (3, 0)])
def test_collision_budget_and_pager(monkeypatch, limit, safe):
    # Intentionally not sorted by size. Postcollision states must not count twice.
    rows = []
    for rid, count in enumerate([1, 3, 4, 0], start=1):
        for t in range(count):
            rows.append(event(len(rows) + 1, "collision", rid, t, 1))
            rows.append(event(len(rows) + 1, "postcollision", rid, t, 1))
        rows.append(event(len(rows) + 1, "escape", rid, 10, 5))
    members, events, pathways, regions = cohort([1, 2, 3, 4], *rows)
    events = events.with_columns((pl.col("id") % 3 == 0).alias("accepted"))
    monkeypatch.setattr(session, "MAX_COLLISIONS", limit)
    monkeypatch.setattr(
        session, "load_data", lambda *args: (members, events, pathways, regions)
    )
    doc = Document()
    session.build_document(None, doc, 1, 1)
    collision = doc.select_one({"name": "event:collision"})
    notice = next(d for d in doc.select({"type": Div}) if "Collisions hidden" in d.text)

    def widget(label):
        return next(
            w for w in doc.select({"type": CheckboxGroup}) if w.labels == [label]
        )

    def plotted_types():
        return {
            kind
            for renderer in doc.select_one({"name": "events"}).renderers
            if hasattr(renderer, "data_source")
            for kind in renderer.data_source.data.get("type", [])
        }

    assert collision.disabled
    assert "collision" not in plotted_types()
    assert "escape" in plotted_types()
    if safe:
        assert f"no more than {safe} realization" in notice.text
    else:
        assert "a single realization can exceed the limit" in notice.text
    widget("Separate rejected collisions").active = [0]
    rejected = doc.select_one({"name": "event:rejected_collision"})
    assert collision.disabled and rejected.disabled
    assert "collision" not in plotted_types()

    pager = widget("Realizations pager")
    slider = doc.select_one({"type": RangeSlider})
    pager.active = [0]
    # Includes exactly the limit for limit=7, and a smaller actual total otherwise.
    slider.value = (2, 3) if limit == 7 else (1, 1)
    assert not collision.disabled
    assert "Collisions hidden" not in notice.text
    assert "collision" in plotted_types()
    # Expanding or switching the pager off must protect the plot again.
    pager.active = []
    assert collision.disabled and rejected.disabled
    assert "collision" not in plotted_types()
    assert "Collisions hidden" in notice.text
    pager.active = [0]
    assert not collision.disabled
    slider.value = (4, 4)
    assert "Collisions hidden" not in notice.text
    assert "collision" not in plotted_types()


def test_pathway_collision_budget_is_shared(monkeypatch):
    from .helpers import pathway_cohort

    members, events, pathways, regions = pathway_cohort()
    # Put one collision in each of two simulated pathways, sharing page number 1.
    events = events.with_columns(
        pl.when(pl.col("id").is_in([1, 6]))
        .then(pl.lit("collision"))
        .otherwise(pl.col("type"))
        .alias("type")
    )
    monkeypatch.setattr(session, "MAX_COLLISIONS", 1)
    monkeypatch.setattr(
        session, "load_data", lambda *args: (members, events, pathways, regions)
    )
    doc = Document()
    session.build_document(None, doc, 1, 1)
    collision = doc.select_one({"name": "event:collision"})
    pager = next(
        w
        for w in doc.select({"type": CheckboxGroup})
        if w.labels == ["Realizations pager"]
    )
    pager.active = [0]
    doc.select_one({"type": RangeSlider}).value = (1, 1)
    assert collision.disabled  # One realization per pathway is two in total.
    fate = next(
        w for w in doc.select({"type": CheckboxGroup}) if "Parent → C + D" in w.labels
    )
    fate.active = [fate.labels.index("Parent → C + D")]
    assert not collision.disabled


def test_collision_limit_browser(monkeypatch, page):
    from playwright.sync_api import expect

    from .test_browser import serve_explorer

    frames = cohort(
        [1, 2],
        event(1, "collision", 1, 0, 1),
        event(2, "escape", 1, 1, 5),
        event(3, "collision", 2, 0, 1),
        event(4, "escape", 2, 1, 5),
    )
    monkeypatch.setattr(session, "MAX_COLLISIONS", 1)
    with serve_explorer(monkeypatch, frames) as url:
        page.goto(url)
        collision = page.get_by_label("Collision", exact=True)
        expect(collision).to_be_disabled()
        expect(page.get_by_text("Collisions hidden:")).to_contain_text(
            "no more than 1 realization in total"
        )
        page.get_by_label("Realizations pager", exact=True).check()
        page.evaluate("""() => {
            const slider = [...Bokeh.documents[0]._all_models.values()]
                .find(m => m.type === 'RangeSlider');
            slider.value = [1, 1];
        }""")
        expect(collision).to_be_enabled()
        expect(page.get_by_text("Collisions hidden:")).to_have_count(0)
        page.get_by_label("Realizations pager", exact=True).uncheck()
        expect(collision).to_be_disabled()


def test_details_reveal_one_collision_above_the_limit(monkeypatch, page):
    from playwright.sync_api import expect

    from .helpers import collision_cohort
    from .test_browser import serve_explorer

    members, events, pathways, regions = collision_cohort()
    overview = members.with_columns(
        pl.lit(3).alias("collisions"),
        pl.lit(1).alias("rejected_collisions"),
        pl.lit(0).alias("clamped_events"),
        pl.lit(5.0).alias("max_radial"),
    )
    sparse = events.filter(pl.col("type") == "escape")
    monkeypatch.setattr(session, "load_histories", lambda *args, **kwargs: events)
    monkeypatch.setattr(session, "MAX_COLLISIONS", 1)
    with serve_explorer(monkeypatch, (overview, sparse, pathways, regions)) as url:
        page.set_viewport_size({"width": 1280, "height": 900})
        page.goto(url)
        expect(page.get_by_label("Collision", exact=True)).to_be_disabled()
        page.get_by_label("X coordinate").select_option(label="Axial distance (mm)")
        page.wait_for_function("""() => {
            const plots = Bokeh.documents[0].get_model_by_name('explorer_plots');
            const plot = plots.children.find(p => p.name === 'events');
            const source = plot?.renderers.find(r => r.data_source?.data.id?.includes(6))?.data_source;
            if (!plot?.inner_width || source?.data.x[0] !== 5000) return false;
            source.selected.indices = [0];
            return true;
        }""")

        def check_plot(temporary_x, highlight_x):
            page.wait_for_function(
                """([temporaryX, highlightX]) => {
                    const plots = Bokeh.documents[0].get_model_by_name('explorer_plots');
                    const plot = plots.children.find(p => p.name === 'events');
                    const temporary = plot.renderers.find(
                        r => r.data_source?.name === 'selected_hidden_collision'
                    )?.data_source.data;
                    const highlight = plot.renderers.find(
                        r => r.glyph?.size?.value === 20
                    ).data_source.data;
                    const plotted = plot.renderers.flatMap(r => r.data_source?.data.id ?? []);
                    return plotted.length === 1 && plotted[0] === 6 &&
                        temporary?.x.length === (temporaryX === null ? 0 : 1) &&
                        (temporaryX === null || (temporary.x[0] === temporaryX && temporary.y[0] === 1)) &&
                        highlight.position.length === (highlightX === null ? 0 : 1) &&
                        (highlightX === null || (highlight.position[0] === highlightX && highlight.plot_y[0] === 1));
                }""",
                arg=[temporary_x, highlight_x],
                timeout=5000,
            )

        # The paired after-state shares the before-state's location on the plot.
        page.get_by_role("button", name="Inspect event #2", exact=True).click()
        expect(page.get_by_text("Selected hidden event #1")).to_be_visible()
        check_plot(1000, 1000)
        page.locator("#event-3 td").nth(1).click()
        expect(page.get_by_text("Selected hidden event #3")).to_be_visible()
        check_plot(2000, 2000)
        page.get_by_role("button", name="Inspect event #6", exact=True).click()
        expect(page.get_by_text("Selected hidden event")).to_have_count(0)
        check_plot(None, 5000)
        page.get_by_role("button", name="Inspect event #1", exact=True).click()
        check_plot(1000, 1000)
        page.get_by_role("button", name="Clear selection", exact=True).click()
        expect(page.get_by_text("Selected hidden event")).to_have_count(0)
        check_plot(None, None)
