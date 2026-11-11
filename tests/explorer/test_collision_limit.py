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
