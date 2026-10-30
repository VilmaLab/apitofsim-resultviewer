"""Independent pathway cohorts retain their own outcomes and denominators."""

import pytest
from bokeh.document import Document
from bokeh.models import CheckboxGroup, Div, HoverTool, RangeSlider, Select

from apitofresview.plotting.explorer import build_document, session
from apitofresview.plotting.explorer.plot import EVENT_TOOLTIP, _pathway_colors

from .helpers import pathway_cohort
from .test_rates import RateDatabase


def checkbox(doc, label):
    return next(w for w in doc.select({"type": CheckboxGroup}) if w.labels == [label])


def charts(doc, name):
    return [
        p for p in doc.select_one({"name": "explorer_plots"}).children if p.name == name
    ]


def points(chart):
    return next(r.data_source for r in chart.renderers if "id" in r.data_source.data)


@pytest.fixture
def pathway_doc(monkeypatch):
    monkeypatch.setattr(session, "load_data", lambda *args: pathway_cohort())
    doc = Document()
    build_document(None, doc, 2, 1)
    return doc


def test_pathway_facets_ranges_and_identity(pathway_doc):
    doc = pathway_doc
    assert not checkbox(doc, "Facet pathways").visible
    assert not checkbox(doc, "Group fragmentations").visible
    events = charts(doc, "events")
    assert len(events) == 3  # Excludes a pathway without a recorded result.
    assert points(events[0]).data["realization_id"] == [100, 200, 300, 400, 400]
    assert points(events[1]).data["realization_id"] == [500]
    assert points(events[1]).data["realization"] == [1]
    assert "4 shown / 4 in range" in events[0].title.text
    assert events[0].y_range is events[1].y_range is events[2].y_range
    source = points(events[1])
    source.selected.indices = [0]
    assert any(
        "<h3>Details</h3>" in d.text and "Parent → C + D" in d.text
        for d in doc.select({"type": Div})
    )
    assert not any("Database realization" in d.text for d in doc.select({"type": Div}))
    assert not any("Parent → Parent" in d.text for d in doc.select({"type": Div}))
    checkbox(doc, "Realizations pager").active = [0]
    slider = next(iter(doc.select({"type": RangeSlider})))
    assert slider.end == 4
    slider.value = (2, 3)
    events = charts(doc, "events")
    assert points(events[0]).data["realization_id"] == [200, 300]
    assert "0 shown / 0 in range" in events[1].title.text
    assert len(events) == 3
    assert events[0].y_range.start == 1.5 and events[0].y_range.end == 3.5
    assert not any("<h3>Details</h3>" in d.text for d in doc.select({"type": Div}))


def test_pathway_probabilities_colors_and_simple_presentation(pathway_doc):
    doc = pathway_doc
    assert doc.select_one({"name": "outcomes"}) is None
    assert not any(
        d.visible and ("shown /" in d.text or "Fractions use" in d.text)
        for d in doc.select({"type": Div})
    )
    checkbox(doc, "Cumulative").active = [0]
    checkbox(doc, "Bar chart").active = [0]
    events = charts(doc, "events")
    assert points(events[0]).data["id"] == [1, 2, 3, 4, 5]
    assert all(
        h.tooltips == EVENT_TOOLTIP
        for p in events
        for h in p.select({"type": HoverTool})
    )
    areas = charts(doc, "cdf")[0].renderers
    assert [r.data_source.data["fraction"][-1] for r in areas] == [0.25, 0.25]
    colors = _pathway_colors(
        ["Parent → A + B", "Parent → C + D", "Parent → E + F"], "Parent → Parent"
    )
    assert areas[0].glyph.fill_color == colors["Parent → A + B"]
    assert points(events[0]).data["color"][0] == areas[0].glyph.fill_color
    bars = charts(doc, "bars")
    assert all(p.y_range.start == 0 and p.y_range.end == 1 for p in bars)
    assert max(r.glyph.top for r in bars[0].renderers) == 0.25
    assert charts(doc, "cdf")[1].renderers[0].data_source.data["fraction"][-1] == 1
    assert not charts(doc, "cdf")[2].renderers
    checkbox(doc, "Realizations pager").active = [0]
    next(iter(doc.select({"type": RangeSlider}))).value = (1, 2)
    assert [
        r.data_source.data["fraction"][-1] for r in charts(doc, "cdf")[0].renderers
    ] == [0.5, 0.5]
    layout = next(w for w in doc.select({"type": Select}) if w.title == "Y coordinate")
    layout.value = "radial"
    events = charts(doc, "events")
    assert events[0].y_range is events[1].y_range
    assert events[0].y_range.end > 0


class IndependentRateDatabase(RateDatabase):
    def pathway_rate_curves(self, *args, **kwargs):
        curves = super().pathway_rate_curves(*args, **kwargs)
        for curve in curves:
            curve.participating = curve.pathway_id == (
                7 if kwargs["result_id"] == 10 else 8
            )
        return curves


def test_rates_follow_simulated_pathway_even_when_all_escape(monkeypatch):
    monkeypatch.setattr(session, "load_data", lambda *args: pathway_cohort())
    db = IndependentRateDatabase()
    doc = Document()
    build_document(db, doc, 2, 1)
    checkbox(doc, "Show rate constants").active = [0]
    points(charts(doc, "events")[0]).selected.indices = [0]
    first = list(doc.select({"name": "pathway_rates"}))[0].renderers[0].glyph.line_color
    points(charts(doc, "events")[1]).selected.indices = [0]
    rate = list(doc.select({"name": "pathway_rates"}))[0]
    assert len(rate.renderers) == 1
    assert rate.renderers[0].glyph.line_color != first
    assert db.reads[-1][2] == 11
    checkbox(doc, "Show all pathways").active = [0]
    assert len(list(doc.select({"name": "pathway_rates"}))[0].renderers) == 2


def test_recorded_pathways_without_histories_remain_visible(monkeypatch):
    members, events, pathways, regions = pathway_cohort()
    monkeypatch.setattr(
        session,
        "load_data",
        lambda *args: (members.head(0), events.head(0), pathways, regions),
    )
    doc = Document()
    build_document(None, doc, 2, 1)
    assert next(iter(doc.select({"type": RangeSlider}))).disabled
    assert len(charts(doc, "events")) == 3
    assert all("0 shown / 0 in range" in p.title.text for p in charts(doc, "events"))
