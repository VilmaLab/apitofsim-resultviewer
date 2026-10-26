"""Rate views use cached curves and share the Explorer's event selection."""

from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
from apitofsim.api import Histogram, Q_
from apitofsim.workflow.db import PathwayRateCurve
from bokeh.document import Document
from bokeh.models import (
    CheckboxGroup,
    Div,
    GroupBox,
    LinearScale,
    LogScale,
    Range1d,
    Select,
    Span,
)

from apitofresview.plotting.explorer import build_document, rates, session
from apitofresview.plotting.explorer.details import render_choices, render_details

from .helpers import cohort, event


class CachedCurve(SimpleNamespace):
    rate_at = PathwayRateCurve.rate_at


class RateDatabase:
    def __init__(self):
        self.reads = []

    def pathway_rate_curves(self, run, parent, *, result_id, all_pathways):
        self.reads.append((run, parent, result_id, all_pathways))
        return [
            CachedCurve(
                pathway_id=id,
                bonding_energy=Q_(bond, "J"),
                histogram=Histogram(
                    Q_(np.array([1, 2, 3]) * 1e-20, "J"), np.array([0.0, 10.0, 100.0])
                ),
            )
            for id, bond in [(7, 1e-20), (8, 8e-20)]
        ]

    def collision_profile_inputs(self, **kwargs):
        return {}, object()


def rate_frames():
    members, events, pathways, regions = cohort(
        [1, 2],
        event(1, "collision", 1, 0, 0),
        event(2, "fragmentation", 1, 1, 2, 7),
        event(3, "escape", 2, 1, 5),
    )
    events = events.with_columns(
        pl.Series("rot_energy", [1e-20, 2e-20, None]),
        pl.Series("vibrational_energy", [2e-20, 3e-20, None]),
        pl.Series(
            "velocity", [dict(x=3.0, y=4.0, z=5.0), dict(x=6.0, y=7.0, z=8.0), None]
        ),
    )
    pathways = pl.concat(
        [pathways, pl.DataFrame({"pathway_id": [8], "pathway": ["Parent → C + D"]})]
    )
    return members, events, pathways, regions


def check(doc, label):
    return next(w for w in doc.select({"type": CheckboxGroup}) if w.labels == [label])


def select(doc, id):
    plot = doc.select_one({"name": "events"})
    source = next(
        r.data_source
        for r in plot.renderers
        if hasattr(r, "data_source") and id in r.data_source.data.get("id", [])
    )
    source.selected.indices = [source.data["id"].index(id)]


@pytest.fixture
def rate_doc(monkeypatch):
    monkeypatch.setattr(session, "load_data", lambda *args: rate_frames())
    db = RateDatabase()
    doc = Document()
    build_document(db, doc, 1, 1)
    return doc, db


def test_energy_summary_and_missing():
    members, events, _, _ = rate_frames()
    members = members.with_row_index("realization_number", offset=1)
    for render in (
        lambda summary: render_details(members, events, 1, summarise_energy=summary),
        lambda summary: render_choices(
            events.with_columns(pl.lit(1).alias("realization_number")),
            summarise_energy=summary,
        ),
    ):
        html = render(True)
        assert "Internal energy (J)" in html and "Rotational energy" not in html
        assert "<td>3e-20</td>" in html and "<td>—</td>" in html
        html = render(False)
        assert "Rotational energy (J)" in html and "Vibrational energy (J)" in html
        assert "Internal energy" not in html


def test_rate_selection_modes_and_layout(rate_doc):
    doc, db = rate_doc
    show = check(doc, "Show rate constants")
    assert not show.visible and not show.active and not db.reads
    main = doc.select_one({"name": "events"})
    main.x_range.start, main.x_range.end = 1, 2
    select(doc, 1)
    assert show.visible and not show.active and not db.reads
    show.active = [0]
    assert doc.select_one({"name": "details_layout_right"}).active
    chart = doc.select_one({"name": "pathway_rates"})
    assert isinstance(chart.y_scale, LogScale)
    assert np.isnan(chart.renderers[0].data_source.data["y"][0])
    marker = chart.select_one({"type": Span})
    assert marker.location == pytest.approx(3e-20, abs=1e-32)
    assert chart.title.text == "Fragmentation rate"
    assert chart.xaxis.axis_label == "Internal energy (J)"
    assert chart.legend[0] in chart.below
    assert chart.legend[0] not in chart.center
    np.testing.assert_allclose(
        chart.renderers[0].data_source.data["x"], np.array([2, 3, 4]) * 1e-20
    )
    color = chart.renderers[0].glyph.line_color
    check(doc, "Show all pathways").active = [0]
    chart = doc.select_one({"name": "pathway_rates"})
    assert chart.renderers[0].glyph.line_color == color
    assert sorted(m.location for m in chart.select({"type": Span})) == pytest.approx(
        [3e-20], abs=1e-32
    )
    check(doc, "Share plot").active = []
    charts = list(doc.select({"name": "pathway_rates"}))
    assert len(charts) == 2
    assert charts[0].x_range is charts[1].x_range
    assert charts[0].y_range is charts[1].y_range
    check(doc, "Logarithmic rates").active = []
    assert all(
        isinstance(p.y_scale, LinearScale)
        for p in doc.select({"name": "pathway_rates"})
    )
    bridge = doc.roots[0].children[1]
    bridge.text = "2"
    assert sorted(
        m.location
        for p in doc.select({"name": "pathway_rates"})
        for m in p.select({"type": Span})
    ) == pytest.approx([5e-20, 5e-20], abs=1e-32)
    check(doc, "Summarise energy").active = []
    assert len(db.reads) == 1
    assert doc.select_one({"name": "events"}) is main
    assert (main.x_range.start, main.x_range.end) == (1, 2)
    doc.select_one({"name": "details_layout_bottom"}).active = True
    assert show.active
    select(doc, 3)
    assert show.active
    assert all(
        not m.visible
        for p in doc.select({"name": "pathway_rates"})
        for m in p.select({"type": Span})
    )
    doc.select_one({"name": "details_layout_sidebar"}).active = True
    assert not show.active


def test_collision_velocity_scales_and_coordinates(rate_doc, monkeypatch):
    doc, db = rate_doc
    calls = []

    def frequency(z, velocity, **kwargs):
        calls.append((z.copy(), velocity))
        return Q_(np.full(len(z), sum(velocity) + 1.0), "1/s")

    prepared = []
    monkeypatch.setattr(rates, "skimmer", lambda config: prepared.append(config))
    monkeypatch.setattr(
        rates,
        "collision_profile",
        lambda *args, **kwargs: SimpleNamespace(frequency=frequency),
    )
    main = doc.select_one({"name": "events"})
    main.x_range.start, main.x_range.end = 1, 2
    check(doc, "Gas collision rate").active = [0]
    chart = doc.select_one({"name": "collision_rates"})
    assert chart.x_range is main.x_range
    assert isinstance(chart.y_range, Range1d)
    assert calls[-1][1] == [0, 0, 0]
    assert doc.select_one({"name": "collision_velocity"}).text == "v = 0 m/s"
    assert not doc.select_one({"name": "collision_event_marker"}).visible
    assert not doc.select_one({"name": "collision_event_readout"}).visible
    assert isinstance(chart.y_scale, LogScale)
    select(doc, 1)
    assert calls[-1][1] == [3, 4, 5]
    assert doc.select_one({"name": "collision_event_marker"}).visible
    assert doc.select_one({"name": "collision_event_marker"}).location == 0
    readout = doc.select_one({"name": "collision_event_readout"})
    assert readout.visible and readout.y == 13 and readout.text == "13 s⁻¹"
    check(doc, "Show rate constants").active = [0]
    fragmentation = doc.select_one({"name": "fragmentation_readout"})
    assert fragmentation.visible and fragmentation.y == pytest.approx(10)
    assert fragmentation.x == pytest.approx(3e-20, abs=1e-32)
    assert fragmentation.text == "10 s⁻¹"
    show_values = check(doc, "Show readoff values")
    assert show_values.active == [0]
    show_values.active = []
    assert not fragmentation.visible
    assert not doc.select_one({"name": "collision_event_readout"}).visible
    assert doc.select_one({"name": "collision_event_marker"}).visible
    assert doc.select_one({"name": "pathway_rates"}).select_one({"type": Span}).visible
    show_values.active = [0]
    assert fragmentation.visible
    assert doc.select_one({"name": "collision_event_readout"}).visible
    doc.roots[0].children[1].text = "2"
    assert not fragmentation.visible  # Above the cached rate curve.
    assert calls[-1][1] == [6, 7, 8]
    bounds = (chart.y_range.start, chart.y_range.end)
    doc.roots[0].children[1].text = "2"
    assert (chart.y_range.start, chart.y_range.end) == bounds
    assert (
        doc.select_one({"name": "collision_velocity"}).text
        == f"v = {np.linalg.norm([6, 7, 8]):.6g} m/s"
    )
    check(doc, "Logarithmic rates").active = []
    assert isinstance(doc.select_one({"name": "collision_rates"}).y_scale, LinearScale)
    assert isinstance(doc.select_one({"name": "pathway_rates"}).y_scale, LinearScale)
    assert doc.select_one({"name": "events"}) is main
    assert (main.x_range.start, main.x_range.end) == (1, 2)
    select(doc, 3)
    source = doc.select_one({"name": "collision_rates_source"})
    assert len(source.data["rate"]) == 0
    assert any("no velocity" in d.text for d in doc.select({"type": Div}))
    doc.roots[0].children[1].text = "clear"
    assert calls[-1][1] == [0, 0, 0]
    mode = next(w for w in doc.select({"type": Select}) if w.title == "X coordinate")
    mode.value = "physical"
    source = doc.select_one({"name": "collision_rates_source"})
    np.testing.assert_allclose(source.data["x"], calls[-1][0] * 1000)
    mode.value = "time"
    assert check(doc, "Gas collision rate").disabled
    assert check(doc, "Gas collision rate").active
    mode.value = "schematic"
    assert not check(doc, "Gas collision rate").disabled
    assert len(prepared) == 1


def test_collision_exact_readout_and_facet_spacing(monkeypatch):
    members, events, pathways, regions = rate_frames()
    position = 2.12345
    events = events.with_columns(
        pl.when(pl.col("id") == 2).then(position).otherwise(pl.col("z")).alias("z")
    )
    monkeypatch.setattr(
        session, "load_data", lambda *args: (members, events, pathways, regions)
    )
    doc = Document()
    build_document(RateDatabase(), doc, 1, 1)
    monkeypatch.setattr(rates, "skimmer", lambda config: None)
    monkeypatch.setattr(
        rates,
        "collision_profile",
        lambda *args, **kwargs: SimpleNamespace(
            frequency=lambda z, velocity, **kwargs: Q_(z**2 + 1, "1/s"),
        ),
    )
    check(doc, "Gas collision rate").active = [0]
    chart = doc.select_one({"name": "collision_rates"})
    main = doc.select_one({"name": "events"})
    assert chart.margin == main.margin == 0
    select(doc, 1)
    doc.roots[0].children[1].text = "2"
    readout = doc.select_one({"name": "collision_event_readout"})
    assert readout.y == pytest.approx(position**2 + 1)
    assert readout.text == f"{position**2 + 1:.4g} s⁻¹"
    source = doc.select_one({"name": "collision_rates_source"})
    assert source.data["rate"][list(source.data["x"]).index(readout.x)] == readout.y
    check(doc, "Facet pathways").active = [0]
    assert doc.select_one({"name": "collision_rates"}).margin is None
    assert all(p.margin is None for p in doc.select({"name": "events"}))
    doc.roots[0].children[1].text = "clear"
    assert not doc.select_one({"name": "collision_event_marker"}).visible
    assert not doc.select_one({"name": "collision_event_readout"}).visible


def test_bottom_layout_and_rate_controls(rate_doc):
    doc, _ = rate_doc
    sidebar_log = doc.select_one({"name": "log_rates"})
    detail_log = doc.select_one({"name": "detail_log_rates"})
    assert detail_log.disabled
    views = next(
        group for group in doc.select({"type": GroupBox}) if group.title == "Views"
    )
    assert views.child.children[0] is check(doc, "Gas collision rate")
    select(doc, 1)
    doc.select_one({"name": "details_layout_bottom"}).active = True
    check(doc, "Show rate constants").active = [0]
    assert doc.select_one({"name": "details_layout_bottom"}).active
    assert not detail_log.disabled
    assert sidebar_log.active == detail_log.active == [0]
    detail_log.active = []
    assert sidebar_log.active == detail_log.active == []
    sidebar_log.active = [0]
    assert sidebar_log.active == detail_log.active == [0]
    check(doc, "Show rate constants").active = []
    assert detail_log.disabled


def test_chooser_hides_rates_until_realization_selected(rate_doc):
    doc, _ = rate_doc
    select(doc, 1)
    check(doc, "Show rate constants").active = [0]
    source = next(
        r.data_source
        for r in doc.select_one({"name": "events"}).renderers
        if hasattr(r, "data_source") and 1 in r.data_source.data.get("id", [])
    )
    source.selected.indices = [source.data["id"].index(1), source.data["id"].index(2)]
    assert not check(doc, "Show rate constants").visible
    assert check(doc, "Show rate constants").active
    doc.roots[0].children[1].text = "choose:1:2"
    assert check(doc, "Show rate constants").visible
    assert check(doc, "Show rate constants").active


def test_unavailable_inputs_stay_local(rate_doc, monkeypatch):
    doc, db = rate_doc

    def unavailable(*args, **kwargs):
        raise ValueError("Missing provenance")

    monkeypatch.setattr(db, "pathway_rate_curves", unavailable)
    monkeypatch.setattr(db, "collision_profile_inputs", unavailable)
    select(doc, 1)
    check(doc, "Show rate constants").active = [0]
    check(doc, "Gas collision rate").active = [0]
    assert any(
        "Rate curves unavailable: Missing provenance" in d.text
        for d in doc.select({"type": Div})
    )
    assert any(
        "Collision rates unavailable: Missing provenance" in d.text
        for d in doc.select({"type": Div})
    )
    assert doc.select_one({"name": "events"}) is not None


def test_zero_and_missing_curves(rate_doc, monkeypatch):
    doc, db = rate_doc
    curves = db.pathway_rate_curves(1, 1, result_id=10, all_pathways=True)
    curves[0].histogram.y[:] = 0
    curves[1].histogram = None
    monkeypatch.setattr(db, "pathway_rate_curves", lambda *args, **kwargs: curves)
    select(doc, 1)
    check(doc, "Show rate constants").active = [0]
    assert any("No positive rates" in d.text for d in doc.select({"type": Div}))
    check(doc, "Show all pathways").active = [0]
    assert any(
        "Rate curve unavailable: Parent → C + D" in d.text
        for d in doc.select({"type": Div})
    )
    check(doc, "Logarithmic rates").active = []
    assert not any("No positive rates" in d.text for d in doc.select({"type": Div}))
    assert list(
        doc.select_one({"name": "pathway_rates"}).renderers[0].data_source.data["y"]
    ) == [0, 0, 0]


def test_shared_log_rates_use_largest_curve_minimum(rate_doc, monkeypatch):
    doc, db = rate_doc
    curves = db.pathway_rate_curves(1, 1, result_id=10, all_pathways=True)
    curves[0].histogram.y = np.array([0.0, 1e-6, 100.0])
    curves[1].histogram.y = np.array([0.0, 10.0, 200.0])
    monkeypatch.setattr(db, "pathway_rate_curves", lambda *args, **kwargs: curves)
    select(doc, 1)
    check(doc, "Show all pathways").active = [0]
    check(doc, "Show rate constants").active = [0]
    chart = doc.select_one({"name": "pathway_rates"})
    assert chart.y_range.start == pytest.approx(10 / 1.2)
    assert chart.y_range.end == pytest.approx(200 * 1.2)
    check(doc, "Share plot").active = []
    assert all(
        p.y_range.start == pytest.approx(1e-6 / 1.2)
        for p in doc.select({"name": "pathway_rates"})
    )
    check(doc, "Logarithmic rates").active = []
    assert all(p.y_range.start == 0 for p in doc.select({"name": "pathway_rates"}))


def test_fragmentation_readouts_thresholds_and_coincident_values(rate_doc, monkeypatch):
    doc, db = rate_doc
    curves = db.pathway_rate_curves(1, 1, result_id=10, all_pathways=True)
    monkeypatch.setattr(db, "pathway_rate_curves", lambda *args, **kwargs: curves)
    select(doc, 1)
    check(doc, "Show all pathways").active = [0]
    check(doc, "Show rate constants").active = [0]
    labels = list(doc.select({"name": "fragmentation_readout"}))
    assert sum(label.visible for label in labels) == 1
    check(doc, "Logarithmic rates").active = []
    assert sorted(
        label.y
        for label in doc.select({"name": "fragmentation_readout"})
        if label.visible
    ) == pytest.approx([0, 10])
    curves[1].bonding_energy = curves[0].bonding_energy
    check(doc, "Logarithmic rates").active = [0]
    labels = list(doc.select({"name": "fragmentation_readout"}))
    assert all(label.visible for label in labels)
    assert len({label.y_offset for label in labels}) == 2
    check(doc, "Share plot").active = []
    assert all(label.visible for label in doc.select({"name": "fragmentation_readout"}))
    select(doc, 3)
    assert not any(
        label.visible for label in doc.select({"name": "fragmentation_readout"})
    )
