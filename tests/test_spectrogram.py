"""Peak labels render and follow the page's shared spectrogram setting."""

from math import sqrt

import holoviews
import pandas as pd
import pytest
from bokeh.document import Document
from bokeh.embed import file_html
from bokeh.models import GlyphRenderer, Text
from bokeh.resources import INLINE
from playwright.sync_api import expect

from apitofresview.plotting.spectrogram import bokeh_labels, matplotlib_spectrogram


@pytest.fixture(scope="module", autouse=True)
def headless_matplotlib():
    import matplotlib

    matplotlib.use("Agg")


@pytest.mark.parametrize("backend", ["matplotlib", "bokeh"])
@pytest.mark.parametrize("labels", [True, False])
def test_peak_labels(backend, labels):
    renderer = holoviews.renderer(backend)
    df = pd.DataFrame(
        {
            "atomic_mass": [40.0, 60.0, 100.0],
            "intensity": [0.4, 0.0, 0.6],
            "product_name": ["Product A", "Neutral", "Parent"],
        }
    )
    if backend == "matplotlib":
        figure = matplotlib_spectrogram(df, labels=labels, scale="sum")
        names = [text.get_text() for text in figure.axes[0].texts]
    else:
        from apitofsim.plotting.spectrogram import basic_spectrogram

        plot = renderer.get_plot(basic_spectrogram(df, scale="sum"))
        if labels:
            bokeh_labels(plot.state, df)
        names = [
            name
            for item in plot.state.renderers
            if isinstance(item, GlyphRenderer) and isinstance(item.glyph, Text)
            for name in item.data_source.data[item.glyph.text]
        ]
    assert names == (["Product A", "Parent"] if labels else [])


def test_label_checkbox_controls_both_views(page, live_server):
    page.goto(f"{live_server}/experiment/cluster/spectrogram?experiment=1&cluster=1")
    checkbox = page.get_by_role("checkbox", name="Show labels")
    expect(checkbox).to_be_checked()
    page.get_by_role("link", name="Bokeh", exact=True).click()

    def expect_bokeh_labels(visible):
        page.wait_for_function(
            """visible => window.Bokeh?.documents.some(doc => {
                const labels = [...doc.all_models.values()].filter(
                    model => model.type === 'GlyphRenderer' && model.glyph.type === 'Text'
                );
                return visible ? labels.some(model =>
                    model.data_source.data[model.glyph.text.field].includes('Parent')
                ) : labels.length === 0;
            })""",
            arg=visible,
            timeout=30_000,
        )

    expect_bokeh_labels(True)
    page.wait_for_function(
        """() => window.Bokeh.documents.some(doc =>
            [...doc.all_models.values()].some(model =>
                model.type === 'Figure' && model.y_range.end > 1
            )
        )"""
    )
    checkbox.uncheck()
    expect(page).to_have_url(
        f"{live_server}/experiment/cluster/spectrogram?experiment=1&cluster=1&labels=false"
    )
    expect(checkbox).not_to_be_checked()
    expect_bokeh_labels(False)
    checkbox.check()
    expect(page).to_have_url(
        f"{live_server}/experiment/cluster/spectrogram?experiment=1&cluster=1&labels=true"
    )
    expect(checkbox).to_be_checked()
    expect_bokeh_labels(True)


@pytest.mark.parametrize("labels", [True, False])
def test_matplotlib_handles_zero_intensities(labels):
    figure = matplotlib_spectrogram(
        pd.DataFrame(
            {"atomic_mass": [100.0], "intensity": [0.0], "product_name": ["Parent"]}
        ),
        labels=labels,
    )
    assert not figure.axes[0].texts


def test_matplotlib_uses_upstream_layout(monkeypatch, tmp_path):
    from apitofsim.plotting import spectrogram

    original = spectrogram.relayout_labels
    calls = []

    def layout(*args):
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(spectrogram, "relayout_labels", layout)
    figure = matplotlib_spectrogram(
        pd.DataFrame(
            {
                "atomic_mass": [99.0, 100.0, 101.0],
                "intensity": [0.5, 0.5, 0.5],
                "product_name": ["Long product A", "Long product B", "Long product C"],
            }
        )
    )
    assert len(calls) == 1
    assert len(figure.axes[0].collections[-1].get_segments()) == 3
    assert any(text.get_position()[1] > 0.5 for text in figure.axes[0].texts)
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    renderer = canvas.get_renderer()
    boxes = []
    for text in figure.axes[0].texts:
        x, y = text.get_transform().transform(text.get_position())
        rx, ry = (x + y) / sqrt(2), (y - x) / sqrt(2)
        width, height, _ = renderer.get_text_width_height_descent(
            text.get_text(), text.get_fontproperties(), ismath=False
        )
        box = (rx, ry, rx + width, ry + height)
        assert not any(
            box[0] < other[2]
            and box[2] > other[0]
            and box[1] < other[3]
            and box[3] > other[1]
            for other in boxes
        )
        boxes.append(box)
    figure.savefig(tmp_path / "matplotlib.png", bbox_inches="tight")


def test_bokeh_dense_labels_relayout(page, tmp_path):
    from bokeh.layouts import layout
    from bokeh.plotting import figure

    doc = Document()
    plot = figure(width=800, height=400, x_range=(90, 120), y_range=(0, 1.15))
    frame = pd.DataFrame(
        {
            "atomic_mass": [100 + i * 0.1 for i in range(12)],
            "intensity": [0.5 + 0.03 * (i % 3) for i in range(12)],
            "product_name": [f"Long product name {i}" for i in range(12)],
        }
    )
    plot.segment(x0=frame.atomic_mass, y0=0, x1=frame.atomic_mass, y1=frame.intensity)
    bokeh_labels(plot, frame)
    doc.add_root(layout([[plot]], sizing_mode="fixed"))
    path = tmp_path / "spectrogram.html"
    path.write_text(file_html(doc, INLINE, "Dense spectrogram"))
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(path.as_uri())
    check = r"""() => {
        const doc = Bokeh.documents[0];
        const source = doc.get_model_by_name('spectrogram_labels');
        const plot = [...doc.all_models.values()].find(m => m.type === 'Figure');
        if (!source || !(plot.inner_width > 0)) return false;
        const d = source.data;
        const ctx = document.createElement('canvas').getContext('2d');
        ctx.font = '10pt sans-serif';
        const boxes = [];
        for (let i = 0; i < d.x.length; i++) {
            if (!Number.isFinite(d.x[i] + d.y[i])) continue;
            const x = (d.x[i] - plot.x_range.start) * plot.inner_width /
                (plot.x_range.end - plot.x_range.start);
            const y = (d.y[i] - plot.y_range.start) * plot.inner_height /
                (plot.y_range.end - plot.y_range.start);
            const metrics = ctx.measureText(d.text[i]);
            const rx = (x + y) * Math.SQRT1_2;
            const ry = (y - x) * Math.SQRT1_2;
            const box = [rx, ry, rx + metrics.width,
                ry + metrics.actualBoundingBoxAscent + metrics.actualBoundingBoxDescent];
            if (boxes.some(b => box[0] < b[2] && box[2] > b[0] &&
                box[1] < b[3] && box[3] > b[1])) return false;
            const textRight = (box[2] - box[1]) * Math.SQRT1_2;
            const textTop = (box[2] + box[3]) * Math.SQRT1_2;
            if (textRight > plot.inner_width + plot.min_border_right ||
                textTop > plot.inner_height + plot.min_border_top) return false;
            boxes.push(box);
        }
        return boxes.length === 12 &&
            doc.get_model_by_name('spectrogram_label_leaders').data.xs.length === 12;
    }"""
    page.wait_for_function(check)
    before = page.evaluate(
        "Bokeh.documents[0].get_model_by_name('spectrogram_labels').data.y.slice()"
    )
    page.evaluate("""() => {
        const plot = [...Bokeh.documents[0].all_models.values()].find(m => m.type === 'Figure');
        plot.x_range.start = 99;
        plot.x_range.end = 104;
        plot.y_range.start = .1;
        plot.y_range.end = .9;
        plot.width = 600;
    }""")
    page.wait_for_function(
        """before => {
        const after = Bokeh.documents[0].get_model_by_name('spectrogram_labels').data.y;
        return after.some((y, i) => Math.abs(y - before[i]) > .001);
    }""",
        arg=before,
    )
    page.wait_for_function(check)
    page.evaluate(
        "() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))"
    )
    page.screenshot(path=str(tmp_path / "spectrogram.png"), full_page=True)
    page.evaluate("""() => {
        const plot = [...Bokeh.documents[0].all_models.values()].find(m => m.type === 'Figure');
        plot.x_range.start = 100.5;
        plot.x_range.end = 101.1;
    }""")
    page.wait_for_function("""() => {
        const doc = Bokeh.documents[0];
        return doc.get_model_by_name('spectrogram_labels').data.x.filter(Number.isFinite).length === 7 &&
            doc.get_model_by_name('spectrogram_label_leaders').data.xs.length === 7;
    }""")
    assert errors == []
