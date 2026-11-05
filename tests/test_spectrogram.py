"""Peak labels render and follow the page's shared spectrogram setting."""

import holoviews
import pandas as pd
import pytest
from bokeh.models import GlyphRenderer, Text
from playwright.sync_api import expect

from apitofresview.webapp import spectrogram_plot


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
    plot = renderer.get_plot(spectrogram_plot(df, labels=labels, scale="sum"))
    if backend == "matplotlib":
        names = [text.get_text() for text in plot.state.axes[0].texts]
    else:
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
