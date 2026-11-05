"""Spectrogram labels using apitofsim's right-to-left placement strategy."""

from importlib.resources import files
from math import pi


def matplotlib_spectrogram(df, *, labels=True, scale=None):
    import holoviews
    from apitofsim.plotting.spectrogram import basic_spectrogram, relayout_labels

    holoviews.extension("matplotlib")
    plot = basic_spectrogram(df, scale=scale)
    peaks = df[df["intensity"] > 0]
    fig_inches, aspect = (6, 3), 2
    xlim = (0, max(1, float(df["atomic_mass"].max()) * 1.1))
    ylim = (0, max(1.05, float(df["intensity"].max()) * 1.05))

    # Upstream measures artists before HoloViews finalizes the overlay ranges.
    # Initialize both draft and final axes identically before those hooks run.
    def fix_limits(rendered, element):
        rendered.handles["axis"].set(xlim=xlim, ylim=ylim)

    plot = plot.opts(xlim=xlim, ylim=ylim, initial_hooks=[fix_limits])
    if labels and not peaks.empty:
        names = holoviews.Labels(
            (peaks["atomic_mass"], peaks["intensity"], peaks["product_name"]),
            plot.kdims + plot.vdims,
            "Product",
        ).opts(xlim=xlim, ylim=ylim, initial_hooks=[fix_limits])
        hook = relayout_labels(plot, names, fig_inches, aspect)
        plot *= names.opts(color="black", size=10, rotation=45, hooks=[hook])
    return holoviews.render(plot.opts(fig_inches=fig_inches, aspect=aspect))


def bokeh_labels(figure, df):
    from bokeh.events import DocumentReady
    from bokeh.models import ColumnDataSource, CustomJS

    peaks = df[df["intensity"] > 0]
    if peaks.empty:
        return
    source = ColumnDataSource(
        dict(
            mass=peaks["atomic_mass"].tolist(),
            intensity=peaks["intensity"].tolist(),
            text=peaks["product_name"].tolist(),
            x=peaks["atomic_mass"].tolist(),
            y=peaks["intensity"].tolist(),
        ),
        name="spectrogram_labels",
    )
    leaders = ColumnDataSource(dict(xs=[], ys=[]), name="spectrogram_label_leaders")
    figure.multi_line(
        xs="xs",
        ys="ys",
        source=leaders,
        line_color="gray",
        line_width=0.5,
        line_dash="dashed",
        level="annotation",
    )
    # Annotation level permits labels in the extra border reserved by the layout.
    figure.text(
        x="x",
        y="y",
        text="text",
        source=source,
        angle=pi / 4,
        text_align="left",
        text_baseline="bottom",
        text_font_size="10pt",
        text_font="sans-serif",
        text_color="black",
        level="annotation",
    )
    callback = CustomJS(
        args=dict(plot=figure, source=source, leaders=leaders),
        code=(files("apitofresview.plotting") / "spectrogram_labels.js").read_text(),
    )
    figure.js_on_event(DocumentReady, callback)
    for name in ("inner_width", "inner_height"):
        figure.js_on_change(name, callback)
    for axis in (figure.x_range, figure.y_range):
        for name in ("start", "end"):
            axis.js_on_change(name, callback)
