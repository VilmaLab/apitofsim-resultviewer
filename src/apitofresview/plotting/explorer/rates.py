"""Read-only rate inputs and selection-linked Bokeh views for one session."""

from html import escape

import numpy as np
import polars as pl
from apitofsim import collision_profile, skimmer
from apitofsim.api import ApiTofRateConstantOverflow, internal_energy
from bokeh.layouts import column
from bokeh.models import (
    BasicTickFormatter,
    BasicTicker,
    ColumnDataSource,
    CustomJS,
    Div,
    Label,
    Range1d,
    Span,
)
from bokeh.plotting import figure

from .controls import enabled, prepare_toolbar
from .layout import position_events
from .plot import COLORS, PATHWAY_COLORS


def message(text):
    return Div(text=f"<p>{escape(text)}</p>", sizing_mode="stretch_width")


def rate_range(values, logarithmic):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0 if logarithmic else values >= 0)]
    if not len(values):
        return Range1d(1, 10) if logarithmic else Range1d(0, 1)
    lo, hi = float(values.min()), float(values.max())
    if logarithmic:
        return Range1d(lo / 1.2, hi * 1.2)
    return Range1d(0, hi * 1.05 if hi else 1)


def responsive_legend(chart, legend):
    """Fit row-major legend columns to the current frame using actual text widths."""
    legend.orientation = "horizontal"
    legend.ncols = 1
    chart.js_on_change(
        "inner_width",
        CustomJS(
            args=dict(chart=chart, legend=legend),
            code="""
        if (!(chart.inner_width > 0)) return;
        const ctx = document.createElement('canvas').getContext('2d');
        ctx.font = `${legend.label_text_font_style} ${legend.label_text_font_size} ${legend.label_text_font}`;
        const widths = legend.items.map(item => legend.glyph_width + legend.label_standoff +
            Math.max(legend.label_width, ctx.measureText(item.label.value).width));
        const available = chart.inner_width - 2 * (legend.padding + legend.margin + 1);
        let columns = widths.length;
        for (; columns > 1; columns--) {
            const sizes = Array(columns).fill(0);
            widths.forEach((width, i) => sizes[i % columns] = Math.max(sizes[i % columns], width));
            if (sizes.reduce((sum, width) => sum + width, 0) + (columns - 1) * legend.spacing <= available)
                break;
        }
        legend.ncols = Math.max(columns, 1);
        chart.height = 260 + 25 * Math.ceil(widths.length / legend.ncols);
    """,
        ),
    )


def readout_label(name, color="#111827"):
    return Label(
        x=0,
        y=0,
        text_font_size="9pt",
        text_color=color,
        background_fill_color="white",
        background_fill_alpha=0.85,
        visible=False,
        name=name,
    )


def place_readout(label, chart, x, y, logarithmic, offset=0):
    right = x > (chart.x_range.start + chart.x_range.end) / 2
    bounds = chart.y_range
    middle = (
        np.sqrt(bounds.start * bounds.end)
        if logarithmic
        else (bounds.start + bounds.end) / 2
    )
    above = y > middle
    label.update(
        x=x,
        y=y,
        text=f"{y:.4g} s⁻¹",
        visible=True,
        text_align="right" if right else "left",
        x_offset=-6 if right else 6,
        y_offset=-(6 + offset) if above else 6 + offset,
        text_baseline="top" if above else "bottom",
    )


class RateViews:
    def __init__(
        self, db, run, cluster, pathways, fate_names, colors, ui, *, pathway_mode=False
    ):
        self.db, self.run, self.cluster, self.ui = db, run, cluster, ui
        self.pathway_mode = pathway_mode
        self.names = dict(pathways.select("pathway_id", "pathway").iter_rows())
        self.default_names = set(fate_names)
        self.colors = dict(colors)
        palette = [
            color for color in PATHWAY_COLORS if color not in colors.values()
        ] or PATHWAY_COLORS
        for index, name in enumerate(
            sorted(set(self.names.values()) - self.colors.keys())
        ):
            self.colors[name] = palette[index % len(palette)]
        self.curve_cache = {}
        self.signature = None
        self.markers = []
        self.readouts = []
        self.energy_range = None
        self.profile = None
        self.profile_error = None
        self.collision = column(Div(), sizing_mode="stretch_width", visible=False)
        self.collision_source = None
        self.collision_plot = None
        self.collision_notice = message("")
        self.event = None

    def curves(self, result):
        if result not in self.curve_cache:
            try:
                self.curve_cache[result] = self.db.pathway_rate_curves(
                    self.run, self.cluster, result_id=result, all_pathways=True
                )
            except (ValueError, RuntimeError, AttributeError) as exc:
                self.curve_cache[result] = str(exc)
        return self.curve_cache[result]

    def update(self, selected, event):
        self.event = event
        self.update_collision()
        if selected.is_empty() or not enabled(self.ui.show_rates):
            return
        result = selected["experiment_result_id"][0]
        signature = (
            result,
            enabled(self.ui.share_rates),
            enabled(self.ui.all_pathways),
            enabled(self.ui.log_rates),
        )
        if signature != self.signature:
            self.signature = signature
            self.build_rates(result)
        energy = internal_energy(
            event.get("rot_energy") if event else None,
            event.get("vibrational_energy") if event else None,
        )
        value = None if energy is None else float(energy.to("J").magnitude)
        for marker in self.markers:
            marker.visible = value is not None and np.isfinite(value)
            if marker.visible:
                marker.location = value
                # Extend rather than reset the user's energy zoom on selection.
                padding = (self.energy_range.end - self.energy_range.start) * 0.03
                if value <= self.energy_range.start:
                    self.energy_range.start = value - padding
                if value >= self.energy_range.end:
                    self.energy_range.end = value + padding
        placed = {}
        for curve, chart, label in self.readouts:
            label.visible = False
            if (
                not enabled(self.ui.show_readoffs)
                or value is None
                or not np.isfinite(value)
            ):
                continue
            try:
                rate = float(
                    curve.rate_at(energy - curve.bonding_energy).to("1/s").magnitude
                )
            except ValueError, ApiTofRateConstantOverflow:
                continue
            if (
                not np.isfinite(rate)
                or not chart.y_range.start <= rate <= chart.y_range.end
            ):
                continue
            # Stack equal-value readouts so coincident pathways remain readable.
            key = (chart.id, rate)
            offset = placed.get(key, 0)
            place_readout(label, chart, value, rate, enabled(self.ui.log_rates), offset)
            placed[key] = offset + 14

    def build_rates(self, result):
        self.markers = []
        self.readouts = []
        curves = self.curves(result)
        if isinstance(curves, str):
            self.ui.rate_plots.children = [
                message(f"Rate curves unavailable: {curves}")
            ]
            return
        curves = [
            curve
            for curve in curves
            if enabled(self.ui.all_pathways)
            or (
                curve.participating
                if self.pathway_mode
                else self.names.get(curve.pathway_id) in self.default_names
            )
        ]
        if not curves:
            self.ui.rate_plots.children = [message("No rate pathways to display.")]
            return
        logarithmic = enabled(self.ui.log_rates)
        shared = enabled(self.ui.share_rates)
        xs, ys, minima = [], [], []
        for curve in curves:
            if curve.histogram is not None:
                xs.extend(self.internal_energies(curve))
                ys.extend(curve.histogram.y)
                positive = np.asarray(curve.histogram.y)
                positive = positive[np.isfinite(positive) & (positive > 0)]
                if len(positive):
                    minima.append(float(positive.min()))
        finite_x = np.asarray(xs, dtype=float)
        finite_x = finite_x[np.isfinite(finite_x)]
        maximum = float(finite_x.max()) if len(finite_x) else 1e-20
        minimum = min(0, float(finite_x.min())) if len(finite_x) else 0
        self.energy_range = Range1d(
            minimum, maximum if maximum > minimum else minimum + 1e-20
        )
        y_range = rate_range(ys, logarithmic)
        if shared and logarithmic and len(minima) > 1:
            y_range.start = max(minima) / 1.2
        plots, notices = [], []
        for curve in curves:
            name = self.names.get(curve.pathway_id, f"Pathway {curve.pathway_id}")
            color = self.colors.get(name, COLORS["fragmentation"])
            if not shared or not plots:
                chart = figure(
                    title="Fragmentation rate" if shared else name,
                    x_range=self.energy_range,
                    y_range=y_range,
                    x_axis_label="Internal energy (J)",
                    y_axis_label="Fragmentation rate (s⁻¹)",
                    y_axis_type="log" if logarithmic else "linear",
                    height=260,
                    sizing_mode="stretch_width",
                    tools="pan,wheel_zoom,box_zoom,reset,save",
                    name="pathway_rates",
                )
                prepare_toolbar(chart)
                chart.xaxis.ticker = BasicTicker(desired_num_ticks=4)
                chart.xaxis.formatter = BasicTickFormatter(precision=1)
                plots.append(chart)
            chart = plots[-1]
            if curve.histogram is None:
                notices.append(message(f"Rate curve unavailable: {name}"))
            else:
                x = self.internal_energies(curve)
                y = np.asarray(curve.histogram.y, dtype=float).copy()
                valid = np.isfinite(y) & (y > 0 if logarithmic else y >= 0)
                y[~valid] = np.nan
                chart.line(
                    x,
                    y,
                    color=color,
                    line_width=2,
                    **({"legend_label": name} if shared else {}),
                )
                label = readout_label("fragmentation_readout", color)
                chart.add_layout(label)
                self.readouts.append((curve, chart, label))
                if not valid.any():
                    notices.append(
                        message(f"No {'positive ' if logarithmic else ''}rates: {name}")
                    )
        for chart in plots:
            marker = Span(
                dimension="height",
                line_color="#111827",
                line_dash="dashed",
                visible=False,
            )
            chart.add_layout(marker)
            self.markers.append(marker)
            if chart.legend:
                legend = chart.legend[0]
                legend.click_policy = "hide"
                legend.label_text_font_size = "9pt"
                chart.center.remove(legend)
                chart.add_layout(legend, "below")
                chart.height += 25 * len(legend.items)
                responsive_legend(chart, legend)
        self.ui.rate_plots.children = [*plots, *notices]

    @staticmethod
    def internal_energies(curve):
        # apitofsim stores k(delta_E); E_internal = delta_E + bonding energy.
        return np.asarray(curve.histogram.x.to("J").magnitude) + float(
            curve.bonding_energy.to("J").magnitude
        )

    def configure_collision(self, mapped_regions, shared_x, spatial):
        self.collision.visible = spatial and enabled(self.ui.collision_rates)
        self.ui.collision_rates.disabled = not spatial
        if not self.collision.visible:
            return
        if self.profile is None and self.profile_error is None:
            try:
                config, cluster = self.db.collision_profile_inputs(
                    run_id=self.run, cluster_id=self.cluster
                )
                self.profile = collision_profile(
                    config, cluster, skimmer=skimmer(config)
                )
            except (ValueError, RuntimeError, AttributeError, KeyError) as exc:
                self.profile_error = str(exc)
        if self.profile_error is not None:
            self.collision.children = [
                message(f"Collision rates unavailable: {self.profile_error}")
            ]
            return
        positions, mapped = [], []
        for lo, hi, left, right in mapped_regions.select(
            "physical_left", "physical_right", "left", "right"
        ).iter_rows():
            if hi <= lo:
                continue
            z = np.linspace(lo, np.nextafter(hi, lo), 128)
            positions.extend(z)
            mapped.extend(left + (z - lo) * (right - left) / (hi - lo))
        self.positions = np.asarray(positions)
        self.mapped = mapped
        self.mapped_regions = mapped_regions
        self.shared_x = shared_x
        self.build_collision()

    def build_collision(self):
        logarithmic = enabled(self.ui.log_rates)
        self.collision_source = ColumnDataSource(
            data=dict(x=[], rate=[]), name="collision_rates_source"
        )
        self.collision_plot = figure(
            name="collision_rates",
            height=170,
            sizing_mode="stretch_width",
            x_range=self.shared_x,
            y_axis_label="Gas collision rate (s⁻¹)",
            y_range=rate_range([], logarithmic),
            y_axis_type="log" if logarithmic else "linear",
            tools="pan,wheel_zoom,reset,save",
        )
        self.collision_plot.line(
            "x",
            "rate",
            source=self.collision_source,
            color=COLORS["collision"],
            line_width=2,
        )
        self.collision_plot.xaxis.visible = False
        self.collision_plot.yaxis.formatter = BasicTickFormatter(precision=1)
        self.collision_plot.yaxis.major_label_text_font_size = "10px"
        self.collision_plot.yaxis.axis_label_text_font_size = "12px"
        self.collision_plot.yaxis.major_label_standoff = 3
        self.velocity_label = Label(
            x=8,
            y=8,
            x_units="screen",
            y_units="screen",
            text_font_size="9pt",
            background_fill_color="white",
            background_fill_alpha=0.85,
            name="collision_velocity",
        )
        self.collision_plot.add_layout(self.velocity_label)
        self.collision_marker = Span(
            dimension="height",
            line_color="#111827",
            line_width=1,
            visible=False,
            name="collision_event_marker",
        )
        self.collision_readout = readout_label("collision_event_readout")
        self.collision_plot.add_layout(self.collision_marker)
        self.collision_plot.add_layout(self.collision_readout)
        prepare_toolbar(self.collision_plot)
        self.collision.children = [self.collision_plot, self.collision_notice]
        self.update_collision()

    def change_scale(self):
        if self.collision.visible and self.profile is not None:
            self.build_collision()

    def update_collision(self):
        if not self.collision.visible or self.collision_source is None:
            return
        velocity = self.event.get("velocity") if self.event else dict(x=0, y=0, z=0)
        self.collision_marker.visible = self.collision_readout.visible = False
        try:
            if velocity is None or any(velocity.get(axis) is None for axis in "xyz"):
                raise ValueError("Selected event has no velocity")
            vector = [velocity[axis] for axis in "xyz"]
            positions, mapped, index = self.collision_positions()
            rates = np.asarray(
                self.profile.frequency(
                    positions,
                    vector,
                    quantities_strict=False,
                )
                .to("1/s")
                .magnitude,
                dtype=float,
            ).copy()
            logarithmic = enabled(self.ui.log_rates)
            valid = np.isfinite(rates) & (rates > 0 if logarithmic else rates >= 0)
            rates[~valid] = np.nan
            self.collision_source.data = dict(x=mapped, rate=rates)
            bounds = rate_range(rates, logarithmic)
            self.collision_plot.y_range.start = bounds.start
            self.collision_plot.y_range.end = bounds.end
            self.collision_plot.y_range.reset_start = bounds.start
            self.collision_plot.y_range.reset_end = bounds.end
            self.velocity_label.text = f"v = {np.linalg.norm(vector):.6g} m/s"
            if index is not None:
                self.collision_marker.location = mapped[index]
                self.collision_marker.visible = True
                if valid[index] and enabled(self.ui.show_readoffs):
                    place_readout(
                        self.collision_readout,
                        self.collision_plot,
                        mapped[index],
                        rates[index],
                        logarithmic,
                    )
            self.collision_notice.text = (
                ""
                if valid.any()
                else "No positive collision rates."
                if logarithmic
                else "No collision rates."
            )
        except (ValueError, RuntimeError) as exc:
            self.collision_source.data = dict(x=[], rate=[])
            self.velocity_label.text = "v unavailable"
            self.collision_notice.text = (
                f"Collision rates unavailable: {escape(str(exc))}"
            )
        self.collision_notice.visible = bool(self.collision_notice.text) or enabled(
            self.ui.fate_facets
        )

    def collision_positions(self):
        """Include the selected physical position in the plotted profile and readout."""
        z = self.event.get("z") if self.event else None
        if (
            z is None
            or not np.isfinite(z)
            or not 0 <= z <= self.mapped_regions["physical_right"][-1]
        ):
            return self.positions, self.mapped, None
        index = int(np.searchsorted(self.positions, z))
        if index < len(self.positions) and self.positions[index] == z:
            return self.positions, self.mapped, index
        x = position_events(
            pl.DataFrame(dict(z=[z], t=[0.0], type=["collision"])),
            self.mapped_regions,
            self.ui.mode.value,
            own_zones=False,
        )["position"][0]
        return (
            np.insert(self.positions, index, z),
            np.insert(self.mapped, index, x),
            index,
        )
