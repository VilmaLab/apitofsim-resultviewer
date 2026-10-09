"""Explorer details behavior."""

import polars as pl

from apitofresview.plotting.explorer.details import render_details

from .helpers import cohort, event


def test_details_use_cohort_number_and_convert_units():
    realizations, events, _, _ = cohort(
        [16052], event(42, "collision", 16052, 2e-9, 0.003)
    )
    events = events.with_columns(
        pl.Series("velocity", [{"x": -3.0, "y": 4.0, "z": -12.0}]),
        pl.Series("omega", [{"x": 2.0, "y": -3.0, "z": 6.0}]),
        pl.lit(7).alias("particle_index"),
    )
    selected = realizations.with_row_index("realization_number", offset=1)
    details = render_details(selected, events, 42)
    assert "Realization #1</h3>" in details
    assert "Realization #16052" not in details
    assert "Result #" not in details and "disabled" not in details
    assert (
        "<td>2</td><td>3</td><td>5000</td><td>-12</td><td>5</td><td>7</td>" in details
    )
    assert all(
        f"<th>{heading}</th>" in details
        for heading in (
            "Axial velocity (m/s)",
            "Radial speed (m/s)",
            "Angular speed (rad/s)",
        )
    )
    assert "Particle index" not in details and "Pathway ID" not in details
    cartesian = render_details(selected, events, 42, radial=False)
    assert (
        "<td>2</td><td>3000</td><td>4000</td><td>3</td><td>-3</td><td>4</td><td>-12</td><td>2</td><td>-3</td><td>6</td>"
        in cartesian
    )
    assert all(
        f"<th>{label} {axis} ({unit})</th>" in cartesian
        for label, unit in (("Velocity", "m/s"), ("Angular velocity", "rad/s"))
        for axis in "xyz"
    )
    assert "Radial speed" not in cartesian and "Angular speed" not in cartesian
    assert "Particle index" not in cartesian and "Pathway ID" not in cartesian


def test_details_format_missing_and_boolean_values():
    members, events, _, _ = cohort(
        [1], event(1, "collision", 1, 0, 0), event(2, "escape", 1, 1, 5)
    )
    members = members.with_row_index("realization_number", offset=1)
    events = events.with_columns(
        pl.Series("velocity", [{"x": 1.0, "y": 2.0, "z": 3.0}, None]),
        pl.Series("omega", [{"x": 4.0, "y": 5.0, "z": 6.0}, None]),
        pl.Series("accepted", [False, None]),
    )
    html = render_details(members, events, 1)
    assert "Angular speed (rad/s)" in html
    assert "<td>No</td>" in html and "<td>—</td>" in html
    assert 'id="event-1" style="background:#fde68a"' in html


def test_details_escape_fate_and_prompt_when_unselected():
    members, events, _, _ = cohort([1], event(1, "collision", 1, 0, 0))
    members = members.with_row_index("realization_number", offset=1).with_columns(
        pl.lit("A < B & C").alias("fate")
    )
    assert "<em>A &lt; B &amp; C</em>" in render_details(members, events, None)
    assert (
        render_details(members.head(0), events.head(0), None)
        == "<p><em>Click an event to inspect its realization.</em></p>"
    )
