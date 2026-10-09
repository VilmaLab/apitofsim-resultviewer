"""Explorer layout behavior."""

from math import hypot, sin

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from apitofresview.plotting.explorer import layout as event_layout

from .helpers import cohort, event, outcome_cohort


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("physical", [0, 1000, 2000, 5000, 6000, 5000]),
        ("equal", [0, 1, 2, 5, 6, 5.325]),
        ("schematic", [0, 3, 6, 9, 10, 9.325]),
        ("time", [0, 1, 2, 3, 4, 5]),
    ],
)
def test_coordinate_modes_and_boundaries(mode, expected):
    _, events, _, regions = cohort(
        [1],
        *[
            event(i + 1, "escape" if i == 5 else "collision", 1, i, z)
            for i, z in enumerate([0, 1, 2, 5, 6, 5])
        ],
    )
    assert events["region"].to_list() == [0, 1, 2, 4, None, 4]
    positioned = event_layout.position_events(
        events, event_layout.coordinate_regions(regions, mode), mode
    )
    assert positioned["position"].to_list() == pytest.approx(expected)


def test_zero_width_regions():
    _, events, _, regions = cohort(
        [1],
        event(1, "fragmentation", 1, 1, 0, 7),
        boundaries=(0.0, 0.0, 1.0, 2.0, 3.0, 4.0),
    )
    assert events["region"][0] == 1
    positioned = event_layout.position_events(
        events, event_layout.coordinate_regions(regions, "schematic"), "schematic"
    )
    assert positioned["position"][0] == 0


def test_layouts_and_envelope():
    _, events, _, regions = cohort(
        [1],
        event(3, "collision", 1, 0, 0),
        event(1, "collision", 1, 1, 0),
        event(2, "collision", 1, 2, 0),
        event(4, "collision", 1, 3, 2),
        event(5, "escape", 1, 4, 5),
    )
    events = event_layout.position_events(
        events, event_layout.coordinate_regions(regions, "physical"), "physical"
    )
    assert (
        event_layout.layout_events(events, "radial")["plot_y"].to_list() == [5000] * 5
    )
    assert event_layout.layout_events(events, "strip")[
        "plot_y"
    ].to_list() == pytest.approx([0.25 * sin(i * 12.9898) for i in events["id"]])

    def pack(frame):
        return event_layout.pack_beeswarm(frame, 0, 5000, 600, 380)[0]

    packed = pack(events)
    assert packed["plot_y"].to_list() == pytest.approx([-10, 0, 10, 0, 0])
    assert_frame_equal(
        packed.select("id", "plot_y").sort("id"),
        pack(events.reverse()).select("id", "plot_y").sort("id"),
    )
    envelope = event_layout.beeswarm_envelope(packed, 5 * 5000 / 600, 5)
    assert envelope["plot_y"].max() == 15
    assert envelope["plot_y"].min() == -15
    assert envelope["position"].min() < 0
    assert envelope["position"].max() > 5000
    assert_frame_equal(
        envelope, event_layout.beeswarm_envelope(packed.reverse(), 5 * 5000 / 600, 5)
    )
    for subset in (events.head(0), events.head(1), events.head(3)):
        packed = pack(subset)
        outline = event_layout.beeswarm_envelope(packed, 1, 5)
        assert outline.is_empty() == subset.is_empty()
        assert packed.height == subset.height
        if not subset.is_empty():
            assert outline["plot_y"].max() > packed["plot_y"].max()
            assert outline["plot_y"].min() < packed["plot_y"].min()


@pytest.mark.parametrize("width,height", [(600, 380), (300, 200), (900, 600)])
def test_beeswarm_circles_touch_without_overlaps(width, height):
    # A dense stack exercises more than the old seven levels; nearby columns
    # exercise diagonal tangencies and resizing.
    points = pl.DataFrame(
        {"id": range(80), "position": [i // 20 * 0.005 for i in range(80)]}
    )
    packed, diameter = event_layout.pack_beeswarm(points, 0, 1, width, height)
    centers = list(zip(packed["position"] * width, packed["plot_y"], strict=True))
    for index, (x, y) in enumerate(centers):
        distances = [hypot(x - px, y - py) for px, py in centers[:index]]
        if distances:
            assert min(distances) >= diameter - 1e-7
            if abs(y) > 1e-7:
                assert min(distances) == pytest.approx(diameter)
    assert max(abs(y) for _, y in centers) + diameter / 2 <= height / 2
    identical = points.with_columns(pl.lit(0.0).alias("position"))
    stacked, size = event_layout.pack_beeswarm(identical, 0, 1, width, height)
    assert [
        b - a for a, b in zip(sorted(stacked["plot_y"]), sorted(stacked["plot_y"])[1:])
    ] == pytest.approx([size] * 79)


@pytest.mark.parametrize("mode", ["schematic", "equal", "physical", "time"])
def test_initial_events_position(mode):
    _, events, _, regions = cohort(
        [1], event(1, "init", 1, 0, 0), event(2, "escape", 1, 1, 5)
    )
    mapped = event_layout.coordinate_regions(regions, mode)
    positioned = event_layout.position_events(events, mapped, mode)
    if mode in ("schematic", "equal"):
        lo, hi = event_layout.initial_slot(mapped)
        escape_lo, escape_hi = event_layout.escape_slot(mapped)
        assert hi - lo == pytest.approx(escape_hi - escape_lo)
        assert positioned["position"][0] == (lo + hi) / 2 < mapped["left"][0]
    else:
        assert positioned["position"][0] == 0


@pytest.mark.parametrize("mode", ["schematic", "equal", "physical", "time"])
def test_collapsed_terminal_coordinates(mode):
    _, events, _, regions = cohort(
        [1], event(1, "init", 1, 0, 0), event(2, "escape", 1, 1, 5)
    )
    mapped = event_layout.coordinate_regions(regions, mode)
    collapsed = event_layout.position_events(events, mapped, mode, own_zones=False)
    if mode in ("schematic", "equal"):
        assert collapsed["position"].to_list() == [
            mapped["left"][0],
            mapped["right"][-1],
        ]
    else:
        assert_frame_equal(
            collapsed, event_layout.position_events(events, mapped, mode)
        )


@pytest.mark.parametrize("mode", ["schematic", "equal"])
def test_terminal_spreading_modes(mode):
    _, events, _, regions = cohort(
        list(range(1, 21)),
        *[event(i, "init", i, 0, 0) for i in range(1, 21)],
        *[event(20 + i, "escape", i, 1, 5) for i in range(1, 21)],
        event(41, "collision", 1, 0.5, 2),
    )
    mapped = event_layout.coordinate_regions(regions, mode)
    events = event_layout.position_events(events, mapped, mode).with_columns(
        pl.col("realization_id").alias("realization_number")
    )
    original = event_layout.layout_events(events, "realization")
    start, end = (
        event_layout.initial_slot(mapped)[0],
        event_layout.escape_slot(mapped)[1],
    )

    def spread(frame, layout="realization", height=380):
        return event_layout.spread_terminal_x(
            frame, mapped, layout, start, end, 600, height, (0.5, 20.5)
        )

    displayed = spread(original)
    assert displayed["plot_y"].to_list() == original["plot_y"].to_list()
    assert (
        displayed.filter(pl.col("type") == "collision")["position"].to_list()
        == original.filter(pl.col("type") == "collision")["position"].to_list()
    )
    for kind, slot in (
        ("init", event_layout.initial_slot(mapped)),
        ("escape", event_layout.escape_slot(mapped)),
    ):
        xs = displayed.filter(pl.col("type") == kind)["position"].to_list()
        assert xs[0] != xs[1] and xs[:2] == xs[2:4]
        assert slot[0] < min(xs) < max(xs) < slot[1]
    assert_frame_equal(displayed.sort("id"), spread(original.reverse()).sort("id"))
    dense = spread(original, height=20)
    assert dense.filter(pl.col("type") == "init")["position"].n_unique() > 2
    strip = event_layout.layout_events(events, "strip")
    jittered = spread(strip, "strip")
    assert jittered["plot_y"].to_list() == strip["plot_y"].to_list()
    assert jittered.filter(pl.col("type") == "init")["position"].n_unique() == 20
    assert_frame_equal(jittered.sort("id"), spread(strip.reverse(), "strip").sort("id"))


@pytest.mark.parametrize("kind", ["init", "escape"])
def test_beeswarm_terminal_spill_stages(kind):
    _, events, _, regions = cohort(
        [1],
        *[event(i, "collision", 1, i, 2) for i in range(1, 4)],
        *[event(i, kind, 1, i, 0 if kind == "init" else 5) for i in range(4, 24)],
    )
    mapped = event_layout.coordinate_regions(regions, "equal")
    events = event_layout.position_events(events, mapped, "equal")
    slot = (
        event_layout.initial_slot(mapped)
        if kind == "init"
        else event_layout.escape_slot(mapped)
    )
    start, end = (
        event_layout.initial_slot(mapped)[0],
        event_layout.escape_slot(mapped)[1],
    )
    # 100 px per unit: this 65 px slot holds five centered columns of 10 px circles.
    packed, diameter = event_layout.pack_beeswarm(
        events, start, end, (end - start) * 100, 380, terminal_regions=mapped
    )
    assert diameter == 10
    ordinary = packed.filter(pl.col("type") == "collision")
    assert ordinary["plot_y"].abs().max() == 10
    terminals = packed.filter(pl.col("type") == kind).sort("id")
    assert terminals["position"][:3].to_list() == pytest.approx([sum(slot) / 2] * 3)
    assert terminals["plot_y"][:3].to_list() == [0, 10, -10]
    assert terminals["position"][3] != terminals["position"][0]
    assert terminals["plot_y"][:15].abs().max() == 10
    assert terminals["plot_y"][15:].abs().max() == 20
    assert terminals["position"][15:].n_unique() == 5
    centers = list(zip(packed["position"] * 100, packed["plot_y"], strict=True))
    for i, (x, y) in enumerate(centers):
        assert all(hypot(x - px, y - py) >= diameter - 1e-7 for px, py in centers[:i])
    assert terminals["position"].min() >= slot[0] + diameter / 200
    assert terminals["position"].max() <= slot[1] - diameter / 200
    shuffled, _ = event_layout.pack_beeswarm(
        events.reverse(), start, end, (end - start) * 100, 380, terminal_regions=mapped
    )
    assert_frame_equal(packed.sort("id"), shuffled.sort("id"))


@pytest.mark.parametrize("count", [0, 1, 10, 200])
def test_beeswarm_terminal_only_and_height_fit(count):
    _, events, _, regions = cohort(
        [1], *[event(i, "init", 1, i, 0) for i in range(1, count + 1)]
    )
    mapped = event_layout.coordinate_regions(regions, "equal")
    events = event_layout.position_events(events, mapped, "equal")
    start, end = (
        event_layout.initial_slot(mapped)[0],
        event_layout.escape_slot(mapped)[1],
    )
    packed, diameter = event_layout.pack_beeswarm(
        events, start, end, 600, 60, terminal_regions=mapped
    )
    assert packed.height == count
    if count:
        assert packed["plot_y"].abs().max() + diameter / 2 <= 30
        if count >= 10:
            assert packed["position"].n_unique() > 1
            assert packed.sort("id")["plot_y"][:3].to_list() == [0, 0, 0]
        centers = list(
            zip(
                (packed["position"] - start) * 600 / (end - start),
                packed["plot_y"],
                strict=True,
            )
        )
        for i, (x, y) in enumerate(centers):
            assert all(
                hypot(x - px, y - py) >= diameter - 1e-7 for px, py in centers[:i]
            )


def test_schematic_and_time_coordinates():
    _, events, _, regions = outcome_cohort()
    mapping = event_layout.coordinate_regions(regions, "schematic")
    assert mapping["left"].to_list() == [0, 3, 6, 7, 8]
    assert mapping["right"].to_list() == [3, 6, 7, 8, 9]
    positioned = event_layout.position_events(events, mapping, "schematic")
    assert positioned.filter(pl.col("id") == 2)["position"][0] == 6
    assert (
        positioned.filter(pl.col("id") == 4)["position"][0]
        == sum(event_layout.escape_slot(mapping)) / 2
    )
    assert (
        event_layout.position_events(
            events, event_layout.coordinate_regions(regions, "time"), "time"
        )["position"].to_list()
        == events["t"].to_list()
    )
