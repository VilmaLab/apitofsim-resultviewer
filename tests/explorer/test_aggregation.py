"""Explorer aggregation behavior."""

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from apitofresview.plotting.explorer import aggregation, data
from apitofresview.plotting.explorer import layout as event_layout

from .helpers import cohort, event, outcome_cohort


def test_regional_fractions_keep_unresolved_in_denominator():
    realizations, events, _, regions = outcome_cohort()
    mapping = event_layout.coordinate_regions(regions, "schematic")
    positioned = event_layout.position_events(events, mapping, "schematic")
    bars = aggregation.regional_fractions(realizations, positioned, mapping)
    assert bars["fraction"].to_list() == [0, 0, 0.4, 0, 0, 0.2]
    assert bars["name"].to_list() == [*data.REGIONS, "Escaped"]
    assert (bars["left"][-1], bars["right"][-1]) == event_layout.escape_slot(mapping)
    empty = aggregation.regional_fractions(
        realizations.head(0), positioned.head(0), mapping
    )
    assert empty["fraction"].sum() == 0


@pytest.mark.parametrize("rid", [1, 3])
def test_regional_fractions_selected_denominator(rid):
    realizations, events, _, regions = outcome_cohort()
    mapping = event_layout.coordinate_regions(regions, "schematic")
    positioned = event_layout.position_events(events, mapping, "schematic")
    members = realizations.filter(pl.col("id") == rid)
    bars = aggregation.regional_fractions(
        members, data.events_for(positioned, members), mapping
    )
    assert bars["fraction"].sum() == 1


@pytest.mark.parametrize("mode", ["schematic", "equal", "physical", "time"])
def test_initial_events_do_not_contribute_to_region_fractions(mode):
    members, events, _, regions = cohort(
        [1], event(1, "init", 1, 0, 0), event(2, "escape", 1, 1, 5)
    )
    mapping = event_layout.coordinate_regions(regions, mode)
    positioned = event_layout.position_events(events, mapping, mode)
    assert aggregation.regional_fractions(members, positioned, mapping)[
        "fraction"
    ].to_list() == [0, 0, 0, 0, 0, 1]


def test_cumulative_areas_share_tied_grid_and_preserve_pathway_order():
    members, events, _, regions = cohort(
        [1, 2, 3, 4, 5],
        event(1, "fragmentation", 1, 1, 2, 7),
        event(2, "fragmentation", 2, 1, 2, 7),
        event(3, "fragmentation", 3, 1, 2, 8),
        event(4, "escape", 4, 2, 5),
        event(5, "collision", 5, 1, 1),
    )
    pathways = pl.DataFrame(
        {"pathway_id": [7, 8], "pathway": ["Parent → A + B", "Parent → C + D"]}
    )
    members = data.classify(
        members.select("id", "experiment_result_id"), events, pathways, "Parent"
    )
    positioned = event_layout.position_events(
        events, event_layout.coordinate_regions(regions, "time"), "time"
    )
    fates = ["Parent → C + D", "Parent → A + B", "Parent → Parent", "Incomplete"]
    areas = aggregation.cumulative_areas(members, positioned, 0, 3, fates)
    assert list(areas) == fates[:-1]
    first, second, third = areas.values()
    for frame in areas.values():
        assert frame["position"].to_list() == [0, 1, 1, 2, 2, 3]
    assert first["fraction"].to_list() == [0, 0, 0.2, 0.2, 0.2, 0.2]
    assert second["fraction"].to_list() == [0, 0, 0.4, 0.4, 0.4, 0.4]
    assert third["fraction"].to_list() == [0, 0, 0, 0, 0.2, 0.2]
    assert first["bottom"].to_list() == [0] * 6
    assert second["bottom"].to_list() == first["top"].to_list()
    assert third["bottom"].to_list() == second["top"].to_list()
    assert third["top"][-1] == pytest.approx(0.8)
    reordered = aggregation.cumulative_areas(
        members.reverse(), positioned.reverse(), 0, 3, fates
    )
    for name in areas:
        assert_frame_equal(areas[name], reordered[name])


@pytest.mark.parametrize("empty", [False, True])
def test_cumulative_areas_without_terminal_events(empty):
    members, events, _, regions = cohort([] if empty else [1, 2])
    positioned = event_layout.position_events(
        events, event_layout.coordinate_regions(regions, "time"), "time"
    )
    assert aggregation.cumulative_areas(members, positioned, 0, 1, ["Incomplete"]) == {}
