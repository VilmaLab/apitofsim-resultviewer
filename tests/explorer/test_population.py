"""Charged species persist across escape and follow axial history order."""

import polars as pl

from apitofresview.plotting.explorer import population
from apitofresview.plotting.explorer.data import events_for
from apitofresview.plotting.explorer.layout import coordinate_regions, position_events

from .helpers import cohort, event


def test_axial_population_preserves_history_order_and_handles_missing_histories():
    members, events, _, regions = cohort(
        [1, 2, 3, 4],
        event(1, "init", 1, 0, 0),
        event(2, "collision", 1, 1, 3),
        event(3, "fragmentation", 1, 2, 1, 7),
        event(4, "init", 2, 0, 0),
        event(5, "escape", 2, 3, 5),
        event(6, "init", 3, 0, 0),
        event(7, "collision", 3, 3, 4),
    )
    events = position_events(
        events,
        coordinate_regions(regions, "physical"),
        "physical",
    )
    labels = {1: "Parent", 2: "A", 3: "B"}
    products = {7: (2, 3)}
    areas = population.population_areas(members, events, 0, 5200, 1, labels, products)
    product = areas[2]
    assert product.filter(pl.col("fraction") > 0)["position"][0] == 3000
    assert product["fraction"][-1] == 0.25
    assert areas[3]["fraction"][-1] == 0.25
    assert areas[1]["fraction"][-1] == 0.25
    assert 5000 not in areas[1]["position"].to_list()  # Escape does not change species.
    single = members.head(1)
    split = population.population_areas(
        single, events_for(events, single), 0, 5200, 1, labels, products
    )
    assert split[3]["top"][-1] == 2  # Each charged product contributes an ion.

    no_events = population.population_areas(
        members, events.head(0), 0, 5200, 1, labels, products
    )
    assert no_events == {}
