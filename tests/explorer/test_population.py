"""Population fractions account for the whole facet through terminal events."""

import polars as pl

from apitofresview.plotting.explorer import population
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
        events.with_columns(pl.lit("A").alias("daughter")),
        coordinate_regions(regions, "physical"),
        "physical",
    )
    areas = population.population_areas(members, events, 0, 5200)
    product = areas["product", "A"]
    assert product.filter(pl.col("fraction") > 0)["position"][0] == 3000
    assert product["fraction"][-1] == 0.25
    assert areas["escaped", None]["fraction"][-1] == 0.25
    assert areas[population.UNKNOWN]["fraction"][-1] == 0.5
    assert areas[population.UNKNOWN]["top"].to_list() == [1.0] * product.height

    no_events = population.population_areas(members, events.head(0), 0, 5200)
    assert set(no_events) == {population.UNKNOWN}
    assert no_events[population.UNKNOWN]["fraction"].to_list() == [1.0, 1.0]
