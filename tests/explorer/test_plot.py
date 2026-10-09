"""Explorer plot behavior."""

from bokeh.palettes import Category20

from apitofresview.plotting.explorer import plot


def test_categorical_pathway_palette():
    names = [f"Pathway {i:02}" for i in range(16)]
    colors = plot._pathway_colors(
        ["Incomplete", *reversed(names), "Survival"], "Survival"
    )
    assert colors[names[0]] == plot.COLORS["fragmentation"] == "#d62728"
    assert colors["Survival"] == plot.COLORS["escape"] == "#2ca02c"
    assert plot.COLORS["collision"] == "#1f77b4"
    assert [colors[name] for name in names[:8]] == [
        "#d62728",
        "#ff7f0e",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    pathway_colors = {colors[name] for name in names}
    assert len(pathway_colors) == len(names)
    assert pathway_colors <= set(Category20[20])
    assert pathway_colors.isdisjoint({"#1f77b4", "#aec7e8", "#2ca02c", "#98df8a"})
    assert plot._pathway_colors(names[:2], "Survival")[names[1]] == colors[names[1]]
