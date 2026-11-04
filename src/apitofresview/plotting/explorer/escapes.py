"""Escape identities and stable colors, independent of display filters."""

import polars as pl


class EscapeGroups:
    def __init__(self, events, cluster, tree=None, root_name="Escape"):
        from .plot import COLORS, PATHWAY_COLORS

        escaped = events.filter(pl.col("type") == "escape")
        if tree:
            self.records = escaped.select(
                "id",
                pl.concat_str(
                    pl.col("experiment_result_id"),
                    pl.lit(":escape:"),
                    pl.col("particle_index"),
                ).alias("route"),
                pl.concat_str(
                    pl.lit("escape:daughter:"),
                    pl.col("current_cluster_id").fill_null(-1),
                ).alias("daughter"),
                pl.col("current_cluster_name").fill_null("Unknown").alias("name"),
                (pl.col("current_cluster_id") == cluster)
                .fill_null(False)
                .alias("root"),
            )
        else:
            self.records = escaped.select("id").with_columns(
                pl.lit(f"escape:daughter:{cluster}").alias("daughter"),
                pl.lit(f"escape:daughter:{cluster}").alias("route"),
                pl.lit(root_name).alias("name"),
                pl.lit(True).alias("root"),
            )
        daughters = (
            self.records.select("daughter", "name", "root")
            .unique()
            .sort(["root", "name", "daughter"], descending=[True, False, False])
        )
        routes = self.records.select("route", "daughter", "name").unique().sort("route")
        self.names = dict(daughters.select("daughter", "name").iter_rows())
        self.route_daughters = dict(routes.select("route", "daughter").iter_rows())
        self.ambiguous = routes.height > daughters.height
        self.labels = {key: f"{name} → {name}" for key, name in self.names.items()}
        if tree:
            full = tree.leaf_labels()
            self.labels.update(
                {
                    key: full.get(key, self.labels[daughter])
                    for key, daughter in self.route_daughters.items()
                }
            )
        self.short_labels = {
            key: f"→ {self.names[daughter]}"
            for key, daughter in self.route_daughters.items()
        } | {key: f"→ {name}" for key, name in self.names.items()}
        palette = (COLORS["escape"], *PATHWAY_COLORS)
        used = set()

        def allocate():
            # Exhaust the existing palette before deriving lighter variants.
            index = len(used)
            while True:
                base = palette[index % len(palette)]
                level = index // len(palette)
                amount = 1 - 1 / (1 + level * 0.2)
                rgb = [
                    round(int(base[i : i + 2], 16) * (1 - amount) + 255 * amount)
                    for i in (1, 3, 5)
                ]
                color = "#" + "".join(f"{channel:02x}" for channel in rgb)
                if color not in used:
                    used.add(color)
                    return color
                index += 1

        self.colors = {key: allocate() for key in self.names}
        first = set()
        for route, daughter in self.route_daughters.items():
            self.colors[route] = (
                self.colors[daughter] if daughter not in first else allocate()
            )
            first.add(daughter)

    def keys(self, distinguish=False):
        if not distinguish:
            return list(self.names)
        return [
            route
            for daughter in self.names
            for route, parent in self.route_daughters.items()
            if parent == daughter
        ]

    def annotate(self, events):
        return events.join(
            self.records.select(
                "id",
                pl.col("route").alias("escape_route"),
                pl.col("daughter").alias("escape_daughter"),
            ),
            on="id",
            how="left",
            maintain_order="left",
        )

    def route_colors(self, distinguish=False):
        return {
            route: self.colors[route if distinguish else daughter]
            for route, daughter in self.route_daughters.items()
        }
