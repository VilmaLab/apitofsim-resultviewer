"""Charged species populations reconstructed from recorded realization histories."""

from collections import Counter, defaultdict
from itertools import groupby
from math import isfinite

import polars as pl

from .plot import COLORS, PATHWAY_COLORS


def species_catalog(cluster, realizations, pathways, events, fate_colors):
    labels = {}
    if realizations.height:
        initial = realizations.row(0, named=True)
        if initial.get("initial_charged", True):
            labels[cluster] = initial.get("initial_species", "Initial cluster")
    products = {}
    if "charged_products" in pathways.columns:
        for pathway, charged in pathways.select(
            "pathway_id", "charged_products"
        ).iter_rows():
            products[pathway] = tuple(product["id"] for product in charged)
            for product in charged:
                labels[product["id"]] = product["name"]

    # Prefer the canonical color of an observed pathway that produces this species.
    # A split that actually tracks the species takes priority over a terminal product.
    colors = {cluster: COLORS["escape"]} if cluster in labels else {}
    used = set(colors.values())
    splits = events.filter(pl.col("type") == "fragmentation")
    if "next_cluster_id" not in splits.columns:
        splits = splits.with_columns(pl.lit(None).alias("next_cluster_id"))
    routes = (
        splits.select("pathway_id", "event_pathway", "next_cluster_id")
        .unique()
        .sort(["next_cluster_id", "pathway_id"], nulls_last=True)
    )
    for pathway, name, destination in routes.iter_rows():
        candidates = (
            (destination,) if destination is not None else products.get(pathway, ())
        )
        color = fate_colors[name]
        for species in candidates:
            if species in labels and species not in colors and color not in used:
                colors[species] = color
                used.add(color)
    available = [color for color in PATHWAY_COLORS if color not in used] + list(
        PATHWAY_COLORS
    )
    for i, species in enumerate(sorted(labels.keys() - colors.keys())):
        colors[species] = available[i % len(available)]
    return labels, products, colors


def history_species(history, terminal_id, start, cluster, labels, products, tree):
    """Yield populations in event order, projected onto a monotone X axis."""
    position = start
    for index, event in enumerate(history):
        x = event["position"]
        kind = event["type"]
        if (
            x is None
            or not isfinite(x)
            or (
                tree
                and "provenance" in event
                and (
                    event["provenance"] != "persisted" or not event["continuity_valid"]
                )
            )
        ):
            break
        if index == 0:
            yield start, (cluster,) if cluster in labels else ()
        position = max(position, x)
        if kind == "fragmentation" and tree and event["next_particle_index"] >= 0:
            destination = event["next_cluster_id"]
            if destination is None:
                break
            yield position, (destination,) if destination in labels else ()
        elif kind in ("fragmentation", "escape"):
            if event["id"] == terminal_id:
                if kind == "fragmentation":
                    yield position, products.get(event["pathway_id"], ())
                return
            break
    yield position, ()


def population_areas(
    members, events, left_edge, right_edge, cluster, labels, products, *, tree=False
):
    """Stack charged species counts per selected realization on a shared step grid."""
    n = members.height
    if not n:
        return {}
    # Events arrive in numeric result, realization, and event index order.
    terminals = dict(members.select("id", "terminal_event_id").iter_rows())
    changes = defaultdict(Counter)
    for rid, rows in groupby(
        events.iter_rows(named=True), key=lambda e: e["realization_id"]
    ):
        previous = ()
        for x, species in history_species(
            rows, terminals[rid], left_edge, cluster, labels, products, tree
        ):
            if species != previous:
                changes[x].subtract(previous)
                changes[x].update(species)
                previous = species

    positions = sorted(x for x, delta in changes.items() if any(delta.values()))
    x = [left_edge, *[p for p in positions for _ in range(2)], right_edge]
    bottom = [0.0] * len(x)
    states = {species for delta in changes.values() for species in delta}
    areas = {}
    for species in sorted(
        states, key=lambda species: (species != cluster, labels[species], species)
    ):
        count = 0
        fractions = [0.0]
        for position in positions:
            fractions.append(count / n)
            count += changes[position][species]
            fractions.append(count / n)
        fractions.append(count / n)
        if not any(fractions):
            continue
        top = [lo + value for lo, value in zip(bottom, fractions, strict=True)]
        areas[species] = pl.DataFrame(
            dict(position=x, bottom=bottom, top=top, fraction=fractions),
            schema_overrides={"position": pl.Float64},
        )
        bottom = top
    return areas
