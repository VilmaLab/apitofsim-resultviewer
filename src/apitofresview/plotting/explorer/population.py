"""Reconstruct one population state per realization from its recorded history."""

from collections import Counter, defaultdict
from itertools import groupby
from math import isfinite

import polars as pl


INITIAL = ("initial", None)
UNKNOWN = ("unknown", None)


def population_labels(pathways, tree):
    species = tree.species if tree else {}
    labels = {
        INITIAL: "Initial cluster (unfragmented)",
        UNKNOWN: "Unknown history",
        ("escaped", None): "Escaped · Initial cluster",
    }
    descendants = set(tree.nodes.values()) if tree else set()
    for cluster in sorted(descendants):
        labels["active", cluster] = f"Simulated descendant · {species[cluster]}"
        labels["escaped", cluster] = f"Escaped · {species[cluster]}"
    if "daughter" in pathways.columns:
        for daughter in sorted(pathways["daughter"].drop_nulls().unique()):
            labels["product", daughter] = f"Terminal product · {daughter}"
    return labels


def history_states(history, terminal_id, start, tree):
    """Yield changes in chronological order, projected onto a monotone X axis."""
    state = UNKNOWN
    position = start
    for index, event in enumerate(history):
        x = event["position"]
        kind = event["type"]
        if (
            x is None
            or not isfinite(x)
            or (
                tree
                and (
                    event["provenance"] != "persisted" or not event["continuity_valid"]
                )
            )
        ):
            break
        if index == 0:
            state = INITIAL
            yield max(start, x) if kind == "init" else start, state
        position = max(position, x)
        if kind == "fragmentation" and tree and event["next_particle_index"] >= 0:
            destination = tree.nodes.get(
                (event["experiment_result_id"], event["next_particle_index"])
            )
            if destination is None:
                break
            state = ("active", destination)
            yield position, state
        elif kind in ("fragmentation", "escape"):
            if event["id"] == terminal_id:
                if kind == "escape":
                    yield position, ("escaped", state[1])
                else:
                    daughter = event.get("daughter")
                    yield position, ("product", daughter) if daughter else UNKNOWN
                return
            break
    yield position, UNKNOWN


def population_areas(members, events, left_edge, right_edge, *, tree=None):
    """Stack fractions on one step grid with a fixed denominator per facet."""
    n = members.height
    if not n:
        return {}
    # Events arrive sorted by realization, raw time, and event ID.
    terminals = dict(members.select("id", "terminal_event_id").iter_rows())
    changes = defaultdict(Counter)
    for rid, rows in groupby(
        events.iter_rows(named=True), key=lambda e: e["realization_id"]
    ):
        state = UNKNOWN
        for x, next_state in history_states(rows, terminals[rid], left_edge, tree):
            if next_state != state:
                changes[x][state] -= 1
                changes[x][next_state] += 1
                state = next_state

    positions = sorted(x for x, delta in changes.items() if any(delta.values()))
    x = [left_edge, *[p for p in positions for _ in range(2)], right_edge]
    bottom = [0.0] * len(x)
    states = {state for delta in changes.values() for state in delta} | {UNKNOWN}
    areas = {}
    for state in sorted(
        states, key=lambda s: (s == UNKNOWN, s != INITIAL, s[0], str(s[1]))
    ):
        count = n if state == UNKNOWN else 0
        fractions = [count / n]
        for position in positions:
            fractions.append(count / n)
            count += changes[position][state]
            fractions.append(count / n)
        fractions.append(count / n)
        starts_at_edge = positions and positions[0] == left_edge
        if not any(fractions[2:] if starts_at_edge else fractions):
            continue
        top = [lo + value for lo, value in zip(bottom, fractions, strict=True)]
        areas[state] = pl.DataFrame(
            dict(position=x, bottom=bottom, top=top, fraction=fractions),
            schema_overrides={"position": pl.Float64},
        )
        bottom = top
    return areas
