"""Summarize one ordered batch of Parquet history."""

import numpy as np
from numba import njit

# Native event_kind codes from the apitofsim archive.
INIT, COLLISION, POSTCOLLISION, FRAGMENTATION, POSTFRAGMENTATION, ESCAPE = range(6)


@njit(cache=True)
def _summarize_batch(columns, nodes, state, next_realization, tree_mode):
    """Keep visible rows, prefix maxima, and one set of totals per history."""
    realizations, indices, kinds, particles, destinations, accepted, x, y, z, t = (
        columns
    )
    (
        realization,
        last_index,
        previous_particle,
        collisions,
        rejected,
        clamped,
        max_radial,
        max_z,
        max_t,
        valid,
    ) = state
    size = len(realizations)
    retained = np.empty(size, np.int64)
    prefix = np.empty((2, size), np.float64)
    totals = np.empty((5, size), np.int64)
    radial = np.empty(size, np.float64)
    continuity = np.empty(size, np.bool_)
    kept = finished = 0
    known_particle = None

    for i in range(size):
        rid, index, kind, particle = realizations[i], indices[i], kinds[i], particles[i]
        if (
            rid < 0
            or index < 0
            or rid < realization
            or (rid == realization and index <= last_index)
            or not INIT <= kind <= ESCAPE
            or not (
                np.isfinite(x[i])
                and np.isfinite(y[i])
                and np.isfinite(z[i])
                and np.isfinite(t[i])
            )
        ):
            return None
        first = rid != realization
        if first:
            realization = rid
            collisions = rejected = clamped = 0
            max_radial = max_z = max_t = -np.inf
            valid = True
        last_index = index
        if tree_mode:
            valid &= (
                particle == 0
                if kind == INIT
                else not first and particle == previous_particle
            )
            previous_particle = destinations[i] if kind == FRAGMENTATION else particle
        collision = kind == COLLISION
        collisions += collision
        rejected += collision and accepted[i] == 0
        clamped += z[i] < 0
        max_radial = max(max_radial, np.sqrt(x[i] * x[i] + y[i] * y[i]))
        max_z, max_t = max(max_z, z[i]), max(max_t, t[i])

        last = (realizations[i + 1] if i + 1 < size else next_realization) != rid
        if kind not in (COLLISION, POSTCOLLISION) or last:
            retained[kept] = i
            prefix[0, kept], prefix[1, kept] = max_z, max_t
            kept += 1
        elif (
            tree_mode
            and valid
            and (known_particle is None or particle != known_particle)
        ):
            # The tree query checks provenance on the retained rows. A
            # dropped collision has persisted provenance iff its node exists.
            node = np.searchsorted(nodes, particle)
            valid = node < len(nodes) and nodes[node] == particle
            if valid:
                known_particle = particle
        if last:
            totals[0, finished] = rid
            totals[1, finished] = collisions
            totals[2, finished] = rejected
            totals[3, finished] = clamped
            totals[4, finished] = index
            radial[finished], continuity[finished] = max_radial, valid
            finished += 1
    state = (
        realization,
        last_index,
        previous_particle,
        collisions,
        rejected,
        clamped,
        max_radial,
        max_z,
        max_t,
        valid,
    )
    return (
        state,
        retained[:kept].copy(),
        prefix[:, :kept].copy(),
        totals[:, :finished].copy(),
        radial[:finished].copy(),
        continuity[:finished].copy(),
    )
