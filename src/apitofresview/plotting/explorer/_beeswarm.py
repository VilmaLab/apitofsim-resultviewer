"""Compiled screen-space beeswarm and envelope kernels."""

from math import floor, sqrt

import numpy as np
from numba import njit


@njit(cache=True)
def nearest_y(lows, highs, count):
    """Find the endpoints of the forbidden interval component containing zero."""
    lo, hi = 0.0, 0.0
    straddles_zero = False
    for j in range(count):
        a, b = lows[j], highs[j]
        straddles_zero |= a < 0 < b
        if a + 1e-9 < 0 < b - 1e-9:
            lo, hi = min(lo, a), max(hi, b)
    if lo == hi:
        if not straddles_zero:
            return 0.0
        # Near zero, two individually tolerated intervals can overlap enough
        # to forbid zero together. Preserve the original sorted merge rule.
        order = np.argsort(lows[:count])
        lo, hi = lows[order[0]], highs[order[0]]
        for j in range(1, count):
            a, b = lows[order[j]], highs[order[j]]
            if a < hi - 1e-9:
                hi = max(hi, b)
            else:
                if lo + 1e-9 < 0 < hi - 1e-9:
                    break
                lo, hi = a, b
        if lo + 1e-9 < 0 < hi - 1e-9:
            return hi if hi <= -lo else lo
        return 0.0
    changed = True
    while changed:
        changed = False
        for j in range(count):
            a, b = lows[j], highs[j]
            if a < hi - 1e-9 and b > lo + 1e-9:
                if a < lo or b > hi:
                    lo, hi = min(lo, a), max(hi, b)
                    changed = True
    return hi if hi <= -lo else lo


@njit(cache=True)
def stack_swarm(xs, width, diameter):
    # No sort: repeatedly close the interval component containing zero.
    n = len(xs)
    ys = np.zeros(n)
    lows, highs = np.empty(n), np.empty(n)
    first = 0
    for i in range(n):
        x = xs[i]
        if x < -diameter or x > width + diameter:
            first = i + 1
            continue
        while first < i and x - xs[first] >= diameter:
            first += 1
        k = i - first
        for j in range(k):
            r = sqrt(max(0.0, diameter * diameter - (x - xs[first + j]) ** 2))
            lows[j], highs[j] = ys[first + j] - r, ys[first + j] + r
        ys[i] = nearest_y(lows, highs, k)
    return ys


@njit(cache=True)
def terminal_swarm(xs, ordinary_xs, ordinary, init, escape, slots, width, diameter):
    # Sparse linked buckets avoid Python per-candidate generators and tuple lists.
    ys = np.zeros(len(xs))
    oy = stack_swarm(ordinary_xs, width, diameter)
    xs = xs.copy()
    extent = 0.0
    buckets = {(0, 0): -1}
    buckets.clear()
    nxt = np.full(len(xs), -1)
    for j in range(len(ordinary)):
        i = ordinary[j]
        ys[i] = oy[j]
        extent = max(extent, abs(oy[j]))
        if -diameter <= xs[i] <= width + diameter:
            key = (int(floor(xs[i] / diameter)), int(floor(ys[i] / diameter)))
            nxt[i] = buckets[key] if key in buckets else -1
            buckets[key] = i
    for kind in range(2):
        indices = init if kind == 0 else escape
        lo, hi = slots[kind, 0], slots[kind, 1]
        if hi < -diameter or lo > width + diameter:
            continue
        center = (lo + hi) / 2
        radius = min(diameter / 2, (hi - lo) / 2)
        steps = min(len(indices), int(floor(((hi - lo) / 2 - radius) / diameter)))
        columns = np.empty(1 + 2 * steps)
        columns[0] = center
        for step in range(1, steps + 1):
            columns[2 * step - 1], columns[2 * step] = (
                center - step * diameter,
                center + step * diameter,
            )
        nlevels = 1 + 2 * int(floor(extent / diameter))
        cursor = 0
        for i in indices:
            while True:
                if cursor < len(columns) * nlevels:
                    col = cursor // nlevels
                    level = cursor % nlevels
                    y = (
                        0.0
                        if level == 0
                        else ((level + 1) // 2) * diameter * (1 if level % 2 else -1)
                    )
                else:
                    offset = cursor - len(columns) * nlevels
                    col = offset % len(columns)
                    row = offset // len(columns)
                    y = (
                        (nlevels // 2 + 1 + row // 2)
                        * diameter
                        * (1 if row % 2 == 0 else -1)
                    )
                x = columns[col]
                cursor += 1
                bx, by = int(floor(x / diameter)), int(floor(y / diameter))
                free = True
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        key = (bx + dx, by + dy)
                        j = buckets[key] if key in buckets else -1
                        while j != -1:
                            if (x - xs[j]) ** 2 + (
                                y - ys[j]
                            ) ** 2 < diameter * diameter * (1 - 1e-9):
                                free = False
                                break
                            j = nxt[j]
                        if not free:
                            break
                    if not free:
                        break
                if free:
                    xs[i], ys[i] = x, y
                    key = (bx, by)
                    nxt[i] = buckets[key] if key in buckets else -1
                    buckets[key] = i
                    break
    return xs, ys


@njit(cache=True)
def envelope_edges(xs, ys, samples, x_padding, y_padding):
    upper, lower = np.empty(len(samples)), np.empty(len(samples))
    first, last = 0, 0
    for i in range(len(samples)):
        x = samples[i]
        while last < len(xs) and xs[last] <= x + x_padding:
            last += 1
        while first < last and xs[first] < x - x_padding:
            first += 1
        hi, lo = -np.inf, np.inf
        for j in range(first, last):
            r = y_padding * sqrt(max(0.0, 1 - ((x - xs[j]) / x_padding) ** 2))
            hi, lo = max(hi, ys[j] + r), min(lo, ys[j] - r)
        upper[i], lower[i] = (hi, lo) if first != last else (0.0, 0.0)
    return upper, lower
