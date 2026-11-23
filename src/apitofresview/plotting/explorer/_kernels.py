"""Use the compiled Explorer kernels when present, with JIT for source installs."""

import sys

try:
    from _apitofresview_kernels import (
        _summarize_batch,
        envelope_edges,
        nearest_y,
        stack_swarm,
        terminal_swarm,
    )
except ModuleNotFoundError as exc:
    if getattr(sys, "frozen", False) or exc.name != "_apitofresview_kernels":
        raise
    from ._beeswarm import envelope_edges, nearest_y, stack_swarm, terminal_swarm
    from ._overview_kernel import _summarize_batch

__all__ = [
    "_summarize_batch",
    "envelope_edges",
    "nearest_y",
    "stack_swarm",
    "terminal_swarm",
]
