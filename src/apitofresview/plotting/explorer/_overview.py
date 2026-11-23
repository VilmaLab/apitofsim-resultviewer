"""Stream large Parquet tree histories before enriching the retained events."""

from itertools import chain, pairwise

import numpy as np
import polars as pl
import pyarrow as pa
from apitofsim.recording import PARQUET_VERSION

from ._kernels import _summarize_batch
from ._parquet import InMemoryEventQueries

# Small selections can use SQL without paying to compile the scanner.
MIN_STREAM_EVENTS = 1_000_000
BATCH_SIZE = 262_144
SUMMARY_INPUTS = (
    "realization_index",
    "event_index",
    "event_kind",
    "particle_index",
    "next_particle_index",
    "accepted",
    "x",
    "y",
    "z",
    "t",
)
SCAN_FIELDS = (*SUMMARY_INPUTS, "result_id", "transition_index")
SUMMARY_COLUMNS = ("collisions", "rejected_collisions", "clamped_events", "max_radial")
MOTION_COLUMNS = (
    "postime, velocity, omega, rot_energy, vibrational_energy, theta, u_norm"
)
HISTORY_KEY = ("result_id", "realization_index")
EVENT_KEY = (*HISTORY_KEY, "event_index")


def _stream_result(archive, result, nodes, retained, histories):
    state = (-1, -1, 0, 0, 0, 0, -np.inf, -np.inf, -np.inf, True)
    query = archive.filter(f"result_id = {result}").select(", ".join(SCAN_FIELDS))
    with query.arrow(batch_size=BATCH_SIZE) as reader:
        # One batch of lookahead identifies the final row of a history without
        # retaining its preceding collision rows across batches.
        batches = (batch for batch in reader if batch.num_rows)
        for batch, following in pairwise(chain(batches, (None,))):
            next_realization = (
                following["realization_index"][0].as_py()
                if following is not None
                else -1
            )
            if next_realization is None or any(
                column.null_count for column in batch.columns
            ):
                return False
            summary = _summarize_batch(
                tuple(
                    batch[name].to_numpy(zero_copy_only=True) for name in SUMMARY_INPUTS
                ),
                nodes,
                state,
                next_realization,
            )
            if summary is None:
                return False
            state, indices, prefix, totals, radial, continuity = summary
            if indices.size:
                retained.append(
                    batch.take(pa.array(indices))
                    .append_column("population_z", pa.array(prefix[0]))
                    .append_column("population_t", pa.array(prefix[1]))
                )
            if radial.size:
                histories.append(
                    pa.RecordBatch.from_arrays(
                        [
                            pa.array(np.full(radial.size, result, dtype=np.int64)),
                            *(pa.array(column) for column in totals),
                            pa.array(radial),
                            pa.array(continuity),
                        ],
                        names=[
                            *HISTORY_KEY,
                            *SUMMARY_COLUMNS[:3],
                            "last_index",
                            "max_radial",
                            "history_continuous",
                        ],
                    )
                )
    return True


def stream_tree_overview(db, experiment, cluster):
    """Return the sparse event frame, or use SQL for unsupported archives."""
    if not db.readonly:
        return None

    results = db.db.execute(
        """select e.result_id, e.event_count
        from event_execution e join cluster_tree_experiment_result r on r.id = e.result_id
        where e.run_id = ? and e.cluster_id = ? and e.recorded
        order by e.result_id""",
        [experiment, cluster],
    ).fetchall()
    if not results or sum(size for _, size in results) < MIN_STREAM_EVENTS:
        return None
    (version,) = db.db.execute(
        "select schema_version from event_recording_state"
    ).fetchone()
    if version != PARQUET_VERSION or db.events_path is None:
        return None

    # ExperimentDatabase has verified the sidecar and holds the shared lock.
    archive = db.db.read_parquet(str(db.events_path), hive_partitioning=False)
    retained, histories = [], []
    for result, _ in results:
        nodes = np.array(
            db.db.execute(
                """select node_index from cluster_tree_node
                where experiment_result_id = ? order by node_index""",
                [result],
            ).fetchnumpy()["node_index"],
            dtype=np.int64,
        )
        if not _stream_result(archive, result, nodes, retained, histories):
            return None
    if not retained:
        return None

    rows = pa.Table.from_batches(retained)
    events = (
        InMemoryEventQueries(db.db, rows)
        .tree_events_query(run_id=experiment, parent=cluster)
        .select(f"* exclude ({MOTION_COLUMNS})")
        .pl()
    )
    return (
        events.join(
            pl.from_arrow(rows.select([*EVENT_KEY, "population_z", "population_t"])),
            on=EVENT_KEY,
            maintain_order="left",
        )
        .join(
            pl.from_arrow(pa.Table.from_batches(histories)),
            on=HISTORY_KEY,
            maintain_order="left",
        )
        .with_columns(
            (
                pl.col("history_continuous")
                & (pl.col("provenance") == "persisted").all().over("realization_id")
            ).alias("history_valid")
        )
        .select(
            *events.columns,
            *SUMMARY_COLUMNS,
            "last_index",
            "population_z",
            "population_t",
            "history_valid",
        )
    )
