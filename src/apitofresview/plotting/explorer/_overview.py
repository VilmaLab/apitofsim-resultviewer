"""Stream large Parquet histories before enriching the retained events."""

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
POPULATION_COLUMNS = ("population_z", "population_t")
MOTION_COLUMNS = (
    "postime, velocity, omega, rot_energy, vibrational_energy, theta, u_norm"
)
HISTORY_KEY = ("result_id", "realization_index")
EVENT_KEY = (*HISTORY_KEY, "event_index")


def _stream_result(archive, result, nodes, tree_mode, retained, histories):
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
                tree_mode,
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
                            *(pa.array(column) for column in totals[:-1]),
                            pa.array(radial),
                            pa.array(continuity),
                        ],
                        names=[
                            *HISTORY_KEY,
                            *SUMMARY_COLUMNS[:3],
                            "max_radial",
                            "history_continuous",
                        ],
                    )
                )
    return True


def stream_overview(db, experiment, cluster, tree_mode):
    """Return ordered sparse events and one row of totals per history."""
    if not db.readonly:
        return None

    tree_results = (
        "join cluster_tree_experiment_result r on r.id = e.result_id"
        if tree_mode
        else ""
    )
    results = db.db.execute(
        f"""select e.result_id, e.event_count from event_execution e {tree_results}
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
    nodes = np.empty(0, dtype=np.int64)
    for result, _ in results:
        if tree_mode:
            nodes = (
                db.db.execute(
                    """select node_index from cluster_tree_node
                where experiment_result_id = ? order by node_index""",
                    [result],
                )
                .fetchnumpy()["node_index"]
                .astype(np.int64, copy=False)
            )
        if not _stream_result(archive, result, nodes, tree_mode, retained, histories):
            return None
    if not retained:
        return None

    rows = pa.Table.from_batches(retained)
    queries = InMemoryEventQueries(db.db, rows)
    query = (
        queries.tree_events_query(run_id=experiment, parent=cluster)
        if tree_mode
        else queries.events_query(run_id=experiment, parent=cluster)
    )
    events = query.select(f"* exclude ({MOTION_COLUMNS})").pl()
    # The projection can reorder rows during its joins. Match the scanner's
    # order before attaching positional columns and fall back for changed keys.
    keys = pl.from_arrow(rows.select(EVENT_KEY), rechunk=False)
    if not events.select(*EVENT_KEY).equals(keys):
        events = events.sort(*EVENT_KEY)
        if not events.select(*EVENT_KEY).equals(keys):
            return None
    events = events.hstack(
        pl.from_arrow(rows.select(POPULATION_COLUMNS), rechunk=False)
    )
    summaries = pl.from_arrow(pa.Table.from_batches(histories))
    if tree_mode:
        history_index = events.get_column("realization_id").rle_id()
        events = events.with_columns(
            (
                summaries.get_column("history_continuous").gather(history_index)
                & (pl.col("provenance") == "persisted").all().over("realization_id")
            ).alias("history_valid")
        )
    return events, summaries.select(
        pl.concat_str(*HISTORY_KEY, separator=":").alias("id"),
        pl.col("result_id").alias("experiment_result_id"),
        *SUMMARY_COLUMNS,
    )
