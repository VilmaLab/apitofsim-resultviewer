"""Read small history selections from the matching Parquet row groups."""

from bisect import bisect_left
from functools import lru_cache

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from apitofsim.recording import EVENT_SCHEMA, PARQUET_VERSION
from apitofsim.workflow.db import _EventQueries


MAX_DIRECT_GROUPS = 16
MAX_DIRECT_BYTES = 128 * 1024**2


class InMemoryEventQueries(_EventQueries):
    """Apply apitofsim's event projections to a filtered archive."""

    def __init__(self, connection, rows):
        self.db = connection
        self.archive = connection.from_arrow(rows)
        # The shared projection binds motion columns dropped by the overview.
        missing = [
            f"NULL::DOUBLE AS {name}"
            for name in EVENT_SCHEMA.names
            if name not in rows.column_names
        ]
        if missing:
            self.archive = self.archive.select("*, " + ", ".join(missing))

    def _recorded_events(self, run_id, parent, result_id, *, columns=None):
        archive = (
            self.archive if columns is None else self.archive.select(", ".join(columns))
        )
        return archive, self._execution_query(run_id, parent, result_id).filter(
            "recorded"
        )


@lru_cache(maxsize=2)
def _row_group_index(path, archive_id, generation):
    # The cache retains footer metadata, without keeping the sidecar open.
    with pq.ParquetFile(path) as archive:
        metadata = archive.metadata
        columns = [
            archive.schema.names.index(k) for k in ("result_id", "realization_index")
        ]
        bounds = []
        for i in range(metadata.num_row_groups):
            group = metadata.row_group(i)
            stats = [group.column(col).statistics for col in columns]
            if any(stat is None or not stat.has_min_max for stat in stats):
                return None
            bounds.append(
                tuple(value for stat in stats for value in (stat.min, stat.max))
            )
        return metadata, bounds


def _intersects(keys, first_result, last_result, first_realization, last_realization):
    for result, realizations in keys.items():
        if first_result <= result <= last_result:
            start = bisect_left(realizations, first_realization)
            if start < len(realizations) and realizations[start] <= last_realization:
                return True
    return False


def parquet_history_queries(db, keys):
    # Use DuckDB for broad selections. Physical row-group ranges can overlap,
    # so filter the rows by the exact key before applying tree continuity.
    if not db.readonly or db.events_path is None or sum(map(len, keys.values())) > 16:
        return None
    version, identity, generation = db.db.execute(
        "select schema_version, archive_id, generation from event_recording_state"
    ).fetchone()
    if version != PARQUET_VERSION:
        return None
    # ExperimentDatabase already validated the archive and holds its shared lock.
    path = str(db.events_path)
    try:
        index = _row_group_index(path, identity, generation)
        if index is None:
            return None
        metadata, bounds = index
        groups = [i for i, limits in enumerate(bounds) if _intersects(keys, *limits)]
        if (
            len(groups) > MAX_DIRECT_GROUPS
            or sum(metadata.row_group(i).total_byte_size for i in groups)
            > MAX_DIRECT_BYTES
        ):
            return None
        with pq.ParquetFile(path, metadata=metadata) as archive:
            rows = archive.read_row_groups(groups)
    except OSError, pa.ArrowException:
        return None
    mask = None
    for result, realizations in keys.items():
        matched = pc.and_(
            pc.equal(rows["result_id"], result),
            pc.is_in(
                rows["realization_index"],
                value_set=pa.array(realizations, type=pa.int64()),
            ),
        )
        mask = matched if mask is None else pc.or_(mask, matched)
    return InMemoryEventQueries(db.db, rows.filter(mask))
