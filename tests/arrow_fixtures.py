"""Small, explicit IPC fixtures for query tests, independent of the simulator."""

import json

import pyarrow as pa
from apitofsim.recording import EVENT_SCHEMA


def publish_events(db, rows, *, statuses=None):
    db._release_event_reader()
    identity, generation, _ = db.db.execute(
        "select archive_id,generation,file_size from event_recording_state"
    ).fetchall()[0]
    schema = pa.schema(
        [pa.field(f.name, f.type, nullable=False) for f in EVENT_SCHEMA]
    ).with_metadata(
        {b"apitofsim:event_schema": b"1", b"apitofsim:archive_id": identity.encode()}
    )
    values = [
        {
            **{field.name: 0 for field in schema},
            "transition_index": -1,
            "next_particle_index": -1,
            **row,
        }
        for row in rows
    ]
    statuses = statuses or {}
    ids = sorted({row["result_id"] for row in values})
    metadata = {str(id): {"status": statuses.get(id, "complete")} for id in ids}
    with pa.ipc.new_file(
        str(db.events_path),
        schema,
        metadata={b"apitofsim:executions": json.dumps(metadata).encode()},
    ) as writer:
        # Scoped readers use execution batch ranges; mirror the recorder's layout.
        for id in ids:
            writer.write_batch(
                pa.RecordBatch.from_pylist(
                    [row for row in values if row["result_id"] == id], schema=schema
                )
            )
    for batch_start, id in enumerate(ids):
        db.db.execute(
            "update event_execution set recorded=true,status=?,event_count=?,"
            "batch_start=?,batch_count=1 where result_id=?",
            [
                statuses.get(id, "complete"),
                sum(row["result_id"] == id for row in values),
                batch_start,
                id,
            ],
        )
    db.db.execute(
        "update event_recording_state set generation=?,file_size=?",
        [generation + 1, db.events_path.stat().st_size],
    )
    db._ensure_event_reader()


def event(result_id, realization_index, event_index, kind, t, z, **fields):
    return {
        "result_id": result_id,
        "realization_index": realization_index,
        "event_index": event_index,
        "event_kind": (
            "init",
            "collision",
            "postcollision",
            "fragmentation",
            "postfragmentation",
            "escape",
        ).index(kind),
        "t": t,
        "z": z,
        **fields,
    }
