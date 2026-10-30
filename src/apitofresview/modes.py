"""Recorded simulation modes shared by metadata and the Explorer."""

import json

MODE_LABELS = {
    "PATHWAY_AT_A_TIME": "Pathway at a time",
    "SINGLE_CLUSTER": "Single cluster",
    "CLUSTER_TREE": "Cluster tree",
}


def simulation_mode(db, run):
    result = db.db.execute("select * from experiment_run where id = ?", [run])
    row = result.fetchone()
    if row is None:
        return None
    metadata = dict(zip((c[0] for c in result.description), row, strict=True))
    if metadata.get("simulation_mode"):
        return metadata["simulation_mode"]
    config = json.loads(metadata.get("run_config") or "{}")
    if isinstance(config, str):
        config = json.loads(config)
    return config.get("simulation_mode")


def mode_label(db, run, is_single_pathway):
    return MODE_LABELS.get(simulation_mode(db, run)) or (
        "Single pathway" if is_single_pathway else "Multi-pathway"
    )
