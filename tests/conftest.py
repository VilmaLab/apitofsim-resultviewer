import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest
from apitofsim.workflow.db import ExperimentDatabase


SERVER_STARTUP_TIMEOUT = 30


@pytest.fixture(scope="session")
def browser_database(tmp_path_factory):
    path = tmp_path_factory.mktemp("browser") / "test.duckdb"
    db = ExperimentDatabase(path)
    try:
        db.create_tables()
        db.db.executemany(
            "insert into cluster values (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (1, "Parent", 100.0, 1, -10.0, [1.0, 2.0, 3.0], [], "{}"),
                (2, "Product A", 40.0, 1, -4.0, [1.0, 2.0, 3.0], [], "{}"),
                (3, "Product B", 60.0, 0, -5.0, [1.0, 2.0, 3.0], [], "{}"),
            ],
        )
        db.db.execute("insert into pathway values (1, 1, 2, 3)")
        db.db.execute(
            "insert into experiment_config values (1, 'Browser test', '{}')"
        )
        db.db.execute(
            "insert into experiment_config values (2, 'Other run', '{}')"
        )
        db.db.execute(
            "insert into experiment_run values (1, 1, '{}', '2026-01-02 03:04:05')"
        )
        db.db.execute(
            "insert into experiment_run values (2, 2, '{}', '2026-01-03 03:04:05')"
        )
        db.db.execute(
            """
            insert into single_pathway_experiment_result
            values (1, 1, 1, 10, 20, 0, 4, 6, 12, 0)
            """
        )
        db.db.execute(
            """
            insert into single_pathway_experiment_result
            values (2, 2, 1, 10, 20, 0, 1, 9, 12, 0)
            """
        )
        db.refresh_views()
    finally:
        db.close()
    return path


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_until_serving(base_url, process):
    deadline = time.monotonic() + SERVER_STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stdout, _ = process.communicate()
            raise RuntimeError(f"Server exited with {process.returncode}:\n{stdout}")
        try:
            with urllib.request.urlopen(base_url, timeout=1) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            pass
        time.sleep(0.1)
    process.terminate()
    stdout, _ = process.communicate(timeout=5)
    raise TimeoutError(f"Server did not start:\n{stdout}")


@pytest.fixture(scope="session")
def live_server(browser_database):
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "apitofresview",
            "--database",
            str(browser_database),
            "--port",
            str(port),
            "--no-window",
            "--no-browser",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        _wait_until_serving(base_url, process)
        yield base_url
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
