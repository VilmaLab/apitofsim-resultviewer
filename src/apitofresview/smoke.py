"""Self-check used to validate frozen builds.

A plain request to `/` would pass even in a bundle where bokeh, panel,
holoviews or duckdb are unusable: the heavy machinery is only reached when an
experiment's spectrogram is opened, and those libraries resolve their
submodules lazily. This exercises those paths directly, because missing
modules are the failure mode PyInstaller actually produces.
"""

import json
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from apitofresview import desktop  # type: ignore[reportMissingImports]

HTTP_CHECKS = [
    # The overview page (served at the root) renders base.html, which needs
    # the middleware's request context for the mplbed head injection and the
    # experiment-summary query, so this covers both paths.
    ("", None),
    ("static/index.js", None),
    ("static/index.css", None),
    ("webagg/mpl.js", None),
    ("webagg/webaggext.js", None),
    ("webagg/_static/js/mpl.js", None),
]

IMPORT_CHECKS = [
    ("apitofresview.webapp", "apitofresview.webapp", "create_app"),
    ("apitofsim.plotting", "apitofsim.plotting", "get_report"),
    ("apitofsim.workflow.db", "apitofsim.workflow.db", "ExperimentDatabase"),
    ("bokeh.server.asgi", "bokeh.server.asgi", "BokehASGI"),
    ("bokeh.embed", "bokeh.embed", "server_document"),
    ("panel", "panel", "layout"),
    ("holoviews", "holoviews", "renderer"),
    ("mplbed", "mplbed", "FigureCollector"),
    ("duckdb", "duckdb", "connect"),
    (
        "uvicorn websockets protocol",
        "uvicorn.protocols.websockets.websockets_impl",
        "WebSocketProtocol",
    ),
]


def make_test_database():
    """Create a minimal, valid experiment database for the HTTP checks.

    The viewer reads the schema (tables plus the ``*_report`` views) that
    ``create_tables`` builds, so we build that once writable and hand the
    resulting file to the read-only app.
    """
    from apitofsim.workflow.db import (
        ExperimentDatabase,  # type: ignore[reportMissingImports]
    )

    tmp = tempfile.TemporaryDirectory(prefix="apitofresview-smoke-")
    path = Path(tmp.name) / "smoke.duckdb"
    db = ExperimentDatabase(path)
    try:
        db.create_tables()
        # The *_report views (cluster_report, experiment_summary, ...) are
        # materialised by refresh_views, and the report routes query them.
        db.refresh_views()
    finally:
        db.close()
    return tmp, path


def _check_http(url, failures, contains=None):
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            assert response.status == 200, response.status
            body = response.read()
            assert body, "empty body"
        if contains is not None:
            assert contains.encode() in body, f"{contains!r} missing from body"
    except Exception as err:
        failures.append(f"GET {url}: {err!r}")
        print(f"FAIL GET {url}: {err!r}", flush=True)
    else:
        print(f"OK   GET {url}", flush=True)


def _check_import(label, module, attr, failures):
    try:
        imported = __import__(module, fromlist=[attr])
        getattr(imported, attr)
    except Exception as err:
        failures.append(f"import {label}: {err!r}")
        print(f"FAIL import {label}: {err!r}", flush=True)
    else:
        print(f"OK   import {label}", flush=True)


def _check_mpl_backend(failures):
    """Render a figure through mplbed's backend.

    The backend is selected by the string "module://mplbed.webaggext._impl",
    which static analysis cannot see, so this is exactly the kind of thing a
    bundle drops.
    """
    try:
        import matplotlib
        from matplotlib.figure import Figure
        from matplotlib.pyplot import _get_backend_mod

        backend = matplotlib.get_backend()
        assert "mplbed" in backend, f"unexpected backend {backend!r}"
        figure = Figure()
        figure.gca().plot([0, 1], [0, 1])
        manager = _get_backend_mod().new_figure_manager_given_figure(id(figure), figure)
        manager.canvas.draw()
    except Exception as err:
        failures.append(f"mplbed backend: {err!r}")
        print(f"FAIL mplbed backend: {err!r}", flush=True)
    else:
        print("OK   mplbed backend", flush=True)


def _check_report_route(url, failures):
    """Fetch one report page through the running server.

    Exercises the app handlers plus duckdb's relation/limit/fetchdf path
    against the synthetic database.
    """
    try:
        with urllib.request.urlopen(
            url + "report/data?report=cluster-report&page=1&size=100", timeout=60
        ) as response:
            assert response.status == 200, response.status
            payload = json.loads(response.read())
        assert payload["last_page"] == 1
    except Exception as err:
        failures.append(f"report route: {err!r}")
        print(f"FAIL report route: {err!r}", flush=True)
    else:
        print("OK   report route", flush=True)


def run_smoke_test(sock, database_path=None, debug=False):
    from apitofresview.webapp import create_app  # type: ignore[reportMissingImports]

    # The smoke test must run on a database that exists. When the caller does
    # not supply one, build a minimal valid one on the fly so the check works
    # in CI with no fixtures.
    tmpdir = None
    if database_path is None:
        tmpdir, database_path = make_test_database()
    app = create_app(database_path, debug=debug)

    server = desktop.ServerThread(app, sock, log_level="warning").start()
    failures = []
    try:
        print(f"Serving at {server.url}", flush=True)
        for path, contains in HTTP_CHECKS:
            _check_http(server.url + path, failures, contains=contains)
        for label, module, attr in IMPORT_CHECKS:
            _check_import(label, module, attr, failures)
        _check_mpl_backend(failures)
        _check_report_route(server.url, failures)
    finally:
        server.stop()
        if tmpdir is not None:
            tmpdir.cleanup()

    if failures:
        print("\nSMOKE TEST FAILED:", flush=True)
        for failure in failures:
            print(f" - {failure}", flush=True)
        return 1
    print("\nSMOKE TEST PASSED", flush=True)
    return 0
