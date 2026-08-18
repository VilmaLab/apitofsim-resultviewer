"""Backwards-compatible entry point.

Keeps the previous way of running the viewer working unchanged: ``uvicorn
main:app`` with ``$DATABASE`` set, and ``import main`` in tests. The
implementation now lives in the ``apitofresview`` package.
"""

from apitofresview.webapp import _db as db  # type: ignore[reportMissingImports]
from apitofresview.webapp import (  # type: ignore[reportMissingImports]
    app,
    create_app,
    report_data,
    report_download,
)

__all__ = ["app", "create_app", "report_data", "report_download", "db"]
