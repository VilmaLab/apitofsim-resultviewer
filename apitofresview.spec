# PyInstaller spec for the APiToF result viewer desktop app.
# Build with: uv run pyinstaller apitofresview.spec --noconfirm --clean
# The shared helper builds a onedir bundle and a macOS .app.

import sys
from pathlib import Path

sys.path.insert(0, "src")

from apitofresview.build import ASSETS
from apitofresview.desktop import APP
from workingtitle.desktop.freezing import (
    BundleSpec,
    PackageData,
    Submodules,
    build_bundle,
)


def keep_holoviews_module(name):
    # Tests and the unused Plotly backend abort submodule collection.
    return not name.startswith(("holoviews.tests", "holoviews.plotting.plotly"))


recipe = BundleSpec(
    app=APP,
    entrypoint="src/apitofresview/__main__.py",
    root=Path(SPECPATH),
    packages=(
        PackageData("apitofresview"),
        PackageData("apitofsim"),
        PackageData("mplbed"),
        PackageData("holoviews"),
    ),
    metadata=("bokeh", "panel", "holoviews", "mplbed"),
    submodules=(
        Submodules("bokeh"),
        Submodules("panel"),
        Submodules("holoviews", filter=keep_holoviews_module),
    ),
    hiddenimports=(
        "mplbed.webaggext._impl",
        "mplbed.integration.starlette",
        "matplotlib.backends.backend_webagg_core",
        "matplotlib.backends.backend_agg",
        "duckdb",
    ),
    excludes=(
        "tkinter",
        "PyQt5",
        "PyQt6",
        "PySide2",
        "PySide6",
        "qtpy",
        "IPython",
        "pytest",
        "notebook",
        "marimo",
        "cefpython3",
        "jnius",
        "matplotlib.backends.backend_tk",
        "matplotlib.backends._backend_tk",
        "PIL._tkinter_finder",
    ),
    matplotlib=True,
    bundle_identifier="fi.helsinki.vilma.apitofsimresultviewer",
    version="0.1.0",
)

bundle = build_bundle(recipe, globals(), assets=ASSETS)
