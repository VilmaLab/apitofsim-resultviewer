# PyInstaller spec for the APiToF result viewer desktop app.
# Compile the Explorer kernels before building. See README.md.
# The shared helper builds a onedir bundle and a macOS .app.

import sys
from importlib.machinery import EXTENSION_SUFFIXES
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


root = Path(SPECPATH)
kernel_dir = root / "build" / "explorer-kernels"
kernel_module = "_apitofresview_kernels"
tbb_library = {
    "linux": "libtbb.so.12",
    "darwin": "libtbb.12.dylib",
    "win32": "tbb12.dll",
}[sys.platform]
if not (
    any((kernel_dir / (kernel_module + suffix)).is_file() for suffix in EXTENSION_SUFFIXES)
    and (kernel_dir / tbb_library).is_file()
):
    raise SystemExit("Compile the Explorer kernels before bundling. See README.md.")

recipe = BundleSpec(
    app=APP,
    entrypoint="src/apitofresview/__main__.py",
    root=root,
    pathex=("src", str(kernel_dir)),
    binaries=((str(kernel_dir / tbb_library), "."),),
    datas=(("licenses/numba.txt", "licenses"), ("licenses/onetbb.txt", "licenses")),
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
        kernel_module,
        "mplbed.webaggext._impl",
        "mplbed.integration.starlette",
        "matplotlib.backends.backend_webagg_core",
        "matplotlib.backends.backend_agg",
        "duckdb",
    ),
    excludes=(
        "numba",
        "llvmlite",
        "apitofresview.plotting.explorer._beeswarm",
        "apitofresview.plotting.explorer._overview_kernel",
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
