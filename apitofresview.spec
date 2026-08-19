# PyInstaller spec for the APiToF result viewer desktop app.
#
# onedir, not onefile: the bundle carries duckdb, bokeh, panel, holoviews and
# matplotlib's mpl-data, so onefile would re-extract several hundred megabytes
# to a temp directory on every single launch. onedir is also the only sane
# basis for a macOS .app bundle.
#
# Build with:  uv run pyinstaller apitofresview.spec --noconfirm --clean

import sys

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_submodules,
    copy_metadata,
)

IS_MACOS = sys.platform == "darwin"
IS_LINUX = sys.platform.startswith("linux")

datas = []
# templates/ and static/ live inside the package and are found through
# importlib.resources, so they must land at apitofresview/... in the bundle.
datas += collect_data_files("apitofresview")
# mplbed reads webaggext.js through importlib.resources.
datas += collect_data_files("mplbed")
# mplbed serves matplotlib's backends/web_backend and mpl-data as static dirs.
datas += collect_data_files("matplotlib")
# holoviews bundles logo/font data and its package-data dirs.
datas += collect_data_files("holoviews")
# bokeh and panel have their own hooks (collect data + metadata), but the
# metadata is still needed by the lazy __version__ lookups.
datas += copy_metadata("bokeh")
datas += copy_metadata("panel")
datas += copy_metadata("holoviews")
datas += copy_metadata("matplotlib")
datas += copy_metadata("mplbed")

hiddenimports = []
# bokeh selects its server extensions and command subcommands by string, and
# panel/holoviews lazily import most of their machinery at runtime.
hiddenimports += collect_submodules("bokeh")
hiddenimports += collect_submodules("panel")
# holoviews.tests.plotting.plotly calls pytest.importorskip("plotly") at import
# time, and holoviews.plotting.plotly imports plotly at module level; both abort
# submodule collection. Tests are never needed in the bundle anyway, and we
# deliberately don't ship the plotly backend.
hiddenimports += collect_submodules(
    "holoviews",
    filter=lambda name: not name.startswith(
        ("holoviews.tests", "holoviews.plotting.plotly")
    ),
)
# uvicorn picks its protocol/loop/lifespan implementations by string.
hiddenimports += collect_submodules("uvicorn")
hiddenimports += [
    # Selected via matplotlib.use("module://mplbed.webaggext._impl").
    "mplbed.webaggext._impl",
    "mplbed.integration.starlette",
    "matplotlib.backends.backend_webagg_core",
    "matplotlib.backends.backend_agg",
    "duckdb",
    "websockets",
    "anyio._backends._asyncio",
    "encodings.idna",
]

excludes = [
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
]

if IS_LINUX:
    # No native window on Linux; see apitofresview/desktop.py.
    excludes += ["webview", "gi", "pythonnet", "clr"]
else:
    # PyInstaller pulls in every pywebview backend it can find, used or not.
    excludes += ["webview.platforms.gtk", "webview.platforms.qt", "webview.platforms.android"]
    if IS_MACOS:
        hiddenimports += ["webview.platforms.cocoa"]
    else:
        hiddenimports += ["webview.platforms.edgechromium", "webview.platforms.winforms"]


a = Analysis(
    ["src/apitofresview/__main__.py"],
    pathex=["src"],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=["rthook_mplconfig.py"],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="apitofresview",
    debug=False,
    strip=False,
    upx=False,
    # Keep the console for now: while this packaging is new, a crash with no
    # console is an invisible flash. Revisit once builds are reliably green.
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="apitofresview",
)

if IS_MACOS:
    app = BUNDLE(
        coll,
        name="APiToF Result Viewer.app",
        bundle_identifier="fi.helsinki.vilma.apitofsimresultviewer",
        info_plist={
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "11.0",
            "CFBundleShortVersionString": "0.1.0",
        },
    )
