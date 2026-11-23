"""Bundle Panel resources used by the desktop viewer."""

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

unused_bundles = (
    "abstractvtkplot",
    "aceplot",
    "deckglplot",
    "echarts",
    "perspective",
    "plotlyplot",
    "terminal",
    "vegaplot",
)
datas = collect_data_files(
    "panel",
    excludes=["**/*.map", *(f"dist/bundled/{name}" for name in unused_bundles)],
)
hiddenimports = collect_submodules("panel.models")
