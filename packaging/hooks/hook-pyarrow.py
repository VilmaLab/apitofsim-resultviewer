"""Keep the local Arrow, IPC, and Parquet support in the Linux bundle."""

from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)

excludedimports = ["pyarrow.flight", "pyarrow._flight", "pyarrow._s3fs"]
unused_libraries = ("libarrow_flight.", "libarrow_python_flight.", "libarrow_s3.")
hiddenimports = collect_submodules(
    "pyarrow", filter=lambda name: "tests" not in name and name not in excludedimports
)
datas = collect_data_files(
    "pyarrow",
    excludes=[
        "include",
        "includes",
        "src",
        "tests",
        "**/*.h",
        "**/*.hpp",
        "**/*.pxd",
        "**/*.pyx",
        "**/*.pxi",
        "**/*.cmake",
        *(f"{prefix}*" for prefix in unused_libraries),
    ],
)
binaries = [
    (source, dest)
    for source, dest in collect_dynamic_libs("pyarrow")
    if not Path(source).name.startswith(unused_libraries)
]
