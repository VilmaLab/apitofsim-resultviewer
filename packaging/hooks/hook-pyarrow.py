"""Keep local Arrow IPC and Parquet support in the desktop bundle."""

from fnmatch import fnmatchcase
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)

excludedimports = [
    "pyarrow.flight",
    "pyarrow._flight",
    "pyarrow._s3fs",
    # PyInstaller's module scan also discovers the unversioned support library.
    "pyarrow.libarrow_python_flight",
]
unused_libraries = [
    f"{prefix}{name}.*"
    for prefix in ("", "lib")
    for name in ("arrow_flight", "arrow_python_flight", "arrow_s3")
]
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
        "**/*.lib",
        *unused_libraries,
    ],
)
binaries = [
    (source, dest)
    for source, dest in collect_dynamic_libs("pyarrow")
    if not any(fnmatchcase(Path(source).name, pattern) for pattern in unused_libraries)
]
