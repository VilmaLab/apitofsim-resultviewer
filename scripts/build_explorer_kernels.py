"""Compile the Explorer kernels with the oneTBB-enabled Numba fork."""

import argparse
import os
import shlex
import shutil
import sys
import sysconfig
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "build" / "explorer-kernels"
    )
    args = parser.parse_args()

    prefix = Path(sys.prefix)
    tbb_root = Path(
        os.environ.get("TBBROOT")
        or (prefix / "Library" if sys.platform == "win32" else prefix)
    )
    runtime, rpath = {
        "linux": ("lib/libtbb.so.12", "$ORIGIN"),
        "darwin": ("lib/libtbb.12.dylib", "@loader_path"),
        "win32": ("bin/tbb12.dll", None),
    }[sys.platform]
    tbb_runtime = tbb_root / runtime
    if not tbb_runtime.is_file():
        raise FileNotFoundError(tbb_runtime)
    if rpath:
        # Conda's Python adds its own library directory to extension rpaths.
        # Resolve the oneTBB library staged next to the extension first.
        build_rpath = f"-Wl,-rpath,{prefix / 'lib'}"
        for name in ("LDSHARED", "LDCXXSHARED"):
            flags = os.environ.get(name) or sysconfig.get_config_var(name)
            if flags:
                os.environ[name] = shlex.join(
                    flag for flag in shlex.split(flags) if flag != build_rpath
                )
        linker_flag = shlex.quote(f"-Wl,-rpath,{rpath}")
        os.environ["LDFLAGS"] = f"{os.environ.get('LDFLAGS', '')} {linker_flag}"
    os.environ["NUMBA_CPU_NAME"] = "generic"
    os.environ["NUMBA_CACHE_DIR"] = str(args.output_dir.resolve() / "numba-cache")

    from numba import types
    from numba.pycc import CC

    # Import the kernel sources without executing the Explorer package initializer.
    sys.path.insert(0, str(ROOT / "src/apitofresview/plotting/explorer"))
    from _beeswarm import envelope_edges, nearest_y, stack_swarm, terminal_swarm
    from _overview_kernel import _summarize_batch

    compiler = CC("_apitofresview_kernels")
    if not hasattr(compiler, "parallel"):
        raise RuntimeError("The oneTBB-enabled Numba fork is required")
    compiler.parallel = True
    args.output_dir.mkdir(parents=True, exist_ok=True)
    compiler.output_dir = str(args.output_dir)

    i64, f64, boolean = types.int64, types.float64, types.boolean
    floats = types.Array(f64, 1, "C", readonly=True)
    indices = types.Array(i64, 1, "C", readonly=True)
    slots = types.Array(f64, 2, "C", readonly=True)
    float_pair = types.UniTuple(f64[::1], 2)
    state = types.Tuple((i64,) * 6 + (f64,) * 3 + (boolean,))
    columns = types.Tuple((indices,) * 6 + (floats,) * 4)
    summary = types.Optional(
        types.Tuple((state, i64[::1], f64[:, ::1], i64[:, ::1], f64[::1], boolean[::1]))
    )

    compiler.export("nearest_y", f64(floats, floats, i64))(nearest_y.py_func)
    compiler.export("stack_swarm", f64[::1](floats, f64, f64))(stack_swarm.py_func)
    compiler.export(
        "terminal_swarm",
        float_pair(floats, floats, indices, indices, indices, slots, f64, f64),
    )(terminal_swarm.py_func)
    compiler.export("envelope_edges", float_pair(floats, floats, floats, f64, f64))(
        envelope_edges.py_func
    )

    # The ordered scanner is a serial helper. This also keeps the parallel
    # array analysis out of its optional tuple return.
    @compiler.export("_summarize_batch", summary(columns, indices, state, i64, boolean))
    def summarize_batch(columns, nodes, state, next_realization, tree_mode):
        return _summarize_batch(columns, nodes, state, next_realization, tree_mode)

    compiler.compile()
    shutil.copy2(tbb_runtime, args.output_dir / tbb_runtime.name)
    if sys.platform == "darwin":
        # The Conda-built extension and oneTBB both link to this dylib.
        libcxx = prefix / "lib" / "libc++.1.dylib"
        shutil.copy2(libcxx, args.output_dir / libcxx.name)


if __name__ == "__main__":
    main()
