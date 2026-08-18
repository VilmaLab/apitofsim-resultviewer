import argparse
import sys
from pathlib import Path

from apitofresview import config, desktop  # type: ignore[reportMissingImports]
from apitofresview.webapp import create_app  # type: ignore[reportMissingImports]


def build_parser():
    parser = argparse.ArgumentParser(
        prog="apitofresview",
        description="Read-only web viewer for APi-ToF simulation results.",
    )
    parser.add_argument(
        "--database",
        type=Path,
        metavar="PATH",
        help=f"Experiment database path. Overrides ${config.ENV_VAR}. "
        "Without either, the viewer cannot start.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--port", type=int, default=0, help="0 (the default) picks a free port"
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Serve only; do not open a native window or a browser",
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="Do not open the system browser"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Show tracebacks in the browser"
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Start up, self-check, and exit. Used to validate frozen builds.",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    sock = desktop.bind_socket(args.host, args.port)

    if args.smoke_test:
        # Self-contained: builds its own throwaway database, so it validates
        # the bundle without needing a real database path or $DATABASE.
        from apitofresview.smoke import (
            run_smoke_test,  # type: ignore[reportMissingImports]
        )

        return run_smoke_test(sock, args.database, args.debug)

    try:
        settings = config.load(args.database)
    except config.ConfigError as exc:
        parser.error(str(exc))

    app = create_app(settings.database, debug=args.debug)

    if args.no_window or not desktop.native_window_supported():
        desktop.run_browser(app, sock, open_browser=not args.no_browser)
        return 0

    try:
        desktop.run_window(app, sock)
    except Exception as err:
        # A missing or broken webview backend should degrade to the browser,
        # not to nothing at all.
        print(
            f"Could not open a native window ({err}); falling back to the browser.",
            file=sys.stderr,
        )
        # uvicorn closes the sockets it was handed when it shuts down, so the
        # socket run_window used is dead by now; the fallback needs a fresh one.
        sock.close()
        sock = desktop.bind_socket(args.host, args.port)
        desktop.run_browser(app, sock, open_browser=not args.no_browser)
    return 0


if __name__ == "__main__":
    sys.exit(main())
