"""Result viewer options over the shared desktop runner."""

from pathlib import Path

from workingtitle.desktop import run_cli

from apitofresview import config
from apitofresview.build import ASSETS
from apitofresview.desktop import APP, SESSION
from apitofresview.smoke import smoke_test


def add_arguments(parser):
    parser.add_argument(
        "--database",
        type=Path,
        metavar="PATH",
        help=f"Experiment database path. Overrides ${config.ENV_VAR}.",
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Serve only; do not open a native window or browser",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open the system browser",
    )


def prepare(args):
    try:
        settings = config.load(args.database)
    except config.ConfigError as exc:
        raise ValueError(str(exc)) from exc
    if args.no_window or args.no_browser:
        args.mode = "server"
    return {config.ENV_VAR: str(settings.database)}


def main(argv=None):
    return run_cli(
        APP,
        argv,
        add_arguments=add_arguments,
        prepare=prepare,
        assets=ASSETS,
        smoke_test=smoke_test,
        session=SESSION,
    )


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    raise SystemExit(main())
