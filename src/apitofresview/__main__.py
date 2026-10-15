import sys

from apitofresview.cli import main  # type: ignore[reportMissingImports]

if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    sys.exit(main())
