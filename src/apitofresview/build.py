"""Frontend asset plan shared by startup, reload, and packaging."""

import sys
from pathlib import Path

from workingtitle.desktop import AssetPlan, NpmBuild

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _installed_assets():
    from importlib.resources import files

    return AssetPlan(
        root=Path(str(files("apitofresview"))).parent,
        source=False,
        steps=(
            NpmBuild(
                name="frontend",
                inputs=(),
                outputs=(
                    "apitofresview/static/index.js",
                    "apitofresview/static/index.css",
                ),
            ),
        ),
    )


if getattr(sys, "frozen", False) or not (PROJECT_ROOT / "package.json").is_file():
    ASSETS = _installed_assets()
else:
    ASSETS = AssetPlan(
        root=PROJECT_ROOT,
        steps=(
            NpmBuild(
                name="frontend",
                inputs=(
                    "src/js/**",
                    "src/css/**",
                    "src/apitofresview/templates/**",
                ),
                outputs=(
                    "src/apitofresview/static/index.js",
                    "src/apitofresview/static/index.css",
                ),
            ),
        ),
    )


if __name__ == "__main__":
    ASSETS.ensure_built(force="--force" in sys.argv)
