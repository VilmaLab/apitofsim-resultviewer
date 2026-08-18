"""Settings for the result viewer.

The viewer is read-only, so its only required setting is the path to the
APi-ToF experiment database. It can be supplied on the command line with
``--database`` or, for the current ``uvicorn main:app`` / ``$DATABASE`` way of
running, through the environment variable below.
"""

import os
from dataclasses import dataclass
from pathlib import Path

ENV_VAR = "DATABASE"


class ConfigError(Exception):
    """Raised when no usable configuration can be found."""


@dataclass(frozen=True)
class Settings:
    database: Path


def load(database=None):
    """Resolve settings from an explicit path or the environment.

    ``database`` wins over $DATABASE. Raise ConfigError when neither yields a
    value.
    """
    if database is None:
        database = os.environ.get(ENV_VAR)
    if not database:
        raise ConfigError(
            f"no database: pass --database or set ${ENV_VAR} "
            f"(the current way of running)"
        )
    return Settings(database=Path(database))
