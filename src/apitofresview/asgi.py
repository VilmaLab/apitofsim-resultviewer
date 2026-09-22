from . import config
from .webapp import create_app


app = create_app()


def create_app_from_env():
    """Application factory for uvicorn's reloader.

    The reloader imports the application in a spawned subprocess, so it cannot
    receive this process's parsed arguments. Configuration and the debug flag
    travel through the environment instead.
    """
    settings = config.load()
    return create_app(settings.database, debug=config.debug_from_env())
