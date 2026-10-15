"""Result viewer identity for the shared desktop runner."""

from workingtitle.desktop import AppSpec, DesktopSession, WindowSpec

APP = AppSpec(
    name="apitofresview",
    title="APiToF Result Viewer",
    description="Read-only web viewer for APi-ToF simulation results.",
    factory="apitofresview.asgi:create_app_from_env",
    debug_env_var="APITOF_DEBUG",
    window=WindowSpec(width=1400, height=900),
    native_platforms=("win32", "darwin"),
)

SESSION = DesktopSession(APP)
