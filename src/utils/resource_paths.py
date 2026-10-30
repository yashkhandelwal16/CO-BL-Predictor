import os
import sys
from pathlib import Path


def is_packaged_application() -> bool:
    return bool(
        getattr(sys, "frozen", False)
        or getattr(sys, "_MEIPASS", None)
    )


def application_resource_root() -> Path:
    """Return the read-only root for source or PyInstaller-bundled resources."""
    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root:
        return Path(bundle_root)
    return Path(__file__).resolve().parents[2]


def application_resource_path(*parts: str) -> Path:
    return application_resource_root().joinpath(*parts)


def application_data_root() -> Path:
    """Return writable per-user data outside a PyInstaller extraction folder."""
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "CO-BL-Predictor"
    if os.name == "nt":
        return Path.home() / "AppData" / "Local" / "CO-BL-Predictor"

    xdg_data_home = os.environ.get("XDG_DATA_HOME")
    data_home = (
        Path(xdg_data_home)
        if xdg_data_home
        else Path.home() / ".local" / "share"
    )
    return data_home / "CO-BL-Predictor"
