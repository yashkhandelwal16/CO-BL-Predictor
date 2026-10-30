import json
import os
import sys
from pathlib import Path

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QMessageBox

from src.ai.llm_client import InferenceError, LLMClient, ModelConfigurationError
from src.database.db_manager import (
    DatabaseError,
    DatabaseManager,
    default_database_path,
)
from src.prediction.bloom_rules import (
    BloomConfigurationError,
    load_bloom_config,
)
from src.ui.main_window import MainWindow
from src.utils.resource_paths import (
    application_resource_path,
    application_resource_root,
    is_packaged_application,
)


def validate_packaged_assets() -> None:
    if is_packaged_application():
        LLMClient().validate_model_asset()


def run_packaged_self_test() -> int:
    result: dict[str, str | bool] = {
        "frozen": is_packaged_application(),
        "bundle_root": str(application_resource_root()),
    }
    if not is_packaged_application():
        result["error"] = "The executable is not running from a bundled application."
        return _write_self_test_result(result, 2)
    status = 1
    try:
        client = LLMClient()
        client.validate_model_asset()
        load_bloom_config()
        database_path = default_database_path()
        if application_resource_root() in database_path.parents:
            result["error"] = "Database path is inside the resource directory."
            return _write_self_test_result(result, status)
        database = DatabaseManager(database_path)
        database.close()
        result["database_path"] = str(database_path)
        response = client.chat(
            [{"role": "user", "content": "Reply with one short greeting."}],
            max_tokens=8,
            timeout_seconds=120,
        )
        if not response.strip():
            result["error"] = "The bundled model returned an empty response."
            return _write_self_test_result(result, status)
        result["inference"] = "passed"
        status = 0
    except (
        BloomConfigurationError,
        DatabaseError,
        InferenceError,
        ModelConfigurationError,
        OSError,
    ) as error:
        result["error"] = str(error)
    return _write_self_test_result(result, status)


def _write_self_test_result(result: dict[str, str | bool], status: int) -> int:
    report_path = os.environ.get("CO_BL_SELF_TEST_REPORT")
    marker_path = Path(sys.executable).resolve().parent / ".packaged-self-test"
    if report_path is None and marker_path.is_file():
        report_path = marker_path.read_text(encoding="utf-8").strip()
    if report_path:
        try:
            path = Path(report_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        except OSError:
            return 1
    return status


def main() -> int:
    self_test_marker = Path(sys.executable).resolve().parent / ".packaged-self-test"
    if (
        "--self-test" in sys.argv
        or os.environ.get("CO_BL_SELF_TEST_REPORT")
        or self_test_marker.is_file()
    ):
        return run_packaged_self_test()

    app = QApplication(sys.argv)
    app.setApplicationName("CO & BL Predictor")
    app.setOrganizationName("CO & BL Predictor")
    app.setApplicationVersion("1.0.0")
    icon_path = application_resource_path("CO-BL-Predictor.ico")
    if icon_path.is_file():
        app.setWindowIcon(QIcon(str(icon_path)))

    try:
        validate_packaged_assets()
    except (InferenceError, ModelConfigurationError):
        QMessageBox.critical(
            None,
            "Unable to open the application",
            "Required application files could not be verified. Reinstall the "
            "application and try again.",
        )
        return 1

    try:
        window = MainWindow(database_path=default_database_path())
    except BloomConfigurationError:
        QMessageBox.critical(
            None,
            "Unable to open the application",
            "The classification rules could not be loaded. Check the rules "
            "configuration and restart the application.",
        )
        return 1
    except DatabaseError:
        QMessageBox.critical(
            None,
            "Unable to open the application",
            "Your local data could not be opened. Check that your user data "
            "folder is available and writable.",
        )
        return 1
    app.aboutToQuit.connect(window.close_storage)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
