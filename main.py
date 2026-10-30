import sys

from PySide6.QtWidgets import QApplication, QMessageBox

from src.database.db_manager import DatabaseError, default_database_path
from src.prediction.bloom_rules import BloomConfigurationError
from src.ui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("CO & BL Predictor")
    app.setOrganizationName("CO & BL Predictor")

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
