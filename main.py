import sys

from PySide6.QtWidgets import QApplication
from PySide6.QtWidgets import QMessageBox

from src.prediction.bloom_rules import BloomConfigurationError
from src.ui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("CO & BL Predictor")
    app.setOrganizationName("CO & BL Predictor")

    try:
        window = MainWindow()
    except BloomConfigurationError:
        QMessageBox.critical(
            None,
            "Unable to open the application",
            "The classification rules could not be loaded. Check the rules "
            "configuration and restart the application.",
        )
        return 1
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
