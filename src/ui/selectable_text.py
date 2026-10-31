from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QLabel,
    QListWidget,
    QMessageBox,
    QTableWidget,
    QWidget,
)


class _SelectableTextFilter(QObject):
    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.Type.Show and isinstance(watched, QWidget):
            _set_selectable_labels(watched)
            if isinstance(watched, QMessageBox):
                watched.setTextInteractionFlags(
                    Qt.TextInteractionFlag.TextSelectableByMouse
                    | Qt.TextInteractionFlag.TextSelectableByKeyboard
                )

        if event.type() != QEvent.Type.KeyPress:
            return False
        if not isinstance(event, QKeyEvent):
            return False
        if (
            event.key() != Qt.Key.Key_C
            or not event.modifiers() & Qt.KeyboardModifier.ControlModifier
        ):
            return False

        widget = watched if isinstance(watched, QWidget) else None
        while widget is not None and not isinstance(
            widget, (QListWidget, QTableWidget, QComboBox)
        ):
            widget = widget.parentWidget()
        if isinstance(widget, QListWidget):
            text = "\n".join(item.text() for item in widget.selectedItems())
        elif isinstance(widget, QTableWidget):
            items = sorted(
                widget.selectedItems(),
                key=lambda item: (item.row(), item.column()),
            )
            text = "\n".join(item.text() for item in items)
        elif isinstance(widget, QComboBox):
            text = widget.currentText()
        else:
            return False

        if text:
            QApplication.clipboard().setText(text)
            return True
        return False


_text_filter: _SelectableTextFilter | None = None


def enable_text_selection(root: QWidget) -> None:
    global _text_filter
    application = QApplication.instance()
    if application is not None and _text_filter is None:
        _text_filter = _SelectableTextFilter(application)
        application.installEventFilter(_text_filter)
    _set_selectable_labels(root)


def _set_selectable_labels(root: QWidget) -> None:
    for label in root.findChildren(QLabel):
        label.setTextInteractionFlags(
            label.textInteractionFlags()
            | Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
