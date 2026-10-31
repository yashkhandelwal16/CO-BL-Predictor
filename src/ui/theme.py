from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QColor,
    QIcon,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import QGraphicsDropShadowEffect, QFrame, QWidget


COLORS = {
    "background": "#F6F2FF",
    "surface": "#FFFFFF",
    "primary": "#7C3AED",
    "primary_dark": "#6D28D9",
    "lavender": "#EDE5FF",
    "pale_lavender": "#F3EDFF",
    "heading": "#172554",
    "body": "#64748B",
    "border": "#E7DFF3",
    "hover": "#F0E7FF",
}


def workspace_stylesheet() -> str:
    colors = COLORS
    return f"""
        QWidget {{
            background: transparent;
            color: {colors["heading"]};
            font-family: "Segoe UI";
            font-size: 14px;
        }}
        QWidget#workspace {{
            background: {colors["background"]};
        }}
        QWidget#sidebar {{
            background: {colors["surface"]};
            border-right: 1px solid {colors["border"]};
        }}
        QDialog {{
            background: {colors["background"]};
        }}
        QLabel#brand {{
            color: {colors["heading"]};
            font-size: 24px;
            font-weight: 700;
        }}
        QLabel#sidebarCaption {{
            color: {colors["body"]};
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 1px;
        }}
        QPushButton#navigationButton {{
            background: transparent;
            border: 0;
            border-radius: 9px;
            color: {colors["body"]};
            min-height: 44px;
            padding: 0 14px;
            text-align: left;
            spacing: 12px;
        }}
        QPushButton#navigationButton:hover {{
            background: {colors["hover"]};
            color: {colors["primary_dark"]};
        }}
        QPushButton#navigationButton:checked,
        QPushButton#navigationButton:checked:hover {{
            background: {colors["primary"]};
            color: {colors["surface"]};
            font-weight: 700;
        }}
        QLabel#pageTitle {{
            color: {colors["heading"]};
            font-size: 27px;
            font-weight: 700;
        }}
        QLabel#sectionTitle {{
            color: {colors["heading"]};
            font-size: 20px;
            font-weight: 700;
        }}
        QFrame#card {{
            background: {colors["surface"]};
            border: 1px solid {colors["border"]};
            border-radius: 16px;
        }}
        QLabel#cardTitle {{
            color: {colors["heading"]};
            font-size: 16px;
            font-weight: 700;
        }}
        QLabel#bodyText {{
            color: {colors["body"]};
            font-size: 14px;
        }}
        QLabel#mutedText {{
            color: {colors["body"]};
            font-size: 13px;
        }}
        QLabel#fieldLabel {{
            color: {colors["heading"]};
            font-weight: 600;
        }}
        QLabel#noticeText {{
            background: {colors["pale_lavender"]};
            border: 1px solid {colors["lavender"]};
            border-radius: 9px;
            color: {colors["primary_dark"]};
            padding: 10px;
        }}
        QPushButton#primaryButton, QPushButton#secondaryButton {{
            border-radius: 9px;
            font-weight: 600;
            min-height: 40px;
            padding: 0 16px;
        }}
        QPushButton#primaryButton {{
            background: {colors["primary"]};
            border: 1px solid {colors["primary"]};
            color: {colors["surface"]};
        }}
        QPushButton#primaryButton:hover:enabled {{
            background: {colors["primary_dark"]};
            border-color: {colors["primary_dark"]};
        }}
        QPushButton#primaryButton:pressed:enabled {{
            background: {colors["primary_dark"]};
        }}
        QPushButton#primaryButton:disabled {{
            background: {colors["lavender"]};
            border-color: {colors["lavender"]};
            color: {colors["body"]};
        }}
        QPushButton#secondaryButton {{
            background: {colors["surface"]};
            border: 1px solid {colors["primary"]};
            color: {colors["primary_dark"]};
        }}
        QPushButton#secondaryButton:hover:enabled {{
            background: {colors["hover"]};
        }}
        QPushButton#secondaryButton:pressed:enabled {{
            background: {colors["lavender"]};
        }}
        QComboBox, QLineEdit, QTextEdit, QPlainTextEdit {{
            background: {colors["surface"]};
            border: 1px solid {colors["border"]};
            border-radius: 8px;
            padding: 8px;
            selection-background-color: {colors["primary"]};
            selection-color: {colors["surface"]};
        }}
        QComboBox:focus, QLineEdit:focus, QTextEdit:focus,
        QPlainTextEdit:focus {{
            border: 1px solid {colors["primary"]};
        }}
        QComboBox:disabled, QLineEdit:disabled, QTextEdit:disabled,
        QPlainTextEdit:disabled {{
            color: {colors["body"]};
            background: {colors["pale_lavender"]};
        }}
        QListWidget, QTableWidget {{
            background: {colors["surface"]};
            alternate-background-color: {colors["pale_lavender"]};
            border: 1px solid {colors["border"]};
            border-radius: 10px;
            gridline-color: {colors["border"]};
            selection-background-color: {colors["lavender"]};
            selection-color: {colors["heading"]};
        }}
        QListWidget {{
            padding: 4px;
        }}
        QListWidget::item {{
            border-bottom: 1px solid {colors["border"]};
            padding: 10px 12px;
        }}
        QListWidget::item:selected {{
            background: {colors["lavender"]};
            color: {colors["primary_dark"]};
        }}
        QHeaderView::section {{
            background: {colors["pale_lavender"]};
            border: 0;
            border-bottom: 1px solid {colors["border"]};
            color: {colors["heading"]};
            font-weight: 600;
            padding: 8px;
        }}
        QTabWidget::pane {{
            background: {colors["surface"]};
            border: 1px solid {colors["border"]};
            border-radius: 8px;
        }}
        QTabBar::tab {{
            background: {colors["pale_lavender"]};
            color: {colors["body"]};
            padding: 8px 14px;
        }}
        QTabBar::tab:selected {{
            background: {colors["surface"]};
            color: {colors["primary_dark"]};
        }}
        QScrollBar:vertical {{
            background: {colors["pale_lavender"]};
            border: 0;
            border-radius: 7px;
            margin: 4px 3px 4px 3px;
            width: 16px;
        }}
        QScrollBar::handle:vertical {{
            background: {colors["primary"]};
            border: 3px solid {colors["pale_lavender"]};
            border-radius: 7px;
            min-height: 44px;
        }}
        QScrollBar::handle:vertical:hover {{
            background: {colors["primary_dark"]};
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            background: transparent;
            border: 0;
            height: 0;
        }}
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
            background: transparent;
        }}
        QScrollBar:horizontal {{
            background: {colors["pale_lavender"]};
            border: 0;
            border-radius: 7px;
            margin: 3px 4px 3px 4px;
            height: 16px;
        }}
        QScrollBar::handle:horizontal {{
            background: {colors["primary"]};
            border: 3px solid {colors["pale_lavender"]};
            border-radius: 7px;
            min-width: 44px;
        }}
        QScrollBar::handle:horizontal:hover {{
            background: {colors["primary_dark"]};
        }}
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
            background: transparent;
            border: 0;
            width: 0;
        }}
        QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
            background: transparent;
        }}
        QStatusBar {{
            background: {colors["surface"]};
            border-top: 1px solid {colors["border"]};
            color: {colors["body"]};
        }}
    """


class DecorativeWorkspace(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("workspace")

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        bounds = self.rect()
        width = bounds.width()
        height = bounds.height()
        painter.fillRect(bounds, QColor(COLORS["background"]))
        self._draw_waves(painter, width, height)
        self._draw_dots(painter, width, height)
        self._draw_graduation_cap(painter, width, height)

    @staticmethod
    def _draw_waves(painter: QPainter, width: int, height: int) -> None:
        outer_wave = QPainterPath()
        outer_wave.moveTo(width, height * 0.10)
        outer_wave.cubicTo(
            width * 0.87,
            height * 0.22,
            width * 0.88,
            height * 0.46,
            width * 0.72,
            height * 0.61,
        )
        outer_wave.cubicTo(
            width * 0.57,
            height * 0.77,
            width * 0.46,
            height * 0.92,
            width * 0.30,
            height,
        )
        outer_wave.lineTo(width, height)
        outer_wave.closeSubpath()
        gradient = QLinearGradient(
            QPointF(width * 0.84, height * 0.12),
            QPointF(width * 0.75, height),
        )
        gradient.setColorAt(0, QColor(COLORS["pale_lavender"]))
        gradient.setColorAt(0.55, QColor(COLORS["lavender"]))
        gradient.setColorAt(1, QColor(COLORS["lavender"]))
        painter.fillPath(outer_wave, gradient)

        middle_wave = QPainterPath()
        middle_wave.moveTo(width, height * 0.27)
        middle_wave.cubicTo(
            width * 0.88,
            height * 0.34,
            width * 0.85,
            height * 0.56,
            width * 0.70,
            height * 0.70,
        )
        middle_wave.cubicTo(
            width * 0.55,
            height * 0.85,
            width * 0.43,
            height * 0.95,
            width * 0.22,
            height,
        )
        middle_wave.lineTo(width, height)
        middle_wave.closeSubpath()
        gradient = QLinearGradient(
            QPointF(width * 0.82, height * 0.26),
            QPointF(width * 0.74, height),
        )
        gradient.setColorAt(0, QColor(COLORS["lavender"]))
        gradient.setColorAt(1, QColor(COLORS["lavender"]))
        painter.fillPath(middle_wave, gradient)

        inner_wave = QPainterPath()
        inner_wave.moveTo(width, height * 0.48)
        inner_wave.cubicTo(
            width * 0.88,
            height * 0.49,
            width * 0.85,
            height * 0.67,
            width * 0.71,
            height * 0.79,
        )
        inner_wave.cubicTo(
            width * 0.57,
            height * 0.90,
            width * 0.48,
            height * 0.98,
            width * 0.36,
            height,
        )
        inner_wave.lineTo(width, height)
        inner_wave.closeSubpath()
        gradient = QLinearGradient(
            QPointF(width, height * 0.46),
            QPointF(width * 0.62, height),
        )
        gradient.setColorAt(0, QColor(COLORS["lavender"]))
        accent = QColor(COLORS["primary"])
        accent.setAlpha(58)
        gradient.setColorAt(1, accent)
        painter.fillPath(inner_wave, gradient)

        lower_wave = QPainterPath()
        lower_wave.moveTo(width, height * 0.66)
        lower_wave.cubicTo(
            width * 0.87,
            height * 0.66,
            width * 0.82,
            height * 0.84,
            width * 0.68,
            height * 0.91,
        )
        lower_wave.cubicTo(
            width * 0.54,
            height * 0.98,
            width * 0.47,
            height,
            width * 0.40,
            height,
        )
        lower_wave.lineTo(width, height)
        lower_wave.closeSubpath()
        gradient = QLinearGradient(
            QPointF(width, height * 0.63),
            QPointF(width * 0.66, height),
        )
        gradient.setColorAt(0, QColor(COLORS["lavender"]))
        accent = QColor(COLORS["primary"])
        accent.setAlpha(44)
        gradient.setColorAt(1, accent)
        painter.fillPath(lower_wave, gradient)

    @staticmethod
    def _draw_dots(painter: QPainter, width: int, height: int) -> None:
        painter.save()
        painter.setPen(Qt.PenStyle.NoPen)
        dot_color = QColor(COLORS["primary"])
        dot_color.setAlpha(32)
        painter.setBrush(dot_color)
        spacing = max(22.0, min(width * 0.025, height * 0.04))
        radius = max(2.0, min(width, height) * 0.0045)
        origin_x = width * 0.49
        origin_y = height * 0.07
        for row in range(4):
            for column in range(6):
                painter.drawEllipse(
                    QPointF(
                        origin_x + column * spacing,
                        origin_y + row * spacing,
                    ),
                    radius,
                    radius,
                )
        painter.restore()

    @staticmethod
    def _draw_graduation_cap(painter: QPainter, width: int, height: int) -> None:
        side = max(92.0, min(width * 0.13, height * 0.17, 190.0))
        center_x = width * 0.79
        center_y = height * 0.115
        half = side * 0.50
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen_color = QColor(COLORS["primary"])
        pen_color.setAlpha(54)
        painter.setPen(QPen(pen_color, max(1.7, side * 0.021)))
        painter.setBrush(Qt.BrushStyle.NoBrush)

        mortarboard = QPainterPath()
        mortarboard.moveTo(center_x - half, center_y - side * 0.16)
        mortarboard.lineTo(center_x, center_y - side * 0.40)
        mortarboard.lineTo(center_x + half, center_y - side * 0.16)
        mortarboard.lineTo(center_x, center_y + side * 0.08)
        mortarboard.closeSubpath()
        painter.drawPath(mortarboard)

        cap_base = QPainterPath()
        cap_base.moveTo(center_x - side * 0.31, center_y - side * 0.05)
        cap_base.lineTo(center_x - side * 0.31, center_y + side * 0.31)
        cap_base.lineTo(center_x, center_y + side * 0.47)
        cap_base.lineTo(center_x + side * 0.31, center_y + side * 0.31)
        cap_base.lineTo(center_x + side * 0.31, center_y - side * 0.05)
        painter.drawPath(cap_base)
        painter.drawLine(
            QPointF(center_x - half, center_y - side * 0.16),
            QPointF(center_x - side * 0.56, center_y + side * 0.19),
        )
        painter.drawLine(
            QPointF(center_x + side * 0.03, center_y - side * 0.16),
            QPointF(center_x + side * 0.29, center_y - side * 0.10),
        )
        painter.drawLine(
            QPointF(center_x + side * 0.29, center_y - side * 0.10),
            QPointF(center_x + side * 0.40, center_y - side * 0.02),
        )
        painter.drawEllipse(
            QPointF(center_x + side * 0.03, center_y - side * 0.16),
            side * 0.025,
            side * 0.025,
        )
        tassel_color = QColor(COLORS["primary"])
        tassel_color.setAlpha(76)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(tassel_color)
        painter.drawPolygon(
            [
                QPointF(center_x - side * 0.56, center_y + side * 0.15),
                QPointF(center_x - side * 0.51, center_y + side * 0.21),
                QPointF(center_x - side * 0.56, center_y + side * 0.27),
                QPointF(center_x - side * 0.61, center_y + side * 0.21),
            ]
        )
        painter.restore()


def make_navigation_icon(name: str) -> QIcon:
    icon = QIcon()
    for color, mode in (
        (COLORS["body"], QIcon.Mode.Normal),
        (COLORS["surface"], QIcon.Mode.Active),
        (COLORS["surface"], QIcon.Mode.Selected),
    ):
        pixmap = QPixmap(24, 24)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(
            QPen(
                QColor(color),
                1.8,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
                Qt.PenJoinStyle.RoundJoin,
            )
        )
        painter.setBrush(Qt.BrushStyle.NoBrush)
        _paint_navigation_glyph(painter, name)
        painter.end()
        state = QIcon.State.On if mode != QIcon.Mode.Normal else QIcon.State.Off
        icon.addPixmap(pixmap, mode, state)
    return icon


def _paint_navigation_glyph(painter: QPainter, name: str) -> None:
    path = QPainterPath()
    if name == "Dashboard":
        path.moveTo(3, 10)
        path.lineTo(12, 3)
        path.lineTo(21, 10)
        path.moveTo(5.5, 9)
        path.lineTo(5.5, 20)
        path.lineTo(10, 20)
        path.lineTo(10, 14)
        path.lineTo(14, 14)
        path.lineTo(14, 20)
        path.lineTo(18.5, 20)
        path.lineTo(18.5, 9)
    elif name == "Courses":
        path.moveTo(3, 5)
        path.quadTo(8, 3, 12, 6)
        path.quadTo(16, 3, 21, 5)
        path.lineTo(21, 19)
        path.quadTo(16, 17, 12, 20)
        path.quadTo(8, 17, 3, 19)
        path.closeSubpath()
        path.moveTo(12, 6)
        path.lineTo(12, 20)
    elif name == "Syllabus":
        path.addRoundedRect(QRectF(5, 3, 14, 18), 1.5, 1.5)
        path.moveTo(8, 9)
        path.lineTo(16, 9)
        path.moveTo(8, 13)
        path.lineTo(16, 13)
        path.moveTo(8, 17)
        path.lineTo(14, 17)
    elif name == "Predict":
        path.moveTo(3, 19)
        path.lineTo(8, 13)
        path.lineTo(12, 16)
        path.lineTo(20, 7)
        path.moveTo(15, 7)
        path.lineTo(20, 7)
        path.lineTo(20, 12)
        path.moveTo(3, 21)
        path.lineTo(21, 21)
    elif name == "History":
        path.addEllipse(QRectF(3, 3, 18, 18))
        path.moveTo(12, 7)
        path.lineTo(12, 12)
        path.lineTo(16, 14)
    painter.drawPath(path)


def add_card_shadow(card: QFrame) -> None:
    shadow = QGraphicsDropShadowEffect(card)
    shadow.setBlurRadius(16)
    shadow.setOffset(0, 3)
    color = QColor(COLORS["primary_dark"])
    color.setAlpha(18)
    shadow.setColor(color)
    card.setGraphicsEffect(shadow)
