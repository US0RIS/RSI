"""Small dependency-free chart of variant scores over time."""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from ..archive import Variant


class ScoreChart(QWidget):
    def __init__(self):
        super().__init__()
        self.variants: list[Variant] = []
        self.setMinimumHeight(150)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

    def set_variants(self, variants: list[Variant]) -> None:
        self.variants = variants
        self.update()

    def paintEvent(self, _ev) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        pal = self.palette()
        text = pal.text().color()
        grid = QColor(text)
        grid.setAlpha(40)
        accent = pal.highlight().color()
        dev_c = QColor("#3a7bd5")
        hold_c = QColor("#2e9d5a")
        fail_c = QColor(text)
        fail_c.setAlpha(110)

        m_left, m_right, m_top, m_bot = 40, 12, 10, 22
        area = QRectF(m_left, m_top, max(10, self.width() - m_left - m_right),
                      max(10, self.height() - m_top - m_bot))
        p.setPen(QPen(grid, 1))
        for i in range(5):
            y = area.bottom() - area.height() * i / 4
            p.drawLine(QPointF(area.left(), y), QPointF(area.right(), y))
            p.setPen(text)
            p.drawText(QRectF(0, y - 8, m_left - 6, 16), Qt.AlignRight | Qt.AlignVCenter, f"{i / 4:.2f}")
            p.setPen(QPen(grid, 1))

        n = len(self.variants)
        if n == 0:
            p.setPen(text)
            p.drawText(area, Qt.AlignCenter, "No variants yet: press Start.")
            return

        def pt(i: int, score: float) -> QPointF:
            x = area.left() + (area.width() * (i + 0.5) / n)
            return QPointF(x, area.bottom() - area.height() * max(0.0, min(1.0, score)))

        # Best-so-far hold-out frontier.
        best = None
        path = QPainterPath()
        started = False
        for i, v in enumerate(self.variants):
            if v.holdout is not None and (best is None or v.holdout > best):
                best = v.holdout
            if best is not None:
                q = pt(i, best)
                if started:
                    path.lineTo(q)
                else:
                    path.moveTo(q)
                    started = True
        p.setPen(QPen(hold_c, 2))
        p.drawPath(path)

        for i, v in enumerate(self.variants):
            if v.status in ("failed", "rejected"):
                q = pt(i, v.smoke or 0.0)
                p.setPen(QPen(fail_c, 1.5))
                p.drawLine(q + QPointF(-3, -3), q + QPointF(3, 3))
                p.drawLine(q + QPointF(-3, 3), q + QPointF(3, -3))
                continue
            if v.dev is None:
                continue
            q = pt(i, v.dev)
            r = 6 if v.is_champion else 4
            p.setPen(QPen(accent if v.is_champion else dev_c, 2 if v.is_champion else 1))
            p.setBrush(accent if (v.is_champion or v.awaiting_approval) else dev_c)
            p.drawEllipse(q, r, r)

        p.setPen(text)
        legend = "● dev score   — best hold-out   × rejected/failed   large dot = champion"
        p.drawText(QRectF(area.left(), area.bottom() + 4, area.width(), 16), Qt.AlignLeft, legend)
