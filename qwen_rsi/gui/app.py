"""Main window and application bootstrap (Windows 11 look and integration)."""
from __future__ import annotations

import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (QApplication, QLabel, QMainWindow, QStyleFactory, QSystemTrayIcon, QTabWidget)

from ..archive import Archive
from ..config import APP_NAME, Settings, data_dir
from .chat_tab import ChatTab
from .evolution_tab import EvolutionTab
from .rsi_tab import RSITab
from .settings_tab import SettingsTab
from .workers import StatusPoller


def make_icon() -> QIcon:
    """Drawn at runtime so the repository carries no binary assets."""
    icon = QIcon()
    for size in (16, 24, 32, 48, 64, 128, 256):
        pm = QPixmap(size, size)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)
        p.setBrush(QColor("#5b5bd6"))
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(0, 0, size, size, size * 0.22, size * 0.22)
        p.setPen(QColor("white"))
        f = QFont("Segoe UI", max(6, int(size * 0.42)))
        f.setBold(True)
        p.setFont(f)
        p.drawText(pm.rect(), Qt.AlignCenter, "Q↻")
        p.end()
        icon.addPixmap(pm)
    return icon


class MainWindow(QMainWindow):
    def __init__(self, settings: Settings):
        super().__init__()
        self.s = settings
        self.archive = Archive(data_dir())
        self.setWindowTitle(f"Qwen RSI · {settings.model}")
        self.resize(1280, 860)

        self.chat = ChatTab(settings, self.archive)
        self.rsi = RSITab(settings, self.archive)
        self.evo = EvolutionTab(settings, self.archive)
        self.rsi.other_busy = lambda: self.evo.worker is not None
        self.evo.other_busy = lambda: self.rsi.worker is not None
        self.settings_tab = SettingsTab(settings)
        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        tabs.addTab(self.rsi, "RSI")
        tabs.addTab(self.chat, "Chat")
        tabs.addTab(self.evo, "Agent evolution")
        tabs.addTab(self.settings_tab, "Settings")
        self.setCentralWidget(tabs)

        self.conn = QLabel("Checking Ollama…")
        self.statusBar().addPermanentWidget(self.conn)

        self.tray = QSystemTrayIcon(self.windowIcon(), self) if QSystemTrayIcon.isSystemTrayAvailable() else None
        if self.tray:
            self.tray.setToolTip(APP_NAME)
            self.tray.show()
        self.evo.notify.connect(self.toast)
        self.rsi.notify.connect(self.toast)

        self.poller = StatusPoller(settings)
        self.poller.status.connect(self.on_status)
        self.poller.start()

    def on_status(self, ok: bool, text: str) -> None:
        self.setWindowTitle(f"Qwen RSI · {self.s.model}")
        self.conn.setText(("● " if ok else "○ ") + text)
        self.conn.setStyleSheet("" if ok else "color: #c42b1c;")

    def toast(self, title: str, message: str) -> None:
        if self.tray:
            self.tray.showMessage(title, message[:250], QSystemTrayIcon.Information, 8000)
        self.statusBar().showMessage(f"{title}: {message[:150]}", 15000)

    def closeEvent(self, ev) -> None:
        self.poller.halt()
        self.chat.shutdown()
        self.rsi.shutdown()
        self.evo.shutdown()
        self.poller.wait(2000)
        self.archive.close()
        super().closeEvent(ev)


def _windows_integration() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes
        # Own taskbar group and icon instead of python.exe's.
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("QwenRSI.Desktop.1")
    except (AttributeError, OSError):
        pass


def run(argv: list[str] | None = None) -> int:
    _windows_integration()
    app = QApplication(argv or sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(APP_NAME)
    styles = {k.lower(): k for k in QStyleFactory.keys()}
    # Qt >= 6.7 ships a native Windows 11 style (rounded controls, Fluent
    # metrics, follows the system light/dark setting).
    app.setStyle(styles.get("windows11") or styles.get("windowsvista") or "Fusion")
    icon = make_icon()
    app.setWindowIcon(icon)
    win = MainWindow(Settings.load())
    win.setWindowIcon(icon)
    win.show()
    return app.exec()
