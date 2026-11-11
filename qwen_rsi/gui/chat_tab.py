from __future__ import annotations

import html

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QKeySequence, QShortcut, QTextCursor
from PySide6.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QTextBrowser,
                               QVBoxLayout, QWidget)

from ..archive import Archive
from ..config import Settings
from .workers import ChatWorker

THINK_LEVELS = ["off", "low", "medium", "high", "max"]


class ChatTab(QWidget):
    def __init__(self, settings: Settings, archive: Archive):
        super().__init__()
        self.s = settings
        self.archive = archive
        self.history: list[dict] = []
        self.worker: ChatWorker | None = None
        self._in_thinking = False

        self.view = QTextBrowser()
        self.view.setOpenExternalLinks(True)
        self.input = QPlainTextEdit()
        self.input.setPlaceholderText("Message Qwen…  (Ctrl+Enter to send)")
        self.input.setFixedHeight(96)

        self.mode = QComboBox()
        self.mode.addItems(["Direct model (streaming)", "Champion agent (self-improved)"])
        self.mode.setToolTip("Champion agent runs your message through the current champion genome's solve().")
        self.think = QComboBox()
        self.think.addItems(THINK_LEVELS)
        self.think.setCurrentText(settings.chat_think if settings.chat_think in THINK_LEVELS else "medium")
        self.send_btn = QPushButton("Send")
        self.send_btn.setDefault(True)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.clear_btn = QPushButton("New chat")

        bar = QHBoxLayout()
        bar.addWidget(QLabel("Mode"))
        bar.addWidget(self.mode)
        bar.addWidget(QLabel("Thinking"))
        bar.addWidget(self.think)
        bar.addStretch(1)
        bar.addWidget(self.clear_btn)
        bar.addWidget(self.stop_btn)
        bar.addWidget(self.send_btn)

        lay = QVBoxLayout(self)
        lay.addWidget(self.view, 1)
        lay.addWidget(self.input)
        lay.addLayout(bar)

        self.send_btn.clicked.connect(self.send)
        self.stop_btn.clicked.connect(self.stop)
        self.clear_btn.clicked.connect(self.clear)
        QShortcut(QKeySequence("Ctrl+Return"), self.input, activated=self.send)
        self.mode.currentIndexChanged.connect(lambda i: self.think.setEnabled(i == 0))
        self.render()

    # ------------------------------------------------------------------ view
    def render(self) -> None:
        parts = []
        for m in self.history:
            who = "You" if m["role"] == "user" else "Qwen"
            if m.get("thinking"):
                parts.append(f"**{who} (thinking)**\n\n> " + m["thinking"].strip().replace("\n", "\n> "))
            parts.append(f"**{who}**\n\n{m['content']}")
        if not parts:
            parts.append(f"*Chat with `{self.s.model}`. Switch the mode to talk to the current self-improved "
                         f"champion agent instead of the raw model.*")
        self.view.setMarkdown("\n\n---\n\n".join(parts))
        self.view.moveCursor(QTextCursor.End)

    def _append(self, kind: str, text: str) -> None:
        cur = self.view.textCursor()
        cur.movePosition(QTextCursor.End)
        fmt = cur.charFormat()
        if kind == "thinking":
            fmt.setForeground(self.palette().placeholderText())
            fmt.setFontItalic(True)
        else:
            if self._in_thinking:
                cur.insertText("\n\n")
            fmt.setForeground(self.palette().text())
            fmt.setFontItalic(False)
        self._in_thinking = kind == "thinking"
        cur.insertText(text, fmt)
        self.view.setTextCursor(cur)
        self.view.ensureCursorVisible()

    # --------------------------------------------------------------- actions
    def send(self) -> None:
        text = self.input.toPlainText().strip()
        if not text or self.worker is not None:
            return
        self.input.clear()
        self.history.append({"role": "user", "content": text})
        self.render()
        self.view.append("")
        cur = self.view.textCursor()
        cur.movePosition(QTextCursor.End)
        cur.insertHtml("<p><b>Qwen</b></p>")
        self._in_thinking = False
        msgs = [{"role": m["role"], "content": m["content"]} for m in self.history]
        champion = self.mode.currentIndex() == 1
        if champion and self.archive.champion() is None:
            self.history.append({"role": "assistant", "content": "*No champion yet: start the evolution loop first.*"})
            self.render()
            return
        self.worker = ChatWorker(self.s, msgs, self.s.think_value(self.think.currentText()),
                                 champion_archive=self.archive if champion else None)
        self.worker.token.connect(self._append)
        self.worker.finished_ok.connect(self._done)
        self.worker.failed.connect(self._failed)
        self.worker.finished.connect(self._cleanup)
        self.send_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.worker.start()

    def _done(self, thinking: str, content: str) -> None:
        self.history.append({"role": "assistant", "content": content, "thinking": thinking})
        self.render()

    def _failed(self, err: str) -> None:
        self.history.append({"role": "assistant", "content": f"**Error:** {html.escape(err)}"})
        self.render()

    def _cleanup(self) -> None:
        self.worker = None
        self.send_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def stop(self) -> None:
        if self.worker:
            self.worker.request_stop()

    def clear(self) -> None:
        if self.worker is None:
            self.history.clear()
            self.render()

    def shutdown(self) -> None:
        if self.worker:
            self.worker.request_stop()
            self.worker.wait(3000)
