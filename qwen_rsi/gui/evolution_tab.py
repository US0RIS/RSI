from __future__ import annotations

import difflib
import time

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QFont, QFontDatabase
from PySide6.QtWidgets import (QAbstractItemView, QHBoxLayout, QHeaderView, QLabel, QMessageBox, QPlainTextEdit,
                               QPushButton, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget,
                               QVBoxLayout, QWidget)

from ..archive import GENOME_FILES, Archive
from ..config import Settings
from .score_chart import ScoreChart
from .workers import EvolutionWorker

COLUMNS = ["Variant", "Parent", "Gen", "Status", "Smoke", "Dev", "Hold-out", "Improver", "Kids", "Tokens", "Role"]


def _fmt(x: float | None) -> str:
    return "–" if x is None else f"{x:.3f}"


class EvolutionTab(QWidget):
    notify = Signal(str, str)   # title, message (for tray notifications)

    def __init__(self, settings: Settings, archive: Archive):
        super().__init__()
        self.s = settings
        self.archive = archive
        self.worker: EvolutionWorker | None = None
        self.other_busy = lambda: False   # set by the main window (RSI loop running?)
        self._dirty = True

        mono = QFontDatabase.systemFont(QFontDatabase.FixedFont)

        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.gens = QSpinBox()
        self.gens.setRange(0, 10000)
        self.gens.setValue(0)
        self.gens.setSpecialValueText("∞")
        self.gens.setToolTip("Generations to run (∞ = until stopped)")
        self.approve_btn = QPushButton("Approve as champion")
        self.approve_btn.setToolTip("Promote the selected variant. Also used to roll back to an older variant.")
        self.status = QLabel("Idle")
        self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)

        top = QHBoxLayout()
        top.addWidget(self.start_btn)
        top.addWidget(self.stop_btn)
        top.addWidget(QLabel("Generations"))
        top.addWidget(self.gens)
        top.addSpacing(16)
        top.addWidget(self.approve_btn)
        top.addSpacing(16)
        top.addWidget(self.status, 1)

        self.chart = ScoreChart()

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setToolTip("Improver = fraction of this variant's children that beat it (how good its improve() is).")

        self.rationale = QPlainTextEdit(readOnly=True)
        self.diff = QPlainTextEdit(readOnly=True)
        self.diff.setFont(mono)
        self.diff.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.files = QPlainTextEdit(readOnly=True)
        self.files.setFont(mono)
        self.files.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.results = QPlainTextEdit(readOnly=True)
        self.results.setFont(mono)
        self.detail = QTabWidget()
        self.detail.addTab(self.rationale, "Rationale")
        self.detail.addTab(self.diff, "Diff vs parent")
        self.detail.addTab(self.files, "Files")
        self.detail.addTab(self.results, "Dev results")

        self.log = QPlainTextEdit(readOnly=True)
        self.log.setFont(mono)
        self.log.setMaximumBlockCount(5000)

        mid = QSplitter(Qt.Horizontal)
        mid.addWidget(self.table)
        mid.addWidget(self.detail)
        mid.setSizes([620, 480])
        outer = QSplitter(Qt.Vertical)
        outer.addWidget(self.chart)
        outer.addWidget(mid)
        outer.addWidget(self.log)
        outer.setSizes([170, 420, 160])

        lay = QVBoxLayout(self)
        lay.addLayout(top)
        lay.addWidget(outer, 1)

        self.start_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self.stop)
        self.approve_btn.clicked.connect(self.approve_selected)
        self.table.itemSelectionChanged.connect(self.show_selected)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._maybe_refresh)
        self.timer.start(1000)
        for ts, kind, msg in self.archive.recent_events(300):
            self.log.appendPlainText(f"{time.strftime('%H:%M:%S', time.localtime(ts))} [{kind}] {msg}")
        self.refresh()

    # ------------------------------------------------------------------ loop
    def start(self) -> None:
        if self.worker is not None:
            return
        if self.other_busy():
            QMessageBox.information(self, "Busy", "The RSI loop is running; it already includes agent evolution.")
            return
        self.worker = EvolutionWorker(self.s, self.archive, self.gens.value() or None)
        self.worker.event.connect(self.on_event)
        self.worker.stopped.connect(self.on_stopped)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status.setText("Running…")
        self.worker.start()

    def stop(self) -> None:
        if self.worker:
            self.status.setText("Stopping (current sandbox will be killed)…")
            self.worker.request_stop()

    def on_stopped(self, reason: str) -> None:
        self.status.setText(f"Idle: {reason}")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        if self.worker:
            self.worker.wait(2000)
        self.worker = None
        self._dirty = True
        if reason.startswith("Ollama problem"):
            self.notify.emit("Evolution paused", reason)

    def on_event(self, kind: str, msg: str) -> None:
        self.log.appendPlainText(f"{time.strftime('%H:%M:%S')} [{kind}] {msg}")
        if kind != "genome":
            self.status.setText(msg[:160])
            self._dirty = True
        if kind == "approval":
            self.notify.emit("New champion candidate", msg)
        elif kind == "champion":
            self.notify.emit("Champion updated", msg)

    # ----------------------------------------------------------------- table
    def _maybe_refresh(self) -> None:
        if self._dirty:
            self._dirty = False
            self.refresh()

    def refresh(self) -> None:
        variants = self.archive.all()
        self.chart.set_variants(variants)
        selected = self.selected_id()
        self.table.setRowCount(len(variants))
        highlight = self.palette().highlight().color()
        highlight.setAlpha(60)
        for row, v in enumerate(reversed(variants)):
            prod = self.archive.productivity(v.id)
            kids = len(self.archive.children(v.id))
            role = "★ champion" if v.is_champion else ("awaiting approval" if v.awaiting_approval else "")
            vals = [v.id, v.parent_id or "–", str(v.generation), v.status, _fmt(v.smoke), _fmt(v.dev),
                    _fmt(v.holdout), "–" if prod is None else f"{prod:.0%}", str(kids),
                    f"{(v.prompt_tokens + v.output_tokens) / 1000:.0f}k", role]
            for col, text in enumerate(vals):
                item = QTableWidgetItem(text)
                if col in (2, 4, 5, 6, 7, 8, 9):
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if v.is_champion or v.awaiting_approval:
                    item.setBackground(QBrush(highlight))
                if v.error and col == 3:
                    item.setToolTip(v.error[:1000])
                self.table.setItem(row, col, item)
        if selected:
            for row in range(self.table.rowCount()):
                if self.table.item(row, 0).text() == selected:
                    self.table.selectRow(row)
                    break

    def selected_id(self) -> str | None:
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        return self.table.item(rows[0].row(), 0).text() if rows else None

    def show_selected(self) -> None:
        vid = self.selected_id()
        if not vid:
            return
        v = self.archive.get(vid)
        head = f"{v.id}  (parent {v.parent_id or '–'}, generation {v.generation}, {v.status})\n"
        if v.error:
            head += f"\nError: {v.error}\n"
        self.rationale.setPlainText(head + "\n" + (v.rationale or "(no rationale)"))
        files = self.archive.read_files(vid)
        self.files.setPlainText("\n\n".join(f"===== {n} =====\n{files[n]}" for n in GENOME_FILES if n in files))
        if v.parent_id:
            pf = self.archive.read_files(v.parent_id)
            chunks = []
            for n in GENOME_FILES:
                a, b = pf.get(n, "").splitlines(True), files.get(n, "").splitlines(True)
                d = list(difflib.unified_diff(a, b, f"{v.parent_id}/{n}", f"{vid}/{n}"))
                if d:
                    chunks.append("".join(d))
            self.diff.setPlainText("\n".join(chunks) or "(identical to parent)")
        else:
            self.diff.setPlainText("(seed genome: no parent)")
        lines = []
        for e in sorted(self.archive.evals(vid), key=lambda e: (e["split"], e["task_id"])):
            if e["split"] == "holdout":
                continue  # keep hold-out details out of sight, even for the human, to avoid hand-tuning to it
            lines.append(f"{e['task_id']:<24} {e['passed']:>2}/{e['total']:<2} {e['llm_calls']:>2} calls "
                         f"{e['seconds']:>6.1f}s  {e['error'][:80]}")
            for f in e["failures"][:3]:
                lines.append(f"    - {f[:160]}")
        self.results.setPlainText("\n".join(lines) or "(not evaluated)")

    def approve_selected(self) -> None:
        vid = self.selected_id()
        if not vid:
            return
        v = self.archive.get(vid)
        if v.status != "evaluated":
            QMessageBox.warning(self, "Cannot promote", f"{vid} is '{v.status}', only evaluated variants can lead.")
            return
        champ = self.archive.champion()
        msg = (f"Make {vid} the champion?\n\nDev {_fmt(v.dev)}, hold-out {_fmt(v.holdout)}"
               + (f"\nCurrent champion {champ.id}: dev {_fmt(champ.dev)}, hold-out {_fmt(champ.holdout)}" if champ else "")
               + "\n\nReview the 'Diff vs parent' tab before approving.")
        if QMessageBox.question(self, "Approve champion", msg) == QMessageBox.Yes:
            self.archive.set_champion(vid)
            self.archive.log("champion", f"{vid} approved as champion by user")
            self.on_event("champion", f"{vid} approved as champion by user")

    def shutdown(self) -> None:
        if self.worker:
            self.worker.request_stop()
            self.worker.wait(10000)
