"""The one-button full RSI tab: weights + agent self-improvement."""
from __future__ import annotations

import json
import time
import traceback

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QBrush, QFont, QFontDatabase
from PySide6.QtWidgets import (QAbstractItemView, QHBoxLayout, QHeaderView, QLabel, QMessageBox, QPlainTextEdit,
                               QPushButton, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from ..archive import Archive
from ..config import Settings
from ..evolve import InfraError
from ..selftrain import recipe as R
from ..selftrain.engine import RSIEngine
from ..selftrain.trainer import TrainerBridge

COLUMNS = ["Model", "Parent", "Round", "Status", "Examples", "Pass rate", "Loss", "Dev", "Hold-out", "Audit", ""]

EXPLAINER = (
    "Each round: (1) the agent rewrites its own code and prompts, (2) the current model designs its own training "
    "recipe from past results, (3) writes new coding problems (kept only if verified), (4) solves them, keeping "
    "only answers that pass every hidden check, (5) is fine-tuned on those answers (QLoRA), (6) the new weights "
    "replace the old ones only if they score higher on hold-out tasks they have never seen. Improved weights then "
    "run every later step, so better models write better data and recipes."
)


class RSIWorker(QThread):
    event = Signal(str, str)
    stopped = Signal(str)

    def __init__(self, settings: Settings, archive: Archive, rounds: int | None):
        super().__init__()
        self.rounds = rounds
        self.engine = RSIEngine(settings, archive=archive, on_event=lambda k, m: self.event.emit(k, m))

    def run(self) -> None:
        reason = "finished"
        try:
            self.engine.run(self.rounds)
            if self.engine.stop.is_set():
                reason = "stopped by user"
        except InfraError as e:
            reason = f"Ollama problem: {e}"
        except Exception as e:  # never crash the UI
            reason = f"{type(e).__name__}: {e}"
            self.event.emit("error", traceback.format_exc()[-2000:])
        self.stopped.emit(reason)

    def request_stop(self) -> None:
        self.engine.stop.set()


class TrainerCheck(QThread):
    result = Signal(bool, str)

    def __init__(self, settings: Settings):
        super().__init__()
        self.s = settings

    def run(self) -> None:
        ok, info = TrainerBridge(self.s).check()
        if ok:
            text = f"Trainer ready: {info.get('gpu')}, {info.get('vram_gb')} GB, Unsloth {info.get('unsloth')}"
            if info.get("warning"):
                text += f" (warning: {info['warning']})"
        else:
            text = "Trainer not ready: " + (info.get("error") or info.get("unsloth_error") or info.get("torch_error")
                                            or json.dumps(info))[:300]
        self.result.emit(ok, text)


def _f(x) -> str:
    return "–" if x is None else f"{x:.3f}"


class RSITab(QWidget):
    notify = Signal(str, str)

    def __init__(self, settings: Settings, archive: Archive):
        super().__init__()
        self.s = settings
        self.archive = archive
        self.worker: RSIWorker | None = None
        self.other_busy = lambda: False   # set by the main window (scaffold-only loop running?)
        self._dirty = True
        mono = QFontDatabase.systemFont(QFontDatabase.FixedFont)

        self.go = QPushButton("▶  Start RSI")
        f = self.go.font()
        f.setPointSize(f.pointSize() + 6)
        f.setBold(True)
        self.go.setFont(f)
        self.go.setMinimumHeight(56)
        self.go.setMinimumWidth(260)
        self.go.setToolTip("Run the full loop: agent self-modification + self-generated training data + "
                           "fine-tuning + hold-out gated promotion of new weights.")
        self.rounds = QSpinBox()
        self.rounds.setRange(0, 1000)
        self.rounds.setSpecialValueText("∞")
        self.rounds.setToolTip("Rounds to run (∞ = until stopped)")
        self.check_btn = QPushButton("Check trainer")
        self.stage = QLabel("Idle")
        self.stage.setWordWrap(True)
        sf = self.stage.font()
        sf.setPointSize(sf.pointSize() + 1)
        self.stage.setFont(sf)
        self.trainer_status = QLabel("Trainer: not checked yet")
        self.trainer_status.setWordWrap(True)
        self.current = QLabel()
        explainer = QLabel(EXPLAINER)
        explainer.setWordWrap(True)
        explainer.setStyleSheet("color: palette(placeholder-text);")

        side = QVBoxLayout()
        side.addWidget(self.current)
        side.addWidget(self.stage)
        side.addWidget(self.trainer_status)
        controls = QHBoxLayout()
        controls.addWidget(self.go)
        rc = QVBoxLayout()
        rc.addWidget(QLabel("Rounds"))
        rc.addWidget(self.rounds)
        rc.addWidget(self.check_btn)
        controls.addLayout(rc)
        controls.addSpacing(12)
        controls.addLayout(side, 1)

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.detail = QPlainTextEdit(readOnly=True)
        self.detail.setFont(mono)
        self.log = QPlainTextEdit(readOnly=True)
        self.log.setFont(mono)
        self.log.setMaximumBlockCount(5000)

        mid = QSplitter(Qt.Horizontal)
        mid.addWidget(self.table)
        mid.addWidget(self.detail)
        mid.setSizes([700, 420])
        outer = QSplitter(Qt.Vertical)
        outer.addWidget(mid)
        outer.addWidget(self.log)
        outer.setSizes([380, 260])

        lay = QVBoxLayout(self)
        lay.addLayout(controls)
        lay.addWidget(explainer)
        lay.addWidget(outer, 1)

        self.go.clicked.connect(self.toggle)
        self.check_btn.clicked.connect(self.check_trainer)
        self.table.itemSelectionChanged.connect(self.show_selected)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(1000)
        self._checker: TrainerCheck | None = None
        self.refresh()

    # ----------------------------------------------------------------- run
    def toggle(self) -> None:
        if self.worker:
            self.go.setEnabled(False)
            self.go.setText("Stopping…")
            self.worker.request_stop()
            return
        if self.other_busy():
            QMessageBox.information(self, "Busy", "Stop the agent-only loop on the Self-improvement tab first.")
            return
        self.worker = RSIWorker(self.s, self.archive, self.rounds.value() or None)
        self.worker.event.connect(self.on_event)
        self.worker.stopped.connect(self.on_stopped)
        self.go.setText("■  Stop RSI")
        self.worker.start()

    def on_stopped(self, reason: str) -> None:
        if self.worker:
            self.worker.wait(5000)
        self.worker = None
        self.go.setEnabled(True)
        self.go.setText("▶  Start RSI")
        self.stage.setText(f"Idle: {reason}")
        self._dirty = True
        if reason.startswith("Ollama problem"):
            self.notify.emit("RSI paused", reason)

    def on_event(self, kind: str, msg: str) -> None:
        self.log.appendPlainText(f"{time.strftime('%H:%M:%S')} [{kind}] {msg}")
        if kind not in ("genome", "eval", "taskgen", "collect", "train"):
            self._dirty = True
        if kind == "champion" and "NEW WEIGHTS" in msg:
            self.notify.emit("New model weights promoted", msg)
        elif kind == "trainer":
            self.trainer_status.setText(msg[:300])

    def _tick(self) -> None:
        if self.worker:
            self.stage.setText(f"Stage: {self.worker.engine.stage}")
        if self._dirty:
            self._dirty = False
            self.refresh()

    def check_trainer(self) -> None:
        if self._checker:
            return
        self.trainer_status.setText("Checking trainer environment…")
        self._checker = TrainerCheck(self.s)
        self._checker.result.connect(lambda ok, text: self.trainer_status.setText(text))
        self._checker.finished.connect(lambda: setattr(self, "_checker", None))
        self._checker.start()

    # -------------------------------------------------------------- views
    def refresh(self) -> None:
        self.current.setText(f"<b>Current model:</b> {self.s.model}")
        models = self.archive.models()
        selected = self._selected()
        self.table.setRowCount(len(models))
        hl = self.palette().highlight().color()
        hl.setAlpha(60)
        for row, m in enumerate(reversed(models)):
            st = m.stats or {}
            loss = (f"{st['first_loss']:.2f}→{st['final_loss']:.2f}"
                    if st.get("first_loss") is not None and st.get("final_loss") is not None else "–")
            vals = [m.tag, m.parent_tag or "–", str(m.round), m.status, str(st.get("examples", "–")),
                    f"{st['mean_pass_rate']:.2f}" if "mean_pass_rate" in st else "–", loss,
                    _f(m.dev), _f(m.holdout), _f(m.audit), "★ current" if m.is_current else ""]
            for col, text in enumerate(vals):
                item = QTableWidgetItem(text)
                if 2 <= col <= 9:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if m.is_current:
                    item.setBackground(QBrush(hl))
                if m.error:
                    item.setToolTip(m.error[:1000])
                self.table.setItem(row, col, item)
        if selected:
            for row in range(self.table.rowCount()):
                if self.table.item(row, 0).text() == selected:
                    self.table.selectRow(row)

    def _selected(self) -> str | None:
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        return self.table.item(rows[0].row(), 0).text() if rows else None

    def show_selected(self) -> None:
        tag = self._selected()
        m = self.archive.get_model(tag) if tag else None
        if m is None:
            return
        parent = self.archive.get_model(m.parent_tag) if m.parent_tag else None
        lines = [f"{m.tag}  round {m.round}  status {m.status}"]
        if m.error:
            lines += ["", "Result: " + m.error]
        if m.round:
            changes = R.diff(R.clamp(parent.recipe) if parent else R.DEFAULT_RECIPE, R.clamp(m.recipe))
            lines += ["", "Recipe changes vs parent:"] + [f"  {c}" for c in changes or ["(none)"]]
            rfile = self.archive.root / "selftrain" / f"round_{m.round:03d}" / "recipe.json"
            if rfile.exists():
                try:
                    lines += ["", "Model's reasoning for this recipe:",
                              json.loads(rfile.read_text(encoding="utf-8")).get("analysis", "")]
                except (OSError, ValueError):
                    pass
        if m.stats:
            lines += ["", "Data and training:"] + [f"  {k}: {v}" for k, v in m.stats.items()]
        lines += ["", "Full recipe:", json.dumps(R.clamp(m.recipe), indent=2)]
        self.detail.setPlainText("\n".join(lines))

    def shutdown(self) -> None:
        if self.worker:
            self.worker.request_stop()
            self.worker.wait(10000)
