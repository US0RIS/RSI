from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import fields

from PySide6.QtWidgets import (QCheckBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                               QMessageBox, QProgressBar, QPushButton, QScrollArea, QSpinBox, QVBoxLayout, QWidget)

from ..config import Settings, data_dir
from ..ollama_client import OllamaClient, OllamaError
from .workers import PullWorker

GROUPS = {
    "Ollama": ["ollama_url", "model", "num_ctx", "keep_alive", "request_timeout_s", "chat_think"],
    "Evolution": ["smoke_tasks", "smoke_min_frac", "holdout_margin", "require_approval", "selection_lambda",
                  "selection_midpoint"],
    "Self-training (RSI button)": [
        "base_model", "hf_base_model", "trainer_python", "ollama_exe", "rsi_scaffold_generations",
        "rsi_auto_promote", "rsi_model_margin", "rsi_dev_tolerance", "rsi_min_examples", "rsi_train_timeout_h",
        "rsi_keep_failed_models"],
    "Budgets and sandbox (enforced by the supervisor)": [
        "llm_calls_per_task", "llm_calls_per_improve", "exec_calls_per_task", "task_timeout_s",
        "improve_timeout_s", "exec_timeout_s", "sandbox_memory_mb"],
}
LABELS = {
    "ollama_url": "Ollama URL", "model": "Model", "num_ctx": "Context window (tokens)", "keep_alive": "Keep alive",
    "request_timeout_s": "Request timeout (s)", "chat_think": "Chat thinking level",
    "smoke_tasks": "Smoke-test tasks", "smoke_min_frac": "Smoke pass ratio vs parent",
    "holdout_margin": "Promotion margin (hold-out)", "require_approval": "Require my approval for new champions",
    "selection_lambda": "Selection sharpness λ", "selection_midpoint": "Selection midpoint",
    "llm_calls_per_task": "LLM calls per task", "llm_calls_per_improve": "LLM calls per improve()",
    "exec_calls_per_task": "Code runs per task", "task_timeout_s": "Task timeout (s)",
    "improve_timeout_s": "improve() timeout (s)", "exec_timeout_s": "Code run timeout (s)",
    "sandbox_memory_mb": "Sandbox memory (MB)",
    "base_model": "Ollama base model", "hf_base_model": "Training weights (Hugging Face id)",
    "trainer_python": "Trainer Python (blank = default venv)", "ollama_exe": "ollama executable",
    "rsi_scaffold_generations": "Agent generations per round", "rsi_auto_promote": "Promote automatically when gates pass",
    "rsi_model_margin": "New weights: hold-out margin", "rsi_dev_tolerance": "New weights: max dev drop",
    "rsi_min_examples": "Min. verified examples to train", "rsi_train_timeout_h": "Training timeout (h)",
    "rsi_keep_failed_models": "Keep rejected models in Ollama",
}


class SettingsTab(QWidget):
    def __init__(self, settings: Settings):
        super().__init__()
        self.s = settings
        self.widgets: dict[str, QWidget] = {}
        types = {f.name: f.type for f in fields(Settings)}

        inner = QWidget()
        v = QVBoxLayout(inner)
        for title, names in GROUPS.items():
            box = QGroupBox(title)
            form = QFormLayout(box)
            for name in names:
                val = getattr(settings, name)
                t = str(types[name])
                if t == "bool":
                    w = QCheckBox()
                    w.setChecked(bool(val))
                elif t == "int":
                    w = QSpinBox()
                    w.setRange(0, 10_000_000)
                    w.setValue(int(val))
                elif t == "float":
                    w = QDoubleSpinBox()
                    w.setRange(0, 100_000)
                    w.setDecimals(3)
                    w.setValue(float(val))
                else:
                    w = QLineEdit(str(val))
                self.widgets[name] = w
                form.addRow(LABELS.get(name, name), w)
            v.addWidget(box)

        self.save_btn = QPushButton("Save")
        self.test_btn = QPushButton("Test connection")
        self.pull_btn = QPushButton("Pull model")
        self.folder_btn = QPushButton("Open data folder")
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.info = QLabel(f"Data folder: {data_dir()}\nAdd your own tasks as JSON files in the 'tasks' sub-folder.")
        self.info.setWordWrap(True)
        row = QHBoxLayout()
        for b in (self.save_btn, self.test_btn, self.pull_btn, self.folder_btn):
            row.addWidget(b)
        row.addStretch(1)
        v.addLayout(row)
        v.addWidget(self.progress)
        v.addWidget(self.info)
        v.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)
        QVBoxLayout(self).addWidget(scroll)

        self.save_btn.clicked.connect(self.save)
        self.test_btn.clicked.connect(self.test)
        self.pull_btn.clicked.connect(self.pull)
        self.folder_btn.clicked.connect(self.open_folder)
        self.pull_worker: PullWorker | None = None

    def apply(self) -> None:
        for name, w in self.widgets.items():
            if isinstance(w, QCheckBox):
                val = w.isChecked()
            elif isinstance(w, (QSpinBox, QDoubleSpinBox)):
                val = w.value()
            else:
                val = w.text().strip()
            setattr(self.s, name, val)

    def save(self) -> None:
        self.apply()
        self.s.save()
        self.info.setText(f"Saved to {Settings.path()}. Changes apply to the next chat message / generation.")

    def test(self) -> None:
        self.apply()
        c = OllamaClient(self.s.ollama_url, 10)
        try:
            ver = c.version()
            if not c.has_model(self.s.model):
                self.info.setText(f"Ollama {ver} is running, but {self.s.model} is not pulled. Click 'Pull model'.")
                return
            caps = c.capabilities(self.s.model)
            ctx = c.context_length(self.s.model)
            self.info.setText(f"Ollama {ver} OK. {self.s.model}: capabilities {', '.join(caps)}; "
                              f"max context {ctx}.")
        except OllamaError as e:
            self.info.setText(f"{e}\nIs Ollama running? Install it with: winget install Ollama.Ollama")

    def pull(self) -> None:
        if self.pull_worker:
            self.pull_worker.stop_event.set()
            return
        self.apply()
        self.pull_worker = PullWorker(self.s)
        self.pull_worker.progress.connect(self._pull_progress)
        self.pull_worker.done.connect(self._pull_done)
        self.progress.setVisible(True)
        self.pull_btn.setText("Cancel pull")
        self.pull_worker.start()

    def _pull_progress(self, status: str, completed: int, total: int) -> None:
        if total:
            self.progress.setRange(0, 1000)
            self.progress.setValue(int(1000 * completed / total))
            self.progress.setFormat(f"{status}: {completed / 1e9:.2f} / {total / 1e9:.2f} GB")
        else:
            self.progress.setRange(0, 0)
            self.progress.setFormat(status)

    def _pull_done(self, err: str) -> None:
        self.pull_worker = None
        self.pull_btn.setText("Pull model")
        self.progress.setVisible(False)
        self.info.setText(f"Pull failed: {err}" if err else f"{self.s.model} is ready.")
        if err and err != "cancelled":
            QMessageBox.warning(self, "Pull failed", err)

    def open_folder(self) -> None:
        path = str(data_dir())
        if sys.platform == "win32":
            os.startfile(path)  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", path])
