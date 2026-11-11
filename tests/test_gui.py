import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")


def _wait(app, cond, timeout=120):
    end = time.time() + timeout
    while time.time() < end and not cond():
        app.processEvents()
        time.sleep(0.02)
    return cond()


def test_main_window_runs_a_generation_and_chats(home, fake_ollama):
    _, url = fake_ollama
    from qwen_rsi.config import Settings
    from qwen_rsi.gui.app import MainWindow, make_icon

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    s = Settings(ollama_url=url, require_approval=True)
    w = MainWindow(s)
    w.setWindowIcon(make_icon())
    w.show()
    assert _wait(app, lambda: "qwen3.8:27b" in w.conn.text(), 20)

    w.evo.gens.setValue(1)
    w.evo.start()
    assert _wait(app, lambda: w.evo.worker is None, 180), w.evo.status.text()
    w.evo.refresh()
    assert w.evo.table.rowCount() == 2
    assert "awaiting approval" in [w.evo.table.item(r, 10).text() for r in range(2)]
    w.evo.table.selectRow(0)
    assert "Always check empty inputs" in w.evo.diff.toPlainText()

    w.chat.input.setPlainText("hello")
    w.chat.send()
    assert _wait(app, lambda: w.chat.worker is None, 30)
    assert w.chat.history[-1]["content"] == "Hello from the fake model."
    w.chat.mode.setCurrentIndex(1)
    w.chat.input.setPlainText("hi champion")
    w.chat.send()
    assert _wait(app, lambda: w.chat.worker is None, 30)
    assert "Hello" in w.chat.history[-1]["content"]

    w.settings_tab.test()
    assert "capabilities" in w.settings_tab.info.text()
    w.close()


def test_rsi_button_runs_a_full_round(home, fake_ollama, monkeypatch):
    import copy
    import sys

    from test_selftrain import FAKE_TRAINER, SMALL, FakeImporter
    from qwen_rsi.config import Settings
    from qwen_rsi.gui import rsi_tab
    from qwen_rsi.gui.app import MainWindow
    from qwen_rsi.ollama_client import OllamaClient
    from qwen_rsi.selftrain.engine import RSIEngine
    from qwen_rsi.selftrain.trainer import TrainerBridge

    state, url = fake_ollama
    s = Settings(ollama_url=url, rsi_scaffold_generations=0, rsi_min_examples=3, trainer_python=sys.executable)

    def factory(settings, archive, on_event):
        client = OllamaClient(url, 30)
        if archive.get_model(settings.model) is None:
            archive.add_model(settings.model, None, 0, "base", recipe=copy.deepcopy(SMALL))
        return RSIEngine(settings, archive=archive, client=client, on_event=on_event,
                         trainer=TrainerBridge(settings, script=FAKE_TRAINER),
                         importer=FakeImporter(settings, client, state))
    monkeypatch.setattr(rsi_tab, "RSIEngine", factory)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = MainWindow(s)
    w.show()
    w.rsi.rounds.setValue(1)
    w.rsi.go.click()
    assert w.rsi.worker is not None
    assert w.evo.other_busy()          # the agent-only loop refuses to start while RSI runs
    assert _wait(app, lambda: w.rsi.worker is None, 240), w.rsi.stage.text()
    w.rsi.refresh()
    tags = [w.rsi.table.item(r, 0).text() for r in range(w.rsi.table.rowCount())]
    assert "qwen3.8-rsi:r1" in tags and s.model == "qwen3.8-rsi:r1"
    w.rsi.table.selectRow(0)
    assert "Data and training" in w.rsi.detail.toPlainText()
    w.close()
