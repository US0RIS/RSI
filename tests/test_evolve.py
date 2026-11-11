import random

import pytest

from qwen_rsi.archive import Archive
from qwen_rsi.config import Settings
from qwen_rsi.evolve import Evolver, InfraError
from qwen_rsi.ollama_client import OllamaClient


def make(home, url, **kw):
    s = Settings(ollama_url=url, **kw)
    events = []
    ev = Evolver(s, archive=Archive(home), client=OllamaClient(url, 30),
                 on_event=lambda k, m: events.append((k, m)), rng=random.Random(0))
    return ev, events


def test_seed_then_improving_generation(home, fake_ollama):
    state, url = fake_ollama
    ev, events = make(home, url, require_approval=True)
    seed = ev.ensure_seed()
    s = ev.archive.get(seed)
    assert s.is_champion and 0.5 < s.dev < 1.0 and s.holdout is not None
    child = ev.step()
    c = ev.archive.get(child)
    assert c.parent_id == seed and c.status == "evaluated"
    assert c.dev == 1.0 and c.holdout == 1.0
    assert c.awaiting_approval and not c.is_champion  # human must approve
    assert "Always check empty inputs" in ev.archive.read_files(child)["system_prompt.md"]
    # Hold-out task text never reaches the model during improve().
    improve_reqs = [r for r in state.requests if "You are improving" in str(r.get("messages"))]
    assert improve_reqs and all("run_program" not in str(r) and "word_break_all" not in str(r) for r in improve_reqs)
    assert all(r["options"]["num_ctx"] == ev.s.num_ctx for r in state.requests if "options" in r)
    ev.archive.set_champion(child)
    assert ev.archive.champion().id == child
    assert ev.archive.productivity(seed) == 1.0
    assert "Hello" in ev.chat_with_champion([{"role": "user", "content": "hi"}])


def test_auto_promotion(home, fake_ollama):
    _, url = fake_ollama
    ev, _ = make(home, url, require_approval=False)
    ev.run(generations=1)
    champ = ev.archive.champion()
    assert champ.parent_id is not None and champ.holdout == 1.0


@pytest.mark.parametrize("mode,expect", [("bad_search", "no applicable improvement"), ("guard_violation", "guard")])
def test_bad_proposals_are_archived_as_failed(home, fake_ollama, mode, expect):
    state, url = fake_ollama
    ev, _ = make(home, url)
    ev.ensure_seed()
    state.improve_mode = mode
    child = ev.step()
    c = ev.archive.get(child)
    assert c.status == "failed" and expect in c.error
    assert ev.archive.champion().parent_id is None


def test_infra_error_aborts(home):
    ev, events = make(home, "http://127.0.0.1:9")
    with pytest.raises(InfraError):
        ev.run(generations=1)
    assert any(k == "error" for k, _ in events)


def test_report_hides_holdout(home, fake_ollama):
    _, url = fake_ollama
    ev, _ = make(home, url)
    seed = ev.ensure_seed()
    report = ev.build_report(seed)
    assert "decode_nested" in report
    for t in ev.holdout:
        assert t.id not in report and t.prompt[:40] not in report
