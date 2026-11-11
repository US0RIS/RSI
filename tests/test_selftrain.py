import copy
import json
import random
import sys
from pathlib import Path

import pytest

from qwen_rsi.archive import Archive
from qwen_rsi.benchmark import load_tasks
from qwen_rsi.config import Settings
from qwen_rsi.ollama_client import OllamaClient
from qwen_rsi.selftrain import recipe as R
from qwen_rsi.selftrain.engine import RSIEngine
from qwen_rsi.selftrain.ollama_import import Importer
from qwen_rsi.selftrain.taskgen import TaskValidator
from qwen_rsi.selftrain.train_lora import build_example
from qwen_rsi.selftrain.trainer import TrainerBridge

FAKE_TRAINER = Path(__file__).with_name("fake_trainer.py")


# ------------------------------------------------------------------ units
def test_recipe_clamp_and_mutate():
    r = R.clamp({"train": {"learning_rate": 1.0, "lora_r": 2, "lr_scheduler": "evil"}, "bogus": {"x": 1},
                 "task_gen": {"topics": ["a", ""]}})
    assert r["train"]["learning_rate"] == 5e-4 and r["train"]["lora_r"] == 4
    assert r["train"]["lr_scheduler"] == "cosine" and "bogus" not in r and r["task_gen"]["topics"] == ["a"]
    base = R.clamp({"train": {"epochs": 3}})
    assert R.clamp({"data": {"replay_fraction": 0.5}}, base=base)["train"]["epochs"] == 3
    new, changes = R.mutate(R.DEFAULT_RECIPE, random.Random(1))
    assert len(changes) >= 1 and R.clamp(new) == new
    assert "train.learning_rate" in R.describe_bounds()


def _qwen_render(msgs, add_generation_prompt):
    out = ""
    for i, m in enumerate(msgs):
        body = m["content"]
        if m["role"] == "assistant" and m.get("reasoning_content") and i == len(msgs) - 1:
            body = f"<think>\n{m['reasoning_content']}\n</think>\n\n" + body
        out += f"<|im_start|>{m['role']}\n{body}<|im_end|>\n"
    return out + ("<|im_start|>assistant\n" if add_generation_prompt else "")


def test_build_example_masks_prompt_and_keeps_thinking():
    enc = lambda t: [ord(c) for c in t]  # noqa: E731
    msgs = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "A"}]
    ids, labels = build_example(msgs, "because", _qwen_render, enc, 10_000)
    prompt_len = len(_qwen_render(msgs[:1], True))
    assert labels[:prompt_len] == [-100] * prompt_len
    completion = "".join(chr(c) for c in labels[prompt_len:])
    assert completion == "<think>\nbecause\n</think>\n\nA<|im_end|>\n"
    assert build_example(msgs, "", _qwen_render, enc, 10) is None   # too long
    # Template that drops reasoning: falls back to explicit <think> block + eos.
    plain = lambda m, g: _qwen_render([dict(x, reasoning_content=None) for x in m], g)  # noqa: E731
    ids, labels = build_example(msgs, "why", plain, enc, 10_000, eos="<|im_end|>")
    assert "".join(chr(c) for c in labels if c != -100) == "<think>\nwhy\n</think>\n\nA<|im_end|>"


def test_task_validator():
    v = TaskValidator(load_tasks(), [])
    good = {"entry_point": "double_all", "prompt": "Write `double_all(xs: list[int]) -> list[int]` doubling each "
            "element; return [] for an empty list.", "reference_solution": "def double_all(xs):\n    return [2*x for x in xs]",
            "checks": ["double_all([1]) == [2]", "double_all([]) == []", "double_all([0, 3]) == [0, 6]",
                       "double_all([-1]) == [-2]", "double_all([5, 5]) == [10, 10]"]}
    t, why = v.validate(good)
    assert t is not None, why
    v.accept(t)
    assert "duplicate" in v.validate(good)[1]
    bad = dict(good, reference_solution="def double_all(xs):\n    return xs")
    assert "reference fails" in v.validate(dict(bad, entry_point="double_all2",
                                                 checks=[c.replace("double_all", "double_all2") for c in good["checks"]]))[1]
    def rename(d, ep):
        return dict(d, entry_point=ep, checks=[c.replace("double_all", ep) for c in d["checks"]],
                    prompt=d["prompt"].replace("double_all", ep))
    assert "collides" in v.validate(rename(good, "knapsack"))[1]
    holdout = next(t for t in load_tasks() if t.id == "word_break_all")
    leak = dict(rename(good, "seg_all"), prompt=holdout.prompt.replace("word_break_all", "seg_all"))
    assert "too similar" in v.validate(leak)[1]
    weak = dict(good, entry_point="noop_fn", reference_solution="def noop_fn(x):\n    return None",
                checks=["noop_fn(1) is None"] * 5)
    assert "too weak" in v.validate(weak)[1]


# ------------------------------------------------------------------ engine
SMALL = R.clamp({"task_gen": {"tasks_per_round": 10, "tasks_per_call": 5},
                 "sampling": {"attempts_per_task": 2}})


class FakeImporter(Importer):
    def __init__(self, s, client, state, fail_adapter=False):
        super().__init__(s, client)
        self.state, self.fail_adapter, self.calls = state, fail_adapter, []

    def _run(self, args, cwd, timeout=0):
        self.calls.append(args)
        if args[0] == "create":
            mf = Path(args[-1]).read_text()
            if self.fail_adapter and "ADAPTER" in mf:
                return 1, "unsupported adapter architecture"
            self.state.models.add(args[1])
            return 0, "success"
        if args[0] == "rm":
            self.state.models.discard(args[1])
        return 0, ""


def make_engine(home, url, state, *, trainer_ok=True, fail_adapter=False, **kw):
    s = Settings(ollama_url=url, rsi_min_examples=3, **kw)
    if trainer_ok:
        s.trainer_python = sys.executable
    else:
        s.trainer_python = str(home / "missing" / "python.exe")
    client = OllamaClient(url, 30)
    archive = Archive(home)
    archive.add_model(s.model, None, 0, "base", recipe=copy.deepcopy(SMALL))
    events = []
    eng = RSIEngine(s, archive=archive, client=client, on_event=lambda k, m: events.append((k, m)),
                    trainer=TrainerBridge(s, script=FAKE_TRAINER),
                    importer=FakeImporter(s, client, state, fail_adapter), rng=random.Random(0))
    return eng, events


def test_round_trains_and_promotes_new_weights(home, fake_ollama):
    state, url = fake_ollama
    eng, events = make_engine(home, url, state, rsi_scaffold_generations=0)
    eng.run(rounds=2)
    m1 = eng.archive.get_model("qwen3.8-rsi:r1")
    assert m1.status == "promoted", m1.error
    assert eng.s.model == "qwen3.8-rsi:r1" and Settings.load().model == "qwen3.8-rsi:r1"
    assert m1.stats["examples"] >= 3 and m1.stats["final_loss"] == 0.33 and m1.audit is not None
    champ = eng.archive.champion()
    assert champ.model == "qwen3.8-rsi:r1" and champ.holdout == 1.0
    # Contaminating and weak generated tasks never entered the pool.
    pool = [json.loads(l)["entry_point"] for l in eng.pool_path.read_text().splitlines()]
    assert "knapsack" not in pool and "weak_thing" not in pool and len(pool) >= 10
    # Round 2: the PROMOTED model designed the recipe and generated the data.
    m2 = eng.archive.get_model("qwen3.8-rsi:r2")
    assert m2.parent_tag == "qwen3.8-rsi:r1" and m2.recipe["train"]["learning_rate"] == 0.0002
    recipe_reqs = [r for r in state.requests if "OWN post-training" in str(r.get("messages"))]
    assert recipe_reqs and recipe_reqs[-1]["model"] == "qwen3.8-rsi:r1"
    # The trained model solves everything, so nothing is left to learn: round 2 can't train.
    assert m2.status == "failed" and "examples" in m2.error and m2.stats["too_easy"] == m2.stats["tasks"]
    # Hold-out task text never reached the model in any generation/recipe/attempt prompt.
    holdout = [t for t in load_tasks() if t.split == "holdout"]
    for r in state.requests:
        msgs = str(r.get("messages"))
        if "Create " in msgs or "OWN post-training" in msgs or "scale_sum" in msgs:
            assert not any(t.prompt[:60] in msgs for t in holdout)


def test_round_rejects_weights_that_are_not_better(home, fake_ollama):
    state, url = fake_ollama
    eng, _ = make_engine(home, url, state, rsi_scaffold_generations=1)
    eng.run(rounds=1)
    m1 = eng.archive.get_model("qwen3.8-rsi:r1")
    assert m1.status == "rejected" and "hold-out 1.000->1.000" in m1.error
    assert eng.s.model == "qwen3.8:27b"
    assert ["rm", "qwen3.8-rsi:r1"] in eng.importer.calls
    assert eng.archive.champion().model == "qwen3.8:27b"


def test_without_trainer_data_is_still_collected(home, fake_ollama):
    state, url = fake_ollama
    eng, events = make_engine(home, url, state, trainer_ok=False, rsi_scaffold_generations=0)
    eng.run(rounds=1)
    m1 = eng.archive.get_model("qwen3.8-rsi:r1")
    assert m1.status == "skipped" and eng.examples_path.exists()
    assert any(k == "trainer" and "NOT available" in m for k, m in events)


def test_training_failure_is_recorded(home, fake_ollama, monkeypatch):
    state, url = fake_ollama
    monkeypatch.setenv("FAKE_TRAINER_FAIL", "1")
    eng, _ = make_engine(home, url, state, rsi_scaffold_generations=0)
    eng.run(rounds=1)
    m1 = eng.archive.get_model("qwen3.8-rsi:r1")
    assert m1.status == "failed" and "out of memory" in m1.error


def test_adapter_import_falls_back_to_merged(home, fake_ollama):
    state, url = fake_ollama
    eng, events = make_engine(home, url, state, fail_adapter=True, rsi_scaffold_generations=0)
    eng.run(rounds=1)
    assert eng.archive.get_model("qwen3.8-rsi:r1").status == "promoted"
    merged_mf = next(Path(c[-1]) for c in eng.importer.calls if "--quantize" in c)
    text = merged_mf.read_text()
    assert "RENDERER qwen3.8" in text and "PARSER qwen3.8" in text and "PARAMETER temperature 0.6" in text
