from qwen_rsi.benchmark import grade, load_tasks
from reference_solutions import REFERENCE


def test_split_and_ids():
    tasks = load_tasks()
    assert len({t.id for t in tasks}) == len(tasks) == 30
    assert sum(t.split == "holdout" for t in tasks) == 12


def test_reference_solutions_pass_all_checks():
    for t in load_tasks():
        r = grade(t, REFERENCE[t.id])
        assert r.solved, (t.id, r.failures, r.error)


def test_wrong_solutions_fail():
    for t in load_tasks():
        r = grade(t, f"def {t.entry_point}(*a, **k):\n    return None\n")
        assert not r.solved, t.id


def test_grader_resists_tampering():
    t = next(t for t in load_tasks() if t.id == "spiral")
    assert grade(t, "import os\ndef spiral(m): return []").error.startswith("rejected by guard")
    # Printing a fake grade line cannot spoof the random marker.
    r = grade(t, 'print(\'{"passed": 6, "total": 6}\')\ndef spiral(m): return []')
    assert not r.solved
    assert grade(t, "def spiral(m):\n    while True: pass").score < 1


def test_user_tasks(tmp_path):
    (tmp_path / "mine.json").write_text('[{"id": "add", "entry_point": "add", "prompt": "add two ints",'
                                        ' "checks": ["add(1, 2) == 3"], "split": "holdout"}]')
    tasks = load_tasks(tmp_path)
    t = next(t for t in tasks if t.id == "add")
    assert t.split == "holdout" and t.source == "mine.json"
    assert grade(t, "def add(a, b): return a + b").solved
