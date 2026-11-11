"""A scripted stand-in for the Ollama HTTP API used by the tests.

Solve requests get the reference solution, except that tasks in BROKEN are
answered wrongly unless the system prompt contains the marker MAGIC. Improve
requests get a JSON proposal that inserts MAGIC into system_prompt.md, so a
single generation produces a measurably better child.
"""
import json
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from reference_solutions import REFERENCE  # noqa: E402

MAGIC = "Always check empty inputs first."
BROKEN = {"decode_nested", "merge_intervals", "eval_expr", "spiral", "knapsack", "islands", "rotate_cw",
          "next_permutation"}


class State:
    def __init__(self):
        self.requests = []
        self.improve_mode = "good"   # good | bad_search | guard_violation
        self.models = {"qwen3.8:27b"}
        self.gen_counter = 0
        self.attempt_counter = 0


def _gen_tasks(state, n):
    """Mostly valid generated tasks, plus one contaminating and one weak task."""
    out = []
    for _ in range(n):
        state.gen_counter += 1
        k = state.gen_counter
        ep = f"scale_sum_{k}"
        out.append({
            "entry_point": ep,
            "prompt": f"Write `{ep}(xs: list[int]) -> int` returning the sum of xs multiplied by {k + 1}. "
                      f"Return 0 for an empty list. Raise TypeError if xs is not a list.",
            "reference_solution": f"def {ep}(xs):\n    if not isinstance(xs, list):\n        raise TypeError\n"
                                  f"    return sum(xs) * {k + 1}\n",
            "checks": [f"{ep}([1, 2]) == {3 * (k + 1)}", f"{ep}([]) == 0", f"{ep}([5]) == {5 * (k + 1)}",
                       f"{ep}([-1, 1]) == 0", f"_raises(TypeError, {ep}, 'x')"],
        })
    out.append({"entry_point": "knapsack", "prompt": "Write `knapsack(weights, values, capacity) -> int` returning "
                "the best value of a 0/1 knapsack selection within capacity.",
                "reference_solution": "def knapsack(w, v, c):\n    return 0\n",
                "checks": ["knapsack([], [], 1) == 0"] * 5})
    out.append({"entry_point": "weak_thing", "prompt": "Write `weak_thing(x) -> None` that does nothing at all "
                "and returns None for every input value given.",
                "reference_solution": "def weak_thing(x):\n    return None\n",
                "checks": ["weak_thing(1) is None"] * 5})
    return out


class Handler(BaseHTTPRequestHandler):
    state: State = None  # set by start()

    def log_message(self, *a):
        pass

    def _send_json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/version":
            return self._send_json({"version": "0.99.0-fake"})
        if self.path == "/api/tags":
            return self._send_json({"models": [{"name": m} for m in sorted(self.state.models)]})
        if self.path == "/api/ps":
            return self._send_json({"models": [{"name": "qwen3.8:27b", "size_vram": 19000000000}]})
        self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        self.state.requests.append(req)
        if self.path == "/api/show":
            return self._send_json({"capabilities": ["completion", "tools", "thinking", "vision"],
                                    "model_info": {"qwen38.context_length": 262144},
                                    "renderer": "qwen3.8", "parser": "qwen3.8", "parameters": "temperature 0.6"})
        if self.path != "/api/chat":
            return self._send_json({"error": "unsupported"}, 404)
        if req.get("model") not in self.state.models:
            return self._send_json({"error": f"model '{req.get('model')}' not found"}, 404)
        content = self._answer(req)
        chunks = [{"message": {"role": "assistant", "content": "", "thinking": "Let me think."}, "done": False}]
        for i in range(0, len(content), 40):
            chunks.append({"message": {"role": "assistant", "content": content[i:i + 40]}, "done": False})
        chunks.append({"message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "stop",
                       "prompt_eval_count": 100, "eval_count": 50, "total_duration": 1000})
        if req.get("stream") is False:
            return self._send_json({"message": {"role": "assistant", "content": content}, "done": True})
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        for c in chunks:
            self.wfile.write((json.dumps(c) + "\n").encode())
            self.wfile.flush()

    def _answer(self, req):
        msgs = req.get("messages", [])
        text = "\n".join(m.get("content", "") for m in msgs)
        trained = req.get("model", "").startswith("qwen3.8-rsi:")
        if "What is 17 * 23" in text:
            return "391"
        if "new, original Python programming problems" in text:
            n = int(re.search(r"Create (\d+) new", text).group(1))
            return json.dumps({"tasks": _gen_tasks(self.state, n)})
        if "designing the next round of your OWN post-training" in text:
            return json.dumps({"analysis": "loss was fine; more data", "recipe": {"train": {"learning_rate": 0.0002},
                               "task_gen": {"tasks_per_round": 12}}})
        if "You are improving the source code" in text:
            mode = self.state.improve_mode
            if mode == "guard_violation":
                edits = [{"file": "agent.py", "search": "import ast\n", "replace": "import ast\nimport os\n"}]
            elif mode == "bad_search":
                edits = [{"file": "system_prompt.md", "search": "THIS TEXT DOES NOT EXIST", "replace": "x"}]
            else:
                edits = [{"file": "system_prompt.md", "search": "You are a careful senior Python engineer.",
                          "replace": "You are a careful senior Python engineer. " + MAGIC}]
            return json.dumps({"analysis": "edge cases are missed", "plan": "add an edge-case rule", "edits": edits})
        m = re.search(r"defines `(\w+)`", text)
        if m:
            entry = m.group(1)
            system = msgs[0]["content"] if msgs and msgs[0]["role"] == "system" else ""
            if entry.startswith("scale_sum_"):
                k = int(entry.rsplit("_", 1)[1])
                self.state.attempt_counter += 1
                ok = trained or self.state.attempt_counter % 2 == 0
                code = (f"def {entry}(xs):\n    if not isinstance(xs, list):\n        raise TypeError\n"
                        f"    return sum(xs) * {k + 1}\n") if ok else f"def {entry}(xs):\n    return sum(xs)\n"
            elif entry in BROKEN and MAGIC not in system and not trained:
                code = f"def {entry}(*a, **k):\n    return None\n"
            else:
                code = REFERENCE.get(entry, f"def {entry}(*a):\n    return None\n")
            return "Here you go:\n```python\n" + code.strip() + "\n```"
        return "Hello from the fake model."


def start():
    state = State()
    handler = type("H", (Handler,), {"state": state})
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, state, f"http://127.0.0.1:{srv.server_address[1]}"
