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
            return self._send_json({"models": [{"name": "qwen3.8:27b"}]})
        if self.path == "/api/ps":
            return self._send_json({"models": [{"name": "qwen3.8:27b", "size_vram": 19000000000}]})
        self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        self.state.requests.append(req)
        if self.path == "/api/show":
            return self._send_json({"capabilities": ["completion", "tools", "thinking", "vision"],
                                    "model_info": {"qwen38.context_length": 262144}})
        if self.path != "/api/chat":
            return self._send_json({"error": "unsupported"}, 404)
        if req.get("model") != "qwen3.8:27b":
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
            if entry in BROKEN and MAGIC not in system:
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
