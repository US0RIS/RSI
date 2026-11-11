"""Trusted shim that runs INSIDE the sandbox process.

Usage: python -I runner_shim.py <genome_dir> <entry>

Reads one JSON line ``{"payload": ...}`` from stdin, imports ``agent.py`` from
the genome directory, calls ``agent.<entry>(payload, llm, tools)`` and writes
``{"op": "result", "value": ...}`` to the protocol channel. Any LLM call or
code execution is a request to the supervisor, which owns the budgets.

Standard library only: this file is copied next to the genome and executed by
the bare base interpreter.
"""
import importlib.util
import json
import sys
import traceback

_PROTO_OUT = sys.stdout
_PROTO_IN = sys.stdin
# Genome print() output goes to stderr so it can't corrupt the protocol.
sys.stdout = sys.stderr


def _send(msg):
    _PROTO_OUT.write(json.dumps(msg) + "\n")
    _PROTO_OUT.flush()


def _call(msg):
    _send(msg)
    line = _PROTO_IN.readline()
    if not line:
        raise SystemExit("supervisor closed the channel")
    return json.loads(line)


class LLMError(RuntimeError):
    pass


class LLM:
    """Handle the genome uses to talk to the model (through the supervisor)."""

    def chat(self, messages, think=None, format=None, options=None, tools=None):
        """Return a dict: {"content", "thinking", "tool_calls", "prompt_tokens",
        "output_tokens", "done_reason"}. Raises LLMError when the budget is
        exhausted or the model call fails."""
        resp = _call({"op": "chat", "messages": messages, "think": think,
                      "format": format, "options": options, "tools": tools})
        if "error" in resp:
            raise LLMError(resp["error"])
        return resp

    def ask(self, prompt, system=None, **kw):
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        return self.chat(msgs, **kw)["content"]

    def budget(self):
        return _call({"op": "budget"})


class Tools:
    def run_python(self, code, stdin="", timeout=None):
        """Run Python source in a separate sandbox. Returns a dict with
        returncode, stdout, stderr, timed_out, seconds (or an "error" key)."""
        return _call({"op": "exec", "code": code, "stdin": stdin, "timeout": timeout})

    def log(self, message):
        _send({"op": "log", "message": str(message)})


def main():
    genome_dir, entry = sys.argv[1], sys.argv[2]
    header = json.loads(_PROTO_IN.readline())
    try:
        spec = importlib.util.spec_from_file_location("agent", genome_dir + "/agent.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["agent"] = mod
        spec.loader.exec_module(mod)
        fn = getattr(mod, entry)
        value = fn(header["payload"], LLM(), Tools())
        json.dumps(value)  # must be JSON-serialisable
        _send({"op": "result", "value": value})
    except BaseException as e:  # report everything, including SystemExit from the genome
        _send({"op": "error", "message": f"{type(e).__name__}: {e}\n{traceback.format_exc()[-4000:]}"})


if __name__ == "__main__":
    main()
