"""Seed genome: the agent that solves tasks AND improves itself.

This file is the part of the system the model rewrites. The supervisor calls
two entry points, each in its own sandbox process:

    solve(payload, llm, tools)   -> {"code": str} for code tasks,
                                    {"content": str} for chat tasks
    improve(payload, llm, tools) -> {"files": {name: text}, "rationale": str}

Because improve() is part of the genome, a child that improves improve()
becomes a better self-improver: that is the recursive part of the loop.

Contract (enforced by the supervisor, see GENOME_CONTRACT in evolve.py):
* standard-library imports only from the allowlist; no file, network or
  process access (use llm / tools instead);
* llm.chat(messages, think=None, format=None, options=None, tools=None) -> dict
  with keys content, thinking, tool_calls, prompt_tokens, output_tokens;
  raises an exception when the per-call budget is exhausted;
* tools.run_python(code, stdin="", timeout=None) -> dict(returncode, stdout,
  stderr, timed_out, seconds) runs code in a fresh sandbox;
* tools.log(message) appends to the run log shown in the app.
"""
import ast
import json
import re

GENOME_FILES = ["agent.py", "system_prompt.md", "improve_prompt.md", "config.json", "NOTES.md"]


# --------------------------------------------------------------------------- helpers
def load_config(payload):
    try:
        return json.loads(payload["genome"].get("config.json", "{}"))
    except ValueError:
        return {}


def extract_code(text, entry_point=None):
    """Return the most relevant ```python block, or the raw text if none."""
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    if not blocks:
        return text.strip()
    if entry_point:
        for b in reversed(blocks):
            if re.search(r"def\s+" + re.escape(entry_point) + r"\s*\(", b):
                return b.strip()
    return blocks[-1].strip()


def extract_json(text):
    text = text.strip()
    try:
        return json.loads(text)
    except ValueError:
        pass
    m = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except ValueError:
            pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return json.loads(text[start:end + 1])
    raise ValueError("no JSON object found in model output")


# --------------------------------------------------------------------------- solve
def solve(payload, llm, tools):
    cfg = load_config(payload)
    system = payload["genome"].get("system_prompt.md", "")
    task = payload["task"]

    if task.get("kind") == "chat":
        msgs = [{"role": "system", "content": system}] + task["messages"]
        r = llm.chat(msgs, think=cfg.get("think_chat", "medium"), options=cfg.get("options_solve"))
        return {"content": r["content"]}

    entry = task["entry_point"]
    user = (
        task["prompt"]
        + "\n\nReturn a complete, self-contained Python module in a single ```python block that defines `"
        + entry + "`. Only standard-library imports. No example usage, no tests, no input()."
    )
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    r = llm.chat(msgs, think=cfg.get("think_solve", "medium"), options=cfg.get("options_solve"))
    code = extract_code(r["content"], entry)

    # Cheap self-check: does the module load and define the entry point?
    for _ in range(int(cfg.get("max_fix_rounds", 1))):
        probe = code + "\n\nassert callable(" + entry + ")\nprint('OK')\n"
        res = tools.run_python(probe, timeout=5)
        if res.get("returncode") == 0 and "OK" in res.get("stdout", ""):
            break
        tools.log("self-check failed for " + entry + "; asking for a fix")
        msgs += [
            {"role": "assistant", "content": r["content"]},
            {"role": "user", "content": "Loading your module failed:\n" + str(res.get("stderr", ""))[-2000:]
                + "\nFix it and return the full corrected module in one ```python block."},
        ]
        r = llm.chat(msgs, think=cfg.get("think_solve", "medium"), options=cfg.get("options_solve"))
        code = extract_code(r["content"], entry)
    return {"code": code}


# --------------------------------------------------------------------------- improve
EDIT_SCHEMA = {
    "type": "object",
    "properties": {
        "analysis": {"type": "string"},
        "plan": {"type": "string"},
        "edits": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "file": {"type": "string", "enum": GENOME_FILES},
                    "search": {"type": "string"},
                    "replace": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["file"],
            },
        },
    },
    "required": ["analysis", "plan", "edits"],
}


def apply_edits(files, edits):
    """Apply search/replace or whole-file edits. Returns (new_files, errors)."""
    new = dict(files)
    errors = []
    for i, e in enumerate(edits):
        name = e.get("file")
        if name not in GENOME_FILES:
            errors.append("edit %d: unknown file %r" % (i, name))
            continue
        if "content" in e and e.get("content") is not None and not e.get("search"):
            new[name] = e["content"]
            continue
        search, replace = e.get("search", ""), e.get("replace", "")
        if not search:
            errors.append("edit %d: needs either content or search" % i)
            continue
        current = new.get(name, "")
        count = current.count(search)
        if count != 1:
            errors.append("edit %d on %s: search text found %d times (must be exactly once)" % (i, name, count))
            continue
        new[name] = current.replace(search, replace, 1)
    if "agent.py" in new:
        try:
            ast.parse(new["agent.py"])
        except SyntaxError as exc:
            errors.append("agent.py no longer parses: %s (line %s)" % (exc.msg, exc.lineno))
    try:
        json.loads(new.get("config.json", "{}"))
    except ValueError as exc:
        errors.append("config.json is not valid JSON: %s" % exc)
    return new, errors


def render_files(files):
    parts = []
    for name in GENOME_FILES:
        if name in files:
            parts.append("=== FILE: " + name + " ===\n" + files[name])
    return "\n\n".join(parts)


def improve(payload, llm, tools):
    cfg = load_config(payload)
    files = payload["genome"]
    template = files.get("improve_prompt.md", "")
    prompt = (template
              .replace("{{CONTRACT}}", payload.get("contract", ""))
              .replace("{{FILES}}", render_files(files))
              .replace("{{REPORT}}", payload.get("report", ""))
              .replace("{{HISTORY}}", payload.get("history", "")))
    msgs = [{"role": "user", "content": prompt}]
    use_format = cfg.get("improve_use_format", True)
    attempts = int(cfg.get("improve_attempts", 3))

    last_errors = []
    for attempt in range(attempts):
        r = llm.chat(msgs, think=cfg.get("think_improve", "high"),
                     format=EDIT_SCHEMA if use_format else None,
                     options=cfg.get("options_improve"))
        try:
            proposal = extract_json(r["content"])
        except ValueError as exc:
            last_errors = [str(exc)]
        else:
            new_files, last_errors = apply_edits(files, proposal.get("edits", []))
            changed = any(new_files.get(k) != files.get(k) for k in new_files)
            if not last_errors and changed:
                rationale = (proposal.get("plan", "") + "\n\n" + proposal.get("analysis", "")).strip()
                return {"files": new_files, "rationale": rationale[:4000]}
            if not changed and not last_errors:
                last_errors = ["the edits did not change anything"]
        tools.log("improve attempt %d rejected: %s" % (attempt + 1, "; ".join(last_errors)[:500]))
        msgs += [
            {"role": "assistant", "content": r["content"]},
            {"role": "user", "content": "Your proposal could not be applied:\n- " + "\n- ".join(last_errors)
                + "\nReturn a corrected JSON proposal. Search strings must be copied exactly from the current "
                  "files and occur exactly once."},
        ]
    raise RuntimeError("no applicable improvement: " + "; ".join(last_errors))
