"""Minimal, dependency-free client for the Ollama REST API.

Covers the endpoints the app needs: /api/chat (streaming and not),
/api/tags, /api/show, /api/ps, /api/pull and /api/version.
"""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator


class OllamaError(RuntimeError):
    pass


@dataclass
class ChatResult:
    content: str = ""
    thinking: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    done_reason: str = ""
    prompt_tokens: int = 0
    output_tokens: int = 0
    total_duration_ns: int = 0

    def as_message(self) -> dict:
        msg: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.thinking:
            msg["thinking"] = self.thinking
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        return msg


class OllamaClient:
    def __init__(self, base_url: str = "http://127.0.0.1:11434", timeout: float = 900.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # ------------------------------------------------------------------ http
    def _request(self, method: str, path: str, body: dict | None = None, timeout: float | None = None):
        data = None if body is None else json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            return urllib.request.urlopen(req, timeout=timeout or self.timeout)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")
            try:
                detail = json.loads(detail).get("error", detail)
            except (json.JSONDecodeError, AttributeError):
                pass
            raise OllamaError(f"{method} {path} -> HTTP {e.code}: {detail}") from None
        except urllib.error.URLError as e:
            raise OllamaError(f"Cannot reach Ollama at {self.base_url}: {e.reason}") from None

    def _json(self, method: str, path: str, body: dict | None = None, timeout: float | None = None) -> dict:
        with self._request(method, path, body, timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _stream(self, path: str, body: dict, stop: threading.Event | None = None) -> Iterator[dict]:
        with self._request("POST", path, body) as resp:
            for raw in resp:
                if stop is not None and stop.is_set():
                    return
                line = raw.strip()
                if not line:
                    continue
                chunk = json.loads(line)
                if "error" in chunk:
                    raise OllamaError(chunk["error"])
                yield chunk

    # ------------------------------------------------------------- endpoints
    def version(self) -> str:
        return self._json("GET", "/api/version", timeout=5).get("version", "?")

    def list_models(self) -> list[str]:
        return [m["name"] for m in self._json("GET", "/api/tags", timeout=10).get("models", [])]

    def running(self) -> list[dict]:
        return self._json("GET", "/api/ps", timeout=10).get("models", [])

    def show(self, model: str) -> dict:
        return self._json("POST", "/api/show", {"model": model}, timeout=30)

    def context_length(self, model: str) -> int | None:
        info = self.show(model).get("model_info", {}) or {}
        for k, v in info.items():
            if k.endswith(".context_length"):
                return int(v)
        return None

    def capabilities(self, model: str) -> list[str]:
        return list(self.show(model).get("capabilities", []) or [])

    def has_model(self, model: str) -> bool:
        names = self.list_models()
        if model in names:
            return True
        return ":" not in model and f"{model}:latest" in names

    def pull(self, model: str, progress: Callable[[str, int, int], None] | None = None,
             stop: threading.Event | None = None) -> None:
        for chunk in self._stream("/api/pull", {"model": model, "stream": True}, stop):
            if progress:
                progress(chunk.get("status", ""), int(chunk.get("completed", 0) or 0),
                         int(chunk.get("total", 0) or 0))

    # ------------------------------------------------------------------ chat
    def build_chat_body(self, model: str, messages: list[dict], *, think: Any = None,
                        tools: list[dict] | None = None, format: Any = None,
                        options: dict | None = None, keep_alive: str | None = None,
                        stream: bool = True) -> dict:
        body: dict[str, Any] = {"model": model, "messages": messages, "stream": stream}
        if think is not None:
            body["think"] = think
        if tools:
            body["tools"] = tools
        if format is not None:
            body["format"] = format
        if options:
            body["options"] = options
        if keep_alive:
            body["keep_alive"] = keep_alive
        return body

    def chat_stream(self, model: str, messages: list[dict], *, stop: threading.Event | None = None,
                    **kw) -> Iterator[dict]:
        body = self.build_chat_body(model, messages, stream=True, **kw)
        try:
            yield from self._stream("/api/chat", body, stop)
        except OllamaError as e:
            # Older Ollama builds / non-thinking models reject `think`; retry once without it.
            if "think" in body and "think" in str(e).lower():
                body.pop("think")
                yield from self._stream("/api/chat", body, stop)
            else:
                raise

    def chat(self, model: str, messages: list[dict], *,
             on_token: Callable[[str, str], None] | None = None,
             stop: threading.Event | None = None, **kw) -> ChatResult:
        """Streams internally (keeps the HTTP connection alive for long thinking
        phases) and returns the aggregated result. ``on_token(kind, text)`` is
        called with kind ``"thinking"`` or ``"content"``."""
        res = ChatResult()
        content: list[str] = []
        thinking: list[str] = []
        for chunk in self.chat_stream(model, messages, stop=stop, **kw):
            msg = chunk.get("message") or {}
            if msg.get("thinking"):
                thinking.append(msg["thinking"])
                if on_token:
                    on_token("thinking", msg["thinking"])
            if msg.get("content"):
                content.append(msg["content"])
                if on_token:
                    on_token("content", msg["content"])
            if msg.get("tool_calls"):
                res.tool_calls.extend(msg["tool_calls"])
            if chunk.get("done"):
                res.done_reason = chunk.get("done_reason", "")
                res.prompt_tokens = int(chunk.get("prompt_eval_count", 0) or 0)
                res.output_tokens = int(chunk.get("eval_count", 0) or 0)
                res.total_duration_ns = int(chunk.get("total_duration", 0) or 0)
        res.content = "".join(content)
        res.thinking = "".join(thinking)
        return res
