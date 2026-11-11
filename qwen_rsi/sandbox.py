"""Process sandbox for model-written code.

Every piece of model-written code (genome code and task solutions) runs in a
separate Python process that:

* starts with ``-I`` (isolated mode: no user site-packages, no PYTHON* env vars,
  script directory not trusted),
* runs in a fresh temporary working directory with a scrubbed environment,
* is killed on a wall-clock timeout,
* on Windows, is placed in a Job Object that caps total memory, caps the number
  of processes and kills everything when the job handle closes,
* on POSIX, gets an address-space rlimit and its own session.

Genome processes talk to the supervisor only through a JSON-lines broker on
stdin/stdout (see ``GenomeSession``); the supervisor enforces the LLM-call and
code-execution budgets, so a genome cannot raise its own limits.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import guard

IS_WINDOWS = sys.platform == "win32"
SHIM = Path(__file__).with_name("runner_shim.py")


def sandbox_python() -> str:
    # On Windows a venv's python.exe is a launcher that spawns the base
    # interpreter as a child; use the base interpreter directly so the Job
    # Object's process limit is not consumed by the launcher. Sandboxed code
    # only needs the standard library.
    exe = Path(getattr(sys, "_base_executable", None) or sys.executable)
    # The GUI runs under pythonw.exe; sandboxed children need python.exe so
    # their stdio pipes behave normally (CREATE_NO_WINDOW hides the console).
    if exe.name.lower() == "pythonw.exe" and (exe.parent / "python.exe").exists():
        exe = exe.parent / "python.exe"
    return str(exe)


def _scrubbed_env(workdir: Path) -> dict[str, str]:
    env = {
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TEMP": str(workdir),
        "TMP": str(workdir),
        "TMPDIR": str(workdir),
        "HOME": str(workdir),
    }
    if IS_WINDOWS:
        for key in ("SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT"):
            if key in os.environ:
                env[key] = os.environ[key]
        env["PATH"] = os.environ.get("SYSTEMROOT", r"C:\Windows") + r"\System32"
        env["USERPROFILE"] = str(workdir)
    else:
        env["PATH"] = "/usr/bin:/bin"
    return env


# --------------------------------------------------------------------------- Windows Job Object
class _WinJob:
    """Kill-on-close Job Object with job-wide memory and process-count limits."""

    def __init__(self, memory_mb: int, max_processes: int = 4):
        import ctypes
        from ctypes import wintypes

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [(n, ctypes.c_ulonglong) for n in (
                "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

        class BASIC(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class EXTENDED(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BASIC),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 0x00000008
        JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200
        JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION = 0x00000400
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
        JobObjectExtendedLimitInformation = 9

        self._k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._k32.CreateJobObjectW.restype = wintypes.HANDLE
        self._k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self._k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        self._k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self._k32.CloseHandle.argtypes = [wintypes.HANDLE]

        self.handle = self._k32.CreateJobObjectW(None, None)
        if not self.handle:
            raise OSError(ctypes.get_last_error(), "CreateJobObjectW failed")
        info = EXTENDED()
        info.BasicLimitInformation.LimitFlags = (
            JOB_OBJECT_LIMIT_ACTIVE_PROCESS | JOB_OBJECT_LIMIT_JOB_MEMORY
            | JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION | JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE)
        info.BasicLimitInformation.ActiveProcessLimit = max_processes
        info.JobMemoryLimit = memory_mb * 1024 * 1024
        if not self._k32.SetInformationJobObject(self.handle, JobObjectExtendedLimitInformation,
                                                 ctypes.byref(info), ctypes.sizeof(info)):
            err = ctypes.get_last_error()
            self.close()
            raise OSError(err, "SetInformationJobObject failed")

    def assign(self, proc: subprocess.Popen) -> None:
        import ctypes
        if not self._k32.AssignProcessToJobObject(self.handle, int(proc._handle)):  # type: ignore[attr-defined]
            raise OSError(ctypes.get_last_error(), "AssignProcessToJobObject failed")

    def close(self) -> None:
        if self.handle:
            self._k32.CloseHandle(self.handle)
            self.handle = None


def _spawn(args: list[str], workdir: Path, memory_mb: int) -> tuple[subprocess.Popen, Any]:
    kwargs: dict[str, Any] = dict(
        cwd=str(workdir), env=_scrubbed_env(workdir),
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    job = None
    if IS_WINDOWS:
        CREATE_NO_WINDOW = 0x08000000
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        kwargs["creationflags"] = CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP
        proc = subprocess.Popen(args, **kwargs)
        try:
            job = _WinJob(memory_mb)
            job.assign(proc)
        except OSError:
            # Never run unconstrained if limits can't be applied.
            proc.kill()
            if job:
                job.close()
            raise
    else:
        def _limits() -> None:  # runs in the child before exec
            import resource
            os.setsid()
            limit = memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        kwargs["preexec_fn"] = _limits
        proc = subprocess.Popen(args, **kwargs)
    return proc, job


def _kill(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        if not IS_WINDOWS:
            import signal
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except OSError:
            pass


# --------------------------------------------------------------------------- one-shot code runs
@dataclass
class RunResult:
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool
    seconds: float

    def to_dict(self) -> dict:
        return {"returncode": self.returncode, "stdout": self.stdout[-8000:],
                "stderr": self.stderr[-8000:], "timed_out": self.timed_out,
                "seconds": round(self.seconds, 3)}


def run_code(code: str, *, stdin: str = "", timeout: float = 10.0, memory_mb: int = 512) -> RunResult:
    """Run untrusted Python source in a fresh sandboxed interpreter."""
    workdir = Path(tempfile.mkdtemp(prefix="qrsi_run_"))
    try:
        script = workdir / "main.py"
        script.write_text(code, encoding="utf-8")
        start = time.monotonic()
        proc, job = _spawn([sandbox_python(), "-I", "-X", "utf8", str(script)], workdir, memory_mb)
        timed_out = False
        try:
            out, err = proc.communicate(stdin.encode("utf-8"), timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill(proc)
            out, err = proc.communicate()
        finally:
            if job:
                job.close()
        return RunResult(proc.returncode, out.decode("utf-8", "replace"),
                         err.decode("utf-8", "replace"), timed_out, time.monotonic() - start)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# --------------------------------------------------------------------------- genome sessions
class BudgetExceeded(Exception):
    pass


@dataclass
class SessionStats:
    llm_calls: int = 0
    exec_calls: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    logs: list[str] = field(default_factory=list)


@dataclass
class SessionOutcome:
    ok: bool
    result: Any
    error: str
    stats: SessionStats
    stderr: str


class GenomeSession:
    """Runs one entry point of a genome in a sandbox process and brokers its
    requests. ``chat_fn(request_dict) -> response_dict`` performs LLM calls."""

    def __init__(self, genome_dir: Path, *, chat_fn: Callable[[dict], dict],
                 max_llm_calls: int, max_exec_calls: int, timeout_s: float,
                 exec_timeout_s: float = 10.0, memory_mb: int = 1024,
                 on_event: Callable[[str, str], None] | None = None,
                 stop: threading.Event | None = None):
        self.genome_dir = Path(genome_dir)
        self.chat_fn = chat_fn
        self.max_llm_calls = max_llm_calls
        self.max_exec_calls = max_exec_calls
        self.timeout_s = timeout_s
        self.exec_timeout_s = exec_timeout_s
        self.memory_mb = memory_mb
        self.on_event = on_event or (lambda kind, text: None)
        self.stop = stop or threading.Event()

    def _handle(self, req: dict, stats: SessionStats) -> dict:
        op = req.get("op")
        if op == "chat":
            if stats.llm_calls >= self.max_llm_calls:
                return {"error": f"LLM call budget exhausted ({self.max_llm_calls})"}
            stats.llm_calls += 1
            self.on_event("llm", f"call {stats.llm_calls}/{self.max_llm_calls}")
            resp = self.chat_fn(req)
            stats.prompt_tokens += resp.get("prompt_tokens", 0)
            stats.output_tokens += resp.get("output_tokens", 0)
            return resp
        if op == "exec":
            if stats.exec_calls >= self.max_exec_calls:
                return {"error": f"exec budget exhausted ({self.max_exec_calls})"}
            stats.exec_calls += 1
            code = str(req.get("code", ""))
            g = guard.check_solution_code(code)
            if not g.ok:
                return {"returncode": None, "stdout": "", "stderr": "Rejected by guard:\n" + "\n".join(g.problems),
                        "timed_out": False, "seconds": 0.0}
            timeout = min(float(req.get("timeout") or self.exec_timeout_s), self.exec_timeout_s)
            return run_code(code, stdin=str(req.get("stdin", "")), timeout=timeout,
                            memory_mb=min(self.memory_mb, 1024)).to_dict()
        if op == "budget":
            return {"llm_calls_left": self.max_llm_calls - stats.llm_calls,
                    "exec_calls_left": self.max_exec_calls - stats.exec_calls}
        return {"error": f"unknown op {op!r}"}

    def run(self, entry: str, payload: dict) -> SessionOutcome:
        stats = SessionStats()
        agent_src = (self.genome_dir / "agent.py").read_text(encoding="utf-8")
        g = guard.check_genome_code(agent_src)
        if not g.ok:
            return SessionOutcome(False, None, "Genome rejected by guard:\n" + "\n".join(g.problems), stats, "")

        workdir = Path(tempfile.mkdtemp(prefix="qrsi_genome_"))
        start = time.monotonic()
        proc = job = None
        stderr_chunks: list[str] = []
        try:
            gdir = workdir / "genome"
            shutil.copytree(self.genome_dir, gdir)
            shim = workdir / "runner_shim.py"
            shutil.copy2(SHIM, shim)
            proc, job = _spawn([sandbox_python(), "-I", "-X", "utf8", str(shim), str(gdir), entry],
                               workdir, self.memory_mb)

            def _drain_stderr() -> None:
                assert proc is not None and proc.stderr is not None
                for line in iter(proc.stderr.readline, b""):
                    text = line.decode("utf-8", "replace").rstrip()
                    stderr_chunks.append(text)
                    if len(stderr_chunks) > 2000:
                        del stderr_chunks[:1000]
                    self.on_event("stderr", text)
            threading.Thread(target=_drain_stderr, daemon=True).start()

            timed_out = threading.Event()

            def _watchdog() -> None:
                while proc.poll() is None:
                    if time.monotonic() - start > self.timeout_s:
                        timed_out.set()
                        _kill(proc)
                        return
                    if self.stop.is_set():
                        _kill(proc)
                        return
                    time.sleep(0.2)
            threading.Thread(target=_watchdog, daemon=True).start()

            assert proc.stdin and proc.stdout
            proc.stdin.write((json.dumps({"payload": payload}) + "\n").encode("utf-8"))
            proc.stdin.flush()
            for raw in iter(proc.stdout.readline, b""):
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                kind = msg.get("op")
                if kind == "result":
                    stats.seconds = time.monotonic() - start
                    return SessionOutcome(True, msg.get("value"), "", stats, "\n".join(stderr_chunks))
                if kind == "error":
                    stats.seconds = time.monotonic() - start
                    return SessionOutcome(False, None, msg.get("message", "error"), stats, "\n".join(stderr_chunks))
                if kind == "log":
                    text = str(msg.get("message", ""))[:2000]
                    stats.logs.append(text)
                    self.on_event("log", text)
                    continue
                try:
                    resp = self._handle(msg, stats)
                except Exception as e:  # broker failure must not crash the supervisor
                    resp = {"error": f"{type(e).__name__}: {e}"}
                proc.stdin.write((json.dumps(resp) + "\n").encode("utf-8"))
                proc.stdin.flush()
            stats.seconds = time.monotonic() - start
            why = "timed out" if timed_out.is_set() else ("stopped" if self.stop.is_set()
                                                           else f"exited with code {proc.wait()}")
            return SessionOutcome(False, None, f"genome process {why}", stats, "\n".join(stderr_chunks))
        except (BrokenPipeError, OSError) as e:
            stats.seconds = time.monotonic() - start
            return SessionOutcome(False, None, f"sandbox error: {e}", stats, "\n".join(stderr_chunks))
        finally:
            if proc is not None:
                _kill(proc)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
            if job:
                job.close()
            shutil.rmtree(workdir, ignore_errors=True)
