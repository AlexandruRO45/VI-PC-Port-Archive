"""Run the repository's standard-library unit tests in parallel.

    python -I scripts/run_stdlib_tests.py

Selects the same ``tests/test_*.py`` subset that ``.github/workflows/ci.yml`` assigns to the
``check`` job (everything outside ``tests/semantic`` that does not match ``PYTEST_FILE_MARKERS``
and contains ``__main__``), runs each file in its own isolated ``python -I`` subprocess across a
worker pool, streams each file's output atomically on completion, runs every file even after
earlier failures, emits GitHub Actions ``::error`` annotations for every failed file, and exits
non-zero if any file fails or if no files are discovered.

Standard library only; no game files, no bash, no third-party packages.
"""

import argparse
import concurrent.futures
from dataclasses import dataclass
import fnmatch
import locale
import os
import re
import signal
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TESTS_DIR = "tests"
DEFAULT_PRUNE_DIR = "tests/semantic"
DEFAULT_GLOB = "test_*.py"
DEFAULT_PYTEST_MARKERS = r"^import pytest|^ *HAVE_NUMPY ="
DEFAULT_TIMEOUT = 600


@dataclass(frozen=True)
class TestResult:
    path: str
    returncode: int
    elapsed: float
    output: str
    timed_out: bool = False


def resolve_pytest_markers(explicit=None):
    """Return explicit markers, else PYTEST_FILE_MARKERS from the environment, else the default."""
    if explicit is not None:
        return explicit
    env_markers = os.environ.get("PYTEST_FILE_MARKERS")
    return env_markers if env_markers else DEFAULT_PYTEST_MARKERS


def discover_tests(
    root,
    tests_dir=DEFAULT_TESTS_DIR,
    prune_dir=DEFAULT_PRUNE_DIR,
    pattern=DEFAULT_GLOB,
    pytest_markers=None,
):
    """Return sorted POSIX-relative paths of stdlib test files matching the CI selection rule."""
    markers_re = re.compile(resolve_pytest_markers(pytest_markers), re.MULTILINE)
    tests_rel = tests_dir.replace("\\", "/").strip("/")
    prune_rel = prune_dir.replace("\\", "/").strip("/")
    start_dir = os.path.join(root, *tests_rel.split("/")) if tests_rel else root
    if not os.path.isdir(start_dir):
        return []

    selected = []
    for base, dirs, files in os.walk(start_dir):
        rel_base = os.path.relpath(base, root).replace(os.sep, "/")
        if rel_base == ".":
            rel_base = ""
        if prune_rel and rel_base == prune_rel:
            dirs[:] = []
            continue

        kept_dirs = []
        for d in sorted(dirs):
            if d == "__pycache__":
                continue
            child_rel = f"{rel_base}/{d}" if rel_base else d
            if prune_rel and child_rel == prune_rel:
                continue
            kept_dirs.append(d)
        dirs[:] = kept_dirs

        for name in sorted(files):
            if not fnmatch.fnmatchcase(name, pattern):
                continue
            rel = f"{rel_base}/{name}" if rel_base else name
            full_path = os.path.join(base, name)
            with open(full_path, encoding="utf-8", errors="replace") as f:
                source = f.read()
            if markers_re.search(source) or "__main__" not in source:
                continue
            selected.append(rel)

    selected.sort()
    return selected


def child_output_encoding():
    """Encoding a piped `python -I` child uses for stdout/stderr on this platform."""
    get_encoding = getattr(locale, "getencoding", None)
    if get_encoding is not None:
        return get_encoding()
    return locale.getpreferredencoding(False) or "utf-8"


def _decode_bytes(data, encoding=None):
    if not data:
        return ""
    if isinstance(data, bytes):
        text = data.decode(encoding or child_output_encoding(), errors="replace")
    else:
        text = str(data)
    return text.replace("\r\n", "\n")


class _Win32Job:
    """Windows Job Object retaining ownership of a child and its descendants past parent exit."""

    def __init__(self):
        self._handle = None
        self._kernel32 = None
        if os.name != "nt":
            return
        try:
            import ctypes
            from ctypes import wintypes

            class _BasicLimitInfo(ctypes.Structure):
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

            class _IoCounters(ctypes.Structure):
                _fields_ = [
                    ("ReadOperationCount", ctypes.c_uint64),
                    ("WriteOperationCount", ctypes.c_uint64),
                    ("OtherOperationCount", ctypes.c_uint64),
                    ("ReadTransferCount", ctypes.c_uint64),
                    ("WriteTransferCount", ctypes.c_uint64),
                    ("OtherTransferCount", ctypes.c_uint64),
                ]

            class _ExtendedLimitInfo(ctypes.Structure):
                _fields_ = [
                    ("BasicLimitInformation", _BasicLimitInfo),
                    ("IoInfo", _IoCounters),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t),
                ]

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
            kernel32.CreateJobObjectW.restype = wintypes.HANDLE
            kernel32.SetInformationJobObject.argtypes = [
                wintypes.HANDLE,
                ctypes.c_int,
                ctypes.c_void_p,
                wintypes.DWORD,
            ]
            kernel32.SetInformationJobObject.restype = wintypes.BOOL
            kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
            kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
            kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
            kernel32.TerminateJobObject.restype = wintypes.BOOL
            kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
            kernel32.CloseHandle.restype = wintypes.BOOL

            job = kernel32.CreateJobObjectW(None, None)
            if not job:
                return
            info = _ExtendedLimitInfo()
            # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE (0x2000), JobObjectExtendedLimitInformation (9).
            info.BasicLimitInformation.LimitFlags = 0x2000
            if not kernel32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
                kernel32.CloseHandle(job)
                return
            self._kernel32 = kernel32
            self._handle = job
        except (ImportError, OSError, AttributeError):
            self._handle = None
            self._kernel32 = None

    def assign(self, proc):
        if not self._handle:
            return False
        proc_handle = getattr(proc, "_handle", None)
        if not proc_handle:
            return False
        if self._kernel32.AssignProcessToJobObject(self._handle, int(proc_handle)):
            return True
        self.close()
        return False

    def terminate(self):
        if self._handle:
            return bool(self._kernel32.TerminateJobObject(self._handle, 1))
        return False

    def close(self):
        if self._handle:
            handle, self._handle = self._handle, None
            self._kernel32.CloseHandle(handle)


def _terminate_process_tree(proc, job=None):
    """Terminate ``proc`` and any descendants so helpers cannot outlive a timeout or hold pipes open."""
    if os.name == "nt":
        if job is not None and job.terminate():
            pass
        else:
            # Fallback when Job Object assignment is unavailable: kill tree while parent PID is alive.
            try:
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=15,
                )
            except (OSError, subprocess.TimeoutExpired):
                pass
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass
    try:
        proc.kill()
    except OSError:
        pass


def run_one(root, rel, python_exe=None, timeout=DEFAULT_TIMEOUT):
    """Run one test file in an isolated `python -I` subprocess and return its TestResult."""
    exe = python_exe or sys.executable
    script_arg = os.path.normpath(rel)
    effective_timeout = timeout if (timeout is not None and timeout > 0) else None
    popen_kwargs = {
        "cwd": root,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        popen_kwargs["start_new_session"] = True

    job = _Win32Job() if os.name == "nt" else None
    t0 = time.monotonic()
    try:
        proc = subprocess.Popen([exe, "-I", script_arg], **popen_kwargs)
    except OSError as exc:
        if job is not None:
            job.close()
        elapsed = time.monotonic() - t0
        return TestResult(
            path=rel,
            returncode=127,
            elapsed=elapsed,
            output=f"[run_stdlib_tests] failed to start {rel}: {exc}\n",
        )

    if job is not None:
        job.assign(proc)

    try:
        stdout_bytes, _ = proc.communicate(timeout=effective_timeout)
        elapsed = time.monotonic() - t0
        return TestResult(
            path=rel,
            returncode=proc.returncode,
            elapsed=elapsed,
            output=_decode_bytes(stdout_bytes),
        )
    except subprocess.TimeoutExpired:
        _terminate_process_tree(proc, job)
        partial_bytes = b""
        try:
            partial_bytes, _ = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            if proc.stdout is not None:
                try:
                    proc.stdout.close()
                except OSError:
                    pass
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        elapsed = time.monotonic() - t0
        partial = _decode_bytes(partial_bytes)
        if partial and not partial.endswith("\n"):
            partial += "\n"
        partial += f"[run_stdlib_tests] timed out after {effective_timeout}s\n"
        return TestResult(
            path=rel,
            returncode=124,
            elapsed=elapsed,
            output=partial,
            timed_out=True,
        )
    except BaseException:
        _terminate_process_tree(proc, job)
        if proc.stdout is not None:
            try:
                proc.stdout.close()
            except OSError:
                pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        raise
    finally:
        if job is not None:
            job.close()


def run_suite(
    root,
    files,
    jobs,
    python_exe=None,
    timeout=DEFAULT_TIMEOUT,
    stream=None,
):
    """Run all selected test files in parallel, report all failures, and return an exit code."""
    out = stream if stream is not None else sys.stdout
    if not files:
        out.write("::error::run_stdlib_tests discovered 0 standard-library test files\n")
        out.flush()
        return 1

    workers = max(1, min(jobs, len(files)))
    wall_start = time.monotonic()
    failed = []

    interrupted = False
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
    try:
        futures = [
            executor.submit(run_one, root, rel, python_exe, timeout)
            for rel in files
        ]
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            block = [f"== {res.path}\n"]
            if res.output:
                block.append(res.output if res.output.endswith("\n") else res.output + "\n")
            if res.returncode != 0:
                failed.append(res)
            out.write("".join(block))
            out.flush()
    except BaseException:
        interrupted = True
        raise
    finally:
        executor.shutdown(wait=not interrupted, cancel_futures=interrupted)

    wall = time.monotonic() - wall_start
    failed.sort(key=lambda r: r.path)
    passed = len(files) - len(failed)
    out.write(
        f"\nRan {len(files)} stdlib test files in {wall:.2f}s ({workers} workers): "
        f"{passed} passed, {len(failed)} failed.\n"
    )
    for res in failed:
        if res.timed_out:
            detail = f"timed out after {timeout}s"
        else:
            detail = f"failed (exit {res.returncode})"
        out.write(f"::error file={res.path}::{res.path} {detail}\n")
    out.flush()
    return 1 if failed else 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the standard-library unit tests in parallel with `python -I`."
    )
    parser.add_argument(
        "--root",
        default=REPO_ROOT,
        help="Repository root directory (default: parent of scripts/).",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=os.cpu_count() or 1,
        help="Maximum number of test files to run in parallel (default: CPU count).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"Per-file timeout in seconds; 0 disables (default: {DEFAULT_TIMEOUT}).",
    )
    parser.add_argument(
        "--tests-dir",
        default=DEFAULT_TESTS_DIR,
        help=f"Test directory relative to --root (default: {DEFAULT_TESTS_DIR}).",
    )
    parser.add_argument(
        "--prune-dir",
        default=DEFAULT_PRUNE_DIR,
        help=f"Directory relative to --root to skip (default: {DEFAULT_PRUNE_DIR}).",
    )
    parser.add_argument(
        "--glob",
        default=DEFAULT_GLOB,
        help=f"Filename glob for test files (default: {DEFAULT_GLOB}).",
    )
    parser.add_argument(
        "--pytest-markers",
        default=None,
        help="Regex marking pytest-handled files (default: $PYTEST_FILE_MARKERS or CI default).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print discovered stdlib test files and exit without running them.",
    )
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    if args.timeout < 0:
        parser.error("--timeout must be >= 0")
    return args


def main(argv=None):
    args = parse_args(argv)
    files = discover_tests(
        root=args.root,
        tests_dir=args.tests_dir,
        prune_dir=args.prune_dir,
        pattern=args.glob,
        pytest_markers=args.pytest_markers,
    )
    if args.list:
        for rel in files:
            sys.stdout.write(f"{rel}\n")
        return 0
    try:
        return run_suite(
            root=args.root,
            files=files,
            jobs=args.jobs,
            timeout=args.timeout,
        )
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    sys.exit(main())
