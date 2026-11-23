"""Parallel SHA-256 manifest integrity verifier with progress reporting.

Verifies a target directory against a SHA-256 JSON manifest. Files are verified
concurrently using up to --jobs threads (defaulting to half the CPU count; 1 = serial).
Live progress is reported continuously during verification.

Usage:
    python -I manifest_verifier.py <target_dir> [--manifest FILE] [--jobs N]
                                   [--cache CACHEFILE] [--quiet] [--verbose] [--json] [--strict]
    python -I manifest_verifier.py verify <target_dir> [--manifest FILE] [--jobs N]
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
import time
from typing import Callable, Optional, TextIO

CHUNK = 8 * 1024 * 1024
DEFAULT_MANIFEST = os.path.join(os.path.dirname(__file__), "..", "reports", "manifest.json")
# A cache entry is written only for a file whose modification time is older than this: a file changed within the
# same timestamp tick as the run could keep its stamps, which is the "racy" window git leaves alone for the same
# reason. A cache is tied to one manifest: re-hashing a different manifest ignores it.
CACHE_VERSION = 1
CACHE_QUIET_NS = 3 * 1_000_000_000


def default_jobs() -> int:
    return max(1, (os.cpu_count() or 2) // 2)


def positive_int(text: str) -> int:
    try:
        val = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid int value: {text!r}") from None
    if val < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return val


def below_normal() -> None:
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            k32 = ctypes.windll.kernel32
            k32.GetCurrentProcess.argtypes = ()
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            k32.SetPriorityClass.argtypes = (wintypes.HANDLE, wintypes.DWORD)
            k32.SetPriorityClass.restype = wintypes.BOOL
            k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00004000)  # BELOW_NORMAL_PRIORITY_CLASS
        except Exception:
            pass


class ManifestError(Exception):
    """Raised when a manifest file is missing, unparseable, or invalid."""


def _file_stamp(st: os.stat_result) -> tuple[int, int, int]:
    return st.st_size, st.st_mtime_ns, st.st_ctime_ns


def sha256_file(path: str, chunk_size: int = CHUNK, *, expected_size: int | None = None) -> str:
    """Hash a stable open file, refusing a resize, rewrite or path replacement during the read."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    h = hashlib.sha256()
    with open(path, "rb", buffering=0) as f:
        before = os.fstat(f.fileno())
        if expected_size is not None and before.st_size != expected_size:
            raise OSError(f"{path}: file size changed before hashing")
        remaining = before.st_size
        if remaining:
            with memoryview(bytearray(min(chunk_size, remaining))) as buf:
                while remaining:
                    count = f.readinto(buf[:remaining])
                    if not count:
                        break
                    h.update(buf[:count])
                    remaining -= count
        # Compare handle timestamps to each other; Windows may report different times by path.
        # The final path stat checks that the name still refers to the same open file.
        if (remaining or f.read(1) or _file_stamp(os.fstat(f.fileno())) != _file_stamp(before)
                or not os.path.samestat(before, os.stat(path))):
            raise OSError(f"{path}: file changed while hashing")
    return h.hexdigest()


def walk(target_dir: str, *, exclude: str | None = None, excludes: Optional[list[str]] = None):
    """Yields (relative_path, absolute_path) using forward slashes, excluding the selected manifest and cache.

    `exclude` is the manifest (the historical single exclusion); `excludes` adds paths that belong to the tool
    rather than to the tree under test, such as a cache file written inside the game folder."""
    def scan_failed(error: OSError) -> None:
        raise error

    skipped: set[tuple[str, str]] = set()
    for path in ([exclude] if exclude is not None else []) + list(excludes or []):
        directory, name = os.path.split(os.path.abspath(path))
        skipped.add((os.path.normcase(os.path.realpath(directory)), os.path.normcase(name)))

    # os.walk otherwise treats missing or unreadable directories as empty.
    for root, _, files in os.walk(target_dir, onerror=scan_failed):
        root_key = os.path.normcase(os.path.realpath(root))
        for name in sorted(files):
            if (root_key, os.path.normcase(name)) in skipped:
                continue
            abs_p = os.path.join(root, name)
            rel_p = os.path.relpath(abs_p, target_dir).replace(os.sep, "/")
            yield rel_p, abs_p


def _file_stamp_ns(st: os.stat_result) -> dict:
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "ctime_ns": st.st_ctime_ns}


def load_cache(path: str, manifest_sha256: str) -> dict[str, dict]:
    """The cached verdicts that belong to this manifest, or {} for a missing, broken, stale or foreign cache."""
    try:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, ValueError):
        return {}
    if (not isinstance(doc, dict) or doc.get("version") != CACHE_VERSION
            or doc.get("manifest_sha256") != manifest_sha256):
        return {}
    files = doc.get("files")
    if not isinstance(files, dict):
        return {}
    entries: dict[str, dict] = {}
    for rel, entry in files.items():
        if not isinstance(rel, str) or not isinstance(entry, dict):
            continue
        if not isinstance(entry.get("size"), int) or not isinstance(entry.get("sha256"), str):
            continue
        if not isinstance(entry.get("mtime_ns"), int) or not isinstance(entry.get("ctime_ns"), int):
            continue
        entries[rel.replace("\\", "/")] = entry
    return entries


def write_cache(path: str, manifest_sha256: str, files: dict[str, dict]) -> None:
    """Replace `path` atomically with an LF-terminated cache; a failure keeps the previous file."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".manifest-cache-", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump({"version": CACHE_VERSION, "manifest_sha256": manifest_sha256, "files": files},
                      f, indent=0, sort_keys=True)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise



def load_manifest(path: str) -> dict[str, dict]:
    """Loads and validates a manifest file.

    Supports:
      {"files": {<rel_path>: {"size": <int>, "sha256": <str>}, ...}}
      or flat:
      {<rel_path>: {"size": <int>, "sha256": <str>}, ...}

    Rejects repeated JSON keys and normalizes relative paths to forward
    slashes and hashes to lowercase hex.
    """
    try:
        mode = os.stat(path).st_mode
    except FileNotFoundError:
        raise ManifestError(f"manifest file not found: {path}") from None
    except OSError as err:
        raise ManifestError(f"cannot read manifest {path}: {err}") from None
    if not stat.S_ISREG(mode):
        raise ManifestError(f"manifest path is not a regular file: {path}")

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ManifestError(f"{path}: duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f, object_pairs_hook=unique_object)
    except ValueError as err:
        raise ManifestError(f"{path} is not valid JSON ({err})") from None
    except OSError as err:
        raise ManifestError(f"cannot read manifest {path}: {err}") from None

    if not isinstance(doc, dict):
        raise ManifestError(f"{path}: root must be a JSON object")

    nested = "files" in doc
    if nested:
        raw_files = doc["files"]
        if not isinstance(raw_files, dict):
            raise ManifestError(f"{path}: 'files' must be a JSON object")
    else:
        raw_files = doc

    entries: dict[str, dict] = {}
    for raw_rel, info in raw_files.items():
        if not nested and raw_rel in ("created", "title", "version", "note"):
            continue
        rel = str(raw_rel).replace("\\", "/")
        if rel in entries:
            raise ManifestError(f"{path}: duplicate normalized path {rel!r}")
        if isinstance(info, dict):
            size = info.get("size")
            digest = info.get("sha256")
            if not isinstance(digest, str):
                raise ManifestError(f"{path}: entry {rel!r} missing 'sha256' string")
            if size is not None and not (isinstance(size, int) and not isinstance(size, bool) and size >= 0):
                raise ManifestError(f"{path}: entry {rel!r} has invalid integer 'size'")
        elif isinstance(info, str):
            size, digest = None, info
        else:
            raise ManifestError(f"{path}: entry {rel!r} must be an object or digest string")
        if re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None:
            raise ManifestError(f"{path}: entry {rel!r} needs a SHA-256 digest of 64 hexadecimal characters")
        entries[rel] = {"size": size, "sha256": digest.lower()}
    return entries


@dataclass
class ProgressInfo:
    completed_files: int
    total_files: int
    completed_bytes: int
    total_bytes: int
    rate_mib: float
    pct: float
    rel_path: str
    status: str


@dataclass
class ManifestReport:
    passed: bool
    total_manifest_files: int
    matched: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    size_mismatch: list[str] = field(default_factory=list)
    hash_mismatch: list[str] = field(default_factory=list)
    unreadable: list[tuple[str, str]] = field(default_factory=list)
    untracked: list[str] = field(default_factory=list)
    problems: dict[str, str] = field(default_factory=dict)
    bytes_verified: int = 0
    cached_count: int = 0
    bytes_cached: int = 0
    elapsed_seconds: float = 0.0
    strict: bool = False

    def count_problems(self, *, strict: Optional[bool] = None) -> int:
        count = len(self.missing) + len(self.size_mismatch) + len(self.hash_mismatch) + len(self.unreadable)
        if strict is None:
            strict = self.strict
        if strict:
            count += len(self.untracked)
        return count

    @property
    def problem_count(self) -> int:
        return self.count_problems()

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "total_manifest_files": self.total_manifest_files,
            "matched_count": len(self.matched),
            "problem_count": self.problem_count,
            "missing": self.missing,
            "size_mismatch": self.size_mismatch,
            "hash_mismatch": self.hash_mismatch,
            "unreadable": self.unreadable,
            "untracked": self.untracked,
            "problems": self.problems,
            "bytes_verified": self.bytes_verified,
            "cached_count": self.cached_count,
            "bytes_cached": self.bytes_cached,
            "elapsed_seconds": round(self.elapsed_seconds, 4),
        }


def verify_manifest(
    target_dir: str,
    manifest_path: str = DEFAULT_MANIFEST,
    jobs: int = 1,
    progress_callback: Optional[Callable[[ProgressInfo], None]] = None,
    progress_stream: Optional[TextIO] = None,
    strict: bool = False,
    cache_path: Optional[str] = None,
) -> ManifestReport:
    """Verifies target_dir against manifest_path, optionally in parallel.

    With `cache_path`, a file whose size, mtime and ctime match a cached verdict for the same manifest (and whose
    cached digest is the expected one) is not hashed again; only files that hashed OK are remembered, and only once
    their modification time is older than the racy window. A missing, unreadable or foreign cache is ignored and
    simply means a full verification."""
    try:
        mode = os.stat(target_dir).st_mode
    except FileNotFoundError:
        raise ManifestError(f"target directory does not exist: {target_dir}") from None
    except OSError as err:
        raise ManifestError(f"cannot access target directory {target_dir}: {err}") from None
    if not stat.S_ISDIR(mode):
        raise ManifestError(f"target path is not a directory: {target_dir}")

    want = load_manifest(manifest_path)
    manifest_sha256 = sha256_file(manifest_path) if cache_path else ""
    cached = load_cache(cache_path, manifest_sha256) if cache_path else {}
    excludes = [cache_path] if cache_path else None
    have = dict(walk(target_dir, exclude=manifest_path, excludes=excludes))
    if not want and have:
        # A cut-short create or a mistyped --manifest would otherwise print OK without checking a file.
        raise ManifestError(f"{manifest_path} lists no files, but {target_dir} holds {len(have)}")
    untracked = sorted(set(have.keys()) - set(want.keys()))

    problems: dict[str, str] = {}
    unreadable_map: dict[str, str] = {}
    pending: list[tuple[str, str, int, str]] = []
    stamps: dict[str, tuple[int, int]] = {}
    sizes: dict[str, int] = {}
    total_manifest_bytes = 0
    cached_count = 0
    cached_bytes = 0
    quiet_before = time.time_ns() - CACHE_QUIET_NS

    for rel in sorted(want.keys()):
        info = want[rel]
        abs_p = have.get(rel)
        if abs_p is None:
            problems[rel] = "MISSING"
            continue
        try:
            st = os.stat(abs_p)
        except OSError as err:
            problems[rel] = "UNREADABLE"
            unreadable_map[rel] = str(err)
            continue
        actual_size = st.st_size
        expected_size = info.get("size")
        if expected_size is not None and actual_size != expected_size:
            problems[rel] = "SIZE"
            continue
        expected_sha256 = info["sha256"]
        entry = cached.get(rel)
        if (entry is not None and entry["size"] == actual_size and entry["sha256"] == expected_sha256
                and entry["mtime_ns"] == st.st_mtime_ns and entry["ctime_ns"] == st.st_ctime_ns):
            cached_count += 1
            cached_bytes += actual_size
            continue
        stamps[rel] = (st.st_mtime_ns, st.st_ctime_ns)
        sizes[rel] = actual_size
        pending.append((rel, abs_p, actual_size, expected_sha256))
        total_manifest_bytes += actual_size

    total_pending_files = len(pending)
    completed_files = 0
    completed_bytes = 0
    bytes_verified = 0
    t0 = time.time()

    def hash_worker(rel: str, abs_p: str, size: int, expected_hash: str) -> tuple[str, int, str, Optional[str]]:
        try:
            actual_hash = sha256_file(abs_p, expected_size=size)
            if actual_hash.lower() == expected_hash:
                return rel, size, "OK", None
            return rel, size, "CHANGED", None
        except OSError as err:
            return rel, size, "UNREADABLE", str(err)

    def on_file_done(rel: str, size: int, status: str, err_msg: Optional[str]) -> None:
        nonlocal completed_files, completed_bytes, bytes_verified
        completed_files += 1
        completed_bytes += size
        if status != "UNREADABLE":
            bytes_verified += size
        if status != "OK":
            problems[rel] = status
            if status == "UNREADABLE" and err_msg:
                unreadable_map[rel] = err_msg

        elapsed = max(time.time() - t0, 1e-6)
        rate_mib = (completed_bytes / elapsed) / (1024 * 1024)
        pct = (
            completed_bytes / total_manifest_bytes
            if total_manifest_bytes > 0
            else (completed_files / total_pending_files if total_pending_files > 0 else 1.0)
        )
        info = ProgressInfo(
            completed_files=completed_files,
            total_files=total_pending_files,
            completed_bytes=completed_bytes,
            total_bytes=total_manifest_bytes,
            rate_mib=rate_mib,
            pct=pct,
            rel_path=rel,
            status=status,
        )
        if progress_callback is not None:
            progress_callback(info)
        if progress_stream is not None:
            progress_stream.write(f"[{pct:6.1%}] {completed_files}/{total_pending_files} files {rate_mib:7.0f} MiB/s\n")
            progress_stream.flush()

    if jobs <= 1:
        for rel, abs_p, size, expected_hash in pending:
            rel_res, size_res, status, err_msg = hash_worker(rel, abs_p, size, expected_hash)
            on_file_done(rel_res, size_res, status, err_msg)
    else:
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            futures = [
                pool.submit(hash_worker, rel, abs_p, size, expected_hash)
                for rel, abs_p, size, expected_hash in pending
            ]
            for future in as_completed(futures):
                rel_res, size_res, status, err_msg = future.result()
                on_file_done(rel_res, size_res, status, err_msg)

    matched: list[str] = []
    missing: list[str] = []
    size_mismatch: list[str] = []
    hash_mismatch: list[str] = []
    unreadable: list[tuple[str, str]] = []

    for rel in sorted(want.keys()):
        kind = problems.get(rel)
        if kind == "MISSING":
            missing.append(rel)
        elif kind == "SIZE":
            size_mismatch.append(rel)
        elif kind == "CHANGED":
            hash_mismatch.append(rel)
        elif kind == "UNREADABLE":
            unreadable.append((rel, unreadable_map.get(rel, "unreadable file")))
        else:
            matched.append(rel)

    problem_count = len(missing) + len(size_mismatch) + len(hash_mismatch) + len(unreadable)
    passed = (problem_count == 0) and (not strict or len(untracked) == 0)
    elapsed_total = time.time() - t0

    if cache_path:
        files: dict[str, dict] = {}
        for rel in sorted(want.keys()):
            if rel in problems:
                continue
            if rel in cached and rel not in stamps:
                files[rel] = cached[rel]
                continue
            mtime_ns, ctime_ns = stamps[rel]
            if mtime_ns <= quiet_before:
                files[rel] = {"size": sizes[rel], "mtime_ns": mtime_ns, "ctime_ns": ctime_ns,
                              "sha256": want[rel]["sha256"]}
        try:
            write_cache(cache_path, manifest_sha256, files)
        except OSError:
            # The cache is an optimisation: a read-only or full disk must not turn a good verify into a failure.
            pass

    return ManifestReport(
        passed=passed,
        total_manifest_files=len(want),
        matched=matched,
        missing=missing,
        size_mismatch=size_mismatch,
        hash_mismatch=hash_mismatch,
        unreadable=unreadable,
        untracked=untracked,
        problems={rel: problems[rel] for rel in sorted(problems.keys())},
        bytes_verified=bytes_verified,
        cached_count=cached_count,
        bytes_cached=cached_bytes,
        elapsed_seconds=elapsed_total,
        strict=strict,
    )


def print_report(
    report: ManifestReport,
    out: TextIO = sys.stdout,
    verbose: bool = False,
    strict: Optional[bool] = None,
) -> None:
    unreadable_reasons = dict(report.unreadable)
    for rel in sorted(report.problems.keys()):
        kind = report.problems[rel]
        if kind == "UNREADABLE":
            reason = unreadable_reasons.get(rel, "unreadable file")
            out.write(f"{kind} {rel}: {reason}\n")
        else:
            out.write(f"{kind:<10}{rel}\n")
    for rel in report.untracked:
        out.write(f"{'NEW':<10}{rel}\n")

    bad = report.count_problems(strict=strict)

    if bad == 0:
        if verbose:
            gib = report.bytes_verified / (1024**3)
            out.write(f"OK ({len(report.matched)} files, {gib:.2f} GiB in {report.elapsed_seconds:.2f}s)\n")
        else:
            out.write("OK\n")
    else:
        out.write(f"{bad} problem(s)\n")
    out.flush()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="manifest_verifier.py",
        description="Verify SHA-256 manifest integrity in parallel with progress reporting.",
    )
    ap.add_argument("cmd_or_dir", metavar="PATH", help="Target directory to verify (or 'verify' subcommand)")
    ap.add_argument("target_dir", nargs="?", default=None, metavar="TARGET_DIR",
                    help="Target directory (if 'verify' subcommand was used)")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST, help=f"Manifest JSON file (default: {DEFAULT_MANIFEST})")
    ap.add_argument(
        "--jobs", "-j", type=positive_int, default=default_jobs(), metavar="N",
        help="Hash up to N files concurrently (default: %(default)s; 1 = serial)",
    )
    ap.add_argument("--quiet", "-q", action="store_true", help="Suppress live progress output")
    ap.add_argument("--progress-stream", choices=["stderr", "stdout"], default="stderr",
                    help="Stream for live progress reporting (default: stderr)")
    ap.add_argument("--verbose", "-v", action="store_true", help="Print verbose summary statistics")
    ap.add_argument("--json", action="store_true", help="Output verification report as JSON")
    ap.add_argument("--strict", action="store_true", help="Fail if untracked (NEW) files exist")
    ap.add_argument("--cache", metavar="CACHEFILE", default=None,
                    help="Remember files that verified, and skip re-hashing them while size, mtime and ctime are "
                         "unchanged and the manifest is the same (off by default)")
    return ap


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cmd_or_dir == "verify":
        game_dir = args.target_dir
        if not game_dir:
            parser.error("target directory required when using 'verify'")
    else:
        if args.target_dir is not None:
            parser.error(f"unrecognized arguments: {args.target_dir}")
        game_dir = args.cmd_or_dir

    below_normal()

    progress_stream = None
    if not args.quiet and not args.json:
        progress_stream = sys.stdout if args.progress_stream == "stdout" else sys.stderr

    try:
        report = verify_manifest(
            target_dir=game_dir,
            manifest_path=args.manifest,
            jobs=args.jobs,
            progress_stream=progress_stream,
            strict=args.strict,
            cache_path=args.cache,
        )
    except ManifestError as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 2
    except OSError as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print_report(report, out=sys.stdout, verbose=args.verbose)

    return 0 if report.passed else 1


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
