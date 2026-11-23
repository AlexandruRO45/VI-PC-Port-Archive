"""Run the Windows build's CTest tests under Wine, on Linux.

Usage:
    python -I scripts/run_wine_tests.py --tests ctest.json --windows-root D:/a/Leonida/Leonida
        [--root DIR] [--wine CMD] [--require NAME ...] [--timeout SECONDS] [--jobs N] [--summary FILE] [--label TEXT]

--tests is the output of `ctest --show-only=json-v1` from the Windows build, and --windows-root the checkout it was
made in (forward slashes, as `cygpath -m` prints it). Paths under it are mapped to --root, the Linux checkout (default:
this repository), which must hold the Windows build's runtime/build folder at the same place.

Some tests open files through absolute paths compiled into them (the GLSL test shaders, LEONIDA_TEST_*_SPV under
runtime/build/generated). For those, the drive of --windows-root is pointed at --root in the Wine prefix for the run
($WINEPREFIX, else ~/.wine), unless that drive is already in use there.

Every test is reported. The exit code is 1 if a test named by --require fails or is missing, else 0: the
others show what does and doesn't work under Wine yet without failing the job.
--timeout caps each process; a shorter positive CTest TIMEOUT takes precedence for that test.
--jobs N runs up to N tests at once (default 1, one at a time in the order listed). Tests of one program never overlap
(they may share a scratch folder named after it), nor do tests with a CTest DEPENDS, fixture or RESOURCE_LOCK (CI runs
them in order); a RUN_SERIAL test runs alone. The results are the same; only the order of the report lines can differ.
"""
import argparse
import concurrent.futures
from graphlib import CycleError, TopologicalSorter
import json
import math
import ntpath
import os
import re
import shlex
import signal
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# "wine: Unhandled page fault ... starting debugger..." and "wine: Unimplemented function ... starting debugger...".
ABORTED = re.compile(r"^wine: .*starting debugger\.\.\.", re.MULTILINE)


def _make_parent(path: str) -> None:
    """Create the folder a report lives in: the run is over by the time the summary is written."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def norm(path: str) -> str:
    return ntpath.normpath(path).replace("\\", "/").rstrip("/")


def map_path(arg: str, windows_root: str, root: str) -> str | None:
    """The Linux path for a path under the Windows checkout, or None if it isn't one."""
    a = norm(arg)
    w = norm(windows_root)
    if a.lower() == w.lower():
        return root
    if a.lower().startswith(w.lower() + "/"):
        return os.path.join(root, *a[len(w) + 1:].split("/"))
    return None


class DriveMap:
    """Points the drive of the Windows checkout at the Linux one inside the Wine prefix, for the run.

    A test that opens D:/a/Leonida/Leonida/runtime/build/generated/x.spv (a path compiled into it) then reads
    <root>/runtime/build/generated/x.spv. The drive is a dosdevices link to a scratch folder holding a/Leonida/Leonida
    as a link to the root; both are removed afterwards. A drive the prefix already has (c:, z:, or one set up by
    hand) is left alone, and so is a root without a drive letter.
    """

    def __init__(self, windows_root: str, root: str, prefix: str):
        self.windows_root, self.root, self.prefix = norm(windows_root), os.path.abspath(root), prefix
        self.link = self.scratch = None
        self.note = ""

    def __enter__(self):
        m = re.fullmatch(r"([A-Za-z]):/(.+)", self.windows_root)
        if not m:
            self.note = f"{self.windows_root} has no drive letter: no drive mapped"
            return self
        drive, rest = m.group(1).lower(), [p for p in m.group(2).split("/") if p]
        dosdevices = os.path.join(self.prefix, "dosdevices")
        link = os.path.join(dosdevices, drive + ":")
        if not os.path.isdir(dosdevices):
            self.note = f"no Wine prefix at {self.prefix}: drive {drive}: not mapped"
            return self
        already_set = f"drive {drive}: is already set in {dosdevices}: left as it is"
        if os.path.lexists(link):
            self.note = already_set
            return self
        self.scratch = tempfile.mkdtemp(prefix="wine-drive-")
        try:
            parent = os.path.join(self.scratch, *rest[:-1])
            os.makedirs(parent, exist_ok=True)
            os.symlink(self.root, os.path.join(parent, rest[-1]))
            try:
                os.symlink(self.scratch, link)
            except FileExistsError:
                self.note = already_set
                return self
            self.link = link
        finally:
            # __exit__ only runs after a successful __enter__.
            if self.link is None:
                shutil.rmtree(self.scratch, ignore_errors=True)
                self.scratch = None
        self.note = f"drive {drive}: maps {self.windows_root} to {self.root} for this run"
        return self

    def __exit__(self, *exc):
        if self.link and os.path.islink(self.link):
            os.unlink(self.link)
        if self.scratch:
            shutil.rmtree(self.scratch, ignore_errors=True)   # removes the links, never what they point to
        return False


class WineDebugger:
    """Makes a process that Wine aborts end at once, for the run, instead of waiting for its crash debugger.

    Wine's default crash handler (HKLM\\Software\\Microsoft\\Windows NT\\CurrentVersion\\AeDebug, "winedbg --auto") is
    started for an unhandled exception, such as a call to a function this Wine does not implement ("Unimplemented
    function KERNEL32.dll.CopyFile2 ... starting debugger..."). Under Xvfb it never ends, so the test sat until
    --timeout (120 s each). With Debugger set to "false" the process dies in under a second with a non-zero exit code,
    which run_test reports as the failure it was (see ABORTED). The old value is put back afterwards.

    The value is only touched when it is exactly the REG_SZ default (DEFAULT_DEBUGGER), or when the query shows it is
    absent. Anything else (another command, an empty or REG_EXPAND_SZ value, a query that failed for another reason,
    a missing prefix, keep=True) is left alone, and so is a test whose CTest ENVIRONMENT sets its own WINEPREFIX. A
    failing "wine reg" never fails the run: a restore that fails is tried twice, then reported with the command that
    restores it by hand. A run killed with SIGKILL cannot restore it, and two runs on one prefix must not overlap.
    """

    KEY = r"HKLM\Software\Microsoft\Windows NT\CurrentVersion\AeDebug"
    DEFAULT_DEBUGGER = "winedbg --auto %ld %ld"

    def __init__(self, wine: list[str], prefix: str, keep: bool = False):
        self.wine, self.prefix, self.keep = list(wine), prefix, keep
        self.active, self.previous = False, None
        self.note = ""

    def _reg(self, *argv: str) -> tuple[int, str] | None:
        try:
            proc = subprocess.run(self.wine + ["reg", *argv], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return proc.returncode, proc.stdout.decode("utf-8", "replace")

    def _set(self, value: str) -> bool:
        result = self._reg("add", self.KEY, "/v", "Debugger", "/t", "REG_SZ", "/d", value, "/f")
        return result is not None and result[0] == 0

    def __enter__(self):
        if self.keep or not os.path.isdir(os.path.join(self.prefix, "dosdevices")):
            return self
        query = self._reg("query", self.KEY, "/v", "Debugger")
        if query is None:
            self.note = "wine reg did not run: the crash debugger is left as it is"
            return self
        code, out = query
        found = re.search(r"^\s*Debugger\s+(REG_\w+)(?:\s+(.*?))?\s*$", out, re.MULTILINE)
        if code == 0 and found:
            kind, data = found.group(1), found.group(2) or ""
            if (kind, data) != ("REG_SZ", self.DEFAULT_DEBUGGER):
                self.note = f"the prefix has its own crash debugger setting ({kind} {data!r}): left as it is"
                return self
            self.previous = data
        elif code == 0 or "unable to find" not in out.lower():
            self.note = "could not read the crash debugger setting: left as it is"
            return self
        if not self._set("false"):
            self.note = "could not set the crash debugger: a crashing test waits for winedbg"
            return self
        self.active = True
        # Printed before the tests run, so a run killed with SIGKILL still left the exact way back in its log.
        self.note = ("Wine's crash debugger is off for this run: an aborted test ends at once. "
                     f"If this run is killed, put it back with: {self.restore_command()}")
        return self

    def restore_command(self) -> str:
        """The command that puts the prefix back as it was: the default value again, or no value when there was none."""
        if self.previous is not None:
            return (f'WINEPREFIX="{self.prefix}" wine reg add "{self.KEY}" /v Debugger /t REG_SZ '
                    f'/d "{self.previous}" /f')
        return f'WINEPREFIX="{self.prefix}" wine reg delete "{self.KEY}" /v Debugger /f'

    def __exit__(self, *exc):
        if not self.active:
            return False
        for _ in range(2):
            if self.previous is not None:
                restored = self._set(self.previous)
            else:
                result = self._reg("delete", self.KEY, "/v", "Debugger", "/f")
                restored = result is not None and result[0] == 0
            if restored:
                return False
        print(f"WARNING: could not restore Wine's crash debugger in {self.prefix}; it is still 'false'. "
              f"Restore it with: {self.restore_command()}", file=sys.stderr, flush=True)
        return False


def wine_path(path: str) -> str:
    """Wine's Z: drive is the Unix root, so a Windows program reads /x/y as Z:\\x\\y."""
    return "Z:" + os.path.abspath(path).replace("/", "\\")


def properties(test: dict) -> dict:
    return {p["name"]: p["value"] for p in test.get("properties", [])}


def fixture_dependencies(tests: list[dict]) -> dict[int, tuple[int, ...]]:
    """Find the listed setup tests that must finish before each fixture consumer."""
    metadata = [properties(test) for test in tests]
    setups: dict[str, list[int]] = {}
    for index, props in enumerate(metadata):
        for fixture in props.get("FIXTURES_SETUP", []):
            setups.setdefault(fixture, []).append(index)
    return {
        index: tuple(dict.fromkeys(
            setup
            for fixture in props.get("FIXTURES_REQUIRED", [])
            for setup in setups.get(fixture, ())
            if setup != index
        ))
        for index, props in enumerate(metadata)
    }


def truthy(value) -> bool:
    """A ctest boolean property. The JSON carries it as a string, so "0" must stay false."""
    return str(value).strip().lower() in ("1", "on", "true", "yes")


def regex_list(value: str | list[str]) -> list[str]:
    """Treat a single string as one output expression, like a one-element CTest list."""
    return [value] if isinstance(value, str) else value


def run_test(test: dict, args, wine: list[str], *, failed_setups: tuple[str, ...] = ()) -> tuple[str, str]:
    """Returns (status, detail): status is pass, fail or skip."""
    props = properties(test)
    if truthy(props.get("DISABLED")):
        return "skip", "disabled by CTest"
    command = test.get("command")
    if not command:
        return "fail", "no command (target not built?)"
    exe = map_path(command[0], args.windows_root, args.root)
    if exe is None:
        # A test that runs a tool of the Windows runner (cmake -P, python) rather than one this build made.
        return "skip", f"runs {command[0]}, not a program from the build"
    if not os.path.isfile(exe):
        return "fail", f"executable not in the artifact: {command[0]}"
    if failed_setups:
        return "fail", "failed fixture setup: " + ", ".join(failed_setups)
    argv = list(wine) + [exe]
    for a in command[1:]:
        mapped = map_path(a, args.windows_root, args.root)
        argv.append(wine_path(mapped) if mapped is not None else a)
    listed_cwd = props.get("WORKING_DIRECTORY")
    cwd = map_path(listed_cwd, args.windows_root, args.root) if listed_cwd else None
    if cwd is not None and not os.path.isdir(cwd):
        return "fail", f"CTest working directory is not a directory: {listed_cwd} (mapped to {cwd})"
    if cwd is None:
        cwd = os.path.dirname(exe)
    assignments = props.get("ENVIRONMENT", [])
    if not isinstance(assignments, list):
        return "fail", "invalid CTest ENVIRONMENT: expected a list of NAME=value entries"
    env = os.environ.copy()
    for assignment in assignments:
        if not isinstance(assignment, str):
            return "fail", "invalid CTest ENVIRONMENT entry: expected NAME=value"
        key, equals, value = assignment.partition("=")
        if not key or not equals or "\0" in assignment:
            return "fail", "invalid CTest ENVIRONMENT entry: expected NAME=value"
        env[key] = value
    timeout = args.timeout
    if "TIMEOUT" in props:
        try:
            ctest_timeout = float(props["TIMEOUT"])
        except (TypeError, ValueError):
            return "fail", "invalid CTest TIMEOUT"
        if not math.isfinite(ctest_timeout) or ctest_timeout < 0:
            return "fail", "invalid CTest TIMEOUT"
        if ctest_timeout > 0:
            timeout = min(timeout, ctest_timeout)
    skip_code = props.get("SKIP_RETURN_CODE")
    if skip_code is not None:
        error = "invalid CTest SKIP_RETURN_CODE: expected an integer from 0 to 255"
        try:
            skip_code = int(str(skip_code))
        except ValueError:
            return "fail", error
        if not 0 <= skip_code <= 255:
            return "fail", error
    start = time.monotonic()
    try:
        proc = run_test_process(argv, cwd=cwd, env=env, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"").decode("utf-8", "replace")
        return "fail", f"timed out after {timeout} s\n{out}"
    except OSError as e:                  # a mistyped --wine, or a missing one: report the test, do not abort the run
        return "fail", f"could not start {argv[0]}: {e.strerror or e}"
    elapsed = time.monotonic() - start
    out = proc.stdout.decode("utf-8", "replace")
    code = proc.returncode
    if code < 0:
        # A negative return code is a host-signal termination (e.g. SIGSEGV), not a process exit code:
        # report it as a failure before SKIP_RETURN_CODE or WILL_FAIL can reinterpret it.
        return "fail", f"killed by signal {-code} in {elapsed:.1f} s\n{out}"
    if ABORTED.search(out):
        # Wine's own report of an unhandled exception. Before WineDebugger the crash debugger kept such a process
        # alive until the timeout, which is a failure whatever WILL_FAIL, the exit code or an output expression say.
        return "fail", f"aborted by Wine (exit {code} in {elapsed:.1f} s)\n{out}"
    if code == skip_code:
        return "skip", f"exit {code} (skip) in {elapsed:.1f} s\n{out}"
    detail = f"exit {code} in {elapsed:.1f} s\n{out}"
    skip_patterns = regex_list(props.get("SKIP_REGULAR_EXPRESSION", []))
    required = regex_list(props.get("PASS_REGULAR_EXPRESSION", []))
    forbidden = regex_list(props.get("FAIL_REGULAR_EXPRESSION", []))
    # A required match takes the place of the exit code. A forbidden match overrides either kind of success.
    try:
        if any(re.search(pattern, out, re.DOTALL) for pattern in skip_patterns):
            return "skip", f"matched CTest SKIP_REGULAR_EXPRESSION: {detail}"
        passed = any(re.search(pattern, out, re.DOTALL) for pattern in required) if required else code == 0
        if any(re.search(pattern, out, re.DOTALL) for pattern in forbidden):
            passed = False
    except re.error as e:
        return "fail", f"invalid output regular expression: {e}\n{detail}"
    if truthy(props.get("WILL_FAIL")):
        passed = not passed
    return ("pass" if passed else "fail"), detail


KILL_WAIT = 10.0   # seconds to collect a killed test's output before giving up on it


def run_test_process(argv: list[str], *, cwd: str, env: dict[str, str],
                     timeout: float) -> subprocess.CompletedProcess:
    """Runs one test in its own process group (its own session on POSIX): at the timeout, or on Ctrl-C, the test and
    everything it started (Wine's Windows children, launcher wrappers, a relaunched gta6rt, a helper) are killed, so
    nothing of it runs on beside the next tests on the shared prefix. The launcher may have exited while a helper
    still holds the output pipe, so the output after the kill is read for at most KILL_WAIT seconds. Raises
    TimeoutExpired with the output so far, as subprocess.run does."""
    group = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt"
             else {"start_new_session": True})
    options = dict(cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **group)
    with subprocess.Popen(argv, **options) as proc:
        try:
            out, _ = proc.communicate(timeout=timeout)
        except KeyboardInterrupt:
            kill_tree(proc)
            proc.wait()
            raise
        except subprocess.TimeoutExpired:
            kill_tree(proc)
            try:   # a descendant that escaped the kill may still hold the pipe: do not wait for it forever
                out, _ = proc.communicate(timeout=KILL_WAIT)
            except subprocess.TimeoutExpired as late:
                proc.kill()
                out = late.stdout
            raise subprocess.TimeoutExpired(argv, timeout, output=out) from None
        return subprocess.CompletedProcess(argv, proc.returncode, out)


def kill_tree(proc) -> None:
    """Stop a test and everything it started (its process group, or on Windows its tree)."""
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True, timeout=KILL_WAIT)
        else:
            os.killpg(proc.pid, signal.SIGKILL)
    except (OSError, subprocess.SubprocessError):
        proc.kill()


def positive_int(text: str) -> int:
    """--jobs: a whole number of at least 1."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid int value: {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


ORDERING_PROPERTIES = ("DEPENDS", "FIXTURES_SETUP", "FIXTURES_REQUIRED", "FIXTURES_CLEANUP", "RESOURCE_LOCK")


def plan_lanes(tests: list[dict], args) -> tuple[list[list[dict]], list[list[dict]]]:
    """Split the tests for --jobs > 1 into (alone, lanes): the tests CTest never runs beside another one (RUN_SERIAL)
    come first, one at a time; the rest are lanes that run side by side, each lane in list order. A lane holds the
    tests of one executable (they may share a scratch folder named after the program), and a single lane holds every
    test with an ordering of its own (DEPENDS, fixtures, RESOURCE_LOCK) and every test one of them names in DEPENDS,
    so the order CTest would keep is kept."""
    named = set()
    for test in tests:
        depends = properties(test).get("DEPENDS", [])
        named.update([depends] if isinstance(depends, str) else depends)
    def program(test):
        command = test.get("command") or [""]
        return map_path(command[0], args.windows_root, args.root) or command[0]

    def has_order(test):
        return test.get("name") in named or any(p in properties(test) for p in ORDERING_PROPERTIES)

    # the ordered lane also takes every test of a program that has an ordered test, so no two tests of one program
    # overlap even when only one of them has an ordering
    ordered_programs = {program(t) for t in tests if has_order(t)}
    alone, ordered, by_exe = [], [], {}
    for test in tests:
        if has_order(test) or program(test) in ordered_programs:
            ordered.append(test)
        elif truthy(properties(test).get("RUN_SERIAL")):
            alone.append([test])
        else:
            by_exe.setdefault(program(test), []).append(test)
    lanes = list(by_exe.values())
    if any(truthy(properties(t).get("RUN_SERIAL")) for t in ordered):
        # a RUN_SERIAL test inside an ordering: the whole ordered lane runs alone, in order, so the test still
        # follows its setup or the test it depends on and has nothing beside it
        alone.append(ordered)
    elif ordered:
        lanes.insert(0, ordered)
    return alone, lanes


def run_tests(tests: list[dict], args, wine: list[str], report, order: list[int],
              dependencies: dict[int, tuple[int, ...]]) -> dict[str, str]:
    """Runs every test and calls report(name, status, detail) as each one finishes; returns {name: status} in the
    order of `tests`. `order` puts every fixture setup before the tests that need it (fixture_dependencies), and a
    test whose setup failed fails without running. With --jobs 1 (the default) one test runs at a time in that order.
    With more, up to --jobs tests run at once (see plan_lanes; the setups and their consumers share the ordered lane,
    which keeps that order). Each test is bounded by its own timeout, so a test that hangs holds one worker for at most
    that long and the others go on; an error inside a worker fails that test, not the run."""
    done: dict[int, str] = {}   # by test index: two listed tests can share a name, and the setup that failed must count
    lock = threading.Lock()
    names = [test.get("name", "?") for test in tests]
    index_of = {id(test): index for index, test in enumerate(tests)}

    def run_one(test: dict) -> None:
        index = index_of[id(test)]
        name = names[index]
        with lock:
            failed_setups = tuple(names[s] for s in dependencies.get(index, ()) if done.get(s) == "fail")
        try:
            status, detail = run_test(test, args, wine, failed_setups=failed_setups)
        except Exception as e:
            if args.jobs <= 1:
                raise
            # beside other tests a bug in the runner fails this test; it must not leave the others unreported
            status, detail = "fail", f"the runner raised {type(e).__name__}: {e}"
        with lock:
            done[index] = status
            report(name, status, detail)

    in_order = [tests[index] for index in order]
    if args.jobs <= 1:
        for test in in_order:
            run_one(test)
    else:
        alone, lanes = plan_lanes(in_order, args)
        for lane in alone:
            for test in lane:
                run_one(test)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            for future in [pool.submit(lambda lane=lane: [run_one(t) for t in lane]) for lane in lanes]:
                future.result()
    return {names[index]: done[index] for index in range(len(tests))}


def positive_seconds(text: str) -> float:
    """--timeout: a finite value greater than 0, so subprocess.run never gets a bad timeout to raise on."""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid float value: {text!r}") from None
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError("must be a finite number")
    if value <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return value


def wine_command(text: str) -> list[str]:
    """--wine: split into a command the way a shell would, refusing blank or malformed values up front."""
    try:
        parts = shlex.split(text)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"malformed --wine: {e}") from None
    if not parts:
        raise argparse.ArgumentTypeError("must be a nonempty command")
    return parts


def valid_test_property(entry) -> bool:
    if not isinstance(entry, dict) or not isinstance(entry.get("name"), str) or "value" not in entry:
        return False
    name, value = entry["name"], entry["value"]
    if name == "WORKING_DIRECTORY":
        return isinstance(value, str) and "\0" not in value
    if name in ("PASS_REGULAR_EXPRESSION", "FAIL_REGULAR_EXPRESSION", "SKIP_REGULAR_EXPRESSION"):
        return isinstance(value, str) or (isinstance(value, list) and all(isinstance(p, str) for p in value))
    return True


def valid_test_row(test) -> bool:
    if not isinstance(test, dict) or not isinstance(test.get("name"), str):
        return False
    # CTest can omit the command for a target that was not built.
    command = test.get("command", [])
    if not isinstance(command, list) or not all(isinstance(arg, str) and "\0" not in arg for arg in command):
        return False
    entries = test.get("properties", [])
    return isinstance(entries, list) and all(valid_test_property(entry) for entry in entries)


def read_tests(path: str) -> list[dict] | None:
    """The tests in a `ctest --show-only=json-v1` file, or None when it is not one."""
    try:
        with open(path, encoding="utf-8-sig") as f:
            doc = json.load(f)
    except (OSError, ValueError) as e:
        print(f"ERROR: cannot read {path}: {e}", file=sys.stderr)
        return None
    tests = doc.get("tests") if isinstance(doc, dict) else None
    if not isinstance(tests, list) or not all(valid_test_row(t) for t in tests):
        print(f"ERROR: {path} is not the output of ctest --show-only=json-v1", file=sys.stderr)
        return None
    return tests


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tests", required=True, help="ctest --show-only=json-v1 output from the Windows build")
    ap.add_argument("--windows-root", required=True, help="the Windows checkout the tests were listed in")
    ap.add_argument("--root", type=os.path.abspath, default=ROOT, help="the Linux checkout (default: this repository)")
    ap.add_argument("--wine", type=wine_command, default=os.environ.get("WINE", "wine"),
                    help="Wine command (default: $WINE or wine)")
    ap.add_argument("--require", action="append", default=[], help="a test that must pass (repeatable)")
    ap.add_argument("--timeout", type=positive_seconds, default=120.0,
                    help="maximum seconds per test; a shorter CTest TIMEOUT applies (default 120)")
    ap.add_argument("--jobs", type=positive_int, default=1,
                    help="tests to run at once (default 1: one at a time, in order). Tests of one program, and tests "
                         "with a CTest dependency, fixture or resource lock, never overlap; RUN_SERIAL tests run alone")
    ap.add_argument("--summary", help="append a Markdown table to this file (e.g. $GITHUB_STEP_SUMMARY)")
    ap.add_argument("--label", help="named in the summary's heading, e.g. the Wine version")
    ap.add_argument("--keep-wine-debugger", action="store_true",
                    help="leave Wine's crash debugger (winedbg) on: an aborted test then waits for it until --timeout")
    args = ap.parse_args()

    tests = read_tests(args.tests)
    if tests is None:
        return 2
    names = [test.get("name", "?") for test in tests]
    indices = {name: index for index, name in enumerate(names)}
    setups = fixture_dependencies(tests)
    sorter = TopologicalSorter(setups)
    for index, test in enumerate(tests):
        # DEPENDS controls order. Only failed fixture setups block a consumer.
        sorter.add(index, *(indices[name] for name in properties(test).get("DEPENDS", []) if name in indices))
    try:
        order = list(sorter.static_order())
    except CycleError:
        print("ERROR: CTest test dependencies form a cycle", file=sys.stderr)
        return 2
    wine = args.wine
    prefix = os.environ.get("WINEPREFIX") or os.path.join(os.path.expanduser("~"), ".wine")
    with DriveMap(args.windows_root, args.root, prefix) as drive, \
            WineDebugger(wine, prefix, args.keep_wine_debugger) as debugger:
        print(drive.note, flush=True)
        if debugger.note:
            print(debugger.note, flush=True)
        def report(name: str, status: str, detail: str) -> None:
            print(f"[{status.upper()}] {name}", flush=True)
            if status != "pass":
                for line in detail.rstrip().splitlines()[-40:]:
                    print(f"    {line}")

        results = run_tests(tests, args, wine, report, order, setups)

    missing = [n for n in args.require if n not in results]
    failed_required = [n for n in args.require if results.get(n) != "pass"]
    counts = {s: sum(1 for v in results.values() if v == s) for s in ("pass", "fail", "skip")}
    print(f"\n{counts['pass']} passed, {counts['fail']} failed, {counts['skip']} skipped under Wine "
          f"({len(results)} tests)")
    for n in missing:
        print(f"required test not listed by the Windows build: {n}")
    if failed_required:
        print("required tests that did not pass: " + ", ".join(failed_required))

    if args.summary:
        try:
            _make_parent(args.summary)
            with open(args.summary, "a", encoding="utf-8", newline="\n") as f:
                f.write(f"### Windows tests under {args.label or 'Wine'}\n\n")
                f.write(f"{counts['pass']} passed, {counts['fail']} failed, {counts['skip']} skipped.\n\n")
                f.write("| Test | Result | Required |\n|---|---|---|\n")
                for name, status in results.items():
                    f.write(f"| `{name}` | {status} | {'yes' if name in args.require else ''} |\n")
                f.write("\n")
        except OSError as e:
            print(f"ERROR: cannot write the summary {args.summary}: {e.strerror or e}", file=sys.stderr)
            return 1
    return 1 if failed_required else 0


if __name__ == "__main__":
    sys.exit(main())
