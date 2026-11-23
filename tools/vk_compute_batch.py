"""Run many vk_compute_test programs in one process (vk_compute_test --batch; runtime/test/vk_compute_test.cpp).

Opening the Vulkan device costs seconds per process (2.5 s on an idle RTX 5080 with driver 580.65.06, about 18 s when
some 90 processes open it at once, as parallel fuzzers do: the driver serialises device creation), while one fuzzed
program runs for microseconds. run_jobs() sends the programs to vk_compute_test in batches of batch_size, one process
and one device per batch, and yields one Result per program, in order, with what a single run would have given: the
exit code, stdout and stderr. Each job gets its own pipeline and buffers in that process. A job that crashes the
process, runs past the timeout or loses the device is reported for that job, and the jobs after it run in a new
process. With batch_size 1, or a vk_compute_test built before --batch existed, every program runs in its own process
as before. Standard library only; no game files.
"""
import locale
import os
import queue
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field

# The settings a single run reads from its environment and a batch job gives on its line (vk_compute_test.cpp).
SETTINGS = ("VKC_OUT_WORDS", "VKC_DUMP", "VKC_HEAP_INIT", "VKC_HEAP_FILE", "VKC_SUBGROUP_SIZE", "VKC_SGPRS",
            "VKC_BENCH", "LEONIDA_PC_TAIL_UBO")
# Programs per process. Measured on an RTX 5080 (valu_fuzz --family mem --emulate, 600 programs, while a fuzzing
# campaign kept opening the device): the GPU phase took 21.5 s in batches of 16, 3.6 s in batches of 64 and 1.2 s in
# one batch of 1024. A program costs about 2 ms once the device is open; what varies is each device open (2 s idle, up
# to 25 s under contention), so fewer opens is better, and a crash only reruns the programs after it. With 1024 a
# usual fuzzer run (50 to 1000 programs) opens the device once. See docs/shader_validation.md ("Batch mode").
DEFAULT_BATCH_SIZE = 1024
JOB_TIMEOUT = 60        # seconds one dispatch may run (vk_compute_test --job-timeout)
STARTUP_GRACE = 300     # seconds for the device open before the first job starts (18 s measured under load)
JOB_GRACE = 120         # seconds a job may take beyond its dispatch limit (shader compile, a busy machine)
TIMED_OUT = 124         # the exit code of a job that ran past its limit


@dataclass(frozen=True)
class Job:
    """One program: vk_compute_test's arguments (cs.spv, groups, expect[, --require-wave64]) and the environment a
    single run of it gets (None: this process's)."""
    args: tuple
    env: dict = field(default=None, hash=False, compare=False)


@dataclass(frozen=True)
class Result:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def output(self):
        """stdout then stderr, as the fuzzers read a single run's output."""
        return self.stdout + self.stderr


_SUPPORT = {}


def command(exe):
    """vk_compute_test as an argument list: a path, or a list such as [runner, path]."""
    return [os.fspath(exe)] if isinstance(exe, (str, os.PathLike)) else [os.fspath(x) for x in exe]


def supports_batch(exe):
    """True when this vk_compute_test has --batch (it prints its batch version without opening a device; an older
    build prints its usage and exits 2). Cached per command."""
    cmd = tuple(command(exe))
    if cmd not in _SUPPORT:
        try:
            r = subprocess.run([*cmd, "--batch-version"], capture_output=True, text=True, timeout=60)
            _SUPPORT[cmd] = r.returncode == 0 and r.stdout.startswith("vk_compute_test batch ")
        except (OSError, subprocess.SubprocessError):
            _SUPPORT[cmd] = False
    return _SUPPORT[cmd]


def run_single(exe, job, timeout=None):
    """One program in its own process, the way the fuzzers always ran it."""
    try:
        r = subprocess.run([*command(exe), *map(os.fspath, job.args)], capture_output=True, text=True,
                           env=job.env, timeout=timeout)
    except subprocess.TimeoutExpired:
        return Result(TIMED_OUT, "", f"vk_compute_test timed out after {timeout} s\n", timed_out=True)
    return Result(r.returncode, r.stdout, r.stderr)


def job_line(job):
    """The job's line in a batch file: the arguments and its settings, tab-separated."""
    args = [os.fspath(a) for a in job.args]
    if len(args) not in (3, 4) or (len(args) == 4 and args[3] != "--require-wave64"):
        raise ValueError(f"vk_compute_test arguments must be cs.spv groups expect [--require-wave64], got {args}")
    env = os.environ if job.env is None else job.env
    fields = args + [f"{k}={env[k]}" for k in SETTINGS if k in env]
    if any(c in f for f in fields for c in "\t\r\n") or args[0].startswith("#") or not args[0]:
        raise ValueError(f"a batch job cannot hold a tab, a line break or a leading #: {fields}")
    return "\t".join(fields)


def _device_env(job):
    env = os.environ if job.env is None else job.env
    return {k: v for k, v in env.items() if k not in SETTINGS}


def _text(data):
    """Bytes as subprocess's text mode decodes them (universal newlines)."""
    text = data.decode(locale.getpreferredencoding(False), errors="replace")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _read_records(stream, events):
    """Reader thread: vk_compute_test's batch records, as events (kind, ...), then ("eof",)."""
    try:
        while True:
            line = stream.readline()
            if not line:
                break
            words = line.decode("ascii", errors="replace").split()
            if words[:1] == ["@vkc-batch"]:
                events.put(("header",))
            elif words[:1] == ["@vkc-job-begin"] and len(words) == 2:
                events.put(("begin", int(words[1])))
            elif words[:1] == ["@vkc-job-end"] and len(words) == 6:
                index, code, status, n_out, n_err = int(words[1]), int(words[2]), words[3], int(words[4]), int(words[5])
                body = b""
                while len(body) < n_out + n_err:
                    chunk = stream.read(n_out + n_err - len(body))
                    if not chunk:
                        break
                    body += chunk
                if len(body) < n_out + n_err:
                    break          # the process died while writing the record: the job has no result
                events.put(("end", index, code, status, body[:n_out], body[n_out:]))
    finally:
        events.put(("eof",))


def _drain(stream, sink):
    for chunk in iter(lambda: stream.read(65536), b""):
        sink.append(chunk)


@dataclass
class _Batch:
    results: dict
    begun: set
    header: bool
    returncode: int
    stderr: str
    stopped: bool      # the watchdog stopped a process that went quiet


def _run_process(cmd, jobs, timeout, env):
    """Runs jobs in one vk_compute_test --batch process and collects the records it wrote."""
    fd, path = tempfile.mkstemp(prefix="vkc-batch-", suffix=".txt")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write("".join(job_line(j) + "\n" for j in jobs))
        proc = subprocess.Popen([*cmd, "--batch", path, "--job-timeout", str(timeout or 0)], stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        events, err = queue.Queue(), []
        threads = [threading.Thread(target=_read_records, args=(proc.stdout, events), daemon=True),
                   threading.Thread(target=_drain, args=(proc.stderr, err), daemon=True)]
        for t in threads:
            t.start()
        results, begun, header, stopped = {}, set(), False, False
        limit = (timeout + JOB_GRACE) if timeout else None
        deadline = time.monotonic() + STARTUP_GRACE
        while True:
            wait = None if deadline is None else max(0.0, deadline - time.monotonic())
            try:
                event = events.get(timeout=wait)
            except queue.Empty:
                stopped = True
                proc.kill()
                break
            if event[0] == "eof":
                break
            if event[0] == "header":
                header = True
            elif event[0] == "begin":
                begun.add(event[1])
                deadline = None if limit is None else time.monotonic() + limit
            elif event[0] == "end":
                _, index, code, status, out, errb = event
                results[index] = Result(code, _text(out), _text(errb), timed_out=status == "timeout")
                deadline = None if limit is None else time.monotonic() + limit
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        for t in threads:
            t.join(timeout=10)
        return _Batch(results, begun, header, proc.returncode, _text(b"".join(err)), stopped)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def _run_chunk(cmd, jobs, timeout):
    """Results for jobs that share one device environment: one process, then a new one after each job that ended
    the process (crash, timeout, lost device)."""
    done = []
    env = _device_env(jobs[0])
    while len(done) < len(jobs):
        rest = jobs[len(done):]
        b = _run_process(cmd, rest, timeout, env)
        k = 0
        while k in b.results:
            done.append(b.results[k])
            k += 1
        if k == len(rest):
            break
        if not b.header:
            # No device (or no Vulkan): what a single run of each job would have said.
            done += [Result(b.returncode, "", b.stderr)] * len(rest)
            break
        if k not in b.begun and k > 0 and not b.stopped:
            continue       # the process ended between two jobs: the next one never started, so it just runs again
        if b.stopped:
            done.append(Result(TIMED_OUT, "", f"{b.stderr}vk_compute_test gave no result in "
                               f"{timeout + JOB_GRACE if timeout else STARTUP_GRACE} s and was stopped\n", timed_out=True))
        else:
            done.append(Result(b.returncode if b.returncode else 1, "",
                               f"{b.stderr}vk_compute_test ended (exit {b.returncode}) while running this program\n"))
    return done


def run_jobs(exe, jobs, batch_size=DEFAULT_BATCH_SIZE, timeout=JOB_TIMEOUT):
    """Yields one Result per job, in order. batch_size programs share a process (1: a process each, also used when
    vk_compute_test has no --batch); timeout is the seconds one program may run (None: no limit)."""
    jobs = list(jobs)
    cmd = command(exe)
    if batch_size <= 1 or not jobs or not supports_batch(cmd):
        for job in jobs:
            yield run_single(cmd, job, timeout)
        return
    start = 0
    while start < len(jobs):
        env = _device_env(jobs[start])
        end = start + 1
        while end < len(jobs) and end - start < batch_size and _device_env(jobs[end]) == env:
            end += 1
        yield from _run_chunk(cmd, jobs[start:end], timeout)
        start = end
