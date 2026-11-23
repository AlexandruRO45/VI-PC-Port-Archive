"""tools/vk_compute_batch.py with a stand-in vk_compute_test (a Python script that speaks the same --batch protocol):
the batched results equal one process per program, and a crash, a timeout, a silent hang, a missing device or an
older binary without --batch still give every program its result."""
import os
import sys
import textwrap
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import vk_compute_batch as vkb  # noqa: E402

FAKE = textwrap.dedent(r'''
    import os, sys, time
    args = sys.argv[1:]
    old = args[:1] == ["--old"]
    if old:
        args = args[1:]
    with open(os.environ["FAKE_LOG"], "a") as log:
        log.write(("batch" if args[:1] == ["--batch"] else "single") + "\n")

    def program(spv, settings):
        """(exit code, stdout, stderr) of one program, from its name: ok-N, fail-N, crash-N, slow-N, hang-N."""
        kind = os.path.basename(spv).split("-")[0]
        if kind == "crash":
            os._exit(139)
        if kind == "hang":
            time.sleep(600)
        if "VKC_DUMP" in settings:
            with open(settings["VKC_DUMP"], "wb") as f:
                f.write(spv.encode())
        return (1 if kind == "fail" else 0), f"ran {spv} s={settings.get('VKC_SGPRS')}\n", "device: fake\n"

    if args == ["--batch-version"]:
        if old:
            sys.stderr.write("usage: vk_compute_test cs.spv groups expect [--require-wave64]\n")
            sys.exit(2)
        print("vk_compute_test batch 1")
        sys.exit(0)
    if os.environ.get("FAKE_NO_DEVICE"):
        sys.stderr.write("no Vulkan physical device found\n")
        sys.exit(1)
    if args[:1] == ["--batch"]:
        out = sys.stdout.buffer
        lines = [ln.rstrip("\n").split("\t") for ln in open(args[1], encoding="utf-8")]
        out.write(b"@vkc-batch 1 jobs %d\n" % len(lines))
        for i, f in enumerate(lines):
            out.write(b"@vkc-job-begin %d\n" % i)
            out.flush()
            if os.path.basename(f[0]).startswith("slow"):
                out.write(b"@vkc-job-end %d 124 timeout 0 0\n" % i)
                out.flush()
                os._exit(3)
            code, o, e = program(f[0], dict(x.split("=", 1) for x in f[3:]))
            o, e = o.encode(), e.encode()
            out.write(b"@vkc-job-end %d %d done %d %d\n" % (i, code, len(o), len(e)) + o + e)
            out.flush()
        out.write(b"@vkc-batch-end %d\n" % len(lines))
        sys.exit(0)
    code, o, e = program(args[0], {k: v for k, v in os.environ.items() if k.startswith("VKC_")})
    sys.stdout.write(o)
    sys.stderr.write(e)
    sys.exit(code)
''')


@pytest.fixture
def fake(tmp_path, monkeypatch):
    script = tmp_path / "fake_vk_compute_test.py"
    script.write_text(FAKE, encoding="utf-8")
    log = tmp_path / "processes.log"
    monkeypatch.setenv("FAKE_LOG", str(log))
    monkeypatch.delenv("FAKE_NO_DEVICE", raising=False)

    def processes():
        return log.read_text().split() if log.exists() else []
    return [sys.executable, str(script)], processes


def jobs(tmp_path, *names):
    """One job per name, each with its own VKC_SGPRS and dump file."""
    return [vkb.Job((str(tmp_path / name), "1", "dump"),
                    dict(os.environ, VKC_SGPRS=str(i), VKC_DUMP=str(tmp_path / f"{name}.bin")))
            for i, name in enumerate(names)]


def test_batched_results_equal_one_process_per_program(fake, tmp_path):
    exe, processes = fake
    names = ["ok-0", "fail-1", "ok-2", "ok-3", "fail-4", "ok-5", "ok-6"]
    single = list(vkb.run_jobs(exe, jobs(tmp_path, *names), batch_size=1))
    dumps = {n: (tmp_path / f"{n}.bin").read_bytes() for n in names}
    for n in names:
        (tmp_path / f"{n}.bin").unlink()
    before = len(processes())
    batched = list(vkb.run_jobs(exe, jobs(tmp_path, *names), batch_size=3))
    assert batched == single
    assert [r.returncode for r in batched] == [0, 1, 0, 0, 1, 0, 0]
    assert batched[2].output == f"ran {tmp_path / 'ok-2'} s=2\ndevice: fake\n"
    assert {n: (tmp_path / f"{n}.bin").read_bytes() for n in names} == dumps
    assert processes()[before:].count("batch") == 3          # 7 programs in batches of 3
    assert processes()[:before].count("single") == 7


def test_a_crash_is_charged_to_its_program_and_the_rest_run_again(fake, tmp_path):
    exe, processes = fake
    got = list(vkb.run_jobs(exe, jobs(tmp_path, "ok-0", "crash-1", "ok-2", "ok-3"), batch_size=10))
    assert [r.returncode for r in got] == [0, 139, 0, 0]
    assert "while running this program" in got[1].stderr
    assert got[3].stdout == f"ran {tmp_path / 'ok-3'} s=3\n"
    assert processes().count("batch") == 2


def test_a_timed_out_program_is_marked_and_the_rest_run_again(fake, tmp_path):
    exe, processes = fake
    got = list(vkb.run_jobs(exe, jobs(tmp_path, "ok-0", "slow-1", "ok-2"), batch_size=10))
    assert [(r.returncode, r.timed_out) for r in got] == [(0, False), (124, True), (0, False)]
    assert processes().count("batch") == 2


def test_a_silent_hang_is_stopped_within_the_limit(fake, tmp_path, monkeypatch):
    exe, processes = fake
    monkeypatch.setattr(vkb, "JOB_GRACE", 1)
    start = time.monotonic()
    got = list(vkb.run_jobs(exe, jobs(tmp_path, "ok-0", "hang-1", "ok-2"), batch_size=10, timeout=0.5))
    assert time.monotonic() - start < 30
    assert [(r.returncode, r.timed_out) for r in got] == [(0, False), (124, True), (0, False)]


def test_no_device_gives_every_program_the_single_run_error(fake, tmp_path, monkeypatch):
    exe, _ = fake
    monkeypatch.setenv("FAKE_NO_DEVICE", "1")
    single = list(vkb.run_jobs(exe, jobs(tmp_path, "ok-0", "ok-1"), batch_size=1))
    assert list(vkb.run_jobs(exe, jobs(tmp_path, "ok-0", "ok-1"), batch_size=10)) == single
    assert single[0] == vkb.Result(1, "", "no Vulkan physical device found\n")


def test_a_binary_without_batch_runs_one_process_per_program(fake, tmp_path):
    exe, processes = fake
    got = list(vkb.run_jobs(exe + ["--old"], jobs(tmp_path, "ok-0", "fail-1"), batch_size=10))
    assert [r.returncode for r in got] == [0, 1]
    assert "batch" not in processes()
