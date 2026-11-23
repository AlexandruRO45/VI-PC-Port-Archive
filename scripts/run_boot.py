"""Run one bounded gta6rt first-boot pass into a fresh private local directory."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
OPTIONS = ("LEONIDA_D3D12", "LEONIDA_UNIMPLEMENTED", "LEONIDA_VK_VALIDATION",
           "LEONIDA_VK_DEVICE", "LEONIDA_SHADERS", "LEONIDA_SAVE_DIR", "LEONIDA_GAME_DIR",
           "LEONIDA_PRESENTX_SLOT", "LEONIDA_CAPTURE", "DISABLE_RTSS_LAYER", "DISABLE_VULKAN_OBS_CAPTURE")
if os.name == "nt":
    sys.path.insert(0, str(HERE))
    import boot_windows


def seconds(value):
    n = float(value)
    if not math.isfinite(n) or not 0 < n <= 86400:
        raise argparse.ArgumentTypeError("expected seconds in (0, 86400]")
    return n


def adapters():
    """Installed display adapters and their driver versions, when registered."""
    import winreg
    path = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
    result = []
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path) as root:
            for i in range(winreg.QueryInfoKey(root)[0]):
                name = winreg.EnumKey(root, i)
                if name.isdecimal():
                    try:
                        with winreg.OpenKey(root, name) as key:
                            result.append({k: winreg.QueryValueEx(key, k)[0]
                                           for k in ("DriverDesc", "DriverVersion")})
                    except OSError:
                        continue
    except OSError:
        pass
    return result


def revision(directory):
    try:
        return subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL, timeout=3, text=True).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def diagnostic(pid, out, reason, budget):
    target = out / (reason + ".dmp")
    partial = target.with_suffix(".partial.dmp")
    command = [sys.executable, "-I", str(HERE / "boot_windows.py"), str(pid), str(partial)]
    result = {"reason": reason}
    try:
        with (out / (reason + "-capture.log")).open("xb") as log:
            with boot_windows.launch(command, out, None, log, log) as (writer, _):
                status = boot_windows.wait(writer, budget)
        result["exit_code"] = status
        if status == 0 and partial.stat().st_size:
            partial.rename(target)
            result["status"] = "captured"
        else:
            result["status"] = "timeout" if status is None else "failed"
    except OSError as exc:
        result.update(status="failed", error=str(exc))
    return result


def loader_command(runtime, mode, image, package, out, startup_only=False):
    command = list(map(str, [runtime, "--" + mode, image, "--game-dir", package, "--overlay",
                          out / "overlay", "--trace", out / "startup.jsonl"]))
    if mode != "dry-run" and not startup_only:
        command.append("--trace-d3d12")
    return command


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["dry-run", "init", "run"])
    for flag in ("runtime", "image", "game-dir", "out"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--deadline", type=seconds, required=True, metavar="SECONDS")
    parser.add_argument("--diagnose-after", type=seconds, metavar="SECONDS",
                        help="also take a snapshot if the loader is still active at this time")
    parser.add_argument("--capture-timeout", type=seconds, default=10, metavar="SECONDS")
    parser.add_argument("--guest-arg", action="append", default=[], metavar="ARG",
                        help="append one argument to the guest command line in run mode")
    parser.add_argument("--startup-only", action="store_true",
                        help="collect and parse only the startup trace for workloads without D3D12 calls")
    a = parser.parse_args(argv)
    if a.guest_arg and a.mode != "run":
        parser.error("--guest-arg requires run mode")
    if os.name != "nt":
        parser.error("the capture runner requires Windows")
    if a.diagnose_after and a.diagnose_after >= a.deadline:
        parser.error("--diagnose-after must precede the deadline")
    runtime, image, package, out = (p.resolve() for p in (a.runtime, a.image, a.game_dir, a.out))
    if not image.is_file() or not package.is_dir():
        parser.error("expected an existing image and package directory")
    shaders = os.environ.get("LEONIDA_SHADERS")
    if shaders and out.is_relative_to(Path(shaders).resolve()):
        parser.error("output overlaps shader input")
    if out.is_relative_to(package) or any(p.is_relative_to(out) for p in (runtime, image)):
        parser.error("output overlaps inputs")
    try:
        out.mkdir(parents=True)
    except OSError as exc:
        parser.error(str(exc))
    for directory in ("overlay", "saves"):
        (out / directory).mkdir()
    env = os.environ.copy()
    env.update(LEONIDA_GAME_DIR=str(package), LEONIDA_SAVE_DIR=str(out / "saves"))
    if env.get("LEONIDA_CAPTURE"):
        (out / "capture").mkdir()
        env["LEONIDA_CAPTURE"] = str(out / "capture")
    if shaders:
        env["LEONIDA_SHADERS"] = str(Path(shaders).resolve())
    env.setdefault("LEONIDA_D3D12", "vulkan")
    command = loader_command(runtime, a.mode, image, package, out, a.startup_only)
    if a.guest_arg:
        command.extend(["--", *a.guest_arg])
    report = dict(command=command, deadline=a.deadline, diagnose_after=a.diagnose_after,
                  capture_timeout=a.capture_timeout, runtime_sha256=None,
                  runner_checkout_revision=revision(HERE), runtime_checkout_revision=revision(runtime.parent),
                  installed_adapters=adapters(), options={k: env[k] for k in OPTIONS if k in env},
                  loader_exit_code=None, captures=[], status="launch_failure")

    def save():
        tmp = out / "result.tmp"
        tmp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
        tmp.replace(out / "result.json")

    save()
    with (out / "stdout.log").open("xb") as stdout, (out / "stderr.log").open("xb") as stderr:
        try:
            with runtime.open("rb") as f:
                report["runtime_sha256"] = hashlib.file_digest(f, "sha256").hexdigest()
            with boot_windows.launch(command, out, env, stdout, stderr) as (process, pid):
                end = time.monotonic() + a.deadline
                report.update(status="running", pid=pid)
                save()
                if a.diagnose_after:
                    delay = max(0, end - a.deadline + a.diagnose_after - time.monotonic())
                    if boot_windows.wait(process, delay) is None:
                        budget = max(0, min(a.capture_timeout, end - time.monotonic()))
                        report["captures"].append(diagnostic(pid, out, "requested", budget))
                code = boot_windows.wait(process, max(0, end - time.monotonic()))
                report.update(status="exited", loader_exit_code=code)
                if code is None:
                    report["status"] = "timeout"
                    report["captures"].append(diagnostic(pid, out, "deadline", a.capture_timeout))
                    # The loader can return naturally while the dump is being written.
                    report["loader_exit_code"] = boot_windows.wait(process, 0)
        except OSError as exc:
            report["process_error"] = str(exc)
            if report["status"] == "running":
                report["status"] = "supervision_failure"
    save()
    with (out / "summary.json").open("xb") as summary, (out / "parser.log").open("xb") as log:
        try:
            cmd = [sys.executable, "-I", "-X", "utf8", str(HERE.parent / "tools" / "boot_trace.py"),
                   str(out / "startup.jsonl"), "--json"]
            if a.mode != "dry-run" and not a.startup_only:
                cmd.extend(["--d3d12", str(out / "stderr.log")])
            report["parser_exit_code"] = subprocess.run(cmd, stdout=summary, stderr=log).returncode
        except OSError as exc:
            report.update(parser_exit_code=None, parser_error=str(exc))
    save()
    print(report["status"])
    return int(report["status"] != "exited" or report["loader_exit_code"] != 0
               or report["parser_exit_code"] != 0 or "process_error" in report
               or any(c["status"] != "captured" for c in report["captures"]))


if __name__ == "__main__":
    sys.exit(main())
