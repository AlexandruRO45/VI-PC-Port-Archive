"""Exercise Windows first-boot diagnostics on the tiny_startup_cmdline fixtures."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
GUEST = "tiny_startup_cmdline.exe"
DLL = "tiny_startup_cmdline_dll.dll"
IMPORT = "startup_cmdline_matches"


def require(condition, detail):
    if not condition:
        raise RuntimeError(detail)


def tool(name, *args):
    result = subprocess.run([sys.executable, "-I", ROOT / "tools" / name, *args],
                            stdout=subprocess.PIPE, timeout=30, check=True)
    return result.stdout


def rehearse(loader, fixtures, work):
    guest = work / GUEST
    shutil.copyfile(fixtures / GUEST, guest)
    shutil.copyfile(fixtures / DLL, work / DLL)
    tool("pe_inspect.py", guest, "--game-dir", work, "--out", work)
    report = json.loads((work / (GUEST + ".json")).read_bytes())
    require(any(i["dll"].lower() == DLL and i["where"] == "bundled" and IMPORT in i["functions"]
                for i in report["imports"]), "inspection missed the bundled startup import")
    tool("gdk_runtime_calls.py", guest, "--out", work)
    tool("find_hash_lookup.py", guest, "--json", work / "hash_lookup.json")
    tool("exe_checklist.py", "--gdk-calls", work / (GUEST + ".gdk_calls.json"),
         "--out", work / "checklist.md")

    def phase(label, mode, code, *args, directory=work):
        trace = work / (label + ".jsonl")
        command = [loader, mode, directory / GUEST, "--trace", trace, "--mods", work / "empty-mods"]
        if args:
            command.extend(["--", *args])
        # This relocatable fixture runs in the loader process. run() kills and waits on timeout.
        try:
            result = subprocess.run(command, cwd=directory, capture_output=True,
                                    timeout=5 if code is None else 30)
            actual, output = result.returncode, result.stdout + result.stderr
        except subprocess.TimeoutExpired as error:
            actual, output = None, (error.stdout or b"") + (error.stderr or b"")
        require(actual == code, f"{label}: expected {code}, got {actual}: {output!r}")
        records = [json.loads(line) for line in trace.read_bytes().splitlines()]
        summary = json.loads(tool("boot_trace.py", trace, "--json"))["startup"]
        require(summary["bad_lines"] == 0, f"{label}: malformed trace")
        modules = sorted(m["module"] for m in summary["modules"])
        expected = [GUEST] if directory != work else sorted([GUEST, DLL])
        require(modules == expected, f"{label}: unexpected mapping {modules}")
        print(f"{label}: process={actual}, trace exit={summary['exit']}, error={summary['error']}")
        return records, summary, output

    for label, mode, code, args in [
        ("map", "--dry-run", 0, []), ("attach", "--init", 0, []),
        ("entry", "--run", 42, ["exit-code"]), ("abnormal", "--run", 73, ["abnormal"]),
        ("timeout", "--run", None, ["timeout"]),
    ]:
        records, summary, output = phase(label, mode, code, *args)
        attach = [(r["kind"], r["module"]) for r in records
                  if r["kind"] == "dllmain" or (r["kind"] == "tls_callback" and r["reason"] == 1)]
        if mode == "--dry-run":
            require(not attach, f"{label}: unexpected initialization {attach}")
        else:
            require(attach.count(("dllmain", DLL)) == 1 and ("tls_callback", GUEST) in attach
                    and attach.index(("dllmain", DLL)) < attach.index(("tls_callback", GUEST))
                    and summary["dllmain"]["ok"] == 1, f"{label}: wrong attach order {attach}")
        require((b"tiny_startup_cmdline: PASS" in output) == (mode == "--run"),
                f"{label}: unexpected guest entry output {output!r}")
        expected_exit = None if label in ("abnormal", "timeout") else code
        require(summary["exit"] == expected_exit and summary["error"] is None,
                f"{label}: unexpected final summary {summary}")

    _, summary, output = phase("failed-init", "--run", 1, "fail-init")
    require(summary["dllmain"]["failed_modules"] == [DLL] and summary["exit"] is None
            and summary["error"] == DLL + ": DllMain returned FALSE" and b"running " not in output,
            f"failed-init: expected a rejected DLL attach before entry: {summary}")

    missing = work / "missing"
    missing.mkdir()
    shutil.copyfile(guest, missing / GUEST)
    _, summary, output = phase("missing-import", "--dry-run", 0, directory=missing)
    require(f"UNIMPLEMENTED {DLL}!{IMPORT}".encode() in b" ".join(output.split())
            and summary["exit"] == 0, f"missing-import: expected unresolved fixture import: {output!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loader", type=Path, default=ROOT / "runtime/build/gta6rt.exe")
    parser.add_argument("--fixtures", type=Path,
                        default=ROOT / "runtime/build/fixtures/game/RelWithDebInfo")
    args = parser.parse_args()
    loader = args.loader.resolve()
    for path in (loader, loader.with_name("gta6rt_tls_host.dll"), args.fixtures / GUEST, args.fixtures / DLL):
        if not path.is_file():
            parser.error(f"missing required build artifact: {path}")
    with tempfile.TemporaryDirectory(prefix="leonida-rehearsal-") as temp:
        rehearse(loader, args.fixtures, Path(temp))


if __name__ == "__main__":
    main()
