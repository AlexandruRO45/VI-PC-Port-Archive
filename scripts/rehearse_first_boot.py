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
    package = work / "package"
    package.mkdir()
    guest = package / GUEST
    shutil.copyfile(fixtures / GUEST, guest)
    shutil.copyfile(fixtures / DLL, package / DLL)
    tool("pe_inspect.py", guest, "--game-dir", package, "--out", work)
    report = json.loads((work / (GUEST + ".json")).read_bytes())
    require(any(i["dll"].lower() == DLL and i["where"] == "bundled" and IMPORT in i["functions"]
                for i in report["imports"]), "inspection missed the bundled startup import")
    tool("gdk_runtime_calls.py", guest, "--out", work)
    tool("find_hash_lookup.py", guest, "--json", work / "hash_lookup.json")
    tool("exe_checklist.py", "--gdk-calls", work / (GUEST + ".gdk_calls.json"),
         "--out", work / "checklist.md")

    def phase(label, mode, code, *args, directory=package):
        out = work / label
        command = [sys.executable, "-I", ROOT / "scripts/run_boot.py", mode,
                   "--runtime", loader, "--image", directory / GUEST, "--game-dir", directory,
                   "--out", out, "--startup-only", "--deadline", "5" if code is None else "30"]
        for arg in args:
            command.extend(["--guest-arg", arg])
        result = subprocess.run(command, timeout=60)
        capture = json.loads((out / "result.json").read_bytes())
        output = (out / "stdout.log").read_bytes() + (out / "stderr.log").read_bytes()
        expected_status = "timeout" if code is None else "exited"
        require((capture["status"], capture["loader_exit_code"], capture["parser_exit_code"])
                == (expected_status, code, 0) and result.returncode == int(code != 0),
                f"{label}: unexpected runner result {capture}: {output!r}")
        if code is None:
            require([(c["reason"], c["status"]) for c in capture["captures"]] == [("deadline", "captured")]
                    and (out / "deadline.dmp").stat().st_size > 0, f"{label}: missing deadline dump")
        trace = out / "startup.jsonl"
        records = [json.loads(line) for line in trace.read_bytes().splitlines()]
        summary = json.loads((out / "summary.json").read_bytes())["startup"]
        require(summary["bad_lines"] == 0, f"{label}: malformed trace")
        modules = sorted(m["module"] for m in summary["modules"])
        expected = [GUEST] if directory != package else sorted([GUEST, DLL])
        require(modules == expected, f"{label}: unexpected mapping {modules}")
        print(f"{label}: {expected_status}, process={code}, trace exit={summary['exit']}, error={summary['error']}")
        return records, summary, output

    for label, mode, code, args in [
        ("map", "dry-run", 0, []), ("attach", "init", 0, []),
        ("entry", "run", 42, ["exit-code"]), ("abnormal", "run", 73, ["abnormal"]),
    ]:
        records, summary, output = phase(label, mode, code, *args)
        attach = [(r["kind"], r["module"]) for r in records
                  if r["kind"] == "dllmain" or (r["kind"] == "tls_callback" and r["reason"] == 1)]
        if mode == "dry-run":
            require(not attach, f"{label}: unexpected initialization {attach}")
        else:
            require(attach.count(("dllmain", DLL)) == 1 and ("tls_callback", GUEST) in attach
                    and attach.index(("dllmain", DLL)) < attach.index(("tls_callback", GUEST))
                    and summary["dllmain"]["ok"] == 1, f"{label}: wrong attach order {attach}")
        require((b"tiny_startup_cmdline: PASS" in output) == (mode == "run"),
                f"{label}: unexpected guest entry output {output!r}")
        entries = [r for r in records if r["kind"] == "game_entry"]
        require(len(entries) == int(mode == "run"), f"{label}: unexpected entry records {entries}")
        if entries:
            entry = entries[0]
            require(entry["tid"] > 0 and entry["tid"] != records[0]["tid"]
                    and entry["at"].startswith(GUEST + "+")
                    and any(r["kind"] == "tls_callback_return" and r["seq"] < entry["seq"] for r in records),
                    f"{label}: guest entry did not follow initialization on a guest thread")
        # CRT exit can end the guest thread without returning from its entry function.
        outstanding = summary["outstanding_phases"]
        if label == "entry" and summary["kinds"].get("game_entry_return"):
            require(not outstanding, f"{label}: completed entry still outstanding")
        else:
            expected = [("game_entry", e["tid"], e["phase_id"]) for e in entries]
            require([(p["phase"], p["tid"], p["phase_id"]) for p in outstanding] == expected,
                    f"{label}: unexpected outstanding phases {outstanding}")
        expected_exit = None if label == "abnormal" else code
        require(summary["exit"] == expected_exit and summary["error"] is None,
                f"{label}: unexpected final summary {summary}")

    records, summary, output = phase("timeout", "run", None, "trace-wait")
    outstanding = summary["outstanding_phases"]
    entered = [r for r in records if r["kind"] == "dllmain_enter" and r["module"] == DLL]
    require(len(entered) == len(outstanding) == 1 and outstanding[0]["phase"] == "dllmain"
            and outstanding[0]["module"] == DLL and outstanding[0]["reason"] == 1
            and (outstanding[0]["tid"], outstanding[0]["phase_id"]) == (entered[0]["tid"], entered[0]["phase_id"])
            and "game_entry" not in summary["kinds"] and summary["exit"] is None
            and b"inside synthetic DllMain" in output, f"timeout: expected an outstanding DLL attach: {summary}")

    _, summary, output = phase("failed-init", "run", 1, "fail-init")
    require(summary["dllmain"]["failed_modules"] == [DLL] and summary["exit"] is None
            and not summary["outstanding_phases"] and "game_entry" not in summary["kinds"]
            and summary["error"] == DLL + ": DllMain returned FALSE" and b"running " not in output,
            f"failed-init: expected a rejected DLL attach before entry: {summary}")

    missing = work / "missing"
    missing.mkdir()
    shutil.copyfile(guest, missing / GUEST)
    _, summary, output = phase("missing-import", "dry-run", 0, directory=missing)
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
