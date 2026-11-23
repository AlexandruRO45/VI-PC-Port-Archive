"""Compare real msvcrt.dll argument imports natively and through the PE loader.

The same original fixture imports both legacy APIs from an import-library-only
DEF file. Raw command lines retain quoted wildcards and identical argv[0].
All inputs are synthetic; native Windows supplies the ACP-dependent oracle.
"""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile


def run(executable, command, cwd, environment):
    child = subprocess.run(
        command, executable=executable, cwd=cwd, env=environment,
        capture_output=True, timeout=15,
    )
    output = child.stdout.decode("utf-8", "replace")
    return {
        "command": command, "exit_code": child.returncode,
        "lines": [line for line in output.splitlines()
                  if line.startswith(("A step=", "W step=", "arg"))],
        "stdout": output, "stderr": child.stderr.decode("utf-8", "replace"),
    }


def states(result, mode):
    """Require successful calls, complete terminated tables, and the requested flags."""
    lines = iter(result["lines"])
    tables = []
    for step, flag in enumerate(mode[1:], 1):
        header = next(lines, "")
        match = re.fullmatch(
            rf"{mode[0]} step={step} expand={flag} rc=0 argc=(\d+) env=1 term=1", header,
        )
        if not match or not 1 <= int(match[1]) < 256:
            return None
        table = []
        for index in range(int(match[1])):
            argument = re.fullmatch(rf"arg{index}=([0-9a-f]*)", next(lines, ""))
            if not argument:
                return None
            table.append(argument[1])
        tables.append(table)
    return tables if next(lines, None) is None else None


def ascii_hex(text, width):
    return "".join(f"{ord(char):0{2 if width == 'A' else 4}x}" for char in text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loader", type=Path, required=True)
    parser.add_argument("--guest", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if os.name != "nt":
        parser.error("this native CRT comparison requires Windows")
    loader, guest, work = (path.resolve(strict=True) for path in (args.loader, args.guest, args.work_dir))
    environment = {key: value for key, value in os.environ.items()
                   if not key.upper().startswith("LEONIDA_")}
    environment["LEONIDA_UNIMPLEMENTED"] = "stop"
    cases = [
        ("literal-a", "A0", "*.dat"), ("expand-a", "A1", "*.dat"),
        ("literal-w", "W0", "*.dat"), ("expand-w", "W1", "*.dat"),
        ("quoted-a", "A1", '"*.dat"'), ("quoted-w", "W1", '"*.dat"'),
        ("unmatched-a", "A1", "none*.missing"), ("unmatched-w", "W1", "none*.missing"),
        ("prefix-a", "A1", r"sub\*.dat"), ("prefix-w", "W1", r"sub\*.dat"),
        ("reconfigure-a", "A01", "*.dat"), ("reverse-a", "A10", "*.dat"),
        ("reconfigure-w", "W01", "*.dat"), ("reverse-w", "W10", "*.dat"),
        ("encoding-a", "A1", "unicode/\u6f22*.txt"), ("encoding-w", "W1", "unicode/\u6f22*.txt"),
        ("order-a", "A1", "order/*.txt literal second/*.txt"),
        ("order-w", "W1", "order/*.txt literal second/*.txt"),
        ("order-reconfigure-a", "A01", "order/*.txt literal second/*.txt"),
        ("order-reconfigure-w", "W10", "order/*.txt literal second/*.txt"),
    ]
    rows, program_rows, failures = [], [], []
    with tempfile.TemporaryDirectory(prefix="legacy_argv_", dir=work) as temporary:
        root = Path(temporary)
        for name in ("sub", "unicode", "package", "order", "second"):
            (root / name).mkdir()
        seeded = {}
        for name in ("a.dat", "b.dat", "z.txt", "file-\u6f22.dat", "sub/inside.dat",
                     "unicode/A-only.txt", "unicode/\u6f22-only.txt", "package/sentinel.txt",
                     "programA.exe", "programB.exe"):
            seeded[name] = b"original synthetic input\n"
            (root / name).write_bytes(seeded[name])
        for name in ("Z.txt", "a.txt", "B.txt", "_x.txt", "[x.txt", "\u00e9.txt", "\u00c4.txt",
                     "aA.txt", "Ab.txt", "-x.txt", "0x.txt", "\u6f22.txt"):
            seeded["order/" + name] = b"original synthetic order input\n"
            (root / "order" / name).write_bytes(seeded["order/" + name])
        for name in ("Z.txt", "a.txt"):
            seeded["second/" + name] = b"original synthetic second input\n"
            (root / "second" / name).write_bytes(seeded["second/" + name])
        plain = {}
        for label, mode, tail in cases:
            native_command = subprocess.list2cmdline([guest.name, mode]) + " " + tail
            mapped_command = subprocess.list2cmdline([
                loader.name, "--run", str(guest), "--game-dir", str(root / "package"),
                "--overlay", str(root / "overlay"), "--", mode,
            ]) + " " + tail
            native = run(guest, native_command, root, environment)
            mapped = run(loader, mapped_command, root, environment)
            problems = []
            native_tables = states(native, mode)
            for route, result in (("native", native), ("mapped", mapped)):
                if result["exit_code"] or states(result, mode) is None:
                    problems.append(f"{route} did not complete valid argument calls")
            if native_tables is not None:
                if label.startswith("order"):
                    if [len(table) for table in native_tables] != [5 if flag == "0" else 17 for flag in mode[1:]]:
                        problems.append("native ordering control did not match both pattern groups")
                    for flag, table in zip(mode[1:], native_tables):
                        if flag == "1" and len(table) == 17 and (
                                table[14] != ascii_hex("literal", mode[0]) or
                                any(not value.startswith(ascii_hex("order/", mode[0])) for value in table[2:14]) or
                                any(not value.startswith(ascii_hex("second/", mode[0])) for value in table[15:])):
                            problems.append("native sorting moved arguments across pattern groups")
                elif not label.startswith("encoding"):
                    counts = [3 if flag == "0" or label.startswith(("unmatched", "prefix")) else 5
                              for flag in mode[1:]]
                    if [len(table) for table in native_tables] != counts:
                        problems.append("native argument counts do not exercise the intended case")
                elif len(native_tables[0]) not in ((3,) if mode[0] == "W" else (3, 4)):
                    problems.append("native encoding pattern produced an unexpected number of matches")
                if label.startswith("expand"):
                    plain[mode[0]] = native["lines"]
                if label.startswith("quoted") and native["lines"] != plain[mode[0]]:
                    problems.append("native quoted pattern differs from unquoted")
                if label.startswith("prefix") and native_tables[0][-1] != ascii_hex(r"sub\inside.dat", mode[0]):
                    problems.append("native prefixed match is missing")
            if native["lines"] != mapped["lines"]:
                problems.append("mapped argument tables differ from native")
            if "game exited with code 0" not in mapped["stdout"]:
                problems.append("loader did not complete the guest run")
            rows.append({"case": label, "native": native, "mapped": mapped, "problems": problems})
            if problems:
                failures.append(label + ": " + "; ".join(problems))
                print("FAIL " + failures[-1], flush=True)

        # CreateProcess permits an independent command-line program token. This native-only control establishes
        # the argv[0] expansion rule also exercised through the production shim by process_cmdline_test.
        for mode in ("A0", "A1", "W0", "W1"):
            command = subprocess.list2cmdline(["program?.exe", mode, "literal"])
            result = run(guest, command, root, environment)
            tables = states(result, mode)
            expected = [ascii_hex(value, mode[0]) for value in (mode, "literal")]
            problems = []
            if result["exit_code"] or tables is None:
                problems.append("native program-token control did not complete")
            elif mode[1] == "0":
                if tables != [[ascii_hex("program?.exe", mode[0]), *expected]]:
                    problems.append("native zero flag changed the program token")
            elif (len(tables[0]) != 4 or tables[0][2:] != expected or
                  set(tables[0][:2]) != {ascii_hex(name, mode[0]) for name in ("programA.exe", "programB.exe")}):
                problems.append("native wildcard flag did not expand the program token")
            program_rows.append({"mode": mode, "native": result, "problems": problems})
            failures.extend(mode + ": " + problem for problem in problems)
        for name, content in seeded.items():
            path = root / name
            if not path.is_file() or path.read_bytes() != content:
                failures.append("synthetic input changed: " + name)
        if {path.name for path in (root / "package").iterdir()} != {"sentinel.txt"}:
            failures.append("synthetic package entries changed")
    report = {
        "acp": ctypes.windll.kernel32.GetACP(), "pairs": len(rows), "native_program_controls": len(program_rows),
        "sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                   for name, path in (("loader", loader), ("guest", guest))},
        "rows": rows, "program_rows": program_rows, "failures": failures,
    }
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"legacy argv: {len(rows)} native/mapped pairs, {len(program_rows)} native program-token controls, "
          f"{len(failures)} failures, ACP {report['acp']}")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
