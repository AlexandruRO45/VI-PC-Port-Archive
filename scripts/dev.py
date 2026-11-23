#!/usr/bin/env python3
"""Use development tools from this Leonida checkout."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BOOT_TRACE = "tools/boot_trace.py"
TOOLS = (
    (
        "boot-trace",
        BOOT_TRACE,
        ("Python 3", "Startup JSONL from gta6rt --trace or stderr from gta6rt --trace-d3d12"),
    ),
    (
        "guest-scene",
        "scripts/run_guest_scene.py",
        (
            "Python 3",
            "gta6rt and tiny_guest_draw built with LEONIDA_GUEST_SCENE (static SDL3)",
            "Vulkan and the synthetic M1.5 shader fixture with a spirv/ directory",
            "Wine when running the Windows loader outside Windows",
        ),
    ),
)


def discover() -> dict:
    tools = []
    for name, path, requirements in TOOLS:
        present = (ROOT / path).is_file()
        tools.append({
            "id": name,
            "path": path,
            "present": present,
            "help_argv": ["python", "-I", path, "--help"] if present else None,
            "requirements": requirements,
        })
    return {
        "schema_version": 1,
        "tools": tools,
        "live_controls": {
            "available": False,
            "reason": "This entry point does not expose live attach, pause, step, input or screenshots.",
        },
    }


def boot_trace(args: list[str]) -> int:
    producer = ROOT / BOOT_TRACE
    if not producer.is_file():
        print(f"error: {BOOT_TRACE} is missing; use a complete Leonida checkout", file=sys.stderr)
        return 2
    try:
        return subprocess.run(
            [sys.executable, "-I", str(producer), *args],
            cwd=Path.cwd(),
        ).returncode
    except OSError as exc:
        print(f"error: could not run {BOOT_TRACE}: {exc}", file=sys.stderr)
        return 2


def main() -> int:
    # The producer owns its arguments and help. Forward before argparse handles them.
    argv = sys.argv[1:]
    if argv[:1] == ["boot-trace"]:
        return boot_trace(argv[1:])

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inventory = commands.add_parser("discover", help="list producer scripts and their requirements")
    inventory.add_argument("--json", action="store_true", help="write a JSON inventory to stdout")
    commands.add_parser("boot-trace", help="summarize startup and D3D12 boot traces")
    args = parser.parse_args(argv)

    result = discover()
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("Paths and help commands are relative to the Leonida checkout root.")
        for tool in result["tools"]:
            status = "present" if tool["present"] else "missing"
            print(f"{tool['id']}: {tool['path']} ({status})")
            if tool["help_argv"]:
                print(f"  Help: {' '.join(tool['help_argv'])}")
            print("  Requirements (not checked):")
            for requirement in tool["requirements"]:
                print(f"    - {requirement}")
        print(f"Live controls: unavailable. {result['live_controls']['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
