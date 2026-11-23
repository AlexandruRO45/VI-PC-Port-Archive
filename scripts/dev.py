#!/usr/bin/env python3
"""Discover development tools in this Leonida checkout."""

import argparse
import json
import os
import shlex
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = (
    (
        "boot-trace",
        "tools/boot_trace.py",
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
            "help_argv": [sys.executable, "-I", path, "--help"] if present else None,
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


def format_help_command(argv: list[str]) -> str:
    if os.name == "nt":
        quoted = ("'" + arg.replace("'", "''") + "'" for arg in argv)
        return "& " + " ".join(quoted)
    return shlex.join(argv)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inventory = commands.add_parser("discover", help="list producer scripts and their requirements")
    inventory.add_argument("--json", action="store_true", help="write a JSON inventory to stdout")
    args = parser.parse_args()

    result = discover()
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        sys.stdout.reconfigure(encoding="utf-8")
        print("Paths and help commands are relative to the Leonida checkout root.")
        for tool in result["tools"]:
            status = "present" if tool["present"] else "missing"
            print(f"{tool['id']}: {tool['path']} ({status})")
            if tool["help_argv"]:
                print(f"  Help: {format_help_command(tool['help_argv'])}")
            print("  Requirements (not checked):")
            for requirement in tool["requirements"]:
                print(f"    - {requirement}")
        print(f"Live controls: unavailable. {result['live_controls']['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
