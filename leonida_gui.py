"""Entry point for the Leonida Desktop GUI (with --version and --self-check for CI/release validation)."""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from gui import APP_NAME, APP_VERSION  # noqa: E402
from gui.core import resolve_repo_root  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Leonida Desktop Builder & Validator GUI")
    ap.add_argument("--version", action="store_true", help="Print version and exit")
    ap.add_argument("--self-check", action="store_true", help="Run built-in codebase/bundle validation and exit")
    args, _unknown = ap.parse_known_args(argv)

    if args.version:
        print(f"{APP_NAME} GUI {APP_VERSION}")
        return 0

    if args.self_check:
        root = resolve_repo_root()
        tools_dir = os.path.join(root, "tools")
        if tools_dir not in sys.path:
            sys.path.insert(0, tools_dir)
        import validate  # noqa: E402

        rep = validate.validate_codebase(root_dir=root)
        print(f"[{'PASS' if rep.ok else 'FAIL'}] {rep.summary}")
        for c in rep.checks:
            print(f"  [{c['status'].upper():<5}] {c['name']}: {c['detail']}")
        return 0 if rep.ok else 1

    try:
        from gui.app import run_gui  # noqa: E402
    except ImportError as exc:
        print(
            f"PySide6 is required for the Leonida GUI ({exc}).\n"
            "Install it with:  pip install -r gui/requirements.txt\n"
            "Or run the CLI directly:  python -I make_game.py",
            file=sys.stderr,
        )
        return 1
    return run_gui()


if __name__ == "__main__":
    sys.exit(main())
