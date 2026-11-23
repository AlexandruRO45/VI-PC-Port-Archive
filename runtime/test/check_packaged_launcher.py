"""The packaged launcher on Windows: make_game's runtime stage installs this gta6rt build in a port folder, and
`<out>\\gta6rt.cmd --dry-run`, `--init` and `--run` start it on the package named in game_path.txt, with the port's
overlay for writes, as gta6rt.sh does under Wine.

    python -I runtime/test/check_packaged_launcher.py --loader gta6rt.exe --tls-host gta6rt_tls_host.dll \\
        --guest tiny_game.exe --work-dir <scratch>

The package is synthetic: the stand-in guest under the executable name the version manifest expects. The launcher
passes the game no arguments (gta6rt.sh neither), so tiny_game reports argc=1 and fails its own argv check; what is
checked is that it was started from the package and that its exit code comes back through gta6rt.cmd. The folders
carry non-ASCII characters and cmd's special characters, as a player's may.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for name in ("loader", "tls-host", "guest", "work-dir"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    if os.name != "nt":
        parser.error("gta6rt.cmd is the Windows launcher; under Wine make_game writes gta6rt.sh")
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")   # the folder names are not ANSI
    sys.path.insert(0, str(ROOT))
    import make_game  # noqa: E402  (the runtime stage under test)

    names = set()
    for manifest in glob.glob(os.path.join(make_game.VERSIONS, "*.json")):
        with open(manifest, encoding="utf-8") as f:
            names.add(json.load(f)["expected_exe"])
    if len(names) != 1:
        parser.error(f"expected one executable name across the version manifests, found {sorted(names)}")
    exe = names.pop()
    environment = {k: v for k, v in os.environ.items() if not k.upper().startswith("LEONIDA_")}
    failures: list[str] = []
    args.work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="packaged_launcher_", dir=args.work_dir.resolve()) as temp:
        root = Path(os.path.realpath(temp))
        build = root / "build"
        game = root / "Gäme λ 100% & (x86) !"
        out = root / "Port Ü (x86) & co"
        build.mkdir()
        game.mkdir()
        shutil.copyfile(args.loader, build / "gta6rt.exe")
        shutil.copyfile(args.tls_host, build / "gta6rt_tls_host.dll")
        shutil.copyfile(args.guest, game / exe)
        make_game.runtime(str(game), str(out), {"expected_exe": exe}, runtime_build=str(build), work=str(out / "work"))
        launcher = out / "gta6rt.cmd"
        if not launcher.is_file():
            print(f"FAIL the runtime stage wrote no {launcher.name}")
            return 1
        for mode in ("--dry-run", "--init", "--run"):
            # A console of its own: cmd starts in the system's OEM code page there, and the launcher's code page
            # switch never reaches the console ctest runs in.
            child = subprocess.run([str(launcher), mode], cwd=root, env=environment, capture_output=True,
                                   timeout=60, creationflags=subprocess.CREATE_NO_WINDOW)
            output = (child.stdout + child.stderr).decode("utf-8", "replace").replace("\r\n", "\n")
            errors = []
            if mode != "--run":
                if child.returncode != 0:
                    errors.append(f"exit code {child.returncode}")
                if not re.search(r"(^|\n)1 module\(s\) mapped, 0 unimplemented import\(s\)\n", output):
                    errors.append("no binding report for the packaged executable")
            else:
                paths = f"console paths: GAME: -> {game}, writes -> {out / 'overlay'}\n"
                if paths not in output:
                    errors.append(f"expected {paths!r}")
                if f"running {exe}\n" not in output or "tiny_game: started, argc=1\n" not in output:
                    errors.append("the packaged executable did not start")
                exited = re.search(r"(^|\n)game exited with code (-?\d+)\n", output)
                if not exited or int(exited.group(2)) != child.returncode:
                    errors.append(f"gta6rt.cmd returned {child.returncode}, not the game's exit code")
            print(f"{'FAIL' if errors else 'PASS'} gta6rt.cmd {mode}: exit {child.returncode}" +
                  "".join(f"\n  {e}" for e in errors))
            if errors:
                print(output)
                failures += errors
    print(f"packaged launcher: {len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
