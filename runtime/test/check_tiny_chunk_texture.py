"""Run a mapped guest across both chunks and check the verdict outside its address space."""
import argparse
import os
import re
import subprocess
from pathlib import Path

PASS = "tiny_chunk_texture: PASS 5 frames, 10240 pixels, 2 texture allocations and final releases"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("loader", type=Path)
    ap.add_argument("guest", type=Path)
    ap.add_argument("fixture", type=Path)
    ap.add_argument("--runner", action="append", default=[])
    opt = ap.parse_args()
    # Packed Windows CTest entries can be replayed on Wine. All transferred paths are under Wine's Z: drive.
    def guest_path(path):
        path = str(path.resolve())
        return "Z:" + path.replace("/", "\\") if opt.runner and os.name != "nt" else path

    fixture = opt.fixture.resolve()
    env = dict(os.environ, LEONIDA_D3D12="vulkan", LEONIDA_SHADERS=guest_path(fixture / "spirv"))
    proc = subprocess.run([*opt.runner, str(opt.loader.resolve()), "--run", opt.guest.name, "--", guest_path(fixture)],
                          cwd=opt.guest.resolve().parent, env=env, timeout=120, capture_output=True,
                          text=True, encoding="utf-8", errors="replace")
    output = proc.stdout + proc.stderr
    lines = output.splitlines()
    bound = any(re.fullmatch(r"\s*shim\s+d3d12_xs\.dll!D3D12XboxCreateDevice", line) for line in lines)
    if (proc.returncode or lines.count(PASS) != 1 or "game exited with code 0" not in lines
            or not bound or "(Vulkan device " not in output):
        print(output)
        print(f"guest texture check failed (loader exit {proc.returncode})")
        return 1
    print(PASS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
