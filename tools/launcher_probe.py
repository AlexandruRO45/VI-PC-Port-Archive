r"""What Leonida.exe (launcher-wpf/) needs to know about this PC, as one JSON object on stdout.

    python -I tools/launcher_probe.py                 # defaults, stages, options and the tool checks
    python -I tools/launcher_probe.py --find          # also the game copies make_game.py would find
    python -I tools/launcher_probe.py --gpu           # also the GPU the runtime would use (tools/vk_subgroup_info.py)
    python -I tools/launcher_probe.py --free PATH     # also the free space on PATH's drive

Every answer comes from make_game.py's own functions (the same checks its preflight makes, the same search as
--find), so the launcher never keeps a second copy of the rules that could drift from the builder. Nothing is
written, and no game file is read beyond what make_game.py --find reads (folder names and a few file sizes).
"""
import argparse
import contextlib
import io
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import make_game  # noqa: E402  (its imports put tools/ on sys.path)
from mc_command import mc_command  # noqa: E402


def tool(name: str, ok: bool, detail: str, needed_for: str, hint: str = "", made_by_build: bool = False) -> dict:
    """made_by_build: missing now, but the build makes it itself when it gets there (not something to install)."""
    return {"name": name, "ok": bool(ok), "detail": detail, "needed_for": needed_for, "hint": hint,
            "made_by_build": bool(made_by_build) and not ok}


def which_detail(exe: str) -> tuple[bool, str]:
    path = shutil.which(exe)
    return (path is not None, path or "not found on PATH")


def tools() -> list:
    out = []
    py = sys.version_info
    out.append(tool("Python", py >= (3, 12), f"{py.major}.{py.minor}.{py.micro} at {sys.executable}", "every stage",
                     "Python 3.12 or newer: https://www.python.org/downloads/"))
    mc = mc_command()
    found = os.path.isfile(mc[0]) or shutil.which(mc[0]) is not None
    out.append(tool("llvm-mc", found, " ".join(mc) + ("" if found else " (not found)"), "shaders",
                    "LLVM 21+ with the AMDGPU target: scripts\\build_llvm_mc.ps1, or set LLVM_MC (README)"
                    + make_game.appdata_hint(mc[0])))
    for t in ("spirv-as", "spirv-val"):
        ok, detail = which_detail(t)
        out.append(tool(t, ok, detail, "shaders", "install the Vulkan SDK: https://vulkan.lunarg.com/"))
    out.append(tool(".NET 8 SDK", make_game.dotnet_sdk_ok(), shutil.which("dotnet") or "dotnet not found", "assets",
                    "https://dotnet.microsoft.com/download/dotnet/8.0"))
    ok, detail = which_detail("ffmpeg")
    out.append(tool("ffmpeg", ok, detail, "assets (--wav, XMA2 streams)", "https://ffmpeg.org/download.html"))
    ok, detail = which_detail("git")
    out.append(tool("git", ok, detail, "assets (fetches third_party/rpf9-explorer when missing)",
                    "https://git-scm.com/download/win"))
    if os.name == "nt":
        out.append(tool("Visual Studio C++ tools", make_game.visual_cpp_tools_ok(), "vswhere: x64 C++ tools",
                        "runtime", "Visual Studio 2022 with 'Desktop development with C++'"))
    sdk = os.environ.get("VULKAN_SDK")
    out.append(tool("VULKAN_SDK", bool(sdk), sdk or "not set", "runtime (builds vk_pipeline_check), pipelines",
                    "install the Vulkan SDK and reopen Leonida.exe"))
    checker = make_game.pipeline_check_exe()
    # The runtime stage builds it on Windows when VULKAN_SDK is set (pipeline_check_prerequisite).
    builds_it = os.name == "nt" and bool(sdk) and make_game.visual_cpp_tools_ok()
    out.append(tool("vk_pipeline_check", checker is not None,
                    checker or ("not built yet: the runtime stage builds it" if builds_it else "not built yet"), "pipelines",
                    make_game.pipeline_check_prerequisite(True), made_by_build=builds_it))
    sub = os.path.join(ROOT, "third_party", "rpf9-explorer")
    has_sub = os.path.isdir(sub) and bool(os.listdir(sub))
    out.append(tool("rpf9-explorer submodule", has_sub, sub if has_sub else "empty: the assets stage fetches it with git",
                    "assets", "git submodule update --init --recursive", made_by_build=shutil.which("git") is not None))
    return out


def find() -> dict:
    containers: list = []
    with contextlib.redirect_stdout(io.StringIO()):   # find_dumps prints nothing today; keep stdout pure JSON anyway
        found = make_game.find_dumps(containers)
    return {
        "copies": [{"path": d, "title": v.get("title"), "version": v.get("version")} for d, v in found],
        "unusable": [{"path": p, "kind": h.get("kind"), "version": h.get("version")} for p, h in containers],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--find", action="store_true", help="also list the game copies found on this PC")
    ap.add_argument("--gpu", action="store_true", help="also ask tools/vk_subgroup_info.py about the runtime's GPU")
    ap.add_argument("--free", metavar="PATH", help="also report the free space on PATH's drive")
    a = ap.parse_args()
    versions = []
    for name in sorted(os.listdir(make_game.VERSIONS)):
        if name.endswith(".json"):
            try:
                with open(os.path.join(make_game.VERSIONS, name), encoding="utf-8") as f:
                    v = json.load(f)
                versions.append({"file": name, "title": v.get("title"), "version": v.get("version"),
                                 "files": len(v.get("files", []))})
            except (OSError, ValueError):
                pass
    info = {
        "root": ROOT,
        "python": sys.executable,
        # The launcher reads make_game.py's output through a pipe, as this probe's: decode it the same way.
        "stdout_encoding": sys.stdout.encoding,
        "stages": make_game.STAGES,
        "wave64_modes": list(make_game.WAVE64_MODES),
        "default_out": make_game.default_out(),
        "default_jobs": max(1, (os.cpu_count() or 2) // 2),
        "cpu_count": os.cpu_count(),
        "min_free_gb": make_game.MIN_FREE_GB,
        "versions": versions,
        "tools": tools(),
    }
    if a.find:
        info.update(find())
    if a.gpu:
        info["gpu"] = make_game.runtime_gpu_info()
    if a.free:
        probe = os.path.abspath(a.free)
        while not os.path.exists(probe) and os.path.dirname(probe) != probe:
            probe = os.path.dirname(probe)
        try:
            info["free_bytes"] = shutil.disk_usage(probe).free
        except OSError as e:
            info["free_error"] = str(e)
    json.dump(info, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
