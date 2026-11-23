"""Automatic installer for LLVM llvm-mc (AMDGPU target, from llvm/llvm-project).

Usage:
    python -I scripts/install_llvm.py [--version 21.1.8] [--dest DIR] [--no-env] [--check]
    python -I scripts/install_llvm.py --prebuilt-url URL --expected-sha256 HEX [--dest DIR]

1. Respects any working LLVM_MC already configured (including command lines such as
   `wsl -d Ubuntu -e /usr/lib/llvm-21/bin/llvm-mc`), or a working llvm-mc on PATH / in --dest.
2. By default, builds llvm-mc (AMDGPU target only) directly from official
   https://github.com/llvm/llvm-project sources (scripts/build_llvm_mc.ps1 on Windows,
   CMake + Ninja on Linux/macOS).
3. Optional prebuilt archive download requires BOTH --prebuilt-url and --expected-sha256
   and fails closed if the SHA-256 does not match.

Standard library only.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPTS_DIR)
TOOLS_DIR = os.path.join(ROOT_DIR, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from mc_command import split_command  # noqa: E402

DEFAULT_VERSION = "21.1.8"
LLVM_REPO = "https://github.com/llvm/llvm-project"
PROBE_BYTES = "0x00 0x00 0x81 0xbf\n"
HEX64_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def default_dest() -> str:
    return os.path.join(os.path.expanduser("~"), ".leonida", "llvm-mc")


def exe_name() -> str:
    return "llvm-mc.exe" if os.name == "nt" else "llvm-mc"


def probe_llvm_mc(cmd: list[str]) -> bool:
    """Verify an llvm-mc command line can disassemble a gfx1013 s_endpgm instruction."""
    if not cmd:
        return False
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8", newline="\n") as f:
        f.write(PROBE_BYTES)
        probe_path = f.name
    try:
        r = subprocess.run(
            [*cmd, "--disassemble", "-triple=amdgcn", "-mcpu=gfx1013", probe_path],
            capture_output=True, text=True, errors="replace", timeout=20,
        )
        return r.returncode == 0 and "s_endpgm" in r.stdout
    except (OSError, subprocess.SubprocessError):
        return False
    finally:
        try:
            os.remove(probe_path)
        except OSError:
            pass


def find_existing_llvm_mc(dest: str) -> tuple[str, bool] | None:
    """Return (command_or_path, from_env) for an existing working llvm-mc, checking LLVM_MC first."""
    env_raw = os.environ.get("LLVM_MC", "").strip()
    if env_raw:
        try:
            cmd = split_command(env_raw)
        except SystemExit:
            cmd = []
        if cmd and probe_llvm_mc(cmd):
            return env_raw, True

    local_bin = os.path.join(dest, exe_name())
    if os.path.isfile(local_bin) and probe_llvm_mc([local_bin]):
        return local_bin, False

    which_mc = shutil.which("llvm-mc")
    if which_mc and probe_llvm_mc([which_mc]):
        return which_mc, False

    return None


def _download(url: str, dest_path: str) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "Leonida-LLVM-Installer/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(dest_path, "wb") as out:
        shutil.copyfileobj(resp, out)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_extract_zip(archive: str, dest: str) -> None:
    real_dest = os.path.realpath(dest)
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            target = os.path.realpath(os.path.join(real_dest, info.filename))
            if os.path.commonpath([real_dest, target]) != real_dest:
                raise RuntimeError(f"Unsafe zip member path: {info.filename}")
        zf.extractall(real_dest)


def install_prebuilt(url: str, expected_sha256: str, dest: str) -> str:
    """Download and verify a prebuilt llvm-mc archive; fails closed if SHA-256 is missing or mismatches."""
    if not expected_sha256 or not HEX64_RE.match(expected_sha256.strip()):
        raise RuntimeError("--expected-sha256 must be a valid 64-character hex SHA-256 digest.")
    expected = expected_sha256.strip().lower()

    with tempfile.TemporaryDirectory(prefix="leonida-llvm-") as tmp:
        name = os.path.basename(url.split("?", 1)[0]) or "llvm-mc-archive"
        archive = os.path.join(tmp, name)
        print(f"Downloading {url} ...", flush=True)
        _download(url, archive)

        actual = _sha256(archive)
        if actual != expected:
            raise RuntimeError(f"SHA-256 mismatch for {name}: got {actual}, expected {expected}")

        os.makedirs(dest, exist_ok=True)
        if name.lower().endswith(".zip"):
            _safe_extract_zip(archive, dest)
        else:
            with tarfile.open(archive, "r:*") as tf:
                tf.extractall(dest, filter="data")

    binary = os.path.join(dest, exe_name())
    if not os.path.isfile(binary):
        raise RuntimeError(f"Expected {exe_name()} was not found after extracting {url}.")
    if os.name != "nt":
        os.chmod(binary, os.stat(binary).st_mode | 0o111)
    if not probe_llvm_mc([binary]):
        raise RuntimeError(f"Downloaded {binary} failed the gfx1013 s_endpgm probe.")
    return binary


def build_from_source(version: str, dest: str) -> str:
    if os.name == "nt":
        ps1 = os.path.join(SCRIPTS_DIR, "build_llvm_mc.ps1")
        if not os.path.isfile(ps1):
            raise RuntimeError(f"Missing {ps1}")
        r = subprocess.run(
            ["powershell", "-ExecutionPolicy", "Bypass", "-File", ps1, "-Version", version, "-Dest", dest, "-NoEnv"],
            check=False,
        )
        if r.returncode != 0:
            raise RuntimeError(f"build_llvm_mc.ps1 failed with exit code {r.returncode}")
        return os.path.join(dest, "llvm-mc.exe")

    for tool in ("git", "cmake", "ninja"):
        if not shutil.which(tool):
            raise RuntimeError(f"'{tool}' is required to build llvm-mc from source.")

    with tempfile.TemporaryDirectory(prefix="leonida-llvm-src-") as tmp:
        src = os.path.join(tmp, "src")
        bld = os.path.join(tmp, "build")
        print(f"Cloning {LLVM_REPO} (llvmorg-{version}, sparse) ...", flush=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", f"llvmorg-{version}", "--filter=blob:none", "--sparse", LLVM_REPO, src],
            check=True,
        )
        subprocess.run(["git", "-C", src, "sparse-checkout", "set", "llvm", "cmake", "third-party"], check=True)
        jobs = str(max(1, (os.cpu_count() or 2) // 2))
        subprocess.run(
            [
                "cmake", "-S", os.path.join(src, "llvm"), "-B", bld, "-G", "Ninja",
                "-DCMAKE_BUILD_TYPE=Release",
                "-DLLVM_TARGETS_TO_BUILD=AMDGPU",
                "-DLLVM_INCLUDE_TESTS=OFF",
                "-DLLVM_INCLUDE_BENCHMARKS=OFF",
                "-DLLVM_INCLUDE_EXAMPLES=OFF",
                "-DLLVM_INCLUDE_DOCS=OFF",
                "-DLLVM_ENABLE_ZLIB=OFF",
                "-DLLVM_ENABLE_ZSTD=OFF",
                "-DLLVM_ENABLE_LIBXML2=OFF",
            ],
            check=True,
        )
        subprocess.run(["cmake", "--build", bld, "--target", "llvm-mc", "-j", jobs], check=True)
        built = os.path.join(bld, "bin", "llvm-mc")
        if not probe_llvm_mc([built]):
            raise RuntimeError("Built llvm-mc failed the gfx1013 s_endpgm probe.")
        os.makedirs(dest, exist_ok=True)
        final = os.path.join(dest, "llvm-mc")
        shutil.copy2(built, final)
        os.chmod(final, os.stat(final).st_mode | 0o111)
        return final


def set_user_env(binary: str) -> None:
    os.environ["LLVM_MC"] = binary
    if os.name == "nt":
        import ctypes
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0, winreg.KEY_SET_VALUE) as key:
            winreg.SetValueEx(key, "LLVM_MC", 0, winreg.REG_SZ, binary)
        ctypes.windll.user32.SendMessageTimeoutW(0xFFFF, 0x001A, 0, "Environment", 0x0002, 5000, None)
        print(f"LLVM_MC set to {binary} (User environment).", flush=True)
    else:
        print(f"Installed to {binary}. Add to your shell profile: export LLVM_MC=\"{binary}\"", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Automatic LLVM llvm-mc (AMDGPU) installer from llvm/llvm-project.")
    ap.add_argument("--version", default=DEFAULT_VERSION, help=f"LLVM version (default {DEFAULT_VERSION})")
    ap.add_argument("--dest", default=default_dest(), help="Destination directory (default ~/.leonida/llvm-mc)")
    ap.add_argument("--prebuilt-url", default="", help="Optional prebuilt archive URL (requires --expected-sha256)")
    ap.add_argument("--expected-sha256", default="", help="Pinned 64-char hex SHA-256 for --prebuilt-url")
    ap.add_argument("--from-source", action="store_true", help="Build directly from llvm/llvm-project (default)")
    ap.add_argument("--no-env", action="store_true", help="Do not update the LLVM_MC user environment variable")
    ap.add_argument("--check", action="store_true", help="Only verify whether a working llvm-mc (AMDGPU) is already installed")
    args = ap.parse_args(argv)

    existing = find_existing_llvm_mc(args.dest)
    if existing:
        found, from_env = existing
        print(f"[OK] llvm-mc (AMDGPU) already configured: {found}", flush=True)
        if not from_env and not args.no_env and not args.check and not os.environ.get("LLVM_MC", "").strip():
            set_user_env(found)
        return 0
    if args.check:
        print("[MISSING] llvm-mc (AMDGPU) not found.", flush=True)
        return 1

    print(f"Platform: {platform.system()} ({platform.machine()})", flush=True)
    if args.prebuilt_url and not args.from_source:
        binary = install_prebuilt(args.prebuilt_url, args.expected_sha256, args.dest)
    else:
        binary = build_from_source(args.version, args.dest)

    if not args.no_env:
        set_user_env(binary)
    print(f"[OK] llvm-mc ready at {binary}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
