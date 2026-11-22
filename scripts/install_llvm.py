"""Automatic installer for LLVM llvm-mc (AMDGPU target, from llvm/llvm-project).

Usage:
    python -I scripts/install_llvm.py [--version 21.1.8] [--dest DIR] [--from-source] [--no-env] [--check]

1. Checks if a working llvm-mc with AMDGPU (gfx1013) is already available.
2. Otherwise downloads the prebuilt llvm-mc built from https://github.com/llvm/llvm-project
   (verified with SHA-256) from GitHub Releases.
3. Falls back to building from https://github.com/llvm/llvm-project source (--from-source).
4. Installs to ~/.leonida/llvm-mc and sets LLVM_MC on Windows.

Standard library only.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
import zipfile

DEFAULT_VERSION = "21.1.8"
RELEASE_REPO = "FelipeMzero/Leonida"
LLVM_REPO = "https://github.com/llvm/llvm-project"
PROBE_BYTES = "0x00 0x00 0x81 0xbf\n"


def default_dest() -> str:
    return os.path.join(os.path.expanduser("~"), ".leonida", "llvm-mc")


def exe_name() -> str:
    return "llvm-mc.exe" if os.name == "nt" else "llvm-mc"


def probe_llvm_mc(cmd: list[str]) -> bool:
    """Verify llvm-mc can disassemble a gfx1013 s_endpgm instruction."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
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


def find_existing_llvm_mc(dest: str) -> str | None:
    candidates = []
    env_mc = os.environ.get("LLVM_MC", "").strip().strip('"')
    if env_mc:
        candidates.append(env_mc)
    candidates.append(os.path.join(dest, exe_name()))
    which_mc = shutil.which("llvm-mc")
    if which_mc:
        candidates.append(which_mc)
    for c in candidates:
        if os.path.isfile(c) and probe_llvm_mc([c]):
            return c
    return None


def platform_asset_name() -> str:
    if os.name == "nt":
        return "llvm-mc-windows-x64.zip"
    if sys.platform == "darwin":
        return "llvm-mc-macos-arm64.tar.gz"
    return "llvm-mc-linux-x64.tar.gz"


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


def _parse_checksums(path: str) -> dict[str, str]:
    table: dict[str, str] = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                table[os.path.basename(parts[1].lstrip("*"))] = parts[0].lower()
    return table


def install_prebuilt(version: str, dest: str) -> str | None:
    asset = platform_asset_name()
    base = f"https://github.com/{RELEASE_REPO}/releases/download/llvm-{version}"
    asset_url = f"{base}/{asset}"
    sums_url = f"{base}/SHA256SUMS.txt"

    with tempfile.TemporaryDirectory(prefix="leonida-llvm-") as tmp:
        archive = os.path.join(tmp, asset)
        sums_path = os.path.join(tmp, "SHA256SUMS.txt")
        print(f"Downloading {asset_url} ...", flush=True)
        try:
            _download(asset_url, archive)
            _download(sums_url, sums_path)
        except (urllib.error.URLError, OSError) as exc:
            print(f"Prebuilt download unavailable ({exc}).", flush=True)
            return None

        sums = _parse_checksums(sums_path)
        expected = sums.get(asset)
        actual = _sha256(archive)
        if expected and actual != expected:
            raise RuntimeError(f"SHA-256 mismatch for {asset}: got {actual}, expected {expected}")

        os.makedirs(dest, exist_ok=True)
        if asset.endswith(".zip"):
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(dest)
        else:
            with tarfile.open(archive, "r:*") as tf:
                tf.extractall(dest)

    binary = os.path.join(dest, exe_name())
    if not os.path.isfile(binary):
        return None
    if os.name != "nt":
        os.chmod(binary, os.stat(binary).st_mode | 0o111)
    if not probe_llvm_mc([binary]):
        raise RuntimeError(f"Downloaded {binary} failed the gfx1013 s_endpgm probe.")
    return binary


def build_from_source(version: str, dest: str) -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.name == "nt":
        ps1 = os.path.join(script_dir, "build_llvm_mc.ps1")
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
    ap.add_argument("--from-source", action="store_true", help="Compile directly from llvm/llvm-project instead of downloading prebuilt")
    ap.add_argument("--no-env", action="store_true", help="Do not update the LLVM_MC user environment variable")
    ap.add_argument("--check", action="store_true", help="Only verify whether a working llvm-mc (AMDGPU) is already installed")
    args = ap.parse_args(argv)

    existing = find_existing_llvm_mc(args.dest)
    if existing:
        print(f"[OK] llvm-mc (AMDGPU) already installed: {existing}", flush=True)
        if not args.no_env and not args.check and os.environ.get("LLVM_MC") != existing:
            set_user_env(existing)
        return 0
    if args.check:
        print("[MISSING] llvm-mc (AMDGPU) not found.", flush=True)
        return 1

    print(f"Platform: {platform.system()} ({platform.machine()})", flush=True)
    binary = None if args.from_source else install_prebuilt(args.version, args.dest)
    if not binary:
        print("Falling back to building llvm-mc from llvm/llvm-project source...", flush=True)
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
