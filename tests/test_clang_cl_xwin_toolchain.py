"""Unit tests for runtime/cmake/clang-cl-xwin.cmake native Windows and cross-compile behavior.

Exercises the toolchain file via ``cmake -P`` against a synthetic xwin tree (no compiler, SDK download,
Vulkan SDK install, or game files required); skipped when cmake is not on PATH.
"""
import os
import shutil
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOLCHAIN = ROOT / "runtime" / "cmake" / "clang-cl-xwin.cmake"
CMAKE = shutil.which("cmake")


def make_fake_xwin(base: Path) -> Path:
    xwin = base / "xwin-sdk"
    (xwin / "VC" / "Tools" / "MSVC" / "14.42.34433" / "lib" / "x64").mkdir(parents=True)
    (xwin / "Windows Kits" / "10" / "Lib" / "10.0.26100.0" / "um" / "x64").mkdir(parents=True)
    (xwin / "Windows Kits" / "10" / "Include" / "10.0.26100.0" / "um").mkdir(parents=True)
    return xwin


def make_fake_wine(bin_dir: Path) -> Path:
    bin_dir.mkdir(parents=True, exist_ok=True)
    name = "wine.exe" if os.name == "nt" else "wine"
    wine = bin_dir / name
    wine.write_bytes(b"#!/bin/sh\nexit 0\n")
    if os.name != "nt":
        wine.chmod(wine.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return wine


def run_toolchain(
    tmp: Path,
    *,
    host_win32: bool,
    xwin_dir: Path,
    pass_xwin_via_env: bool = False,
    cmake_defs: dict[str, str] | None = None,
    env_extra: dict[str, str | None] | None = None,
    path_dir: Path | None = None,
):
    build_dir = tmp / "build"
    build_dir.mkdir(exist_ok=True)
    driver = tmp / "driver.cmake"
    driver.write_text(
        "\n".join(
            [
                f"set(CMAKE_HOST_WIN32 {'1' if host_win32 else '0'})",
                f'set(CMAKE_BINARY_DIR "{build_dir.as_posix()}")',
                "set(CMAKE_FIND_USE_CMAKE_SYSTEM_PATH OFF)",
                "set(CMAKE_FIND_USE_CMAKE_PATH OFF)",
                "set(CMAKE_FIND_USE_PACKAGE_REGISTRY OFF)",
                f'include("{TOOLCHAIN.as_posix()}")',
                'message(STATUS "RESULT:CMAKE_CXX_SCAN_FOR_MODULES=${CMAKE_CXX_SCAN_FOR_MODULES}")',
                'message(STATUS "RESULT:Vulkan_INCLUDE_DIR=${Vulkan_INCLUDE_DIR}")',
                'message(STATUS "RESULT:CMAKE_CROSSCOMPILING_EMULATOR=${CMAKE_CROSSCOMPILING_EMULATOR}")',
                'message(STATUS "RESULT:XWIN_WINE=${XWIN_WINE}")',
                'message(STATUS "RESULT:XWIN_DIR=${XWIN_DIR}")',
                'message(STATUS "RESULT:CMAKE_CXX_FLAGS_INIT=${CMAKE_CXX_FLAGS_INIT}")',
            ]
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    args = [CMAKE]
    if not pass_xwin_via_env:
        args.append(f"-DXWIN_DIR={xwin_dir.as_posix()}")
    for k, v in (cmake_defs or {}).items():
        args.append(f"-D{k}={v}")
    args += ["-P", str(driver)]

    env = dict(os.environ)
    env.pop("VULKAN_SDK", None)
    env.pop("XWIN_DIR", None)
    if pass_xwin_via_env:
        env["XWIN_DIR"] = str(xwin_dir)
    env["PATH"] = str(path_dir) if path_dir is not None else ""
    for k, v in (env_extra or {}).items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = v

    proc = subprocess.run(args, capture_output=True, text=True, env=env, timeout=60)
    results: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        if line.startswith("-- RESULT:"):
            kv = line[len("-- RESULT:") :]
            k, _, v = kv.partition("=")
            results[k] = v
    return proc, results, build_dir


@unittest.skipUnless(CMAKE, "cmake not on PATH")
class ClangClXwinToolchainTest(unittest.TestCase):
    def test_defaults_cxx_scan_for_modules_off(self):
        for host_win32 in (True, False):
            with self.subTest(host_win32=host_win32):
                with tempfile.TemporaryDirectory() as td:
                    tmp = Path(td)
                    xwin = make_fake_xwin(tmp)
                    proc, res, _ = run_toolchain(tmp, host_win32=host_win32, xwin_dir=xwin)
                    self.assertEqual(proc.returncode, 0, proc.stderr)
                    self.assertEqual(res.get("CMAKE_CXX_SCAN_FOR_MODULES"), "OFF")

    def test_respects_explicit_cxx_scan_for_modules_on(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            xwin = make_fake_xwin(tmp)
            proc, res, _ = run_toolchain(
                tmp,
                host_win32=True,
                xwin_dir=xwin,
                cmake_defs={"CMAKE_CXX_SCAN_FOR_MODULES": "ON"},
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(res.get("CMAKE_CXX_SCAN_FOR_MODULES"), "ON")

    def test_windows_host_skips_wine_lookup_and_warning(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            xwin = make_fake_xwin(tmp)
            # Even when a wine binary is on PATH, native Windows must not use it or warn.
            fake_bin = tmp / "bin"
            make_fake_wine(fake_bin)
            for path_dir in (None, fake_bin):
                with self.subTest(has_wine_on_path=path_dir is not None):
                    proc, res, _ = run_toolchain(tmp, host_win32=True, xwin_dir=xwin, path_dir=path_dir)
                    self.assertEqual(proc.returncode, 0, proc.stderr)
                    self.assertEqual(res.get("CMAKE_CROSSCOMPILING_EMULATOR"), "")
                    self.assertEqual(res.get("XWIN_WINE"), "")
                    self.assertNotIn("Wine not found", proc.stdout + proc.stderr)

    def test_windows_host_respects_explicit_emulator(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            xwin = make_fake_xwin(tmp)
            proc, res, _ = run_toolchain(
                tmp,
                host_win32=True,
                xwin_dir=xwin,
                cmake_defs={"CMAKE_CROSSCOMPILING_EMULATOR": "/custom/emu"},
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(res.get("CMAKE_CROSSCOMPILING_EMULATOR"), "/custom/emu")

    def test_non_windows_host_warns_when_wine_missing(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            xwin = make_fake_xwin(tmp)
            proc, res, _ = run_toolchain(tmp, host_win32=False, xwin_dir=xwin, path_dir=None)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(res.get("CMAKE_CROSSCOMPILING_EMULATOR"), "")
            self.assertIn("Wine not found", proc.stdout + proc.stderr)

    def test_non_windows_host_finds_wine_when_present(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            xwin = make_fake_xwin(tmp)
            fake_bin = tmp / "bin"
            wine = make_fake_wine(fake_bin)
            proc, res, _ = run_toolchain(tmp, host_win32=False, xwin_dir=xwin, path_dir=fake_bin)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(Path(res.get("CMAKE_CROSSCOMPILING_EMULATOR", "")).resolve(), wine.resolve())
            self.assertNotIn("Wine not found", proc.stdout + proc.stderr)

    def test_windows_host_discovers_vulkan_sdk_include(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            xwin = make_fake_xwin(tmp)
            sdk = tmp / "VulkanSDK" / "1.4.304.0"
            (sdk / "Include" / "vulkan").mkdir(parents=True)
            (sdk / "Include" / "vulkan" / "vulkan.h").write_text("// stub\n", encoding="utf-8")

            # Trailing separator exercises normalization on both Windows and POSIX.
            sdk_env = str(sdk) + os.sep
            proc, res, build_dir = run_toolchain(
                tmp,
                host_win32=True,
                xwin_dir=xwin,
                env_extra={"VULKAN_SDK": sdk_env},
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(res.get("Vulkan_INCLUDE_DIR"), (sdk / "Include").as_posix())
            self.assertFalse((build_dir / "xwin-vulkan-include").exists())

    def test_windows_host_discovers_vulkan_sdk_lowercase_include(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            xwin = make_fake_xwin(tmp)
            sdk = tmp / "vulkan-headers"
            (sdk / "include" / "vulkan").mkdir(parents=True)
            (sdk / "include" / "vulkan" / "vulkan.h").write_text("// stub\n", encoding="utf-8")

            proc, res, _ = run_toolchain(
                tmp,
                host_win32=True,
                xwin_dir=xwin,
                env_extra={"VULKAN_SDK": str(sdk)},
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            # On case-insensitive filesystems (Windows), Include matches first; either resolves to the same dir.
            found = Path(res.get("Vulkan_INCLUDE_DIR", ""))
            self.assertEqual(found.resolve(), (sdk / "include").resolve())
            self.assertTrue((found / "vulkan" / "vulkan.h").is_file())

    def test_windows_host_respects_explicit_vulkan_include_dir(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            xwin = make_fake_xwin(tmp)
            sdk = tmp / "VulkanSDK"
            (sdk / "Include" / "vulkan").mkdir(parents=True)
            (sdk / "Include" / "vulkan" / "vulkan.h").write_text("// stub\n", encoding="utf-8")
            custom = tmp / "custom-vk"
            (custom / "vulkan").mkdir(parents=True)
            (custom / "vulkan" / "vulkan.h").write_text("// custom\n", encoding="utf-8")

            proc, res, _ = run_toolchain(
                tmp,
                host_win32=True,
                xwin_dir=xwin,
                cmake_defs={"Vulkan_INCLUDE_DIR": custom.as_posix()},
                env_extra={"VULKAN_SDK": str(sdk)},
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(res.get("Vulkan_INCLUDE_DIR"), custom.as_posix())

    def test_windows_host_replaces_notfound_vulkan_include_dir(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            xwin = make_fake_xwin(tmp)
            sdk = tmp / "VulkanSDK"
            (sdk / "Include" / "vulkan").mkdir(parents=True)
            (sdk / "Include" / "vulkan" / "vulkan.h").write_text("// stub\n", encoding="utf-8")

            proc, res, _ = run_toolchain(
                tmp,
                host_win32=True,
                xwin_dir=xwin,
                cmake_defs={"Vulkan_INCLUDE_DIR": "Vulkan_INCLUDE_DIR-NOTFOUND"},
                env_extra={"VULKAN_SDK": str(sdk)},
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(res.get("Vulkan_INCLUDE_DIR"), (sdk / "Include").as_posix())

    def test_windows_host_ignores_vulkan_sdk_without_headers(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            xwin = make_fake_xwin(tmp)
            empty_sdk = tmp / "EmptyVulkanSDK"
            empty_sdk.mkdir()

            proc, res, _ = run_toolchain(
                tmp,
                host_win32=True,
                xwin_dir=xwin,
                env_extra={"VULKAN_SDK": str(empty_sdk)},
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(res.get("Vulkan_INCLUDE_DIR"), "")

    def test_xwin_dir_from_env_normalizes_separators(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            xwin = make_fake_xwin(tmp)
            proc, res, _ = run_toolchain(tmp, host_win32=True, xwin_dir=xwin, pass_xwin_via_env=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(res.get("XWIN_DIR"), xwin.as_posix())
            self.assertNotIn("\\", res.get("CMAKE_CXX_FLAGS_INIT", ""))


if __name__ == "__main__":
    unittest.main()
