"""scripts/build_runtime.ps1's cl.exe fallback builds every GPU test harness the Python tools need (no build run).

The fallback is what runs when Visual Studio has no "C++ CMake tools" component. Without these programs,
check.py / run_gpu_tests.py stop with "missing vk_compute_test", and so do the fuzzers.

    python -I tests/test_build_runtime_fallback.py
"""
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "runtime" / "test" / "shaders"), str(ROOT / "tools")]
import check  # noqa: E402
import image_fuzz  # noqa: E402
import run_gpu_tests  # noqa: E402

SCRIPT = ROOT / "scripts" / "build_runtime.ps1"


def fallback_body(text: str) -> str:
    """The cl.exe branch: from the line-starting `} else {` after the CMake test to the end of its here-string."""
    start = re.search(r"^\} else \{", text, re.M).start()
    return text[start:text.index('"@', text.index('$body = @"', start))]


class FallbackBuildsTheHarnesses(unittest.TestCase):
    def setUp(self):
        self.body = fallback_body(SCRIPT.read_text(encoding="utf-8"))
        self.programs = set(re.findall(r"/Fe:([A-Za-z0-9_]+)\.exe", self.body))

    def test_the_harnesses_the_gpu_tools_require_are_built(self):
        needed = set(run_gpu_tests.REQUIRED_TEST_PROGRAMS) | set(check.TEST_PROGRAMS) | {image_fuzz.HARNESS}
        self.assertEqual(needed - self.programs, set(), "missing from build_runtime.ps1's cl.exe fallback")

    def test_every_harness_the_gpu_suite_starts_is_built(self):
        """A harness run_gpu_tests.py or check.py starts by name, but the fallback did not build, fails its test
        with "could not start" (ps_ssharp: vk_sampler_test.exe, WinError 2)."""
        named = set()
        for path in (ROOT / "scripts" / "run_gpu_tests.py", ROOT / "runtime" / "test" / "shaders" / "check.py"):
            named |= set(re.findall(r'"(vk_[a-z0-9_]+)"', path.read_text(encoding="utf-8")))
        harnesses = {n for n in named if (ROOT / "runtime" / "test" / f"{n}.cpp").is_file()}
        self.assertIn("vk_sampler_test", harnesses)
        self.assertEqual(harnesses - self.programs, set(), "missing from build_runtime.ps1's cl.exe fallback")

    def test_the_launcher_compiles_every_source_cmake_gives_gta6rt(self):
        """BUILD-680: main.cpp calls rt::fs::configure, but the fallback left out shims/file_system.cpp (LNK2019).
        Every source of CMake's add_executable(gta6rt ...) must be in the fallback's launcher list ($files)."""
        cmake = (ROOT / "runtime" / "CMakeLists.txt").read_text(encoding="utf-8")
        block = re.search(r"add_executable\(gta6rt\s+(.*?)\)", cmake, re.S).group(1)
        wanted = {item.replace("/", "\\") for item in block.split() if item.endswith(".cpp")}
        files = re.search(r"\$files = (.*?) \| ForEach-Object", self.body).group(1)
        listed = set(re.findall(r'"([^"]+\.cpp)"', files))
        self.assertTrue(wanted)
        self.assertEqual(wanted - listed, set(), "in CMake's gta6rt sources but not in the cl.exe fallback")

    def test_every_test_source_it_compiles_exists(self):
        sources = re.findall(r'"\$src\\test\\([A-Za-z0-9_]+\.cpp)"', self.body)
        self.assertTrue(sources)
        for name in sources:
            with self.subTest(source=name):
                self.assertTrue((ROOT / "runtime" / "test" / name).is_file())

    def test_the_pipeline_checker_gets_the_stack_reserve_cmake_gives_it(self):
        """BUILD-1457: a driver compiles on vk_pipeline_check's threads, and AMD's Windows compiler overflows the
        1 MB PE default. CMake links the checker with /STACK:<LEONIDA_PIPELINE_CHECK_STACK>; the fallback has to pass
        the same number on the checker's own cl line (a /link option applies to that one program)."""
        cmake = (ROOT / "runtime" / "CMakeLists.txt").read_text(encoding="utf-8")
        reserve = re.search(r"set\(LEONIDA_PIPELINE_CHECK_STACK (\d+)\)", cmake).group(1)
        self.assertGreaterEqual(int(reserve), 8 * 1024 * 1024)
        self.assertIn("target_link_options(vk_pipeline_check PRIVATE /STACK:${LEONIDA_PIPELINE_CHECK_STACK})", cmake)
        lines = [line for line in self.body.splitlines() if "/Fe:vk_pipeline_check.exe" in line]
        self.assertEqual(len(lines), 1)
        self.assertRegex(lines[0], rf"/Fe:vk_pipeline_check\.exe /link (\S+ )*/STACK:{reserve}( |$)")

    def test_the_loader_gets_the_stack_reserve_cmake_gives_it(self):
        """BUILD-1457, the loader's side: a game thread with the default stack size gets the reserve in gta6rt.exe's
        header, and the driver compiles a pipeline's shaders on the thread that creates it. CMake links gta6rt with
        /STACK:<GTA6RT_STACK>; the fallback has to pass the same number, and it is the number run_entry gives the
        entry thread, so every game thread that asks for no size has the same room."""
        cmake = (ROOT / "runtime" / "CMakeLists.txt").read_text(encoding="utf-8")
        reserve = re.search(r"set\(GTA6RT_STACK (\d+)\)", cmake).group(1)
        self.assertGreaterEqual(int(reserve), 8 * 1024 * 1024)
        self.assertRegex(cmake, r"target_link_options\(gta6rt PRIVATE [^)]*/STACK:\$\{GTA6RT_STACK\}\)")
        lines = [line for line in self.body.splitlines() if "/Fe:gta6rt.exe" in line]
        self.assertEqual(len(lines), 1)
        self.assertRegex(lines[0], rf"/Fe:gta6rt\.exe /link (\S+ )*/STACK:{reserve}( |$)")
        shim = (ROOT / "runtime" / "shims" / "process_shim.h").read_text(encoding="utf-8")
        megabytes = re.search(r"run_entry\(void\* entry, std::size_t stack_bytes = (\d+)u << 20\);", shim).group(1)
        self.assertEqual(int(megabytes) << 20, int(reserve))

    def test_build_cmd_sets_utf8_codepage_for_ninja_showincludes(self):
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("chcp 65001 >nul", text)
        self.assertLess(text.index("chcp 65001 >nul"), text.index('call `"$vcvars`"'))

    def test_cmake_exe_is_invoked_without_call(self):
        """cmd's CALL doubles ^ in the rest of the line, so cmake.exe must not be launched with it (#3554).

        .cmd/.bat wrappers are launched with nested cmd /c (no CALL) so they return and carets stay intact.
        """
        text = SCRIPT.read_text(encoding="utf-8-sig")
        self.assertNotRegex(
            text, r'(?m)^call "\$cmake"',
            "call \"$cmake\" doubles ^ in the checkout path; use Get-BatchCommand so cmake.exe is quoted only",
        )
        start = text.index("function Get-BatchCommand")
        end = text.index("$root = Split-Path")
        helper = text[start:end]
        self.assertIn(r"\.(cmd|bat)$", helper)
        self.assertIn("cmd /d /s /c", helper)
        self.assertNotRegex(helper, r'return "call')
        self.assertIn('return "`"$Tool`" $CommandArgs"', helper)
        self.assertIn("$runConfigure = Get-BatchCommand $cmake", text)
        self.assertIn("$runBuild = Get-BatchCommand $cmake", text)
        start = text.index("$runConfigure = Get-BatchCommand $cmake")
        cmake_body = text[start:text.index("} else {", start)]
        self.assertIn("$runConfigure || exit /b 1", cmake_body)
        self.assertIn("$runBuild || exit /b 1", cmake_body)
        self.assertNotIn("call", cmake_body)

    @unittest.skipUnless(sys.platform == "win32", "requires Windows PowerShell")
    def test_path_fallback_prefers_vs_and_skips_msys_cmake(self):
        text = SCRIPT.read_text(encoding="utf-8")
        # Extract the helper functions and the two Find-BuildTool assignments directly from scripts/build_runtime.ps1.
        helpers_start = text.index("function Invoke-ToolProbe")
        helpers_end = text.index("$root = Split-Path")
        lookup_start = text.index("$cmakeDir = Join-Path")
        lookup_end = text.index("$cfg = if")
        snippet = text[helpers_start:helpers_end] + "\n" + text[lookup_start:lookup_end]

        def run_probe(vs_dir: Path, path_dirs: list[Path]) -> tuple[str, str]:
            ps = (
                "$ErrorActionPreference = 'Stop'\n"
                "$vs = $env:LEONIDA_VS_DIR\n"
                + snippet
                + "\nWrite-Output \"CMAKE=$cmake\"\nWrite-Output \"NINJA=$ninja\"\n"
            )
            env = dict(os.environ, PATH=";".join(str(p) for p in path_dirs), LEONIDA_VS_DIR=str(vs_dir))
            res = subprocess.run(
                ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", ps],
                env=env, capture_output=True, text=True, check=True, timeout=30,
            )
            found = {}
            for line in res.stdout.splitlines():
                if "=" in line:
                    k, v = line.split("=", 1)
                    found[k.strip()] = v.strip()
            return found.get("CMAKE", ""), found.get("NINJA", "")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vs_empty = root / "vs_empty"
            msys_dir = root / "msys_bin"
            native_cmake_dir = root / "native_cmake_bin"
            native_ninja_dir = root / "native_ninja_bin"
            for d in (vs_empty, msys_dir, native_cmake_dir, native_ninja_dir):
                d.mkdir()

            # MSYS/devkitPro-like cmake: lacks Visual Studio generators in --help.
            msys_cmake = msys_dir / "cmake.cmd"
            msys_cmake.write_text("@echo off\r\necho * Ninja\r\necho   Unix Makefiles\r\nexit /b 0\r\n", encoding="ascii")

            # Native Windows cmake: lists both Visual Studio and Ninja generators.
            native_cmake = native_cmake_dir / "cmake.cmd"
            native_cmake.write_text(
                "@echo off\r\necho   Visual Studio 18 2026\r\necho * Ninja\r\nexit /b 0\r\n",
                encoding="ascii",
            )
            native_ninja = native_ninja_dir / "ninja.cmd"
            native_ninja.write_text("@echo off\r\necho 1.12.1\r\nexit /b 0\r\n", encoding="ascii")

            # 1) No VS CMake tools, MSYS cmake comes first on PATH -> must skip MSYS and pick native PATH cmake + ninja.
            cmake, ninja = run_probe(vs_empty, [msys_dir, native_cmake_dir, native_ninja_dir])
            self.assertEqual(Path(cmake).resolve(), native_cmake.resolve())
            self.assertEqual(Path(ninja).resolve(), native_ninja.resolve())

            # 2) Only MSYS cmake on PATH -> must reject it so build_runtime.ps1 falls back to cl.exe.
            cmake, ninja = run_probe(vs_empty, [msys_dir])
            self.assertEqual(cmake, "")
            self.assertEqual(ninja, "")

            # 3) Native cmake on PATH without ninja -> ninja stays empty so CMake branch does not run.
            cmake, ninja = run_probe(vs_empty, [native_cmake_dir])
            self.assertEqual(Path(cmake).resolve(), native_cmake.resolve())
            self.assertEqual(ninja, "")

            # 4) VS bundled CMake + Ninja present -> must prefer VS over PATH.
            vs_full = root / "vs_full"
            vs_cmake = vs_full / "Common7" / "IDE" / "CommonExtensions" / "Microsoft" / "CMake" / "CMake" / "bin" / "cmake.exe"
            vs_ninja = vs_full / "Common7" / "IDE" / "CommonExtensions" / "Microsoft" / "CMake" / "Ninja" / "ninja.exe"
            vs_cmake.parent.mkdir(parents=True)
            vs_ninja.parent.mkdir(parents=True)
            vs_cmake.write_bytes(b"")
            vs_ninja.write_bytes(b"")
            cmake, ninja = run_probe(vs_full, [native_cmake_dir, native_ninja_dir])
            self.assertEqual(Path(cmake).resolve(), vs_cmake.resolve())
            self.assertEqual(Path(ninja).resolve(), vs_ninja.resolve())

    @unittest.skipUnless(sys.platform == "win32", "requires Windows PowerShell and cmd.exe")
    def test_cmake_batch_wrapper_preserves_caret_paths_and_status(self):
        """Get-BatchCommand's .cmd/.bat path must keep a single ^ in the wrapper, checkout, and Ninja paths,
        return after the wrapper, and pass the wrapper's exit status through to || exit /b 1."""
        text = SCRIPT.read_text(encoding="utf-8-sig")
        helper = text[text.index("function Get-BatchCommand"):text.index("$root = Split-Path")]
        cmake_block = text[text.index("if ($cmake -and $ninja) {"):text.index("} else {", text.index("if ($cmake -and $ninja) {"))]

        wrapper_src = (
            "@echo off\r\n"
            "setlocal EnableExtensions EnableDelayedExpansion\r\n"
            "set \"LOG=%~dp0cmake-log.txt\"\r\n"
            ">>\"%LOG%\" echo(---\r\n"
            "set \"CODE=0\"\r\n"
            "if defined CMAKE_FAKE_CONFIG_EXIT set \"CODE=%CMAKE_FAKE_CONFIG_EXIT%\"\r\n"
            ":loop\r\n"
            "if \"%~1\"==\"\" goto done\r\n"
            "set \"ARG=%~1\"\r\n"
            "if /I \"!ARG!\"==\"--build\" if defined CMAKE_FAKE_BUILD_EXIT set \"CODE=%CMAKE_FAKE_BUILD_EXIT%\"\r\n"
            ">>\"%LOG%\" echo(!ARG!\r\n"
            "shift\r\n"
            "goto loop\r\n"
            ":done\r\n"
            "exit /b !CODE!\r\n"
        )

        def emit_body(cmake: Path, ninja: Path, src: Path, out: Path, body_path: Path) -> str:
            driver = body_path.parent / "emit-body.ps1"
            driver.write_text(
                "$ErrorActionPreference = 'Stop'\n"
                + helper
                + "$cmake = $env:LEONIDA_CMAKE\n"
                + "$ninja = $env:LEONIDA_NINJA\n"
                + "$src = $env:LEONIDA_SRC\n"
                + "$out = $env:LEONIDA_OUT\n"
                + "$cfg = 'RelWithDebInfo'\n"
                + "$jobs = 2\n"
                + "$sdl = ''\n"
                + "$vk = ''\n"
                + cmake_block
                + "}\n[System.IO.File]::WriteAllText($env:LEONIDA_BODY, $body, [System.Text.UTF8Encoding]::new($false))\n",
                encoding="utf-8",
                newline="\n",
            )
            env = dict(
                os.environ,
                LEONIDA_CMAKE=str(cmake),
                LEONIDA_NINJA=str(ninja),
                LEONIDA_SRC=str(src),
                LEONIDA_OUT=str(out),
                LEONIDA_BODY=str(body_path),
            )
            subprocess.run(
                ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                 "-File", str(driver)],
                env=env, check=True, timeout=30, capture_output=True, text=True,
            )
            return body_path.read_text(encoding="utf-8")

        def run_body(directory: Path, body: str, env_extra: dict[str, str]) -> tuple[int, str]:
            batch = directory / "run-cmake.cmd"
            batch.write_bytes(("@echo off\r\n" + body).encode("utf-8"))
            log = directory / "caret^dir" / "cmake-log.txt"
            if log.is_file():
                log.unlink()
            env = dict(os.environ, **env_extra)
            res = subprocess.run(
                ["cmd.exe", "/d", "/c", "run-cmake.cmd"],
                cwd=directory, env=env, capture_output=True, timeout=30,
            )
            logged = log.read_text(encoding="utf-8") if log.is_file() else ""
            return res.returncode, logged

        def invocations(logged: str) -> list[list[str]]:
            blocks = []
            current = []
            for line in logged.splitlines():
                if line == "---":
                    if current:
                        blocks.append(current)
                    current = []
                else:
                    current.append(line)
            if current:
                blocks.append(current)
            return blocks

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            root = directory / "caret^dir"
            src = root / "runtime"
            out = src / "build"
            out.mkdir(parents=True)
            ninja = root / "tools^bin" / "ninja.exe"
            ninja.parent.mkdir()
            ninja.write_bytes(b"")

            for suffix in (".cmd", ".bat"):
                with self.subTest(wrapper=suffix):
                    cmake = root / f"cmake{suffix}"
                    cmake.write_bytes(wrapper_src.encode("ascii"))
                    body = emit_body(cmake, ninja, src, out, directory / "cmake-body.txt")
                    self.assertNotIn("call ", body.lower())
                    self.assertIn("cmd /d /s /c", body)
                    self.assertIn("caret^dir", body)
                    self.assertNotIn("caret^^dir", body)
                    self.assertIn(str(cmake), body)
                    self.assertIn(f"-DCMAKE_MAKE_PROGRAM={ninja}", body)

                    code, logged = run_body(directory, body, {
                        "CMAKE_FAKE_CONFIG_EXIT": "0",
                        "CMAKE_FAKE_BUILD_EXIT": "0",
                    })
                    self.assertEqual(code, 0, logged)
                    self.assertNotIn("caret^^dir", logged)
                    calls = invocations(logged)
                    self.assertEqual(len(calls), 2, logged)
                    self.assertIn("-S", calls[0])
                    self.assertIn(str(src), calls[0])
                    self.assertIn(str(out), calls[0])
                    self.assertIn(f"-DCMAKE_MAKE_PROGRAM={ninja}", calls[0])
                    self.assertIn("--build", calls[1])
                    self.assertIn(str(out), calls[1])
                    for arg in calls[0] + calls[1]:
                        self.assertIn("caret^dir", arg)
                        self.assertNotIn("caret^^dir", arg)

                    code, logged = run_body(directory, body, {
                        "CMAKE_FAKE_CONFIG_EXIT": "42",
                    })
                    self.assertEqual(code, 1, logged)
                    calls = invocations(logged)
                    self.assertEqual(len(calls), 1, logged)
                    self.assertIn("-S", calls[0])
                    self.assertFalse(any("--build" in c for c in calls))

                    code, logged = run_body(directory, body, {
                        "CMAKE_FAKE_CONFIG_EXIT": "0",
                        "CMAKE_FAKE_BUILD_EXIT": "42",
                    })
                    self.assertEqual(code, 1, logged)
                    calls = invocations(logged)
                    self.assertEqual(len(calls), 2, logged)
                    self.assertIn("--build", calls[1])



if __name__ == "__main__":
    unittest.main()
