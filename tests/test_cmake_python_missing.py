"""A runtime configure without Python 3 must say so loudly, or fail when asked to.

The D3D12 COM objects need Python at build time (tools/gen_d3d12_stubs.py generates their stubs from d3d12.h), so
without it they are left out, and with them the Vulkan D3D12 device in gta6rt and every test that links
gta6rt_d3d12_com. It used to be one STATUS line: a Windows PC whose only `python` was the Microsoft Store alias built
and ran ctest without any D3D12 layer test and reported "100% tests passed".

Runs runtime/cmake/python_missing.cmake with ``cmake -P`` (no compiler needed); skipped without cmake.
"""
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODULE = ROOT / "runtime" / "cmake" / "python_missing.cmake"
CMAKE_LISTS = ROOT / "runtime" / "CMakeLists.txt"
CMAKE = shutil.which("cmake")


def run_module(require):
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "driver.cmake"
        script.write_text(f'include("{MODULE.as_posix()}")\nleonida_python_missing()\n', encoding="utf-8")
        args = [CMAKE]
        if require is not None:
            args.append(f"-DLEONIDA_REQUIRE_VULKAN={require}")
        args += ["-P", str(script)]
        return subprocess.run(args, capture_output=True, text=True, timeout=60)


@unittest.skipUnless(CMAKE, "cmake not on PATH")
class PythonMissingMessageTest(unittest.TestCase):
    def test_default_warns_and_names_what_is_left_out(self):
        r = run_module(None)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("CMake Warning", r.stderr)
        text = " ".join(r.stderr.split())
        for part in ("D3D12 COM objects", "tools/gen_d3d12_stubs.py", "Vulkan D3D12 device in gta6rt",
                     "LEONIDA_D3D12=vulkan", "Microsoft Store", "-DPython3_EXECUTABLE=", "CMakeCache.txt",
                     "LEONIDA_REQUIRE_VULKAN=ON"):
            self.assertIn(part, text)

    def test_off_still_warns(self):
        r = run_module("OFF")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("CMake Warning", r.stderr)

    def test_require_fails_the_configure(self):
        r = run_module("ON")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("CMake Error", r.stderr)
        self.assertIn("Python 3 not found", r.stderr)


class CMakeListsUsesTheModuleTest(unittest.TestCase):
    def test_no_python_branch_calls_the_module(self):
        text = CMAKE_LISTS.read_text(encoding="utf-8")
        # The branch taken when the D3D12 COM objects are not built: elseif(TARGET gta6rt) after the
        # if(TARGET gta6rt AND Python3_FOUND AND _stubs_header_ok) that builds them.
        m = re.search(r"^  if\(TARGET gta6rt AND Python3_FOUND AND _stubs_header_ok\)\n.*?^  elseif\(TARGET gta6rt\)\n"
                      r"(.*?)^  endif\(\)", text, re.S | re.M)
        self.assertIsNotNone(m, "the D3D12 COM objects' if/elseif(TARGET gta6rt) block was not found")
        branch = m.group(1)
        self.assertIn("if(NOT Python3_FOUND)", branch)
        self.assertIn("leonida_python_missing()", branch)
        self.assertNotIn('message(STATUS "Python 3', branch)


if __name__ == "__main__":
    unittest.main()
