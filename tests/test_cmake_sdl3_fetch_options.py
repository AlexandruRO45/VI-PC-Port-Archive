"""The pinned SDL3 fetch builds without SDL's libusb backend in a cross build, unless asked.

A clang-cl + xwin cross build on a Mac with Homebrew's libusb used to stop at configure: SDL3's find_package(LibUSB)
took the host's Mach-O libusb-1.0.a and its import-library check failed. Without SDL3 there is no GameInput shim, and
under Wine the retail boot stops at GameInputCreate.

Runs runtime/cmake/sdl3_fetch_options.cmake with ``cmake -P`` (no compiler, no download); skipped without cmake.
"""
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODULE = ROOT / "runtime" / "cmake" / "sdl3_fetch_options.cmake"
CMAKE_LISTS = ROOT / "runtime" / "CMakeLists.txt"
CMAKE = shutil.which("cmake")


def run_module(*defines):
    """Return (SDL_HIDAPI_LIBUSB, LEONIDA_SDL3_LIBUSB) after the macro ran with the given -D definitions."""
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "driver.cmake"
        script.write_text(f'include("{MODULE.as_posix()}")\nleonida_sdl3_fetch_options()\n'
                          'message(STATUS "sdl=[${SDL_HIDAPI_LIBUSB}] leonida=[${LEONIDA_SDL3_LIBUSB}]")\n',
                          encoding="utf-8")
        r = subprocess.run([CMAKE, *(f"-D{d}" for d in defines), "-P", str(script)],
                           capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        raise AssertionError(r.stderr)
    m = re.search(r"sdl=\[(\w*)\] leonida=\[(\w*)\]", r.stdout + r.stderr)
    if not m:
        raise AssertionError(r.stdout + r.stderr)
    return m.group(1), m.group(2)


@unittest.skipUnless(CMAKE, "cmake not on PATH")
class Sdl3FetchOptionsTest(unittest.TestCase):
    def test_cross_build_turns_libusb_off(self):
        self.assertEqual(run_module("CMAKE_CROSSCOMPILING=TRUE"), ("OFF", "OFF"))

    def test_cross_build_can_turn_it_back_on(self):
        self.assertEqual(run_module("CMAKE_CROSSCOMPILING=TRUE", "LEONIDA_SDL3_LIBUSB=ON"), ("ON", "ON"))

    def test_native_build_keeps_sdls_default(self):
        self.assertEqual(run_module(), ("ON", "ON"))

    def test_native_build_can_turn_it_off(self):
        self.assertEqual(run_module("LEONIDA_SDL3_LIBUSB=OFF"), ("OFF", "OFF"))

    def test_a_failed_configure_s_cached_on_does_not_win(self):
        # SDL3 caches its own SDL_HIDAPI_LIBUSB=ON before the check that failed; the next configure must not keep it.
        self.assertEqual(run_module("CMAKE_CROSSCOMPILING=TRUE", "SDL_HIDAPI_LIBUSB=ON"), ("OFF", "OFF"))


class CMakeListsUsesTheModuleTest(unittest.TestCase):
    def test_the_fetch_calls_the_module_before_sdl3_is_configured(self):
        text = CMAKE_LISTS.read_text(encoding="utf-8")
        start = text.index("FetchContent_Declare(SDL3")
        end = text.index("FetchContent_MakeAvailable(SDL3)", start)
        block = text[start:end]
        self.assertIn("cmake/sdl3_fetch_options.cmake", block)
        self.assertIn("leonida_sdl3_fetch_options()", block)


if __name__ == "__main__":
    unittest.main()
