"""MSVC must read every runtime source as UTF-8 (BUILD-1005).

Without ``/utf-8``, MSVC reads a source in the system code page and reports C4819 for any character the code page
can't hold (``→`` in ``runtime/src/vulkan_replay.h`` on DBCS code pages such as 936), or misreads a literal. The
flag used to be added per target, and the standalone Vulkan and replay test targets didn't have it. It is now set
once, before the first target, so a new target can't miss it. It is limited to C and C++: ml64 warns A4018 on an
unknown /utf-8.
"""
import os
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CMAKE = Path(os.environ.get("LEONIDA_RUNTIME_CMAKE", str(ROOT / "runtime" / "CMakeLists.txt")))

GLOBAL_UTF8 = re.compile(r"^if\(MSVC\)\s*\n"
                         r"\s*add_compile_options\(\$<\$<COMPILE_LANGUAGE:C,CXX>:/utf-8>\)\s*\n"
                         r"endif\(\)", re.M)
FIRST_TARGET = re.compile(r"^\s*add_(executable|library)\(", re.M)


def non_ascii_sources():
    found = []
    for folder in ("src", "shims", "test"):
        for path in sorted((ROOT / "runtime" / folder).rglob("*")):
            if path.suffix in (".cpp", ".h", ".hpp", ".c") and any(b > 0x7F for b in path.read_bytes()):
                found.append(path)
    return found


class RuntimeUtf8Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cmake = CMAKE.read_text(encoding="utf-8")

    def test_utf8_is_set_for_every_msvc_target(self):
        match = GLOBAL_UTF8.search(self.cmake)
        self.assertIsNotNone(match, "runtime/CMakeLists.txt must set /utf-8 for MSVC with add_compile_options")
        first = FIRST_TARGET.search(self.cmake)
        self.assertIsNotNone(first)
        # add_compile_options only reaches targets created after it.
        self.assertLess(match.start(), first.start(), "/utf-8 must come before the first add_executable/add_library")

    def test_the_flag_is_still_needed(self):
        # If no source has a non-ASCII byte any more, the flag (and this test) can go.
        self.assertTrue(non_ascii_sources(), "no runtime source has non-ASCII bytes")

    def test_non_ascii_sources_are_valid_utf8(self):
        for path in non_ascii_sources():
            with self.subTest(path=path.relative_to(ROOT).as_posix()):
                path.read_bytes().decode("utf-8")


if __name__ == "__main__":
    unittest.main()
