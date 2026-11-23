"""Every source a runtime target compiles lives under runtime/ (or the build folder).

    python -I tests/test_runtime_sources_in_tree.py

CMake names the object of a source outside the target's source and binary folders after the source's absolute path
(``CMakeFiles\\gta6rt_xtaskqueue.dir\\C_\\...\\third_party\\...\\TaskQueue.cpp.obj``), and the Ninja generator passes
object names to ``lib.exe`` / ``link.exe`` unquoted through ``cmd.exe /C``.  A checkout in a folder whose name has
``&`` then fails to link: cmd splits the object name at the ``&`` (``LNK1181``).  Vendored code outside runtime/ is
compiled through a one-line wrapper under runtime/ instead (``runtime/shims/xtaskqueue/*.cpp``).

The check reads ``runtime/CMakeLists.txt`` only: each compiled source in ``add_library``, ``add_executable`` and
``target_sources``, with the variables the file sets expanded, must not leave runtime/.  No build tools are needed.
"""
import pathlib
import re
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
CMAKE = ROOT / "runtime" / "CMakeLists.txt"
COMPILED = (".c", ".cc", ".cpp", ".cxx", ".asm", ".rc")
BUILD_DIRS = ("${CMAKE_CURRENT_BINARY_DIR}", "${CMAKE_BINARY_DIR}", "${PROJECT_BINARY_DIR}")


def _strip_comments(text):
    return "\n".join(re.sub(r"(^|\s)#.*", "", line) for line in text.splitlines())


def outside_sources(text):
    """The compiled sources of the targets in ``text`` (a runtime/CMakeLists.txt) that resolve outside runtime/."""
    text = _strip_comments(text)
    values = {}
    for m in re.finditer(r"\bset\s*\(\s*(\w+)\s+([^\s)]+)", text):
        values.setdefault(m.group(1), m.group(2).strip('"'))

    def expand(word, depth=0):
        if depth > 10:
            return word
        out = re.sub(r"\$\{(\w+)\}", lambda v: values.get(v.group(1), v.group(0)) if v.group(1) != "CMAKE_CURRENT_SOURCE_DIR" else v.group(0), word)
        return out if out == word else expand(out, depth + 1)

    bad = []
    for m in re.finditer(r"\b(add_library|add_executable|target_sources)\s*\(([^)]*)\)", text):
        for word in m.group(2).split()[1:]:
            word = expand(word.strip('"'))
            if not word.lower().endswith(COMPILED) or word.startswith(BUILD_DIRS):
                continue
            rel = word.replace("${CMAKE_CURRENT_SOURCE_DIR}", ".")
            if "${" in rel and not rel.startswith("."):
                continue  # a variable this file does not set (a loop variable): its folder is checked where it is used
            rel = rel.replace("\\", "/")
            depth = 0
            for part in rel.split("/"):
                depth += -1 if part == ".." else (0 if part in (".", "") else 1)
                if depth < 0:
                    break
            if depth < 0 or re.match(r"[A-Za-z]:/|/", rel):
                bad.append(f"{m.group(1)}: {word}")
    return bad


class RuntimeSourcesInTree(unittest.TestCase):
    def test_runtime_cmakelists_compiles_only_sources_under_runtime(self):
        self.assertEqual(outside_sources(CMAKE.read_text(encoding="utf-8")), [])

    def test_a_vendored_source_outside_runtime_is_reported(self):
        old = (
            "set(XTASKQUEUE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/xtaskqueue)\n"
            "add_library(gta6rt_xtaskqueue STATIC\n"
            "  ${XTASKQUEUE_DIR}/Source/Task/TaskQueue.cpp  # the form before the wrappers\n"
            "  shims/xtaskqueue/AsyncLib.cpp\n"
            ")\n"
            "add_executable(t ../tools/x.cpp C:/abs/y.cpp)\n"
        )
        self.assertEqual(outside_sources(old), [
            "add_library: ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/xtaskqueue/Source/Task/TaskQueue.cpp",
            "add_executable: ../tools/x.cpp",
            "add_executable: C:/abs/y.cpp",
        ])

    def test_generated_and_in_tree_sources_pass(self):
        ok = (
            "set(GEN ${CMAKE_CURRENT_BINARY_DIR}/generated/stubs.cpp)\n"
            "add_executable(t src/a.cpp ${CMAKE_CURRENT_SOURCE_DIR}/shims/b.cpp ${GEN} test/${tool}.cpp\n"
            "  ../runtime/c.h)\n"
        )
        self.assertEqual(outside_sources(ok), [])

    def test_the_wrappers_include_the_vendored_files(self):
        for name in ("TaskQueue", "AsyncLib", "ThreadPool_win32", "WaitTimer_win32"):
            wrapper = (ROOT / "runtime" / "shims" / "xtaskqueue" / f"{name}.cpp").read_text(encoding="utf-8")
            target = f"../../../third_party/xtaskqueue/Source/Task/{name}.cpp"
            self.assertIn(f'#include "{target}"', wrapper)
            self.assertTrue((ROOT / "runtime" / "shims" / "xtaskqueue" / target).resolve().is_file(), target)


if __name__ == "__main__":
    unittest.main()
