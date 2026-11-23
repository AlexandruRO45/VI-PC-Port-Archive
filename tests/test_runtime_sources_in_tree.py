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
    # Every value a variable is given anywhere in the file: each entry of a set() list and of list(APPEND ...).
    values = {}
    for m in re.finditer(r"\b(?:set\s*\(\s*(\w+)|list\s*\(\s*APPEND\s+(\w+))\s+([^)]*)\)", text):
        name = m.group(1) or m.group(2)
        words = [w.strip('"') for w in m.group(3).split() if w not in ("CACHE", "PARENT_SCOPE", "FORCE")]
        values.setdefault(name, []).extend(words)

    def expand(word, depth=0):
        """Every word ``word`` can stand for; a ``${VAR}`` with several values gives one result per value."""
        m = next((v for v in re.finditer(r"\$\{(\w+)\}", word) if v.group(1) in values), None)
        if depth > 10 or not m:
            return [word]
        out = []
        for value in values[m.group(1)]:
            out.extend(expand(word[:m.start()] + value + word[m.end():], depth + 1))
        return out

    bad = []
    for m in re.finditer(r"\b(add_library|add_executable|target_sources)\s*\(([^)]*)\)", text):
        for word in (w for raw in m.group(2).split()[1:] for w in expand(raw.strip('"'))):
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

    def test_every_entry_of_a_list_variable_is_checked(self):
        # Found by DoubleD17 in #6246: only the first value of a set() list was expanded.
        lists = (
            "set(PROBE_SOURCES probe.cpp ../outside/helper.cpp)\n"
            "add_executable(scope_probe ${PROBE_SOURCES})\n"
            "set(MORE\n  in/a.cpp\n  in/b.cpp\n)\n"
            "list(APPEND MORE ${CMAKE_CURRENT_SOURCE_DIR}/../vendored/c.cpp)\n"
            "add_library(more STATIC ${MORE})\n"
        )
        self.assertEqual(outside_sources(lists), [
            "add_executable: ../outside/helper.cpp",
            "add_library: ${CMAKE_CURRENT_SOURCE_DIR}/../vendored/c.cpp",
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
