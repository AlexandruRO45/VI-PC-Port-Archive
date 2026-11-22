"""Every compiled-in shader fixture macro (`LEONIDA_*_SPV`) in `runtime/test/` resolves via `vktest::shader_fixture_path`.

Tests built on Windows CI receive absolute build-tree paths in `LEONIDA_*_SPV` macros
(`D:/a/Leonida/Leonida/runtime/build/generated/<name>.spv`). When the `runtime-windows` artifact is run on
another machine or under Wine without a mapped build drive, opening those paths directly fails; resolving them
through `vktest::shader_fixture_path(...)` falls back to `generated/<filename>` inside the artifact directory.

Standard library only, no game data, no GPU.
"""
import os
import re
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RUNTIME_TEST_DIR = os.path.join(ROOT, "runtime", "test")
RUNTIME_CMAKELISTS = os.path.join(ROOT, "runtime", "CMakeLists.txt")
RUNTIME_WORKFLOW = os.path.join(ROOT, ".github", "workflows", "runtime.yml")

SPV_MACRO_RE = re.compile(r"\b(LEONIDA_[A-Z0-9_]*_SPV)\b")
INCLUDE_HELPER_RE = re.compile(r'#\s*include\s*"shader_fixture_path\.h"')
WRAPPED_MACRO_RE = re.compile(r"\b(?:vktest::)?shader_fixture_path\s*\(\s*(LEONIDA_[A-Z0-9_]*_SPV)\s*\)")
HELPER_VAR_CALL_RE = re.compile(r"\b(?:vktest::)?shader_fixture_path\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)")
DIRECT_OPEN_RE = re.compile(
    r"(?:\b(?:copy_file|fopen|CreateFile[AW])\s*\("
    r"|\b(?:ifstream|fstream)(?:\s+[A-Za-z_][A-Za-z0-9_]*)?\s*[({]"
    r"|\.\s*open\s*\()"
    r"[^;{}]*\b(LEONIDA_[A-Z0-9_]*_SPV)\b"
)
PREPROCESSOR_LINE_RE = re.compile(r"^\s*#")


def strip_comments_and_strings(source: str) -> str:
    """Replace C/C++ comments and string/char literals with spaces while preserving line numbers."""
    out = []
    i = 0
    n = len(source)
    while i < n:
        ch = source[i]
        nxt = source[i + 1] if i + 1 < n else ""
        if ch == "/" and nxt == "/":
            i += 2
            while i < n and source[i] != "\n":
                i += 1
        elif ch == "/" and nxt == "*":
            i += 2
            while i + 1 < n and not (source[i] == "*" and source[i + 1] == "/"):
                out.append("\n" if source[i] == "\n" else " ")
                i += 1
            i += 2
        elif ch in ('"', "'"):
            quote = ch
            out.append(" ")
            i += 1
            while i < n and source[i] != quote:
                if source[i] == "\\" and i + 1 < n:
                    out.append("\n" if source[i + 1] == "\n" else " ")
                    i += 2
                else:
                    out.append("\n" if source[i] == "\n" else " ")
                    i += 1
            if i < n:
                out.append(" ")
                i += 1
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def check_source(source: str) -> list[str]:
    """Return human-readable violations for `LEONIDA_*_SPV` usage in a C++ test source."""
    cleaned = strip_comments_and_strings(source)
    non_pp_lines = []
    continuation = False
    for lineno, line in enumerate(cleaned.splitlines(), 1):
        if continuation:
            continuation = line.rstrip().endswith("\\")
            continue
        if PREPROCESSOR_LINE_RE.match(line):
            continuation = line.rstrip().endswith("\\")
            continue
        non_pp_lines.append((lineno, line))

    code_only = "\n".join(line for _, line in non_pp_lines)
    macro_uses = []
    for lineno, line in non_pp_lines:
        for m in SPV_MACRO_RE.finditer(line):
            macro_uses.append((lineno, m.group(1), line, m.start(), m.end()))

    if not macro_uses:
        return []

    violations = []
    if not INCLUDE_HELPER_RE.search(source):
        violations.append('missing `#include "shader_fixture_path.h"`')

    # Distinguish files that wrap a helper parameter (`shader_fixture_path(path)`) from files that wrap
    # `LEONIDA_*_SPV` directly at each call site.
    helper_calls = [m.group(1) for m in HELPER_VAR_CALL_RE.finditer(code_only)]
    wraps_indirect_var = any(not SPV_MACRO_RE.fullmatch(arg) for arg in helper_calls)
    wraps_direct_macro = bool(WRAPPED_MACRO_RE.search(code_only))

    if not wraps_indirect_var and not wraps_direct_macro:
        for lineno, macro, *_ in macro_uses:
            violations.append(f"line {lineno}: {macro} is used without vktest::shader_fixture_path(...)")
        return violations

    # Even in a file with a helper function, never allow a direct file-open call on a bare `LEONIDA_*_SPV`.
    for lineno, line in non_pp_lines:
        masked = WRAPPED_MACRO_RE.sub(" ", line)
        for m in DIRECT_OPEN_RE.finditer(masked):
            violations.append(
                f"line {lineno}: direct file open on {m.group(1)} without vktest::shader_fixture_path(...)"
            )

    # In files that wrap macros directly at call sites (no indirect helper wrapping a variable), every single
    # non-preprocessor reference to `LEONIDA_*_SPV` must be wrapped.
    if wraps_direct_macro and not wraps_indirect_var:
        for lineno, line in non_pp_lines:
            masked = WRAPPED_MACRO_RE.sub(" ", line)
            for m in SPV_MACRO_RE.finditer(masked):
                violations.append(f"line {lineno}: bare {m.group(1)} without vktest::shader_fixture_path(...)")

    return violations


class ShaderFixturePathsTest(unittest.TestCase):
    def test_detector_catches_bare_and_partial_uses(self):
        good_direct = (
            '#include "shader_fixture_path.h"\n'
            '#ifndef LEONIDA_TEST_CS_SPV\n'
            '#error "LEONIDA_TEST_CS_SPV missing"\n'
            '#endif\n'
            'void f() {\n'
            '    std::filesystem::copy_file(vktest::shader_fixture_path(LEONIDA_TEST_CS_SPV), dst);\n'
            '    std::ifstream in(vktest::shader_fixture_path(LEONIDA_TEST_CS_SPV), std::ios::binary);\n'
            '}\n'
        )
        self.assertEqual(check_source(good_direct), [])

        good_helper = (
            '#include "shader_fixture_path.h"\n'
            'void load(const char* path) { std::ifstream f(vktest::shader_fixture_path(path)); }\n'
            'void f() { load(LEONIDA_TEST_CS_SAMPLE_SPV); }\n'
        )
        self.assertEqual(check_source(good_helper), [])

        missing_helper = (
            'void f() { std::filesystem::copy_file(LEONIDA_TEST_VS_SPV, dst); }\n'
        )
        self.assertTrue(check_source(missing_helper))

        partial_miss = (
            '#include "shader_fixture_path.h"\n'
            'void f() {\n'
            '    std::filesystem::copy_file(vktest::shader_fixture_path(LEONIDA_TEST_CS_SPV), dst);\n'
            '    std::ifstream in(LEONIDA_TEST_CS_SPV, std::ios::binary);\n'
            '}\n'
        )
        self.assertTrue(any("line 4:" in v for v in check_source(partial_miss)))

        unwrapped_read_helper = (
            'bool read_spv(const char* path) { std::ifstream f(path); return bool(f); }\n'
            'void f() { read_spv(LEONIDA_TEST_VS_TRI_SPV); }\n'
        )
        self.assertTrue(check_source(unwrapped_read_helper))

        helper_with_direct_bypass = (
            '#include "shader_fixture_path.h"\n'
            'void load(const char* path) { std::ifstream f(vktest::shader_fixture_path(path)); }\n'
            'void f() { load(LEONIDA_TEST_CS_SAMPLE_SPV); std::ifstream bad(LEONIDA_TEST_CS_SAMPLE_SPV); }\n'
        )
        self.assertTrue(any("direct file open" in v for v in check_source(helper_with_direct_bypass)))

    def test_every_runtime_test_uses_shader_fixture_path(self):
        checked = 0
        failures = []
        for name in sorted(os.listdir(RUNTIME_TEST_DIR)):
            if not name.endswith((".cpp", ".h")):
                continue
            if name in ("shader_fixture_path.h", "shader_fixture_path_test.cpp"):
                continue
            path = os.path.join(RUNTIME_TEST_DIR, name)
            with open(path, encoding="utf-8") as f:
                src = f.read()
            if SPV_MACRO_RE.search(src):
                checked += 1
            for issue in check_source(src):
                failures.append(f"runtime/test/{name}: {issue}")
        self.assertGreaterEqual(checked, 12, "expected at least 12 runtime tests using LEONIDA_*_SPV macros")
        self.assertEqual(failures, [], "bare shader fixture paths:\n" + "\n".join(failures))

    def test_workflow_packages_all_generated_spv_fixtures(self):
        with open(RUNTIME_CMAKELISTS, encoding="utf-8") as f:
            cmake = f.read()
        with open(RUNTIME_WORKFLOW, encoding="utf-8") as f:
            workflow = f.read()
        self.assertIn("runtime/build/generated/*.spv", workflow)
        # Every compiled-in LEONIDA_*_SPV macro in runtime/CMakeLists.txt must output directly under generated/
        # so `runtime/build/generated/*.spv` packages it and `shader_fixture_path` finds it as `generated/<filename>`.
        for m in re.finditer(r'\$\{CMAKE_CURRENT_BINARY_DIR\}/generated/([^\s")]+)\.spv', cmake):
            rel = m.group(1)
            self.assertNotIn("/", rel, f"generated/{rel}.spv is nested; runtime/build/generated/*.spv would miss it")


if __name__ == "__main__":
    unittest.main()
