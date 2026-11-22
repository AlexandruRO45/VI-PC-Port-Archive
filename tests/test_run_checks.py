"""Unit tests for scripts/run_checks.py (standard library only, no game data)."""
from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import run_checks  # noqa: E402


class TestPythonTestDiscoveryAndPartition(unittest.TestCase):
    def test_discovery_excludes_semantic_and_includes_self(self):
        discovered = run_checks.discover_python_tests(REPO_ROOT)
        rels = [p.relative_to(REPO_ROOT).as_posix() for p in discovered]
        self.assertIn("tests/test_run_checks.py", rels)
        self.assertFalse(
            any(rel.startswith("tests/semantic/") for rel in rels),
            "tests/semantic/* must be excluded from top-level python test discovery",
        )
        self.assertEqual(rels, sorted(rels))

    def test_recursive_discovery_finds_nested_non_semantic_tests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tests" / "semantic").mkdir(parents=True)
            (root / "tests" / "sub").mkdir(parents=True)
            (root / "tests" / "test_top.py").write_text("if __name__ == '__main__': pass\n", encoding="utf-8")
            (root / "tests" / "sub" / "test_nested.py").write_text("if __name__ == '__main__': pass\n", encoding="utf-8")
            (root / "tests" / "semantic" / "test_sem.py").write_text("if __name__ == '__main__': pass\n", encoding="utf-8")
            (root / "tests" / "helper.py").write_text("", encoding="utf-8")

            discovered = run_checks.discover_python_tests(root)
            rels = [p.relative_to(root).as_posix() for p in discovered]
            self.assertEqual(rels, ["tests/sub/test_nested.py", "tests/test_top.py"])

    def test_exact_disjoint_partition_on_repo(self):
        discovered = run_checks.discover_python_tests(REPO_ROOT)
        stdlib_files, pytest_files = run_checks.partition_python_tests(REPO_ROOT)

        discovered_set = set(discovered)
        stdlib_set = set(stdlib_files)
        pytest_set = set(pytest_files)

        self.assertEqual(stdlib_set | pytest_set, discovered_set)
        self.assertEqual(stdlib_set & pytest_set, set())

        pytest_rels = {p.relative_to(REPO_ROOT).as_posix() for p in pytest_files}
        stdlib_rels = {p.relative_to(REPO_ROOT).as_posix() for p in stdlib_files}
        self.assertIn("tests/test_shader_index.py", pytest_rels)
        self.assertIn("tests/test_bc_feature.py", pytest_rels)
        self.assertIn("tests/test_run_checks.py", stdlib_rels)

        targets = run_checks.pytest_targets(REPO_ROOT)
        self.assertIn(run_checks.TEX_HEADERS_EXTRA_NODE, targets)

    def test_this_file_does_not_match_pytest_markers(self):
        self_path = Path(__file__).resolve()
        text = self_path.read_text(encoding="utf-8")
        self.assertIsNone(
            run_checks.PYTEST_FILE_MARKERS_RE.search(text),
            "tests/test_run_checks.py must not match PYTEST_FILE_MARKERS",
        )
        self.assertIn("__main__", text)
        self.assertFalse(run_checks.is_pytest_file(self_path))


class TestNoGameDataRegex(unittest.TestCase):
    def test_matches_forbidden_extensions_case_insensitively(self):
        forbidden = [
            "archive.rpf",
            "path/to/BIN.DLL",
            "shaders/out.spv",
            "runtime/gta6rt.exe",
            "x/a.ttpf",
            "x/b.XVD",
            "x/c.dxbc",
            "x/d.fssm",
            "x/e.spvasm",
            "x/f.dds",
            "x/g.bk2",
            "x/h.awc",
        ]
        allowed = [
            "scripts/run_checks.py",
            "CONTRIBUTING.md",
            "runtime/src/main.cpp",
            "tests/test_viewer_streaming.cjs",
            "docs/rpf_notes.txt",
        ]
        matched = run_checks.find_game_data_files(forbidden + allowed)
        self.assertEqual(matched, forbidden)


class TestCLI(unittest.TestCase):
    def test_list_flag_outputs_checks_and_exits_zero(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = run_checks.main(["--list"])
        self.assertEqual(rc, 0)
        out = buf.getvalue()
        for spec in run_checks.CHECKS:
            self.assertIn(spec.name, out)
        self.assertIn("default", out)
        self.assertIn("opt-in", out)

    def test_only_and_skip_selection(self):
        selected = run_checks.resolve_selected_checks(
            only=["no-game-data,compileall", "help"],
            skip=["compileall"],
        )
        self.assertEqual([c.name for c in selected], ["no-game-data", "help"])

    def test_default_selection_and_opt_ins(self):
        defaults = [c.name for c in run_checks.resolve_selected_checks()]
        self.assertNotIn("runtime", defaults)
        self.assertNotIn("lint", defaults)
        self.assertIn("no-game-data", defaults)
        self.assertIn("dotnet", defaults)

        with_opt_ins = [
            c.name
            for c in run_checks.resolve_selected_checks(
                include_runtime=True,
                include_lint=True,
            )
        ]
        self.assertIn("runtime", with_opt_ins)
        self.assertIn("lint", with_opt_ins)

    def test_unknown_check_name_rejected(self):
        with self.assertRaises(ValueError):
            run_checks.resolve_selected_checks(only=["does-not-exist"])
        with self.assertRaises(ValueError):
            run_checks.resolve_selected_checks(skip=["no-game-data,bogus"])

        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            with self.assertRaises(SystemExit) as ctx:
                run_checks.main(["--only", "not-a-real-check"])
        self.assertEqual(ctx.exception.code, 2)
        self.assertIn("not-a-real-check", err.getvalue())


class TestJactionlintProfile(unittest.TestCase):
    def test_reads_profile_from_workflow_and_falls_back_to_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Missing workflow file falls back to "default"
            self.assertEqual(run_checks.jactionlint_profile(root), "default")

            wf = root / ".github" / "workflows" / "jactionlint.yml"
            wf.parent.mkdir(parents=True)

            wf.write_text("steps:\n  - uses: jdx/jactionlint\n", encoding="utf-8")
            self.assertEqual(run_checks.jactionlint_profile(root), "default")

            wf.write_text(
                "steps:\n  - uses: jdx/jactionlint\n    with:\n      profile: correctness\n",
                encoding="utf-8",
            )
            self.assertEqual(run_checks.jactionlint_profile(root), "correctness")

            wf.write_text(
                "steps:\n  - uses: jdx/jactionlint\n    with:\n      profile: default\n",
                encoding="utf-8",
            )
            self.assertEqual(run_checks.jactionlint_profile(root), "default")

    def test_repo_workflow_profile_is_detected(self):
        self.assertIn(run_checks.jactionlint_profile(REPO_ROOT), {"correctness", "default"})


class TestRuntimePresetSelection(unittest.TestCase):
    def test_darwin_returns_none_even_when_presets_exist(self):
        presets = {
            "configurePresets": [
                {"name": "windows-ninja"},
                {"name": "linux-ninja"},
            ]
        }
        self.assertIsNone(run_checks.select_cmake_preset(presets, platform="darwin"))
        self.assertEqual(run_checks.select_cmake_preset(presets, platform="win32"), "windows-ninja")
        self.assertEqual(run_checks.select_cmake_preset(presets, platform="linux"), "linux-ninja")

    def test_returns_none_when_preset_absent_from_file(self):
        only_linux = {"configurePresets": [{"name": "linux-ninja"}]}
        self.assertIsNone(run_checks.select_cmake_preset(only_linux, platform="win32"))
        self.assertEqual(run_checks.select_cmake_preset(only_linux, platform="linux"), "linux-ninja")

        empty = {"configurePresets": []}
        self.assertIsNone(run_checks.select_cmake_preset(empty, platform="win32"))
        self.assertIsNone(run_checks.select_cmake_preset(empty, platform="linux"))
        self.assertIsNone(run_checks.select_cmake_preset({}, platform="linux"))


if __name__ == "__main__":
    unittest.main()
