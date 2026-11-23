"""Tests for scripts/run_wine_tests.py: path mapping, per-test statuses and the job's exit code.

    python -I tests/test_run_wine_tests.py

No Wine and no game data: `wine` is the Python interpreter and every "build artifact" is a tiny script that exits
with a chosen code, so the runner's own logic is what is under test.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_wine_tests as rwt  # noqa: E402

WINDOWS_ROOT = "D:/win/root"


def ctest_entry(name, command, properties=None):
    entry = {"name": name}
    if command is not None:
        entry["command"] = command
    if properties:
        entry["properties"] = [{"name": k, "value": v} for k, v in properties.items()]
    return entry


class PathMappingTest(unittest.TestCase):
    def test_paths_under_the_windows_checkout_are_mapped(self):
        self.assertEqual(rwt.map_path("D:/win/root", WINDOWS_ROOT, "/repo"), "/repo")
        self.assertEqual(rwt.map_path("D:\\win\\root\\runtime\\build", WINDOWS_ROOT, "/repo"),
                         os.path.join("/repo", "runtime", "build"))
        self.assertEqual(rwt.map_path("d:/WIN/root/a/b.exe", WINDOWS_ROOT, "/repo"), os.path.join("/repo", "a", "b.exe"))
        self.assertEqual(rwt.map_path("D:/win/root/", WINDOWS_ROOT, "/repo"), "/repo")

    def test_paths_outside_the_checkout_are_not_mapped(self):
        for path in ("D:/win/root2/a.exe", "C:/Windows/system32/cmd.exe", "/usr/bin/cmake", ""):
            with self.subTest(path=path):
                self.assertIsNone(rwt.map_path(path, WINDOWS_ROOT, "/repo"))

    def test_parent_segments_cannot_escape_windows_root(self):
        self.assertIsNone(rwt.map_path("D:/win/root/../outside.exe", WINDOWS_ROOT, "/repo"))
        self.assertEqual(rwt.map_path("D:/win/root/inside/../runner.exe", WINDOWS_ROOT, "/repo"),
                         os.path.join("/repo", "runner.exe"))

    def test_dot_and_repeated_separators_normalize_before_mapping(self):
        self.assertEqual(rwt.map_path("D:/win/root/./runner.exe", WINDOWS_ROOT, "/repo"),
                         os.path.join("/repo", "runner.exe"))
        self.assertEqual(rwt.map_path("D:/win/root//sub//runner.exe", WINDOWS_ROOT, "/repo"),
                         os.path.join("/repo", "sub", "runner.exe"))
        self.assertIsNone(rwt.map_path("D:/win/root/../../outside.exe", WINDOWS_ROOT, "/repo"))

    def test_wine_path_uses_the_z_drive(self):
        self.assertEqual(rwt.wine_path("/tmp/x y"), "Z:" + os.path.abspath("/tmp/x y").replace("/", "\\"))

    def test_properties_are_flattened(self):
        self.assertEqual(rwt.properties(ctest_entry("t", ["x"], {"SKIP_RETURN_CODE": "77", "WILL_FAIL": "1"})),
                         {"SKIP_RETURN_CODE": "77", "WILL_FAIL": "1"})
        self.assertEqual(rwt.properties({"name": "t"}), {})

    def test_ctest_booleans_are_not_python_truthiness(self):
        for value in ("1", "ON", "true", "yes", " 1 "):
            with self.subTest(value=value):
                self.assertTrue(rwt.truthy(value))
        for value in ("0", "", None, "false", "no"):
            with self.subTest(value=value):
                self.assertFalse(rwt.truthy(value))


class RunTestTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.artifact = self.root / "artifact"
        self.artifact.mkdir()
        self.args = argparse.Namespace(windows_root=WINDOWS_ROOT, root=str(self.root), timeout=10.0)
        self.wine = [sys.executable]

    def tearDown(self):
        self._tmp.cleanup()

    def program(self, name, body):
        path = self.artifact / name
        path.write_text(body, encoding="utf-8")
        return str(path)

    def exiting(self, code, name=None):
        return self.program(name or f"exit{code}.py", f"import sys\nsys.exit({code})\n")

    def command_for(self, exe, *args):
        """The same program as ctest would list it: a path in the Windows checkout."""
        parts = Path(exe).relative_to(self.root).parts
        return [WINDOWS_ROOT + "/" + "/".join(parts), *args]

    def run_entry(self, entry):
        return rwt.run_test(entry, self.args, self.wine)

    def test_a_passing_test(self):
        status, detail = self.run_entry(ctest_entry("ok", self.command_for(self.exiting(0))))
        self.assertEqual(status, "pass")
        self.assertIn("exit 0", detail)

    def test_a_failing_test_reports_the_exit_code(self):
        status, detail = self.run_entry(ctest_entry("bad", self.command_for(self.exiting(3))))
        self.assertEqual(status, "fail")
        self.assertIn("exit 3", detail)

    def test_a_disabled_test_is_skipped_without_launching(self) -> None:
        built = self.command_for(self.exiting(0))
        missing = self.command_for(self.artifact / "missing.exe")
        for command in (built, missing, None):
            with self.subTest(command=command), \
                    mock.patch.object(rwt.subprocess, "run", wraps=rwt.subprocess.run) as launch:
                status, detail = self.run_entry(ctest_entry("disabled", command, {"DISABLED": True}))
                launch.assert_not_called()
                self.assertEqual(status, "skip", detail)
                self.assertIn("disabled by CTest", detail)

    def test_a_false_disabled_property_still_runs_the_test(self) -> None:
        command = self.command_for(self.exiting(0))
        status, detail = self.run_entry(ctest_entry("enabled", command, {"DISABLED": "0"}))
        self.assertEqual(status, "pass", detail)

    def test_a_test_without_a_command_fails(self):
        status, detail = self.run_entry(ctest_entry("unbuilt", None))
        self.assertEqual(status, "fail")
        self.assertIn("no command", detail)

    def test_a_test_that_runs_a_runner_program_is_skipped(self):
        status, detail = self.run_entry(ctest_entry("python-test", ["/usr/bin/cmake", "-P", "x.cmake"]))
        self.assertEqual(status, "skip")
        self.assertIn("not a program from the build", detail)

    def test_a_missing_executable_fails(self):
        status, detail = self.run_entry(ctest_entry("gone", [WINDOWS_ROOT + "/artifact/gone.py"]))
        self.assertEqual(status, "fail")
        self.assertIn("not in the artifact", detail)

    def test_will_fail_inverts_the_exit_code(self):
        exe = self.exiting(2)
        status, _ = self.run_entry(ctest_entry("expected", self.command_for(exe), {"WILL_FAIL": "1"}))
        self.assertEqual(status, "pass")

    def test_will_fail_zero_does_not_invert_the_exit_code(self):
        # A JSON property value is a string; "0" must stay false.
        status, detail = self.run_entry(ctest_entry("normal", self.command_for(self.exiting(0)), {"WILL_FAIL": "0"}))
        self.assertEqual(status, "pass", detail)

    def test_the_skip_return_code_is_a_skip(self):
        status, detail = self.run_entry(
            ctest_entry("skipped", self.command_for(self.exiting(77)), {"SKIP_RETURN_CODE": "77"}))
        self.assertEqual(status, "skip")
        self.assertIn("exit 77", detail)

    def test_ctest_environment_is_applied_only_to_its_child(self) -> None:
        exe = self.program("show_environment.py",
                           "import os\n"
                           "for key in ('LEONIDA_HOST_ADAPTER', 'LEONIDA_WINE_TEST_INHERITED', "
                           "'LEONIDA_WINE_TEST_VALUE', 'LEONIDA_WINE_TEST_EMPTY'):\n"
                           "    print(repr(os.environ.get(key)))\n")
        command = self.command_for(exe)
        env = {"LEONIDA_HOST_ADAPTER": "inherited", "LEONIDA_WINE_TEST_INHERITED": "keep"}
        props = {"ENVIRONMENT": ["LEONIDA_HOST_ADAPTER=warp", "LEONIDA_WINE_TEST_VALUE=path=a;b",
                                 "LEONIDA_WINE_TEST_EMPTY="]}
        with mock.patch.dict(os.environ, env):
            os.environ.pop("LEONIDA_WINE_TEST_VALUE", None)
            os.environ.pop("LEONIDA_WINE_TEST_EMPTY", None)
            status, detail = self.run_entry(ctest_entry("with_environment", command, props))
            self.assertEqual(status, "pass", detail)
            self.assertEqual(detail.splitlines()[1:], ["'warp'", "'keep'", "'path=a;b'", "''"])
            status, detail = self.run_entry(ctest_entry("without_environment", command))
            self.assertEqual(status, "pass", detail)
            self.assertEqual(detail.splitlines()[1:], ["'inherited'", "'keep'", "None", "None"])

    def test_malformed_ctest_environment_fails_before_launch(self) -> None:
        command = self.command_for(self.exiting(0))
        props = {"ENVIRONMENT": ["missing_equals"]}
        with mock.patch.object(rwt.subprocess, "run", wraps=rwt.subprocess.run) as launch:
            status, detail = self.run_entry(ctest_entry("bad_environment", command, props))
        launch.assert_not_called()
        self.assertEqual(status, "fail", detail)
        self.assertIn("invalid CTest ENVIRONMENT", detail)

    def test_ctest_output_expressions_determine_the_result(self):
        usage = {"PASS_REGULAR_EXPRESSION": ["alternative", "usage: gta6rt"]}
        forbidden = {"FAIL_REGULAR_EXPRESSION": ["running ", r"module\(s\) mapped"]}
        for code, output, props, expected in (
            (2, "usage: gta6rt", usage | forbidden, "pass"),
            (0, "unexpected output", usage, "fail"),
            (2, "usage: gta6rt\n1 module(s) mapped", usage | forbidden, "fail"),
            (0, "running tiny_game.exe", forbidden, "fail"),
            (2, "unexpected output", forbidden, "fail"),
            (77, "unexpected output", usage | {"SKIP_RETURN_CODE": 77}, "skip"),
            (0, "running tiny_game.exe", forbidden | {"WILL_FAIL": True}, "pass"),
        ):
            with self.subTest(code=code, output=output, props=props):
                exe = self.program("regex.py", f"import sys\nprint({output!r}, file=sys.stderr)\nsys.exit({code})\n")
                status, detail = self.run_entry(ctest_entry("regex", self.command_for(exe), props))
                self.assertEqual(status, expected, detail)

    def test_a_scalar_output_expression_is_matched_as_one_pattern(self):
        required = {"PASS_REGULAR_EXPRESSION": "usage: gta6rt"}
        forbidden = {"FAIL_REGULAR_EXPRESSION": r"module\(s\) mapped"}
        for code, output, props, expected in (
            (3, "unexpected output", required, "fail"),
            (2, "usage: gta6rt", required, "pass"),
            (0, "unexpected output", forbidden, "pass"),
            (0, "1 module(s) mapped", forbidden, "fail"),
        ):
            with self.subTest(output=output, props=props):
                exe = self.program("regex.py", f"import sys\nprint({output!r})\nsys.exit({code})\n")
                status, detail = self.run_entry(ctest_entry("regex", self.command_for(exe), props))
                self.assertEqual(status, expected, detail)

    def test_skip_output_overrides_a_normal_exit_status(self):
        skip_pattern = "SKIP: missing adapter"
        exe = self.program("skip.py", f"print({skip_pattern!r})\nraise SystemExit(3)\n")
        for expression in (skip_pattern, ["different reason", skip_pattern]):
            with self.subTest(expression=expression):
                entry = ctest_entry("skip", self.command_for(exe), {"SKIP_REGULAR_EXPRESSION": expression})
                status, detail = rwt.run_test(entry, self.args, [sys.executable, "-I"])
                self.assertEqual(status, "skip", detail)
                self.assertIn("SKIP_REGULAR_EXPRESSION", detail)

        entry = ctest_entry("no_match", self.command_for(exe), {"SKIP_REGULAR_EXPRESSION": ["different reason"]})
        status, detail = rwt.run_test(entry, self.args, [sys.executable, "-I"])
        self.assertEqual(status, "fail", detail)

    def test_a_signal_termination_fails_even_with_success_overrides(self):
        # The mocked negative returncode keeps the test portable across Windows and POSIX.
        exe = self.exiting(0)
        fake = mock.Mock(returncode=-11, stdout=b"usage: gta6rt")
        for props in ({"WILL_FAIL": True}, {"PASS_REGULAR_EXPRESSION": ["usage: gta6rt"]},
                      {"SKIP_REGULAR_EXPRESSION": ["usage: gta6rt"]}):
            with self.subTest(props=props), mock.patch.object(rwt.subprocess, "run", return_value=fake):
                entry = ctest_entry("segv", self.command_for(exe), props | {"SKIP_RETURN_CODE": 77})
                status, detail = self.run_entry(entry)
                self.assertEqual(status, "fail")
                self.assertIn("signal 11", detail)

    def test_a_timeout_is_reported_without_hanging(self):
        slow = self.program("slow.py", "import time\ntime.sleep(5)\n")
        self.args.timeout = 0.2
        status, detail = self.run_entry(ctest_entry("slow", self.command_for(slow)))
        self.assertEqual(status, "fail")
        self.assertIn("timed out after 0.2 s", detail)

    def test_a_short_ctest_timeout_still_fails_a_will_fail_test(self):
        slow = self.program("slow_failure.py", "import sys, time\ntime.sleep(1)\nsys.exit(1)\n")
        entry = ctest_entry("slow_failure", self.command_for(slow), {"TIMEOUT": 0.2, "WILL_FAIL": True})
        status, detail = self.run_entry(entry)
        self.assertEqual(status, "fail", detail)
        self.assertIn("timed out after 0.2 s", detail)

    def test_ctest_timeout_cannot_remove_or_extend_the_runner_limit(self):
        exe = self.exiting(0)
        for timeout in (0, "20"):
            with self.subTest(timeout=timeout):
                entry = ctest_entry("bounded", self.command_for(exe), {"TIMEOUT": timeout})
                result = rwt.subprocess.CompletedProcess([], 0, stdout=b"")
                with mock.patch.object(rwt.subprocess, "run", return_value=result) as run:
                    status, detail = self.run_entry(entry)
                self.assertEqual(status, "pass", detail)
                self.assertEqual(run.call_args.kwargs["timeout"], self.args.timeout)

    def test_an_unstartable_wine_fails_the_test_instead_of_aborting_the_run(self):
        status, detail = rwt.run_test(ctest_entry("ok", self.command_for(self.exiting(0))), self.args,
                                      ["definitely-not-a-real-wine-xyz"])
        self.assertEqual(status, "fail")
        self.assertIn("could not start", detail)

    def test_arguments_and_working_directory_are_mapped(self):
        report = self.root / "report.json"
        script = self.program("report.py", (
            "import json, os, sys\n"
            "json.dump({'argv': sys.argv[1:], 'cwd': os.getcwd()}, open(os.environ['REPORT_PATH'], 'w'))\n"))
        (self.artifact / "sub").mkdir()
        entry = ctest_entry("mapped", self.command_for(script, WINDOWS_ROOT + "/artifact/data.bin", "--flag"),
                           {"WORKING_DIRECTORY": WINDOWS_ROOT + "/artifact/sub"})
        with mock.patch.dict(os.environ, {"REPORT_PATH": str(report)}):
            status, detail = self.run_entry(entry)
        self.assertEqual(status, "pass", detail)
        seen = json.loads(report.read_text(encoding="utf-8"))
        self.assertEqual(len(seen["argv"]), 2)
        self.assertTrue(seen["argv"][0].startswith("Z:"), seen["argv"][0])
        self.assertTrue(seen["argv"][0].endswith("data.bin"), seen["argv"][0])
        self.assertEqual(seen["argv"][1], "--flag")
        self.assertEqual(os.path.normcase(os.path.realpath(seen["cwd"])),
                         os.path.normcase(os.path.realpath(self.artifact / "sub")))


@unittest.skipIf(os.name == "nt", "the runner and its dosdevices links are for Linux and macOS")
class DriveMapTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        base = Path(self._tmp.name)
        self.root = base / "checkout"
        (self.root / "runtime" / "build" / "generated").mkdir(parents=True)
        (self.root / "runtime" / "build" / "generated" / "cs.spv").write_text("spirv", encoding="utf-8")
        self.prefix = base / "prefix"
        (self.prefix / "dosdevices").mkdir(parents=True)
        self.link = self.prefix / "dosdevices" / "d:"

    def tearDown(self):
        self._tmp.cleanup()

    def test_the_drive_leads_to_the_checkout_and_is_removed_afterwards(self):
        with rwt.DriveMap("D:/a/Leonida/Leonida", str(self.root), str(self.prefix)) as drive:
            seen = self.link / "a" / "Leonida" / "Leonida" / "runtime" / "build" / "generated" / "cs.spv"
            self.assertEqual(seen.read_text(encoding="utf-8"), "spirv")
            self.assertIn("drive d: maps", drive.note)
            scratch = drive.scratch
        self.assertFalse(os.path.lexists(self.link))
        self.assertFalse(os.path.exists(scratch))
        self.assertTrue((self.root / "runtime" / "build" / "generated" / "cs.spv").is_file())   # links only

    def test_backslashes_and_a_lower_case_drive_are_accepted(self):
        with rwt.DriveMap("d:\\work\\leonida\\", str(self.root), str(self.prefix)):
            self.assertTrue((self.link / "work" / "leonida" / "runtime" / "build").is_dir())

    def test_a_drive_the_prefix_already_has_is_left_alone(self):
        other = self.prefix / "elsewhere"
        other.mkdir()
        os.symlink(other, self.link)
        with rwt.DriveMap("D:/a/Leonida/Leonida", str(self.root), str(self.prefix)) as drive:
            self.assertIn("already set", drive.note)
        self.assertEqual(os.readlink(self.link), str(other))

    def test_no_prefix_or_no_drive_letter_maps_nothing(self):
        with rwt.DriveMap("D:/a/b", str(self.root), str(self.prefix / "absent")) as drive:
            self.assertIn("no Wine prefix", drive.note)
        with rwt.DriveMap("/home/runner/work", str(self.root), str(self.prefix)) as drive:
            self.assertIn("no drive letter", drive.note)
        self.assertFalse(os.path.lexists(self.link))

    def test_main_maps_the_drive_while_the_tests_run(self):
        # The "Wine" here is the Python interpreter, so the program reads the drive through its dosdevices link, which
        # is where Wine resolves D:/win/root/... too.
        probe = self.root / "probe.py"
        target = self.link / "win" / "root" / "runtime" / "build" / "generated" / "cs.spv"
        probe.write_text(f"import sys\nsys.exit(0 if open({str(target)!r}).read() == 'spirv' else 1)\n", encoding="utf-8")
        listing = self.root / "ctest.json"
        listing.write_text(json.dumps({"tests": [ctest_entry("compiled_in_path", [WINDOWS_ROOT + "/probe.py"])]}),
                           encoding="utf-8")
        argv = ["run_wine_tests.py", "--tests", str(listing), "--windows-root", WINDOWS_ROOT,
                "--root", str(self.root), "--wine", sys.executable.replace(os.sep, "/")]
        out = io.StringIO()
        with mock.patch.dict(os.environ, {"WINEPREFIX": str(self.prefix)}), mock.patch.object(sys, "argv", argv), \
                contextlib.redirect_stdout(out):
            code = rwt.main()
        self.assertEqual(code, 0, out.getvalue())
        self.assertIn("[PASS] compiled_in_path", out.getvalue())
        self.assertFalse(os.path.lexists(self.link))


class WineDebuggerTest(unittest.TestCase):
    """The crash debugger is switched off for the run and put back after it, through a stand-in for `wine reg`."""

    DEFAULT = "winedbg --auto %ld %ld"
    FAKE_WINE = (
        "import json, sys\n"
        "state_path, args = sys.argv[1], sys.argv[3:]\n"
        "state = json.load(open(state_path))\n"
        "state['calls'].append(args)\n"
        "code = 0\n"
        "if args[0] == 'query':\n"
        "    if state.get('query_error'):\n"
        "        print(state['query_error'])\n"
        "        code = 1\n"
        "    elif state['debugger'] is None:\n"
        "        print('reg: Error: The system was unable to find the specified registry key or value.')\n"
        "        code = 1\n"
        "    else:\n"
        "        print('HKEY_LOCAL_MACHINE\\\\Software\\\\AeDebug')\n"
        "        print('    Debugger    ' + state['type'] + '    ' + state['debugger'])\n"
        "elif args[0] == 'add':\n"
        "    adds = sum(1 for c in state['calls'] if c[0] == 'add')\n"
        "    if adds > state.get('adds_that_work', 99):\n"
        "        code = 1\n"
        "    else:\n"
        "        state['debugger'], state['type'] = args[args.index('/d') + 1], 'REG_SZ'\n"
        "elif args[0] == 'delete':\n"
        "    state['debugger'] = None\n"
        "json.dump(state, open(state_path, 'w'))\n"
        "sys.exit(code)\n"
    )

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        base = Path(self._tmp.name)
        self.prefix = base / "prefix"
        (self.prefix / "dosdevices").mkdir(parents=True)
        self.state = base / "state.json"
        script = base / "fake_wine.py"
        script.write_text(self.FAKE_WINE, encoding="utf-8")
        self.wine = [sys.executable, str(script), str(self.state)]

    def tearDown(self):
        self._tmp.cleanup()

    def set_debugger(self, value, kind="REG_SZ", **extra):
        self.state.write_text(json.dumps({"debugger": value, "type": kind, "calls": [], **extra}), encoding="utf-8")

    def read(self):
        return json.loads(self.state.read_text(encoding="utf-8"))

    def enter(self):
        return rwt.WineDebugger(self.wine, str(self.prefix))

    def test_the_debugger_is_off_during_the_run_and_restored_after(self):
        self.set_debugger(self.DEFAULT)
        with self.enter() as dbg:
            self.assertEqual(self.read()["debugger"], "false")
            self.assertIn("crash debugger is off", dbg.note)
        self.assertEqual(self.read()["debugger"], self.DEFAULT)

    def test_a_value_the_query_proves_absent_is_removed_again(self):
        self.set_debugger(None)
        with self.enter():
            self.assertEqual(self.read()["debugger"], "false")
        self.assertIsNone(self.read()["debugger"])

    def test_any_other_value_is_left_alone(self):
        cases = {
            "another command": ("gdb --pid %ld", "REG_SZ"),
            "a wrapper around winedbg": ("/opt/my-winedbg-wrapper --auto %ld %ld", "REG_SZ"),
            "an expandable string": ("%WINDIR%\\winedbg --auto %ld %ld", "REG_EXPAND_SZ"),
            "the default text in the wrong type": (self.DEFAULT, "REG_EXPAND_SZ"),
        }
        for label, (value, kind) in cases.items():
            with self.subTest(label):
                self.set_debugger(value, kind)
                with self.enter() as dbg:
                    self.assertEqual(self.read()["debugger"], value)
                    self.assertIn("left as it is", dbg.note)
                self.assertEqual(self.read()["debugger"], value)
                self.assertEqual([c[0] for c in self.read()["calls"]], ["query"])

    def test_the_way_back_is_printed_before_the_run_for_both_prior_states(self):
        # A run killed with SIGKILL cannot restore the value: the note printed at the start must undo exactly what was done.
        self.set_debugger(self.DEFAULT)
        with self.enter() as dbg:
            self.assertIn('reg add', dbg.note)
            self.assertIn('/d "winedbg --auto %ld %ld"', dbg.note)
        self.set_debugger(None)
        with self.enter() as dbg:
            self.assertIn('reg delete', dbg.note)
            self.assertNotIn('reg add', dbg.note)

    def test_an_empty_value_is_left_alone(self):
        self.set_debugger("", "REG_SZ")
        with self.enter():
            self.assertEqual(self.read()["debugger"], "")
        self.assertEqual(self.read()["debugger"], "")

    def test_a_query_that_failed_for_another_reason_changes_nothing(self):
        self.set_debugger(self.DEFAULT, query_error="reg: Error: Access is denied.")
        with self.enter():
            self.assertEqual(self.read()["debugger"], self.DEFAULT)
        self.assertEqual([c[0] for c in self.read()["calls"]], ["query"])

    def test_keep_and_a_missing_prefix_change_nothing(self):
        self.set_debugger(self.DEFAULT)
        with rwt.WineDebugger(self.wine, str(self.prefix), keep=True):
            pass
        with rwt.WineDebugger(self.wine, str(self.prefix / "missing")):
            pass
        self.assertEqual(self.read()["calls"], [])

    def test_an_unstartable_wine_does_not_stop_the_run(self):
        with rwt.WineDebugger(["/nonexistent/wine"], str(self.prefix)) as dbg:
            self.assertIn("left as it is", dbg.note)

    def test_a_failed_restore_is_tried_again_and_then_reported(self):
        err = io.StringIO()
        self.set_debugger(self.DEFAULT, adds_that_work=1)         # only the "false" add works: both restores fail
        with contextlib.redirect_stderr(err):
            with self.enter():
                pass
        adds = [c for c in self.read()["calls"] if c[0] == "add"]
        self.assertEqual(len(adds), 3)                            # false, restore, one retry
        self.assertEqual(self.read()["debugger"], "false")
        self.assertIn(str(self.prefix), err.getvalue())
        self.assertIn("reg add", err.getvalue())
        self.assertIn(self.DEFAULT, err.getvalue())

    def test_a_restore_that_works_on_the_second_try_is_quiet(self):
        self.set_debugger(self.DEFAULT)
        err = io.StringIO()
        real = self.enter()
        calls = []
        original = real._reg

        def flaky(*argv):
            calls.append(argv[0])
            if argv[0] == "add" and calls.count("add") == 2:        # the first restore fails
                return None
            return original(*argv)

        real._reg = flaky
        with contextlib.redirect_stderr(err):
            with real:
                pass
        self.assertEqual(self.read()["debugger"], self.DEFAULT)
        self.assertEqual(err.getvalue(), "")


class AbortedTestTest(unittest.TestCase):
    """A process Wine aborts stays a failure, however CTest would read its exit code or output."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "artifact").mkdir()
        self.args = argparse.Namespace(windows_root=WINDOWS_ROOT, root=str(self.root), timeout=10.0)

    def tearDown(self):
        self._tmp.cleanup()

    def aborting(self, code=1, before=""):
        path = self.root / "artifact" / "abort.py"
        path.write_text("import sys\n" + (f"print({before!r})\n" if before else "") +
                        "print('wine: Unimplemented function KERNEL32.dll.CopyFile2 called at address 0x1 (thread 1), "
                        f"starting debugger...')\nsys.exit({code})\n", encoding="utf-8")
        return [WINDOWS_ROOT + "/artifact/abort.py"]

    def run_entry(self, entry):
        return rwt.run_test(entry, self.args, [sys.executable])

    def test_an_abort_fails_even_when_the_test_is_expected_to_fail(self):
        status, detail = self.run_entry(ctest_entry("will_fail", self.aborting(), {"WILL_FAIL": "1"}))
        self.assertEqual(status, "fail")
        self.assertIn("aborted by Wine", detail)

    def test_an_abort_fails_even_when_the_required_output_was_already_printed(self):
        entry = ctest_entry("printed", self.aborting(before="ALL OK"), {"PASS_REGULAR_EXPRESSION": "ALL OK"})
        self.assertEqual(self.run_entry(entry)[0], "fail")

    def test_an_abort_with_a_zero_exit_code_still_fails(self):
        self.assertEqual(self.run_entry(ctest_entry("zero", self.aborting(code=0)))[0], "fail")


class MainTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.artifact = self.root / "artifact"
        self.artifact.mkdir()
        for name, code in (("ok.py", 0), ("bad.py", 1)):
            (self.artifact / name).write_text(f"import sys\nsys.exit({code})\n", encoding="utf-8")
        # main() maps the checkout's drive into $WINEPREFIX: keep it away from a real prefix (~/.wine).
        env = mock.patch.dict(os.environ, {"WINEPREFIX": str(self.root / "no-prefix")})
        env.start()
        self.addCleanup(env.stop)

    def tearDown(self):
        self._tmp.cleanup()

    def list_tests(self, entries):
        path = self.root / "ctest.json"
        path.write_text(json.dumps({"tests": entries}), encoding="utf-8")
        return str(path)

    def command(self, name):
        return [WINDOWS_ROOT + "/artifact/" + name]

    def run_main(self, entries, *extra, stderr=False):
        listing = self.list_tests(entries)
        argv = ["run_wine_tests.py", "--tests", listing, "--windows-root", WINDOWS_ROOT,
                "--root", str(self.root), "--wine", sys.executable.replace(os.sep, "/"), *extra]
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = rwt.main()
        return code, out.getvalue(), err.getvalue()

    def test_a_required_test_that_passes_exits_zero(self):
        code, out, _ = self.run_main([ctest_entry("ok", self.command("ok.py"))], "--require", "ok")
        self.assertEqual(code, 0)
        self.assertIn("1 passed, 0 failed, 0 skipped under Wine (1 tests)", out)

    def test_a_required_test_that_fails_exits_one(self):
        code, out, _ = self.run_main([ctest_entry("bad", self.command("bad.py"))], "--require", "bad")
        self.assertEqual(code, 1)
        self.assertIn("required tests that did not pass: bad", out)

    def test_a_required_test_that_is_not_listed_exits_one(self):
        code, out, _ = self.run_main([ctest_entry("ok", self.command("ok.py"))], "--require", "absent")
        self.assertEqual(code, 1)
        self.assertIn("required test not listed", out)

    def test_unrequired_failures_are_reported_but_do_not_fail_the_job(self):
        code, out, _ = self.run_main([ctest_entry("bad", self.command("bad.py"))])
        self.assertEqual(code, 0)
        self.assertIn("[FAIL] bad", out)

    def test_the_summary_table_is_appended(self):
        summary = self.root / "summary.md"
        code, _, _ = self.run_main([ctest_entry("ok", self.command("ok.py")), ctest_entry("bad", self.command("bad.py"))],
                                   "--summary", str(summary), "--label", "Wine 11", "--require", "ok")
        self.assertEqual(code, 0)
        text = summary.read_text(encoding="utf-8")
        self.assertIn("### Windows tests under Wine 11", text)
        self.assertIn("| `ok` | pass | yes |", text)
        self.assertIn("| `bad` | fail |  |", text)

    def test_a_summary_in_a_missing_folder_creates_it(self):
        summary = self.root / "missing" / "sub" / "summary.md"
        code, _, err = self.run_main([ctest_entry("ok", self.command("ok.py"))],
                                     "--summary", str(summary), "--label", "Wine 12")
        self.assertEqual(code, 0)
        self.assertEqual(err, "")
        self.assertIn("### Windows tests under Wine 12", summary.read_text(encoding="utf-8"))

    def test_an_unwritable_summary_is_one_error_line(self):
        blocker = self.root / "blocker"
        blocker.write_text("not a folder", encoding="utf-8")
        summary = blocker / "summary.md"       # its parent is a file, not a folder
        code, _, err = self.run_main([ctest_entry("ok", self.command("ok.py"))], "--summary", str(summary))
        self.assertEqual(code, 1)
        self.assertNotIn("Traceback", err)
        self.assertIn("ERROR: cannot write the summary", err)
        self.assertEqual(len(err.strip().splitlines()), 1, err)

    def test_a_file_that_is_not_json_is_one_error_line(self):
        listing = self.root / "notes.txt"
        listing.write_text("this is not ctest output\n", encoding="utf-8")
        argv = ["run_wine_tests.py", "--tests", str(listing), "--windows-root", WINDOWS_ROOT,
                "--root", str(self.root), "--wine", sys.executable.replace(os.sep, "/")]
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = rwt.main()
        self.assertEqual(code, 2)
        self.assertIn("ERROR: cannot read", err.getvalue())
        self.assertNotIn("Traceback", err.getvalue())

    def test_json_that_is_not_ctest_output_is_one_error_line(self):
        code, _, err = self.run_main_json([{"name": "t"}])
        self.assertEqual(code, 2)
        self.assertIn("not the output of ctest", err)

    def test_missing_tests_is_not_a_zero_test_run(self):
        code, _, err = self.run_main_json({})
        self.assertEqual(code, 2)
        self.assertIn("not the output of ctest", err)

    def test_scalar_tests_rows_are_rejected(self):
        code, _, err = self.run_main_json({"tests": ["binary", 3, None]})
        self.assertEqual(code, 2)
        self.assertIn("not the output of ctest", err)

    def test_an_explicit_empty_tests_list_is_accepted(self):
        code, out, _ = self.run_main_json({"tests": []})
        self.assertEqual(code, 0)
        self.assertIn("0 passed, 0 failed, 0 skipped under Wine (0 tests)", out)

    def run_main_json(self, document):
        listing = self.root / "wrong.json"
        listing.write_text(json.dumps(document), encoding="utf-8")
        argv = ["run_wine_tests.py", "--tests", str(listing), "--windows-root", WINDOWS_ROOT,
                "--root", str(self.root), "--wine", sys.executable.replace(os.sep, "/")]
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = rwt.main()
        return code, out.getvalue(), err.getvalue()

    def test_a_bad_timeout_is_refused_by_argparse(self):
        for text in ("0", "-1", "abc", "nan", "inf", "-inf"):
            with self.subTest(text=text):
                with self.assertRaises(argparse.ArgumentTypeError):
                    rwt.positive_seconds(text)
        self.assertEqual(rwt.positive_seconds("0.5"), 0.5)

    def test_a_nonfinite_timeout_is_a_usage_error(self):
        for text in ("nan", "inf", "-inf"):
            with self.subTest(text=text):
                with self.assertRaises(SystemExit) as ctx:
                    self.run_main([ctest_entry("ok", self.command("ok.py"))], "--timeout", text)
                self.assertEqual(ctx.exception.code, 2)

    def test_a_blank_wine_is_a_usage_error(self):
        with self.assertRaises(SystemExit) as ctx:
            self.run_main([ctest_entry("ok", self.command("ok.py"))], "--wine", "")
        self.assertEqual(ctx.exception.code, 2)

    def test_an_unclosed_quote_in_wine_is_a_usage_error(self):
        with self.assertRaises(SystemExit) as ctx:
            self.run_main([ctest_entry("ok", self.command("ok.py"))], "--wine", "wine 'unclosed")
        self.assertEqual(ctx.exception.code, 2)

    def test_wine_quoting_is_still_split(self):
        self.assertEqual(rwt.wine_command("prog 'two words'"), ["prog", "two words"])
        with self.assertRaises(argparse.ArgumentTypeError):
            rwt.wine_command("   ")


    # --jobs N: more than one test at a time, with the same results, and no overlap CTest would not allow.

    def marker(self, name, body="pass"):
        """A program that logs its start and end (with the time) to markers.log, runs `body`, and exits 0."""
        log = (self.root / "markers.log").as_posix()
        (self.artifact / name).write_text(
            "import os, sys, time\n"
            f"def mark(what):\n    open({log!r}, 'a').write(f'{{what}} {{os.path.basename(sys.argv[0])}} {{time.monotonic():.3f}}\\n')\n"
            f"mark('start')\n{body}\nmark('end')\n", encoding="utf-8")
        return self.command(name)

    def marks(self):
        rows = [line.split() for line in (self.root / "markers.log").read_text().splitlines()]
        return [(what, name, float(t)) for what, name, t in rows]

    def spans(self, name):
        rows = [r for r in self.marks() if r[1] == name]
        return min(t for _, _, t in rows), max(t for _, _, t in rows)

    def test_the_results_are_those_of_one_test_at_a_time(self):
        (self.artifact / "skip.py").write_text("import sys\nsys.exit(77)\n", encoding="utf-8")
        entries = [ctest_entry("ok", self.command("ok.py")), ctest_entry("bad", self.command("bad.py")),
                   ctest_entry("skipped", self.command("skip.py"), {"SKIP_RETURN_CODE": "77"}),
                   ctest_entry("inverted", self.command("bad.py"), {"WILL_FAIL": "1"}),
                   ctest_entry("missing", self.command("nothing.py")), ctest_entry("no_command", None)]
        one = self.run_main(entries, "--require", "ok")
        many = self.run_main(entries, "--require", "ok", "--jobs", "4")
        self.assertEqual(many[0], one[0])
        self.assertEqual(sorted(l for l in many[1].splitlines() if l.startswith("[")),
                         sorted(l for l in one[1].splitlines() if l.startswith("[")))
        self.assertIn("2 passed, 3 failed, 1 skipped", many[1])

    def test_a_summary_lists_the_tests_in_list_order(self):
        names = [f"t{i}" for i in range(6)]
        summary = self.root / "summary.md"
        self.run_main([ctest_entry(n, self.marker(f"{n}.py", "time.sleep(0.2 * (6 - int(sys.argv[0][-4])))"))
                       for n in names], "--jobs", "6", "--summary", str(summary))
        rows = [l.split("|")[1].strip(" `") for l in summary.read_text().splitlines() if l.startswith("| `")]
        self.assertEqual(rows, names)

    def test_different_programs_overlap_and_the_same_program_does_not(self):
        entries = [ctest_entry("a1", self.marker("a.py", "time.sleep(0.4)")),
                   ctest_entry("a2", self.marker("a.py", "time.sleep(0.4)")),
                   ctest_entry("b", self.marker("b.py", "time.sleep(0.4)")),
                   ctest_entry("c", self.marker("c.py", "time.sleep(0.4)"))]
        code, out, _ = self.run_main(entries, "--jobs", "4")
        self.assertEqual(code, 0, out)
        starts = [t for what, name, t in self.marks() if what == "start"]
        a_runs = sorted((t, what) for what, name, t in self.marks() if name.endswith("a.py"))
        self.assertEqual([w for _, w in a_runs], ["start", "end", "start", "end"])      # a1 ended before a2 began
        b_start, c_start = (self.spans(n)[0] for n in ("b.py", "c.py"))
        self.assertLess(abs(b_start - c_start), 0.35)                                     # b and c ran together
        self.assertEqual(len(starts), 4)

    def test_a_run_serial_test_runs_alone(self):
        entries = [ctest_entry("x", self.marker("x.py", "time.sleep(0.3)")),
                   ctest_entry("alone", self.marker("alone.py", "time.sleep(0.3)"), {"RUN_SERIAL": "1"}),
                   ctest_entry("y", self.marker("y.py", "time.sleep(0.3)"))]
        self.run_main(entries, "--jobs", "4")
        alone = self.spans("alone.py")
        for other in ("x.py", "y.py"):
            start, end = self.spans(other)
            self.assertTrue(end <= alone[0] or start >= alone[1], (other, alone, (start, end)))

    def test_a_fixture_is_set_up_before_the_tests_that_need_it(self):
        flag = (self.root / "fixture.flag").as_posix()
        (self.artifact / "setup.py").write_text(f"import time\ntime.sleep(0.5)\nopen({flag!r}, 'w').write('x')\n",
                                                encoding="utf-8")
        (self.artifact / "needs.py").write_text(f"import os, sys\nsys.exit(0 if os.path.exists({flag!r}) else 1)\n",
                                                encoding="utf-8")
        entries = [ctest_entry("setup", self.command("setup.py"), {"FIXTURES_SETUP": "cases"}),
                   ctest_entry("needs", self.command("needs.py"), {"FIXTURES_REQUIRED": "cases"}),
                   ctest_entry("other", self.command("ok.py"))]
        code, out, _ = self.run_main(entries, "--require", "needs", "--jobs", "4")
        self.assertEqual(code, 0, out)
        self.assertIn("3 passed", out)

    def test_a_test_that_hangs_costs_its_own_timeout_and_no_other_test(self):
        entries = [ctest_entry("hang", self.marker("hang.py", "time.sleep(60)"))]
        entries += [ctest_entry(f"quick{i}", self.marker(f"quick{i}.py", "time.sleep(0.2)")) for i in range(6)]
        code, out, _ = self.run_main(entries, "--jobs", "2", "--timeout", "1.5", "--require", "quick5")
        self.assertEqual(code, 0, out)
        self.assertIn("6 passed, 1 failed", out)
        self.assertIn("timed out after 1.5 s", out)
        hang_start = self.spans("hang.py")[0]
        # the hang held one of the two workers for its 1.5 s; the other test programs ran and ended beside it
        beside = [n for what, n, t in self.marks() if what == "end" and n.startswith("quick") and t < hang_start + 1.0]
        self.assertGreaterEqual(len(beside), 3, self.marks())

    def test_a_runner_error_in_one_test_fails_that_test_and_not_the_others(self):
        real = rwt.run_test

        def flaky(test, args, wine):
            if test["name"] == "boom":
                raise RuntimeError("synthetic")
            return real(test, args, wine)

        entries = [ctest_entry("boom", self.command("ok.py")), ctest_entry("fine", self.command("bad.py"))]
        with mock.patch.object(rwt, "run_test", flaky):
            code, out, _ = self.run_main(entries, "--jobs", "2")
        self.assertIn("[FAIL] boom", out)
        self.assertIn("the runner raised RuntimeError: synthetic", out)
        self.assertIn("[FAIL] fine", out)

    def test_jobs_must_be_a_whole_number_of_at_least_one(self):
        for bad in ("0", "-2", "x", "1.5"):
            with self.subTest(jobs=bad), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    self.run_main([ctest_entry("ok", self.command("ok.py"))], "--jobs", bad)
                self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
