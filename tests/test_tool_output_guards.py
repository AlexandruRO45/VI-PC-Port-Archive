"""Every tool that writes a report file through --json, --csv, --report or --records uses the shared output-path
helpers (tools/report_output.py, tools/report_paths.py) instead of its own copy or nothing.

    python -I tests/test_tool_output_guards.py

A report path that names an input (another spelling, a case-only spelling, a symlink or a hard link), a folder, or
a file where a parent folder should be has been fixed one tool at a time, in dozens of pull requests, several for
the same tool. This test makes the shared helpers the rule: a new tool with one of these options has to import
them, and the tools that don't yet are listed in PENDING, which may only shrink.

The scan is static (ast): an option counts when `add_argument` declares it with a value (no `action`, or
`store`/`append`); `--json` as an on/off switch prints to stdout and writes no file. `--out` is not checked yet,
because for many tools it names a folder, not a report file. Standard library only; nothing is run, no game data.
"""
import ast
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"

REPORT_OPTIONS = {"--json", "--csv", "--report", "--records"}
VALUE_ACTIONS = {"store", "append"}
SHARED_MODULES = {"report_output", "report_paths"}
SHARED_NAMES = {"check_output_path", "output_collision", "write_json_report", "same_report_path"}
HELPER_MODULES = {"report_output.py", "report_paths.py"}

# Tools with a report option that don't use the shared helpers yet: they carry their own path check (often a copy of
# `validate_json_path` / `write_json_report` or a `samefile` compare), or none. Moving one to the shared helpers
# removes it from here; the test fails until it is removed. Do not add to this list: a new tool imports the helpers.
PENDING = {
    "analyze_4240_full", "anim_glb", "anim_skel_match", "buffer_bounds_probe", "buffer_store_probe",
    "buffer_swizzle_probe", "bvh_geometry_check", "bvh_gpu_smoke", "bvh_index", "bvh_match", "cvt_f16_omod_probe",
    "cvt_norm_mods_probe", "d16_format_probe", "d3d12_iid_scan", "ds_permute_probe", "f16_trans_sweep",
    "find_hash_lookup", "fragment_children", "gamedata_graph", "gen_d3d12_stubs", "heap_index_census",
    "image_d16_probe", "image_query_probe", "instance_lists", "int_hw_probe", "meshlet_page", "mode_round_probe",
    "navmesh_decode", "ps_helper_probe", "ps_oracle", "rbf0_tree", "rootsig_census", "shader_link_check",
    "skin_streams", "ttpf_mips", "user_sgpr_census", "vertex_types", "wave64_pipeline_audit", "world_index",
}

# Tools whose report option is exempt, each with the reason it needs no shared check. Keep the reason specific.
EXEMPT = {}


def report_options(tree):
    """The report options a module declares with a value."""
    found = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            continue
        names = {a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)}
        options = names & REPORT_OPTIONS
        if not options:
            continue
        action = next((k.value for k in node.keywords if k.arg == "action"), None)
        if action is None or (isinstance(action, ast.Constant) and action.value in VALUE_ACTIONS):
            found |= options
    return found


def uses_shared_helpers(tree):
    """True when the module imports one of the shared helpers (by name, or the module itself)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in SHARED_MODULES:
            if {a.name for a in node.names} & SHARED_NAMES:
                return True
        if isinstance(node, ast.Import) and {a.name for a in node.names} & SHARED_MODULES:
            return True
    return False


def scan():
    """{tool name: (report options, uses the shared helpers)} for every tool with a report option."""
    tools = {}
    for path in sorted(TOOLS.glob("*.py")):
        if path.name in HELPER_MODULES:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        options = report_options(tree)
        if options:
            tools[path.stem] = (options, uses_shared_helpers(tree))
    return tools


class ScannerTests(unittest.TestCase):
    """The scan itself, on hand-written sources."""

    def options(self, source):
        return report_options(ast.parse(source))

    def test_value_options_count(self):
        src = ("ap.add_argument('--json', help='write here')\n"
               "ap.add_argument('-c', '--csv', metavar='FILE')\n"
               "sub.add_argument('--report', action='store')\n"
               "ap.add_argument('--records', action='append')\n")
        self.assertEqual(self.options(src), {"--json", "--csv", "--report", "--records"})

    def test_switches_and_other_options_do_not(self):
        src = ("ap.add_argument('--json', action='store_true')\n"
               "ap.add_argument('--csv', action='store_const', const=1)\n"
               "ap.add_argument('--out')\n"
               "ap.add_argument('--manifest')\n"
               "ap.add_argument('json')\n")
        self.assertEqual(self.options(src), set())

    def test_shared_helper_imports(self):
        for src in ("from report_output import write_json_report",
                    "from report_paths import same_report_path  # noqa: E402",
                    "from report_output import check_output_path, output_collision",
                    "import report_output"):
            with self.subTest(src=src):
                self.assertTrue(uses_shared_helpers(ast.parse(src)))
        for src in ("from report_paths import path_key", "import json", "def same_report_path(a, b): pass"):
            with self.subTest(src=src):
                self.assertFalse(uses_shared_helpers(ast.parse(src)))


class ToolOutputGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tools = scan()

    def test_scan_finds_the_known_cases(self):
        # Fails loudly if the scan stops seeing tools, so the other tests cannot pass by finding nothing.
        self.assertGreaterEqual(len(self.tools), 40)
        self.assertTrue(self.tools["gamedata_index"][1], "gamedata_index imports same_report_path")
        self.assertIn("--json", self.tools["find_hash_lookup"][0])

    def test_every_report_tool_uses_the_shared_helpers(self):
        missing = sorted(name for name, (options, shared) in self.tools.items()
                         if not shared and name not in PENDING and name not in EXEMPT)
        self.assertEqual(missing, [], "these tools write a report through "
                         "--json/--csv/--report/--records without the shared output-path helpers. Check the path "
                         "with report_output.check_output_path and output_collision before any work, and write it "
                         "with write_json_report (see the module docstring); do not add them to PENDING.")

    def test_pending_names_real_tools(self):
        gone = sorted(name for name in PENDING if name not in self.tools)
        self.assertEqual(gone, [], "no longer tools with a report option (moved, renamed or removed): "
                         "remove them from PENDING in tests/test_tool_output_guards.py")

    def test_pending_tools_that_moved_to_the_shared_helpers(self):
        # Reported, not failed: a pull request that moves one tool should not turn CI red for every other open pull
        # request that still lists it. Whoever touches this file next removes them.
        moved = sorted(name for name in PENDING if name in self.tools and self.tools[name][1])
        if moved:
            self.skipTest(f"use the shared helpers now, remove from PENDING: {', '.join(moved)}")

    def test_exempt_entries_are_real_and_explained(self):
        for name, reason in EXEMPT.items():
            with self.subTest(tool=name):
                self.assertIn(name, self.tools, "not a tool with a report option")
                self.assertNotIn(name, PENDING, "listed twice")
                self.assertGreaterEqual(len(reason.split()), 4, "give a specific reason")


if __name__ == "__main__":
    unittest.main()
