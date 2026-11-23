"""Every LEONIDA_* / GTA6RT_* environment variable the runtime, make_game.py, the tools or the scripts read is named
in a document.

    python -I tests/test_docs_env_vars.py

`runtime/README.md` documents the runtime's variables one by one, but five that the `xgameruntime` shim reads
(`LEONIDA_DEVICE_TYPE` and four more) were in no document at all. This collects the variable names that appear as
string literals in `runtime/src`, `runtime/shims`, `make_game.py`, `tools/` and `scripts/` and checks that each is
written in some tracked Markdown file. `tools/` and `scripts/` were added after: `docs/testing.md` documents
`tools/pc_checker.py` without the switch that silences its weak-PC prompt, and `tools/first_light.py`'s harness path
was in no page either. Sources and Markdown only: no build, no GPU, no game files.
"""
import os
import re
import subprocess
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# A variable name as a string literal: "LEONIDA_X", L"LEONIDA_X" or 'LEONIDA_X'.
LITERAL = re.compile(r"""["']((?:LEONIDA|GTA6RT)_[A-Z0-9_]+)["']""")
CODE = ("runtime/src/", "runtime/shims/", "make_game.py", "tools/", "scripts/")
# Set by the code for itself or for a program it starts, not by a person: gta6rt passes the first to the copy of
# itself it starts with the image base reserved (runtime/src/base_reservation.h), the second is a name that is only
# looked up to make the CRT build its environment table (runtime/shims/process.cpp), and
# scripts/run_guest_scene.py sets the last two for the guest it starts (--play-audio picks winmm), overwriting
# whatever a person set.
INTERNAL = {"LEONIDA_BASE_RESERVED", "LEONIDA_RT_ENV_INIT",
            "LEONIDA_SCENE_AUDIO", "LEONIDA_SCENE_MODE"}
# Named in an open pull request rather than in this branch: drop an entry when its pull request lands (the test
# below fails if a name here stops being read, so a stale entry cannot hide a removal).
PENDING = {"LEONIDA_TOOLS_DIR": "#7130, tests/semantic/README.md",
           "LEONIDA_VK_HARNESS": "#6138, docs/first_light.md"}


def tracked_files():
    try:
        out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
        return [p for p in out.splitlines() if p]
    except (OSError, subprocess.CalledProcessError):
        found = []
        for root, dirs, files in os.walk(ROOT):
            dirs[:] = [d for d in dirs if d not in (".git", "build", "__pycache__", "third_party")]
            found += [os.path.relpath(os.path.join(root, f), ROOT).replace(os.sep, "/") for f in files]
        return found


def read(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8", errors="replace") as f:
        return f.read()


def variables_read(sources):
    """{variable: first "file:line" that names it} over `sources` ({name: text})."""
    found = {}
    for name in sorted(sources):
        for number, line in enumerate(sources[name].splitlines(), 1):
            for variable in LITERAL.findall(line):
                found.setdefault(variable, f"{name}:{number}")
    return found


def undocumented(sources, pages):
    """[(variable, where it is read)] for variables no page of `pages` ({name: text}) names as a whole word."""
    text = "\n".join(pages.values())
    return [(variable, where) for variable, where in sorted(variables_read(sources).items())
            if variable not in INTERNAL and variable not in PENDING
            and not re.search(rf"(?<![A-Z0-9_]){variable}(?![A-Z0-9_])", text)]


class EnvironmentVariablesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        files = tracked_files()
        cls.sources = {f: read(f) for f in files
                       if f.startswith(CODE) and f.endswith((".cpp", ".h", ".py"))}
        cls.pages = {f: read(f) for f in files if f.endswith(".md") and not f.startswith("third_party/")}

    def test_the_scan_finds_the_variables(self):
        # A moved folder or a changed spelling must not turn the check below into a no-op.
        found = variables_read(self.sources)
        self.assertGreaterEqual(len(found), 20, sorted(found))
        for known in ("LEONIDA_UNIMPLEMENTED", "LEONIDA_D3D12", "LEONIDA_DEVICE_TYPE"):
            self.assertIn(known, found)
        # One producer from each of the two folders added last: a folder renamed out of the scan shows up here.
        for known in ("LEONIDA_SKIP_SPEC_CHECK", "LEONIDA_SCENE_MODE"):
            self.assertIn(known, found)
        self.assertGreater(len(self.pages), 20)

    def test_the_internal_variables_are_still_read(self):
        self.assertEqual(sorted(INTERNAL - set(variables_read(self.sources))), [])

    def test_the_pending_variables_are_still_read(self):
        self.assertEqual(sorted(set(PENDING) - set(variables_read(self.sources))), [])

    def test_every_variable_is_named_in_a_document(self):
        missing = undocumented(self.sources, self.pages)
        self.assertEqual(missing, [], "read by the code and named in no .md file:\n" + "\n".join(
            f"  {variable} ({where})" for variable, where in missing))

    def test_an_undocumented_variable_is_reported(self):
        sources = {"runtime/shims/x.cpp": 'GetEnvironmentVariableA("LEONIDA_ONE", s, n);\n'
                                          'kName[] = L"LEONIDA_TWO";\n// LEONIDA_COMMENT is not a literal\n',
                   "make_game.py": "os.environ.get('GTA6RT_THREE')\nos.environ.get(\"LEONIDA_BASE_RESERVED\")\n"
                                   "os.environ.get(\"LEONIDA_VK_HARNESS\")\n"}
        self.assertEqual(variables_read(sources),
                         {"GTA6RT_THREE": "make_game.py:1", "LEONIDA_BASE_RESERVED": "make_game.py:2",
                          "LEONIDA_ONE": "runtime/shims/x.cpp:1", "LEONIDA_TWO": "runtime/shims/x.cpp:2",
                          "LEONIDA_VK_HARNESS": "make_game.py:3"})
        # A longer name that contains the variable does not document it, an internal one needs no page, and a
        # pending one waits for its pull request.
        pages = {"a.md": "Set `LEONIDA_ONE=1`.", "b.md": "LEONIDA_TWO_MORE and XGTA6RT_THREE"}
        self.assertEqual(undocumented(sources, pages),
                         [("GTA6RT_THREE", "make_game.py:1"), ("LEONIDA_TWO", "runtime/shims/x.cpp:2")])


if __name__ == "__main__":
    unittest.main()
