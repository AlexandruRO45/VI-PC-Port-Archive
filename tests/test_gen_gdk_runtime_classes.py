"""Tests for tools/gen_gdk_runtime_classes.py on a synthetic xgameruntime-docs clone.

    python -I tests/test_gen_gdk_runtime_classes.py
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import gen_gdk_runtime_classes as gen  # noqa: E402

CLASS_ID = "11111111-1111-1111-1111-111111111111"
IFACE_ID = "22222222-2222-2222-2222-222222222222"


def make_docs(root: Path) -> None:
    folder = root / "COM" / "XUser"
    folder.mkdir(parents=True)
    (folder / "README.md").write_text(
        f"Class Id: `{CLASS_ID}`\n\nImplements: [IXUser](IXUser.md)\n", encoding="utf-8")
    (folder / "IXUser.md").write_text(
        f"Interface Id: `{IFACE_ID}`\n\nInherits: IUnknown\n\nLayout:\n\n"
        "- [GetThing](GetThing.md)\n- *unknown*\n",
        encoding="utf-8")


def run_main(argv):
    err = io.StringIO()
    with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
        try:
            code = gen.main(argv)
        except SystemExit as e:
            return e.code, err.getvalue()
    return code, err.getvalue()


class GenGdkRuntimeClassesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.docs = self.dir / "docs"
        make_docs(self.docs)
        self.out = self.dir / "out" / "gdk_runtime_classes.json"

    def test_writes_the_table(self):
        code, err = run_main([str(self.docs), "--out", str(self.out)])
        self.assertEqual(code, 0, err)
        with open(self.out, encoding="utf-8") as f:
            table = json.load(f)
        self.assertEqual(table["classes"]["XUser"]["implements"], ["IXUser"])
        self.assertEqual(table["interfaces"]["IXUser"]["methods"], ["GetThing", None])

    def test_a_missing_folder_is_created(self):
        nested = self.dir / "reports" / "deeper" / "table.json"
        code, err = run_main([str(self.docs), "--out", str(nested)])
        self.assertEqual(code, 0, err)
        self.assertTrue(nested.is_file())

    def test_out_must_not_overwrite_a_docs_file(self):
        readme = self.docs / "COM" / "XUser" / "README.md"
        before = readme.read_bytes()
        code, err = run_main([str(self.docs), "--out", str(readme)])
        self.assertEqual(code, 2)
        self.assertIn("would overwrite the docs input", err)
        self.assertEqual(readme.read_bytes(), before)

    def test_out_must_be_a_file(self):
        code, err = run_main([str(self.docs), "--out", str(self.docs / "COM")])
        self.assertEqual(code, 2)
        self.assertIn("is a folder, not a file", err)

    def test_a_failed_write_is_one_line(self):
        blocker = self.dir / "not_a_folder"
        blocker.write_bytes(b"keep me")
        code, err = run_main([str(self.docs), "--out", str(blocker / "table.json")])
        self.assertEqual(code, 2)
        self.assertIn("ERROR: cannot write", err)
        self.assertNotIn("Traceback", err)
        self.assertEqual(blocker.read_bytes(), b"keep me")

    def test_a_missing_docs_clone_is_one_line(self):
        code, err = run_main([str(self.dir / "missing-clone"), "--out", str(self.out)])
        self.assertEqual(code, 2)
        self.assertIn("ERROR: cannot read the docs", err)
        self.assertNotIn("Traceback", err)


class NonUtf8DocsTests(unittest.TestCase):
    def test_a_non_utf8_docs_file_is_one_error_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            docs = Path(tmp) / "docs"
            make_docs(docs)
            (docs / "COM" / "XUser" / "Notes.md").write_bytes(b"Interface Id: \xff\xfe not utf-8\n")
            out = Path(tmp) / "table.json"
            err = io.StringIO()
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                rc = gen.main([str(docs), "--out", str(out)])
            message = err.getvalue()
        self.assertEqual(rc, 2)
        self.assertIn("ERROR: cannot read the docs", message)
        self.assertNotIn("Traceback", message)
        self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
