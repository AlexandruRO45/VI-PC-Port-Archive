import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import find_assets  # noqa: E402


class TestFindAssets(unittest.TestCase):
    def _create_tree(self, root: Path) -> None:
        assets = root / "assets"
        (assets / "char").mkdir(parents=True)
        (assets / "map").mkdir(parents=True)

        (assets / "char" / "cs_alpha.obj").write_bytes(b"x" * (2 * 1024))
        (assets / "char" / "cs_big.obj").write_bytes(b"x" * (2 * 1024 * 1024))
        (assets / "char" / "hair_01.OBJ").write_bytes(b"x" * (20 * 1024))
        (assets / "map" / "s01_w02.obj").write_bytes(b"x" * (4 * 1024))
        (assets / "random.obj").write_bytes(b"x" * 1024)

    def test_default_size_filter_and_descending_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._create_tree(root)
            out = io.StringIO()
            with redirect_stdout(out):
                code = find_assets.main(["--work", str(root)])
            output = out.getvalue()
            self.assertEqual(code, 0)
            self.assertIn("Found 2 matching .obj models", output)
            self.assertLess(output.index("cs_big.obj"), output.index("hair_01.OBJ"))

    def test_category_filter_and_top_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._create_tree(root)
            out = io.StringIO()
            with redirect_stdout(out):
                code = find_assets.main(
                    ["--work", str(root), "--category", "characters", "--min-size", "1", "--top", "1"]
                )
            output = out.getvalue()
            self.assertEqual(code, 0)
            self.assertIn("Found 2 matching .obj models (showing top 1)", output)
            self.assertIn("cs_big.obj", output)
            self.assertNotIn("cs_alpha.obj", output)

    def test_invalid_cli_arguments_rejected(self) -> None:
        for bad_args in (
            ["--work", ".", "--top", "-1"],
            ["--work", ".", "--min-size", "nan"],
            ["--work", ".", "--min-size", "inf"],
            ["--work", ".", "--min-size", "-5"],
        ):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as ctx:
                find_assets.main(bad_args)
            self.assertNotEqual(ctx.exception.code, 0)

    def test_missing_or_file_work_dir_returns_non_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "does_not_exist"
            with redirect_stderr(io.StringIO()):
                self.assertNotEqual(find_assets.main(["--work", str(missing)]), 0)

            file_path = Path(tmp) / "not_a_dir.txt"
            file_path.write_text("hi", encoding="utf-8")
            with redirect_stderr(io.StringIO()):
                self.assertNotEqual(find_assets.main(["--work", str(file_path)]), 0)


if __name__ == "__main__":
    unittest.main()
