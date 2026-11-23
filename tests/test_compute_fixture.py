"""The synthetic compute shader metadata reaches the translator's workgroup declaration."""

from pathlib import Path
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime" / "test" / "shaders"))
sys.path.insert(0, str(ROOT / "tools"))

from compute_fixture import compute_header  # noqa: E402
from rdna2spirv import translate  # noqa: E402


class ComputeFixtureTests(unittest.TestCase):
    def test_group_tuple_whitespace_reaches_spirv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            disasm = root / "disasm"
            fssm = root / "fssm"
            disasm.mkdir()
            fssm.mkdir()
            for index, group in enumerate(("4,2,1", "4, 2, 1", "4 , 2,\t1", "4,2,1 ; four by two")):
                with self.subTest(group=group):
                    shader = f"cs_group_{index}"
                    listing = disasm / f"{shader}.s"
                    listing.write_text(f"; group: {group}\n\ts_endpgm\n", encoding="utf-8", newline="\n")
                    (fssm / f"{shader}.fssm").write_bytes(compute_header(listing))
                    assembly = translate(str(listing))
                    local_size = next(line.strip() for line in assembly.splitlines()
                                      if line.lstrip().startswith("OpExecutionMode %main LocalSize "))
                    self.assertEqual(local_size, "OpExecutionMode %main LocalSize 4 2 1")

    def test_malformed_group_dimensions_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            listing = Path(tmp) / "cs_bad_group.asm"
            for group in ("4,2", "4,2,1,8", "4,,1", "0,2,1", "4294967296,1,1"):
                with self.subTest(group=group):
                    listing.write_text(f"; group: {group}\n\ts_endpgm\n", encoding="utf-8", newline="\n")
                    with self.assertRaisesRegex(ValueError, "group"):
                        compute_header(listing)

    def test_absent_group_directive_keeps_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            listing = Path(tmp) / "cs_default_group.asm"
            listing.write_text("\ts_endpgm\n", encoding="utf-8", newline="\n")
            header = compute_header(listing)
        self.assertEqual(struct.unpack_from("<3I", header, 0x13C), (64, 1, 1))


if __name__ == "__main__":
    unittest.main()
