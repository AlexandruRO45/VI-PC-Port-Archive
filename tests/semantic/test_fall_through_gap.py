"""A path that runs into words llvm-mc skipped is refused; code that ends before them is not (SHADER-813).

llvm-mc leaves out the words it cannot decode and the listing says "invalid instruction encoding" there. The translator
re-anchors the printed instructions to the blob. If a reachable instruction falls through into such a gap, everything
after the gap (exports, stores) would be missing from a module that still passes spirv-val, so it is refused. Hand-built
listing and blob, no game data; the gap is a made-up marker pattern (two little-endian words 0xDEADBEEF, 0xBAADF00D).

    python -I tests/semantic/test_fall_through_gap.py -v
"""
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from support import rdna2spirv, translate_listing  # noqa: E402

NAME = "ps_gap_semantic"
MOV0 = "f2 02 00 7e"            # v_mov_b32_e32 v0, 1.0
MOV1 = "f2 02 02 7e"            # v_mov_b32_e32 v1, 1.0
EXPORT = "0f 18 00 f8 00 01 00 01"
ENDPGM = "00 00 81 bf"
GAP = struct.pack("<II", 0xDEADBEEF, 0xBAADF00D)
EXPORT_TEXT = "exp mrt0 v0, v1, v0, v1 done vm"


def line(text, enc):
    printed = ",".join(f"0x{b:02x}" for b in bytes.fromhex(enc))
    return f"\t{text:<40}; encoding: [{printed}]\n"


MOVS = line("v_mov_b32_e32 v0, 1.0", MOV0) + line("v_mov_b32_e32 v1, 1.0", MOV1)
GAP_LINE = "; invalid instruction encoding at 0x8\n"
TAIL = line(EXPORT_TEXT, EXPORT) + line("s_endpgm", ENDPGM)


def raw(*parts):
    return b"".join(p if isinstance(p, bytes) else bytes.fromhex(p) for p in parts)


class FallThroughGapTest(unittest.TestCase):
    def test_instruction_running_into_the_gap_is_refused(self):
        listing = ("; " + NAME + "  code 0x0..0x1c (header), gfx1013, 2 invalid words\n" + MOVS + GAP_LINE + TAIL)
        blob = raw(MOV0, MOV1, GAP, EXPORT, ENDPGM)
        with self.assertRaisesRegex(rdna2spirv.Unsupported, "fall-through into undecodable words"):
            translate_listing(NAME, listing.encode("utf-8"), fssm=blob)

    def test_gap_after_s_endpgm_is_not_reached_and_translates(self):
        listing = ("; " + NAME + "  code 0x0..0x1c (header), gfx1013, 2 invalid words\n"
                   + MOVS + line(EXPORT_TEXT, EXPORT) + line("s_endpgm", ENDPGM) + "; invalid instruction encoding at 0x14\n")
        blob = raw(MOV0, MOV1, EXPORT, ENDPGM, GAP)
        self.assertIn("OpEntryPoint Fragment", translate_listing(NAME, listing.encode("utf-8"), fssm=blob))


if __name__ == "__main__":
    unittest.main()
