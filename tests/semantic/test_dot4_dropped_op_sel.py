"""v_dot4_i32_i8's op_sel/op_sel_hi come from the blob when llvm-mc drops them.

llvm-mc 23 refuses op_sel and op_sel_hi on gfx1030's v_dot4_i32_i8. When it disassembles one that has them, it
prints the instruction without them and re-encodes the defaults (op_sel:[0,0,0] op_sel_hi:[1,1,1]). The hardware
reads them (docs/design.md, "Signed-byte dot products"), so translating the listing as printed would pick the wrong
bytes. The blob holds the real fields, and translate() now reads them from there: in a clean listing
(check_printed_encodings) and in one with undecodable words (the re-anchoring path). Any other difference is still
refused. Hand-built listing and blob, no game data.

    python -I tests/semantic/test_dot4_dropped_op_sel.py -v
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from support import mrt0_words, rdna2spirv, translate_listing  # noqa: E402

# llvm-mc -mcpu=gfx1030 -show-encoding (LLVM 23.1.6). S0 bytes 1, 2, 3, 4; S1 bytes 2, 3, -2, -3; addend 100.
PROGRAM = [
    ("v_mov_b32_e32 v1, 0x4030201", "ff 02 02 7e 01 02 03 04"),
    ("v_mov_b32_e32 v2, 0xfdfe0302", "ff 02 04 7e 02 03 fe fd"),
    ("v_mov_b32_e32 v3, 0x64", "ff 02 06 7e 64 00 00 00"),
    ("v_dot4_i32_i8 v0, v1, v2, v3", "00 40 16 cc 01 05 0e 1c"),
    ("exp mrt0 v0, v0, v0, v0 done vm", "0f 18 00 f8 00 00 00 00"),
    ("s_endpgm", "00 00 81 bf"),
]
DOT = 3
CODE_START = 0x100
NAME = "ps_dot4_dropped_op_sel"


def listing(invalid=False):
    size = sum(len(bytes.fromhex(enc)) for _, enc in PROGRAM)
    lines = [f"; {NAME}  code 0x{CODE_START:x}..0x{CODE_START + size:x} (scan), gfx1030, "
             f"{int(invalid)} invalid words"]
    for text, enc in PROGRAM:
        printed = ",".join(f"0x{b:02x}" for b in bytes.fromhex(enc))
        lines.append(f"\t{text:<40}; encoding: [{printed}]")
    if invalid:
        # What shader_disasm.py appends when llvm-mc skipped words; it sends translate() down the re-anchoring path.
        lines += ["; --- llvm-mc", "; <stdin>:1:1: warning: invalid instruction encoding"]
    return ("\n".join(lines) + "\n").encode("utf-8")


def blob(dot_bytes=None):
    """The program's bytes, with the dot4 stored as `dot_bytes` (what the compiler wrote, not what LLVM printed)."""
    parts = [bytes.fromhex(enc) for _, enc in PROGRAM]
    if dot_bytes is not None:
        parts[DOT] = bytes.fromhex(dot_bytes)
    return bytes(CODE_START) + b"".join(parts)


def result(spvasm):
    return mrt0_words(spvasm)[0]


class Dot4DroppedOpSelTest(unittest.TestCase):
    def translate(self, stored, invalid=False):
        return translate_listing(NAME, listing(invalid), fssm=blob(stored))

    def test_default_selects(self):
        # 1*2 + 2*3 + 3*-2 + 4*-3 + 100
        self.assertEqual(result(self.translate(None)), 90)

    def test_op_sel_hi_from_the_blob(self):
        # op_sel_hi:[1,0,1] (word 1 bit 28 clear): S1's byte pairs 2..3 come from its low half, bytes 2, 3 again.
        for invalid in (False, True):
            with self.subTest(re_anchored=invalid):
                spvasm = self.translate("00 40 16 cc 01 05 0e 0c", invalid)
                self.assertEqual(result(spvasm), 1 * 2 + 2 * 3 + 3 * 2 + 4 * 3 + 100)

    def test_op_sel_and_op_sel_hi_from_the_blob(self):
        # op_sel:[0,1,0] op_sel_hi:[1,0,1]: pairs 0..1 of S1 from its high half (-2, -3), pairs 2..3 from the low one.
        self.assertEqual(result(self.translate("00 50 16 cc 01 05 0e 0c")), 1 * -2 + 2 * -3 + 3 * 2 + 4 * 3 + 100)

    def test_constant_s1_broadcast(self):
        # The form the FSPX programs store: S1 an inline constant with op_sel_hi[1] = 0, so both byte pairs come from
        # its low half. -6 reads as its 32-bit value 0xfffffffa, as #2224 measured for packed 16-bit ops: low half
        # bytes -6, -1. The default would take pairs 2..3 from the high half, -1, -1.
        program = list(PROGRAM)
        program[DOT] = ("v_dot4_i32_i8 v0, v1, -6, v3", "00 40 16 cc 01 8d 0d 1c")
        size = sum(len(bytes.fromhex(enc)) for _, enc in program)
        head = f"; {NAME}  code 0x{CODE_START:x}..0x{CODE_START + size:x} (scan), gfx1030, 0 invalid words"
        text = "\n".join([head] + [f"\t{t:<40}; encoding: [{','.join(f'0x{b:02x}' for b in bytes.fromhex(e))}]"
                                   for t, e in program]) + "\n"
        stored = bytes(CODE_START) + b"".join(bytes.fromhex(e) for _, e in program[:DOT]) + \
            bytes.fromhex("00 40 16 cc 01 8d 0d 0c") + b"".join(bytes.fromhex(e) for _, e in program[DOT + 1:])
        default = bytes(CODE_START) + b"".join(bytes.fromhex(e) for _, e in program)
        self.assertEqual(result(translate_listing(NAME, text.encode(), fssm=default)),
                         1 * -6 + 2 * -1 + 3 * -1 + 4 * -1 + 100)
        self.assertEqual(result(translate_listing(NAME, text.encode(), fssm=stored)),
                         1 * -6 + 2 * -1 + 3 * -6 + 4 * -1 + 100)

    def test_a_restored_addend_select_is_still_refused(self):
        # op_sel_hi[2] = 0 (word 0 bit 14 clear) would half-select the 32-bit addend, which the dot4 rule refuses.
        with self.assertRaises(rdna2spirv.Unsupported) as cm:
            self.translate("00 00 16 cc 01 05 0e 1c")
        self.assertIn("v_dot4_i32_i8 op_sel", str(cm.exception))

    def test_other_differences_are_still_refused(self):
        # neg_lo[0] (word 1 bit 29) is not a select: the printed instruction is not the stored one.
        with self.assertRaises(rdna2spirv.Unsupported) as cm:
            self.translate("00 40 16 cc 01 05 0e 3c")
        self.assertIn("dropped", str(cm.exception))
        # Re-anchored, the same word can't be placed and reads as normalized; it is reachable, so it is refused.
        with self.assertRaises(rdna2spirv.Unsupported) as cm:
            self.translate("00 40 16 cc 01 05 0e 3c", invalid=True)
        self.assertIn("normalized encoding", str(cm.exception))

    def test_a_live_source_stored_as_s0_is_still_refused(self):
        # same_encoding lets an unused SRC1/SRC2 stored as 0 (s0) match a printed 0x80 (inline 0). v_dot4 reads all
        # three sources, so with a dropped select as well the stored s0 must not be translated as the constant 0.
        # Printed: v_dot4_i32_i8 v0, v1, 0, v3 (SRC1 = 0x80) and v_dot4_i32_i8 v0, v1, v2, 0 (SRC2 = 0x80).
        for field, printed_hex, stored_hex in (
                ("SRC1", "00 40 16 cc 01 01 0d 1c", "00 40 16 cc 01 01 0c 0c"),
                ("SRC2", "00 40 16 cc 01 05 02 1a", "00 40 16 cc 01 05 02 08")):
            printed, stored = bytes.fromhex(printed_hex), bytes.fromhex(stored_hex)
            self.assertIsNone(rdna2spirv.dropped_op_sel(stored, printed, "v_dot4_i32_i8"), field)
            program = list(PROGRAM)
            program[DOT] = (program[DOT][0], printed_hex)
            size = sum(len(bytes.fromhex(enc)) for _, enc in program)
            for invalid in (False, True):
                with self.subTest(field=field, re_anchored=invalid):
                    lines = [f"; {NAME}  code 0x{CODE_START:x}..0x{CODE_START + size:x} (scan), gfx1030, "
                             f"{int(invalid)} invalid words"]
                    lines += [f"\t{t:<40}; encoding: [{','.join(f'0x{b:02x}' for b in bytes.fromhex(e))}]"
                              for t, e in program]
                    if invalid:
                        lines += ["; --- llvm-mc", "; <stdin>:1:1: warning: invalid instruction encoding"]
                    fssm = bytes(CODE_START) + b"".join(stored if i == DOT else bytes.fromhex(e)
                                                        for i, (_, e) in enumerate(program))
                    with self.assertRaises(rdna2spirv.Unsupported):
                        translate_listing(NAME, ("\n".join(lines) + "\n").encode(), fssm=fssm)

    def test_only_the_listed_ops(self):
        stored = bytes.fromhex("00 40 16 cc 01 05 0e 0c")
        printed = bytes.fromhex("00 40 16 cc 01 05 0e 1c")
        self.assertEqual(rdna2spirv.dropped_op_sel(stored, printed, "v_dot4_i32_i8"), ([0, 0, 0], [1, 0, 1]))
        self.assertIsNone(rdna2spirv.dropped_op_sel(stored, printed, "v_dot2_i32_i16"))
        # A printed instruction that already carries non-default selects is LLVM's own reading: nothing to restore.
        self.assertIsNone(rdna2spirv.dropped_op_sel(printed, stored, "v_dot4_i32_i8"))


if __name__ == "__main__":
    unittest.main()
